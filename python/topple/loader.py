"""Load a submission: a ``.py`` file, or a directory with ``controller.py`` plus assets.

In-process loading runs the submission's code inside this interpreter, so it is for trusted code
only (your own controllers, the shipped examples). Untrusted submissions go through the
subprocess/container runner (Phase 3).

The entry file is read once: those bytes are hashed and those bytes are compiled and run, so the
submission sha256 always covers the code that runs (no bytecode cache is consulted and no second
read can see a swapped file).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from topple.contract import Controller, TaskSpec, class_name, describe_exception

ENTRY_FILE = "controller.py"
FACTORY = "make_controller"
MAX_FILES = 50
MAX_DIRECTORIES = 50
"""Most subdirectories a submission directory may hold (any depth; itself not counted)."""
MAX_BYTES = 20 * 1024 * 1024
SHA_TAG = b"topple.submission/1\n"
CONTROLLER_METHODS = ("reset", "act")  # the members of the Controller protocol (contract §8)


class LoadError(Exception):
    """The submission could not be loaded or does not follow the contract."""


@dataclass(frozen=True)
class Submission:
    id: str
    sha256: str
    entry: Path
    factory: Callable[[TaskSpec], Controller[Any]]


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _check_entry(entry: os.DirEntry[str], rel: str) -> bool:
    """True for a directory to descend into, False for a regular file; else ``LoadError``.

    Names are quoted with ``repr``: they are untrusted and can hold terminal escapes.
    """
    if entry.is_symlink():
        raise LoadError(f"symlinks are not allowed in a submission: {rel!r}")
    if entry.is_dir(follow_symlinks=False):
        if entry.name == "__pycache__":
            raise LoadError(f"remove compiled Python from the submission: {rel!r}")
        return True
    if not entry.is_file(follow_symlinks=False):
        raise LoadError(f"{rel!r} is not a regular file")
    if entry.name.endswith(".pyc"):
        raise LoadError(f"remove compiled Python (.pyc) from the submission: {rel!r}")
    return False


def _files(path: Path) -> list[Path]:
    """Every file of a submission in POSIX-path order, refusing anything that is not plain data.

    The walk stops as soon as it passes ``MAX_FILES`` files or ``MAX_DIRECTORIES`` directories,
    and a file or directory that cannot be read is a ``LoadError``.
    """
    if path.is_symlink():
        raise LoadError(f"{path.name!r} is a symlink; submit the file itself")
    if path.is_file():
        return [path]
    try:
        files = _walk(path)
    except OSError as exc:
        raise _unreadable(exc, path) from exc
    return sorted(files, key=lambda f: _rel(f, path))


def _walk(root: Path) -> list[Path]:
    files: list[Path] = []
    directories = 0
    pending = [root]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                child = Path(entry.path)
                if not _check_entry(entry, _rel(child, root)):
                    files.append(child)
                    if len(files) > MAX_FILES:
                        raise LoadError(f"submission has more than {MAX_FILES} files")
                    continue
                directories += 1
                if directories > MAX_DIRECTORIES:
                    raise LoadError(f"submission has more than {MAX_DIRECTORIES} directories")
                pending.append(child)
    return files


def _unreadable(exc: OSError, submission: Path) -> LoadError:
    """A ``LoadError`` for an ``OSError`` met while reading ``submission``, naming the path that
    failed relative to the submission's parent (``bot/asset.bin``), else the submission."""
    failed = exc.filename
    shown = submission.name
    if isinstance(failed, (str, os.PathLike)) and Path(failed).is_relative_to(submission.parent):
        shown = Path(failed).relative_to(submission.parent).as_posix()
    return LoadError(f"cannot read {shown!r}: {exc.strerror or type(exc).__name__}")


def _read_capped(path: Path, budget: int) -> bytes:
    """Read at most ``budget + 1`` bytes (never following a symlink) to enforce the size cap."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as fh:
        data: bytes = fh.read(budget + 1)
    if len(data) > budget:
        raise LoadError(f"submission is larger than {MAX_BYTES} bytes")
    return data


def _framed(data: bytes) -> bytes:
    return len(data).to_bytes(8, "big") + data


def _utf8_name(rel: str) -> bytes:
    """``rel`` as UTF-8. A name that is not valid UTF-8 on disk decodes to lone surrogates
    (``os.fsdecode``), which have no UTF-8 form: such a submission is refused."""
    try:
        return rel.encode("utf-8")
    except UnicodeEncodeError:
        raise LoadError(f"file name {rel!r} is not valid UTF-8") from None


def submission_sha256(path: Path) -> str:
    """sha256 over length-framed (POSIX relpath, size, bytes) records of every file.

    Refuses symlinks, special files, ``__pycache__``, ``.pyc``, file names that are not valid
    UTF-8 and files or directories that cannot be read; enforces ``MAX_FILES``,
    ``MAX_DIRECTORIES`` and ``MAX_BYTES`` (the latter while reading).
    """
    return _digest(path)[0]


def _digest(path: Path, entry: Path | None = None) -> tuple[str, bytes | None]:
    """The submission sha256 of ``path`` and the exact bytes of ``entry`` that went into it
    (``None`` if ``entry`` is not one of its files)."""
    digest = hashlib.sha256(SHA_TAG)
    root = path if path.is_dir() and not path.is_symlink() else path.parent
    remaining = MAX_BYTES
    source: bytes | None = None
    for f in _files(path):
        name = _utf8_name(_rel(f, root))
        try:
            data = _read_capped(f, remaining)
        except OSError as exc:
            raise _unreadable(exc, path) from exc
        remaining -= len(data)
        digest.update(_framed(name))
        digest.update(_framed(data))
        if f == entry:
            source = data
    return digest.hexdigest(), source


def _module_name(sha: str) -> str:
    return f"topple_submission_{sha[:16]}"


def _import(entry: Path, sha: str, source: bytes) -> ModuleType:
    """Run ``source``, the entry's hashed bytes, as a new module named after ``sha``.

    The code is compiled from exactly those bytes, as Python's source loader would compile them
    (``dont_inherit``: none of this module's ``__future__`` flags), but nothing is read again
    and no bytecode cache is consulted: a planted ``__pycache__`` (an unchecked-hash .pyc runs
    without being compared to its source) or a file swapped after it was hashed changes nothing.
    """
    name = _module_name(sha)
    spec = importlib.util.spec_from_file_location(name, entry)
    if spec is None:
        raise LoadError(f"cannot import {entry}")
    module = importlib.util.module_from_spec(spec)  # __file__, __spec__, ... as for an import
    sys.modules[name] = module
    stop_writing_bytecode()
    try:
        exec(compile(source, str(entry), "exec", dont_inherit=True), module.__dict__)
    except KeyboardInterrupt:
        sys.modules.pop(name, None)
        raise
    except BaseException as exc:  # SystemExit, GeneratorExit, ...: all the submission's fault
        sys.modules.pop(name, None)
        raise LoadError(f"importing {entry.name} failed: {describe_exception(exc)}") from exc
    return module


def stop_writing_bytecode() -> None:
    """Turn bytecode writing off for the rest of this process (``sys.dont_write_bytecode``).

    A ``__pycache__`` next to the submission would change the directory we hashed and make the
    next load refuse it. The submission's code can import a sibling module at any time, in
    ``make_controller``, ``reset`` or ``act``, not only while it is imported, so the flag is never
    restored: once a submission is loaded, nothing in this process writes bytecode. Restoring it
    after each call would also race with other threads.
    """
    sys.dont_write_bytecode = True


def _factory_arity(raw: Callable[..., Any]) -> tuple[int, bool]:
    """(required positional parameters, whether there are any parameters) of ``raw``."""
    parameters = list(inspect.signature(raw).parameters.values())
    required = [
        p
        for p in parameters
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(required), bool(parameters)


def _wrap_factory(raw: Callable[..., Any], entry: Path) -> Callable[[TaskSpec], Controller[Any]]:
    try:  # a signature can run submission code (``__signature__``, ``__wrapped__``)
        required, has_params = _factory_arity(raw)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        raise LoadError(
            f"{FACTORY} in {entry.name} has no usable signature: {describe_exception(exc)}"
        ) from exc
    if required > 1:
        raise LoadError(f"{FACTORY} must take (spec) or no arguments, got {required} required")
    takes_spec = required == 1 or has_params

    def factory(spec: TaskSpec) -> Controller[Any]:
        controller: Controller[Any] = raw(spec) if takes_spec else raw()
        if not _has_controller_methods(controller):
            raise LoadError(
                f"{FACTORY}() returned {class_name(controller)}, which lacks reset()/act()"
            )
        return controller

    return factory


def _has_controller_methods(obj: object) -> bool:
    """Whether ``obj`` has ``reset`` and ``act``, looked up as ``isinstance(obj, Controller)``
    would (``inspect.getattr_static``: the instance dict and the class, nothing dynamic), but
    without reading ``obj.__class__``: the harness never asks a controller its class (§8)."""
    return all(inspect.getattr_static(obj, name, None) is not None for name in CONTROLLER_METHODS)


def _factory_of(module: ModuleType, entry: Path) -> object:
    """The module's ``make_controller`` (``None`` if absent). The module's own attribute lookup
    may run submission code (a ``ModuleType`` subclass); whatever it raises is a ``LoadError``."""
    try:
        return getattr(module, FACTORY, None)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        raise LoadError(
            f"reading {FACTORY} from {entry.name} failed: {describe_exception(exc)}"
        ) from exc


def submission_id(path: Path) -> str:
    """The controller id of the submission at ``path``: the resolved directory's name or the
    resolved file's stem, so ``.``, ``bot/..`` and ``bot`` all name the directory itself."""
    resolved = path.resolve()
    return resolved.stem if resolved.is_file() else resolved.name


def load_submission(path: Path) -> Submission:
    _utf8_name(path.name)  # the name becomes the controller id
    if path.is_symlink():
        raise LoadError(f"{path} is a symlink; submit the file or directory itself")
    try:
        path = path.resolve()
    except (OSError, RuntimeError) as exc:  # RuntimeError: a symlink loop on the way
        raise LoadError(f"cannot resolve {path}: {exc}") from exc
    if not path.exists():
        raise LoadError(f"{path} does not exist")
    entry = path / ENTRY_FILE if path.is_dir() else path
    if entry.suffix != ".py" or not entry.is_file():
        raise LoadError(f"expected a .py file or a directory containing {ENTRY_FILE}")
    sha, source = _digest(path, entry)
    if source is None:  # cannot happen: the entry is always one of the hashed files
        raise LoadError(f"{entry.name} was not hashed")
    module = _import(entry, sha, source)
    try:
        factory = _factory(module, entry)
    except LoadError:
        sys.modules.pop(_module_name(sha), None)  # a failed load leaves nothing behind
        raise
    return Submission(id=submission_id(path), sha256=sha, entry=entry, factory=factory)


def _factory(module: ModuleType, entry: Path) -> Callable[[TaskSpec], Controller[Any]]:
    """The submission's checked ``make_controller``, wrapped; ``LoadError`` if it has none."""
    raw = _factory_of(module, entry)
    if not callable(raw):
        raise LoadError(f"{entry.name} must define {FACTORY}(spec)")
    return _wrap_factory(raw, entry)
