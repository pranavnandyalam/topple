"""Load a submission: a ``.py`` file, or a directory with ``controller.py`` plus assets.

In-process loading runs the submission's code inside this interpreter, so it is for trusted code
only (your own controllers, the shipped examples). Untrusted submissions go through the
subprocess/container runner (Phase 3).
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

from topple.contract import Controller, TaskSpec, class_name

ENTRY_FILE = "controller.py"
FACTORY = "make_controller"
MAX_FILES = 50
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
    """True for a directory to descend into, False for a regular file; else ``LoadError``."""
    if entry.is_symlink():
        raise LoadError(f"symlinks are not allowed in a submission: {rel}")
    if entry.is_dir(follow_symlinks=False):
        if entry.name == "__pycache__":
            raise LoadError(f"remove compiled Python from the submission: {rel}")
        return True
    if not entry.is_file(follow_symlinks=False):
        raise LoadError(f"{rel} is not a regular file")
    if entry.name.endswith(".pyc"):
        raise LoadError(f"remove compiled Python (.pyc) from the submission: {rel}")
    return False


def _files(path: Path) -> list[Path]:
    """Every file of a submission in POSIX-path order, refusing anything that is not plain data."""
    if path.is_symlink():
        raise LoadError(f"{path.name} is a symlink; submit the file itself")
    if path.is_file():
        return [path]
    files: list[Path] = []
    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                child = Path(entry.path)
                if _check_entry(entry, _rel(child, path)):
                    pending.append(child)
                else:
                    files.append(child)
                if len(files) > MAX_FILES:
                    raise LoadError(f"submission has more than {MAX_FILES} files")
    return sorted(files, key=lambda f: _rel(f, path))


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


def submission_sha256(path: Path) -> str:
    """sha256 over length-framed (POSIX relpath, size, bytes) records of every file.

    Refuses symlinks, special files, ``__pycache__`` and ``.pyc``; enforces ``MAX_FILES`` and
    ``MAX_BYTES`` (the latter while reading).
    """
    digest = hashlib.sha256(SHA_TAG)
    root = path if path.is_dir() and not path.is_symlink() else path.parent
    remaining = MAX_BYTES
    for f in _files(path):
        data = _read_capped(f, remaining)
        remaining -= len(data)
        digest.update(_framed(_rel(f, root).encode("utf-8")))
        digest.update(_framed(data))
    return digest.hexdigest()


def _import(entry: Path, sha: str) -> ModuleType:
    name = f"topple_submission_{sha[:16]}"
    spec = importlib.util.spec_from_file_location(name, entry)
    if spec is None or spec.loader is None:
        raise LoadError(f"cannot import {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    stop_writing_bytecode()
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise LoadError(f"importing {entry.name} failed: {type(exc).__name__}: {exc}") from exc
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


def _wrap_factory(raw: Callable[..., Any], entry: Path) -> Callable[[TaskSpec], Controller[Any]]:
    try:
        params = [
            p
            for p in inspect.signature(raw).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError) as exc:
        raise LoadError(f"{FACTORY} in {entry.name} has no usable signature") from exc
    if len(params) > 1:
        raise LoadError(f"{FACTORY} must take (spec) or no arguments, got {len(params)} required")
    takes_spec = len(params) == 1 or bool(inspect.signature(raw).parameters)

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


def load_submission(path: Path) -> Submission:
    if path.is_symlink():
        raise LoadError(f"{path} is a symlink; submit the file or directory itself")
    path = path.resolve()
    if not path.exists():
        raise LoadError(f"{path} does not exist")
    entry = path / ENTRY_FILE if path.is_dir() else path
    if entry.suffix != ".py" or not entry.is_file():
        raise LoadError(f"expected a .py file or a directory containing {ENTRY_FILE}")
    sha = submission_sha256(path)
    module = _import(entry, sha)
    raw = getattr(module, FACTORY, None)
    if not callable(raw):
        raise LoadError(f"{entry.name} must define {FACTORY}(spec)")
    return Submission(
        id=path.stem if path.is_file() else path.name,
        sha256=sha,
        entry=entry,
        factory=_wrap_factory(raw, entry),
    )
