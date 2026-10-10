"""Load a submission: a ``.py`` file, or a directory with ``controller.py`` plus assets.

In-process loading runs the submission's code inside this interpreter, so it is for trusted code
only (your own controllers, the shipped examples). Untrusted submissions go through the
subprocess/container runner (Phase 3).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from topple.contract import Controller, TaskSpec

ENTRY_FILE = "controller.py"
FACTORY = "make_controller"
MAX_FILES = 50
MAX_BYTES = 20 * 1024 * 1024


class LoadError(Exception):
    """The submission could not be loaded or does not follow the contract."""


@dataclass(frozen=True)
class Submission:
    id: str
    sha256: str
    entry: Path
    factory: Callable[[TaskSpec], Controller[Any]]


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if len(files) > MAX_FILES:
        raise LoadError(f"submission has {len(files)} files (max {MAX_FILES})")
    return files


def submission_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    root = path if path.is_dir() else path.parent
    total = 0
    for f in _files(path):
        data = f.read_bytes()
        total += len(data)
        if total > MAX_BYTES:
            raise LoadError(f"submission is larger than {MAX_BYTES} bytes")
        digest.update(str(f.relative_to(root)).encode())
        digest.update(data)
    return digest.hexdigest()


def _import(entry: Path, sha: str) -> ModuleType:
    name = f"topple_submission_{sha[:16]}"
    spec = importlib.util.spec_from_file_location(name, entry)
    if spec is None or spec.loader is None:
        raise LoadError(f"cannot import {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise LoadError(f"importing {entry.name} failed: {type(exc).__name__}: {exc}") from exc
    return module


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
        controller = raw(spec) if takes_spec else raw()
        if not isinstance(controller, Controller):
            raise LoadError(
                f"{FACTORY}() returned {type(controller).__name__}, which lacks reset()/act()"
            )
        return controller

    return factory


def load_submission(path: Path) -> Submission:
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
