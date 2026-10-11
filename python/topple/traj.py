"""Episode trajectories: ``topple.traj/1`` (docs/contract.md §14).

A trajectory is the complete, replayable record of one episode. Files are plain ``.npz`` with a
small allow-list of dtypes so browsers can decode them. A file is untrusted: loading parses every
member by hand (never unpickles, never allocates more than the file holds) and is only ever scored
through trusted verification (``topple.verify``, contract §16).
"""

from __future__ import annotations

import json
import math
import os
import stat
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import IO, Any, NoReturn

import numpy as np
import numpy.typing as npt

from topple.contract import BALANCE_V1_ACTION, FEATURES

FORMAT = "topple.traj/1"
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_STATE_DIM = 256
"""Most columns ``qpos``, ``qvel`` and ``epilogue_qpos`` may have (nq, nv <= this)."""
MAX_META_BYTES = 256 * 1024
"""Most bytes of ``meta_json`` (an honest one is a few KiB)."""
META = "meta_json"
_ALLOWED_DTYPES = {np.dtype("<f8"), np.dtype("<f4"), np.dtype("<i4"), np.dtype("|u1")}
# name -> (dtype, fixed trailing shape, or None for the model-dependent nq / nv widths)
_SCHEMA: dict[str, tuple[str, tuple[int, ...] | None]] = {
    "qpos": ("<f8", None),
    "qvel": ("<f8", None),
    "obs": ("<f4", (len(FEATURES),)),
    "action": ("<f8", (BALANCE_V1_ACTION.dim,)),
    "ctrl": ("<f8", (BALANCE_V1_ACTION.dim,)),
    "enc_counts": ("<i4", (2,)),
    "gyro": ("<f4", (3,)),
    "accel": ("<f4", (3,)),
    "push_force": ("<f8", (3,)),
    "epilogue_qpos": ("<f8", None),
}
_STATE = ("qpos", "qvel")
_MEMBERS = frozenset({f"{n}.npy" for n in _SCHEMA} | {f"{META}.npy"})


def _frozen(arr: npt.ArrayLike, dtype: str) -> npt.NDArray[Any]:
    out = np.ascontiguousarray(np.asarray(arr, dtype=np.dtype(dtype)))
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Trajectory:
    qpos: npt.NDArray[np.float64]  # (T+1, nq)
    qvel: npt.NDArray[np.float64]  # (T+1, nv)
    obs: npt.NDArray[np.float32]  # (T, n_obs)
    action: npt.NDArray[np.float64]  # (T, nu) raw controller output before clipping
    ctrl: npt.NDArray[np.float64]  # (T, nu) clipped command
    enc_counts: npt.NDArray[np.int32]  # (T, 2)
    gyro: npt.NDArray[np.float32]  # (T, 3)
    accel: npt.NDArray[np.float32]  # (T, 3)
    push_force: npt.NDArray[np.float64]  # (T, 3)
    epilogue_qpos: npt.NDArray[np.float64]  # (E, nq)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, (dtype, _) in _SCHEMA.items():
            object.__setattr__(self, name, _frozen(getattr(self, name), dtype))
        self._check_shapes()

    @property
    def steps(self) -> int:
        return int(self.ctrl.shape[0])

    def _check_shapes(self) -> None:
        t = self.steps
        for name in _STATE:
            arr = getattr(self, name)
            if arr.ndim != 2 or arr.shape[0] != t + 1 or arr.shape[1] < 1:
                raise ValueError(
                    f"{name} must have T+1={t + 1} rows and at least one column, "
                    f"got shape {arr.shape}"
                )
            if arr.shape[1] > MAX_STATE_DIM:
                raise ValueError(
                    f"{name} must have at most {MAX_STATE_DIM} columns, got shape {arr.shape}"
                )
        for name, (_, trailing) in _SCHEMA.items():
            if trailing is None:
                continue
            arr = getattr(self, name)
            if arr.shape != (t, *trailing):
                raise ValueError(f"{name} must have shape {(t, *trailing)}, got {arr.shape}")
        if self.epilogue_qpos.ndim != 2 or self.epilogue_qpos.shape[1] != self.qpos.shape[1]:
            raise ValueError(
                f"epilogue_qpos must have {self.qpos.shape[1]} columns (nq), "
                f"got shape {self.epilogue_qpos.shape}"
            )

    def save_npz(self, path: Path) -> None:
        meta = {"format": FORMAT, **self.meta}
        arrays = {name: getattr(self, name) for name in _SCHEMA}
        arrays["meta_json"] = np.frombuffer(
            json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
        for name, arr in arrays.items():
            if arr.dtype not in _ALLOWED_DTYPES or not arr.flags.c_contiguous:
                raise ValueError(f"{name}: dtype {arr.dtype} not allowed in {FORMAT}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)


def load_npz(path: Path, *, max_steps: int) -> Trajectory:
    """Load and validate an untrusted trajectory file (contract §14).

    Nothing is unpickled and nothing is allocated beyond what the file really holds or the format
    allows: every member is size-capped, and its npy header is checked against its size and
    against the limits (T and E at most ``max_steps``, state widths at most ``MAX_STATE_DIM``,
    ``meta_json`` at most ``MAX_META_BYTES``) before any of its data is read. Any problem
    (including an unreadable or missing file) raises ``ValueError``; ``max_steps`` is required.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError(f"max_steps must be a non-negative int, got {max_steps!r}")
    return _guarded(path, lambda: _build(path, max_steps))


def read_meta(path: Path) -> dict[str, Any]:
    """Only the metadata of a trajectory file, validated like ``load_npz`` (no arrays are read)."""
    return _guarded(path, lambda: _read_meta_only(path))


def _guarded[R](path: Path, read: Callable[[], R]) -> R:
    """Run ``read``; report any failure as a ``ValueError`` naming the file."""
    try:
        return read()
    except MemoryError:
        raise
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    except Exception as exc:  # BadZipFile, OSError, RecursionError, zlib.error, EOFError, ...
        raise ValueError(
            f"{path.name}: not a readable {FORMAT} file ({type(exc).__name__})"
        ) from exc


def _open_regular(path: Path) -> IO[bytes]:
    """Open ``path`` for reading only if it is a regular file (after following symlinks).

    The open itself never blocks: a FIFO would wait forever for a writer and a device can be
    endless, so the type is checked on the opened descriptor (no race with a swapped path).
    """
    nonblock = getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | nonblock)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("not a regular file")
        if nonblock:
            os.set_blocking(fd, True)
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def _open_members(path: Path) -> Iterator[zipfile.ZipFile]:
    """The zip archive at ``path``, once its member list and member sizes are acceptable."""
    with _open_regular(path) as fh, zipfile.ZipFile(fh) as zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise ValueError("duplicate members")
        if set(names) != _MEMBERS:
            raise ValueError(f"unexpected members {sorted(set(names) ^ _MEMBERS)}")
        for info in infos:
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"{info.filename} is too large ({info.file_size} bytes)")
        yield zf


def _build(path: Path, max_steps: int) -> Trajectory:
    with _open_members(path) as zf:
        arrays = {
            name: _read_member(
                zf, name, np.dtype(dtype), partial(_array_limits, name, max_steps=max_steps)
            )
            for name, (dtype, _) in _SCHEMA.items()
        }
        meta = _read_meta_member(zf)
    return Trajectory(**arrays, meta=meta)


def _read_meta_only(path: Path) -> dict[str, Any]:
    with _open_members(path) as zf:
        return _read_meta_member(zf)


def _read_meta_member(zf: zipfile.ZipFile) -> dict[str, Any]:
    return _parse_meta(_read_member(zf, META, np.dtype("|u1"), _meta_limits))


def _array_limits(name: str, shape: tuple[int, ...], *, max_steps: int) -> None:
    """Refuse an array header beyond the format's limits (§14) before its data is read.

    Rows: T+1 <= max_steps+1 for ``qpos``/``qvel``, T <= max_steps for the per-step arrays and
    E <= max_steps for the epilogue. Columns: the fixed width of the table, or at most
    ``MAX_STATE_DIM`` for the state arrays. Exact shapes are checked once the arrays are read.
    """
    if len(shape) != 2:
        raise ValueError(f"{name} must be 2-D, got header shape {shape}")
    rows, width = shape
    trailing = _SCHEMA[name][1]
    if name == "epilogue_qpos":
        if rows > max_steps:
            raise ValueError(f"epilogue_qpos has {rows} rows, more than {max_steps}")
    else:
        steps = rows - 1 if name in _STATE else rows
        if steps > max_steps:
            raise ValueError(f"{name}: trajectory has {steps} steps, more than {max_steps}")
    if trailing is None:
        if width > MAX_STATE_DIM:
            raise ValueError(f"{name} must have at most {MAX_STATE_DIM} columns, got shape {shape}")
    elif (width,) != trailing:
        raise ValueError(f"{name} must have shape {(rows, *trailing)}, got {shape}")


def _meta_limits(shape: tuple[int, ...]) -> None:
    size = math.prod(shape)
    if size > MAX_META_BYTES:
        raise ValueError(f"{META} has {size} bytes, more than {MAX_META_BYTES}")


def _read_member(
    zf: zipfile.ZipFile,
    name: str,
    dtype: np.dtype[Any],
    limits: Callable[[tuple[int, ...]], None],
) -> npt.NDArray[Any]:
    """One ``.npy`` member, read by hand so a lying header can never trigger a big allocation:
    its shape must pass ``limits`` and fit the bytes the member holds before any data is read."""
    info = zf.getinfo(f"{name}.npy")
    with zf.open(info) as fh:
        shape, fortran, found = _read_header(fh, name)
        if fortran:
            raise ValueError(f"{name}: Fortran-ordered arrays are not allowed")
        if found.hasobject:
            raise ValueError(f"{name}: Object arrays are not allowed")
        if found != dtype:
            raise ValueError(f"{name}: expected dtype {dtype.str}, got {found.str}")
        if any(n < 0 for n in shape):
            raise ValueError(f"{name}: negative dimension in header shape {shape}")
        limits(shape)
        nbytes = math.prod(shape) * found.itemsize
        if nbytes > info.file_size - fh.tell():
            raise ValueError(f"{name}: header shape {shape} needs more data than the member holds")
        data = fh.read(nbytes)
        if len(data) != nbytes or fh.read(1):
            raise ValueError(f"{name}: member size does not match its header shape {shape}")
    return np.frombuffer(data, dtype=found).reshape(shape)


def _read_header(fh: IO[bytes], name: str) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    else:
        raise ValueError(f"{name}: unsupported npy version {version}")
    return tuple(int(n) for n in shape), bool(fortran), dtype


class _NonFiniteError(ValueError):
    """``meta_json`` holds a number that is not finite (contract §14)."""


def _reject_constant(token: str) -> NoReturn:
    raise _NonFiniteError(f"meta_json: non-finite number {token}")


def _finite_float(literal: str) -> float:
    """A JSON number with a fraction or exponent; ``1e400`` would silently become inf."""
    value = float(literal)
    if not math.isfinite(value):
        shown = literal if len(literal) <= 40 else literal[:37] + "..."
        raise _NonFiniteError(f"meta_json: non-finite number {shown}")
    return value


def _parse_meta(raw: npt.NDArray[Any]) -> dict[str, Any]:
    if raw.ndim != 1:
        raise ValueError("meta_json must be a 1-D uint8 array")
    try:
        meta = json.loads(
            raw.tobytes().decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except RecursionError as exc:
        raise ValueError("meta_json is nested too deeply") from exc
    except _NonFiniteError:
        raise
    except ValueError as exc:
        raise ValueError(f"meta_json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise ValueError(f"meta_json: not a {FORMAT} file")
    meta.pop("format")
    return meta
