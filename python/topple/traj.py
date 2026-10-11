"""Episode trajectories: ``topple.traj/1`` (docs/contract.md §14).

A trajectory is the complete, replayable record of one episode. Files are plain ``.npz`` with a
small allow-list of dtypes so browsers can decode them. A file is untrusted: loading parses the
archive and every member by hand (never unpickles, never evaluates a header, never allocates more
than the format allows) and is only ever scored through trusted verification (``topple.verify``,
contract §16).
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import struct
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import IO, Any, NoReturn

import numpy as np
import numpy.typing as npt

from topple._atomic import replacement
from topple.contract import BALANCE_V1_ACTION, FEATURES

FORMAT = "topple.traj/1"
MAX_FILE_BYTES = 32 * 1024 * 1024
"""Largest trajectory file ``load_npz`` opens (an honest balance-v1 file is about 0.2 MB)."""
MAX_DIRECTORY_BYTES = 16 * 1024
"""Largest zip central directory (an honest one lists its 11 members in under 1 KiB)."""
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 4096
"""Longest npy header (numpy writes 118 bytes for these arrays)."""
MAX_STATE_DIM = 256
"""Most columns ``qpos``, ``qvel`` and ``epilogue_qpos`` may have (nq, nv <= this)."""
MAX_META_BYTES = 256 * 1024
"""Most bytes of ``meta_json`` (an honest one is a few KiB)."""
META = "meta_json"
_READ_CHUNK = 64 * 1024  # most bytes one read asks a zip member for
_COMPRESSION = {zipfile.ZIP_STORED: "stored", zipfile.ZIP_DEFLATED: "deflated"}
_ENCRYPTED = 0x1  # general purpose flag bit 0
_END_RECORD = struct.Struct("<4s4H2LH")  # end of central directory, without its comment
_END_SIGNATURE = b"PK\x05\x06"
_ZIP64_LOCATOR = b"PK\x06\x07"
_ZIP64_LOCATOR_BYTES = 20
_NPY_MAGIC = b"\x93NUMPY"
_NPY_LENGTH = {(1, 0): struct.Struct("<H"), (2, 0): struct.Struct("<I")}
_NPY_HEADER = re.compile(
    r"\{\s*'descr'\s*:\s*'(?P<descr>[\x20-\x26\x28-\x5b\x5d-\x7e]{1,32})'\s*,"
    r"\s*'fortran_order'\s*:\s*(?P<fortran_order>True|False)\s*,"
    r"\s*'shape'\s*:\s*\((?P<shape>[0-9,\s-]{0,256})\)\s*(?:,\s*)?\}\s*",
    re.ASCII,
)
"""The header dict exactly as numpy writes it (key order fixed, whitespace and a trailing comma
free): parsed here, never handed to a Python parser."""
_NPY_DIM = re.compile(r"-?(?:0|[1-9][0-9]{0,18})", re.ASCII)
_SHOWN_NAMES = 5  # unexpected member names quoted in an error
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
        """Write the trajectory to ``path`` atomically (contract §14): through a new temporary
        file, mode 0644, renamed over ``path``, so a symlink at ``path`` is replaced, never
        written through, and a failed write leaves ``path`` as it was."""
        meta = {"format": FORMAT, **self.meta}
        arrays = {name: getattr(self, name) for name in _SCHEMA}
        arrays["meta_json"] = np.frombuffer(
            json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
        for name, arr in arrays.items():
            if arr.dtype not in _ALLOWED_DTYPES or not arr.flags.c_contiguous:
                raise ValueError(f"{name}: dtype {arr.dtype} not allowed in {FORMAT}")
        with replacement(path) as tmp, tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)


def load_npz(path: Path, *, max_steps: int) -> Trajectory:
    """Load and validate an untrusted trajectory file (contract §14).

    Nothing is unpickled or evaluated, and nothing is allocated beyond what the format allows:
    the file, its zip directory and every member are size-capped before they are parsed, only
    stored or deflated members are read, at most ``_READ_CHUNK`` bytes at a time, and each npy
    header is parsed by hand and checked against the member's size and against the limits (T and
    E at most ``max_steps``, state widths at most ``MAX_STATE_DIM``, ``meta_json`` at most
    ``MAX_META_BYTES``) before any of its data is read. Any problem (including an unreadable or
    missing file) raises ``ValueError``; ``max_steps`` is required.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError(f"max_steps must be a non-negative int, got {max_steps!r}")
    return _guarded(path, lambda: _build(path, max_steps))


def read_meta(path: Path) -> dict[str, Any]:
    """Only the metadata of a trajectory file, validated like ``load_npz`` (no arrays are read)."""
    return _guarded(path, lambda: _read_meta_only(path))


def _guarded[R](path: Path, read: Callable[[], R]) -> R:
    """Run ``read``; report any failure as a ``ValueError`` naming the file.

    Only ``MemoryError`` passes through: nothing in the file can make the loader ask for more
    than the format's limits allow (no header is ever evaluated), so it is a real shortage.
    """
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


def _open_regular(path: Path) -> tuple[IO[bytes], int]:
    """Open ``path`` for reading only if it is a regular file (after following symlinks) of at
    most ``MAX_FILE_BYTES``; return it with its size.

    The open itself never blocks: a FIFO would wait forever for a writer and a device can be
    endless, so the type is checked on the opened descriptor (no race with a swapped path).
    """
    nonblock = getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | nonblock)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("not a regular file")
        if info.st_size > MAX_FILE_BYTES:
            raise ValueError(f"the file has {info.st_size} bytes, more than {MAX_FILE_BYTES}")
        if nonblock:
            os.set_blocking(fd, True)
        return os.fdopen(fd, "rb"), info.st_size
    except BaseException:
        os.close(fd)
        raise


def _check_end_record(fh: IO[bytes], size: int) -> None:
    """Refuse an archive whose zip directory ``zipfile`` could not parse cheaply.

    ``zipfile`` parses the whole central directory, one object per entry, before any member can
    be checked, and it takes the directory's size from the end-of-central-directory record. That
    record must end the file (no archive comment) with no zip64 locator before it, so
    ``zipfile`` reads exactly the record checked here, and its directory must be at most
    ``MAX_DIRECTORY_BYTES``.
    """
    tail = min(size, _END_RECORD.size + _ZIP64_LOCATOR_BYTES)
    fh.seek(size - tail)
    data = fh.read(tail)
    if len(data) < _END_RECORD.size:
        raise ValueError("not a zip archive")
    signature, disk, directory_disk, _, _, directory_bytes, _, comment = _END_RECORD.unpack(
        data[-_END_RECORD.size :]
    )
    if signature != _END_SIGNATURE or comment:
        raise ValueError("not a zip archive whose end-of-central-directory record ends the file")
    if len(data) == _END_RECORD.size + _ZIP64_LOCATOR_BYTES and data.startswith(_ZIP64_LOCATOR):
        raise ValueError("zip64 archives are not allowed")
    if disk or directory_disk:
        raise ValueError("multi-disk zip archives are not allowed")
    if directory_bytes > MAX_DIRECTORY_BYTES:
        raise ValueError(
            f"the zip central directory has {directory_bytes} bytes, more than "
            f"{MAX_DIRECTORY_BYTES}"
        )
    fh.seek(0)


@contextmanager
def _open_members(path: Path) -> Iterator[zipfile.ZipFile]:
    """The zip archive at ``path``, once its directory, member list and members are acceptable."""
    fh, size = _open_regular(path)
    with fh:
        _check_end_record(fh, size)
        with zipfile.ZipFile(fh) as zf:
            _check_members(zf.infolist())
            yield zf


def _check_members(infos: list[zipfile.ZipInfo]) -> None:
    """Exactly the format's members, each stored or deflated, unencrypted and size-capped."""
    names = [info.filename for info in infos]
    if len(set(names)) != len(names):
        raise ValueError("duplicate members")
    if set(names) != _MEMBERS:
        raise ValueError(f"unexpected members {_quoted(sorted(set(names) ^ _MEMBERS))}")
    for info in infos:
        if info.compress_type not in _COMPRESSION:
            raise ValueError(
                f"{info.filename}: compression method {info.compress_type} is not allowed "
                f"(only {' or '.join(_COMPRESSION.values())})"
            )
        if info.flag_bits & _ENCRYPTED:
            raise ValueError(f"{info.filename}: encrypted members are not allowed")
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"{info.filename} is too large ({info.file_size} bytes)")


def _quoted(names: list[str]) -> str:
    """The first few ``names``, escaped and shortened: they come from the untrusted archive."""
    shown = ", ".join(_short_repr(name) for name in names[:_SHOWN_NAMES])
    more = len(names) - _SHOWN_NAMES
    return f"[{shown}]" + (f" and {more} more" if more > 0 else "")


def _short_repr(name: str) -> str:
    text = repr(name)
    return text if len(text) <= 40 else text[:37] + "..."


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
        shape = _read_header(fh, name, dtype)
        limits(shape)
        nbytes = math.prod(shape) * dtype.itemsize
        if nbytes > info.file_size - fh.tell():
            raise ValueError(f"{name}: header shape {shape} needs more data than the member holds")
        data = _read_bounded(fh, nbytes)
        if len(data) != nbytes or fh.read(1):
            raise ValueError(f"{name}: member size does not match its header shape {shape}")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _read_bounded(fh: IO[bytes], size: int) -> bytes:
    """Up to ``size`` bytes of a zip member, asking for at most ``_READ_CHUNK`` at a time.

    A deflated member inflates at most as many bytes as one read asks for, so no single read can
    balloon, whatever the member declares.
    """
    parts: list[bytes] = []
    while size > 0:
        chunk = fh.read(min(size, _READ_CHUNK))
        if not chunk:
            break
        parts.append(chunk)
        size -= len(chunk)
    return b"".join(parts)


def _read_header(fh: IO[bytes], name: str, dtype: np.dtype[Any]) -> tuple[int, ...]:
    """The shape in the npy header of member ``name``, which must hold C-ordered ``dtype``.

    The header's length is checked before the header is read, and its text is parsed here,
    never evaluated (``ast.literal_eval`` can exhaust the parser on a crafted header).
    """
    magic = _read_bounded(fh, len(_NPY_MAGIC) + 2)
    if len(magic) != len(_NPY_MAGIC) + 2 or not magic.startswith(_NPY_MAGIC):
        raise ValueError(f"{name}: not an npy member")
    version = (magic[-2], magic[-1])
    length_format = _NPY_LENGTH.get(version)
    if length_format is None:
        raise ValueError(f"{name}: unsupported npy version {version}")
    raw_length = _read_bounded(fh, length_format.size)
    if len(raw_length) != length_format.size:
        raise ValueError(f"{name}: truncated npy header")
    (length,) = length_format.unpack(raw_length)
    if length > MAX_NPY_HEADER_BYTES:
        raise ValueError(f"{name}: npy header has {length} bytes, more than {MAX_NPY_HEADER_BYTES}")
    header = _read_bounded(fh, length)
    if len(header) != length:
        raise ValueError(f"{name}: truncated npy header")
    return _parse_header(header.decode("latin1"), name, dtype)


def _parse_header(text: str, name: str, dtype: np.dtype[Any]) -> tuple[int, ...]:
    match = _NPY_HEADER.fullmatch(text)
    if match is None:
        raise ValueError(f"{name}: malformed npy header")
    if match["fortran_order"] == "True":
        raise ValueError(f"{name}: Fortran-ordered arrays are not allowed")
    descr = match["descr"]
    if descr.lstrip("<>|=").startswith("O"):
        raise ValueError(f"{name}: Object arrays are not allowed")
    if descr != dtype.str:
        raise ValueError(f"{name}: expected dtype {dtype.str}, got {descr}")
    shape = _parse_shape(match["shape"], name)
    if any(n < 0 for n in shape):
        raise ValueError(f"{name}: negative dimension in header shape {shape}")
    return shape


def _parse_shape(text: str, name: str) -> tuple[int, ...]:
    """A tuple literal's contents: ``""`` is ``()``, one dimension needs its trailing comma."""
    parts = [part.strip() for part in text.split(",")]
    if parts == [""]:
        return ()
    if parts[-1] == "":
        parts.pop()
    elif len(parts) == 1:
        raise ValueError(f"{name}: malformed npy header")  # ``(5)`` is an int, not a tuple
    if not all(_NPY_DIM.fullmatch(part) for part in parts):
        raise ValueError(f"{name}: malformed npy header")
    return tuple(int(part) for part in parts)


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
