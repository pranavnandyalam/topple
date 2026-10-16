"""``.npz`` archives and ``.npy`` members read by hand (docs/contract.md §14,
docs/design/phase-2.md §3.3).

Trajectory files (``topple.traj``) and the run-directory inputs of ``topple export``
(``actor.npz``, ``parity_ref.npz``) are untrusted ``.npz`` files: zip archives of ``.npy``
members. Nothing here unpickles or evaluates anything, and nothing allocates more than a bounded
read at a time: the archive must be numpy's canonical layout (``topple._zip``) with exactly the
expected member names, every npy header is parsed by hand (never evaluated) and its shape must
pass the caller's limits and need exactly the bytes the member holds before any data is read, so
a header declaring 2^42 elements allocates nothing.

``read_member`` and its helpers are the trajectory reader's, moved here unchanged;
``read_npz_bytes`` is the export inputs' reader on top of them: an in-memory archive whose
members' declared sizes are also capped in total before any member is read (``Archive`` has no
total cap), and exact shapes for every member.
"""

from __future__ import annotations

import io
import math
import re
import struct
from collections.abc import Callable, Mapping
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from topple._zip import READ_CHUNK, Archive, MemberReader

MAX_NPY_HEADER_BYTES: Final = 4096
"""Longest npy header (numpy writes 118 bytes for these arrays)."""
EXPORT_DTYPES: Final = frozenset({"<f4", "<f8", "|u1", "<U64"})
"""The only dtypes an export input may hold (§3.3 step 2; ``<U64`` is ``onnx_sha256``)."""
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


def read_member(
    archive: Archive,
    name: str,
    dtype: np.dtype[Any],
    limits: Callable[[tuple[int, ...]], None],
) -> npt.NDArray[Any]:
    """One ``.npy`` member, read by hand so a lying header can never trigger a big allocation:
    its shape must pass ``limits`` and need exactly the bytes the member holds before any data
    is read, and the member must then end exactly there (``MemberReader.finish``)."""
    member = archive.open(f"{name}.npy")
    shape = read_header(member, name, dtype)
    limits(shape)
    nbytes = math.prod(shape) * dtype.itemsize
    left = member.size - member.tell()
    if nbytes > left:
        raise ValueError(f"{name}: header shape {shape} needs more data than the member holds")
    if nbytes < left:
        raise ValueError(f"{name}: member size does not match its header shape {shape}")
    data = read_bounded(member, nbytes)
    if len(data) != nbytes:
        raise ValueError(f"{name}: the member holds less data than its size declares")
    member.finish()
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def read_bounded(fh: MemberReader, size: int) -> bytes:
    """Up to ``size`` bytes of a zip member, asking for at most ``READ_CHUNK`` at a time.

    A deflated member inflates at most as many bytes as one read asks for, so no single read can
    balloon, whatever the member declares.
    """
    parts: list[bytes] = []
    while size > 0:
        chunk = fh.read(min(size, READ_CHUNK))
        if not chunk:
            break
        parts.append(chunk)
        size -= len(chunk)
    return b"".join(parts)


def read_header(fh: MemberReader, name: str, dtype: np.dtype[Any]) -> tuple[int, ...]:
    """The shape in the npy header of member ``name``, which must hold C-ordered ``dtype``.

    The header's length is checked before the header is read, and its text is parsed here,
    never evaluated (``ast.literal_eval`` can exhaust the parser on a crafted header).
    """
    magic = read_bounded(fh, len(_NPY_MAGIC) + 2)
    if len(magic) != len(_NPY_MAGIC) + 2 or not magic.startswith(_NPY_MAGIC):
        raise ValueError(f"{name}: not an npy member")
    version = (magic[-2], magic[-1])
    length_format = _NPY_LENGTH.get(version)
    if length_format is None:
        raise ValueError(f"{name}: unsupported npy version {version}")
    raw_length = read_bounded(fh, length_format.size)
    if len(raw_length) != length_format.size:
        raise ValueError(f"{name}: truncated npy header")
    (length,) = length_format.unpack(raw_length)
    if length > MAX_NPY_HEADER_BYTES:
        raise ValueError(f"{name}: npy header has {length} bytes, more than {MAX_NPY_HEADER_BYTES}")
    header = read_bounded(fh, length)
    if len(header) != length:
        raise ValueError(f"{name}: truncated npy header")
    return parse_header(header.decode("latin1"), name, dtype)


def parse_header(text: str, name: str, dtype: np.dtype[Any]) -> tuple[int, ...]:
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
    shape = parse_shape(match["shape"], name)
    if any(n < 0 for n in shape):
        raise ValueError(f"{name}: negative dimension in header shape {shape}")
    return shape


def parse_shape(text: str, name: str) -> tuple[int, ...]:
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


# --- export inputs ------------------------------------------------------------------------------


def read_npz_bytes(
    data: bytes,
    schema: Mapping[str, tuple[str, tuple[int, ...]]],
    *,
    max_bytes: int,
    max_directory_bytes: int = 16 * 1024,
) -> dict[str, npt.NDArray[Any]]:
    """The arrays of the ``.npz`` file ``data`` (already read, at most ``max_bytes``), which must
    hold exactly ``schema``'s members (``name -> (dtype, exact shape)``), each in one of
    ``EXPORT_DTYPES``. ``ValueError`` for anything else.

    Before any member is read: the canonical layout with exactly those names, a central
    directory of at most ``max_directory_bytes``, each member at most ``max_bytes`` and the
    members' declared sizes at most ``max_bytes`` in total. Each header must declare the exact
    dtype and shape (so its data size is known and bounded) before its data is read.
    """
    for name, (dtype, _) in schema.items():
        if dtype not in EXPORT_DTYPES:
            raise ValueError(f"{name}: dtype {dtype} is not an export input dtype")
    if len(data) > max_bytes:
        raise ValueError(f"the file has {len(data)} bytes, more than {max_bytes}")
    archive = Archive(
        io.BytesIO(data),
        len(data),
        frozenset(f"{name}.npy" for name in schema),
        max_directory_bytes=max_directory_bytes,
        max_member_bytes=max_bytes,
    )
    total = sum(m.size for m in archive.members.values())
    if total > max_bytes:
        raise ValueError(f"the members hold {total} bytes in total, more than {max_bytes}")
    return {
        name: read_member(archive, name, np.dtype(dtype), _exactly(name, shape))
        for name, (dtype, shape) in schema.items()
    }


def _exactly(name: str, expected: tuple[int, ...]) -> Callable[[tuple[int, ...]], None]:
    def check(shape: tuple[int, ...]) -> None:
        if shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got header shape {shape}")

    return check
