"""Open untrusted input files without blocking on them (docs/contract.md §14, §15).

An uploaded ``traj_*.npz`` or ``results.json`` is a path someone else chose. A FIFO there makes
a plain ``open`` wait forever for a writer, and a device such as ``/dev/zero`` never ends, so
inputs are opened without blocking and their type is checked on the opened descriptor before
anything is read (no race with a path swapped in between).

``read_regular_file`` is the stricter sibling for files inside submission and run directories
(docs/design/phase-2.md §3.4): no symlinks and no hard links at all, and the file it reads is
the very file it checked.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import IO


def open_regular(path: Path, max_bytes: int) -> tuple[IO[bytes], int]:
    """``path`` opened for reading, with its size, if it is a regular file (symlinks are
    followed) of at most ``max_bytes``; ``ValueError`` if it is anything else, ``OSError`` if it
    cannot be opened at all. The open itself never blocks."""
    nonblock = getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | nonblock)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("not a regular file")
        if info.st_size > max_bytes:
            raise ValueError(f"the file has {info.st_size} bytes, more than {max_bytes}")
        if nonblock:
            os.set_blocking(fd, True)
        return os.fdopen(fd, "rb"), info.st_size
    except BaseException:
        os.close(fd)
        raise


_READ_CHUNK = 1 << 20


def _flag(name: str) -> int:
    """``os.<name>``, or 0 where the platform lacks it (win32 has only ``O_BINARY`` of these)."""
    flag: int = getattr(os, name, 0)
    return flag


def read_regular_file(path: Path, cap: int) -> bytes:
    """The bytes of ``path`` if it is a regular file with one link and at most ``cap`` bytes;
    any other outcome is ``topple.loader.LoadError`` naming the file (contract §8).

    ``lstat`` first (no symlink, ``S_ISREG``, ``st_nlink == 1``, size), then an open that cannot
    follow a symlink (``O_NOFOLLOW``), block on a FIFO swapped in after the check
    (``O_NONBLOCK``), become a controlling terminal or leak into a child; the opened descriptor
    must be a regular file with the lstat's (``st_dev``, ``st_ino``) and still one link, which
    carries the no-symlink guarantee where ``O_NOFOLLOW`` is missing. At most ``cap + 1`` bytes
    are read, so a file that grows after the check is refused without reading it all.
    """
    try:
        return _read_regular(path, cap)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise _load_error(f"cannot read {path.name!r}: {reason}") from exc


def _read_regular(path: Path, cap: int) -> bytes:
    name = path.name
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raise _load_error(f"{name!r} is a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise _load_error(f"{name!r} is not a regular file")
    if info.st_nlink != 1:
        raise _load_error(f"{name!r} has {info.st_nlink} hard links")
    if info.st_size > cap:
        raise _load_error(f"{name!r} is larger than {cap} bytes")
    flags = os.O_RDONLY
    for flag in ("O_NOFOLLOW", "O_NONBLOCK", "O_NOCTTY", "O_CLOEXEC", "O_BINARY"):
        flags |= _flag(flag)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
            or opened.st_nlink != 1
        ):
            raise _load_error(f"{name!r} changed while it was opened")
        data = _read_at_most(fd, cap + 1)
    finally:
        os.close(fd)
    if len(data) > cap:
        raise _load_error(f"{name!r} is larger than {cap} bytes")
    return data


def _read_at_most(fd: int, limit: int) -> bytes:
    """Up to ``limit`` bytes from ``fd`` (fewer only at end of file)."""
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(fd, min(remaining, _READ_CHUNK))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _load_error(message: str) -> Exception:
    """A ``topple.loader.LoadError``. Imported on use: the loader reads submission files
    through this module, so a module-level import would be a cycle."""
    from topple.loader import LoadError

    return LoadError(message)
