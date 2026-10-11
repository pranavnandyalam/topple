"""Open untrusted input files without blocking on them (docs/contract.md §14, §15).

An uploaded ``traj_*.npz`` or ``results.json`` is a path someone else chose. A FIFO there makes
a plain ``open`` wait forever for a writer, and a device such as ``/dev/zero`` never ends, so
inputs are opened without blocking and their type is checked on the opened descriptor before
anything is read (no race with a path swapped in between).
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
