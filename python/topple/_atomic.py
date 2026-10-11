"""Write an output file atomically, never through a symlink planted at its name (contract §14).

Output directories such as ``out/run`` can come from a checkout of someone else's branch, so a
symlink there (``traj_000.npz -> ~/.zshrc``) must never redirect a write to its target.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

OUTPUT_MODE = 0o644
"""Mode of every output file, whatever the umask: trajectories and videos are artifacts that CI
uploads and other people read."""


@contextmanager
def replacement(path: Path, *, suffix: str = "") -> Iterator[Path]:
    """A new, empty file next to ``path`` for the block to write ``path``'s content into.

    The file gets a random name and is created exclusively (``tempfile.mkstemp``), so it is never
    a link someone planted, with mode ``OUTPUT_MODE``. When the block succeeds it is renamed over
    ``path``, and a rename replaces a symlink at ``path`` instead of writing through it. When the
    block fails, the file is removed and ``path`` is left as it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=suffix)
    tmp = Path(name)
    try:
        try:
            _set_mode(fd, tmp)
        finally:
            os.close(fd)
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _set_mode(fd: int, path: Path) -> None:
    """``OUTPUT_MODE`` on the new file; through its path where ``os.fchmod`` does not exist
    (Windows before Python 3.13)."""
    if hasattr(os, "fchmod"):
        os.fchmod(fd, OUTPUT_MODE)
    else:
        os.chmod(path, OUTPUT_MODE)
