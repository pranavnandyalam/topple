"""Write an output file atomically, never through a symlink planted at its name (contract §14).

Output directories such as ``out/run`` can come from a checkout of someone else's branch, so a
symlink there (``traj_000.npz -> ~/.zshrc``) must never redirect a write to its target, and
neither must a symlink committed in place of a directory (``out -> ~/project``), which
``check_output_directory`` refuses for the relative paths the CLI writes to.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

OUTPUT_MODE = 0o644
"""Mode of every output file, whatever the umask: trajectories and videos are artifacts that CI
uploads and other people read."""


class SymlinkedOutputError(OSError):
    """A relative output directory passes through a symlink (contract §14). An ``OSError``, so
    a writer that reports failures to write as "cannot write" reports this one too."""


def check_output_directory(directory: Path) -> None:
    """Raise ``SymlinkedOutputError`` if ``directory``, a relative path, passes through a symlink.

    Each prefix of ``directory`` (``out``, then ``out/run``: the directory itself included) is
    examined without following links, so a checkout that commits ``out`` as a symlink cannot
    send the default ``out/run`` elsewhere. The walk stops at the first prefix that does not
    exist, since the writer creates the rest as real directories. An absolute path is the
    caller's own choice and is not examined (macOS's ``/tmp`` is a symlink). Any other
    ``OSError`` from the examination (a component that is a file, a directory that cannot be
    searched) propagates. The check and the writes that follow are separate steps: a directory
    swapped for a symlink in between, by a process running alongside, is out of its reach.
    """
    if directory.is_absolute():
        return
    prefix = Path()
    for part in directory.parts:
        prefix /= part
        try:
            mode = os.lstat(prefix).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise SymlinkedOutputError(
                f"refusing to write through the symlink {str(prefix)!r}: a relative output "
                "path never follows a symlink (give an absolute path to write there)"
            )


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
