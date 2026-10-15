"""Write an output file atomically, never through a symlink planted at its name (contract §14).

Output directories such as ``out/run`` can come from a checkout of someone else's branch, so a
symlink there (``traj_000.npz -> ~/.zshrc``) must never redirect a write to its target, and
neither must a symlink committed in place of a directory (``out -> ~/project``), which
``check_output_directory`` refuses for the relative paths the CLI writes to.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

OUTPUT_MODE = 0o644
"""Mode of every output file, whatever the umask: trajectories and videos are artifacts that CI
uploads and other people read."""
DIRECTORY_MODE = 0o755
"""Mode of an output directory ``new_directory`` creates (a policy directory is committed and
shared like any other output)."""
_NEW_FILE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_BINARY", 0)
)


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


@contextmanager
def new_directory(path: Path) -> Iterator[Path]:
    """A new, private, empty directory next to ``path`` for the block to fill, renamed to
    ``path`` (mode ``DIRECTORY_MODE``) when the block succeeds, and removed with everything in
    it when the block fails: ``path`` appears complete or not at all.

    ``FileExistsError`` if anything (a symlink included) is at ``path`` before the block, or
    after it: an existing directory is never written into or replaced. A directory created at
    ``path`` in the instant between that last check and the rename is replaced if it is empty
    (POSIX ``rename``) and refused if it is not, so nothing anyone wrote there is lost.
    """
    if os.path.lexists(path):
        raise FileExistsError(errno.EEXIST, "already exists", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial"))
    try:
        yield tmp
        os.chmod(tmp, DIRECTORY_MODE)
        if os.path.lexists(path):
            raise FileExistsError(errno.EEXIST, "already exists", str(path))
        os.rename(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def write_new_file(path: Path, data: bytes) -> None:
    """Create ``path`` (which must not exist, and is never followed if it is a symlink) holding
    ``data``, mode ``OUTPUT_MODE``. For files inside a ``new_directory``."""
    fd = os.open(path, _NEW_FILE_FLAGS, OUTPUT_MODE)
    try:
        _set_mode(fd, path)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
    finally:
        os.close(fd)
