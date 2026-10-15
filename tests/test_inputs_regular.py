"""``read_regular_file``: the strict reader for files inside submission and run directories
(docs/design/phase-2.md §3.4).

Such a directory may come from someone else, who can plant a symlink (to ``/dev/zero`` or to a
file the reader may see but the author may not), a hard link, a FIFO (an open that waits forever
for a writer) or a file far larger than any honest one, and can swap the path between the check
and the open. Every one of these is a ``LoadError`` naming the file, and nothing blocks.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from topple._inputs import read_regular_file
from topple.loader import LoadError

POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="needs POSIX file types")
TIMEOUT_S = 5.0


def read_in_thread(path: Path, cap: int) -> BaseException | bytes:
    """The result of ``read_regular_file``, run in a daemon thread bounded by ``TIMEOUT_S`` so a
    reader that wrongly blocks fails the test instead of hanging the suite."""
    outcome: list[BaseException | bytes] = []

    def target() -> None:
        try:
            outcome.append(read_regular_file(path, cap))
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(TIMEOUT_S)
    assert not thread.is_alive(), "read_regular_file blocked"
    return outcome[0]


def fake_lstat(monkeypatch: pytest.MonkeyPatch, path: Path, seen_as: Path) -> None:
    """Make ``os.lstat(path)`` report ``seen_as``: the path as it was before a swap."""
    real = os.lstat

    def lstat(p: os.PathLike[str] | str, *args: object, **kwargs: object) -> os.stat_result:
        if os.fspath(p) == os.fspath(path):
            return real(seen_as)
        return real(p, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "lstat", lstat)


def test_reads_the_exact_bytes(tmp_path: Path) -> None:
    p = tmp_path / "policy.json"
    p.write_bytes(b'{"a": 1}\n')
    assert read_regular_file(p, 64) == b'{"a": 1}\n'


def test_reads_an_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "README.md"
    p.write_bytes(b"")
    assert read_regular_file(p, 0) == b""


def test_reads_a_file_of_exactly_cap_bytes_across_many_chunks(tmp_path: Path) -> None:
    data = bytes(range(256)) * (3 * 4096 + 7)  # about 3 MiB: more than one read call
    p = tmp_path / "actor.npz"
    p.write_bytes(data)
    assert read_regular_file(p, len(data)) == data


def test_oversize_is_rejected_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "policy.onnx"
    p.write_bytes(b"x" * 11)

    def no_open(*args: object, **kwargs: object) -> int:
        raise AssertionError("opened an oversize file")

    monkeypatch.setattr(os, "open", no_open)
    with pytest.raises(LoadError, match=r"'policy\.onnx' is larger than 10 bytes"):
        read_regular_file(p, 10)


def test_growth_after_the_check_is_cut_at_cap_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The size check uses the lstat; a file that grows after it is still read only to cap+1."""
    p = tmp_path / "policy.onnx"
    p.write_bytes(b"x" * 10)
    real = os.lstat

    def lstat_then_grow(path: os.PathLike[str] | str, *args: object) -> os.stat_result:
        info = real(path)
        if os.fspath(path) == os.fspath(p):
            with p.open("ab") as fh:
                fh.write(b"y" * 1_000_000)
        return info

    monkeypatch.setattr(os, "lstat", lstat_then_grow)
    reads: list[int] = []
    real_read = os.read

    def counting_read(fd: int, n: int) -> bytes:
        reads.append(n)
        return real_read(fd, n)

    monkeypatch.setattr(os, "read", counting_read)
    with pytest.raises(LoadError, match="larger than 10 bytes"):
        read_regular_file(p, 10)
    assert sum(reads) <= 11


@POSIX_ONLY
def test_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "real.onnx"
    target.write_bytes(b"data")
    link = tmp_path / "policy.onnx"
    link.symlink_to(target)
    with pytest.raises(LoadError, match=r"'policy\.onnx' is a symlink"):
        read_regular_file(link, 1024)


@POSIX_ONLY
def test_symlink_to_dev_zero_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "policy.onnx"
    link.symlink_to("/dev/zero")
    assert isinstance(read_in_thread(link, 1024), LoadError)


@POSIX_ONLY
def test_hard_link_is_rejected(tmp_path: Path) -> None:
    a = tmp_path / "policy.json"
    a.write_bytes(b"{}")
    b = tmp_path / "elsewhere.json"
    os.link(a, b)
    for path in (a, b):
        with pytest.raises(LoadError, match="hard link"):
            read_regular_file(path, 64)


def test_directory_is_rejected(tmp_path: Path) -> None:
    d = tmp_path / "policy.onnx"
    d.mkdir()
    with pytest.raises(LoadError, match="not a regular file"):
        read_regular_file(d, 64)


@POSIX_ONLY
def test_fifo_is_rejected_without_blocking(
    tmp_path: Path, make_fifo: Callable[[Path], Path]
) -> None:
    fifo = make_fifo(tmp_path / "policy.onnx")
    outcome = read_in_thread(fifo, 64)
    assert isinstance(outcome, LoadError)
    assert "not a regular file" in str(outcome)


@POSIX_ONLY
def test_character_device_is_rejected() -> None:
    assert isinstance(read_in_thread(Path("/dev/null"), 64), LoadError)


def test_missing_file_is_a_load_error_naming_it(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match=r"cannot read 'missing\.json'") as info:
        read_regular_file(tmp_path / "missing.json", 64)
    assert isinstance(info.value.__cause__, FileNotFoundError)


def test_errors_keep_load_error_identity(tmp_path: Path) -> None:
    with pytest.raises(LoadError) as info:
        read_regular_file(tmp_path / "missing.json", 64)
    assert type(info.value).__module__ == "topple.loader"
    assert type(info.value).__qualname__ == "LoadError"


def test_names_are_quoted_with_repr(tmp_path: Path) -> None:
    name = "evil\x1b[31m.json"
    with pytest.raises(LoadError) as info:
        read_regular_file(tmp_path / name, 64)
    assert "\x1b" not in str(info.value)
    assert repr(name) in str(info.value)


def test_open_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "policy.json"
    p.write_bytes(b"{}")
    seen: list[int] = []
    real_open = os.open

    def recording_open(path: os.PathLike[str] | str, flags: int, *args: object) -> int:
        seen.append(flags)
        return real_open(path, flags, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", recording_open)
    read_regular_file(p, 64)
    (flags,) = seen
    assert flags & (os.O_WRONLY | os.O_RDWR) == 0
    for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_NOCTTY", "O_CLOEXEC", "O_BINARY"):
        flag = getattr(os, name, 0)
        assert flags & flag == flag, name


@POSIX_ONLY
def test_swap_to_a_fifo_after_the_check_never_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare FIFO with no writer: only ``O_NONBLOCK`` keeps the open from waiting forever.
    (``make_fifo``'s writer thread would release a blocking open and hide the bug.)"""
    before = tmp_path / "before.onnx"
    before.write_bytes(b"data")
    path = tmp_path / "policy.onnx"
    os.mkfifo(path)
    fake_lstat(monkeypatch, path, before)  # the check saw a regular file
    try:
        outcome = read_in_thread(path, 64)
    finally:
        _release_a_blocked_reader(path)
    assert isinstance(outcome, LoadError)
    assert "changed" in str(outcome)


def _release_a_blocked_reader(fifo: Path) -> None:
    """Open ``fifo`` for writing once without blocking: a reader wrongly blocked in ``open``
    returns, so its daemon thread ends (ENXIO, no reader waiting, is the passing case)."""
    try:
        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return
    os.close(fd)


def test_swap_to_another_regular_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = tmp_path / "before.onnx"
    before.write_bytes(b"honest")
    path = tmp_path / "policy.onnx"
    path.write_bytes(b"swapped")
    fake_lstat(monkeypatch, path, before)
    with pytest.raises(LoadError, match="changed"):
        read_regular_file(path, 64)


@POSIX_ONLY
@pytest.mark.parametrize("nofollow", [True, False], ids=["O_NOFOLLOW", "no_O_NOFOLLOW"])
def test_swap_to_a_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nofollow: bool
) -> None:
    """Without ``O_NOFOLLOW`` (win32) the fstat identity check still refuses the swap."""
    before = tmp_path / "before.onnx"
    before.write_bytes(b"honest")
    secret = tmp_path / "secret"
    secret.write_bytes(b"secret")
    path = tmp_path / "policy.onnx"
    path.symlink_to(secret)
    fake_lstat(monkeypatch, path, before)
    if not nofollow:
        monkeypatch.delattr(os, "O_NOFOLLOW")
    with pytest.raises(LoadError):
        read_regular_file(path, 64)


@POSIX_ONLY
def test_hard_link_made_after_the_check_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "policy.onnx"
    path.write_bytes(b"data")
    real = os.lstat

    def lstat_then_link(p: os.PathLike[str] | str, *args: object) -> os.stat_result:
        info = real(p)
        if os.fspath(p) == os.fspath(path):
            os.link(path, tmp_path / "second")
        return info

    monkeypatch.setattr(os, "lstat", lstat_then_link)
    with pytest.raises(LoadError, match="changed"):
        read_regular_file(path, 64)
