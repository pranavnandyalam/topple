"""Atomic output files that never write through a planted symlink (contract §14)."""

import os
import stat
from pathlib import Path

import pytest

from topple._atomic import OUTPUT_MODE, replacement


def test_the_block_writes_the_new_content_into_place(tmp_path: Path) -> None:
    path = tmp_path / "out.bin"
    path.write_bytes(b"old")
    with replacement(path, suffix=".partial.bin") as tmp:
        assert tmp.parent == tmp_path
        assert tmp.name.startswith(".out.bin.")
        assert tmp.name.endswith(".partial.bin")
        assert tmp.read_bytes() == b""
        tmp.write_bytes(b"new")
    assert path.read_bytes() == b"new"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]


def test_the_mode_ignores_the_umask(tmp_path: Path) -> None:
    previous = os.umask(0o077)
    try:
        with replacement(tmp_path / "out.bin") as tmp:
            tmp.write_bytes(b"x")
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "out.bin").stat().st_mode) == OUTPUT_MODE == 0o644


def test_the_mode_is_set_through_the_path_without_fchmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows before Python 3.13 has no ``os.fchmod``."""
    monkeypatch.delattr(os, "fchmod")
    with replacement(tmp_path / "out.bin") as tmp:
        tmp.write_bytes(b"x")
    assert stat.S_IMODE((tmp_path / "out.bin").stat().st_mode) == 0o644


def test_a_failing_block_leaves_the_target_and_no_temporary(tmp_path: Path) -> None:
    path = tmp_path / "out.bin"
    path.write_bytes(b"old")

    def interrupted() -> None:
        with replacement(path) as tmp:
            tmp.write_bytes(b"half")
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted()
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]
