"""Atomic output files that never write through a planted symlink (contract §14)."""

import os
import re
import stat
from pathlib import Path

import pytest

from topple._atomic import (
    OUTPUT_MODE,
    SymlinkedOutputError,
    check_output_directory,
    replacement,
)


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


# ------------------------------------------------------------------ planted output directories


def _planted(root: Path, link: str) -> Path:
    """``root/link`` a symlink to a directory ``victim`` holding one file."""
    victim = root / "victim"
    victim.mkdir()
    (victim / "results.json").write_text("precious")
    (root / link).parent.mkdir(parents=True, exist_ok=True)
    (root / link).symlink_to(victim)
    return victim


@pytest.mark.parametrize(
    ("link", "directory"),
    [
        ("out", "out/run"),  # a committed `out` symlink under the default `out/run`
        ("out/run", "out/run"),  # the output directory itself
        ("out", "out"),
        ("out", "sub/../out/run"),  # reached through `..`
    ],
)
def test_a_relative_output_directory_never_passes_through_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link: str, directory: str
) -> None:
    (tmp_path / "sub").mkdir()
    _planted(tmp_path, link)
    monkeypatch.chdir(tmp_path)
    shown = "sub/../out" if directory.startswith("sub/") else link
    expected = f"^refusing to write through the symlink '{re.escape(shown)}'"
    with pytest.raises(SymlinkedOutputError, match=expected):
        check_output_directory(Path(directory))


def test_a_dangling_symlink_is_refused_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "out").symlink_to(tmp_path / "nowhere")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SymlinkedOutputError, match="'out'"):
        check_output_directory(Path("out/run"))


@pytest.mark.parametrize("directory", ["out/run", "out/new/deeper", ".", ""])
def test_real_or_missing_directories_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: str
) -> None:
    """Existing directories that are not links pass, and so does a path from the first missing
    component on: the writer creates those as real directories."""
    (tmp_path / "out" / "run").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    check_output_directory(Path(directory))


def test_an_absolute_output_directory_is_the_callers_choice(tmp_path: Path) -> None:
    """Symlinks on an absolute path (macOS's /tmp, a symlinked home directory) are followed."""
    victim = _planted(tmp_path, "out")
    check_output_directory(tmp_path / "out" / "run")
    assert sorted(p.name for p in victim.iterdir()) == ["results.json"]


def test_a_component_that_cannot_be_examined_raises_its_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "file").write_text("not a directory")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(NotADirectoryError):
        check_output_directory(Path("file/run"))


def test_the_refusal_is_an_oserror() -> None:
    """Writers that report an ``OSError`` as "cannot write" report this one too."""
    assert issubclass(SymlinkedOutputError, OSError)
