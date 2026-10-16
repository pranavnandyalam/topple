"""Atomic output files that never write through a planted symlink (contract §14)."""

import os
import re
import stat
from pathlib import Path

import pytest

from topple._atomic import (
    DIRECTORY_MODE,
    OUTPUT_MODE,
    SymlinkedOutputError,
    check_output_directory,
    new_directory,
    replacement,
    write_new_file,
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


# --- new_directory (``topple policy pack``) ----------------------------------------------------


def test_a_new_directory_appears_complete(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "bot"
    with new_directory(out) as tmp:
        assert tmp.parent == out.parent
        assert tmp.name.startswith(".bot.")
        assert not out.exists()  # nothing at the target until the block succeeds
        write_new_file(tmp / "a.bin", b"a" * 70_000)
        write_new_file(tmp / "b.bin", b"")
    assert sorted(p.name for p in out.iterdir()) == ["a.bin", "b.bin"]
    assert (out / "a.bin").read_bytes() == b"a" * 70_000
    assert stat.S_IMODE(out.stat().st_mode) == DIRECTORY_MODE
    assert stat.S_IMODE((out / "b.bin").stat().st_mode) == OUTPUT_MODE
    assert [p.name for p in out.parent.iterdir()] == ["bot"]


def _fill_then_fail(out: Path) -> None:
    with new_directory(out) as tmp:
        write_new_file(tmp / "a.bin", b"a")
        raise RuntimeError("boom")


def test_a_failed_block_leaves_nothing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _fill_then_fail(tmp_path / "bot")
    assert list(tmp_path.iterdir()) == []


def test_a_failed_block_removes_the_parents_it_created(tmp_path: Path) -> None:
    (tmp_path / "kept").mkdir()
    with pytest.raises(RuntimeError, match="boom"):
        _fill_then_fail(tmp_path / "kept" / "new" / "deeper" / "bot")
    assert [p.name for p in tmp_path.iterdir()] == ["kept"]  # only what was there before
    assert list((tmp_path / "kept").iterdir()) == []


def _theirs_then_fail(out: Path, theirs: Path) -> None:
    with new_directory(out):
        theirs.write_text("keep")  # someone else writes into a parent created here
        raise RuntimeError("boom")


def test_a_created_parent_someone_wrote_into_is_kept(tmp_path: Path) -> None:
    out = tmp_path / "new" / "deeper" / "bot"
    with pytest.raises(RuntimeError, match="boom"):
        _theirs_then_fail(out, tmp_path / "new" / "theirs")
    assert sorted(p.name for p in (tmp_path / "new").iterdir()) == ["theirs"]


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_never_into_or_over_anything_already_there(tmp_path: Path, kind: str) -> None:
    out = tmp_path / "bot"
    if kind == "directory":
        out.mkdir()
    elif kind == "file":
        out.write_text("x")
    else:
        os.symlink(tmp_path / "elsewhere", out)
    with pytest.raises(FileExistsError), new_directory(out):
        pass
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bot"]


def _race(out: Path) -> None:
    with new_directory(out) as tmp:
        write_new_file(tmp / "a.bin", b"new")
        out.mkdir()  # someone else creates the target meanwhile
        (out / "theirs").write_text("keep")


def test_a_directory_that_appears_during_the_block_is_kept(tmp_path: Path) -> None:
    out = tmp_path / "bot"
    with pytest.raises(FileExistsError):
        _race(out)
    assert (out / "theirs").read_text() == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bot"]


def test_write_new_file_never_follows_or_replaces(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("keep")
    os.symlink(target, tmp_path / "link")
    with pytest.raises(FileExistsError):
        write_new_file(tmp_path / "link", b"x")
    with pytest.raises(FileExistsError):
        write_new_file(target, b"x")
    assert target.read_text() == "keep"
