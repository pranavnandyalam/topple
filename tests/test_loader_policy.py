"""Loading ``topple.policy/1`` directories (docs/design/phase-2.md §3.1, §3.6, §6
``test_loader_policy``; contract §8).

A policy submission is data only: exactly ``policy.onnx``, ``policy.json`` and optionally
``README.md``, regular files with one link each, named exactly as ``os.scandir`` returns them.
Anything else is a ``LoadError`` that names the entry (``repr(safe(name))``), the listing
stops at the fourth entry, and each file is read once: the submission sha256 covers the very
bytes that are validated and run. Nothing here ever blocks on a FIFO.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import topple._inputs as inputs_module
import topple.policy.runtime as runtime
from policy_support import (
    first_party_manifest,
    manifest_bytes,
    ppo_bytes,
    sac_bytes,
    sha_of_files,
    write_policy_dir,
)
from topple.challenge import load_challenge
from topple.contract import Controller
from topple.envs.balance import BalanceEnv
from topple.loader import LoadError, load_submission, submission_sha256
from topple.policy.runtime import load_policy_dir, read_policy_dir
from topple.policy.validate import safe

PPO = ppo_bytes()
BALANCE = load_challenge("balance-v1")
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="needs POSIX file types")
LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux", reason="needs a file system that stores any bytes as a name"
)
TIMEOUT_S = 5.0

CONTROLLER = """
class C:
    def reset(self, rng_seed): pass
    def act(self, obs): return (0.0, 0.0)

def make_controller(spec):
    return C()
"""


def load(root: Path) -> Any:
    return load_submission(root, challenge=BALANCE)


def refused(root: Path) -> str:
    with pytest.raises(LoadError) as info:
        load(root)
    return str(info.value)


def in_thread(fn: Callable[[], object]) -> object:
    """``fn()``'s result or exception, run in a daemon thread bounded by ``TIMEOUT_S`` so a
    loader that wrongly blocks fails the test instead of hanging the suite."""
    outcome: list[object] = []

    def target() -> None:
        try:
            outcome.append(fn())
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(TIMEOUT_S)
    assert not thread.is_alive(), "the load blocked"
    return outcome[0]


# --- detection --------------------------------------------------------------------------------


def test_a_policy_directory_is_detected(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "tiny_ppo")
    sub = load(root)
    assert sub.kind == "onnx"
    assert sub.id == "tiny_ppo"
    assert sub.entry == root.resolve() / "policy.onnx"
    assert sub.policy is not None
    assert sub.policy_sha256 == sub.policy.validated.policy_sha256
    controller = sub.factory(BalanceEnv("ideal").task_spec)
    assert isinstance(controller, Controller)


def test_python_submissions_have_no_policy(tmp_path: Path) -> None:
    (tmp_path / "bot.py").write_text(CONTROLLER)
    sub = load_submission(tmp_path / "bot.py")
    assert sub.kind == "python"
    assert sub.policy is None
    assert sub.policy_sha256 is None


def test_readme_is_optional_and_hashed(tmp_path: Path) -> None:
    plain = load(write_policy_dir(tmp_path / "a"))
    documented = load(write_policy_dir(tmp_path / "b", readme=b"# tiny ppo\n"))
    assert plain.sha256 != documented.sha256
    assert plain.policy_sha256 == documented.policy_sha256


def test_missing_challenge_is_a_type_error(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="challenge"):
        load_submission(write_policy_dir(tmp_path / "bot"))


@pytest.mark.parametrize("missing", ["policy.onnx", "policy.json"])
def test_both_files_are_required(tmp_path: Path, missing: str) -> None:
    root = write_policy_dir(tmp_path / "bot")
    (root / missing).unlink()
    if missing == "policy.json":  # without its manifest the directory is a Python one
        assert "controller.py" in refused(root)
    else:
        assert refused(root) == "missing 'policy.onnx'"


# --- ambiguity and unexpected entries ---------------------------------------------------------


@pytest.mark.parametrize("name", ["controller.py", "helper.py", "Helper.PY"])
def test_policy_json_plus_python_is_ambiguous(tmp_path: Path, name: str) -> None:
    root = write_policy_dir(tmp_path / "bot")
    (root / name).write_text(CONTROLLER)
    assert refused(root) == "ambiguous submission kind"


def test_a_python_directory_with_a_policy_json_asset_is_now_ambiguous(tmp_path: Path) -> None:
    """The one Python case §3.1 changes: before, policy.json was just an asset."""
    root = tmp_path / "bot"
    root.mkdir()
    (root / "controller.py").write_text(CONTROLLER)
    (root / "policy.json").write_text('{"gain": 1}')
    for k in range(10):
        (root / f"asset_{k}.txt").write_text("x")
    assert refused(root) == "ambiguous submission kind"
    with pytest.raises(LoadError, match="ambiguous submission kind"):
        load_submission(root)  # refused before the missing challenge matters


@pytest.mark.parametrize(
    "name",
    [
        "notes.txt",
        ".DS_Store",
        ".hidden",
        "model.pkl",
        "model.pickle",
        "a.zip",
        "w.pt",
        "w.pth",
        "m.joblib",
        "c.ckpt",
        "policy.onnx.data",
        "lib.so",
        "x.pyc",
        "readme.md",
        "policy.onnx~",
    ],
)
def test_extra_entries_are_named(tmp_path: Path, name: str) -> None:
    root = write_policy_dir(tmp_path / "bot")
    (root / name).write_bytes(b"x")
    assert refused(root) == f"unexpected entry: {safe(name)!r}"


def test_subdirectories_are_unexpected(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot")
    (root / "sub").mkdir()
    assert refused(root) == "unexpected entry: 'sub'"


def test_entry_names_are_shown_through_safe(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot")
    (root / "<b>\x1b[2J.txt").write_bytes(b"x")
    assert refused(root) == "unexpected entry: '?b???2J.txt'"


@POSIX_ONLY
def test_a_fifo_entry_is_unexpected_and_never_opened(
    tmp_path: Path, make_fifo: Callable[[Path], Path]
) -> None:
    root = write_policy_dir(tmp_path / "bot")
    make_fifo(root / "pipe")
    assert in_thread(lambda: refused(root)) == "unexpected entry: 'pipe'"


@POSIX_ONLY
def test_a_socket_entry_is_unexpected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = write_policy_dir(tmp_path / "bot")
    monkeypatch.chdir(root)  # a short relative name: socket paths are limited to ~100 bytes
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind("s")
        assert refused(root) == "unexpected entry: 's'"


@LINUX_ONLY
def test_a_policy_named_policy_json_in_capitals_is_refused(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot")
    (root / "POLICY.JSON").write_bytes(manifest_bytes(first_party_manifest(PPO)))
    assert refused(root) == "unexpected entry: 'POLICY.JSON'"


def test_capitalized_names_alone_are_refused_on_every_os(tmp_path: Path) -> None:
    """``POLICY.JSON`` is not ``policy.json``: on a case-sensitive system the directory is no
    policy (and no Python submission); on a case-insensitive one the exact-name listing refuses
    the entry. Either way it does not load."""
    root = tmp_path / "bot"
    root.mkdir()
    (root / "POLICY.JSON").write_bytes(manifest_bytes(first_party_manifest(PPO)))
    (root / "policy.onnx").write_bytes(PPO)
    with pytest.raises(LoadError):
        load(root)


@LINUX_ONLY
def test_a_non_utf8_name_is_refused(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot")
    with open(os.path.join(os.fsencode(root), b"\xffname"), "wb") as fh:
        fh.write(b"x")
    assert refused(root) == "file name '?name' is not valid UTF-8"


def test_a_non_utf8_name_is_refused_on_every_os(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS file systems cannot store such a name, so the listing is faked: the rule is the
    loader's, whatever the platform (a name ``os.fsdecode`` turned into lone surrogates)."""
    root = write_policy_dir(tmp_path / "bot")
    real = os.scandir

    class WithBadName:
        def __init__(self, path: Any) -> None:
            self._it = real(path)

        def __enter__(self) -> WithBadName:
            return self

        def __exit__(self, *exc: object) -> None:
            self._it.close()

        def __iter__(self) -> Iterator[Any]:
            yield from self._it
            yield type("Entry", (), {"name": "bad\udcffname"})()

    monkeypatch.setattr(runtime.os, "scandir", WithBadName)
    assert refused(root) == "file name 'bad?name' is not valid UTF-8"


def test_a_directory_of_10000_entries_is_refused_after_at_most_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_policy_dir(tmp_path / "big")
    for k in range(10_000):
        (root / f"f{k:05d}").touch()
    seen: list[str] = []
    real = os.scandir

    class Counting:
        def __init__(self, path: Any) -> None:
            self._it = real(path)

        def __enter__(self) -> Counting:
            return self

        def __exit__(self, *exc: object) -> None:
            self._it.close()

        def __iter__(self) -> Iterator[os.DirEntry[str]]:
            for entry in self._it:
                seen.append(entry.name)
                yield entry

    monkeypatch.setattr(runtime.os, "scandir", Counting)
    assert refused(root).startswith("unexpected entry: 'f")
    assert 1 <= len(seen) <= 4


# --- links, sizes, swaps ----------------------------------------------------------------------


@POSIX_ONLY
def test_a_hard_linked_policy_json_is_refused(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot")
    os.link(root / "policy.json", tmp_path / "elsewhere.json")
    assert refused(root) == "'policy.json' has 2 hard links"


@POSIX_ONLY
def test_a_symlinked_policy_onnx_is_refused(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot")
    (tmp_path / "real.onnx").write_bytes(PPO)
    (root / "policy.onnx").unlink()
    (root / "policy.onnx").symlink_to(tmp_path / "real.onnx")
    assert refused(root) == "'policy.onnx' is a symlink"


@POSIX_ONLY
def test_a_symlink_to_dev_zero_is_refused_without_reading(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot")
    (root / "policy.onnx").unlink()
    (root / "policy.onnx").symlink_to("/dev/zero")
    assert in_thread(lambda: refused(root)) == "'policy.onnx' is a symlink"


@POSIX_ONLY
def test_a_symlinked_directory_is_refused(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot")
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    assert "is a symlink" in refused(link)
    with pytest.raises(LoadError, match=r"'link' is a symlink"):
        load_policy_dir(link, challenge=BALANCE)


def test_load_policy_dir_refuses_a_file(tmp_path: Path) -> None:
    (tmp_path / "policy.onnx").write_bytes(PPO)
    with pytest.raises(LoadError, match=r"'policy\.onnx' is not a directory"):
        load_policy_dir(tmp_path / "policy.onnx", challenge=BALANCE)


def test_a_path_that_cannot_be_examined_is_a_load_error(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="cannot read"):
        load_policy_dir(tmp_path / ("x" * 300), challenge=BALANCE)
    with pytest.raises(LoadError, match="cannot read"):
        load_policy_dir(tmp_path / "absent", challenge=BALANCE)


def test_a_listing_that_fails_is_a_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_policy_dir(tmp_path / "bot")

    def denied(path: Any) -> Any:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(runtime.os, "scandir", denied)
    assert refused(root) == "cannot read 'bot': Permission denied"


def test_oversize_policy_onnx_is_refused_before_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_policy_dir(tmp_path / "bot")
    with (root / "policy.onnx").open("r+b") as fh:
        fh.truncate(2 * 1024 * 1024)  # sparse: nothing is written

    real = inputs_module._read_at_most

    def guarded(fd: int, limit: int) -> bytes:
        if limit > 64 * 1024 + 1:  # only policy.onnx has a cap above README's
            raise AssertionError("an oversize file was read")
        return real(fd, limit)

    monkeypatch.setattr(inputs_module, "_read_at_most", guarded)
    assert refused(root) == "'policy.onnx' is larger than 1048576 bytes"


def test_oversize_readme_is_refused(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot", readme=b"#" * (64 * 1024 + 1))
    assert refused(root) == "'README.md' is larger than 65536 bytes"


def _swap_after_lstat(
    monkeypatch: pytest.MonkeyPatch, target: Path, swap: Callable[[], None]
) -> None:
    """Run ``swap`` right after ``read_regular_file``'s lstat of ``target``: the path changes
    between the check and the open."""
    real = os.lstat

    def lstat(p: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        result = real(p, *args, **kwargs)
        if os.fspath(p) == os.fspath(target):
            swap()
        return result

    monkeypatch.setattr(inputs_module.os, "lstat", lstat)


@POSIX_ONLY
def test_an_lstat_open_swap_to_another_regular_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_policy_dir(tmp_path / "bot")
    (tmp_path / "other.onnx").write_bytes(sac_bytes())
    target = root / "policy.onnx"
    _swap_after_lstat(monkeypatch, target, lambda: os.replace(tmp_path / "other.onnx", target))
    assert in_thread(lambda: refused(root)) == "'policy.onnx' changed while it was opened"


@POSIX_ONLY
def test_an_lstat_open_swap_to_a_fifo_is_refused_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_policy_dir(tmp_path / "bot")
    target = root / "policy.onnx"

    def to_fifo() -> None:
        target.unlink()
        os.mkfifo(target)

    _swap_after_lstat(monkeypatch, target, to_fifo)
    assert in_thread(lambda: refused(root)) == "'policy.onnx' changed while it was opened"


def test_a_directory_swapped_while_it_is_read_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_policy_dir(tmp_path / "bot")
    write_policy_dir(tmp_path / "other")
    real = runtime.read_regular_file
    swapped: list[bool] = []

    def read_then_swap(path: Path, cap: int) -> bytes:
        data = real(path, cap)
        if not swapped:
            swapped.append(True)
            os.rename(root, tmp_path / "aside")
            os.rename(tmp_path / "other", root)
        return data

    monkeypatch.setattr(runtime, "read_regular_file", read_then_swap)
    assert refused(root) == "'bot' changed while it was read"


# --- read once --------------------------------------------------------------------------------


def test_sha256_is_the_submission_recipe_over_the_single_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_policy_dir(tmp_path / "bot", readme=b"# hi\n")
    on_disk = {p.name: p.read_bytes() for p in root.iterdir()}
    real = runtime.read_regular_file
    reads: list[str] = []

    def counting(path: Path, cap: int) -> bytes:
        reads.append(path.name)
        data = real(path, cap)
        path.write_bytes(b"changed after it was read")  # the next reader would see this
        return data

    monkeypatch.setattr(runtime, "read_regular_file", counting)
    sub = load(root)
    assert sorted(reads) == ["README.md", "policy.json", "policy.onnx"]  # once each
    assert sub.sha256 == sha_of_files(on_disk)  # the bytes read, not the disk now
    assert sub.policy is not None
    assert sub.policy.validated.canonical == on_disk["policy.onnx"]


def test_sha256_equals_submission_sha256_of_the_directory(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot", readme=b"# tiny\n")
    assert load(root).sha256 == submission_sha256(root)


def test_read_policy_dir_returns_each_file(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot", readme=b"# x\n")
    files = read_policy_dir(root)
    assert set(files) == {"policy.onnx", "policy.json", "README.md"}
    assert files["policy.onnx"] == PPO
