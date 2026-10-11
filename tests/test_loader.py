import hashlib
import os
from pathlib import Path

import pytest

import topple.loader as loader_module
from topple.contract import FEATURES, Controller, TaskSpec
from topple.envs.balance import BalanceEnv
from topple.loader import LoadError, load_submission, submission_sha256

SPEC: TaskSpec = BalanceEnv("ideal").task_spec

GOOD = """
class C:
    def reset(self, rng_seed):
        self.seed = rng_seed
    def act(self, obs):
        return (0.0, 0.0)

def make_controller(spec):
    assert spec.challenge == "balance-v1"
    return C()
"""

ZERO_ARG = """
class C:
    def reset(self, rng_seed): pass
    def act(self, obs): return [0, 0]

def make_controller():
    return C()
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_load_file(tmp_path: Path) -> None:
    sub = load_submission(write(tmp_path, "mine.py", GOOD))
    assert sub.id == "mine"
    assert len(sub.sha256) == 64
    ctrl = sub.factory(SPEC)
    assert isinstance(ctrl, Controller)


def test_zero_arg_factory(tmp_path: Path) -> None:
    sub = load_submission(write(tmp_path, "lesson1.py", ZERO_ARG))
    assert isinstance(sub.factory(SPEC), Controller)


def test_optional_spec_factory(tmp_path: Path) -> None:
    text = ZERO_ARG.replace(
        "def make_controller():", "def make_controller(spec=None):\n    assert spec is not None"
    )
    sub = load_submission(write(tmp_path, "opt.py", text))
    assert isinstance(sub.factory(SPEC), Controller)


def test_directory_submission_with_assets(tmp_path: Path) -> None:
    root = tmp_path / "team_bot"
    write(root, "gains.txt", "0.5")
    write(
        root,
        "controller.py",
        GOOD.replace(
            "def make_controller(spec):",
            "from pathlib import Path\n"
            "GAIN = float((Path(__file__).parent / 'gains.txt').read_text())\n"
            "def make_controller(spec):",
        ),
    )
    sub = load_submission(root)
    assert sub.id == "team_bot"
    assert sub.entry.name == "controller.py"
    assert isinstance(sub.factory(SPEC), Controller)


def test_sha_changes_with_any_file(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    write(root, "data.txt", "a")
    before = submission_sha256(root)
    write(root, "data.txt", "b")
    assert submission_sha256(root) != before


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("nofactory.py", "x = 1\n", "must define make_controller"),
        ("syntax.py", "def make_controller(:\n", "importing syntax.py failed"),
        ("raises.py", "raise RuntimeError('at import')\n", "RuntimeError: at import"),
        (
            "twoargs.py",
            "def make_controller(a, b):\n    pass\n",
            "must take (spec) or no arguments",
        ),
        ("notpy.txt", "hello", "expected a .py file"),
    ],
)
def test_load_errors(tmp_path: Path, name: str, text: str, message: str) -> None:
    with pytest.raises(LoadError, match=message.replace("(", r"\(").replace(")", r"\)")):
        load_submission(write(tmp_path, name, text))


def test_factory_returning_non_controller(tmp_path: Path) -> None:
    sub = load_submission(write(tmp_path, "bad.py", "def make_controller(spec):\n    return 42\n"))
    with pytest.raises(LoadError, match="lacks reset"):
        sub.factory(SPEC)


def test_missing_path(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="does not exist"):
        load_submission(tmp_path / "nope.py")


def test_directory_without_entry(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(LoadError, match=r"controller\.py"):
        load_submission(tmp_path / "empty")


def test_too_many_files(tmp_path: Path) -> None:
    root = tmp_path / "big"
    write(root, "controller.py", GOOD)
    for i in range(60):
        write(root, f"f{i}.txt", "x")
    with pytest.raises(LoadError, match="files"):
        load_submission(root)


def test_two_submissions_do_not_collide(tmp_path: Path) -> None:
    a = load_submission(write(tmp_path / "a", "bot.py", GOOD))
    b = load_submission(write(tmp_path / "b", "bot.py", GOOD.replace("0.0, 0.0", "0.1, 0.1")))
    assert a.sha256 != b.sha256
    assert a.factory(SPEC).act(None) != b.factory(SPEC).act(None)  # type: ignore[arg-type]


def test_spec_features_match_contract() -> None:
    assert SPEC.obs_features == FEATURES


# ------------------------------------------------------------------ hashing and directory hygiene


def reference_sha(root: Path, rels: list[str]) -> str:
    """Length-framed (POSIX relpath, size, bytes) records, written out independently."""
    digest = hashlib.sha256(b"topple.submission/1\n")
    for rel in sorted(rels):
        data = (root / rel).read_bytes()
        name = rel.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def test_sha_follows_the_framing(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    write(root, "assets/gains.txt", "0.5")
    assert submission_sha256(root) == reference_sha(root, ["assets/gains.txt", "controller.py"])
    single = write(tmp_path, "solo.py", GOOD)
    assert submission_sha256(single) == reference_sha(tmp_path, ["solo.py"])


def test_sha_frames_names_and_contents(tmp_path: Path) -> None:
    """Without framing, ("a", "bc") and ("ab", "c") would hash the same bytes."""
    one, two = tmp_path / "one", tmp_path / "two"
    write(one, "controller.py", GOOD)
    write(two, "controller.py", GOOD)
    write(one, "a", "bc")
    write(two, "ab", "c")
    assert submission_sha256(one) != submission_sha256(two)


@pytest.mark.parametrize("junk", ["__pycache__/controller.cpython-312.pyc", "old.pyc", "lib/x.pyc"])
def test_compiled_python_is_refused(tmp_path: Path, junk: str) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    write(root, junk, "")
    with pytest.raises(LoadError, match=r"__pycache__|\.pyc"):
        load_submission(root)


def test_empty_pycache_directory_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    (root / "__pycache__").mkdir()
    with pytest.raises(LoadError, match="__pycache__"):
        load_submission(root)


def test_loading_writes_no_bytecode_so_it_can_load_again(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    first = load_submission(root)
    assert sorted(p.name for p in root.iterdir()) == ["controller.py"]
    assert load_submission(root).sha256 == first.sha256


def test_symlinked_files_are_refused(tmp_path: Path) -> None:
    secret = write(tmp_path, "secret.txt", "token")
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    (root / "data.txt").symlink_to(secret)
    with pytest.raises(LoadError, match="symlink"):
        load_submission(root)


def test_symlinked_directories_are_refused(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    write(elsewhere, "big.bin", "x")
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    (root / "lib").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(LoadError, match="symlink"):
        load_submission(root)


def test_a_symlinked_submission_path_is_refused(tmp_path: Path) -> None:
    real = write(tmp_path, "real.py", GOOD)
    link = tmp_path / "link.py"
    link.symlink_to(real)
    with pytest.raises(LoadError, match="symlink"):
        load_submission(link)


def test_special_files_are_refused_without_blocking(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    os.mkfifo(root / "pipe")
    with pytest.raises(LoadError, match="regular file"):
        load_submission(root)


def test_size_cap_is_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    monkeypatch.setattr(loader_module, "MAX_BYTES", len(GOOD) + 10)
    write(root, "data.bin", "x" * 10)
    assert len(load_submission(root).sha256) == 64
    write(root, "data.bin", "x" * 11)
    with pytest.raises(LoadError, match="larger than"):
        load_submission(root)


def test_size_cap_stops_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap bounds what is read, not just what is counted afterwards."""
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    write(root, "huge.bin", "x" * 100_000)
    monkeypatch.setattr(loader_module, "MAX_BYTES", len(GOOD) + 100)
    sizes: list[int] = []
    real_open = os.fdopen

    class Spy:
        def __init__(self, fh: object) -> None:
            self.fh = fh

        def __enter__(self) -> "Spy":
            return self

        def __exit__(self, *exc: object) -> None:
            self.fh.close()  # type: ignore[attr-defined]

        def read(self, n: int = -1) -> bytes:
            data: bytes = self.fh.read(n)  # type: ignore[attr-defined]
            sizes.append(len(data))
            return data

    monkeypatch.setattr(loader_module.os, "fdopen", lambda fd, mode: Spy(real_open(fd, mode)))
    with pytest.raises(LoadError, match="larger than"):
        submission_sha256(root)
    assert sizes == [len(GOOD), 101]  # the 100 kB file: only one byte past what was left


def test_hashing_a_symlinked_file_is_refused(tmp_path: Path) -> None:
    real = write(tmp_path, "real.py", GOOD)
    link = tmp_path / "link.py"
    link.symlink_to(real)
    with pytest.raises(LoadError, match="symlink"):
        submission_sha256(link)
