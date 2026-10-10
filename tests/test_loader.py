from pathlib import Path

import pytest

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
