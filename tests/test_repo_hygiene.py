"""What the repository holds and ignores (docs/design/phase-2.md §6 ``test_repo_hygiene``, §7
"Repository").

``submissions/`` is what people copy and run: no pickle-family files, no external ONNX tensor
data and no hidden files there, tracked or about to be. ``.gitignore`` keeps training runs,
pickles, the torch-free eval environment and macOS droppings out, but never the results
package's videos (``*.mp4`` is ignored everywhere else).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# §3.1's refused entries that a submission directory never holds: the pickle family (SB3's
# model.zip is a zip of pickles) and native code
PICKLE_FAMILY = (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".npz.pkl", ".dill", ".zip")
NATIVE_CODE = (".so", ".dylib", ".pyd", ".dll")
EXTERNAL_DATA = (".onnx.data", ".onnx_data", ".data")

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60, check=False
    )


def submission_files() -> list[str]:
    """Tracked files under submissions/, and untracked ones that are not ignored."""
    tracked = git("ls-files", "-z", "--", "submissions")
    untracked = git("ls-files", "-z", "--others", "--exclude-standard", "--", "submissions")
    if tracked.returncode or untracked.returncode:
        pytest.skip("not a git checkout")
    return sorted(set(filter(None, (tracked.stdout + untracked.stdout).split("\0"))))


def refused(rel: str) -> bool:
    """Whether ``rel`` (a path under ``submissions/``) is a file the repository never holds
    there: pickle family, native code, external tensor data or anything hidden."""
    name = rel.lower()
    return name.endswith(PICKLE_FAMILY + NATIVE_CODE + EXTERNAL_DATA) or any(
        part.startswith(".") for part in Path(rel).parts
    )


def test_submissions_hold_no_pickles_external_data_or_hidden_files() -> None:
    files = submission_files()
    assert files, "submissions/ is empty?"
    assert [rel for rel in files if refused(rel)] == []


@pytest.mark.parametrize(
    "rel",
    [
        "submissions/ppo_balance/model.zip",  # spec-SPEC-6
        "submissions/ppo_balance/vecnormalize.pkl",
        "submissions/ppo_balance/policy.onnx.data",
        "submissions/bot/_ext.cpython-312-x86_64-linux-gnu.so",
        "submissions/ppo_balance/.DS_Store",
        "submissions/.hidden/policy.onnx",
    ],
)
def test_the_refused_files(rel: str) -> None:
    assert refused(rel)


@pytest.mark.parametrize(
    "rel",
    [
        "submissions/pid_example.py",
        "submissions/README.md",
        "submissions/ppo_balance/policy.onnx",
        "submissions/ppo_balance/policy.json",
    ],
)
def test_the_kept_files(rel: str) -> None:
    assert not refused(rel)


@pytest.mark.parametrize(
    "path",
    [
        "runs/ppo_s0/ckpt/model.zip",
        "vecnormalize.pkl",
        "deep/dir/x.pkl",
        ".venv-eval/bin/python",
        ".DS_Store",
        "submissions/.DS_Store",
        "out/run/traj_000.npz",
        "out/hero.mp4",
    ],
)
def test_gitignore_covers(path: str) -> None:
    done = git("check-ignore", "-q", "--no-index", path)
    assert done.returncode == 0, f"{path} is not ignored"


@pytest.mark.parametrize(
    "path",
    [
        "docs/results/phase2/hero.mp4",
        "docs/results/phase2/waypoint.mp4",
        "submissions/ppo_balance/policy.onnx",
        "submissions/ppo_balance/policy.json",
    ],
)
def test_gitignore_keeps(path: str) -> None:
    done = git("check-ignore", "-q", "--no-index", path)
    assert done.returncode == 1, f"{path} is ignored ({done.stderr.strip()})"
