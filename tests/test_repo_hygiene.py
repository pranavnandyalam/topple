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
PICKLE_FAMILY = (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".npz.pkl", ".dill")
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


def test_submissions_hold_no_pickles_external_data_or_hidden_files() -> None:
    files = submission_files()
    assert files, "submissions/ is empty?"
    for rel in files:
        name = rel.lower()
        assert not name.endswith(PICKLE_FAMILY), rel
        assert not name.endswith(EXTERNAL_DATA), rel
        assert not any(part.startswith(".") for part in Path(rel).parts), rel


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
