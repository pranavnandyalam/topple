"""The shipped policy (docs/design/phase-2.md §6 ``test_shipped_policy``, §4 "Ubuntu evidence").

``submissions/ppo_balance`` loads without torch, its parity fixture is bound to its
``onnx.sha256`` and holds (batch 1, ≤ 1e-5, with ≥ 300 unsaturated and ≥ 32 saturated rows),
and on eval_v1 it scores above PID. The eval_v1 ``results.json`` is written to
``$TOPPLE_ARTIFACT_DIR`` when set: CI's ``python-slow`` job uploads it as the ubuntu evidence
(G2-G4's ubuntu half, G8). Before a policy is shipped, the tests skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from topple.challenge import load_challenge
from topple.cli import app
from topple.export import (
    MIN_SATURATED,
    MIN_UNSATURATED,
    PARITY_TOLERANCE,
    batch1_max_abs_diff,
    coverage_counts,
    read_parity_fixture,
)
from topple.policy.manifest import policy_task
from topple.policy.runtime import load_policy_dir
from topple.results import read_results

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[1]
SHIPPED = {"ppo_balance": "balance-v1"}  # 2b adds ppo_waypoint
runner = CliRunner()


def _shipped(name: str) -> Path:
    sub = REPO / "submissions" / name
    if not (sub / "policy.onnx").is_file():
        pytest.skip(f"submissions/{name} is not shipped yet")
    return sub


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_the_shipped_policy_loads_and_matches_its_fixture(name: str) -> None:
    sub = _shipped(name)
    challenge = load_challenge(SHIPPED[name])
    policy = load_policy_dir(sub, challenge=challenge)
    task = policy_task(challenge)
    data = (REPO / "tests" / "data" / f"{name}_parity.npz").read_bytes()
    fx = read_parity_fixture(data, obs_dim=task.obs_dim, action_dim=task.action_dim)
    assert fx.onnx_sha256 == policy.policy_sha256
    assert batch1_max_abs_diff(policy.validated, fx.obs, fx.action_ref) <= PARITY_TOLERANCE
    unsaturated, saturated = coverage_counts(fx.action_ref)
    assert unsaturated >= MIN_UNSATURATED
    assert saturated >= MIN_SATURATED


def _eval(sub: Path, out: Path) -> Path:
    done = runner.invoke(
        app, ["eval", str(sub), "-p", "eval_v1", "--out", str(out), "--videos", "none"]
    )
    assert done.exit_code == 0, done.output
    return out / "results.json"


def test_the_shipped_balance_policy_beats_pid_on_eval_v1(tmp_path: Path) -> None:
    sub = _shipped("ppo_balance")
    artifacts = os.environ.get("TOPPLE_ARTIFACT_DIR")
    out = Path(artifacts) if artifacts else tmp_path / "ppo"
    ppo = read_results(_eval(sub, out))
    pid = read_results(_eval(REPO / "submissions" / "pid_example.py", tmp_path / "pid"))
    assert ppo.complete
    assert ppo.controller.policy_sha256 is not None
    assert ppo.versions.onnxruntime is not None
    assert ppo.aggregate.score > pid.aggregate.score
