"""PLAN.md Phase 1 done-when, as executable checks.

- gymnasium env_checker passes (tests/test_env.py)
- PID balances >90% of no-push episodes
- same seed -> identical trajectory on the same platform
- `topple run` writes results.json, traj.npz and an MP4
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from typer.testing import CliRunner

from topple.cli import app
from topple.contract import TaskSpec
from topple.controllers.pid import PIDController
from topple.envs.balance import BalanceEnv
from topple.evaluate import EvalProtocol, evaluate
from topple.results import ControllerInfo

INFO = ControllerInfo(id="pid", sha256="0" * 64)


def pid(_spec: TaskSpec) -> PIDController:
    return PIDController()


@pytest.mark.filterwarnings("ignore:.*Box observation space (minimum|maximum).*:UserWarning")
def test_env_checker_passes_for_every_preset() -> None:
    for preset in ("ideal", "clean", "standard"):
        check_env(BalanceEnv(preset), skip_render_check=True)


def test_pid_balances_over_90_percent_without_pushes() -> None:
    protocol = replace(EvalProtocol.from_id("balance-v1/clean"), seeds=tuple(range(500, 550)))
    results = evaluate(protocol, pid, INFO)
    survived = [e.metrics is not None and e.metrics["survived"] for e in results.episodes or []]
    assert sum(survived) / len(survived) > 0.9


def test_same_seed_identical_trajectory() -> None:
    def run() -> np.ndarray:
        protocol = replace(EvalProtocol.from_id("balance-v1/standard"), seeds=(1003,))
        env = BalanceEnv("standard", record=True)
        from topple.evaluate import run_episode

        return run_episode(env, pid, seed=protocol.seeds[0], index=0).trajectory.qpos

    np.testing.assert_array_equal(run(), run())


@pytest.mark.render
def test_topple_run_writes_mp4(tmp_path: Path) -> None:
    submission = tmp_path / "bot.py"
    submission.write_text(
        "from topple.controllers.pid import PIDController\n"
        "def make_controller(spec):\n    return PIDController()\n"
    )
    out = tmp_path / "run"
    result = CliRunner().invoke(app, ["run", str(submission), "--out", str(out)])
    assert result.exit_code == 0, result.output
    for name in ("results.json", "traj_000.npz", "episode.mp4"):
        assert (out / name).stat().st_size > 0, name
