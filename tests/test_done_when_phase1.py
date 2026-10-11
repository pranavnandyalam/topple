"""PLAN.md Phase 1 done-when, as executable checks.

- gymnasium env_checker passes (tests/test_env.py)
- PID balances >90% of no-push episodes
- same seed -> identical trajectory on the same platform (every array and the metadata, within
  a process and across processes)
- `topple run` writes results.json, traj.npz and an MP4, checked through the installed `topple`
  script on the shipped example, as the plan states it
"""

import dataclasses
import json
import math
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import imageio_ffmpeg
import pytest
from gymnasium.utils.env_checker import check_env
from typer.testing import CliRunner

from topple.cli import app
from topple.contract import BalancerObs, TaskSpec
from topple.controllers.pid import PIDController
from topple.envs.balance import BalanceEnv
from topple.evaluate import (
    ControllerFactory,
    EvalProtocol,
    evaluate,
    iter_episodes,
    run_episode,
)
from topple.loader import load_submission
from topple.results import ControllerInfo, read_results
from topple.traj import Trajectory, load_npz

INFO = ControllerInfo(id="pid", sha256="0" * 64)
REPO_ROOT = Path(__file__).resolve().parents[1]
PID_EXAMPLE = Path("submissions") / "pid_example.py"


def pid(_spec: TaskSpec) -> PIDController:
    return PIDController()


class _Zero:
    """Never drives the wheels, so the robot falls and the record has an epilogue."""

    def reset(self, rng_seed: int) -> None:
        pass

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        return (0.0, 0.0)


def zero(_spec: TaskSpec) -> _Zero:
    return _Zero()


def assert_identical(a: Trajectory, b: Trajectory) -> None:
    """Every array byte for byte, with its dtype and shape (so ``-0.0 != 0.0``), and the
    metadata as canonical JSON."""
    for f in dataclasses.fields(Trajectory):
        if f.name == "meta":
            continue
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert (x.dtype, x.shape) == (y.dtype, y.shape), f.name
        assert x.tobytes() == y.tobytes(), f.name
    assert json.dumps(a.meta, sort_keys=True) == json.dumps(b.meta, sort_keys=True)


@pytest.mark.filterwarnings("ignore:.*Box observation space (minimum|maximum).*:UserWarning")
def test_env_checker_passes_for_every_preset() -> None:
    for preset in ("ideal", "clean", "standard"):
        check_env(BalanceEnv(preset), skip_render_check=True)


@pytest.mark.slow
def test_pid_balances_over_90_percent_without_pushes() -> None:
    protocol = replace(EvalProtocol.from_id("balance-v1/clean"), seeds=tuple(range(500, 550)))
    results = evaluate(protocol, pid, INFO)
    survived = [e.metrics is not None and e.metrics["survived"] for e in results.episodes or []]
    assert sum(survived) / len(survived) > 0.9


@pytest.mark.parametrize(("factory", "seed"), [(pid, 1003), (zero, 1001)])
def test_same_seed_identical_trajectory(factory: ControllerFactory, seed: int) -> None:
    """A survivor (PID) and a fall (zero commands, so the epilogue is compared too), each run
    twice in fresh envs: the whole record is identical, not just ``qpos``."""

    def run() -> Trajectory:
        env = BalanceEnv("standard", record=True)
        try:
            return run_episode(env, factory, seed=seed, index=0).trajectory
        finally:
            env.close()

    first, second = run(), run()
    assert first.epilogue_qpos.shape[0] == (100 if factory is zero else 0)
    assert_identical(first, second)


@pytest.mark.render
def test_done_when_command_writes_results_trajectory_and_mp4(
    tmp_path: Path, headless_env: dict[str, str]
) -> None:
    """The literal done-when: the installed ``topple`` script (``[project.scripts]``) runs the
    shipped ``submissions/pid_example.py`` from the repository root, in a fresh headless
    process with the default GL backend. Its trajectory and result then match the same seed
    run here, in this process: identical on the same platform across processes too."""
    script = shutil.which("topple", path=str(Path(sys.executable).parent))
    assert script is not None, "the topple console script is not installed next to Python"
    out = tmp_path / "run"
    proc = subprocess.run(
        [script, "run", str(PID_EXAMPLE), "--out", str(out)],
        cwd=REPO_ROOT,
        env=headless_env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "balance-v1/standard: score" in proc.stdout
    assert sorted(p.name for p in out.iterdir()) == ["episode.mp4", "results.json", "traj_000.npz"]
    results = read_results(out / "results.json")
    assert (results.protocol, results.complete, results.controller.id) == (
        "balance-v1/standard",
        False,
        "pid_example",
    )
    traj = load_npz(out / "traj_000.npz", max_steps=1000)
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(out / "episode.mp4"))
    assert frames == math.ceil(traj.qpos.shape[0] / 2) == 501  # every recorded row at 50 fps
    assert seconds == pytest.approx(10.02, abs=0.05)

    protocol = EvalProtocol.from_id("balance-v1/standard")
    first = replace(protocol, seeds=protocol.seeds[:1])
    here = next(iter_episodes(first, load_submission(REPO_ROOT / PID_EXAMPLE).factory))
    assert results.episodes == [here.result]
    assert_identical(traj, here.trajectory)


@pytest.mark.render
def test_topple_run_writes_mp4(tmp_path: Path) -> None:
    """The same command in this process, through Typer (measured by coverage)."""
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
    traj = load_npz(out / "traj_000.npz", max_steps=1000)
    assert traj.epilogue_qpos.shape[0] == 0  # the PID survives the first public seed
    frames = math.ceil(traj.qpos.shape[0] / 2)  # every recorded row at 50 fps: 501
    assert imageio_ffmpeg.count_frames_and_secs(str(out / "episode.mp4"))[0] == frames
