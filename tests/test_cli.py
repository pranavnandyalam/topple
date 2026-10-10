import json
import tomllib
from pathlib import Path

import mujoco
import pytest
from typer.testing import CliRunner

from topple import __version__
from topple.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"topple {__version__}"


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert __version__ == pyproject["project"]["version"]


@pytest.mark.render
def test_smoke_command(tmp_path: Path) -> None:
    out = tmp_path / "smoke.png"
    result = runner.invoke(app, ["smoke", "--out", str(out), "--steps", "10"])
    assert result.exit_code == 0, result.output
    assert "smoke ok: 10 steps" in result.output
    assert out.exists()


def test_smoke_command_rejects_zero_steps() -> None:
    result = runner.invoke(app, ["smoke", "--steps", "0"])
    assert result.exit_code == 2


def test_smoke_command_reports_rendering_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delattr(mujoco, "Renderer", raising=False)
    result = runner.invoke(app, ["smoke", "--out", str(tmp_path / "s.png"), "--steps", "1"])
    assert result.exit_code == 1
    assert "headless rendering unavailable" in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


PID_SUBMISSION = """
from topple.controllers.pid import PIDController

def make_controller(spec):
    return PIDController()
"""


def _submission(tmp_path: Path, text: str = PID_SUBMISSION) -> Path:
    path = tmp_path / "bot.py"
    path.write_text(text)
    return path


def test_run_writes_results_and_trajectory(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "run",
            str(_submission(tmp_path)),
            "--protocol",
            "balance-v1/clean",
            "--seed",
            "7",
            "--out",
            str(out),
            "--no-video",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "results.json").exists()
    assert (out / "traj_000.npz").exists()
    assert "balance-v1/clean: score" in result.output
    assert "seed      7" in result.output


def test_eval_alias_protocol_all_seeds(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    result = runner.invoke(
        app,
        [
            "eval",
            str(_submission(tmp_path)),
            "--protocol",
            "eval_v1_clean",
            "--out",
            str(out),
            "--videos",
            "none",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("traj_*.npz"))) == 20
    assert "balance-v1/clean: score" in result.output


def test_verify_and_score_commands(tmp_path: Path) -> None:
    out = tmp_path / "run"
    runner.invoke(
        app,
        [
            "run",
            str(_submission(tmp_path)),
            "--protocol",
            "balance-v1/standard",
            "--out",
            str(out),
            "--no-video",
        ],
    )
    traj = out / "traj_000.npz"
    verified = runner.invoke(app, ["verify", str(traj)])
    assert verified.exit_code == 0, verified.output
    assert verified.output.startswith("OK")
    scored = runner.invoke(app, ["score", str(traj)])
    assert scored.exit_code == 0, scored.output
    payload = json.loads(scored.output)
    results = json.loads((out / "results.json").read_text())
    assert payload["score"] == pytest.approx(results["episodes"][0]["score"])


def test_verify_detects_tampering(tmp_path: Path) -> None:
    from topple.traj import Trajectory, load_npz

    out = tmp_path / "run"
    runner.invoke(
        app,
        [
            "run",
            str(_submission(tmp_path)),
            "--protocol",
            "balance-v1/clean",
            "--out",
            str(out),
            "--no-video",
        ],
    )
    traj = load_npz(out / "traj_000.npz")
    qpos = traj.qpos.copy()
    qpos[-1, 0] += 0.01
    Trajectory(**{**traj.__dict__, "qpos": qpos}).save_npz(out / "forged.npz")
    result = runner.invoke(app, ["verify", str(out / "forged.npz")])
    assert result.exit_code == 1
    assert result.output.startswith("FAIL")


def test_load_error_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(_submission(tmp_path, "x = 1\n")), "--no-video"])
    assert result.exit_code == 2
    assert "make_controller" in result.output


def test_bad_protocol_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", str(_submission(tmp_path)), "--protocol", "balance-v1/hard", "--no-video"]
    )
    assert result.exit_code == 2
    assert "no preset" in result.output
