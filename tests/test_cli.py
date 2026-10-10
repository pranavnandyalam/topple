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
