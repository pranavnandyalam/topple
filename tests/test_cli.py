from pathlib import Path

import pytest
from typer.testing import CliRunner

from topple import __version__
from topple.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.render
def test_smoke_command(tmp_path: Path) -> None:
    out = tmp_path / "smoke.png"
    result = runner.invoke(app, ["smoke", "--out", str(out), "--steps", "10"])
    assert result.exit_code == 0, result.output
    assert out.exists()
