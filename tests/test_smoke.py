import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from topple.smoke import SMOKE_XML, run_smoke


@pytest.mark.render
def test_run_smoke_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "smoke.png"
    result = run_smoke(out, steps=1000, width=160, height=120)

    assert out.exists()
    image = iio.imread(out)
    assert image.shape == (120, 160, 3)
    # A real render has structure: not a single flat color.
    assert image.std() > 1.0
    assert result.steps == 1000
    assert result.image_path == out


@pytest.mark.render
def test_run_smoke_physics_moves(tmp_path: Path) -> None:
    result = run_smoke(tmp_path / "s.png", steps=500, width=64, height=48)
    # The pendulum starts tilted and must swing under gravity.
    assert not np.isclose(result.initial_angle, result.final_angle)
    assert result.sim_time == pytest.approx(500 * result.timestep)


def test_run_smoke_rejects_bad_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="steps"):
        run_smoke(tmp_path / "s.png", steps=0)


def test_smoke_xml_compiles() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_string(SMOKE_XML)
    assert model.njnt == 1


@pytest.mark.render
def test_module_entrypoint_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "cli_smoke.png"
    proc = subprocess.run(
        [sys.executable, "-m", "topple.smoke", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "smoke ok" in proc.stdout


@pytest.mark.parametrize(("width", "height"), [(0, 10), (10, 0), (-1, -1)])
def test_run_smoke_rejects_bad_size(tmp_path: Path, width: int, height: int) -> None:
    with pytest.raises(ValueError, match="image size"):
        run_smoke(tmp_path / "s.png", steps=1, width=width, height=height)


@pytest.mark.render
def test_main_in_process(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from topple.smoke import main

    out = tmp_path / "nested" / "dir" / "smoke.png"
    assert main(["--out", str(out), "--steps", "5"]) == 0
    assert out.exists()
    assert "smoke ok: 5 steps" in capsys.readouterr().out
