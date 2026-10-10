import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import pytest

from topple.rendering import RenderingUnavailableError
from topple.smoke import SMOKE_XML, main, run_smoke


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
def test_run_smoke_physics_is_plausible(tmp_path: Path) -> None:
    result = run_smoke(tmp_path / "s.png", steps=500, width=64, height=48)
    # A damped pendulum released at 0.5 rad has moved, but cannot gain amplitude.
    assert 0.0 < abs(result.final_angle) < result.initial_angle
    assert result.sim_time == pytest.approx(500 * result.timestep)


@pytest.mark.render
def test_run_smoke_equilibrium_stays_put(tmp_path: Path) -> None:
    result = run_smoke(tmp_path / "s.png", steps=500, width=64, height=48, initial_angle=0.0)
    assert result.final_angle == pytest.approx(0.0, abs=1e-12)


def test_run_smoke_rejects_bad_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="steps"):
        run_smoke(tmp_path / "s.png", steps=0)


@pytest.mark.parametrize(("width", "height"), [(0, 10), (10, 0), (-1, -1)])
def test_run_smoke_rejects_bad_size(tmp_path: Path, width: int, height: int) -> None:
    with pytest.raises(ValueError, match="image size"):
        run_smoke(tmp_path / "s.png", steps=1, width=width, height=height)


def test_run_smoke_reports_missing_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # mujoco has no Renderer attribute when its GL context failed to load.
    monkeypatch.delattr(mujoco, "Renderer")
    with pytest.raises(RenderingUnavailableError, match="headless rendering unavailable"):
        run_smoke(tmp_path / "s.png", steps=1)
    assert not (tmp_path / "s.png").exists()


def test_smoke_xml_compiles() -> None:
    model = mujoco.MjModel.from_xml_string(SMOKE_XML)
    assert model.njnt == 1


@pytest.mark.render
def test_main_in_process(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "nested" / "dir" / "smoke.png"
    assert main(["--out", str(out), "--steps", "5"]) == 0
    assert out.exists()
    assert "smoke ok: 5 steps" in capsys.readouterr().out


@pytest.mark.parametrize("steps", ["0", "-3", "abc"])
def test_main_rejects_non_positive_steps(steps: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--steps", steps])
    assert excinfo.value.code == 2


def test_main_reports_rendering_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delattr(mujoco, "Renderer")
    assert main(["--out", str(tmp_path / "s.png"), "--steps", "1"]) == 1
    assert "headless rendering unavailable" in capsys.readouterr().err


@pytest.mark.render
def test_done_when_command_writes_smoke_png(tmp_path: Path, headless_env: dict[str, str]) -> None:
    """The literal Phase 0 done-when: `python -m topple.smoke` with no args, default GL backend."""
    proc = subprocess.run(
        [sys.executable, "-m", "topple.smoke"],
        cwd=tmp_path,
        env=headless_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "smoke ok: 1000 steps" in proc.stdout
    image = iio.imread(tmp_path / "smoke.png")
    assert image.shape == (240, 320, 3)
    assert image.std() > 1.0
