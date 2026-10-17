"""``topple.course_video``: waypoint episodes in one scene on one clock (§8.9 item 8)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from topple import course_video as cv
from topple.evaluate import EvalProtocol, iter_episodes
from topple.loader import load_submission
from topple.traj import Trajectory
from topple.video import WAYPOINT_LADDER, GifRung, GifTooLargeError

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "submissions" / "waypoint_pid_example.py"
CREEPER = REPO / "submissions" / "anchors" / "creeper.py"


def episode(sub: Path, preset: str, seed: int) -> Trajectory:
    proto = EvalProtocol.from_id(f"waypoint-v1/{preset}")
    proto = replace(proto, seeds=(seed,))
    factory = load_submission(sub, challenge=proto.challenge).factory
    return next(iter(iter_episodes(proto, factory))).trajectory


@pytest.fixture(scope="module")
def pair() -> tuple[Trajectory, Trajectory]:
    return episode(DRIVER, "clean", 3000), episode(CREEPER, "clean", 3000)


def test_a_docked_track_ends_at_its_finish(pair: tuple[Trajectory, Trajectory]) -> None:
    driver, creeper = (
        cv.make_track(t, n) for t, n in zip(pair, ("driver", "creeper"), strict=True)
    )
    assert driver.t_finish is not None
    assert driver.t_fall is None
    assert driver.end_s == driver.t_finish
    assert creeper.t_finish is None  # the creeper never docks within 20 s
    assert creeper.end_s == 20.0
    assert cv.end_time([driver]) == pytest.approx(driver.t_finish + cv.TAIL_S)
    assert cv.end_time([driver, creeper]) == 20.0


def test_the_hud_counts_gates_then_shows_the_finish(pair: tuple[Trajectory, Trajectory]) -> None:
    t = cv.make_track(pair[0], "driver")
    assert cv._hud(t, 0) == "driver: gate 0/4"
    assert t.t_finish is not None
    assert cv._hud(t, round(t.t_finish / t.control_dt)) == f"driver: docked {t.t_finish:.2f} s"


def test_the_camera_frames_the_course_within_the_clamp(
    pair: tuple[Trajectory, Trajectory],
) -> None:
    tracks = [cv.make_track(t, "x") for t in pair]
    cam = cv.camera(tracks, 45.0)
    side = 2 * cam.distance * np.tan(np.radians(22.5))
    assert cv.FRAME_M[0] - 1e-9 <= side <= cv.FRAME_M[1] + 1e-9
    assert cam.elevation == cv.ELEVATION_DEG


def test_frames_are_square_and_follow_the_clock(pair: tuple[Trajectory, Trajectory]) -> None:
    frames = list(cv.course_frames([pair[0]], ["driver"], fps=10, size=160))
    t = cv.make_track(pair[0], "driver")
    assert len(frames) == round(cv.end_time([t]) / 0.01) // 10 + 1
    assert all(f.shape == (160, 160, 3) and f.dtype == np.uint8 for f in frames)
    assert frames[0].std() > 0  # something is drawn


def test_two_robots_render_in_one_scene(pair: tuple[Trajectory, Trajectory]) -> None:
    one = next(iter(cv.course_frames([pair[0]], ["a"], fps=10, size=160)))
    two = next(iter(cv.course_frames(list(pair), ["a", "b"], fps=10, size=160)))
    assert one.shape == two.shape
    assert not np.array_equal(one, two)  # the second robot and its HUD line are drawn


def test_trajectories_on_different_seeds_are_refused(pair: tuple[Trajectory, Trajectory]) -> None:
    other = episode(DRIVER, "clean", 3001)
    with pytest.raises(ValueError, match="one scenario"):
        cv.course_frames([pair[0], other], ["a", "b"], fps=10, size=160)


def test_bad_arguments_fail_before_rendering(pair: tuple[Trajectory, Trajectory]) -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        cv.course_frames([pair[0]], ["a"], fps=10, size=100)
    with pytest.raises(ValueError, match="labelled"):
        cv.course_frames([pair[0]], [], fps=10, size=160)
    with pytest.raises(ValueError, match="fps"):
        cv.course_frames([pair[0]], ["a"], fps=7, size=160)


def test_the_gif_ladder_takes_the_first_rung_within_the_cap(
    pair: tuple[Trajectory, Trajectory], tmp_path: Path
) -> None:
    out = tmp_path / "wp.gif"
    rung = cv.render_course_gif([pair[0]], ["a"], out, (GifRung(160, 5),), max_bytes=5 << 20)
    assert rung == GifRung(160, 5)
    assert out.read_bytes()[:6] == b"GIF89a"
    with pytest.raises(GifTooLargeError):
        cv.render_course_gif([pair[0]], ["a"], tmp_path / "small.gif", (GifRung(160, 5),),
                             max_bytes=100)  # fmt: skip
    assert not (tmp_path / "small.gif").exists()
    assert WAYPOINT_LADDER[0] == GifRung(480, 10)
