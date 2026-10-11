import math
import struct
from contextlib import closing
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

import imageio.v2 as iio2
import imageio.v3 as iio
import imageio_ffmpeg
import mujoco
import numpy as np
import numpy.typing as npt
import pytest

import topple.video
from topple.envs import BalanceEnv
from topple.rendering import RenderingUnavailableError
from topple.traj import Trajectory
from topple.video import render_frames, render_trajectory

SEED = 1  # ideal preset: starts ~0.04 rad off vertical and falls after ~50 zero-torque steps
SMALL = (64, 48)  # cheap frames for pixel tests (both multiples of 16)
Frame = npt.NDArray[np.uint8]


@pytest.fixture(scope="module")
def fallen() -> Trajectory:
    """A real recorded episode: zero torque until the robot falls, then 0.5 s of epilogue."""
    env = BalanceEnv(preset="ideal", record=True)
    env.reset(seed=SEED)
    terminated = False
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step([0.0, 0.0])
        if terminated or truncated:
            break
    assert terminated, "a balancer with zero torque must fall"
    env.simulate_epilogue(0.5)
    traj = env.trajectory()
    assert traj.steps < 100
    assert traj.epilogue_qpos.shape[0] == 50
    return traj


def n_rows(traj: Trajectory, include_epilogue: bool = True) -> int:
    return traj.qpos.shape[0] + (traj.epilogue_qpos.shape[0] if include_epilogue else 0)


def first_frames(traj: Trajectory, n: int = 1, **kwargs: Any) -> list[Frame]:
    with closing(render_frames(traj, **kwargs)) as frames:
        return list(islice(frames, n))


def with_push(traj: Trajectory, rows: dict[int, tuple[float, float, float]]) -> Trajectory:
    force = np.zeros_like(traj.push_force)
    for row, value in rows.items():
        force[row] = value
    return replace(traj, push_force=force)


def changed_pixels(a: Frame, b: Frame) -> npt.NDArray[np.bool_]:
    mask: npt.NDArray[np.bool_] = np.any(a != b, axis=2)
    return mask


def red_pixels(frame: Frame) -> npt.NDArray[np.bool_]:
    r, g, b = (frame[..., c].astype(int) for c in range(3))
    mask: npt.NDArray[np.bool_] = (r > 140) & (g < 90) & (b < 90)
    return mask


def blue_pixels(frame: Frame) -> npt.NDArray[np.bool_]:
    """The chassis box (rgba 0.16 0.42 0.86)."""
    r, b = frame[..., 0].astype(int), frame[..., 2].astype(int)
    mask: npt.NDArray[np.bool_] = (b > 140) & (r < 100)
    return mask


def top_level_boxes(path: Path) -> list[str]:
    """Box (atom) types at the top level of an ISO-BMFF/MP4 file, in file order."""
    data = path.read_bytes()
    boxes: list[str] = []
    pos = 0
    while pos + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[pos : pos + 8])
        if size == 1:  # 64-bit size follows the type
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
        elif size == 0:  # box runs to the end of the file
            size = len(data) - pos
        assert size >= 8, f"corrupt box {kind!r} at offset {pos}"
        boxes.append(kind.decode("latin-1"))
        pos += size
    assert pos == len(data), "trailing bytes after the last box"
    return boxes


# ---------------------------------------------------------------- files


@pytest.mark.render
def test_mp4_is_faststart_h264_with_one_frame_per_stride(
    fallen: Trajectory, tmp_path: Path
) -> None:
    path = tmp_path / "videos" / "episode.mp4"
    written = render_trajectory(fallen, path)  # 50 fps: every 2nd control step

    assert written == math.ceil(n_rows(fallen) / 2)
    assert path.stat().st_size > 0
    assert [p.name for p in path.parent.iterdir()] == ["episode.mp4"]  # no temp file left
    assert imageio_ffmpeg.count_frames_and_secs(str(path))[0] == written
    boxes = top_level_boxes(path)
    assert boxes[0] == "ftyp"
    assert boxes.index("moov") < boxes.index("mdat")  # +faststart: playable while downloading
    with iio2.get_reader(path, format="FFMPEG") as reader:
        meta = reader.get_meta_data()
    assert meta["codec"] == "h264"
    assert meta["pix_fmt"].startswith("yuv420p")
    assert meta["size"] == (320, 240)
    assert meta["fps"] == 50


@pytest.mark.render
def test_gif_loops_forever_and_accounts_for_every_frame(fallen: Trajectory, tmp_path: Path) -> None:
    path = tmp_path / "episode.gif"
    written = render_trajectory(fallen, path, fps=25, size=(160, 128))
    assert written == math.ceil(n_rows(fallen) / 4)

    props = iio.improps(path, index=None)
    assert props.shape[1:] == (128, 160, 3)
    stored = props.n_images
    durations = [iio.immeta(path, index=i)["duration"] for i in range(stored)]
    # Pillow stores a run of identical frames as one longer frame, so count frame time: every
    # written frame lasts exactly 1/fps = 40 ms.
    assert 1 <= stored <= written
    assert all(d > 0 and d % 40 == 0 for d in durations)
    assert sum(durations) == written * 40
    assert iio.immeta(path, index=None)["loop"] == 0  # loop forever


@pytest.mark.render
@pytest.mark.parametrize("suffix", [".mp4", ".gif"])
def test_failure_mid_render_leaves_no_file(
    fallen: Trajectory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    real_forward = mujoco.mj_forward
    calls = 0

    def flaky_forward(model: Any, data: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("GPU lost")
        real_forward(model, data)

    monkeypatch.setattr(mujoco, "mj_forward", flaky_forward)
    with pytest.raises(RuntimeError, match="GPU lost"):
        render_trajectory(fallen, tmp_path / f"episode{suffix}", fps=25, size=SMALL)
    assert list(tmp_path.iterdir()) == []


def test_rendering_unavailable_creates_no_files(
    fallen: Trajectory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RenderingUnavailableError("headless rendering unavailable: test")

    monkeypatch.setattr(topple.video, "make_renderer", unavailable)
    with pytest.raises(RenderingUnavailableError):
        render_trajectory(fallen, tmp_path / "out" / "episode.mp4")
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------- frames


@pytest.mark.render
@pytest.mark.parametrize(
    ("fps", "include_epilogue"),
    [(100, True), (50, True), (50, False), (20, True), (1, True), (1, False)],
)
def test_frame_count_is_rows_over_stride(
    fallen: Trajectory, fps: int, include_epilogue: bool
) -> None:
    frames = list(render_frames(fallen, fps=fps, size=SMALL, include_epilogue=include_epilogue))
    assert len(frames) == math.ceil(n_rows(fallen, include_epilogue) / (100 // fps))
    assert all(f.shape == (48, 64, 3) and f.dtype == np.uint8 for f in frames)


@pytest.mark.render
def test_frames_replay_stored_rows_in_order_without_stepping(
    fallen: Trajectory, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_step(*_args: Any) -> None:
        raise AssertionError("rendering must not step the simulation")

    monkeypatch.setattr(mujoco, "mj_step", no_step)
    rewound = replace(
        fallen,
        qpos=fallen.qpos[::-1],
        qvel=fallen.qvel[::-1],
        epilogue_qpos=fallen.epilogue_qpos[::-1],
    )
    frames = list(render_frames(fallen, fps=100, size=SMALL))
    rewound_frames = list(render_frames(rewound, fps=100, size=SMALL))

    # At 100 fps frame i shows stored row i (qpos, then epilogue), so reversing the rows of a
    # segment reverses that segment's frames.
    end = fallen.qpos.shape[0]
    assert len(frames) == end + fallen.epilogue_qpos.shape[0]
    assert not np.array_equal(frames[0], frames[end - 1])  # upright vs falling
    assert not np.array_equal(frames[end], frames[-1])  # the epilogue keeps moving
    for segment in (slice(0, end), slice(end, None)):
        for frame, same_row in zip(frames[segment], reversed(rewound_frames[segment]), strict=True):
            np.testing.assert_array_equal(frame, same_row)


@pytest.mark.render
def test_non_finite_states_hold_the_last_finite_pose(fallen: Trajectory) -> None:
    """A sim_unstable episode can end in a NaN state: the replay freezes instead of going black."""
    qpos = fallen.qpos.copy()
    qpos[-1] = np.nan
    broken = replace(fallen, qpos=qpos, epilogue_qpos=fallen.epilogue_qpos[:0])
    frames = list(render_frames(broken, fps=100, size=SMALL))
    np.testing.assert_array_equal(frames[-1], frames[-2])
    assert frames[-1].std() > 1.0


# ---------------------------------------------------------------- push arrows


@pytest.mark.render
def test_push_arrow_is_red_and_only_drawn_while_the_force_acts(fallen: Trajectory) -> None:
    plain = first_frames(fallen, 2)
    pushed = first_frames(with_push(fallen, {0: (3.0, 0.0, 0.0)}), 2)  # 50 fps: rows 0 and 2

    assert changed_pixels(pushed[0], plain[0]).sum() > 100
    assert red_pixels(pushed[0]).sum() > 60
    assert red_pixels(plain[0]).sum() == 0
    np.testing.assert_array_equal(pushed[1], plain[1])  # push over by row 2


@pytest.mark.render
def test_arrow_head_is_visible_and_touches_the_robot(fallen: Trajectory) -> None:
    """Even a small push's arrow ends on the chassis surface: no gap (MuJoCo draws an arrow over
    half its connector) and the head not buried inside the body."""
    plain = first_frames(fallen)[0]
    arrow = changed_pixels(first_frames(with_push(fallen, {0: (2.0, 0.0, 0.0)}))[0], plain)
    box = blue_pixels(plain)
    rows = [r for r in range(arrow.shape[0]) if arrow[r].any() and box[r].any()]
    # The arrow arrives from the left; its apex row comes closest to the box's left edge.
    gaps = {r: int(np.nonzero(box[r])[0].min() - np.nonzero(arrow[r])[0].max()) for r in rows}
    apex = min(gaps, key=lambda r: gaps[r])
    assert abs(gaps[apex]) <= 2

    thickness = arrow[apex - 10 : apex + 11].sum(axis=0)  # this band excludes the floor shadow
    columns = np.nonzero(thickness)[0]
    shaft = float(np.median(thickness[columns[: len(columns) // 2]]))
    assert thickness.max() >= 1.4 * shaft  # the head (1.75x the shaft radius) is on screen


@pytest.mark.render
def test_show_pushes_false_hides_the_arrow(fallen: Trajectory) -> None:
    pushed = with_push(fallen, {0: (3.0, 0.0, 0.0)})
    np.testing.assert_array_equal(
        first_frames(pushed, show_pushes=False)[0], first_frames(fallen)[0]
    )


@pytest.mark.render
def test_push_between_sampled_rows_is_still_shown(fallen: Trajectory) -> None:
    # 10 fps samples every 10th control step; a push covering only step 7 shows on frame 0.
    plain = first_frames(fallen, fps=10)[0]
    pushed = first_frames(with_push(fallen, {7: (3.0, 0.0, 0.0)}), fps=10)[0]
    assert changed_pixels(pushed, plain).sum() > 100


@pytest.mark.render
def test_arrow_points_along_the_force(fallen: Trajectory) -> None:
    plain = first_frames(fallen)[0]

    def arrow_column(force: tuple[float, float, float]) -> float:
        frame = first_frames(with_push(fallen, {0: force}))[0]
        return float(np.nonzero(changed_pixels(frame, plain))[1].mean())

    # The track camera looks along world +y, so world +x is image right. The arrow's tip touches
    # the robot and its tail trails behind the force: a +x shove arrives from the robot's left.
    robot_column = 320 // 2  # the track camera keeps the robot's center of mass mid-frame
    assert arrow_column((3.0, 0.0, 0.0)) < robot_column < arrow_column((-3.0, 0.0, 0.0))


@pytest.mark.render
def test_arrow_length_grows_with_force_and_is_capped(fallen: Trajectory) -> None:
    plain = first_frames(fallen)[0]

    def frame(newtons: float) -> Frame:
        return first_frames(with_push(fallen, {0: (newtons, 0.0, 0.0)}))[0]

    def length(newtons: float) -> int:
        return int(changed_pixels(frame(newtons), plain).sum())

    assert length(2.0) < length(4.0) < length(20.0)
    np.testing.assert_array_equal(frame(20.0), frame(40.0))  # both beyond the cap


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("size", "message"),
    [
        ((321, 240), "multiples of 16"),
        ((320, 250), "multiples of 16"),
        ((0, 240), "multiples of 16"),
        ((320, -16), "multiples of 16"),
        ((1296, 720), "framebuffer"),
        ((1280, 736), "framebuffer"),
    ],
)
def test_size_must_be_multiples_of_16_within_the_framebuffer(
    fallen: Trajectory, tmp_path: Path, size: tuple[int, int], message: str
) -> None:
    path = tmp_path / "out" / "episode.mp4"
    with pytest.raises(ValueError, match=message):
        render_trajectory(fallen, path, size=size)
    with pytest.raises(ValueError, match=message):
        render_frames(fallen, size=size)  # validated before the first frame is requested
    assert not path.parent.exists()


@pytest.mark.parametrize("fps", [0, -50, 3, 30, 200])
def test_fps_must_divide_the_control_rate(fallen: Trajectory, tmp_path: Path, fps: int) -> None:
    with pytest.raises(ValueError, match="fps"):
        render_trajectory(fallen, tmp_path / "episode.mp4", fps=fps)
    assert list(tmp_path.iterdir()) == []


def test_gif_rejects_frame_delays_that_browsers_slow_down(
    fallen: Trajectory, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="GIF"):
        render_trajectory(fallen, tmp_path / "episode.gif", fps=100)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["episode.avi", "episode.webm", "episode.png", "episode"])
def test_unsupported_suffix(fallen: Trajectory, tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="suffix"):
        render_trajectory(fallen, tmp_path / "out" / name)
    assert not (tmp_path / "out").exists()


def test_qpos_width_must_match_the_model(fallen: Trajectory, tmp_path: Path) -> None:
    narrow = replace(fallen, qpos=fallen.qpos[:, :5], epilogue_qpos=fallen.epilogue_qpos[:, :5])
    with pytest.raises(ValueError, match="nq=9"):
        render_trajectory(narrow, tmp_path / "episode.mp4")


@pytest.mark.parametrize("meta", [{}, {"challenge": 7}])
def test_meta_must_name_the_challenge(
    fallen: Trajectory, tmp_path: Path, meta: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="challenge"):
        render_trajectory(replace(fallen, meta=meta), tmp_path / "episode.mp4")


def test_unknown_camera(fallen: Trajectory, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"camera 'overhead' does not exist; cameras: \['track'\]"):
        render_trajectory(fallen, tmp_path / "episode.mp4", camera="overhead")


# ---------------------------------------------------------------- trust (spec I, C)


@pytest.mark.parametrize("robot_hash", ["0" * 64, None, 7])
def test_refuses_a_trajectory_of_another_robot(
    fallen: Trajectory, tmp_path: Path, robot_hash: Any
) -> None:
    """Never draw the wrong robot: the recorded robot_hash must be the bundle's."""
    meta = {**fallen.meta, "robot_hash": robot_hash}
    with pytest.raises(ValueError, match="robot_hash"):
        render_trajectory(replace(fallen, meta=meta), tmp_path / "episode.mp4")
    assert list(tmp_path.iterdir()) == []


def played_rows(traj: Trajectory, include_epilogue: bool = True) -> int:
    play = topple.video._playback(traj, 100, SMALL, "track", include_epilogue, False)
    return int(play.rows.shape[0])


def test_epilogue_follows_the_epilogue_rule(fallen: Trajectory) -> None:
    """Epilogue frames only after a fall or a controller error (contract §11), never after a
    survival or sim_unstable, whatever rows the file carries."""
    assert played_rows(fallen) == n_rows(fallen)
    assert played_rows(fallen, include_epilogue=False) == fallen.qpos.shape[0]
    for reason in ("timeout", "sim_unstable"):
        meta = {**fallen.meta, "termination": {"step": fallen.steps, "reason": reason}}
        assert played_rows(replace(fallen, meta=meta)) == fallen.qpos.shape[0]
    errored = {
        **fallen.meta,
        "termination": {"step": fallen.steps, "reason": "timeout"},
        "error": {"code": "exception", "step": fallen.steps, "detail": "x"},
    }
    assert played_rows(replace(fallen, meta=errored)) == n_rows(fallen)


@pytest.mark.parametrize("termination", [None, "fell", {"reason": 3}, {}])
def test_malformed_termination_shows_no_epilogue(fallen: Trajectory, termination: Any) -> None:
    meta = {**fallen.meta, "termination": termination}
    assert played_rows(replace(fallen, meta=meta)) == fallen.qpos.shape[0]
