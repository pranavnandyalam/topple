"""Replay videos: render a stored trajectory to MP4 or GIF.

Playback only, never re-simulation: each frame poses the challenge's robot at a recorded state
(``qpos`` rows, then ``epilogue_qpos`` when the epilogue rule of contract §11 calls for one) with
``mj_forward`` and draws it from the chassis-tracking camera. A trajectory recorded with another
robot (``meta.robot_hash``) is refused rather than drawn as the wrong robot. Rows are recorded at
the control rate and subsampled to ``fps``. An active push appears as a red arrow along the
world-frame force, its tip touching the robot where the force's line of action through the
``push_point`` site enters the body, its length proportional to the force.
"""

from __future__ import annotations

import itertools
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import mujoco
import numpy as np
import numpy.typing as npt

from topple._atomic import replacement
from topple.challenge import load_challenge
from topple.evaluate import wants_epilogue
from topple.rendering import make_renderer
from topple.robot import PUSH_SITE, TRACK_CAMERA, build_model, robot_hash
from topple.traj import Trajectory

SUFFIXES = (".mp4", ".gif")
MACRO_BLOCK = 16  # H.264 macroblock: frame sides must be multiples of it
GIF_MAX_FPS = 50  # browsers play GIF frame delays below 20 ms at 10 fps
ARROW_M_PER_N = 0.04  # a standard 2.5-4.5 N shove draws as 10-18 cm
ARROW_MAX_M = 0.4
ARROW_WIDTH_M = 0.01
ARROW_RGBA = np.array([0.9, 0.1, 0.1, 1.0], dtype=np.float32)
_RAY_START_M = 1.0  # cast from this far out along the force line to find the robot's surface

Frame = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class _Playback:
    model: mujoco.MjModel
    rows: npt.NDArray[np.float64]  # one qpos row per frame
    pushes: tuple[npt.NDArray[np.float64] | None, ...]  # push force to draw on each frame
    width: int
    height: int
    camera: str


def render_trajectory(
    traj: Trajectory,
    path: Path,
    *,
    fps: int = 50,
    size: tuple[int, int] = (320, 240),
    camera: str = TRACK_CAMERA,
    include_epilogue: bool = True,
    show_pushes: bool = True,
) -> int:
    """Write ``traj`` to ``path`` and return the number of frames written.

    ``.mp4`` is H.264/yuv420p with the index up front (``+faststart``) so browsers can stream it;
    ``.gif`` loops forever. ``path`` appears only once every frame is encoded. Bad arguments
    (``ValueError``) and a missing GL backend (``RenderingUnavailableError``) raise before any
    file or directory is created.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUFFIXES:
        raise ValueError(f"unsupported video suffix {path.suffix!r}; use one of {SUFFIXES}")
    if suffix == ".gif" and fps > GIF_MAX_FPS:
        raise ValueError(
            f"GIF frame delays below {1000 // GIF_MAX_FPS} ms play at 10 fps in browsers; "
            f"use fps <= {GIF_MAX_FPS} for .gif"
        )
    frames = render_frames(
        traj,
        fps=fps,
        size=size,
        camera=camera,
        include_epilogue=include_epilogue,
        show_pushes=show_pushes,
    )
    try:
        first = next(frames)  # opens the renderer, so a missing GL backend fails before any I/O
        path.parent.mkdir(parents=True, exist_ok=True)
        return _write(path, fps, itertools.chain([first], frames))
    finally:
        frames.close()


def render_frames(
    traj: Trajectory,
    *,
    fps: int = 50,
    size: tuple[int, int] = (320, 240),
    camera: str = TRACK_CAMERA,
    include_epilogue: bool = True,
    show_pushes: bool = True,
) -> Generator[Frame, None, None]:
    """Yield the replay's RGB frames (height x width x 3, uint8).

    Arguments are validated immediately. The renderer opens at the first frame and closes when
    the generator is exhausted or closed.
    """
    return _frames(_playback(traj, fps, size, camera, include_epilogue, show_pushes))


def _playback(
    traj: Trajectory,
    fps: int,
    size: tuple[int, int],
    camera: str,
    include_epilogue: bool,
    show_pushes: bool,
) -> _Playback:
    width, height = size
    if min(width, height) <= 0 or width % MACRO_BLOCK or height % MACRO_BLOCK:
        raise ValueError(
            f"video size must be positive multiples of {MACRO_BLOCK}, got {width}x{height}"
        )
    model, control_dt = _model(traj)
    max_w, max_h = model.vis.global_.offwidth, model.vis.global_.offheight
    if width > max_w or height > max_h:
        raise ValueError(
            f"video size {width}x{height} exceeds the {max_w}x{max_h} offscreen framebuffer"
        )
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
        names = [model.camera(i).name for i in range(model.ncam)]
        raise ValueError(f"camera {camera!r} does not exist; cameras: {names}")
    stride = _stride(fps, control_dt)
    with_epilogue = include_epilogue and _shows_epilogue(traj.meta)
    rows = np.concatenate([traj.qpos, traj.epilogue_qpos]) if with_epilogue else traj.qpos
    starts = range(0, rows.shape[0], stride)
    pushes = tuple(_push_at(traj.push_force, k, stride) if show_pushes else None for k in starts)
    return _Playback(model, rows[::stride], pushes, width, height, camera)


def _model(traj: Trajectory) -> tuple[mujoco.MjModel, float]:
    """The challenge's robot with nominal parameters (a replay needs only its geometry)."""
    challenge_id = traj.meta.get("challenge")
    if not isinstance(challenge_id, str):
        raise ValueError("trajectory meta does not name its challenge")
    challenge = load_challenge(challenge_id)
    expected = robot_hash(challenge.robot_xml, challenge.params)
    recorded = traj.meta.get("robot_hash")
    if recorded != expected:
        shown = recorded[:16] + "..." if isinstance(recorded, str) else repr(recorded)[:20]
        raise ValueError(
            f"trajectory meta.robot_hash {shown} is not {challenge_id}'s robot "
            f"({expected[:16]}...); refusing to draw the wrong robot"
        )
    model = build_model(challenge.robot_xml, challenge.scene_xml, challenge.params)
    if model.nq != traj.qpos.shape[1]:
        raise ValueError(
            f"trajectory qpos has {traj.qpos.shape[1]} columns but {challenge_id} has nq={model.nq}"
        )
    return model, challenge.config.control_dt


def _shows_epilogue(meta: dict[str, Any]) -> bool:
    """Contract §11: epilogue frames only after a fall or a controller error."""
    termination = meta.get("termination")
    reason = termination.get("reason") if isinstance(termination, dict) else None
    if not isinstance(reason, str):
        return False
    return wants_epilogue(reason, has_error=meta.get("error") is not None)


def _stride(fps: int, control_dt: float) -> int:
    rate = round(1.0 / control_dt)
    if fps <= 0 or rate % fps:
        raise ValueError(f"fps must divide the {rate} Hz control rate, got {fps}")
    return rate // fps


def _push_at(
    push_force: npt.NDArray[np.float64], row: int, stride: int
) -> npt.NDArray[np.float64] | None:
    """Strongest push during the control steps a frame stands for, so short pushes still show."""
    window = push_force[row : row + stride]
    if window.shape[0] == 0:
        return None
    norms = np.linalg.norm(window, axis=1)
    k = int(np.argmax(norms))
    force: npt.NDArray[np.float64] = window[k]
    return force if np.isfinite(norms[k]) and norms[k] > 0 else None


def _frames(play: _Playback) -> Generator[Frame, None, None]:
    model = play.model
    data = mujoco.MjData(model)
    push_site = model.site(PUSH_SITE).id
    with make_renderer(model, height=play.height, width=play.width) as renderer:
        for row, push in zip(play.rows, play.pushes, strict=True):
            if np.isfinite(row).all():  # sim_unstable episodes can end in NaN: hold the last pose
                data.qpos[:] = row
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=play.camera)
            if push is not None:
                tip = _entry_point(model, data, data.site_xpos[push_site].copy(), push)
                _draw_arrow(renderer.scene, tip, push)
            frame: Frame = renderer.render()
            yield frame


def _entry_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    point: npt.NDArray[np.float64],
    force: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Where the line along ``force`` through ``point`` enters the robot (``point`` if never)."""
    direction = force / np.linalg.norm(force)
    start = point - direction * _RAY_START_M
    geom = np.zeros(1, dtype=np.int32)
    # flg_static=0 skips world geoms (the floor); returns -1 when nothing is hit.
    dist = float(mujoco.mj_ray(model, data, start, direction, None, 0, -1, geom))
    if 0.0 <= dist < _RAY_START_M:
        hit: npt.NDArray[np.float64] = start + direction * dist
        return hit
    return point


def _draw_arrow(
    scene: mujoco.MjvScene, tip: npt.NDArray[np.float64], force: npt.NDArray[np.float64]
) -> None:
    """Append a red arrow along ``force`` ending at ``tip``, ARROW_M_PER_N long per newton."""
    if scene.ngeom >= scene.maxgeom:
        return
    magnitude = float(np.linalg.norm(force))
    tail = tip - force / magnitude * min(ARROW_M_PER_N * magnitude, ARROW_MAX_M)
    geom = scene.geoms[scene.ngeom]
    arrow = mujoco.mjtGeom.mjGEOM_ARROW
    mujoco.mjv_initGeom(geom, arrow, np.zeros(3), np.zeros(3), np.eye(3).ravel(), ARROW_RGBA)
    # MuJoCo draws an arrow over the first half of its connector (render_gl3.c: shaft to 1/3 of
    # size[2], head to 1/2), so aim it twice as far to put the drawn tip on ``tip``.
    mujoco.mjv_connector(geom, arrow, ARROW_WIDTH_M, tail, 2.0 * tip - tail)
    scene.ngeom += 1


def _write(path: Path, fps: int, frames: Iterable[Frame]) -> int:
    """Encode into a new hidden sibling file and move it into place only once it is complete.

    The sibling has a random name and is created exclusively, and the move replaces a symlink at
    ``path`` instead of writing through it, so links planted in the output directory never
    redirect the video (``topple._atomic``).
    """
    suffix = path.suffix.lower()
    count = 0
    with (
        replacement(path, suffix=f".partial{suffix}") as partial,  # ffmpeg needs the suffix
        _open_writer(partial, suffix, fps) as writer,
    ):
        for frame in frames:
            writer.append_data(frame)
            count += 1
    return count


def _open_writer(path: Path, suffix: str, fps: int) -> Any:
    if suffix == ".mp4":
        return iio.get_writer(
            path,
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=MACRO_BLOCK,
            output_params=["-movflags", "+faststart"],
        )
    # Pillow plugin: duration is per frame in ms; loop=0 repeats forever.
    return iio.get_writer(path, format="pillow", mode="I", duration=1000 // fps, loop=0)
