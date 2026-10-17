"""Overview videos of waypoint episodes (docs/design/phase-2.md §8.9 item 8).

Every trajectory plays in **one** scene on one clock, seen from a fixed camera: the first one's
robot through ``update_scene``, each other's appended with ``mjv_addGeoms`` and drawn at
``GHOST_ALPHA``. The renderer draws the course itself (the scene holds no course geoms, §8.2):
numbered gates and a checkered dock as flat discs of ``gate_radius_m``, a coloured disc of
``DISC_R`` with a heading arrow under each robot, fading trails and the push arrows of
``video.py``. A heads-up line per robot shows its gate count and then its finish or fall time.

Timeline: ``e_i`` = t_finish if docked, t_fall if fell, else the episode length; the video ends
at ``min(T, max_i e_i + TAIL_S)``. A docked robot keeps playing; a fallen one plays its 1 s
epilogue and then holds with ``FELL at t``. The camera looks down at ``ELEVATION_DEG`` along the
course's principal axis and frames the bounding box of the course and every trajectory's path,
plus ``MARGIN_M``, clamped to ``FRAME_M``.

Every input is a trusted replay (a recorded or verified waypoint trajectory); labels are drawn
as printable ASCII.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from topple._atomic import replacement
from topple.challenge import load_challenge
from topple.envs.balance import _quat_pitch_yaw
from topple.evaluate import FALL_REASONS
from topple.kinds import WaypointReplay, waypoint_replay
from topple.rendering import make_renderer
from topple.robot import PUSH_SITE
from topple.traj import Trajectory
from topple.video import (
    MACRO_BLOCK,
    Frame,
    GifRung,
    GifTooLargeError,
    _draw_arrow,
    _entry_point,
    _font,
    _model,
    _open_writer,
    _push_at,
    _resized,
    _shows_epilogue,
    _stride,
    _video_suffix,
    _write,
    fit_text,
)

ELEVATION_DEG: Final = -57.5
"""Within §8.9's -55 to -60 degrees."""
MARGIN_M: Final = 0.5
FRAME_M: Final = (3.4, 5.0)
TAIL_S: Final = 1.5
DISC_R: Final = 0.08
DISC_Z: Final = 0.002
GHOST_ALPHA: Final = 0.5
TRAIL_S: Final = 3.0
TRAIL_EVERY: Final = 5
"""Control steps between trail dots."""
MAX_TRACKS: Final = 2
ROBOT_RGBA: Final = (
    np.array([0.15, 0.45, 0.95, 1.0], dtype=np.float32),  # first: the policy
    np.array([0.95, 0.6, 0.1, 1.0], dtype=np.float32),  # second: the driver
)
GATE_RGBA: Final = np.array([0.2, 0.75, 0.3, 0.55], dtype=np.float32)
GATE_DONE_RGBA: Final = np.array([0.5, 0.5, 0.5, 0.35], dtype=np.float32)
DOCK_RGBA: Final = (  # opaque black and white: the dock must read at GIF scale
    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
)
CHECKER: Final = 4
"""Quadrants of the checkered dock disc (alternating colours)."""
HUD_RGB: Final = ((40, 110, 240), (240, 150, 25))
HEADLIGHT: Final = (0.2, 0.2)
LIGHT_SCALE: Final = 0.45
"""Headlight ambient and diffuse: with the scene's lights made directional, an even floor seen
from above instead of the hot spot of the key spotlight."""


@dataclass(frozen=True)
class Track:
    """One robot on the shared clock: its qpos rows (epilogue included after a fall), the
    course replay, and the times that decide its playback (seconds)."""

    traj: Trajectory
    label: str
    rows: npt.NDArray[np.float64]
    replay: WaypointReplay
    control_dt: float
    t_finish: float | None
    t_fall: float | None
    end_s: float
    """``e_i``: t_finish if docked, t_fall if fell, else the episode length."""


def make_track(traj: Trajectory, label: str) -> Track:
    challenge_id = traj.meta.get("challenge")
    if not isinstance(challenge_id, str):
        raise ValueError("trajectory meta does not name its challenge")
    challenge = load_challenge(challenge_id)
    if challenge.kind != "waypoint":
        raise ValueError(f"a course video needs a waypoint trajectory, got {challenge_id}")
    termination = traj.meta.get("termination")
    reason = termination.get("reason") if isinstance(termination, dict) else None
    if not isinstance(reason, str):
        raise ValueError("trajectory meta records no termination reason")
    replay = waypoint_replay(traj, challenge, reason)
    control_dt = challenge.config.control_dt
    epilogue = _shows_epilogue(traj.meta)
    rows = np.concatenate([traj.qpos, traj.epilogue_qpos]) if epilogue else traj.qpos
    t_fall = (traj.qpos.shape[0] - 1) * control_dt if reason in FALL_REASONS else None
    t_finish = replay.events.t_finish_s
    end = t_finish if t_finish is not None else t_fall
    return Track(
        traj,
        label,
        rows,
        replay,
        control_dt,
        t_finish,
        t_fall,
        end if end is not None else challenge.config.episode_seconds,
    )


def end_time(tracks: Sequence[Track]) -> float:
    """§8.9: ``min(T, max_i e_i + TAIL_S)``."""
    episode_s = load_challenge(str(tracks[0].traj.meta["challenge"])).config.episode_seconds
    return min(episode_s, max(t.end_s for t in tracks) + TAIL_S)


# --- geometry -------------------------------------------------------------------------------------


def world(replay: WaypointReplay, p: tuple[float, float]) -> tuple[float, float]:
    """A start-frame point in the world (the inverse of ``waypoint.to_course_frame``)."""
    c, s = math.cos(replay.heading), math.sin(replay.heading)
    return (replay.origin[0] + c * p[0] - s * p[1], replay.origin[1] + s * p[0] + c * p[1])


def camera(tracks: Sequence[Track], fovy_deg: float) -> mujoco.MjvCamera:
    """A fixed free camera at ``ELEVATION_DEG``, its azimuth along the course's principal axis,
    framing the course and every path plus ``MARGIN_M`` (side clamped to ``FRAME_M``)."""
    first = tracks[0].replay
    course = np.array([world(first, p) for p in ((0.0, 0.0), *first.points)])
    paths = [
        np.array([world(t.replay, (float(x), float(y))) for x, y in t.replay.rows]) for t in tracks
    ]
    xy = np.concatenate([course, *paths])
    xy = xy[np.isfinite(xy).all(axis=1)]
    lo, hi = xy.min(axis=0) - MARGIN_M, xy.max(axis=0) + MARGIN_M
    side = float(np.clip(max(hi - lo), *FRAME_M))
    centered = course - course.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = math.degrees(math.atan2(vt[0][1], vt[0][0]))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [*((lo + hi) / 2), 0.0]
    cam.azimuth = axis  # §8.9: the camera looks along the course's principal axis
    cam.elevation = ELEVATION_DEG
    cam.distance = side / 2 / math.tan(math.radians(fovy_deg) / 2)
    return cam


def project(
    cam: mujoco.MjvCamera, fovy_deg: float, size: int, xyz: Sequence[float]
) -> tuple[float, float] | None:
    """Pixel (x right, y down) of world point ``xyz`` in a square ``size`` frame seen by the
    free camera ``cam`` (MuJoCo's ``mjv_cameraFrame`` convention); None behind the camera."""
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = np.asarray(xyz, dtype=np.float64) - (np.asarray(cam.lookat) - cam.distance * fwd)
    depth = float(rel @ fwd)
    if depth <= 0:
        return None
    f = size / 2 / math.tan(math.radians(fovy_deg) / 2)
    return (size / 2 + f * float(rel @ right) / depth, size / 2 - f * float(rel @ up) / depth)


def _add(
    scene: mujoco.MjvScene,
    kind: int,
    size: Sequence[float],
    pos: Sequence[float],
    rgba: npt.NDArray[np.float32],
    mat: npt.NDArray[np.float64] | None = None,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    rot = np.eye(3).ravel() if mat is None else mat.ravel()
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        kind,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        rot,
        rgba,
    )
    scene.ngeom += 1


def _disc(
    scene: mujoco.MjvScene,
    xy: tuple[float, float],
    r: float,
    rgba: npt.NDArray[np.float32],
    z: float = DISC_Z,
) -> None:
    _add(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, (r, 0.001, 0.0), (xy[0], xy[1], z), rgba)


def _course(scene: mujoco.MjvScene, replay: WaypointReplay, done: int, r: float) -> None:
    """Gates (dimmed once the first robot has passed them) and the checkered dock."""
    *gates, dock = (world(replay, p) for p in replay.points)
    for i, g in enumerate(gates):
        _disc(scene, g, r, GATE_DONE_RGBA if i < done else GATE_RGBA)
    _disc(scene, dock, r, DOCK_RGBA[0])
    for k in range(0, CHECKER, 2):  # light squares in alternate quadrants of a dark disc
        a = (k + 0.5) * 2 * math.pi / CHECKER
        centre = (dock[0] + 0.5 * r * math.cos(a), dock[1] + 0.5 * r * math.sin(a), 2 * DISC_Z)
        half = (0.35 * r, 0.35 * r, 0.001)
        _add(scene, mujoco.mjtGeom.mjGEOM_BOX, half, centre, DOCK_RGBA[1])


def _robot_marks(
    scene: mujoco.MjvScene,
    xy: tuple[float, float],
    yaw: float,
    rgba: npt.NDArray[np.float32],
    trail: npt.NDArray[np.float64],
) -> None:
    _disc(scene, xy, DISC_R, rgba, z=2 * DISC_Z)
    tip = (xy[0] + 1.6 * DISC_R * math.cos(yaw), xy[1] + 1.6 * DISC_R * math.sin(yaw))
    geom_count = scene.ngeom
    if geom_count < scene.maxgeom:
        arrow = mujoco.mjtGeom.mjGEOM_ARROW
        g = scene.geoms[geom_count]
        mujoco.mjv_initGeom(g, arrow, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
        mujoco.mjv_connector(
            g,
            arrow,
            0.012,
            np.array([xy[0], xy[1], 3 * DISC_Z]),
            np.array([2 * tip[0] - xy[0], 2 * tip[1] - xy[1], 3 * DISC_Z]),
        )
        scene.ngeom += 1
    n = len(trail)
    for k, (x, y) in enumerate(trail):
        fade = rgba.copy()
        fade[3] = 0.15 + 0.5 * (k + 1) / n
        _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.012, 0.0, 0.0), (x, y, 0.01), fade)


# --- frames ---------------------------------------------------------------------------------------


def _hud(t: Track, step: int) -> str:
    """The robot's line: gate count, then its finish or fall time once it has one."""
    reached = [s for s in t.replay.events.reached_step if s is not None and s <= step]
    n = len(t.replay.points)
    now = step * t.control_dt
    gates = f"gate {min(len(reached), n - 1)}/{n - 1}"
    if t.t_finish is not None and now >= t.t_finish:
        return f"{t.label}: docked {t.t_finish:.2f} s"
    if t.t_fall is not None and now >= t.t_fall:
        return f"{t.label}: FELL at {t.t_fall:.2f} s, {gates}"
    return f"{t.label}: {gates}"


def _overlay(
    frame: Frame, tracks: Sequence[Track], step: int, numbers: Sequence[tuple[float, float]]
) -> Frame:
    """The HUD and the gate numbers (``numbers``: each gate's pixel)."""
    height, width = frame.shape[:2]
    px = max(10, height // 24)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = _font(px)  # the bundled font, the same on every platform
    for i, (u, v) in enumerate(numbers, 1):
        draw.text(
            (u + 0.7 * px, v - 0.7 * px),  # beside the disc, so the robot on it stays visible
            str(i),
            font=font,
            fill=(255, 255, 255),
            anchor="mm",
            stroke_width=max(1, px // 8),
            stroke_fill=(0, 0, 0),
        )
    y = px // 2
    draw.rectangle((0, 0, width, y + (len(tracks) + 1) * (px + 4)), fill=(0, 0, 0))
    draw.text(
        (px // 2, y), f"t = {step * tracks[0].control_dt:5.2f} s", font=font, fill=(255, 255, 255)
    )
    for i, t in enumerate(tracks):
        y += px + 4
        draw.text((px // 2, y), fit_text(_hud(t, step), width - px, px), font=font, fill=HUD_RGB[i])
    out: Frame = np.array(image, dtype=np.uint8)
    return out


def course_frames(
    trajs: Sequence[Trajectory], labels: Sequence[str], *, fps: int, size: int
) -> Generator[Frame, None, None]:
    """Square ``size`` RGB frames of ``trajs`` in one scene (the first drawn solid, the others
    at ``GHOST_ALPHA``) at ``fps``. Validated before the first frame renders."""
    if not 1 <= len(trajs) <= MAX_TRACKS or len(labels) != len(trajs):
        raise ValueError(f"a course video shows 1 to {MAX_TRACKS} labelled trajectories")
    if size <= 0 or size % MACRO_BLOCK:
        raise ValueError(f"size must be a positive multiple of {MACRO_BLOCK}, got {size}")
    tracks = [make_track(t, lbl) for t, lbl in zip(trajs, labels, strict=True)]
    if len({_canon(t.traj.meta["scenario"]) for t in tracks}) != 1:
        raise ValueError("the trajectories are not on one scenario (seed and preset)")
    model, control_dt = _model(trajs[0])
    if size > min(model.vis.global_.offwidth, model.vis.global_.offheight):
        raise ValueError(f"size {size} exceeds the offscreen framebuffer")
    stride = _stride(fps, control_dt)
    n_steps = round(end_time(tracks) / control_dt)
    return _frames(model, tracks, stride, n_steps, size)


def _canon(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _frames(
    model: mujoco.MjModel, tracks: Sequence[Track], stride: int, n_steps: int, size: int
) -> Generator[Frame, None, None]:
    datas = [mujoco.MjData(model) for _ in tracks]
    model.light_type[:] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    model.light_diffuse[:] *= LIGHT_SCALE
    model.vis.headlight.ambient[:] = HEADLIGHT[0]
    model.vis.headlight.diffuse[:] = HEADLIGHT[1]
    fovy = float(model.vis.global_.fovy)
    cam = camera(tracks, fovy)
    lead = tracks[0]
    numbers = [
        px
        for g in lead.replay.points[:-1]
        if (px := project(cam, fovy, size, (*world(lead.replay, g), 0.0))) is not None
    ]
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    push_site = model.site(PUSH_SITE).id
    r = load_challenge(str(tracks[0].traj.meta["challenge"])).config.course.gate_radius_m
    root = int(tracks[0].traj.meta["joint_qpos"]["root"])
    with make_renderer(model, height=size, width=size) as renderer:
        for step in range(0, n_steps + 1, stride):
            for t, d in zip(tracks, datas, strict=True):
                row = t.rows[min(step, len(t.rows) - 1)]
                if np.isfinite(row).all():
                    d.qpos[:] = row
                mujoco.mj_forward(model, d)
            renderer.update_scene(datas[0], camera=cam)
            scene = renderer.scene
            for d in datas[1:]:
                first = scene.ngeom
                mujoco.mjv_addGeoms(model, d, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
                for g in range(first, scene.ngeom):
                    scene.geoms[g].rgba[3] *= GHOST_ALPHA
            done = sum(s is not None and s <= step for s in lead.replay.events.reached_step)
            _course(scene, lead.replay, done, r)
            for i, (t, d) in enumerate(zip(tracks, datas, strict=True)):
                k = min(step, len(t.rows) - 1)
                q = t.rows[k]
                _, yaw = _quat_pitch_yaw(q[root + 3 : root + 7])
                lo = max(0, k - round(TRAIL_S / t.control_dt))
                lo = -(-lo // TRAIL_EVERY) * TRAIL_EVERY  # dots on a fixed grid, so they stay put
                trail = t.rows[lo:k:TRAIL_EVERY, root : root + 2]
                rgba = ROBOT_RGBA[i].copy()
                if i:
                    rgba[3] *= GHOST_ALPHA
                _robot_marks(scene, (float(q[root]), float(q[root + 1])), yaw, rgba, trail)
                push = (
                    _push_at(t.traj.push_force, step, stride) if step < len(t.traj.qpos) else None
                )
                if push is not None:
                    _draw_arrow(
                        scene, _entry_point(model, d, d.site_xpos[push_site].copy(), push), push
                    )
            yield _overlay(renderer.render(), tracks, step, numbers)


def render_course(
    trajs: Sequence[Trajectory],
    labels: Sequence[str],
    path: Path,
    *,
    fps: int = 50,
    size: int = 720,
) -> int:
    """Write the course video of ``trajs`` to ``path`` (.mp4 or .gif); the frame count."""
    path = Path(path)
    _video_suffix(path, fps)
    frames = course_frames(trajs, labels, fps=fps, size=size)
    try:
        first = next(frames)
        path.parent.mkdir(parents=True, exist_ok=True)
        return _write(path, fps, itertools.chain([first], frames))
    finally:
        frames.close()


def render_course_gif(
    trajs: Sequence[Trajectory],
    labels: Sequence[str],
    out: Path,
    rungs: Sequence[GifRung],
    *,
    max_bytes: int,
) -> GifRung:
    """The course GIF at the first rung of ``rungs`` within ``max_bytes`` (the §5.2 ladder);
    ``GifTooLargeError`` if none is, leaving ``out`` as it was."""
    path = Path(out)
    size = 0
    for rung in rungs:
        _video_suffix(path, rung.fps)
        render = MACRO_BLOCK * math.ceil(rung.width / MACRO_BLOCK)
        frames = course_frames(trajs, labels, fps=rung.fps, size=render)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with replacement(path, suffix=".partial.gif") as partial:
                with _open_writer(partial, ".gif", rung.fps) as writer:
                    for frame in frames:
                        writer.append_data(_resized(frame, (rung.width, rung.width)))
                size = partial.stat().st_size
                if size > max_bytes:
                    raise _RungOverCapError
        except _RungOverCapError:
            continue
        finally:
            frames.close()
        return rung
    raise GifTooLargeError(f"the last course GIF rung is {size} bytes, over {max_bytes}")


class _RungOverCapError(Exception):
    """A rung over its cap (the partial file is removed by ``replacement``)."""
