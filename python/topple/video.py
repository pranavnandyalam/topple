"""Replay videos: render a stored trajectory to MP4 or GIF.

Playback only, never re-simulation: each frame poses the challenge's robot at a recorded state
(``qpos`` rows, then ``epilogue_qpos`` when the epilogue rule of contract §11 calls for one) with
``mj_forward`` and draws it from the chassis-tracking camera. A trajectory recorded with another
robot (``meta.robot_hash``) is refused rather than drawn as the wrong robot. Rows are recorded at
the control rate and subsampled to ``fps``. An active push appears as a red arrow along the
world-frame force, its tip touching the robot where the force's line of action through the
``push_point`` site enters the body, its length proportional to the force.

Side by side (docs/design/phase-2.md §5.2 item 1): square panes, one per trajectory, play on one
clock. A pane whose episode fell or errored plays its epilogue, then holds its last frame
stamped ``FELL at 4.31 s`` (or ``ERROR at``); the video ends when the longest pane does, or
``STAMP_HOLD_S`` after the last stamped pane does, whichever is later. Each pane carries its
label. Every size rendered is a multiple of 16 px (``MACRO_BLOCK``) within the scene's offscreen
buffer, and every fps divides the control rate; a GIF ladder (``render_gif_ladder``) tries
smaller sizes and frame rates in order, resampling the composed frame for a width that no
rendered size gives.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import imageio.v2 as iio
import mujoco
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from topple._atomic import replacement
from topple.challenge import load_challenge
from topple.evaluate import FALL_REASONS, wants_epilogue
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

MAX_PANES: Final = 4
STAMP_HOLD_S: Final = 1.0
"""How long the last stamped pane holds its stamp before the video ends."""
TEXT_RGB: Final = (255, 255, 255)
LABEL_BACKDROP_RGB: Final = (0, 0, 0)
STAMP_BACKDROP_RGB: Final = (170, 20, 20)
ELLIPSIS: Final = "..."


@dataclass(frozen=True)
class _Playback:
    model: mujoco.MjModel
    rows: npt.NDArray[np.float64]  # one qpos row per frame
    pushes: tuple[npt.NDArray[np.float64] | None, ...]  # push force to draw on each frame
    width: int
    height: int
    camera: str
    control_dt: float


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
    _video_suffix(path, fps)
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
    return _Playback(model, rows[::stride], pushes, width, height, camera, control_dt)


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


# --- side by side -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PaneTimeline:
    """One pane on the shared clock: the ``frames`` it plays (its rows, epilogue included, at
    the stride), then its hold; ``stamp`` is the text its hold frames carry (``FELL at 4.31 s``,
    ``ERROR at 0.30 s``), or ``None`` for an episode that neither fell nor errored."""

    frames: int
    stamp: str | None


def side_by_side_timeline(
    trajs: Sequence[Trajectory], fps: int
) -> tuple[tuple[PaneTimeline, ...], int]:
    """Each pane's timeline at ``fps`` and the video's frame count: the longest pane, or one
    ``STAMP_HOLD_S`` hold after the end of the last stamped pane, whichever is later."""
    if not trajs:
        raise ValueError("a side-by-side video has at least one pane")
    return _timeline(trajs, [_model(t)[1] for t in trajs], fps)


def _timeline(
    trajs: Sequence[Trajectory], control_dts: Sequence[float], fps: int
) -> tuple[tuple[PaneTimeline, ...], int]:
    panes = tuple(_pane_timeline(t, dt, fps) for t, dt in zip(trajs, control_dts, strict=True))
    hold = round(STAMP_HOLD_S * fps)
    total = max(p.frames + (hold if p.stamp is not None else 0) for p in panes)
    return panes, total


def _pane_timeline(traj: Trajectory, control_dt: float, fps: int) -> PaneTimeline:
    stride = _stride(fps, control_dt)
    rows = traj.qpos.shape[0] + (traj.epilogue_qpos.shape[0] if _shows_epilogue(traj.meta) else 0)
    return PaneTimeline(frames=math.ceil(rows / stride), stamp=_stamp(traj, control_dt))


def _stamp(traj: Trajectory, control_dt: float) -> str | None:
    """``ERROR at t`` for an episode with an error (``sim_unstable`` included), ``FELL at t`` for
    a fall, else ``None``; t is when the episode ended (its T steps, from the arrays). The meta
    is untrusted (contract §14): a reason that is not a string names no fall."""
    t = traj.steps * control_dt
    if traj.meta.get("error") is not None:
        return f"ERROR at {t:.2f} s"
    termination = traj.meta.get("termination")
    reason = termination.get("reason") if isinstance(termination, dict) else None
    if isinstance(reason, str) and reason in FALL_REASONS:
        return f"FELL at {t:.2f} s"
    return None


def render_side_by_side(
    trajs: Sequence[Trajectory],
    labels: Sequence[str],
    out: Path,
    *,
    pane: int = 480,
    fps: int = 50,
) -> int:
    """Write ``trajs`` side by side (square ``pane`` px panes, left to right, each with its
    label) to ``out`` (``.mp4`` or ``.gif``) and return the number of frames written.

    ``out`` appears only once every frame is encoded, never through a symlink planted there.
    Bad arguments (``ValueError``: pane or label counts, a size that is no multiple of 16 or
    exceeds the offscreen buffer, an fps that does not divide the control rate) and a missing
    GL backend raise before any file or directory is created.
    """
    path = Path(out)
    _video_suffix(path, fps)
    frames = side_by_side_frames(trajs, labels, pane=pane, fps=fps)
    try:
        first = next(frames)  # opens the renderers: a missing GL backend fails before any I/O
        path.parent.mkdir(parents=True, exist_ok=True)
        return _write(path, fps, itertools.chain([first], frames))
    finally:
        frames.close()


def side_by_side_frames(
    trajs: Sequence[Trajectory], labels: Sequence[str], *, pane: int = 480, fps: int = 50
) -> Generator[Frame, None, None]:
    """The composed frames (``pane`` x ``len(trajs) * pane`` x 3, uint8) of
    ``render_side_by_side``. Arguments are validated at once; the renderers open at the first
    frame and close when the generator is exhausted or closed."""
    plays = _side_by_side_plays(trajs, labels, pane, fps)
    panes, total = _timeline(trajs, [play.control_dt for play in plays], fps)
    return _composed(plays, tuple(labels), panes, total)


def _side_by_side_plays(
    trajs: Sequence[Trajectory], labels: Sequence[str], pane: int, fps: int
) -> list[_Playback]:
    if not 1 <= len(trajs) <= MAX_PANES:
        raise ValueError(f"a side-by-side video has 1 to {MAX_PANES} panes, got {len(trajs)}")
    if len(labels) != len(trajs):
        raise ValueError(f"one label per pane: {len(trajs)} panes, {len(labels)} labels")
    return [_playback(t, fps, (pane, pane), TRACK_CAMERA, True, True) for t in trajs]


def _composed(
    plays: Sequence[_Playback],
    labels: Sequence[str],
    panes: Sequence[PaneTimeline],
    total: int,
) -> Generator[Frame, None, None]:
    """Each pane plays its frames, then shows its last one: stamped from then on if it has a
    stamp. Its frame is annotated once per change, so a hold costs nothing."""
    sources = [_frames(play) for play in plays]
    raw: list[Frame | None] = [None] * len(plays)
    shown: list[Frame | None] = [None] * len(plays)
    try:
        for f in range(total):
            for i, (source, label, timeline) in enumerate(zip(sources, labels, panes, strict=True)):
                if f < timeline.frames:
                    raw[i] = next(source)
                    shown[i] = annotate(_present(raw[i]), label, stamp=None)
                elif f == timeline.frames and timeline.stamp is not None:
                    shown[i] = annotate(_present(raw[i]), label, stamp=timeline.stamp)
            yield np.concatenate([_present(frame) for frame in shown], axis=1)
    finally:
        for source in sources:
            source.close()


def _present(frame: Frame | None) -> Frame:
    if frame is None:  # every pane plays at least its first row (qpos has T + 1 >= 1 rows)
        raise RuntimeError("a pane has no frame to show")
    return frame


def _video_suffix(path: Path, fps: int) -> str:
    """``path``'s video suffix (``.mp4`` or ``.gif``), else ``ValueError``; a GIF's fps must
    keep frame delays that browsers honour."""
    suffix = path.suffix.lower()
    if suffix not in SUFFIXES:
        raise ValueError(f"unsupported video suffix {path.suffix!r}; use one of {SUFFIXES}")
    if suffix == ".gif" and fps > GIF_MAX_FPS:
        raise ValueError(
            f"GIF frame delays below {1000 // GIF_MAX_FPS} ms play at 10 fps in browsers; "
            f"use fps <= {GIF_MAX_FPS} for .gif"
        )
    return suffix


# --- text on frames ---------------------------------------------------------------------------

MAX_LABEL_CHARS: Final = 200


@lru_cache(maxsize=16)
def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Pillow's bundled font at ``px`` (its bitmap font where FreeType is missing)."""
    return ImageFont.load_default(size=px)


def _font_px(height: int) -> int:
    return max(10, height // 16)


def printable_ascii(text: str) -> str:
    """``text`` (at most ``MAX_LABEL_CHARS``) with every character outside printable ASCII
    shown as ``?``: a label is untrusted text, drawn the same on every platform."""
    return "".join(ch if " " <= ch <= "~" else "?" for ch in text[:MAX_LABEL_CHARS])


def text_width(text: str, px: int) -> float:
    """The width in pixels of ``text`` drawn at ``px``."""
    return float(_font(px).getlength(text))


@lru_cache(maxsize=64)
def fit_text(text: str, max_width: float, px: int) -> str:
    """``printable_ascii(text)``, cut and ended with ``...`` if it is wider than
    ``max_width`` pixels at ``px``; empty if not even that fits. Cached: a pane's label is
    fitted once, not on every frame (each fit measures up to one candidate per character)."""
    shown = printable_ascii(text)
    if text_width(shown, px) <= max_width:
        return shown
    for n in range(len(shown) - 1, -1, -1):
        candidate = shown[:n].rstrip() + ELLIPSIS
        if text_width(candidate, px) <= max_width:
            return candidate
    return ""


def annotate(frame: Frame, label: str, *, stamp: str | None) -> Frame:
    """A copy of ``frame`` with ``label`` at the top left (white on black) and ``stamp``, if
    any, centred at the bottom (white on red), each trimmed to fit the frame."""
    height, width = frame.shape[:2]
    px = _font_px(height)
    pad = max(2, px // 4)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text = fit_text(label, width - 4 * pad, px)
    if text:
        _boxed(draw, (2 * pad, pad), text, px, pad, LABEL_BACKDROP_RGB)
    if stamp is not None:
        text = fit_text(stamp, width - 4 * pad, px)
        if text:
            x = max(2 * pad, (width - text_width(text, px)) / 2)
            _boxed(draw, (x, height - 3 * pad - px), text, px, pad, STAMP_BACKDROP_RGB)
    out: Frame = np.array(image, dtype=np.uint8)
    return out


def _boxed(
    draw: ImageDraw.ImageDraw,
    at: tuple[float, float],
    text: str,
    px: int,
    pad: int,
    backdrop: tuple[int, int, int],
) -> None:
    font = _font(px)
    left, top, right, bottom = draw.textbbox(at, text, font=font)
    draw.rectangle((left - pad, top - pad, right + pad, bottom + pad), fill=backdrop)
    draw.text(at, text, font=font, fill=TEXT_RGB)


# --- GIF ladders (§5.2) -----------------------------------------------------------------------


@dataclass(frozen=True)
class GifRung:
    """One step of a GIF ladder: the GIF's ``width`` in px (all panes together; panes are
    square) at ``fps``."""

    width: int
    fps: int


def gif_ladder(
    width: int, fps_steps: Sequence[int], shrink_fps: int, *, shrinks: int = 3, factor: float = 0.9
) -> tuple[GifRung, ...]:
    """``width`` at each of ``fps_steps``, then ``width · factor^k`` (rounded) for k = 1 ..
    ``shrinks`` at ``shrink_fps`` (§5.2)."""
    steps = [GifRung(width, f) for f in fps_steps]
    return (*steps, *(GifRung(round(width * factor**k), shrink_fps) for k in range(1, shrinks + 1)))


HERO_LADDER: Final = gif_ladder(480, (20, 10, 5), 5)  # 480x240: 432, 389, 350 px at 5 fps
GALLERY_LADDER: Final = gif_ladder(320, (10, 5, 4), 4)
WAYPOINT_LADDER: Final = gif_ladder(480, (10, 5, 4), 4)


class GifTooLargeError(ValueError):
    """Even the last rung of a GIF ladder is over its size cap."""


class _OverCapError(Exception):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.size = size


def render_gif_ladder(
    trajs: Sequence[Trajectory],
    labels: Sequence[str],
    out: Path,
    rungs: Sequence[GifRung],
    *,
    max_bytes: int,
) -> GifRung:
    """Write the side-by-side GIF of ``trajs`` at the first rung of ``rungs`` whose file is at
    most ``max_bytes``, and return that rung.

    Every rung is checked before anything renders. Each is rendered at its own fps (its own
    stride) with square panes of the smallest multiple of 16 px at least ``width / panes``;
    the composed frame is resampled to ``width`` when no rendered size gives it. ``out``
    appears only for the rung used; ``GifTooLargeError`` if even the last rung is over the
    cap, leaving ``out`` as it was.
    """
    path = Path(out)
    if path.suffix.lower() != ".gif":
        raise ValueError(f"a GIF ladder writes a .gif file, got {path.suffix!r}")
    if not rungs:
        raise ValueError("a GIF ladder needs at least one rung")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError(f"max_bytes must be a positive int, got {max_bytes!r}")
    plans = []
    for rung in rungs:
        pane = _rung_pane(rung, len(trajs))
        _video_suffix(path, rung.fps)
        _side_by_side_plays(trajs, labels, pane, rung.fps)  # sizes, fps, counts: all up front
        plans.append((rung, pane))
    size = 0
    for rung, pane in plans:
        try:
            _write_rung(trajs, labels, path, rung, pane, max_bytes)
        except _OverCapError as exc:
            size = exc.size
            continue
        return rung
    last = plans[-1][0]
    raise GifTooLargeError(
        f"the last GIF rung ({last.width} px at {last.fps} fps) is {size} bytes, "
        f"over the {max_bytes}-byte cap"
    )


def _rung_pane(rung: GifRung, panes: int) -> int:
    """The rendered pane size for ``rung``: the smallest multiple of 16 at least its share."""
    if panes < 1 or rung.width < MACRO_BLOCK * panes:
        raise ValueError(
            f"a rung width must give each of the {panes} panes at least {MACRO_BLOCK} px, "
            f"got {rung.width}"
        )
    return MACRO_BLOCK * math.ceil(rung.width / panes / MACRO_BLOCK)


def _write_rung(
    trajs: Sequence[Trajectory],
    labels: Sequence[str],
    path: Path,
    rung: GifRung,
    pane: int,
    max_bytes: int,
) -> None:
    """Encode ``rung`` into a new sibling file and move it over ``path`` if it fits the cap;
    ``_OverCapError`` (``path`` untouched, the sibling removed) if not."""
    size = (rung.width, round(rung.width / len(trajs)))
    frames = side_by_side_frames(trajs, labels, pane=pane, fps=rung.fps)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with replacement(path, suffix=".partial.gif") as partial:
            with _open_writer(partial, ".gif", rung.fps) as writer:
                for frame in frames:
                    writer.append_data(_resized(frame, size))
            written = partial.stat().st_size
            if written > max_bytes:
                raise _OverCapError(written)
    finally:
        frames.close()


def _resized(frame: Frame, size: tuple[int, int]) -> Frame:
    """``frame`` resampled to ``size`` (width, height), or itself if it already has it."""
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    image = Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)
    out: Frame = np.array(image, dtype=np.uint8)
    return out
