"""Headless renderer creation with actionable errors."""

from __future__ import annotations

import os

import mujoco

from topple._gl import rendering_disabled

_HINT = (
    "install OSMesa (apt-get install libosmesa6) or set MUJOCO_GL=egl on Linux; "
    "use MUJOCO_GL=cgl on macOS"
)


class RenderingUnavailableError(RuntimeError):
    """Raised when MuJoCo cannot create an offscreen OpenGL context."""


def make_renderer(model: mujoco.MjModel, height: int, width: int) -> mujoco.Renderer:
    """Create an offscreen renderer or raise ``RenderingUnavailableError`` explaining why not."""
    if rendering_disabled(os.environ):
        raise RenderingUnavailableError(
            f"headless rendering unavailable: disabled by MUJOCO_GL={os.environ['MUJOCO_GL']}"
        )
    renderer_cls = getattr(mujoco, "Renderer", None)
    if renderer_cls is None:
        raise RenderingUnavailableError(
            f"headless rendering unavailable: mujoco could not load a GL context ({_HINT})"
        )
    try:
        renderer: mujoco.Renderer = renderer_cls(model, height=height, width=width)
    except (OSError, ImportError, RuntimeError, mujoco.FatalError) as exc:
        raise RenderingUnavailableError(f"headless rendering unavailable: {exc} ({_HINT})") from exc
    return renderer
