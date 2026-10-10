"""Choose a headless OpenGL backend for MuJoCo.

MuJoCo reads ``MUJOCO_GL`` once, when ``mujoco`` is first imported, so ``configure()`` runs from
``topple/__init__.py`` before any module imports ``mujoco``. Accepted values mirror MuJoCo 3.15's
own check (``mujoco/rendering/classic/gl_context.py``); only an unset/blank value is replaced by a
headless-friendly default.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Mapping

DISABLED_VALUES = frozenset({"disable", "disabled", "off", "false", "0"})
_GENERIC_VALUES = frozenset({"enable", "enabled", "on", "true", "1", "glfw"})
_PLATFORM_VALUES: Mapping[str, frozenset[str]] = {
    "darwin": frozenset({"cgl"}),
    "linux": frozenset({"glx", "egl", "osmesa"}),
    "win32": frozenset({"wgl"}),
}


def _requested(env: Mapping[str, str]) -> str:
    return env.get("MUJOCO_GL", "").strip().lower()


def rendering_disabled(env: Mapping[str, str]) -> bool:
    """True when ``MUJOCO_GL`` explicitly turns rendering off."""
    return _requested(env) in DISABLED_VALUES


def default_gl_backend(platform: str, env: Mapping[str, str]) -> str:
    """CGL on macOS; on Linux GLFW with a display, OSMesa when headless (CPU-only CI)."""
    if platform == "darwin":
        return "cgl"
    if platform == "linux":
        has_display = bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
        return "glfw" if has_display else "osmesa"
    return "glfw"


def resolve_gl_backend(env: Mapping[str, str], platform: str) -> str:
    """Return the backend to use, honoring any ``MUJOCO_GL`` value MuJoCo itself accepts."""
    requested = _requested(env)
    if not requested:
        return default_gl_backend(platform, env)
    allowed = DISABLED_VALUES | _GENERIC_VALUES | _PLATFORM_VALUES.get(platform, frozenset())
    if requested not in allowed:
        raise ValueError(
            f"MUJOCO_GL={requested!r} is not available on {platform}; use one of {sorted(allowed)}"
        )
    return requested


def configure() -> str:
    """Set ``MUJOCO_GL`` for this process and return the chosen backend.

    Warns when ``mujoco`` was imported first with a different setting: MuJoCo has already
    picked its GL context, so the new value cannot take effect in this process.
    """
    backend = resolve_gl_backend(os.environ, sys.platform)
    if "mujoco" in sys.modules and _requested(os.environ) != backend:
        warnings.warn(
            f"mujoco was imported before topple; MUJOCO_GL={backend!r} cannot take effect in "
            "this process. Import topple first or set MUJOCO_GL yourself.",
            RuntimeWarning,
            stacklevel=2,
        )
    os.environ["MUJOCO_GL"] = backend
    return backend
