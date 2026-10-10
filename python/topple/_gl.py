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
from collections.abc import Mapping, MutableMapping

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


def _mujoco_default(platform: str) -> str:
    """What MuJoCo itself picks when MUJOCO_GL is unset."""
    return "cgl" if platform == "darwin" else "glfw"


def configure(
    env: MutableMapping[str, str] | None = None,
    platform: str | None = None,
    modules: Mapping[str, object] | None = None,
) -> str:
    """Set ``MUJOCO_GL`` for this process and return the chosen backend.

    Warns only when ``mujoco`` was imported first and has already picked a different GL backend
    (in practice: headless Linux, where MuJoCo defaults to GLFW but topple wants OSMesa).
    """
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    modules = sys.modules if modules is None else modules
    backend = resolve_gl_backend(env, platform)
    effective = _requested(env) or _mujoco_default(platform)
    if "mujoco" in modules and effective != backend:
        warnings.warn(
            f"mujoco was imported before topple and is using MUJOCO_GL={effective!r}; "
            f"topple's choice {backend!r} cannot take effect in this process. "
            "Import topple first or set MUJOCO_GL yourself.",
            RuntimeWarning,
            stacklevel=3,
        )
    env["MUJOCO_GL"] = backend
    return backend
