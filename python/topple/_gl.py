"""Choose a headless OpenGL backend for MuJoCo.

MuJoCo reads ``MUJOCO_GL`` once, when ``mujoco`` is first imported, so ``configure()`` runs from
``topple/__init__.py`` before any module imports ``mujoco``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

_DISABLED = frozenset({"disable", "disabled", "off"})
_PLATFORM_BACKENDS: Mapping[str, frozenset[str]] = {
    "darwin": frozenset({"cgl", "glfw"}),
    "linux": frozenset({"egl", "osmesa", "glfw"}),
    "win32": frozenset({"wgl", "glfw"}),
}
_DEFAULTS: Mapping[str, str] = {"darwin": "cgl", "linux": "osmesa"}


def default_gl_backend(platform: str) -> str:
    """Headless-friendly default: CGL on macOS, OSMesa on Linux (works on CPU-only CI)."""
    return _DEFAULTS.get(platform, "glfw")


def resolve_gl_backend(env: Mapping[str, str], platform: str) -> str:
    """Return the backend to use, honoring a valid ``MUJOCO_GL`` already in ``env``."""
    requested = env.get("MUJOCO_GL", "").strip().lower()
    if not requested:
        return default_gl_backend(platform)
    if requested in _DISABLED:
        return requested
    allowed = _PLATFORM_BACKENDS.get(platform, frozenset({"glfw"}))
    if requested not in allowed:
        raise ValueError(
            f"MUJOCO_GL={requested!r} is not available on {platform}; "
            f"use one of {sorted(allowed | _DISABLED)}"
        )
    return requested


def configure() -> str:
    """Set ``MUJOCO_GL`` for this process and return the chosen backend."""
    backend = resolve_gl_backend(os.environ, sys.platform)
    os.environ["MUJOCO_GL"] = backend
    return backend
