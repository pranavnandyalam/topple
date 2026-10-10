"""Topple: CI for robots."""

from importlib.metadata import version as _dist_version

from topple._gl import configure as _configure_gl

# MuJoCo reads MUJOCO_GL at import time; set it before any submodule imports mujoco.
_configure_gl()

__version__ = _dist_version("topple-sim")
