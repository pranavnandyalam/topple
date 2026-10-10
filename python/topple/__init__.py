"""Topple: CI for robots."""

from topple._gl import configure as _configure_gl

# MuJoCo reads MUJOCO_GL at import time; set it before any submodule imports mujoco.
_configure_gl()

__version__ = "0.1.0"
