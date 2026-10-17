"""Anchor "reckless": the shipped driver law at up to 0.60 m/s.

Definition: docs/design/phase-2.md §8.7 (``topple/controllers/anchors.py``).
Anchors bound the scorer from below; compare a submission against them with
``topple compare``.
"""

from __future__ import annotations

from topple.contract import TaskSpec
from topple.controllers.anchors import reckless
from topple.controllers.waypoint_pid import Driver


def make_controller(spec: TaskSpec) -> Driver:
    return reckless(spec)
