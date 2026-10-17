"""Anchor "spin in place": the LQR's drive with a constant turn command of 0.30.

Definition: docs/design/phase-2.md §8.7 (``topple/controllers/anchors.py``).
Anchors bound the scorer from below; compare a submission against them with
``topple compare``.
"""

from __future__ import annotations

from topple.contract import TaskSpec
from topple.controllers.anchors import SpinInPlace, spin_in_place


def make_controller(spec: TaskSpec) -> SpinInPlace:
    return spin_in_place(spec)
