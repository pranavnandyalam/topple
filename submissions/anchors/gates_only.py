"""Anchor "gates-only": the driver law aimed 0.30 m past the dock along the last leg.

It docks only incidentally (a push can stop it inside the dock).

Definition: docs/design/phase-2.md §8.7 (``topple/controllers/anchors.py``).
Anchors bound the scorer from below; compare a submission against them with
``topple compare``.
"""

from __future__ import annotations

from topple.contract import TaskSpec
from topple.controllers.anchors import GatesOnly, gates_only


def make_controller(spec: TaskSpec) -> GatesOnly:
    return gates_only(spec)
