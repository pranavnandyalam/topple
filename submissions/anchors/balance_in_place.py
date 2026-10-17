"""Anchor "balance in place": the shipped LQR, never leaving the start.

Definition: docs/design/phase-2.md §8.7 (``topple/controllers/anchors.py``).
Anchors bound the scorer from below; compare a submission against them with
``topple compare``.
"""

from __future__ import annotations

from topple.contract import TaskSpec
from topple.controllers.anchors import BalanceInPlace, balance_in_place


def make_controller(spec: TaskSpec) -> BalanceInPlace:
    return balance_in_place(spec)
