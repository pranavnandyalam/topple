"""The shipped LQR baseline as a submission: linear feedback on the full balancing state.

Its gains come from ``uv run python -m topple.design.lqr``, which identifies the robot from the
simulator and solves the Riccati equation. Re-run ``design_lqr(q=..., r=...)`` with your own
weights and pass ``LQRGains(...)`` to ``LQRController`` to try a different trade-off.
"""

from __future__ import annotations

from topple.contract import TaskSpec
from topple.controllers.lqr import LQRController


def make_controller(spec: TaskSpec) -> LQRController:
    return LQRController()
