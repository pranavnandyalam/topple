"""The shipped waypoint driver as a submission for ``waypoint-v1``.

It balances with the shipped LQR gains while steering toward the current gate: a speed reference
ramps toward a distance-proportional target (at most 0.30 m/s, 0.8 m/s^2), it backs up toward a
gate behind it, and turns toward the gate's bearing. Copy this file and change ``v_max``,
``a_max`` or ``k_v``, or replace the law. See ``topple/controllers/waypoint_pid.py``.
"""

from __future__ import annotations

from topple.contract import TaskSpec
from topple.controllers.waypoint_pid import Driver

V_MAX = 0.30  # m/s
A_MAX = 0.8  # m/s^2


def make_controller(spec: TaskSpec) -> Driver:
    return Driver.for_task(spec, v_max=V_MAX, a_max=A_MAX)
