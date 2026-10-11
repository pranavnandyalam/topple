"""The shipped cascaded PID baseline as a submission.

Copy this file to start your own controller: tune ``PIDGains`` or replace the controller entirely.
See ``topple/controllers/pid.py`` for how it works.
"""

from __future__ import annotations

from topple.contract import TaskSpec
from topple.controllers.pid import PIDController


def make_controller(spec: TaskSpec) -> PIDController:
    return PIDController()
