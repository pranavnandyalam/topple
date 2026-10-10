"""Cascaded PID baseline for ``balance-v1``.

Inner loop: PD on pitch, chasing a lean target. Outer loop: the lean target pulls the robot back
toward where it started (wheel travel is the integral of wheel speed, so this is the "I" of a
speed PI loop). A separate PD on heading steers. Plain float math only, so the same code ports to
firmware and runs in the browser.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from topple.contract import BalancerObs


def _clamp(value: float, limit: float) -> float:
    return min(max(value, -limit), limit)


@dataclass(frozen=True)
class PIDGains:
    pitch_kp: float = 6.0  # command per rad of pitch error
    pitch_kd: float = 0.6  # command per rad/s of pitch rate
    travel_kp: float = 0.02  # rad of lean per rad of wheel travel
    speed_kp: float = 0.02  # rad of lean per rad/s of wheel speed
    yaw_kp: float = 0.5  # differential command per rad of heading error
    yaw_kd: float = 0.05  # differential command per rad/s of yaw rate
    max_lean_rad: float = 0.35  # 20 deg: a common safety limit on the lean target

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PIDGains:
        return cls(**{k: float(v) for k, v in data.items()})


class PIDController:
    """Stateless cascaded PID: every output depends only on the current observation."""

    def __init__(self, gains: PIDGains | None = None) -> None:
        self.gains = gains or PIDGains()

    def reset(self, rng_seed: int) -> None:
        """Nothing to reset: the controller keeps no state between steps."""

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        g = self.gains
        travel = 0.5 * (obs.wheel_pos[0] + obs.wheel_pos[1])
        speed = 0.5 * (obs.wheel_vel[0] + obs.wheel_vel[1])
        lean_target = _clamp(-(g.travel_kp * travel + g.speed_kp * speed), g.max_lean_rad)
        drive = g.pitch_kp * (obs.pitch - lean_target) + g.pitch_kd * obs.pitch_rate
        turn = -(g.yaw_kp * obs.yaw + g.yaw_kd * obs.yaw_rate)
        return (drive - turn, drive + turn)
