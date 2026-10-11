"""Linear state-feedback (LQR) baseline for ``balance-v1``.

The four sagittal gains are a discrete LQR design on a model identified from the simulator
(``topple.design.lqr``); heading uses the same PD as the PID baseline. Plain float math only, so
the same code ports to firmware and runs in the browser.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from topple.contract import BalancerObs


@dataclass(frozen=True)
class LQRGains:
    k_pitch: float  # command per rad of pitch
    k_pitch_rate: float  # command per rad/s of pitch rate
    k_travel: float  # command per rad of mean wheel travel since reset
    k_speed: float  # command per rad/s of mean wheel speed
    yaw_kp: float = 0.5  # differential command per rad of heading error
    yaw_kd: float = 0.05  # differential command per rad/s of yaw rate

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{f.name} must be a finite number, got {value!r}")
            object.__setattr__(self, f.name, float(value))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LQRGains:
        return cls(**{k: float(v) for k, v in data.items()})


# balance-v1: `uv run python -m topple.design.lqr` (default weights), 4 significant digits.
# tests/test_lqr.py fails when the robot or its params drift away from these: re-run the design.
DEFAULT_GAINS = LQRGains(k_pitch=5.803, k_pitch_rate=0.7635, k_travel=0.2058, k_speed=0.1671)


class LQRController:
    """Stateless feedback on pitch, pitch rate, wheel travel and wheel speed, plus heading PD."""

    def __init__(self, gains: LQRGains | None = None) -> None:
        self.gains = gains if gains is not None else DEFAULT_GAINS

    def reset(self, rng_seed: int) -> None:
        """Nothing to reset: the controller keeps no state between steps."""

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        g = self.gains
        travel = 0.5 * (obs.wheel_pos[0] + obs.wheel_pos[1])
        speed = 0.5 * (obs.wheel_vel[0] + obs.wheel_vel[1])
        drive = (
            g.k_pitch * obs.pitch
            + g.k_pitch_rate * obs.pitch_rate
            + g.k_travel * travel
            + g.k_speed * speed
        )
        turn = -(g.yaw_kp * obs.yaw + g.yaw_kd * obs.yaw_rate)
        return (drive - turn, drive + turn)
