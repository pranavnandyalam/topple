"""Reference waypoint driver for ``waypoint-v1`` (docs/design/phase-2.md §8.7, normative).

The driver steers the shipped LQR balancer toward the current gate of a ``balancer-target.obs.v1``
observation. Each control step it picks a direction (it reverses toward a gate behind it, but
only while nearly stopped), ramps a speed reference toward a distance-proportional target,
integrates that speed into a wheel-position reference, balances around both with the LQR gains,
and turns toward the gate's bearing. Plain float math only, so the same code ports to firmware
and runs in the browser. ``test_driver_golden`` pins this law; change it only with a re-pin.
"""

from __future__ import annotations

import math
import numbers

from topple.contract import BalancerTargetObs, TaskSpec
from topple.controllers.lqr import DEFAULT_GAINS, LQRGains

WHEEL_RADIUS_M = 0.034
"""waypoint-v1's wheel radius (§8.7); the submissions take it from ``TaskSpec.robot``."""
CONTROL_DT_S = 0.01
"""waypoint-v1's control period (§8.7); the submissions take it from ``TaskSpec.control_dt``."""
DEFAULT_K_V = 1.2
"""Speed per metre of distance along the heading (1/s)."""

REVERSE_SWITCH_SPEED_MPS = 0.05
"""The direction may switch only while ``|v_ref|`` is below this (step 1)."""
X_REF_WINDOW_RAD = 3.0
"""``x_ref`` stays within this many radians of the current mean wheel angle (step 3)."""
FEEDFORWARD_DIVISOR = 30.0
"""Drive feedforward: ``(v_ref / r) / 30`` per rad/s of reference wheel speed (step 4)."""
TURN_GAIN = 0.8
"""Differential command per rad of bearing error (step 5)."""
TURN_LIMIT = 0.30
"""Bound on the bearing part of the turn command (step 5)."""
TURN_DAMPING = 0.06
"""Differential command per rad/s of yaw rate (step 5)."""
GATE_DEADBAND_M = 0.03
"""Below this distance the bearing is ignored for turning (step 5)."""


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _check_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not (math.isfinite(out) and out > 0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return out


# --- the six steps of §8.7, as pure functions (unit-tested one by one) -------------------------


def bearing(target: tuple[float, float]) -> tuple[float, float]:
    """Distance and bearing (rad, (-pi, pi]) of a robot-frame point (x forward, y left)."""
    x, y = target
    return math.hypot(x, y), math.atan2(y, x)


def choose_direction(direction: int, herr: float, v_ref: float) -> int:
    """Step 1: ``-1`` iff the target is more than 90 deg off the nose, else ``+1``; the current
    ``direction`` is kept unless ``|v_ref| < 0.05`` m/s."""
    wanted = -1 if abs(herr) > math.pi / 2 else 1
    return wanted if abs(v_ref) < REVERSE_SWITCH_SPEED_MPS else direction


def reverse_bearing(herr: float) -> float:
    """Step 1: the bearing seen from the tail (``herr`` wrapped by pi), in (-pi, pi]."""
    return herr - math.pi if herr > 0 else herr + math.pi


def desired_speed(direction: int, d: float, herr: float, k_v: float, v_max: float) -> float:
    """Step 2: ``dir * clip(k_v * d * max(cos herr, 0)^2, 0, v_max)`` (m/s)."""
    along = d * max(math.cos(herr), 0.0) ** 2
    return direction * _clip(k_v * along, 0.0, v_max)


def ramp(v_ref: float, v_des: float, max_step: float) -> float:
    """Step 2: ``v_ref`` moved toward ``v_des`` by at most ``max_step``."""
    return v_ref + _clip(v_des - v_ref, -max_step, max_step)


def advance_position_ref(
    x_ref: float, v_ref: float, radius: float, dt: float, travel: float
) -> float:
    """Step 3: ``x_ref + (v_ref / r) * dt``, clamped to ``travel +- 3`` rad."""
    return _clip(
        x_ref + (v_ref / radius) * dt, travel - X_REF_WINDOW_RAD, travel + X_REF_WINDOW_RAD
    )


def balance_drive(
    gains: LQRGains,
    obs: BalancerTargetObs,
    v_ref: float,
    x_ref: float,
    radius: float,
) -> float:
    """Step 4: LQR feedback around the speed and position references, plus feedforward.

    With ``v_ref = x_ref = 0`` this is bit for bit the shipped LQR's drive (same terms, same
    order: adding the zero feedforward first changes no bits).
    """
    travel = 0.5 * (obs.wheel_pos[0] + obs.wheel_pos[1])
    speed = 0.5 * (obs.wheel_vel[0] + obs.wheel_vel[1])
    wheel_ref = v_ref / radius
    return (
        wheel_ref / FEEDFORWARD_DIVISOR
        + gains.k_pitch * obs.pitch
        + gains.k_pitch_rate * obs.pitch_rate
        + gains.k_travel * (travel - x_ref)
        + gains.k_speed * (speed - wheel_ref)
    )


def turn_command(herr: float, d: float, yaw_rate: float) -> float:
    """Step 5: ``clip(0.8 * herr, +-0.30) - 0.06 * yaw_rate``, with ``herr = 0`` when d < 0.03."""
    aim = 0.0 if d < GATE_DEADBAND_M else herr
    return _clip(TURN_GAIN * aim, -TURN_LIMIT, TURN_LIMIT) - TURN_DAMPING * yaw_rate


# --- the controller ---------------------------------------------------------------------------


class Driver:
    """The §8.7 reference driver: drives through ``obs.gate`` (the current waypoint).

    State per episode: the direction, the speed reference ``v_ref`` (m/s) and the wheel-angle
    reference ``x_ref`` (rad, 0 at reset like the LQR's travel setpoint). Deterministic: no
    randomness, so ``rng_seed`` is ignored.
    """

    def __init__(
        self,
        v_max: float,
        a_max: float,
        k_v: float = DEFAULT_K_V,
        *,
        wheel_radius_m: float = WHEEL_RADIUS_M,
        control_dt: float = CONTROL_DT_S,
        gains: LQRGains = DEFAULT_GAINS,
    ) -> None:
        self.v_max = _check_positive("v_max", v_max)
        self.a_max = _check_positive("a_max", a_max)
        self.k_v = _check_positive("k_v", k_v)
        self.wheel_radius_m = _check_positive("wheel_radius_m", wheel_radius_m)
        self.control_dt = _check_positive("control_dt", control_dt)
        self.gains = gains
        self._direction = 1
        self._v_ref = 0.0
        self._x_ref = 0.0

    @classmethod
    def for_task(
        cls, spec: TaskSpec, v_max: float, a_max: float, k_v: float = DEFAULT_K_V
    ) -> Driver:
        """A driver with the task's wheel radius and control period."""
        return cls(
            v_max,
            a_max,
            k_v,
            wheel_radius_m=spec.robot.wheel_radius_m,
            control_dt=spec.control_dt,
        )

    @property
    def direction(self) -> int:
        return self._direction

    @property
    def v_ref(self) -> float:
        return self._v_ref

    @property
    def x_ref(self) -> float:
        return self._x_ref

    def reset(self, rng_seed: int) -> None:
        self._direction = 1
        self._v_ref = 0.0
        self._x_ref = 0.0

    def act(self, obs: BalancerTargetObs) -> tuple[float, float]:
        return self.steer(obs, obs.gate)

    def steer(self, obs: BalancerTargetObs, target: tuple[float, float]) -> tuple[float, float]:
        """One control step of the §8.7 law toward ``target`` (robot frame); updates the state."""
        d, herr = bearing(target)
        self._direction = choose_direction(self._direction, herr, self._v_ref)
        if self._direction < 0:
            herr = reverse_bearing(herr)
        v_des = desired_speed(self._direction, d, herr, self.k_v, self.v_max)
        self._v_ref = ramp(self._v_ref, v_des, self.a_max * self.control_dt)
        travel = 0.5 * (obs.wheel_pos[0] + obs.wheel_pos[1])
        self._x_ref = advance_position_ref(
            self._x_ref, self._v_ref, self.wheel_radius_m, self.control_dt, travel
        )
        drive = balance_drive(self.gains, obs, self._v_ref, self._x_ref, self.wheel_radius_m)
        turn = turn_command(herr, d, obs.yaw_rate)
        return (drive - turn, drive + turn)
