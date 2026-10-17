"""Degenerate-strategy anchors for ``waypoint-v1`` (docs/design/phase-2.md §8.7).

Each anchor is a deliberately limited strategy whose score bounds the scorer from below: a
challenge whose scorer rewards one of these too much is mis-calibrated. All of them reuse the
shipped LQR gains and, where they drive, exactly the §8.7 driver law (``Driver``).

| Anchor | Definition |
|---|---|
| Balance in place | the shipped LQR (zero speed reference, heading hold) |
| Spin in place | the LQR's drive with a constant turn command of 0.30 |
| Creeper | ``Driver(v_max=0.12, a_max=0.8)`` |
| Gates-only | the driver law, aiming 0.30 m past the dock along the last leg |
| Reckless | ``Driver(v_max=0.60, a_max=0.8)`` |

How gates-only knows the last leg: ``balancer-target.obs.v1`` shows only the dock once
``final = 1`` (``gate = next = dock``), so the direction cannot be read from that observation
alone. While ``final = 0`` both the current gate and the one after it are visible in the same
robot frame, so ``next - gate`` is the direction of the leg that starts at the current gate.
``GatesOnly`` rotates that vector by the observed ``yaw`` into the reset-heading frame and keeps
the latest one; the one held when ``final`` turns 1 is the last leg (last gate -> dock). On the
final leg it rotates it back by the current ``yaw`` and aims at ``dock + 0.30 m * direction``.
Only the change in ``yaw`` between the last gate and now matters, so odometry drift over the
last leg is the only error. If an episode starts with ``final = 1`` (no gate before the dock),
the last leg is taken to run from the reset pose to the dock.
"""

from __future__ import annotations

import math

from topple.contract import BalancerObs, BalancerTargetObs, TaskSpec
from topple.controllers.lqr import DEFAULT_GAINS, LQRController, LQRGains
from topple.controllers.waypoint_pid import WHEEL_RADIUS_M, Driver, balance_drive

SPIN_TURN = 0.30
"""Spin in place: the constant differential command (positive turns left)."""
CREEPER_V_MAX = 0.12
RECKLESS_V_MAX = 0.60
ANCHOR_A_MAX = 0.8
"""Creeper, gates-only and reckless share the shipped driver's acceleration limit."""
GATES_ONLY_V_MAX = 0.30
"""Gates-only drives at the shipped driver's top speed."""
OVERSHOOT_M = 0.30
"""Gates-only aims this far past the dock along the last leg."""
_MIN_LEG_M = 1e-9
"""A leg shorter than this has no direction; the previous one is kept."""


def balancer_obs(obs: BalancerTargetObs) -> BalancerObs:
    """The ``balancer.obs.v1`` part of a waypoint observation (the same floats)."""
    return BalancerObs(
        obs.t,
        obs.pitch,
        obs.pitch_rate,
        obs.yaw,
        obs.yaw_rate,
        obs.wheel_pos,
        obs.wheel_vel,
        obs.last_action,
    )


def rotate(vector: tuple[float, float], angle: float) -> tuple[float, float]:
    """``vector`` rotated counter-clockwise by ``angle`` (rad)."""
    c, s = math.cos(angle), math.sin(angle)
    x, y = vector
    return (c * x - s * y, s * x + c * y)


class BalanceInPlace:
    """Balance in place: the shipped LQR, fed the balance features of each observation."""

    def __init__(self, gains: LQRGains = DEFAULT_GAINS) -> None:
        self._lqr = LQRController(gains)

    def reset(self, rng_seed: int) -> None:
        self._lqr.reset(rng_seed)

    def act(self, obs: BalancerTargetObs) -> tuple[float, float]:
        return self._lqr.act(balancer_obs(obs))


class SpinInPlace:
    """Spin in place: the LQR's drive (zero speed and position reference), turn fixed at 0.30."""

    def __init__(self, gains: LQRGains = DEFAULT_GAINS) -> None:
        self.gains = gains

    def reset(self, rng_seed: int) -> None:
        """Nothing to reset: the controller keeps no state between steps."""

    def act(self, obs: BalancerTargetObs) -> tuple[float, float]:
        # v_ref = x_ref = 0: the wheel radius only scales the zero speed reference
        drive = balance_drive(self.gains, obs, 0.0, 0.0, WHEEL_RADIUS_M)
        return (drive - SPIN_TURN, drive + SPIN_TURN)


class GatesOnly:
    """Gates-only: the driver law through every gate, then aimed 0.30 m past the dock along the
    last leg, so it docks only incidentally (see the module docstring for the last leg)."""

    def __init__(self, driver: Driver) -> None:
        self.driver = driver
        self._leg: tuple[float, float] | None = None  # unit vector, reset-heading frame

    @property
    def last_leg(self) -> tuple[float, float] | None:
        """The latest leg direction seen (unit vector in the reset-heading frame)."""
        return self._leg

    def reset(self, rng_seed: int) -> None:
        self.driver.reset(rng_seed)
        self._leg = None

    def act(self, obs: BalancerTargetObs) -> tuple[float, float]:
        return self.driver.steer(obs, self.aim(obs))

    def aim(self, obs: BalancerTargetObs) -> tuple[float, float]:
        """The robot-frame point to steer at this step; updates the remembered leg."""
        if obs.final == 0.0:
            self._remember(obs.yaw, obs.next[0] - obs.gate[0], obs.next[1] - obs.gate[1])
            return obs.gate
        if self._leg is None:  # the dock is the first waypoint: the leg starts at the robot
            self._remember(obs.yaw, obs.gate[0], obs.gate[1])
        if self._leg is None:  # standing on the dock with no leg seen: aim at the dock
            return obs.gate
        ux, uy = rotate(self._leg, -obs.yaw)
        return (obs.gate[0] + OVERSHOOT_M * ux, obs.gate[1] + OVERSHOOT_M * uy)

    def _remember(self, yaw: float, dx: float, dy: float) -> None:
        length = math.hypot(dx, dy)
        if length > _MIN_LEG_M:
            self._leg = rotate((dx / length, dy / length), yaw)


def balance_in_place(spec: TaskSpec | None = None) -> BalanceInPlace:
    return BalanceInPlace()


def spin_in_place(spec: TaskSpec | None = None) -> SpinInPlace:
    return SpinInPlace()


def creeper(spec: TaskSpec) -> Driver:
    return Driver.for_task(spec, v_max=CREEPER_V_MAX, a_max=ANCHOR_A_MAX)


def gates_only(spec: TaskSpec) -> GatesOnly:
    return GatesOnly(Driver.for_task(spec, v_max=GATES_ONLY_V_MAX, a_max=ANCHOR_A_MAX))


def reckless(spec: TaskSpec) -> Driver:
    return Driver.for_task(spec, v_max=RECKLESS_V_MAX, a_max=ANCHOR_A_MAX)
