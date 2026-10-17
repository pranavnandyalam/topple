"""``waypoint-v1``: the balancer drives through 4 gates and holds a dock (spec §8).

``WaypointEnv`` is ``BalanceEnv`` with the same physics, sensing, pushes and termination (no
drift termination in eval) plus a course:
- after every step it does not terminate on, it advances the shared ``WaypointTracker``
  (``topple.waypoint``, the scorer's own rules), so its events equal the scorer's;
- its observation is ``balancer-target.obs.v1``: the balance features, then the current
  waypoint, the one after it and ``final``, in the yaw-only robot frame from the true axle pose
  (§8.4); the observation of the step that reaches gate k already shows gate k + 1;
- its training reward is §8.8's (not frozen, never part of a score).
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict

from topple.challenge import Challenge, load_challenge
from topple.contract import BalancerObs, BalancerTargetObs
from topple.course import course_points
from topple.envs.balance import BalanceEnv
from topple.waypoint import WaypointTracker, to_course_frame


class WaypointRewardConfig(BaseModel):
    """§8.8's training reward weights (not frozen; contract §13)."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    balance_scale: float = 0.1
    progress: float = 20.0
    gate: float = 10.0
    dock: float = 20.0
    terminated: float = -50.0
    dock_hold: float = 0.0


class WaypointEnv(BalanceEnv):
    def __init__(
        self,
        preset: str = "standard",
        *,
        challenge: Challenge | None = None,
        waypoint_reward: WaypointRewardConfig | None = None,
        **kwargs: Any,
    ) -> None:
        challenge = challenge if challenge is not None else load_challenge("waypoint-v1")
        if challenge.kind != "waypoint" or challenge.config.course is None:
            raise ValueError(f"{challenge.id} is not a waypoint challenge")
        super().__init__(preset, challenge=challenge, **kwargs)
        self.course_cfg = challenge.config.course
        self.waypoint_reward = waypoint_reward or WaypointRewardConfig()
        self._tracker: WaypointTracker | None = None
        self._points: tuple[tuple[float, float], ...] = ()
        self._step_start_wp = 0
        self._step_start_d = 0.0
        self._advanced = False
        self._docked_now = False

    # --- the course ---------------------------------------------------------------------------

    @property
    def tracker(self) -> WaypointTracker:
        if self._tracker is None:
            raise RuntimeError("reset() first")
        return self._tracker

    def _course_xy(self) -> tuple[float, float]:
        """The true axle midpoint (the chassis origin, contract §3) in the course frame."""
        b, d = self.binding, self.data
        xy = (float(d.qpos[b.root_qposadr]), float(d.qpos[b.root_qposadr + 1]))
        origin = (float(self._xy0[0]), float(self._xy0[1]))
        return to_course_frame(xy, origin, self._yaw0)

    def _distance_to(self, i: int) -> float:
        x, y = self._course_xy()
        px, py = self._points[i]
        return math.hypot(x - px, y - py)

    def _after_reset(self) -> None:
        scenario = self.scenario
        if scenario.course is None:
            raise ValueError("a waypoint scenario needs its course")
        self._points = course_points(scenario.course, self.course_cfg)
        self._tracker = WaypointTracker(
            self._points,
            self._course_xy(),
            gate_radius_m=self.course_cfg.gate_radius_m,
            dock_hold_steps=self.course_cfg.dock_hold_steps,
        )
        self._step_start_wp = 0
        self._step_start_d = self._distance_to(0)
        self._advanced = self._docked_now = False

    def _after_physics(self) -> None:
        tracker = self.tracker
        self._step_start_wp = tracker.current
        reached_before = sum(s is not None for s in tracker.reached)
        docked_before = tracker.docked
        if self._termination is None:  # §8.5: the terminating step is never checked
            tracker.check(self._step_count, self._course_xy())
        reached_after = sum(s is not None for s in tracker.reached)
        self._docked_now = tracker.docked and not docked_before
        self._advanced = reached_after > reached_before and not self._docked_now

    # --- observation --------------------------------------------------------------------------

    def _observe(self, last_action: tuple[float, ...]) -> BalancerTargetObs:  # type: ignore[override]
        base = super()._observe(last_action)
        assert isinstance(base, BalancerObs)
        tracker = self.tracker
        dock = tracker.dock
        current = dock if tracker.docked else tracker.current
        following = min(current + 1, dock)
        b, d = self.binding, self.data
        world = (float(d.qpos[b.root_qposadr]), float(d.qpos[b.root_qposadr + 1]))
        yaw = self._true_state().yaw
        origin = (float(self._xy0[0]), float(self._xy0[1]))

        def robot_frame(i: int) -> tuple[float, float]:
            # the waypoint in world coordinates, then in the yaw-only robot frame
            px, py = self._points[i]
            c0, s0 = math.cos(self._yaw0), math.sin(self._yaw0)
            wx, wy = origin[0] + c0 * px - s0 * py, origin[1] + s0 * px + c0 * py
            return to_course_frame((wx, wy), world, yaw)

        return BalancerTargetObs(
            base.t,
            base.pitch,
            base.pitch_rate,
            base.yaw,
            base.yaw_rate,
            base.wheel_pos,
            base.wheel_vel,
            base.last_action,
            robot_frame(current),
            robot_frame(following),
            1.0 if current == dock else 0.0,
        )

    # --- training reward ----------------------------------------------------------------------

    def _reward(self, u: tuple[float, ...]) -> tuple[float, dict[str, float]]:
        r_bal, costs = super()._reward(u)
        w = self.waypoint_reward
        leg = self.course_cfg.leg_m
        d_now = self._distance_to(self._step_start_wp)
        progress = (self._step_start_d - d_now) / leg
        tracker = self.tracker
        current = tracker.current
        in_dock = current == tracker.dock and self._distance_to(current) <= (
            self.course_cfg.gate_radius_m
        )
        drifting = (
            self.max_drift_m is not None
            and self._termination is None
            and self._drift() > self.max_drift_m
        )
        terminated = self._termination is not None or drifting
        reward = (
            w.balance_scale * r_bal
            + w.progress * progress
            + w.gate * float(self._advanced)
            + w.dock * float(self._docked_now)
            + w.dock_hold * float(in_dock)
            + w.terminated * float(terminated)
        )
        self._step_start_d = self._distance_to(current)
        terms = {**costs, "progress": progress, "gate": float(self._advanced)}
        terms["dock"] = float(self._docked_now)
        return reward, terms
