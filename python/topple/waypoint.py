"""``waypoint-v1`` rules, events, score and success (docs/design/phase-2.md §8.5, §8.6).

Pure functions of the recorded planar path and the course points, used alike by the env (online
events) and the scorer (a trajectory replayed or re-scored), so both always agree. Nothing here
imports the env or the evaluator: the caller extracts the path and passes plain numbers.

Conventions shared by every function:

- ``qpos_xy`` holds rows 0..T of the axle-midpoint position in the course frame (the frame the
  course points are in: origin at the reset pose, x the initial heading). Row 0 is the reset
  state, row k the state after control step k.
- Distances are ``math.hypot`` of Python floats, so they are the same doubles on every platform.
- An episode error scores 0 and is never a success (contract §12): the caller passes
  ``error=True`` to :func:`waypoint_score_parts` and :func:`waypoint_success`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator

FALL_REASONS = frozenset({"fell", "floor_contact"})
"""Recorded reasons that are falls; mirrors ``topple.evaluate.FALL_REASONS`` (a test pins the
two equal). Not imported from there: ``evaluate`` imports the env, and the scorer imports this."""

TERMINATING_REASONS = FALL_REASONS | {"sim_unstable"}
"""Recorded reasons for which the env itself ended the episode on its last step (contract §11);
that step's reach check is skipped, so the events drop the last row."""

FALL_CAUSE_WINDOW_S = 1.5
"""A fall is ``push``-caused when a push started at most this long before it (§8.5)."""

SCORE_PARTS = ("course_pts", "speed_pts", "alive_pts", "battery_pts")

WAYPOINT_METRICS = (
    "steps",
    "termination",
    "survived",
    "time_upright_s",
    "energy_j",
    "saturation_fraction",
    "rms_tilt_rad",
    "waypoints",
    "progress",
    "docked",
    "fell",
    "t_finish_s",
    "max_hold_steps",
    "fall_cause",
    *SCORE_PARTS,
)
"""The 18 per-episode metric names of waypoint-v1, in order (§8.6; contract §15 allows 32)."""

_SUM_TOLERANCE = 1e-9

_Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class WaypointScore(BaseModel):
    """Every waypoint-v1 scorer constant, from ``challenge.yaml`` (so all are in protocol_hash).

    No field has a default: no scorer constant lives in code (§8.1). ``gate_radius_m``,
    ``leg_m`` and ``dock_hold_steps`` belong to the course config, not here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: tuple[_Positive, _Positive, _Positive, _Positive]
    """Points for course, speed, staying up and battery; they sum to 100."""
    t_par_s: _Positive
    """Finish time that earns the full speed points."""
    dock_progress_weights: tuple[_Positive, _Positive]
    """Dock progress split between closeness and hold; they sum to 1."""
    success_energy_factor: _Positive
    e_ref_j: _Positive

    @field_validator("weights")
    @classmethod
    def _weights_sum_to_100(cls, weights: tuple[float, ...]) -> tuple[float, ...]:
        if abs(math.fsum(weights) - 100.0) > _SUM_TOLERANCE:
            raise ValueError(f"weights must sum to 100, got {math.fsum(weights)}")
        return weights

    @field_validator("dock_progress_weights")
    @classmethod
    def _dock_weights_sum_to_1(cls, weights: tuple[float, ...]) -> tuple[float, ...]:
        if abs(math.fsum(weights) - 1.0) > _SUM_TOLERANCE:
            raise ValueError(f"dock_progress_weights must sum to 1, got {math.fsum(weights)}")
        return weights


@dataclass(frozen=True)
class WaypointEvents:
    """What happened on a course (§8.5).

    ``reached_step[i]`` is the step k that completed waypoint i (the last one is the dock: the
    step its hold reached ``dock_hold_steps``), or None. ``waypoints`` (g) counts the completed
    ones. ``d_min`` is the closest distance to the current waypoint (the last one once docked)
    since it became current. ``max_hold_steps`` is the longest dock hold streak (at most
    ``dock_hold_steps``, reached only when docked).
    """

    reached_step: tuple[int | None, ...]
    docked: bool
    t_finish_s: float | None
    d_min: float
    max_hold_steps: int
    waypoints: int

    def __post_init__(self) -> None:
        problem = self._problem()
        if problem:
            raise ValueError(f"WaypointEvents: {problem}")

    def _problem(self) -> str:
        steps = self.reached_step
        if not steps:
            return "reached_step is empty"
        done = [s for s in steps if s is not None]
        if any(s is None for s in steps[: len(done)]):
            return "waypoints must be completed in order"
        if any(s < 1 for s in done) or any(b <= a for a, b in itertools.pairwise(done)):
            return "completion steps must be >= 1 and strictly increasing"
        if self.waypoints != len(done):
            return f"waypoints {self.waypoints} != {len(done)} completed"
        if self.docked != (steps[-1] is not None):
            return "docked must equal 'the last waypoint was completed'"
        if (self.t_finish_s is not None) != self.docked:
            return "t_finish_s must be set exactly when docked"
        if self.t_finish_s is not None and not (
            math.isfinite(self.t_finish_s) and self.t_finish_s >= 0
        ):
            return "t_finish_s must be finite and >= 0"
        if not (math.isfinite(self.d_min) and self.d_min >= 0):
            return "d_min must be finite and >= 0"
        if self.max_hold_steps < 0:
            return "max_hold_steps must be >= 0"
        return ""


def _check_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")


def _check_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")


def _as_points(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 1:
        raise ValueError("points must hold at least the dock")
    out = []
    for p in points:
        if len(p) != 2 or not all(math.isfinite(c) for c in p):
            raise ValueError(f"points must be finite (x, y) pairs, got {p!r}")
        out.append((float(p[0]), float(p[1])))
    return out


def _as_rows(qpos_xy: npt.NDArray[np.float64]) -> list[tuple[float, float]]:
    xy = np.asarray(qpos_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] < 1 or xy.shape[1] != 2:
        raise ValueError(f"qpos_xy must have shape (T + 1, 2), got {xy.shape}")
    if not np.isfinite(xy).all():
        raise ValueError("qpos_xy must be finite")
    return [(float(x), float(y)) for x, y in xy.tolist()]


def env_terminated(termination: str) -> bool:
    """Whether the env ended the episode itself (``fell``, ``floor_contact``, ``sim_unstable``).

    This is ``waypoint_events``' ``terminated``: ``timeout`` and ``error`` keep the last row.
    """
    return termination in TERMINATING_REASONS


class WaypointTracker:
    """§8.5's course rules, one checked row at a time: the single implementation the env (each
    step it does not terminate on) and the scorer (``waypoint_events``, over the recorded rows)
    share, so their events are equal by construction. Positions are in the course frame."""

    def __init__(
        self,
        points: Sequence[tuple[float, float]],
        first_xy: tuple[float, float],
        *,
        gate_radius_m: float,
        dock_hold_steps: int,
    ) -> None:
        _check_positive("gate_radius_m", gate_radius_m)
        _check_count("dock_hold_steps", dock_hold_steps)
        self.points = _as_points(points)
        self.gate_radius_m = gate_radius_m
        self.dock_hold_steps = dock_hold_steps
        self.reached: list[int | None] = [None] * len(self.points)
        self.current = 0
        self.d_min = self._dist(first_xy, 0)
        self.hold = 0
        self.max_hold = 0

    @property
    def dock(self) -> int:
        return len(self.points) - 1

    @property
    def docked(self) -> bool:
        return self.reached[self.dock] is not None

    def _dist(self, xy: tuple[float, float], i: int) -> float:
        return math.hypot(xy[0] - self.points[i][0], xy[1] - self.points[i][1])

    def check(self, k: int, xy: tuple[float, float]) -> None:
        """Check row ``k`` at position ``xy``: at most one waypoint advances, and nothing is
        checked once docked."""
        if self.docked:
            return
        d = self._dist(xy, self.current)
        self.d_min = min(self.d_min, d)
        if self.current < self.dock:
            if d <= self.gate_radius_m:
                self.reached[self.current] = k
                self.current += 1
                self.d_min = self._dist(xy, self.current)
            return
        self.hold = self.hold + 1 if d <= self.gate_radius_m else 0
        self.max_hold = max(self.max_hold, self.hold)
        if self.hold == self.dock_hold_steps:
            self.reached[self.dock] = k

    def events(self, control_dt: float) -> WaypointEvents:
        _check_positive("control_dt", control_dt)
        finish = self.reached[self.dock]
        return WaypointEvents(
            reached_step=tuple(self.reached),
            docked=finish is not None,
            t_finish_s=None if finish is None else finish * control_dt,
            d_min=self.d_min,
            max_hold_steps=self.max_hold,
            waypoints=sum(s is not None for s in self.reached),
        )


def to_course_frame(
    xy: tuple[float, float], origin: tuple[float, float], heading: float
) -> tuple[float, float]:
    """A world position in the course (start) frame: origin = axle midpoint at reset, x = the
    initial heading (§8.3). Env and scorer both use this on ``qpos`` values."""
    c, s = math.cos(heading), math.sin(heading)
    dx, dy = xy[0] - origin[0], xy[1] - origin[1]
    return (c * dx + s * dy, -s * dx + c * dy)


def waypoint_events(
    qpos_xy: npt.NDArray[np.float64],
    points: Sequence[tuple[float, float]],
    *,
    gate_radius_m: float,
    dock_hold_steps: int,
    control_dt: float,
    terminated: bool,
) -> WaypointEvents:
    """Course events of one episode (§8.5); ``points`` are the gates then the dock (last), and
    ``qpos_xy`` rows 0..T are course-frame positions.

    Rows k = 1..T-1 are checked when the env terminated the episode (it skips the check on the
    terminating step), else 1..T. At row k only the current waypoint is checked, so at most one
    waypoint advances per step and the next one is first checked at row k + 1. A gate is reached
    at the first checked row within ``gate_radius_m`` (inclusive). The dock's hold counts the
    consecutive checked rows within the radius from the row after it became current, resetting
    to 0 outside; it docks on the row the hold reaches ``dock_hold_steps``, and nothing is
    checked after that. ``d_min`` starts at the current waypoint's distance on the row it became
    current (row 0 for the first gate) and takes the minimum over the rows checked since.
    """
    _check_positive("control_dt", control_dt)
    rows = _as_rows(qpos_xy)
    tracker = WaypointTracker(
        points, rows[0], gate_radius_m=gate_radius_m, dock_hold_steps=dock_hold_steps
    )
    last = len(rows) - 1 - (1 if terminated else 0)
    for k in range(1, last + 1):
        tracker.check(k, rows[k])
        if tracker.docked:
            break
    return tracker.events(control_dt)


def waypoint_progress(
    events: WaypointEvents,
    *,
    leg_m: float,
    gate_radius_m: float,
    dock_hold_steps: int,
    dock_progress_weights: tuple[float, float],
) -> float:
    """Partial credit p toward the current waypoint, in [0, 1) (§8.6); 0 once docked.

    A gate: ``1 - min(d_min / leg_m, 1)``. The dock: ``w_close * clip((leg_m - d_min) / (leg_m -
    r), 0, 1) + w_hold * max_hold_steps / dock_hold_steps``, monotone in closeness and hold.
    """
    _check_positive("leg_m", leg_m)
    _check_positive("gate_radius_m", gate_radius_m)
    _check_count("dock_hold_steps", dock_hold_steps)
    if leg_m <= gate_radius_m:
        raise ValueError(f"leg_m ({leg_m}) must exceed gate_radius_m ({gate_radius_m})")
    n = len(events.reached_step)
    g = events.waypoints
    if g == n:
        return 0.0
    if g < n - 1:
        return 1.0 - min(events.d_min / leg_m, 1.0)
    if events.max_hold_steps >= dock_hold_steps:
        raise ValueError(
            f"max_hold_steps {events.max_hold_steps} reaches dock_hold_steps {dock_hold_steps} "
            "without docking"
        )
    w_close, w_hold = dock_progress_weights
    closeness = min(max((leg_m - events.d_min) / (leg_m - gate_radius_m), 0.0), 1.0)
    return w_close * closeness + w_hold * events.max_hold_steps / dock_hold_steps


def waypoint_score_parts(
    events: WaypointEvents,
    *,
    t_alive_s: float,
    episode_s: float,
    energy_j: float,
    score: WaypointScore,
    leg_m: float,
    gate_radius_m: float,
    dock_hold_steps: int,
    error: bool = False,
) -> dict[str, float]:
    """The four score parts (§8.6), in ``SCORE_PARTS`` order; :func:`waypoint_score` sums them.

    ``course = w0·(g + p)/n``, ``speed = w1·docked·clip((T - t_finish)/(T - t_par), 0, 1)``,
    ``alive = w2·t_alive/T``, ``battery = w3·(t_alive/T)·(1 - min(E/E_ref, 1))`` with T =
    ``episode_s`` and n the number of waypoints. ``t_alive_s`` is ``time_upright_s`` (contract
    §12). With ``error=True`` (the episode had an error) every part is 0.
    """
    if error:
        return dict.fromkeys(SCORE_PARTS, 0.0)
    _check_positive("episode_s", episode_s)
    if not score.t_par_s < episode_s:
        raise ValueError(f"t_par_s ({score.t_par_s}) must be shorter than episode_s ({episode_s})")
    if not (math.isfinite(t_alive_s) and 0.0 <= t_alive_s <= episode_s):
        raise ValueError(f"t_alive_s must be in [0, {episode_s}], got {t_alive_s!r}")
    if not (math.isfinite(energy_j) and energy_j >= 0.0):
        raise ValueError(f"energy_j must be finite and >= 0, got {energy_j!r}")
    w_course, w_speed, w_alive, w_battery = score.weights
    p = waypoint_progress(
        events,
        leg_m=leg_m,
        gate_radius_m=gate_radius_m,
        dock_hold_steps=dock_hold_steps,
        dock_progress_weights=score.dock_progress_weights,
    )
    n = len(events.reached_step)
    speed = 0.0
    if events.t_finish_s is not None:
        early = (episode_s - events.t_finish_s) / (episode_s - score.t_par_s)
        speed = w_speed * min(max(early, 0.0), 1.0)
    alive = t_alive_s / episode_s
    battery = 1.0 - min(energy_j / score.e_ref_j, 1.0)
    return {
        "course_pts": w_course * (events.waypoints + p) / n,
        "speed_pts": speed,
        "alive_pts": w_alive * alive,
        "battery_pts": w_battery * alive * battery,
    }


def waypoint_score(parts: Mapping[str, float]) -> float:
    """The episode score (0-100): the exact sum (``math.fsum``) of the four parts."""
    if set(parts) != set(SCORE_PARTS):
        raise ValueError(f"parts must be exactly {SCORE_PARTS}, got {sorted(parts)}")
    return math.fsum(parts[name] for name in SCORE_PARTS)


def waypoint_success(
    events: WaypointEvents,
    *,
    survived: bool,
    energy_j: float,
    score: WaypointScore,
    error: bool = False,
) -> bool:
    """``docked ∧ survived ∧ E <= success_energy_factor·E_ref``, never with an error (§8.6)."""
    return (
        not error
        and events.docked
        and survived
        and energy_j <= score.success_energy_factor * score.e_ref_j
    )


def fall_cause(
    termination: str,
    fall_step: int | None,
    push_start_steps: Sequence[int],
    control_dt: float,
) -> str:
    """``push``, ``self`` or ``none`` (§8.5; a metric of both kinds).

    For a fall (``termination`` in ``FALL_REASONS``) at step ``fall_step`` (the episode's step
    count T), ``push`` if some push started on a step s with ``0 < fall_step - s <= W``, where
    W = round(1.5 s / control_dt) steps (150 at 0.01 s), so a push exactly 1.5 s before counts and
    one starting after the fall does not; else ``self``. Every other reason is ``none``.
    Compared in whole steps, so the 1.5 s boundary is exact.
    """
    _check_positive("control_dt", control_dt)
    if termination not in FALL_REASONS:
        return "none"
    if fall_step is None or isinstance(fall_step, bool) or fall_step < 0:
        raise ValueError(f"fall_step must be a step >= 0 for a fall, got {fall_step!r}")
    window = round(FALL_CAUSE_WINDOW_S / control_dt)
    # a push scheduled at step T never ran: the episode ended during step T - 1 (review)
    if any(0 < fall_step - s <= window for s in push_start_steps):
        return "push"
    return "self"
