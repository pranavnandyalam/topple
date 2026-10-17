"""waypoint-v1 rules and events (docs/design/phase-2.md §8.5): pure functions, no env.

Every path here is hand-made: ``qpos_xy`` rows 0..T in the course frame (row 0 the reset state).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from topple.waypoint import (
    FALL_REASONS,
    TERMINATING_REASONS,
    WaypointEvents,
    env_terminated,
    fall_cause,
    waypoint_events,
)

DT = 0.01
R = 0.10
# A straight course along x, every point exactly representable.
LINE = ((1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0))


def path(*rows: tuple[float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def events(
    xy: np.ndarray,
    points: tuple[tuple[float, float], ...] = LINE,
    *,
    r: float = R,
    hold: int = 3,
    terminated: bool = False,
) -> WaypointEvents:
    return waypoint_events(
        xy, points, gate_radius_m=r, dock_hold_steps=hold, control_dt=DT, terminated=terminated
    )


def at(i: int, dx: float = 0.0, dy: float = 0.0) -> tuple[float, float]:
    """A row placed relative to waypoint i of LINE."""
    return (LINE[i][0] + dx, LINE[i][1] + dy)


ORIGIN = (0.0, 0.0)


# --- gates ---------------------------------------------------------------------------------


def test_a_gate_is_reached_at_exactly_the_radius() -> None:
    # dx = 0 exactly and dy = 0.1, so math.hypot is exactly 0.1 == r.
    e = events(path(ORIGIN, at(0, dy=R)))
    assert math.hypot(0.0, R) == R
    assert e.reached_step == (1, None, None, None, None)
    assert e.waypoints == 1


def test_a_gate_is_not_reached_just_outside_the_radius() -> None:
    row = at(0, dx=-(R + 1e-12))
    assert math.hypot(row[0] - LINE[0][0], row[1]) > R
    e = events(path(ORIGIN, row))
    assert e.reached_step == (None,) * 5
    assert e.waypoints == 0


def test_the_radius_comparison_is_inclusive_and_exact() -> None:
    row = (0.875, 0.0)  # exactly 0.125 from (1, 0)
    pts = ((1.0, 0.0), (9.0, 9.0))
    assert events(path(ORIGIN, row), pts, r=0.125).reached_step[0] == 1
    assert events(path(ORIGIN, row), pts, r=math.nextafter(0.125, 0.0)).reached_step[0] is None


def test_at_most_one_waypoint_advances_per_step() -> None:
    # Gates 0 and 1 sit on the same spot: one row inside both completes only gate 0.
    pts = ((1.0, 0.0), (1.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0))
    one = events(path(ORIGIN, (1.0, 0.0)), pts)
    assert one.reached_step == (1, None, None, None, None)
    two = events(path(ORIGIN, (1.0, 0.0), (1.0, 0.0)), pts)
    assert two.reached_step == (1, 2, None, None, None)
    assert two.waypoints == 2


def test_a_path_crossing_two_gates_in_one_step_only_takes_the_current_one() -> None:
    pts = ((1.0, 0.0), (1.05, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0))
    e = events(path(ORIGIN, (1.02, 0.0), (5.0, 5.0)), pts)
    assert e.reached_step[:2] == (1, None)


def test_gates_must_be_taken_in_order() -> None:
    # Passing through gate 1 before gate 0 does not count.
    e = events(path(ORIGIN, at(1), at(0), at(1)))
    assert e.reached_step == (2, 3, None, None, None)


def test_row_zero_never_reaches_a_gate() -> None:
    pts = ((0.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0))
    assert events(path(ORIGIN), pts).reached_step[0] is None
    assert events(path(ORIGIN, ORIGIN), pts).reached_step[0] == 1


def test_terminated_drops_the_last_row() -> None:
    xy = path(ORIGIN, (0.5, 0.0), at(0))
    assert events(xy, terminated=False).reached_step[0] == 2
    assert events(xy, terminated=True).reached_step[0] is None
    assert events(xy, terminated=True).waypoints == 0


def test_terminated_with_one_row_checks_nothing() -> None:
    e = events(path(ORIGIN, at(0)), terminated=True)
    assert e.reached_step == (None,) * 5
    assert e.d_min == 1.0  # the start row only


def test_no_steps_checks_nothing() -> None:
    e = events(path(ORIGIN))
    assert e.reached_step == (None,) * 5
    assert not e.docked
    assert e.t_finish_s is None
    assert e.d_min == 1.0
    assert e.max_hold_steps == 0
    assert e.waypoints == 0


# --- dock ------------------------------------------------------------------------------------


def through_gates() -> list[tuple[float, float]]:
    """Rows 1..4 reach gates 0..3; the dock is current from row 4 on."""
    return [ORIGIN, at(0), at(1), at(2), at(3)]


def test_docked_at_exactly_dock_hold_steps() -> None:
    rows = [*through_gates(), at(4), at(4), at(4)]
    e = events(path(*rows), hold=3)
    assert e.reached_step == (1, 2, 3, 4, 7)
    assert e.docked
    assert e.waypoints == 5
    assert e.max_hold_steps == 3
    assert e.t_finish_s == 7 * DT


def test_one_step_short_of_the_hold_is_not_docked() -> None:
    rows = [*through_gates(), at(4), at(4)]
    e = events(path(*rows), hold=3)
    assert not e.docked
    assert e.reached_step[4] is None
    assert e.t_finish_s is None
    assert e.waypoints == 4
    assert e.max_hold_steps == 2


def test_the_dock_hold_resets_when_leaving() -> None:
    out = at(4, dx=0.5)
    rows = [*through_gates(), at(4), at(4), out, at(4), at(4), out]
    e = events(path(*rows), hold=3)
    assert not e.docked
    assert e.max_hold_steps == 2
    rows = [*through_gates(), at(4), at(4), out, at(4), at(4), at(4)]
    e = events(path(*rows), hold=3)
    assert e.docked
    assert e.reached_step[4] == 10
    assert e.t_finish_s == 10 * DT


def test_the_hold_counts_from_the_step_after_the_dock_becomes_current() -> None:
    # Gate 3 and the dock share a spot: the row that completes gate 3 does not hold the dock.
    pts = ((1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (4.0, 0.0))
    rows = [ORIGIN, (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (4.0, 0.0), (4.0, 0.0)]
    e = events(path(*rows), pts, hold=3)
    assert e.reached_step[3] == 4
    assert not e.docked
    assert e.max_hold_steps == 2
    e = events(path(*rows, (4.0, 0.0)), pts, hold=3)
    assert e.reached_step == (1, 2, 3, 4, 7)


def test_the_hold_stops_counting_once_docked() -> None:
    rows = [*through_gates(), *([at(4)] * 10)]
    e = events(path(*rows), hold=3)
    assert e.reached_step[4] == 7
    assert e.max_hold_steps == 3  # never more than the hold once docked


def test_docking_on_the_last_row_counts_unless_terminated() -> None:
    rows = [*through_gates(), at(4), at(4), at(4)]
    assert events(path(*rows), hold=3, terminated=False).docked
    e = events(path(*rows), hold=3, terminated=True)
    assert not e.docked
    assert e.max_hold_steps == 2


def test_a_hold_of_one_docks_on_the_first_step_inside() -> None:
    rows = [*through_gates(), at(4, dy=R)]
    e = events(path(*rows), hold=1)
    assert e.reached_step[4] == 5
    assert e.t_finish_s == 5 * DT


def test_a_single_point_course_is_just_a_dock() -> None:
    e = events(path(ORIGIN, (1.0, 0.0), (1.0, 0.0)), ((1.0, 0.0),), hold=2)
    assert e.reached_step == (2,)
    assert e.docked
    assert e.waypoints == 1


def test_a_full_course_on_a_dense_path_matches_hand_counts() -> None:
    # x = k / 64 is exact in binary, and so is every distance to the integer waypoints.
    xs = np.arange(0, 321) / 64.0  # 0 .. 5.0
    xy = np.stack([xs, np.zeros_like(xs)], axis=1)
    xy = np.concatenate([xy, np.tile(xy[-1:], (100, 1))])
    e = events(xy, r=0.125, hold=100)
    # Gate i at x = i + 1 is first within 0.125 at x = i + 0.875, row 64 i + 56.
    assert e.reached_step[:4] == (56, 120, 184, 248)
    # The dock is current from row 248; inside from x = 4.875 (row 312), 100 rows in -> row 411.
    assert e.reached_step[4] == 411
    assert e.t_finish_s == 411 * DT
    assert e.max_hold_steps == 100


def test_driving_a_generated_course_docks() -> None:
    """Straight legs through ``course_points`` at 1 cm per step, then a stop on the dock."""
    course = pytest.importorskip("topple.course")
    cfg = course.CourseConfig(
        n_waypoints=5,
        leg_m=0.70,
        turn_deg=(30.0, 150.0),
        max_radius_m=1.5,
        max_attempts=1000,
        gate_radius_m=0.10,
        dock_hold_steps=100,
    )
    for seed in (0, 1, 3000):
        pts = course.course_points(course.make_course(seed, cfg), cfg)
        rows = [np.zeros(2)]
        for a, b in zip(((0.0, 0.0), *pts), pts, strict=False):
            rows += [np.array(a) + (np.array(b) - np.array(a)) * t for t in np.arange(1, 71) / 70]
        rows += [np.array(pts[-1])] * 120
        e = waypoint_events(
            np.array(rows),
            pts,
            gate_radius_m=0.10,
            dock_hold_steps=100,
            control_dt=DT,
            terminated=False,
        )
        assert e.docked, seed
        assert e.waypoints == 5
        # Point i is row 70 (i + 1); 0.1 m (10 rows) before it is within the radius, give or
        # take the last bit of the interpolated distance.
        steps = e.reached_step
        for i in range(4):
            assert 70 * i + 60 <= (steps[i] or 0) <= 70 * i + 61, (seed, i, steps)
        # The dock (row 350) is entered at row 340 or 341 and held for 100 rows.
        assert 439 <= (steps[4] or 0) <= 440, (seed, steps)
        assert e.t_finish_s == steps[4] * DT


# --- d_min ---------------------------------------------------------------------------------


def test_d_min_is_the_closest_distance_to_the_current_gate() -> None:
    e = events(path(ORIGIN, (0.5, 0.0), (0.75, 0.0), (0.25, 0.0)))
    assert e.d_min == 0.25
    assert e.waypoints == 0


def test_d_min_restarts_at_the_row_that_completes_the_previous_waypoint() -> None:
    # Row 2 reaches gate 0 at x = 0.9; gate 1 (x = 2) is then 1.1 away, closest later at 1.5.
    e = events(path(ORIGIN, (0.5, 0.0), (0.9, 0.0), (1.5, 0.0), (1.25, 0.0)))
    assert e.reached_step[0] == 2
    assert e.d_min == 0.5
    # Reached on the last row: the next gate's d_min is its distance at that row.
    e = events(path(ORIGIN, (0.5, 0.0), (0.9, 0.0)))
    assert e.d_min == pytest.approx(1.1, abs=1e-15)
    assert e.d_min == math.hypot(0.9 - 2.0, 0.0)


def test_d_min_ignores_rows_closer_to_other_waypoints() -> None:
    e = events(path(ORIGIN, (0.5, 0.0), at(1), (0.6, 0.0)))
    assert e.waypoints == 0
    assert e.d_min == 0.4


def test_d_min_ignores_the_dropped_terminal_row() -> None:
    e = events(path(ORIGIN, (0.5, 0.0), (0.95, 0.0)), terminated=True)
    assert e.d_min == 0.5


def test_d_min_tracks_the_dock_and_freezes_at_docking() -> None:
    rows = [*through_gates(), at(4, dx=-0.3), at(4, dx=0.05), at(4), at(4), at(4, dx=-0.5)]
    e = events(path(*rows), hold=3)
    assert e.docked
    assert e.d_min == 0.0


def test_the_start_row_counts_for_the_first_gate() -> None:
    # The robot starts 1.0 from gate 0 and only backs away: d_min stays the start distance.
    e = events(path(ORIGIN, (-0.5, 0.0), (-1.0, 0.0)))
    assert e.d_min == 1.0


# --- validation --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "xy",
    [
        np.zeros((0, 2)),
        np.zeros((3,)),
        np.zeros((3, 3)),
        np.array([[0.0, 0.0], [math.nan, 0.0]]),
        np.array([[0.0, 0.0], [math.inf, 0.0]]),
    ],
)
def test_bad_paths_are_refused(xy: np.ndarray) -> None:
    with pytest.raises(ValueError, match="qpos_xy"):
        events(xy)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gate_radius_m": 0.0}, "gate_radius_m"),
        ({"gate_radius_m": -0.1}, "gate_radius_m"),
        ({"gate_radius_m": math.nan}, "gate_radius_m"),
        ({"gate_radius_m": math.inf}, "gate_radius_m"),
        ({"dock_hold_steps": 0}, "dock_hold_steps"),
        ({"dock_hold_steps": True}, "dock_hold_steps"),
        ({"control_dt": 0.0}, "control_dt"),
        ({"control_dt": math.nan}, "control_dt"),
    ],
)
def test_bad_parameters_are_refused(kwargs: dict[str, object], match: str) -> None:
    args: dict[str, object] = {
        "gate_radius_m": R,
        "dock_hold_steps": 3,
        "control_dt": DT,
        "terminated": False,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        waypoint_events(path(ORIGIN), LINE, **args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "points",
    [(), ((1.0,),), ((1.0, 0.0), (math.nan, 0.0)), ((1.0, 0.0, 0.0),)],
)
def test_bad_points_are_refused(points: tuple[tuple[float, ...], ...]) -> None:
    with pytest.raises(ValueError, match="points"):
        events(path(ORIGIN), points)  # type: ignore[arg-type]


# --- WaypointEvents invariants ---------------------------------------------------------------


def good_events(**changes: object) -> WaypointEvents:
    base = WaypointEvents(
        reached_step=(1, 2, None, None, None),
        docked=False,
        t_finish_s=None,
        d_min=0.5,
        max_hold_steps=0,
        waypoints=2,
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def test_events_are_frozen() -> None:
    e = good_events()
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.docked = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"reached_step": ()},
        {"reached_step": (None, 2, None, None, None)},  # out of order
        {"reached_step": (2, 2, None, None, None)},  # not strictly increasing
        {"reached_step": (0, 2, None, None, None)},  # row 0 is never checked
        {"waypoints": 3},
        {"docked": True},
        {"t_finish_s": 1.0},
        {"reached_step": (1, 2, 3, 4, 5), "docked": True, "waypoints": 5, "t_finish_s": math.nan},
        {"d_min": -0.1},
        {"d_min": math.inf},
        {"max_hold_steps": -1},
    ],
)
def test_inconsistent_events_are_refused(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="WaypointEvents"):
        good_events(**changes)


def test_docked_events_must_finish() -> None:
    e = good_events(
        reached_step=(1, 2, 3, 4, 8),
        docked=True,
        t_finish_s=0.08,
        waypoints=5,
        max_hold_steps=4,
    )
    assert e.docked
    with pytest.raises(ValueError, match="WaypointEvents"):
        dataclasses.replace(e, t_finish_s=None)


# --- fall_cause --------------------------------------------------------------------------


def test_fall_reasons_mirror_the_evaluator() -> None:
    from topple.evaluate import FALL_REASONS as EVALUATE_FALL_REASONS

    assert FALL_REASONS == EVALUATE_FALL_REASONS
    assert frozenset({"fell", "floor_contact", "sim_unstable"}) == TERMINATING_REASONS


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("fell", True),
        ("floor_contact", True),
        ("sim_unstable", True),
        ("timeout", False),
        ("error", False),
    ],
)
def test_env_terminated(reason: str, expected: bool) -> None:
    assert env_terminated(reason) is expected


@pytest.mark.parametrize("reason", ["fell", "floor_contact"])
@pytest.mark.parametrize(
    ("pushes", "expected"),
    [
        ((), "self"),
        ((500,), "push"),  # 0.5 s before
        ((350,), "push"),  # exactly 1.5 s before
        ((349,), "self"),  # 1.51 s before
        ((501,), "self"),  # after the fall
        ((200, 349), "self"),
        ((200, 349, 400), "push"),
    ],
)
def test_fall_cause_window(reason: str, pushes: tuple[int, ...], expected: str) -> None:
    assert fall_cause(reason, 500, pushes, DT) == expected


def test_a_push_on_the_fall_step_counts() -> None:
    assert fall_cause("fell", 500, (500,), DT) == "push"


@pytest.mark.parametrize("reason", ["timeout", "error", "sim_unstable"])
def test_fall_cause_is_none_without_a_fall(reason: str) -> None:
    assert fall_cause(reason, None, (100,), DT) == "none"
    assert fall_cause(reason, 120, (100,), DT) == "none"


def test_fall_cause_window_scales_with_control_dt() -> None:
    assert fall_cause("fell", 100, (25,), 0.02) == "push"  # 75 steps = 1.5 s
    assert fall_cause("fell", 100, (24,), 0.02) == "self"


def test_fall_cause_needs_the_fall_step() -> None:
    with pytest.raises(ValueError, match="fall_step"):
        fall_cause("fell", None, (), DT)
    with pytest.raises(ValueError, match="fall_step"):
        fall_cause("fell", -1, (), DT)
    with pytest.raises(ValueError, match="control_dt"):
        fall_cause("fell", 10, (), 0.0)
