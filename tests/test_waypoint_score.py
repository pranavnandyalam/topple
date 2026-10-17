"""waypoint-v1 score, progress and success (docs/design/phase-2.md §8.6): pure functions."""

from __future__ import annotations

import itertools
import math

import pytest
from pydantic import ValidationError

from topple.waypoint import (
    WAYPOINT_METRICS,
    WaypointEvents,
    WaypointScore,
    waypoint_progress,
    waypoint_score,
    waypoint_score_parts,
    waypoint_success,
)

L = 0.70
R = 0.10
HOLD = 100
EPISODE_S = 20.0
E_REF = 22.4
SCORE = WaypointScore(
    weights=(55, 25, 15, 5),
    t_par_s=5.0,
    dock_progress_weights=(0.8, 0.2),
    success_energy_factor=5,
    e_ref_j=E_REF,
)


def ev(
    g: int,
    *,
    d_min: float = 0.35,
    hold: int = 0,
    t_finish_s: float | None = None,
) -> WaypointEvents:
    """Events with g waypoints completed (g = 5 docks at t_finish_s, default 12.5 s)."""
    steps = [10 * (i + 1) for i in range(g)]
    if g == 5:
        t_finish_s = 12.5 if t_finish_s is None else t_finish_s
        steps[4] = round(t_finish_s / 0.01)
    return WaypointEvents(
        reached_step=tuple(steps + [None] * (5 - g)),
        docked=g == 5,
        t_finish_s=t_finish_s if g == 5 else None,
        d_min=d_min,
        max_hold_steps=HOLD if g == 5 else hold,
        waypoints=g,
    )


def progress(e: WaypointEvents) -> float:
    return waypoint_progress(
        e, leg_m=L, gate_radius_m=R, dock_hold_steps=HOLD, dock_progress_weights=(0.8, 0.2)
    )


def parts(
    e: WaypointEvents, *, t_alive_s: float = EPISODE_S, energy_j: float = 0.0, **kw: object
) -> dict[str, float]:
    return waypoint_score_parts(
        e,
        t_alive_s=t_alive_s,
        episode_s=EPISODE_S,
        energy_j=energy_j,
        score=SCORE,
        leg_m=L,
        gate_radius_m=R,
        dock_hold_steps=HOLD,
        **kw,  # type: ignore[arg-type]
    )


# --- WaypointScore ---------------------------------------------------------------------------


def test_score_config_dump_is_pinned() -> None:
    assert SCORE.model_dump(mode="json") == {
        "weights": [55.0, 25.0, 15.0, 5.0],
        "t_par_s": 5.0,
        "dock_progress_weights": [0.8, 0.2],
        "success_energy_factor": 5.0,
        "e_ref_j": 22.4,
    }
    assert list(SCORE.model_dump()) == [
        "weights",
        "t_par_s",
        "dock_progress_weights",
        "success_energy_factor",
        "e_ref_j",
    ]


def test_score_config_round_trips_through_json() -> None:
    assert WaypointScore.model_validate(SCORE.model_dump(mode="json")) == SCORE


def test_score_config_has_no_defaults() -> None:
    """Every scorer constant lives in challenge.yaml (and so in protocol_hash), none in code."""
    with pytest.raises(ValidationError):
        WaypointScore.model_validate({"e_ref_j": 22.4})
    assert all(f.is_required() for f in WaypointScore.model_fields.values())


def test_score_config_is_frozen() -> None:
    with pytest.raises(ValidationError):
        SCORE.t_par_s = 6.0  # type: ignore[misc]


def good_dump(**changes: object) -> dict[str, object]:
    dump: dict[str, object] = SCORE.model_dump(mode="json")
    dump.update(changes)
    return dump


@pytest.mark.parametrize(
    "changes",
    [
        {"weights": [55, 25, 15, 6]},  # sums to 101
        {"weights": [55, 25, 20, 0]},  # a zero weight
        {"weights": [60, 25, 20, -5]},
        {"weights": [55, 25, 20]},
        {"weights": [55, 25, 15, 5, 0]},
        {"weights": [55, 25, 15, math.inf]},
        {"t_par_s": 0.0},
        {"t_par_s": -1.0},
        {"t_par_s": math.nan},
        {"t_par_s": math.inf},
        {"dock_progress_weights": [0.8, 0.3]},
        {"dock_progress_weights": [1.0, 0.0]},
        {"dock_progress_weights": [0.8]},
        {"success_energy_factor": 0.0},
        {"success_energy_factor": math.inf},
        {"e_ref_j": 0.0},
        {"e_ref_j": -22.4},
        {"e_ref_j": math.nan},
        {"e_ref_j": math.inf},
        {"extra": 1},
    ],
)
def test_score_config_validation(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WaypointScore.model_validate(good_dump(**changes))


def test_score_config_accepts_yaml_integers() -> None:
    cfg = WaypointScore.model_validate(good_dump(weights=[55, 25, 15, 5], success_energy_factor=5))
    assert cfg == SCORE


# --- metrics vocabulary --------------------------------------------------------------------


def test_metric_vocabulary() -> None:
    assert WAYPOINT_METRICS == (
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
        "course_pts",
        "speed_pts",
        "alive_pts",
        "battery_pts",
    )
    assert len(set(WAYPOINT_METRICS)) == 18 <= 32  # contract §15's per-episode limit
    assert all(1 <= len(n) <= 64 and n.isprintable() for n in WAYPOINT_METRICS)


def test_score_parts_are_metric_names() -> None:
    assert list(parts(ev(5))) == ["course_pts", "speed_pts", "alive_pts", "battery_pts"]
    assert set(parts(ev(5))) <= set(WAYPOINT_METRICS)


# --- progress ----------------------------------------------------------------------------


def test_progress_is_zero_once_docked() -> None:
    assert progress(ev(5, d_min=0.0)) == 0.0


@pytest.mark.parametrize(
    ("d_min", "expected"),
    [(0.0, 1.0), (0.35, 0.5), (0.7, 0.0), (1.4, 0.0), (0.175, 0.75)],
)
@pytest.mark.parametrize("g", [0, 1, 2, 3])
def test_progress_toward_a_gate(g: int, d_min: float, expected: float) -> None:
    assert progress(ev(g, d_min=d_min)) == expected


@pytest.mark.parametrize(
    ("d_min", "hold", "expected"),
    [
        (0.7, 0, 0.0),  # as far as a leg: no closeness credit
        (1.5, 0, 0.0),  # farther still
        (0.4, 0, 0.8 * 0.5),  # halfway between L and r
        (0.1, 0, 0.8),  # at the radius: full closeness credit
        (0.0, 0, 0.8),
        (0.1, 50, 0.8 + 0.2 * 0.5),
        (0.05, 99, 0.8 + 0.2 * 0.99),
        (0.7, 50, 0.2 * 0.5),
    ],
)
def test_progress_toward_the_dock(d_min: float, hold: int, expected: float) -> None:
    assert progress(ev(4, d_min=d_min, hold=hold)) == pytest.approx(expected, abs=1e-15)


def test_dock_progress_is_monotone_and_below_one_until_docked() -> None:
    ds = [1.0, 0.7, 0.6, 0.4, 0.25, 0.1, 0.05, 0.0]
    holds = [0, 1, 10, 50, 98, 99]
    for d in ds:
        row = [progress(ev(4, d_min=d, hold=h)) for h in holds]
        assert row == sorted(row)
    for h in holds:
        col = [progress(ev(4, d_min=d, hold=h)) for d in ds]
        assert col == sorted(col)
    assert progress(ev(4, d_min=0.0, hold=99)) < 1.0
    for g, d, h in itertools.product(range(5), ds, holds):
        p = progress(ev(g, d_min=d, hold=h))
        assert 0.0 <= p <= 1.0  # 1 only for a gate at d_min 0, which reaching precludes
        assert p < 1.0 or g < 4
        assert g + p < 5


def test_progress_validates_its_course_parameters() -> None:
    with pytest.raises(ValueError, match="leg_m"):
        waypoint_progress(
            ev(4), leg_m=0.1, gate_radius_m=0.1, dock_hold_steps=HOLD, dock_progress_weights=(1, 0)
        )
    with pytest.raises(ValueError, match="dock_hold_steps"):
        waypoint_progress(
            ev(4), leg_m=L, gate_radius_m=R, dock_hold_steps=0, dock_progress_weights=(0.8, 0.2)
        )
    with pytest.raises(ValueError, match="max_hold_steps"):
        waypoint_progress(
            ev(4, hold=100),
            leg_m=L,
            gate_radius_m=R,
            dock_hold_steps=HOLD,
            dock_progress_weights=(0.8, 0.2),
        )


# --- score parts -------------------------------------------------------------------------


def test_a_perfect_run_scores_100() -> None:
    p = parts(ev(5, t_finish_s=5.0))
    assert p == {"course_pts": 55.0, "speed_pts": 25.0, "alive_pts": 15.0, "battery_pts": 5.0}
    assert waypoint_score(p) == 100.0


def test_finishing_faster_than_par_earns_no_more() -> None:
    assert parts(ev(5, t_finish_s=3.0))["speed_pts"] == 25.0


@pytest.mark.parametrize(
    ("t_finish", "speed"), [(5.0, 25.0), (12.5, 12.5), (20.0, 0.0), (18.5, 2.5)]
)
def test_speed_points(t_finish: float, speed: float) -> None:
    assert parts(ev(5, t_finish_s=t_finish))["speed_pts"] == pytest.approx(speed, abs=1e-12)


def test_one_second_faster_earns_25_over_15() -> None:
    a = parts(ev(5, t_finish_s=10.0))["speed_pts"]
    b = parts(ev(5, t_finish_s=11.0))["speed_pts"]
    assert a - b == pytest.approx(25 / 15, abs=1e-12)


def test_course_points() -> None:
    assert parts(ev(3, d_min=0.35))["course_pts"] == 38.5  # 55 * 3.5 / 5
    assert parts(ev(0, d_min=0.7))["course_pts"] == 0.0
    assert parts(ev(4, d_min=0.1, hold=50))["course_pts"] == pytest.approx(55 * 4.9 / 5, abs=1e-12)


def test_no_speed_points_without_docking() -> None:
    assert parts(ev(4, d_min=0.0, hold=99))["speed_pts"] == 0.0


@pytest.mark.parametrize(
    ("t_alive", "energy", "alive", "battery"),
    [
        (20.0, 0.0, 15.0, 5.0),
        (10.0, 0.0, 7.5, 2.5),
        (10.0, 11.2, 7.5, 1.25),  # E = E_ref / 2
        (20.0, 22.4, 15.0, 0.0),  # E = E_ref
        (20.0, 1000.0, 15.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_alive_and_battery_points(
    t_alive: float, energy: float, alive: float, battery: float
) -> None:
    p = parts(ev(2), t_alive_s=t_alive, energy_j=energy)
    assert p["alive_pts"] == alive
    assert p["battery_pts"] == battery


def test_the_parts_sum_to_the_score_and_stay_in_bounds() -> None:
    finishes = [5.0, 7.31, 12.5, 19.99, 20.0]
    alive = [0.0, 0.01, 3.33, 19.99, 20.0]
    energies = [0.0, 1.7, 22.4, 300.0]
    cases = [ev(g, d_min=d, hold=h) for g in range(5) for d in (0.0, 0.3, 0.9) for h in (0, 99)]
    cases += [ev(5, t_finish_s=t) for t in finishes]
    for e, t, energy in itertools.product(cases, alive, energies):
        p = parts(e, t_alive_s=t, energy_j=energy)
        score = waypoint_score(p)
        assert score == math.fsum(p.values())
        assert all(v >= 0.0 for v in p.values())
        assert p["course_pts"] <= 55.0
        assert p["speed_pts"] <= 25.0
        assert p["alive_pts"] <= 15.0
        assert p["battery_pts"] <= 5.0
        assert 0.0 <= score <= 100.0
        if not e.docked:
            assert score < 55.0 * 5 / 5 + 15.0 + 5.0


def test_errors_score_zero() -> None:
    """The caller signals an episode error with ``error=True``: every part is 0 (contract §12)."""
    p = parts(ev(5, t_finish_s=5.0), error=True)
    assert p == {"course_pts": 0.0, "speed_pts": 0.0, "alive_pts": 0.0, "battery_pts": 0.0}
    assert waypoint_score(p) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"t_alive_s": -0.01}, "t_alive_s"),
        ({"t_alive_s": 20.5}, "t_alive_s"),
        ({"t_alive_s": math.nan}, "t_alive_s"),
        ({"energy_j": -1.0}, "energy_j"),
        ({"energy_j": math.inf}, "energy_j"),
        ({"energy_j": math.nan}, "energy_j"),
    ],
)
def test_score_parts_validate_inputs(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parts(ev(2), **kwargs)  # type: ignore[arg-type]


def test_par_must_be_shorter_than_the_episode() -> None:
    with pytest.raises(ValueError, match="t_par_s"):
        waypoint_score_parts(
            ev(5),
            t_alive_s=5.0,
            episode_s=5.0,
            energy_j=0.0,
            score=SCORE,
            leg_m=L,
            gate_radius_m=R,
            dock_hold_steps=HOLD,
        )


def test_score_needs_exactly_the_four_parts() -> None:
    with pytest.raises(ValueError, match="parts"):
        waypoint_score({"course_pts": 1.0})


# --- success -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("events", "survived", "energy", "expected"),
    [
        (ev(5), True, 0.0, True),
        (ev(5), True, 5 * E_REF, True),  # exactly the energy cap
        (ev(5), True, math.nextafter(5 * E_REF, math.inf), False),
        (ev(5), False, 0.0, False),
        (ev(4, d_min=0.0, hold=99), True, 0.0, False),
        (ev(0), True, 0.0, False),
    ],
)
def test_success(events: WaypointEvents, survived: bool, energy: float, expected: bool) -> None:
    assert waypoint_success(events, survived=survived, energy_j=energy, score=SCORE) is expected


def test_an_error_is_never_a_success() -> None:
    assert not waypoint_success(ev(5), survived=True, energy_j=0.0, score=SCORE, error=True)
