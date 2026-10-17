"""Course generation for waypoint-v1 (docs/design/phase-2.md §8.3, contract §9)."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from topple.course import (
    Course,
    CourseConfig,
    _make_course_counted,
    course_points,
    make_course,
    pcos,
    psin,
)
from topple.rng import MAX_SEED, Stream, stream_rng

WAYPOINT_V1: dict[str, Any] = {
    "n_waypoints": 5,
    "leg_m": 0.70,
    "turn_deg": (30.0, 150.0),
    "max_radius_m": 1.5,
    "max_attempts": 1000,
    "gate_radius_m": 0.10,
    "dock_hold_steps": 100,
}
CFG = CourseConfig(**WAYPOINT_V1)


def _cfg(**overrides: Any) -> CourseConfig:
    return CourseConfig(**{**WAYPOINT_V1, **overrides})


# ---------------------------------------------------------------------------------------------
# Portable trigonometry


def test_pcos_psin_exact_at_zero() -> None:
    assert pcos(0.0) == 1.0
    assert psin(0.0) == 0.0
    assert pcos(-0.0) == 1.0


def test_pcos_psin_agree_with_libm_on_bounded_angles() -> None:
    grid = np.linspace(-14.0, 14.0, 200_001).tolist()
    special = [k * math.pi / 4 for k in range(-17, 18)] + [math.pi, -math.pi, 2 * math.pi]
    worst = 0.0
    for theta in grid + special:
        worst = max(worst, abs(pcos(theta) - math.cos(theta)), abs(psin(theta) - math.sin(theta)))
    assert worst <= 2e-15


# Known answers: exact doubles, so a platform whose float arithmetic drifts fails here.
_TRIG_PINS: dict[float, tuple[str, str]] = {
    1e-08: ("0x1.0000000000000p+0", "0x1.5798ee2308c3ap-27"),
    0.5: ("0x1.c1528065b7d50p-1", "0x1.eaee8744b05f0p-2"),
    1.0: ("0x1.14a280fb5068cp-1", "0x1.aed548f090ceep-1"),
    -1.5707963267948966: ("0x0.0p+0", "-0x1.0000000000000p+0"),
    -2.5: ("-0x1.9a2f7ef858b7ep-1", "-0x1.326af0dcfcab2p-1"),
    3.0: ("-0x1.fae04be85e5d4p-1", "0x1.210386db6d56cp-3"),
    3.141592653589793: ("-0x1.0000000000000p+0", "0x1.921fb54442d18p-52"),
    6.0: ("0x1.eb9b7097822f6p-1", "-0x1.1e1f18ab0a2bcp-2"),
    10.0: ("-0x1.ad9ac890c6b1ap-1", "-0x1.1689ef5f34f59p-1"),
    -13.5: ("0x1.309970f142df9p-1", "-0x1.9b89a1e2258b3p-1"),
}


@pytest.mark.parametrize("theta", sorted(_TRIG_PINS))
def test_pcos_psin_known_answers(theta: float) -> None:
    cos_hex, sin_hex = _TRIG_PINS[theta]
    assert pcos(theta).hex() == cos_hex
    assert psin(theta).hex() == sin_hex


@pytest.mark.parametrize("theta", [math.inf, -math.inf, math.nan])
def test_pcos_psin_refuse_non_finite(theta: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        pcos(theta)
    with pytest.raises(ValueError, match="finite"):
        psin(theta)


# ---------------------------------------------------------------------------------------------
# Config


def test_config_defaults_and_frozen() -> None:
    cfg = CourseConfig(**{k: v for k, v in WAYPOINT_V1.items() if k != "n_waypoints"})
    assert cfg.n_waypoints == 5
    with pytest.raises(ValidationError):
        cfg.leg_m = 1.0


def test_config_task_extras_are_str_to_float() -> None:
    extras = CFG.task_extras()
    assert extras == {
        "n_waypoints": 5.0,
        "leg_m": 0.70,
        "turn_deg_min": 30.0,
        "turn_deg_max": 150.0,
        "max_radius_m": 1.5,
        "max_attempts": 1000.0,
        "gate_radius_m": 0.10,
        "dock_hold_steps": 100.0,
    }
    assert all(type(k) is str and type(v) is float for k, v in extras.items())


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_waypoints": 4},
        {"n_waypoints": 6},
        {"n_waypoints": True},
        {"leg_m": 0.0},
        {"leg_m": -0.7},
        {"leg_m": math.inf},
        {"leg_m": math.nan},
        {"leg_m": True},
        {"turn_deg": (150.0, 30.0)},
        {"turn_deg": (30.0, 30.0)},
        {"turn_deg": (0.0, 150.0)},
        {"turn_deg": (30.0, 180.0)},
        {"turn_deg": (30.0,)},
        {"max_radius_m": 0.0},
        {"max_attempts": 0},
        {"max_attempts": 1.5},
        {"gate_radius_m": -0.1},
        {"dock_hold_steps": 0},
        {"unknown": 1.0},
    ],
)
def test_config_refuses_bad_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _cfg(**overrides)


# ---------------------------------------------------------------------------------------------
# make_course / course_points known answers

_COURSE_PINS: dict[int, dict[str, Any]] = {
    0: {  # accepted on attempt 1
        "h0": "0x1.9348665c52b9cp+0",
        "turns": [
            "0x1.7f7088664cd2ep+0",
            "0x1.01c9e0fea292cp+1",
            "-0x1.9ac603ae7efe6p+0",
            "0x1.7ef5d91992036p+0",
        ],
        "points": [
            ("-0x1.9f5df7e6e88ccp-9", "0x1.666575b44ca26p-1"),
            ("-0x1.672edae658a54p-1", "0x1.7ee9a1b90c33dp-1"),
            ("-0x1.c80270416ee47p-2", "0x1.8b0de38136838p-4"),
            ("-0x1.1ae367b90a5d6p+0", "-0x1.19d172f2689dbp-3"),
            ("-0x1.d7897a9185fd1p-1", "-0x1.a03e22997fe03p-1"),
        ],
    },
    3000: {  # accepted on attempt 2
        "h0": "-0x1.4ea7fb082b634p-1",
        "turns": [
            "0x1.c31f3b9ff312ep-1",
            "-0x1.0a6c7483e0e3ep+1",
            "-0x1.9db5f877e9a70p+0",
            "-0x1.dc69df676da04p+0",
        ],
        "points": [
            ("0x1.1c8739f65175dp-1", "-0x1.b3dd0e3fa03edp-2"),
            ("0x1.3cd90bfd8f5d5p+0", "-0x1.1236bf4e6c306p-2"),
            ("0x1.0ac7a63a2db3fp+0", "-0x1.e13bdf77936f6p-1"),
            ("0x1.84a1cb1e1219ep-2", "-0x1.6da14e6c758a0p-1"),
            ("0x1.92282511b4e34p-1", "-0x1.2698f0cc79a28p-3"),
        ],
    },
    MAX_SEED - 1: {  # accepted on attempt 4
        "h0": "0x1.ff1afd858c1d8p-1",
        "turns": [
            "0x1.45f1459f6672bp+1",
            "-0x1.d0ea10973fccep+0",
            "-0x1.02df42c67eac5p+1",
            "-0x1.c2e70c91f4628p-1",
        ],
        "points": [
            ("0x1.84578a7881904p-2", "0x1.2d3e9149444efp-1"),
            ("-0x1.0f029601a83c4p-2", "0x1.41515f2563d21p-2"),
            ("-0x1.7fa8e240dc5b9p-2", "0x1.014d7c79e56cep+0"),
            ("0x1.2e6b4a5c54845p-2", "0x1.9ace10595ed01p-1"),
            ("0x1.218da625b5123p-1", "0x1.40ba4fd0c2a5cp-3"),
        ],
    },
}
_ATTEMPTS = {0: 1, 3000: 2, MAX_SEED - 1: 4}


@pytest.mark.parametrize("seed", sorted(_COURSE_PINS))
def test_make_course_known_answers(seed: int) -> None:
    pin = _COURSE_PINS[seed]
    course, attempts = _make_course_counted(seed, CFG)
    assert make_course(seed, CFG) == course
    assert attempts == _ATTEMPTS[seed]
    assert course.h0.hex() == pin["h0"]
    assert [t.hex() for t in course.turns] == pin["turns"]
    assert [(x.hex(), y.hex()) for x, y in course_points(course, CFG)] == pin["points"]


def test_known_answer_seeds_are_pinned() -> None:
    assert set(_COURSE_PINS) == {0, 3000, MAX_SEED - 1}


def test_make_course_is_deterministic_and_returns_floats() -> None:
    a = make_course(17, CFG)
    b = make_course(17, CFG)
    assert a == b
    assert type(a.h0) is float
    assert isinstance(a.turns, tuple)
    assert all(type(t) is float for t in a.turns)
    assert len(a.turns) == CFG.n_waypoints - 1


def test_course_points_geometry() -> None:
    course = make_course(5, CFG)
    points = course_points(course, CFG)
    assert len(points) == CFG.n_waypoints
    prev = (0.0, 0.0)
    for x, y in points:
        assert math.hypot(x - prev[0], y - prev[1]) == pytest.approx(CFG.leg_m, abs=1e-14)
        prev = (x, y)


def _libm_course(seed: int, cfg: CourseConfig) -> tuple[Course, int]:
    """Independent reference: libm trigonometry, numpy bulk draws (same raw doubles)."""
    rng = stream_rng(seed, Stream.TARGETS)
    lo, hi = (d * (math.pi / 180.0) for d in cfg.turn_deg)
    n = cfg.n_waypoints - 1
    for attempt in range(1, cfg.max_attempts + 1):
        u = rng.random(1 + n).tolist()
        h0 = -math.pi + (math.pi - -math.pi) * u[0]
        mags = [lo + (hi - lo) * v for v in u[1:]]
        signs = rng.integers(0, 2, size=n).tolist()
        turns = tuple(m if s else -m for m, s in zip(mags, signs, strict=True))
        theta, x, y, ok = h0, 0.0, 0.0, True
        for k in range(cfg.n_waypoints):
            if k:
                theta += turns[k - 1]
            x += cfg.leg_m * math.cos(theta)
            y += cfg.leg_m * math.sin(theta)
            ok = ok and x * x + y * y <= cfg.max_radius_m * cfg.max_radius_m
        if ok:
            return Course(h0=h0, turns=turns), attempt
    raise AssertionError("reference exhausted attempts")


def test_same_courses_as_libm_reference() -> None:
    for seed in range(3000):
        course, attempts = _make_course_counted(seed, CFG)
        ref, ref_attempts = _libm_course(seed, CFG)
        assert course == ref, seed
        assert attempts == ref_attempts, seed
        libm_points = _points_libm(ref, CFG)
        for (x, y), (rx, ry) in zip(course_points(course, CFG), libm_points, strict=True):
            assert abs(x - rx) <= 1e-14
            assert abs(y - ry) <= 1e-14


def _points_libm(course: Course, cfg: CourseConfig) -> list[tuple[float, float]]:
    theta, x, y, out = course.h0, 0.0, 0.0, []
    for k in range(cfg.n_waypoints):
        if k:
            theta += course.turns[k - 1]
        x += cfg.leg_m * math.cos(theta)
        y += cfg.leg_m * math.sin(theta)
        out.append((x, y))
    return out


def test_acceptance_rate_and_mean_turn() -> None:
    seeds = range(2000)
    attempts = 0
    turns: list[float] = []
    lo, hi = (math.radians(d) for d in CFG.turn_deg)
    for seed in seeds:
        course, n = _make_course_counted(seed, CFG)
        attempts += n
        turns.extend(abs(t) for t in course.turns)
        assert all(lo <= abs(t) <= hi for t in course.turns)
        assert abs(course.h0) <= math.pi
        for x, y in course_points(course, CFG):
            assert x * x + y * y <= CFG.max_radius_m**2
    rate = len(seeds) / attempts
    mean_turn_deg = math.degrees(sum(turns) / len(turns))
    assert 0.41 <= rate <= 0.47
    assert 96.0 <= mean_turn_deg <= 100.5


def test_both_turn_signs_and_points_inside_radius() -> None:
    signs: set[bool] = set()
    for seed in range(200):
        course = make_course(seed, CFG)
        signs.update(t > 0 for t in course.turns)
        assert all(math.hypot(x, y) <= 1.5 + 1e-12 for x, y in course_points(course, CFG))
    assert signs == {True, False}


def test_make_course_raises_when_radius_impossible() -> None:
    cfg = _cfg(max_radius_m=0.1, max_attempts=3)
    with pytest.raises(RuntimeError, match="3 attempts"):
        make_course(0, cfg)


def test_make_course_validates_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        make_course(-1, CFG)
    with pytest.raises(ValueError, match="seed"):
        make_course(MAX_SEED, CFG)


# ---------------------------------------------------------------------------------------------
# Serialization


def test_to_dict_round_trips_through_json() -> None:
    course = make_course(3000, CFG)
    data = json.loads(json.dumps(course.to_dict()))
    assert set(data) == {"h0", "turns"}
    assert Course.from_dict(data, CFG) == course


def _good() -> dict[str, Any]:
    return {"h0": 0.5, "turns": [0.6, -1.0, 2.0, -2.5]}


def test_from_dict_accepts_tuple_turns_and_bounds() -> None:
    lo, hi = (d * (math.pi / 180.0) for d in CFG.turn_deg)
    course = Course.from_dict({"h0": math.pi, "turns": (lo, -hi, hi, -lo)}, CFG)
    assert course.turns == (lo, -hi, hi, -lo)
    assert Course.from_dict({"h0": -math.pi, "turns": [lo] * 4}, CFG).h0 == -math.pi


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(turns=d["turns"][:3]),
        lambda d: d.update(turns=[*d["turns"], 1.0]),
        lambda d: d.update(turns=[]),
        lambda d: d["turns"].__setitem__(0, 0.1),  # below 30 deg
        lambda d: d["turns"].__setitem__(1, -2.7),  # beyond 150 deg
        lambda d: d["turns"].__setitem__(2, 0.0),
        lambda d: d.update(h0=3.2),
        lambda d: d.update(h0=-3.2),
        lambda d: d.update(h0=math.nan),
        lambda d: d.update(h0=math.inf),
        lambda d: d.update(h0=1),
        lambda d: d.update(h0=True),
        lambda d: d.update(h0="0.5"),
        lambda d: d.update(h0=None),
        lambda d: d["turns"].__setitem__(0, 1),
        lambda d: d["turns"].__setitem__(0, True),
        lambda d: d["turns"].__setitem__(0, math.nan),
        lambda d: d["turns"].__setitem__(0, np.float64(1.0)),
        lambda d: d.update(turns="abcd"),
        lambda d: d.update(turns={"a": 1.0}),
        lambda d: d.update(turns=None),
        lambda d: d.pop("h0"),
        lambda d: d.update(extra=1.0),
        lambda d: d.update({1: 2.0}),
    ],
)
def test_from_dict_refuses(mutate: Any) -> None:
    data = _good()
    assert Course.from_dict(data, CFG).h0 == 0.5
    mutate(data)
    with pytest.raises(ValueError, match="invalid course"):
        Course.from_dict(data, CFG)


@pytest.mark.parametrize("data", [None, [], "course", 1.0, (0.5, [1.0] * 4)])
def test_from_dict_refuses_non_mapping(data: object) -> None:
    with pytest.raises(ValueError, match="invalid course"):
        Course.from_dict(data, CFG)
