"""Waypoint courses for waypoint-v1 (docs/design/phase-2.md §8.3, contract §9).

A course is ``n_waypoints`` points ``p_0 .. p_{n-1}`` in the start frame (origin = axle midpoint at
reset, x = initial heading), reached by legs of exactly ``leg_m`` from ``p_{-1} = (0, 0)``. The
first heading is ``h0`` and each later leg turns by a signed angle of magnitude in ``turn_deg``.
Waypoints ``0 .. n-2`` are gates, ``n-1`` is the dock.

Everything here is portable bit for bit: draws use one raw ``Generator.random()`` double and an
affine map in Python (never ``Generator.uniform``), and the trigonometry is ``pcos``/``psin``,
built from correctly rounded float ``+``, ``-`` and ``*`` only, never libm's or numpy's. Acceptance
decides how many attempts the TARGETS stream consumes, so it must come out the same everywhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, PositiveFloat, StrictInt, field_validator

from topple.rng import Stream, stream_rng

# --------------------------------------------------------------------------------------------
# Portable trigonometry

_TWO_PI = 2.0 * math.pi
_TWO_PI_HI = 6.28125  # 201/32: k * _TWO_PI_HI is exact for any |k| < 2**45
_TWO_PI_LO = _TWO_PI - _TWO_PI_HI  # exact (Sterbenz)
_N_TERMS = 16
# (-1)^n / (2n)! and (-1)^n / (2n+1)!: int / int true division is correctly rounded.
_COS_COEFFS = tuple((-1) ** n / math.factorial(2 * n) for n in range(_N_TERMS))
_SIN_COEFFS = tuple((-1) ** n / math.factorial(2 * n + 1) for n in range(_N_TERMS))


def _reduce(theta: float) -> float:
    """``theta`` minus the nearest multiple of 2π, as ``r`` in about [-π, π]."""
    if not math.isfinite(theta):
        raise ValueError(f"angle must be finite, got {theta!r}")
    k = round(theta / _TWO_PI)
    return (theta - k * _TWO_PI_HI) - k * _TWO_PI_LO


def _horner(coeffs: tuple[float, ...], s: float) -> float:
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        acc = acc * s + c
    return acc


def pcos(theta: float) -> float:
    """Portable cosine: the same double on every IEEE-754 platform (Taylor through r^30)."""
    r = _reduce(float(theta))
    return _horner(_COS_COEFFS, r * r)


def psin(theta: float) -> float:
    """Portable sine: the same double on every IEEE-754 platform (Taylor through r^31)."""
    r = _reduce(float(theta))
    return r * _horner(_SIN_COEFFS, r * r)


def _rad(deg: float) -> float:
    return deg * (math.pi / 180.0)


# --------------------------------------------------------------------------------------------
# Config


class CourseConfig(BaseModel):
    """The ``course`` block of a waypoint challenge.yaml (design §8.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    n_waypoints: StrictInt = 5
    leg_m: PositiveFloat
    turn_deg: tuple[float, float]
    max_radius_m: PositiveFloat
    max_attempts: StrictInt
    gate_radius_m: PositiveFloat
    dock_hold_steps: StrictInt

    @field_validator("*", mode="before")
    @classmethod
    def _no_bools(cls, value: Any) -> Any:
        items = value if isinstance(value, (tuple, list)) else (value,)
        if any(isinstance(v, bool) for v in items):
            raise ValueError("booleans are not numbers here")
        return value

    @field_validator("n_waypoints")
    @classmethod
    def _n_waypoints(cls, value: int) -> int:
        if value != 5:
            raise ValueError("waypoint courses have exactly 5 waypoints in v1")
        return value

    @field_validator("turn_deg")
    @classmethod
    def _turn_deg(cls, value: tuple[float, float]) -> tuple[float, float]:
        lo, hi = value
        if not 0.0 < lo < hi < 180.0:
            raise ValueError(f"turn_deg must satisfy 0 < min < max < 180, got {value}")
        return value

    @field_validator("max_attempts", "dock_hold_steps")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"must be a positive integer, got {value}")
        return value

    def task_extras(self) -> dict[str, float]:
        """The course constants as ``TaskSpec.extras`` (contract §8: str -> float only)."""
        return {
            "n_waypoints": float(self.n_waypoints),
            "leg_m": float(self.leg_m),
            "turn_deg_min": float(self.turn_deg[0]),
            "turn_deg_max": float(self.turn_deg[1]),
            "max_radius_m": float(self.max_radius_m),
            "max_attempts": float(self.max_attempts),
            "gate_radius_m": float(self.gate_radius_m),
            "dock_hold_steps": float(self.dock_hold_steps),
        }


# --------------------------------------------------------------------------------------------
# Course

_INVALID = "invalid course: "
_KEYS = frozenset({"h0", "turns"})


def _exact_float(name: str, value: object) -> float:
    if type(value) is not float:
        raise ValueError(f"{name} must be a float, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


@dataclass(frozen=True)
class Course:
    """The accepted draws only: first heading ``h0`` and signed ``turns`` (radians)."""

    h0: float
    turns: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {"h0": self.h0, "turns": list(self.turns)}

    @classmethod
    def from_dict(cls, data: object, cfg: CourseConfig) -> Course:
        """Parse and validate JSON data; any problem raises ``ValueError``."""
        try:
            return _parse(data, cfg)
        except ValueError as exc:
            raise ValueError(f"{_INVALID}{exc}") from exc


def _parse(data: object, cfg: CourseConfig) -> Course:
    if not isinstance(data, Mapping):
        raise ValueError(f"course must be an object, got {type(data).__name__}")
    if not all(isinstance(key, str) for key in data):
        raise ValueError("keys must be strings")
    if set(data) != _KEYS:
        raise ValueError(f"keys must be exactly {sorted(_KEYS)}, got {sorted(data)}")
    h0 = _exact_float("h0", data["h0"])
    if abs(h0) > math.pi:
        raise ValueError(f"|h0| must be <= pi, got {h0!r}")
    raw = data["turns"]
    n = cfg.n_waypoints - 1
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != n:
        raise ValueError(f"turns must be a list of {n} floats")
    lo, hi = _rad(cfg.turn_deg[0]), _rad(cfg.turn_deg[1])
    turns = tuple(_exact_float(f"turns[{i}]", t) for i, t in enumerate(raw))
    for i, t in enumerate(turns):
        if not lo <= abs(t) <= hi:
            raise ValueError(f"|turns[{i}]| must be in [{lo!r}, {hi!r}], got {t!r}")
    return Course(h0=h0, turns=turns)


def course_points(course: Course, cfg: CourseConfig) -> tuple[tuple[float, float], ...]:
    """The ``n_waypoints`` points in the start frame, with portable trigonometry only."""
    theta = course.h0
    x = y = 0.0
    points = []
    for k in range(cfg.n_waypoints):
        if k:
            theta = theta + course.turns[k - 1]
        x = x + cfg.leg_m * pcos(theta)
        y = y + cfg.leg_m * psin(theta)
        points.append((x, y))
    return tuple(points)


def _uniforms(rng: Any, low: float, high: float, n: int) -> list[float]:
    """Contract §9 portable draws: one raw double each, affine map in Python (never fused)."""
    return [low + (high - low) * u for u in rng.random(n).tolist()]


def _inside(points: tuple[tuple[float, float], ...], radius: float) -> bool:
    r2 = radius * radius
    return all(x * x + y * y <= r2 for x, y in points)


def _make_course_counted(seed: int, cfg: CourseConfig) -> tuple[Course, int]:
    """``make_course`` plus the number of attempts it took (1-based)."""
    rng = stream_rng(seed, Stream.TARGETS)
    n = cfg.n_waypoints - 1
    lo, hi = _rad(cfg.turn_deg[0]), _rad(cfg.turn_deg[1])
    for attempt in range(1, cfg.max_attempts + 1):
        h0 = _uniforms(rng, -math.pi, math.pi, 1)[0]
        mags = _uniforms(rng, lo, hi, n)
        signs = rng.integers(0, 2, size=n).tolist()
        turns = tuple(m if s else -m for m, s in zip(mags, signs, strict=True))
        course = Course(h0=h0, turns=turns)
        if _inside(course_points(course, cfg), cfg.max_radius_m):
            return course, attempt
    raise RuntimeError(
        f"no course within {cfg.max_radius_m} m after {cfg.max_attempts} attempts (seed {seed})"
    )


def make_course(seed: int, cfg: CourseConfig) -> Course:
    """The seed's course: rejection sampling on the TARGETS stream (design §8.3)."""
    return _make_course_counted(seed, cfg)[0]
