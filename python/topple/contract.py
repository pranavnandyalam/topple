"""The controller contract (docs/contract.md §6-8).

Everything a controller touches lives here. This module imports only the standard library
(numpy is imported lazily inside ``to_array``/``from_array``) so the subprocess runner, Pyodide
and lesson-1 submissions can import it cheaply.
"""

from __future__ import annotations

import math
import numbers
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

OBS_SCHEMA_ID = "balancer.obs.v1"
FEATURES: tuple[str, ...] = (
    "pitch",
    "pitch_rate",
    "yaw",
    "yaw_rate",
    "wheel_pos_l",
    "wheel_pos_r",
    "wheel_vel_l",
    "wheel_vel_r",
    "last_action_l",
    "last_action_r",
)
_WIRE = struct.Struct("<11d")  # t + FEATURES, little-endian float64
SATURATION_THRESHOLD = 0.99


def _as_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _as_pair(name: str, value: object) -> tuple[float, float]:
    if not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a pair of numbers")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a pair of numbers") from exc
    if len(items) != 2:
        raise ValueError(f"{name} must have exactly 2 values, got {len(items)}")
    return (_as_float(name, items[0]), _as_float(name, items[1]))


@dataclass(frozen=True, slots=True)
class BalancerObs:
    """What a balance-family controller sees each control step (``balancer.obs.v1``)."""

    t: float
    pitch: float
    pitch_rate: float
    yaw: float
    yaw_rate: float
    wheel_pos: tuple[float, float]
    wheel_vel: tuple[float, float]
    last_action: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("t", "pitch", "pitch_rate", "yaw", "yaw_rate"):
            object.__setattr__(self, name, _as_float(name, getattr(self, name)))
        for name in ("wheel_pos", "wheel_vel", "last_action"):
            object.__setattr__(self, name, _as_pair(name, getattr(self, name)))

    def features(self) -> tuple[float, ...]:
        return (
            self.pitch,
            self.pitch_rate,
            self.yaw,
            self.yaw_rate,
            *self.wheel_pos,
            *self.wheel_vel,
            *self.last_action,
        )

    def to_array(self) -> npt.NDArray[np.float32]:
        """Feature vector in ``FEATURES`` order, float32 (the neural-network input)."""
        import numpy as np

        return np.asarray(self.features(), dtype=np.float32)

    @classmethod
    def from_array(cls, arr: Sequence[float] | npt.NDArray[Any], t: float) -> BalancerObs:
        values = [float(v) for v in arr]
        if len(values) != len(FEATURES):
            raise ValueError(f"expected {len(FEATURES)} features, got {len(values)}")
        p, pr, y, yr, wpl, wpr, wvl, wvr, lal, lar = values
        return cls(t, p, pr, y, yr, (wpl, wpr), (wvl, wvr), (lal, lar))

    def to_wire(self) -> bytes:
        """Lossless 88-byte encoding: ``t`` then the features as little-endian float64."""
        return _WIRE.pack(self.t, *self.features())

    @classmethod
    def from_wire(cls, data: bytes) -> BalancerObs:
        if len(data) != _WIRE.size:
            raise ValueError(f"expected {_WIRE.size} bytes, got {len(data)}")
        t, *features = _WIRE.unpack(data)
        p, pr, y, yr, wpl, wpr, wvl, wvr, lal, lar = features
        return cls(t, p, pr, y, yr, (wpl, wpr), (wvl, wvr), (lal, lar))


Observation = BalancerObs  # permanent alias for obs_v1 submissions


class ActionError(ValueError):
    """The controller returned something that is not a valid action."""


@dataclass(frozen=True, slots=True)
class ActionSpec:
    id: str
    names: tuple[str, ...]
    low: float = -1.0
    high: float = 1.0

    @property
    def dim(self) -> int:
        return len(self.names)


BALANCE_V1_ACTION = ActionSpec(id="diffdrive.effort.v1", names=("left", "right"))


def validate_action(action: object, dim: int) -> tuple[float, ...]:
    """Return the action as a tuple of finite floats, or raise ``ActionError``.

    Accepts a list, tuple or 1-D numpy array of real numbers. Rejects bools, strings, complex
    numbers, NaN/inf and the wrong length. Does not clip.
    """
    if isinstance(action, (str, bytes, Mapping)) or action is None:
        raise ActionError(
            f"action must be a sequence of {dim} numbers, got {type(action).__name__}"
        )
    ndim = getattr(action, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ActionError(f"action must be 1-dimensional, got ndim={ndim}")
    try:
        items = list(action)  # type: ignore[call-overload]
    except TypeError as exc:
        raise ActionError(f"action must be a sequence of {dim} numbers") from exc
    if len(items) != dim:
        raise ActionError(f"action must have exactly {dim} values, got {len(items)}")
    values = []
    for i, item in enumerate(items):
        if isinstance(item, bool) or type(item).__name__ == "bool_":
            raise ActionError(f"action[{i}] is a bool, expected a number")
        if not isinstance(item, numbers.Real):
            raise ActionError(f"action[{i}] must be a real number, got {type(item).__name__}")
        value = float(item)
        if not math.isfinite(value):
            raise ActionError(f"action[{i}] is not finite: {value}")
        values.append(value)
    return tuple(values)


def clip_action(
    values: Sequence[float], low: float = -1.0, high: float = 1.0
) -> tuple[tuple[float, ...], bool]:
    """Clip to ``[low, high]``; also report saturation (any |u| >= 0.99 after clipping)."""
    clipped = tuple(min(max(v, low), high) for v in values)
    saturated = any(abs(v) >= SATURATION_THRESHOLD for v in clipped)
    return clipped, saturated


@dataclass(frozen=True, slots=True)
class RobotGeometry:
    wheel_radius_m: float
    track_width_m: float


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Everything a controller may know about its task, handed to ``make_controller``."""

    challenge: str
    obs_schema: str
    obs_features: tuple[str, ...]
    action: ActionSpec
    control_dt: float
    episode_seconds: float
    robot: RobotGeometry
    extras: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        steps = self.episode_seconds / self.control_dt
        if abs(steps - round(steps)) > 1e-9 or round(steps) < 1:
            raise ValueError(
                f"episode_seconds ({self.episode_seconds}) must be a positive multiple of "
                f"control_dt ({self.control_dt})"
            )

    @property
    def max_steps(self) -> int:
        return round(self.episode_seconds / self.control_dt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenge": self.challenge,
            "obs_schema": self.obs_schema,
            "obs_features": list(self.obs_features),
            "action": {
                "id": self.action.id,
                "names": list(self.action.names),
                "low": self.action.low,
                "high": self.action.high,
            },
            "control_dt": self.control_dt,
            "episode_seconds": self.episode_seconds,
            "robot": {
                "wheel_radius_m": self.robot.wheel_radius_m,
                "track_width_m": self.robot.track_width_m,
            },
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskSpec:
        action = data["action"]
        return cls(
            challenge=data["challenge"],
            obs_schema=data["obs_schema"],
            obs_features=tuple(data["obs_features"]),
            action=ActionSpec(
                id=action["id"],
                names=tuple(action["names"]),
                low=action["low"],
                high=action["high"],
            ),
            control_dt=data["control_dt"],
            episode_seconds=data["episode_seconds"],
            robot=RobotGeometry(**data["robot"]),
            extras=dict(data.get("extras", {})),
        )


ObsT_contra = TypeVar("ObsT_contra", contravariant=True)


@runtime_checkable
class Controller(Protocol[ObsT_contra]):
    """A controller: ``reset`` once per episode, then ``act`` at ``1/control_dt`` Hz."""

    def reset(self, rng_seed: int) -> None: ...

    def act(self, obs: ObsT_contra) -> Sequence[float]: ...
