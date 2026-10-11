"""The controller contract (docs/contract.md §6-8).

Everything a controller touches lives here. This module imports only the standard library
(numpy is imported lazily inside ``to_array``/``controller_obs``) so the subprocess runner,
Pyodide and lesson-1 submissions can import it cheaply.
"""

from __future__ import annotations

import itertools
import math
import numbers
import struct
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, field
from types import MappingProxyType
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


def controller_obs(obs_arr: Sequence[float] | npt.NDArray[Any], t: float) -> BalancerObs:
    """The observation a controller receives: the only way a harness builds it (contract §6).

    The 10 features are quantized to float32 (what the env emits and the trajectory stores), then
    widened exactly to Python floats; ``t`` is kept as given. ``to_wire`` carries these exact
    values, so in-process and out-of-process controllers see identical bits.
    """
    import numpy as np

    with np.errstate(over="ignore"):  # a float64 beyond float32 range becomes inf, as in the env
        arr = np.asarray(obs_arr, dtype=np.float32)
    if arr.shape != (len(FEATURES),):
        raise ValueError(f"expected {len(FEATURES)} features of shape (10,), got shape {arr.shape}")
    return BalancerObs.from_array(arr, t)


_TYPE_NAME = type.__dict__["__name__"]  # type's own descriptor: a metaclass cannot shadow it


def class_name(value: object) -> str:
    """The real name of ``value``'s class as an exact ``str``, running none of its code.

    ``type(value)`` never consults ``value.__class__``; ``type``'s own ``__name__`` descriptor
    bypasses a metaclass ``__name__`` property; and ``str.__str__`` copies a ``str`` subclass to
    an exact ``str``, so the result's methods are ``str``'s own. Untrusted objects and their
    exceptions are only ever named through this (contract §8).
    """
    return str.__str__(_TYPE_NAME.__get__(type(value), type))


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
    """Return the action as a tuple of ``dim`` finite floats, or raise ``ActionError``.

    Accepts any ordered, re-iterable container of real numbers: list, tuple, 1-D numpy array,
    ``range``, ``array.array``, ``memoryview``. Rejects strings and bytes, mappings, sets,
    one-shot iterators and generators (never advanced, so their code never runs), bools, complex
    numbers, values that do not fit a float (huge ints), NaN/inf and the wrong length. At most
    ``dim + 1`` items are read, so an endless container cannot hang the harness. Any ``Exception``
    raised by the object while it is inspected, iterated or converted becomes ``ActionError``;
    ``BaseException``s that are not ``Exception`` (``KeyboardInterrupt``, ``SystemExit``)
    propagate. Classes are named only through ``class_name``, so no metaclass or ``str``
    subclass code runs while a message is built. Does not clip.
    """
    try:
        return _read_action(action, dim)
    except ActionError:
        raise
    except Exception as exc:
        raise ActionError(_unreadable(exc)) from exc


def _unreadable(exc: Exception) -> str:
    """The message for an action whose own code raised ``exc``; never raises an ``Exception``."""
    try:
        return "action could not be read: " + class_name(exc)
    except Exception:  # e.g. MemoryError while copying the name
        return "action could not be read"


def _read_action(action: object, dim: int) -> tuple[float, ...]:
    if isinstance(action, (str, bytes, bytearray, Mapping, Set)) or action is None:
        raise ActionError(f"action must be a sequence of {dim} numbers, got {class_name(action)}")
    if isinstance(action, Iterator):
        raise ActionError(
            f"action must be a sequence of {dim} numbers, not a one-shot generator or iterator"
        )
    ndim = getattr(action, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ActionError(f"action must be 1-dimensional, got ndim={ndim}")
    if not isinstance(action, Iterable):
        raise ActionError(f"action must be a sequence of {dim} numbers, got {class_name(action)}")
    items = list(itertools.islice(action, dim + 1))
    if len(items) > dim:
        raise ActionError(f"action must have exactly {dim} values, got more than {dim}")
    if len(items) != dim:
        raise ActionError(f"action must have exactly {dim} values, got {len(items)}")
    return tuple(_action_value(i, item) for i, item in enumerate(items))


def _action_value(index: int, item: object) -> float:
    if isinstance(item, bool) or _is_numpy_bool(item):
        raise ActionError(f"action[{index}] is a bool, expected a number")
    if not isinstance(item, numbers.Real):
        raise ActionError(f"action[{index}] must be a real number, got {class_name(item)}")
    try:
        value = float(item)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ActionError(
            f"action[{index}] cannot be converted to a float: {type(exc).__name__}"
        ) from exc
    if not math.isfinite(value):
        raise ActionError(f"action[{index}] is not finite: {value}")
    return value


def _is_numpy_bool(item: object) -> bool:
    """Whether ``item`` is a ``numpy.bool_``, without importing numpy (if numpy was never
    imported, nothing can be one). Checked by type, never by a name a class could borrow."""
    np = sys.modules.get("numpy")
    return np is not None and isinstance(item, np.bool_)


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
    """Everything a controller may know about its task, handed to ``make_controller``.

    Immutable all the way down: ``extras`` is a read-only copy (str -> float), so a controller
    cannot change the spec another episode receives. Specs are hashable, compare by value and
    pickle (vectorized training ships them between processes).
    """

    challenge: str
    obs_schema: str
    obs_features: tuple[str, ...]
    action: ActionSpec
    control_dt: float
    episode_seconds: float
    robot: RobotGeometry
    extras: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", _frozen_extras(self.extras))
        if not (math.isfinite(self.control_dt) and self.control_dt > 0):
            raise ValueError(f"control_dt must be positive and finite, got {self.control_dt}")
        steps = self.episode_seconds / self.control_dt
        if not math.isfinite(steps) or abs(steps - round(steps)) > 1e-9 or round(steps) < 1:
            raise ValueError(
                f"episode_seconds ({self.episode_seconds}) must be a positive multiple of "
                f"control_dt ({self.control_dt})"
            )

    def __hash__(self) -> int:
        return hash(
            (
                self.challenge,
                self.obs_schema,
                self.obs_features,
                self.action,
                self.control_dt,
                self.episode_seconds,
                self.robot,
                tuple(sorted(self.extras.items())),
            )
        )

    def __reduce__(self) -> tuple[type[TaskSpec], tuple[Any, ...]]:
        """Pickle/copy support: a mappingproxy cannot be pickled, so rebuild from a plain dict."""
        return (
            TaskSpec,
            (
                self.challenge,
                self.obs_schema,
                self.obs_features,
                self.action,
                self.control_dt,
                self.episode_seconds,
                self.robot,
                dict(self.extras),
            ),
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


def _frozen_extras(extras: object) -> Mapping[str, float]:
    """Read-only copy of ``extras`` with str keys and finite float values."""
    if not isinstance(extras, Mapping):
        raise TypeError(f"extras must be a mapping of str to numbers, got {type(extras).__name__}")
    values: dict[str, float] = {}
    for key, raw in extras.items():
        if not isinstance(key, str):
            raise TypeError(f"extras keys must be str, got {type(key).__name__}")
        value = _as_float(f"extras[{key!r}]", raw)
        if not math.isfinite(value):
            raise ValueError(f"extras[{key!r}] must be finite, got {value}")
        values[key] = value
    return MappingProxyType(values)


ObsT_contra = TypeVar("ObsT_contra", contravariant=True)


@runtime_checkable
class Controller(Protocol[ObsT_contra]):
    """A controller: ``reset`` once per episode, then ``act`` at ``1/control_dt`` Hz."""

    def reset(self, rng_seed: int) -> None: ...

    def act(self, obs: ObsT_contra) -> Sequence[float]: ...
