"""Disturbance presets and scenarios (docs/contract.md §9-10).

A ``Scenario`` is everything random about an episode except per-step sensor noise, materialized
as JSON-serializable data: initial state, pushes, realized model parameters and sensor offsets.
It is drawn from named RNG streams, embedded in every trajectory, and can be replayed exactly.
"""

from __future__ import annotations

import itertools
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from topple.rng import Stream, stream_rng, validate_seed

MAX_PUSHES = 3  # the PUSHES stream always draws this many candidates (fixed consumption)
MAX_PUSH_FORCE_N = 50.0  # contract §10: no push is ever stronger than this
MAX_ACTUATOR_DELAY_S = 0.010  # what the native delay buffer (robot.DELAY_HISTORY) is sized for

# Sane bounds for a scenario read from JSON (contract §9). Scales multiply the robot's params.
# Every preset's randomization ranges lie well inside them; they only keep a hand-edited or
# corrupted scenario from building a nonsensical model.
SCALE_BOUNDS = (0.5, 2.0)  # chassis mass, wheel friction, each motor gain
FRICTIONLOSS_SCALE_BOUNDS = (0.0, 3.0)
MAX_COM_X_OFFSET_M = 0.02
MAX_IMU_PITCH_OFFSET_RAD = 0.1
MAX_GYRO_BIAS_RADS = 0.1
MAX_LATERAL_RATIO = 1.0
# The force cap held for a standard 0.1 s pulse: 11x the strongest calibrated push (0.45 N·s).
MAX_PUSH_IMPULSE_NS = 5.0
MAX_EXACT_INT = 2**53  # a scenario's integers must be exact in a JSON double, like seeds
# The initial state, on every reset path (contract §9): any orientation about the axle, and up
# to 11x the gyro's full scale, far beyond any state a controller could recover from.
MAX_INIT_PITCH_RAD = math.pi
MAX_INIT_PITCH_RATE_RADS = 100.0


@dataclass(frozen=True)
class EpisodeClock:
    """What a scenario's step counts mean in the env that runs it (contract §4, §9).

    Pushes start at a control step and last a number of physics substeps, so only with the
    physics ``timestep``, the substeps per control step (``frame_skip``) and the episode length
    can a push be checked against the episode: starting inside it, lasting no longer than it,
    never overlapping another push, and within ``MAX_PUSH_IMPULSE_NS``. ``BalanceEnv.clock`` is
    the env's own.
    """

    timestep: float
    frame_skip: int
    max_steps: int

    def __post_init__(self) -> None:
        counts = (self.frame_skip, self.max_steps)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
            raise ValueError(f"episode clock needs integer frame_skip and max_steps >= 1: {self}")
        if not (math.isfinite(self.timestep) and self.timestep > 0):
            raise ValueError(f"episode clock needs a finite positive timestep: {self}")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PushConfig(_Frozen):
    count: tuple[int, ...] = (1, 2, 3)
    impulse_ns: tuple[float, float] = (0.25, 0.45)
    duration_s: float = 0.1
    earliest_s: float = 1.0
    latest_s: float = 8.5
    min_spacing_s: float = 1.5
    lateral_fraction_max: float = 0.3
    max_force_n: float = MAX_PUSH_FORCE_N

    @model_validator(mode="after")
    def _check(self) -> PushConfig:
        if not self.count or min(self.count) < 0 or max(self.count) > MAX_PUSHES:
            raise ValueError(f"count values must be in [0, {MAX_PUSHES}]")
        if self.impulse_ns[1] / self.duration_s > self.max_force_n:
            raise ValueError("impulse_ns / duration_s exceeds max_force_n")
        span = self.latest_s - self.earliest_s - (max(self.count) - 1) * self.min_spacing_s
        if span < 0:
            raise ValueError("pushes do not fit between earliest_s and latest_s")
        return self


class RandomizationConfig(_Frozen):
    mass_scale: tuple[float, float] = (0.9, 1.1)
    com_x_offset_m: tuple[float, float] = (-0.003, 0.003)
    wheel_friction_scale: tuple[float, float] = (0.9, 1.1)
    motor_gain_common: tuple[float, float] = (0.9, 1.1)
    motor_gain_per_wheel: tuple[float, float] = (0.95, 1.05)
    frictionloss_scale: tuple[float, float] = (0.5, 1.5)
    actuator_delay_ms: tuple[int, ...] = (0, 2, 4, 6, 8, 10)
    imu_pitch_offset_rad: tuple[float, float] = (-math.radians(1.0), math.radians(1.0))
    gyro_bias_rads: tuple[float, float] = (-0.01, 0.01)
    init_pitch_rad: tuple[float, float] = (-math.radians(3.0), math.radians(3.0))


class Preset(_Frozen):
    name: str
    sensors: Literal["ideal", "full"]
    randomize: bool
    pushes: PushConfig | None = None
    randomization: RandomizationConfig = RandomizationConfig()


PRESETS: dict[str, Preset] = {
    "ideal": Preset(name="ideal", sensors="ideal", randomize=False),
    "clean": Preset(name="clean", sensors="full", randomize=True),
    "standard": Preset(name="standard", sensors="full", randomize=True, pushes=PushConfig()),
}


@dataclass(frozen=True)
class Push:
    start_step: int  # control step at which the push begins
    n_substeps: int  # duration in physics substeps
    magnitude_n: float
    forward_sign: float  # +1 pushes toward the heading at onset, -1 away from it
    lateral_ratio: float  # signed lateral share of the force, |r| <= lateral_fraction_max


@dataclass(frozen=True)
class ModelOverrides:
    chassis_mass_scale: float = 1.0
    com_x_offset_m: float = 0.0
    wheel_friction_scale: float = 1.0
    motor_gain: tuple[float, float] = (1.0, 1.0)
    frictionloss_scale: float = 1.0
    actuator_delay_s: float = 0.0


@dataclass(frozen=True)
class SensorOffsets:
    imu_pitch_offset_rad: float = 0.0
    gyro_bias_rads: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Scenario:
    seed: int
    preset: str
    init_pitch_rad: float
    pushes: tuple[Push, ...]
    model: ModelOverrides
    sensors: SensorOffsets
    init_pitch_rate_rads: float = 0.0  # nonzero only for injected states (fuzzer, RoA maps)

    def to_dict(self) -> dict[str, Any]:
        """JSON data (a new dict: editing it never reaches this frozen scenario)."""
        return {
            "seed": self.seed,
            "preset": self.preset,
            "init_pitch_rad": self.init_pitch_rad,
            "init_pitch_rate_rads": self.init_pitch_rate_rads,
            "pushes": [dict(vars(p)) for p in self.pushes],  # vars() is the Push's own __dict__
            "model": {**vars(self.model), "motor_gain": list(self.model.motor_gain)},
            "sensors": {**vars(self.sensors), "gyro_bias_rads": list(self.sensors.gyro_bias_rads)},
        }

    @classmethod
    def from_dict(cls, data: Any, *, clock: EpisodeClock | None = None) -> Scenario:
        """Parse and validate a scenario from JSON data; any problem raises ``ValueError``.

        Checks key sets (string keys only), types, that every number fits a float (integers
        exactly) and is finite, and the bounds above; pushes must be sorted by ``start_step``
        with no two at the same step. With the env's ``clock`` (``BalanceEnv.reset`` passes
        it), every push must also start inside the episode, last no longer than it, deliver at
        most ``MAX_PUSH_IMPULSE_NS`` and end before the next one starts.
        """
        try:
            scenario = _parse_scenario(data)
            validate_scenario(scenario, clock=clock)
        except (TypeError, OverflowError) as exc:  # a case the checks above missed: still bad data
            raise ValueError(f"{_INVALID}{type(exc).__name__}: {exc}") from exc
        except ValueError as exc:
            message = str(exc)
            if not message.startswith(_INVALID):
                message = f"{_INVALID}{message}"
            raise ValueError(message) from exc
        return scenario


_INVALID = "invalid scenario: "
_SCENARIO_KEYS = frozenset(
    {"seed", "preset", "init_pitch_rad", "init_pitch_rate_rads", "pushes", "model", "sensors"}
)
_PUSH_KEYS = frozenset(f.name for f in fields(Push))
_MODEL_KEYS = frozenset(f.name for f in fields(ModelOverrides))
_SENSOR_KEYS = frozenset(f.name for f in fields(SensorOffsets))


def _mapping(name: str, value: object, keys: frozenset[str]) -> Any:
    """``value`` if it is a mapping with exactly ``keys`` (all strings), else ``ValueError``."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object, got {type(value).__name__}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name}: keys must be strings")
    present = set(value)
    missing = keys - present
    unknown = present - keys
    if missing or unknown:
        raise ValueError(f"{name}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return value


def _real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:  # e.g. a JSON integer above 1.8e308
        raise ValueError(f"{name} does not fit a float ({type(exc).__name__})") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    return result


def _int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if abs(value) > MAX_EXACT_INT:  # never formatted: str() refuses ints beyond 4300 digits
        raise ValueError(f"{name} does not fit a float exactly (|n| > 2**53)")
    return value


def _reals(name: str, value: object, n: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != n:
        raise ValueError(f"{name} must be a list of {n} numbers")
    return tuple(_real(f"{name}[{i}]", v) for i, v in enumerate(value))


def _parse_push(i: int, data: object) -> Push:
    d = _mapping(f"pushes[{i}]", data, _PUSH_KEYS)
    return Push(
        start_step=_int(f"pushes[{i}].start_step", d["start_step"]),
        n_substeps=_int(f"pushes[{i}].n_substeps", d["n_substeps"]),
        magnitude_n=_real(f"pushes[{i}].magnitude_n", d["magnitude_n"]),
        forward_sign=_real(f"pushes[{i}].forward_sign", d["forward_sign"]),
        lateral_ratio=_real(f"pushes[{i}].lateral_ratio", d["lateral_ratio"]),
    )


def _parse_model(data: object) -> ModelOverrides:
    d = _mapping("model", data, _MODEL_KEYS)
    gain = _reals("model.motor_gain", d["motor_gain"], 2)
    return ModelOverrides(
        chassis_mass_scale=_real("model.chassis_mass_scale", d["chassis_mass_scale"]),
        com_x_offset_m=_real("model.com_x_offset_m", d["com_x_offset_m"]),
        wheel_friction_scale=_real("model.wheel_friction_scale", d["wheel_friction_scale"]),
        motor_gain=(gain[0], gain[1]),
        frictionloss_scale=_real("model.frictionloss_scale", d["frictionloss_scale"]),
        actuator_delay_s=_real("model.actuator_delay_s", d["actuator_delay_s"]),
    )


def _parse_sensors(data: object) -> SensorOffsets:
    d = _mapping("sensors", data, _SENSOR_KEYS)
    bias = _reals("sensors.gyro_bias_rads", d["gyro_bias_rads"], 3)
    return SensorOffsets(
        imu_pitch_offset_rad=_real("sensors.imu_pitch_offset_rad", d["imu_pitch_offset_rad"]),
        gyro_bias_rads=(bias[0], bias[1], bias[2]),
    )


def _parse_scenario(data: object) -> Scenario:
    d = _mapping("scenario", data, _SCENARIO_KEYS)
    pushes = d["pushes"]
    if isinstance(pushes, (str, bytes)) or not isinstance(pushes, Sequence):
        raise ValueError("pushes must be a list")
    preset = d["preset"]
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    seed = validate_seed(d["seed"])
    return Scenario(
        seed=seed,
        preset=preset,
        init_pitch_rad=_real("init_pitch_rad", d["init_pitch_rad"]),
        init_pitch_rate_rads=_real("init_pitch_rate_rads", d["init_pitch_rate_rads"]),
        pushes=tuple(_parse_push(i, p) for i, p in enumerate(pushes)),
        model=_parse_model(d["model"]),
        sensors=_parse_sensors(d["sensors"]),
    )


def _check_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ValueError(f"{name} = {value} is outside [{low}, {high}]")


def _check_pushes(pushes: tuple[Push, ...], clock: EpisodeClock | None) -> None:
    last_step = math.inf if clock is None else clock.max_steps - 1
    most_substeps = math.inf if clock is None else clock.max_steps * clock.frame_skip
    for i, push in enumerate(pushes):
        _check_range(f"pushes[{i}].start_step", push.start_step, 0, last_step)
        _check_range(f"pushes[{i}].n_substeps", push.n_substeps, 1, most_substeps)
        _check_range(f"pushes[{i}].magnitude_n", push.magnitude_n, 0.0, MAX_PUSH_FORCE_N)
        if push.forward_sign not in (1.0, -1.0):
            raise ValueError(f"pushes[{i}].forward_sign must be +1 or -1")
        _check_range(
            f"pushes[{i}].lateral_ratio", push.lateral_ratio, -MAX_LATERAL_RATIO, MAX_LATERAL_RATIO
        )
        if clock is not None:
            impulse = push.magnitude_n * push.n_substeps * clock.timestep
            _check_range(f"pushes[{i}] impulse_ns", impulse, 0.0, MAX_PUSH_IMPULSE_NS)
    check_push_schedule(pushes, frame_skip=None if clock is None else clock.frame_skip)


def check_push_schedule(pushes: Sequence[Push], *, frame_skip: int | None = None) -> None:
    """Raise ``ValueError`` unless every push can run in full, exactly as written.

    The env runs one push at a time: a push starting at the step of another, or while another
    is still active, would silently cut that one short. So pushes must be sorted by
    ``start_step`` with no two at one step (an unsorted list would not survive the scenario's
    JSON round trip either) and, with ``frame_skip`` (substeps per control step) known, each
    must end before the next begins. Every path into the env enforces this, including
    in-process ``Scenario`` objects that skip the bounds checks.
    """
    steps = [p.start_step for p in pushes]
    if any(b <= a for a, b in itertools.pairwise(steps)):
        raise ValueError(f"pushes must be sorted by start_step with no repeats, got {steps}")
    if frame_skip is None:
        return
    for i, (a, b) in enumerate(itertools.pairwise(pushes)):
        if a.start_step * frame_skip + a.n_substeps > b.start_step * frame_skip:
            raise ValueError(
                f"pushes[{i + 1}] (start_step {b.start_step}) begins before pushes[{i}] "
                f"(start_step {a.start_step}, {a.n_substeps} substeps) ends: overlapping pushes "
                "would cut each other short"
            )


def _check_model(model: ModelOverrides) -> None:
    _check_range("model.chassis_mass_scale", model.chassis_mass_scale, *SCALE_BOUNDS)
    _check_range("model.wheel_friction_scale", model.wheel_friction_scale, *SCALE_BOUNDS)
    for i, gain in enumerate(model.motor_gain):
        _check_range(f"model.motor_gain[{i}]", gain, *SCALE_BOUNDS)
    _check_range("model.frictionloss_scale", model.frictionloss_scale, *FRICTIONLOSS_SCALE_BOUNDS)
    _check_range(
        "model.com_x_offset_m", model.com_x_offset_m, -MAX_COM_X_OFFSET_M, MAX_COM_X_OFFSET_M
    )
    _check_range("model.actuator_delay_s", model.actuator_delay_s, 0.0, MAX_ACTUATOR_DELAY_S)


def _check_sensors(sensors: SensorOffsets) -> None:
    limit = MAX_IMU_PITCH_OFFSET_RAD
    _check_range("sensors.imu_pitch_offset_rad", sensors.imu_pitch_offset_rad, -limit, limit)
    for i, bias in enumerate(sensors.gyro_bias_rads):
        _check_range(f"sensors.gyro_bias_rads[{i}]", bias, -MAX_GYRO_BIAS_RADS, MAX_GYRO_BIAS_RADS)


def check_initial_state(pitch_rad: float, pitch_rate_rads: float) -> None:
    """Raise ``ValueError`` unless the initial pitch and pitch rate are finite and inside their
    bounds (contract §9). Every reset path checks this, the trusted test hooks included: from
    about 1e84 rad/s MuJoCo cannot compute the reset (its accelerations come out NaN), and the
    float32 records cannot hold a rate beyond 3.4e38 rad/s."""
    if not (math.isfinite(pitch_rad) and math.isfinite(pitch_rate_rads)):
        raise ValueError("initial pitch and pitch rate must be finite")
    _check_range("init_pitch_rad", pitch_rad, -MAX_INIT_PITCH_RAD, MAX_INIT_PITCH_RAD)
    limit = MAX_INIT_PITCH_RATE_RADS
    _check_range("init_pitch_rate_rads", pitch_rate_rads, -limit, limit)


def validate_scenario(scenario: Scenario, *, clock: EpisodeClock | None = None) -> None:
    """Raise ``ValueError`` unless ``scenario`` is inside the contract's sane bounds (§9);
    with the env's ``clock``, its pushes must also fit the episode."""
    try:
        check_initial_state(scenario.init_pitch_rad, scenario.init_pitch_rate_rads)
        _check_pushes(scenario.pushes, clock)
        _check_model(scenario.model)
        _check_sensors(scenario.sensors)
    except ValueError as exc:
        raise ValueError(f"{_INVALID}{exc}") from exc


def _affine(low: float, high: float, u: float) -> float:
    """``low + (high - low)·u``, rounded after each operation (Python floats never fuse)."""
    return low + (high - low) * u


def _uniforms(rng: Any, low: float, high: float, n: int) -> list[float]:
    """``n`` draws from U[low, high), identical on every platform (contract §9).

    Each consumes exactly one raw double ``u`` from ``rng.random`` (exact: a 53-bit integer times
    2**-53), as ``Generator.uniform`` does, but the affine map runs in Python, where every operation
    is rounded. numpy evaluates it in C, where a compiler may contract it into one fused
    multiply-add: numpy's macOS arm64 and Linux x86_64 builds disagree in the last bit. Scenarios
    are trusted data compared bit for bit (contract §16), so they must not.
    """
    return [_affine(low, high, u) for u in rng.random(n).tolist()]


def _uniform(rng: Any, bounds: tuple[float, float]) -> float:
    return _uniforms(rng, bounds[0], bounds[1], 1)[0]


def _draw_pushes(
    seed: int, cfg: PushConfig, timestep: float, control_dt: float
) -> tuple[Push, ...]:
    rng = stream_rng(seed, Stream.PUSHES)
    count = int(cfg.count[int(rng.integers(len(cfg.count)))])
    # Fixed consumption: always draw MAX_PUSHES candidates, keep the first `count`.
    offsets = _uniforms(rng, 0.0, 1.0, MAX_PUSHES)
    impulses = _uniforms(rng, cfg.impulse_ns[0], cfg.impulse_ns[1], MAX_PUSHES)
    signs = rng.integers(0, 2, size=MAX_PUSHES)
    lateral = cfg.lateral_fraction_max
    laterals = _uniforms(rng, -lateral, lateral, MAX_PUSHES)
    if count == 0:
        return ()
    span = cfg.latest_s - cfg.earliest_s - (count - 1) * cfg.min_spacing_s
    starts = sorted(o * span for o in offsets[:count])
    n_substeps = round(cfg.duration_s / timestep)
    pushes = []
    for i in range(count):
        start_s = cfg.earliest_s + starts[i] + i * cfg.min_spacing_s
        pushes.append(
            Push(
                start_step=round(start_s / control_dt),
                n_substeps=n_substeps,
                magnitude_n=impulses[i] / (n_substeps * timestep),
                forward_sign=1.0 if signs[i] else -1.0,
                lateral_ratio=laterals[i],
            )
        )
    return tuple(pushes)


def make_scenario(
    seed: int, preset: Preset, *, timestep: float, control_dt: float, episode_seconds: float
) -> Scenario:
    """Draw an episode's scenario from named streams (independent of each other)."""
    validate_seed(seed)
    r = preset.randomization
    init_pitch = _uniform(stream_rng(seed, Stream.INIT), r.init_pitch_rad)
    pushes = (
        _draw_pushes(seed, preset.pushes, timestep, control_dt) if preset.pushes is not None else ()
    )
    if pushes and pushes[-1].start_step * control_dt > episode_seconds:
        raise ValueError("push scheduled after the episode ends")

    model = ModelOverrides()
    sensors = SensorOffsets()
    if preset.randomize:
        rng = stream_rng(seed, Stream.MODEL)
        common = _uniform(rng, r.motor_gain_common)
        model = ModelOverrides(
            chassis_mass_scale=_uniform(rng, r.mass_scale),
            com_x_offset_m=_uniform(rng, r.com_x_offset_m),
            wheel_friction_scale=_uniform(rng, r.wheel_friction_scale),
            motor_gain=(
                common * _uniform(rng, r.motor_gain_per_wheel),
                common * _uniform(rng, r.motor_gain_per_wheel),
            ),
            frictionloss_scale=_uniform(rng, r.frictionloss_scale),
            actuator_delay_s=0.0,
        )
        sensors = SensorOffsets(
            imu_pitch_offset_rad=_uniform(rng, r.imu_pitch_offset_rad),
            gyro_bias_rads=(
                _uniform(rng, r.gyro_bias_rads),
                _uniform(rng, r.gyro_bias_rads),
                _uniform(rng, r.gyro_bias_rads),
            ),
        )
        delay_rng = stream_rng(seed, Stream.DELAY)
        delay_ms = r.actuator_delay_ms[int(delay_rng.integers(len(r.actuator_delay_ms)))]
        model = ModelOverrides(**{**vars(model), "actuator_delay_s": delay_ms / 1000.0})

    return Scenario(
        seed=seed,
        preset=preset.name,
        init_pitch_rad=init_pitch,
        pushes=pushes,
        model=model,
        sensors=sensors,
    )


def push_force_world(push: Push, yaw: float) -> tuple[float, float, float]:
    """World-frame force for ``push`` given the robot heading ``yaw`` at push onset."""
    forward = math.sqrt(max(0.0, 1.0 - push.lateral_ratio**2))
    hx, hy = math.cos(yaw), math.sin(yaw)  # heading
    lx, ly = -math.sin(yaw), math.cos(yaw)  # left
    fx = push.magnitude_n * (push.forward_sign * forward * hx + push.lateral_ratio * lx)
    fy = push.magnitude_n * (push.forward_sign * forward * hy + push.lateral_ratio * ly)
    return (fx, fy, 0.0)
