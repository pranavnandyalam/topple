"""Disturbance presets and scenarios (docs/contract.md §9-10).

A ``Scenario`` is everything random about an episode except per-step sensor noise, materialized
as JSON-serializable data: initial state, pushes, realized model parameters and sensor offsets.
It is drawn from named RNG streams, embedded in every trajectory, and can be replayed exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from topple.rng import Stream, stream_rng, validate_seed

_MAX_PUSHES = 3  # the PUSHES stream always draws this many candidates (fixed consumption)


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
    max_force_n: float = 50.0

    @model_validator(mode="after")
    def _check(self) -> PushConfig:
        if not self.count or min(self.count) < 0 or max(self.count) > _MAX_PUSHES:
            raise ValueError(f"count values must be in [0, {_MAX_PUSHES}]")
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "preset": self.preset,
            "init_pitch_rad": self.init_pitch_rad,
            "pushes": [vars(p) for p in self.pushes],
            "model": {**vars(self.model), "motor_gain": list(self.model.motor_gain)},
            "sensors": {**vars(self.sensors), "gyro_bias_rads": list(self.sensors.gyro_bias_rads)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        if data["preset"] not in PRESETS:
            raise ValueError(f"unknown preset {data['preset']!r}")
        model = data["model"]
        sensors = data["sensors"]
        gyro_bias = sensors["gyro_bias_rads"]
        return cls(
            seed=validate_seed(data["seed"]),
            preset=data["preset"],
            init_pitch_rad=float(data["init_pitch_rad"]),
            pushes=tuple(Push(**p) for p in data["pushes"]),
            model=ModelOverrides(
                **{**model, "motor_gain": (model["motor_gain"][0], model["motor_gain"][1])}
            ),
            sensors=SensorOffsets(
                imu_pitch_offset_rad=sensors["imu_pitch_offset_rad"],
                gyro_bias_rads=(gyro_bias[0], gyro_bias[1], gyro_bias[2]),
            ),
        )


def _uniform(rng: Any, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _draw_pushes(
    seed: int, cfg: PushConfig, timestep: float, control_dt: float
) -> tuple[Push, ...]:
    rng = stream_rng(seed, Stream.PUSHES)
    count = int(cfg.count[int(rng.integers(len(cfg.count)))])
    # Fixed consumption: always draw _MAX_PUSHES candidates, keep the first `count`.
    offsets = rng.uniform(0.0, 1.0, size=_MAX_PUSHES)
    impulses = rng.uniform(cfg.impulse_ns[0], cfg.impulse_ns[1], size=_MAX_PUSHES)
    signs = rng.integers(0, 2, size=_MAX_PUSHES)
    laterals = rng.uniform(-cfg.lateral_fraction_max, cfg.lateral_fraction_max, size=_MAX_PUSHES)
    if count == 0:
        return ()
    span = cfg.latest_s - cfg.earliest_s - (count - 1) * cfg.min_spacing_s
    starts = sorted(float(o) * span for o in offsets[:count])
    n_substeps = round(cfg.duration_s / timestep)
    pushes = []
    for i in range(count):
        start_s = cfg.earliest_s + starts[i] + i * cfg.min_spacing_s
        pushes.append(
            Push(
                start_step=round(start_s / control_dt),
                n_substeps=n_substeps,
                magnitude_n=float(impulses[i]) / (n_substeps * timestep),
                forward_sign=1.0 if signs[i] else -1.0,
                lateral_ratio=float(laterals[i]),
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
