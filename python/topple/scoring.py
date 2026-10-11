"""Episode metrics and score for ``balance-v1`` (docs/contract.md §12).

Everything here is a pure function of a stored ``Trajectory`` plus frozen challenge parameters, so
any score can be recomputed (and audited) from the trajectory file alone.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from topple.challenge import ScoreConfig
from topple.contract import SATURATION_THRESHOLD
from topple.robot import MotorParams
from topple.traj import Trajectory

SURVIVED = "timeout"


@dataclass(frozen=True)
class EpisodeMetrics:
    steps: int
    survived: bool
    termination: str
    upright_frac: float
    time_upright_s: float
    rms_tilt_rad: float
    max_tilt_rad: float
    drift_m: float
    max_drift_m: float
    energy_j: float
    saturation_fraction: float

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def tilt_from_quat(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Angle between body z and world z for (..., 4) wxyz quaternions."""
    x, y = quat[..., 1], quat[..., 2]
    cos_tilt = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    tilt: npt.NDArray[np.float64] = np.arccos(cos_tilt)
    return tilt


def battery_energy_j(
    ctrl: npt.NDArray[np.float64],
    wheel_angles: npt.NDArray[np.float64],
    control_dt: float,
    motor: MotorParams,
) -> float:
    """Ideal DC motor in slow decay, from commands and mean wheel speed per step.

    ``ctrl`` is (T, 2) forward-positive commands; ``wheel_angles`` is (T+1, 2) forward-positive
    chassis-relative wheel angles. Works the same on sim and real logs.
    """
    omega = np.diff(wheel_angles, axis=0) / control_dt
    power = motor.v_nom * motor.stall_current_a * ctrl * (ctrl - omega / motor.omega_noload_rads)
    return float(math.fsum(np.maximum(power, 0.0).ravel()) * control_dt)


def episode_metrics(traj: Trajectory, motor: MotorParams) -> EpisodeMetrics:
    meta = traj.meta
    steps = traj.steps
    max_steps = int(meta["max_steps"])
    control_dt = float(meta["control_dt"])
    termination = str(meta["termination"]["reason"])
    root = int(meta["joint_qpos"]["root"])
    wheels = [int(a) for a in meta["joint_qpos"]["wheels"]]
    sign = np.asarray(meta.get("wheel_sign", [1.0, 1.0]), dtype=np.float64)

    tilt = tilt_from_quat(traj.qpos[:, root + 3 : root + 7])
    xy = traj.qpos[:, root : root + 2] - traj.qpos[0, root : root + 2]
    dist = np.hypot(xy[:, 0], xy[:, 1])
    alive_tilt = tilt[:steps] if steps > 0 else tilt[:1]
    angles = (traj.qpos[:, wheels] - traj.qpos[0, wheels]) * sign
    survived = termination == SURVIVED and steps == max_steps
    return EpisodeMetrics(
        steps=steps,
        survived=survived,
        termination=termination,
        upright_frac=steps / max_steps,
        time_upright_s=steps * control_dt,
        rms_tilt_rad=float(np.sqrt(np.mean(alive_tilt**2))),
        max_tilt_rad=float(tilt.max()),
        drift_m=float(dist[-1]),
        max_drift_m=float(dist.max()),
        energy_j=battery_energy_j(traj.ctrl, angles, control_dt, motor) if steps else 0.0,
        saturation_fraction=float(np.mean(np.abs(traj.ctrl).max(axis=1) >= SATURATION_THRESHOLD))
        if steps
        else 0.0,
    )


def _lower_is_better(value: float, ref: float) -> float:
    return 1.0 - min(value / ref, 1.0)


def episode_score(m: EpisodeMetrics, cfg: ScoreConfig) -> float:
    """0-100. Secondary terms only count while upright, so falling fast never pays."""
    quality = (
        0.6
        + 0.2 * _lower_is_better(m.rms_tilt_rad, cfg.tilt_ref_rad)
        + 0.1 * _lower_is_better(m.drift_m, cfg.drift_ref_m)
        + 0.1 * _lower_is_better(m.energy_j, cfg.e_ref_j)
    )
    return 100.0 * m.upright_frac * quality


def episode_success(m: EpisodeMetrics, cfg: ScoreConfig) -> bool:
    return (
        m.survived
        and m.drift_m < cfg.success_max_drift_m
        and m.energy_j <= cfg.success_energy_factor * cfg.e_ref_j
    )
