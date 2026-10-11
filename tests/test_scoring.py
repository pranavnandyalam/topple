import math

import numpy as np
import pytest

from topple.challenge import ScoreConfig, load_challenge
from topple.robot import MotorParams
from topple.scoring import (
    EpisodeMetrics,
    battery_energy_j,
    episode_metrics,
    episode_score,
    episode_success,
    tilt_from_quat,
)
from topple.traj import Trajectory

CFG = load_challenge("balance-v1").config.score
MOTOR = load_challenge("balance-v1").params.motor


def quat_pitch(theta: float) -> list[float]:
    return [math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0]


def synthetic(
    steps: int,
    max_steps: int = 10,
    pitch: float = 0.0,
    x_end: float = 0.0,
    ctrl: float = 0.0,
    wheel_rate: float = 0.0,
    termination: str = "timeout",
) -> Trajectory:
    qpos = np.zeros((steps + 1, 9))
    qpos[:, 3:7] = quat_pitch(pitch)
    qpos[:, 0] = np.linspace(0.0, x_end, steps + 1)
    qpos[:, 7] = qpos[:, 8] = wheel_rate * 0.01 * np.arange(steps + 1)
    return Trajectory(
        qpos=qpos,
        qvel=np.zeros((steps + 1, 8)),
        obs=np.zeros((steps, 10), dtype=np.float32),
        action=np.full((steps, 2), ctrl),
        ctrl=np.full((steps, 2), ctrl),
        enc_counts=np.zeros((steps, 2), dtype=np.int32),
        gyro=np.zeros((steps, 3)),
        accel=np.zeros((steps, 3)),
        push_force=np.zeros((steps, 3)),
        epilogue_qpos=np.zeros((0, 9)),
        meta={
            "max_steps": max_steps,
            "control_dt": 0.01,
            "termination": {"step": steps, "reason": termination},
            "joint_qpos": {"root": 0, "wheels": [7, 8]},
        },
    )


def test_tilt_takes_its_arccos_from_the_c_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy picks its float64 arccos at run time (an SVML routine on AVX-512 CPUs, the C library
    elsewhere) and the two can differ in the last bit, so two linux-x86_64 machines could score
    one trajectory differently. The tilt uses ``math.acos`` per row, as the env's fall check
    does (``robot.quat_tilt``), whatever the leading shape."""

    def dispatched(*args: object, **kwargs: object) -> None:
        raise AssertionError("np.arccos depends on numpy's CPU dispatch")

    monkeypatch.setattr(np, "arccos", dispatched)
    rng = np.random.default_rng(0)
    quat = rng.normal(size=(2, 25, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    tilt = tilt_from_quat(quat)
    assert tilt.shape == (2, 25)
    assert tilt.dtype == np.float64
    expected = [
        math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))
        for x, y in zip(quat[..., 1].ravel().tolist(), quat[..., 2].ravel().tolist(), strict=True)
    ]
    assert tilt.ravel().tobytes() == np.array(expected).tobytes()  # bit for bit


def test_tilt_from_quat() -> None:
    q = np.array([quat_pitch(0.3), quat_pitch(-0.5), [1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(tilt_from_quat(q), [0.3, 0.5, 0.0], atol=1e-12)


def test_perfect_episode_scores_100() -> None:
    m = episode_metrics(synthetic(10), MOTOR)
    assert m.survived
    assert m.upright_frac == 1.0
    assert episode_score(m, CFG) == pytest.approx(100.0)
    assert episode_success(m, CFG)


def test_survival_gates_secondary_terms() -> None:
    """Falling immediately must score near zero even with perfect tilt/drift/energy."""
    m = episode_metrics(synthetic(1, max_steps=1000, termination="fell"), MOTOR)
    assert not m.survived
    assert episode_score(m, CFG) == pytest.approx(0.1)
    assert not episode_success(m, CFG)


def test_a_timeout_before_max_steps_is_not_survival() -> None:
    """Survival is a truncation: reason "timeout" and ``T == max_steps`` (§12). A trajectory that
    says "timeout" at step 400 (the env records an error-ended episode as "error", so only a
    hand-made file can) is not a survivor."""
    m = episode_metrics(synthetic(400, max_steps=1000, termination="timeout"), MOTOR)
    assert m.termination == "timeout"
    assert not m.survived
    assert not episode_success(m, CFG)
    assert episode_metrics(synthetic(1000, max_steps=1000), MOTOR).survived


def test_tilt_term() -> None:
    m = episode_metrics(synthetic(10, pitch=0.1), MOTOR)
    assert m.rms_tilt_rad == pytest.approx(0.1)
    assert episode_score(m, CFG) == pytest.approx(100 * (0.6 + 0.2 * 0.5 + 0.1 + 0.1))


def test_drift_term_and_success_cutoff() -> None:
    m = episode_metrics(synthetic(10, x_end=0.5), MOTOR)
    assert m.drift_m == pytest.approx(0.5)
    assert episode_score(m, CFG) == pytest.approx(100 * (0.6 + 0.2 + 0.05 + 0.1))
    far = episode_metrics(synthetic(10, x_end=1.5), MOTOR)
    assert not episode_success(far, CFG)


def test_energy_formula_matches_hand_calculation() -> None:
    motor = MotorParams(
        tau_stall_nm=0.3,
        omega_noload_rads=30.0,
        torque_limit_nm=0.3,
        stall_current_a=1.5,
        v_nom=7.4,
        armature=1e-4,
        frictionloss_nm=0.02,
    )
    ctrl = np.full((10, 2), 0.5)
    angles = np.zeros((11, 2))
    # stalled wheels at u = 0.5: P = V*I*u*u per wheel
    expected = 2 * 10 * 7.4 * 1.5 * 0.25 * 0.01
    assert battery_energy_j(ctrl, angles, 0.01, motor) == pytest.approx(expected)
    # regenerating (u opposite to motion) draws nothing from the battery
    spinning = np.cumsum(np.full((11, 2), 0.2), axis=0)  # 20 rad/s forward
    assert battery_energy_j(np.full((10, 2), -0.5), spinning, 0.01, motor) == pytest.approx(
        2 * 10 * 7.4 * 1.5 * (-0.5) * (-0.5 - 20 / 30) * 0.01
    )
    assert battery_energy_j(np.full((10, 2), 0.5), spinning, 0.01, motor) == 0.0


def test_episode_energy_counts_the_wheels_motion() -> None:
    """episode_metrics feeds the recorded wheel angles into the formula: at u = 0.5 a wheel
    turning at 10 rad/s draws (0.5 - 10/30)/0.5 of its stall power, at 20 rad/s none."""
    p = MOTOR.v_nom * MOTOR.stall_current_a
    turning = episode_metrics(synthetic(10, ctrl=0.5, wheel_rate=10.0), MOTOR)
    assert turning.energy_j == pytest.approx(2 * 10 * p * 0.5 * (0.5 - 10 / 30) * 0.01, rel=1e-12)
    assert episode_metrics(synthetic(10, ctrl=0.5, wheel_rate=20.0), MOTOR).energy_j == 0.0
    stalled = episode_metrics(synthetic(10, ctrl=0.5), MOTOR)
    assert stalled.energy_j == pytest.approx(2 * 10 * p * 0.25 * 0.01, rel=1e-12)


@pytest.mark.parametrize(
    ("ctrl", "fraction"), [(0.99, 1.0), (-0.99, 1.0), (0.9899, 0.0), (-0.9899, 0.0)]
)
def test_saturation_starts_at_0_99(ctrl: float, fraction: float) -> None:
    """§12: a step is saturated when any |u| >= 0.99 after clipping, the boundary included."""
    assert episode_metrics(synthetic(10, ctrl=ctrl), MOTOR).saturation_fraction == fraction


def test_energy_cap_blocks_success() -> None:
    m = episode_metrics(synthetic(10, ctrl=1.0), MOTOR)
    assert m.energy_j > 0
    assert m.saturation_fraction == 1.0
    hungry = EpisodeMetrics(
        **{**m.as_dict(), "energy_j": CFG.e_ref_j * CFG.success_energy_factor + 1}
    )  # type: ignore[arg-type]
    assert not episode_success(hungry, CFG)


def test_mirrored_wheels_use_sign() -> None:
    traj = synthetic(10, ctrl=0.5, wheel_rate=-20.0)
    flipped = Trajectory(**{**traj.__dict__, "meta": {**traj.meta, "wheel_sign": [-1.0, -1.0]}})
    # same physical motion: forward command, wheels rolling forward (joint angles negative)
    forward = synthetic(10, ctrl=0.5, wheel_rate=20.0)
    assert episode_metrics(flipped, MOTOR).energy_j == pytest.approx(
        episode_metrics(forward, MOTOR).energy_j
    )


def test_zero_step_episode() -> None:
    m = episode_metrics(synthetic(0, max_steps=10, termination="error"), MOTOR)
    assert m.steps == 0
    assert m.energy_j == 0.0
    assert episode_score(m, CFG) == 0.0


def test_scoring_module_has_no_aggregate() -> None:
    """Protocol aggregates live in evaluate.summarize (score / N over the expected episodes)."""
    import topple.scoring

    assert not hasattr(topple.scoring, "aggregate_score")


def with_rows(traj: Trajectory, qpos: np.ndarray) -> Trajectory:
    return Trajectory(**{**traj.__dict__, "qpos": qpos})


def test_final_drift_differs_from_max_drift() -> None:
    """Out 0.8 m and back to 0.1 m: drift_m is the final distance, max_drift_m the farthest."""
    traj = synthetic(10)
    qpos = traj.qpos.copy()
    qpos[:, 0] = [0.0, 0.2, 0.4, 0.6, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    qpos[:, 1] = 0.0
    m = episode_metrics(with_rows(traj, qpos), MOTOR)
    assert m.drift_m == pytest.approx(0.1)
    assert m.max_drift_m == pytest.approx(0.8)
    assert episode_score(m, CFG) == pytest.approx(100 * (0.6 + 0.2 + 0.1 * 0.9 + 0.1))


def test_drift_is_planar_and_relative_to_the_start() -> None:
    traj = synthetic(4)
    qpos = traj.qpos.copy()
    qpos[:, 0] = 5.0 + np.array([0.0, 0.0, 0.0, 0.3, 0.3])
    qpos[:, 1] = -2.0 + np.array([0.0, 0.0, 0.0, 0.4, 0.4])
    qpos[:, 2] = np.linspace(0.0, 9.0, 5)  # height never counts
    m = episode_metrics(with_rows(traj, qpos), MOTOR)
    assert m.drift_m == pytest.approx(0.5)
    assert m.max_drift_m == pytest.approx(0.5)


def test_rms_tilt_counts_alive_rows_only() -> None:
    """Row T is the terminal state (already fallen): it counts for max_tilt, never for RMS tilt."""
    traj = synthetic(4, max_steps=10, pitch=0.1, termination="fell")
    qpos = traj.qpos.copy()
    qpos[-1, 3:7] = quat_pitch(1.2)
    m = episode_metrics(with_rows(traj, qpos), MOTOR)
    assert m.rms_tilt_rad == pytest.approx(0.1)
    assert m.max_tilt_rad == pytest.approx(1.2)
    all_rows = math.sqrt((4 * 0.1**2 + 1.2**2) / 5)
    assert m.rms_tilt_rad != pytest.approx(all_rows)


def test_rms_tilt_is_a_root_mean_square() -> None:
    traj = synthetic(4)
    qpos = traj.qpos.copy()
    for row, pitch in enumerate([0.0, 0.1, -0.2, 0.2, 0.9]):
        qpos[row, 3:7] = quat_pitch(pitch)
    m = episode_metrics(with_rows(traj, qpos), MOTOR)
    assert m.rms_tilt_rad == pytest.approx(math.sqrt((0.0 + 0.01 + 0.04 + 0.04) / 4))


def test_score_config_is_frozen_constant() -> None:
    assert ScoreConfig.model_fields.keys() >= {"e_ref_j", "tilt_ref_rad", "drift_ref_m"}
    assert CFG.e_ref_j == pytest.approx(13.0)
