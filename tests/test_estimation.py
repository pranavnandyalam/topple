import math
import subprocess
import sys

import pytest

from topple.estimation import (
    accel_tilt,
    cf_step,
    encoder_counts,
    imu_reading,
    odometry_yaw,
    wheel_angle_from_counts,
    wrap_angle,
)

G = 9.80665


def accel_for_tilt(pitch: float) -> tuple[float, float, float]:
    """Specific force a level-calibrated IMU reads when static at ``pitch``."""
    return (-G * math.sin(pitch), 0.0, G * math.cos(pitch))


@pytest.mark.parametrize("pitch", [-0.6, -0.1, 0.0, 0.05, 0.4])
def test_accel_tilt_recovers_static_pitch(pitch: float) -> None:
    ax, _, az = accel_for_tilt(pitch)
    assert accel_tilt(ax, az) == pytest.approx(pitch, abs=1e-12)


def test_cf_converges_to_accel_tilt_with_zero_gyro() -> None:
    est = 0.0
    ax, _, az = accel_for_tilt(0.2)
    for _ in range(5000):  # 10 s at 500 Hz, tau = 0.5 s
        est = cf_step(est, gyro_y=0.0, accel_x=ax, accel_z=az, tau_s=0.5, dt=0.002)
    assert est == pytest.approx(0.2, abs=1e-6)


def test_cf_follows_gyro_short_term() -> None:
    """Over a short window the filter integrates the gyro, ignoring a lying accelerometer."""
    est = 0.0
    for _ in range(50):  # 0.1 s at 1 rad/s; accelerometer says level
        est = cf_step(est, gyro_y=1.0, accel_x=0.0, accel_z=G, tau_s=0.5, dt=0.002)
    assert 0.08 < est < 0.1


def test_cf_single_step_formula() -> None:
    a = 0.5 / (0.5 + 0.002)
    expected = a * (0.1 + 0.3 * 0.002) + (1 - a) * math.atan2(-1.0, 9.0)
    assert cf_step(0.1, gyro_y=0.3, accel_x=1.0, accel_z=9.0, tau_s=0.5, dt=0.002) == pytest.approx(
        expected
    )


def test_imu_reading_identity() -> None:
    gyro, accel = imu_reading(
        gyro_true=(0.1, 0.2, 0.3),
        accel_true=(1.0, 2.0, 3.0),
        mount_pitch_rad=0.0,
        gyro_bias=(0.0, 0.0, 0.0),
        gyro_noise=(0.0, 0.0, 0.0),
        accel_noise=(0.0, 0.0, 0.0),
        gyro_range=10.0,
        accel_range=40.0,
    )
    assert gyro == (0.1, 0.2, 0.3)
    assert accel == (1.0, 2.0, 3.0)


def test_imu_mounting_offset_biases_accel_tilt() -> None:
    """A board mounted 1 degree nose-down reads 1 degree more pitch at true level."""
    offset = math.radians(1.0)
    _, accel = imu_reading(
        gyro_true=(0.0, 0.0, 0.0),
        accel_true=(0.0, 0.0, G),
        mount_pitch_rad=offset,
        gyro_bias=(0.0, 0.0, 0.0),
        gyro_noise=(0.0, 0.0, 0.0),
        accel_noise=(0.0, 0.0, 0.0),
        gyro_range=10.0,
        accel_range=40.0,
    )
    assert accel_tilt(accel[0], accel[2]) == pytest.approx(offset, abs=1e-12)


def test_imu_bias_noise_and_clamp() -> None:
    gyro, accel = imu_reading(
        gyro_true=(20.0, 0.0, -0.5),
        accel_true=(0.0, -100.0, G),
        mount_pitch_rad=0.0,
        gyro_bias=(0.0, 0.01, 0.02),
        gyro_noise=(0.0, 0.1, 0.0),
        accel_noise=(0.5, 0.0, 0.0),
        gyro_range=8.73,
        accel_range=39.2,
    )
    assert gyro[0] == 8.73  # clamped at full scale
    assert gyro[1] == pytest.approx(0.11)
    assert gyro[2] == pytest.approx(-0.48)
    assert accel[0] == pytest.approx(0.5)
    assert accel[1] == -39.2


@pytest.mark.parametrize(
    ("angle", "sign", "cpr", "expected"),
    [
        (0.0, 1.0, 1320, 0),
        (2 * math.pi, 1.0, 1320, 1320),
        (-math.pi, 1.0, 1320, -660),
        (math.pi, -1.0, 1320, -660),
        (0.0024, 1.0, 1320, 1),
    ],
)
def test_encoder_counts(angle: float, sign: float, cpr: int, expected: int) -> None:
    assert encoder_counts(angle, sign, cpr) == expected


def test_wheel_angle_from_counts() -> None:
    assert wheel_angle_from_counts(1320 + 7, 7, 1320) == pytest.approx(2 * math.pi)


@pytest.mark.parametrize("angle", [0.0, 3.0, 3.2, -3.2, 7.0, -100.0, math.pi])
def test_wrap_angle(angle: float) -> None:
    w = wrap_angle(angle)
    assert -math.pi < w <= math.pi
    assert math.cos(w) == pytest.approx(math.cos(angle))
    assert math.sin(w) == pytest.approx(math.sin(angle), abs=1e-12)


def test_odometry_yaw_turning_left() -> None:
    # right wheel rolled 1 rad more than left: yaw = r * 1 / track, counter-clockwise (positive)
    assert odometry_yaw(0.0, 1.0, wheel_radius=0.034, track_width=0.14) == pytest.approx(
        0.034 / 0.14
    )


def test_odometry_yaw_ignores_common_rotation() -> None:
    assert odometry_yaw(5.0, 5.0, wheel_radius=0.034, track_width=0.14) == 0.0


def test_estimation_imports_only_math() -> None:
    heavy = "('numpy', 'mujoco', 'gymnasium', 'pydantic')"
    code = (
        "import sys; import topple.estimation; "
        f"print(','.join(m for m in {heavy} if m in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
