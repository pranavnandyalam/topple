"""Sensor models and the state estimator (docs/contract.md §5).

Plain float math only (``math`` module): this file is the reference that firmware (C), the browser
arena (JS) and Pyodide port line for line. Every function is pure.
"""

from __future__ import annotations

import math


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi] using atan2 (never ``%``, which differs across languages)."""
    return math.atan2(math.sin(angle), math.cos(angle))


def accel_tilt(accel_x: float, accel_z: float) -> float:
    """Pitch implied by gravity alone: positive when the top leans toward +x."""
    return math.atan2(-accel_x, accel_z)


def cf_step(
    pitch: float, gyro_y: float, accel_x: float, accel_z: float, tau_s: float, dt: float
) -> float:
    """One complementary-filter update: gyro short-term, accelerometer long-term."""
    a = tau_s / (tau_s + dt)
    return a * (pitch + gyro_y * dt) + (1.0 - a) * accel_tilt(accel_x, accel_z)


def _clamp(value: float, limit: float) -> float:
    return min(max(value, -limit), limit)


def _rotate_y(v: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    """Express body-frame vector ``v`` in a board frame rotated by ``angle`` about y."""
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = v
    return (c * x - s * z, y, s * x + c * z)


def imu_reading(
    gyro_true: tuple[float, float, float],
    accel_true: tuple[float, float, float],
    mount_pitch_rad: float,
    gyro_bias: tuple[float, float, float],
    gyro_noise: tuple[float, float, float],
    accel_noise: tuple[float, float, float],
    gyro_range: float,
    accel_range: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """What the IMU reports: mounting rotation, then bias and noise, then full-scale clamp."""
    g = _rotate_y(gyro_true, mount_pitch_rad)
    a = _rotate_y(accel_true, mount_pitch_rad)
    gyro = (
        _clamp(g[0] + gyro_bias[0] + gyro_noise[0], gyro_range),
        _clamp(g[1] + gyro_bias[1] + gyro_noise[1], gyro_range),
        _clamp(g[2] + gyro_bias[2] + gyro_noise[2], gyro_range),
    )
    accel = (
        _clamp(a[0] + accel_noise[0], accel_range),
        _clamp(a[1] + accel_noise[1], accel_range),
        _clamp(a[2] + accel_noise[2], accel_range),
    )
    return gyro, accel


def encoder_counts(joint_angle: float, sign: float, cpr: int) -> int:
    """Quantized, chassis-relative wheel angle in encoder counts (positive = rolling forward)."""
    return round(sign * joint_angle * cpr / (2.0 * math.pi))


def wheel_angle_from_counts(counts: int, counts_at_reset: int, cpr: int) -> float:
    return (counts - counts_at_reset) * 2.0 * math.pi / cpr


def odometry_yaw(
    wheel_pos_l: float, wheel_pos_r: float, wheel_radius: float, track_width: float
) -> float:
    """Heading change from wheel odometry: no gyro drift and immune to pitch motion."""
    return wrap_angle(wheel_radius * (wheel_pos_r - wheel_pos_l) / track_width)
