"""Discrete LQR design for the balancer, identified from the simulator itself.

``design_lqr`` linearizes the challenge robot's sagittal dynamics about upright by central finite
differences of the real MuJoCo model over one control period, then solves the discrete algebraic
Riccati equation by fixed-point iteration. Re-run it whenever the robot or its parameters change
and paste the printed line into ``topple/controllers/lqr.py``::

    uv run python -m topple.design.lqr

State ``x = (pitch, pitch_rate, mean wheel angle, mean wheel speed)`` is in observation units
(wheel angles are chassis-relative, like the encoders) and ``u`` is the common-mode motor command.
The law is ``u = -K x``; ``LQRGains`` stores ``-K``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import mujoco
import numpy as np
import numpy.typing as npt

from topple.challenge import load_challenge
from topple.controllers.lqr import LQRGains
from topple.robot import RobotBinding, bind_robot, build_model

FloatArray = npt.NDArray[np.float64]
Matrix = tuple[tuple[float, ...], ...]

DEFAULT_Q = (10.0, 0.1, 0.5, 0.05)  # pitch, pitch rate, wheel angle, wheel speed
DEFAULT_R = 1.0
N_STATES = 4
SETTLE_S = 0.5  # let the tire contacts reach their equilibrium penetration first
FD_EPS = 1e-4  # central-difference step, in state and command units


@dataclass(frozen=True)
class LQRDesign:
    gains: LQRGains
    a: Matrix  # 4x4 discrete-time state matrix over one control period
    b: Matrix  # 4x1 input matrix
    spectral_radius: float  # of the closed loop A - BK (< 1: stable)


@dataclass(frozen=True)
class _Plant:
    model: mujoco.MjModel
    data: mujoco.MjData
    robot: RobotBinding
    upright_qpos: FloatArray  # settled on the floor, upright, at rest
    substeps: int  # physics steps per control period


def _plant(challenge_id: str) -> _Plant:
    challenge = load_challenge(challenge_id)
    model = build_model(
        challenge.robot_xml, challenge.scene_xml, challenge.params, delay_buffer=False
    )
    robot = bind_robot(model)
    for dof in robot.wheel_dofadr:
        model.dof_frictionloss[dof] = 0.0  # Coulomb friction has no derivative at rest
    model.opt.tolerance = 0.0  # solve contacts exactly: no early-exit noise in the differences
    data = mujoco.MjData(model)
    timestep = float(model.opt.timestep)
    for _ in range(round(SETTLE_S / timestep)):
        mujoco.mj_step(model, data)
    substeps = round(challenge.config.control_dt / timestep)
    return _Plant(model, data, robot, data.qpos.copy(), substeps)


def _set_state(plant: _Plant, x: FloatArray) -> None:
    """Upright pose perturbed by ``x``, with the wheels rolling on the floor without slip."""
    pitch, pitch_rate, angle, speed = (float(v) for v in x)
    m, d, rb = plant.model, plant.data, plant.robot
    qa, da, radius = rb.root_qposadr, rb.root_dofadr, rb.wheel_radius_m
    mujoco.mj_resetData(m, d)
    d.qpos[:] = plant.upright_qpos
    d.qpos[qa] += radius * (angle + pitch)  # the axle sits where the wheels rolled it
    d.qpos[qa + 3 : qa + 7] = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
    d.qvel[da] = radius * (speed + pitch_rate)  # tire contact points at rest
    d.qvel[da + 4] = pitch_rate
    for sign, qadr, dof in zip(rb.wheel_sign, rb.wheel_qposadr, rb.wheel_dofadr, strict=True):
        d.qpos[qadr] = sign * angle
        d.qvel[dof] = sign * speed
    mujoco.mj_forward(m, d)


def _state(plant: _Plant) -> FloatArray:
    """Read ``x`` from qpos/qvel (``data.xmat`` still holds the start of the last substep)."""
    d, rb = plant.data, plant.robot
    qa = rb.root_qposadr
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, d.qpos[qa + 3 : qa + 7])
    pitch = math.atan2(-rot[6], math.hypot(rot[7], rot[8]))
    signs = rb.wheel_sign
    angle = sum(s * float(d.qpos[a]) for s, a in zip(signs, rb.wheel_qposadr, strict=True)) / 2
    speed = sum(s * float(d.qvel[v]) for s, v in zip(signs, rb.wheel_dofadr, strict=True)) / 2
    return np.array([pitch, float(d.qvel[rb.root_dofadr + 4]), angle, speed])


def _step(plant: _Plant, x: FloatArray, u: float) -> FloatArray:
    """The state one control period after ``x`` with common-mode command ``u`` held."""
    _set_state(plant, x)
    for sign, motor in zip(plant.robot.wheel_sign, plant.robot.motor_ids, strict=True):
        plant.data.ctrl[motor] = sign * u
    for _ in range(plant.substeps):
        mujoco.mj_step(plant.model, plant.data)
    return _state(plant)


def _identify(plant: _Plant) -> tuple[FloatArray, FloatArray]:
    """``x' = A x + B u`` about upright, by central differences."""
    h = FD_EPS
    columns = [
        (_step(plant, e, 0.0) - _step(plant, -e, 0.0)) / (2 * h) for e in np.eye(N_STATES) * h
    ]
    zero = np.zeros(N_STATES)
    b = (_step(plant, zero, h) - _step(plant, zero, -h)) / (2 * h)
    return np.column_stack(columns), b.reshape(N_STATES, 1)


def lqr_gain(a: FloatArray, b: FloatArray, r: FloatArray, p: FloatArray) -> FloatArray:
    """Optimal feedback ``K = (R + B'PB)^-1 B'PA`` for cost-to-go matrix ``P`` (``u = -K x``)."""
    k: FloatArray = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return k


def solve_dare(
    a: FloatArray,
    b: FloatArray,
    q: FloatArray,
    r: FloatArray,
    *,
    rtol: float = 1e-12,
    max_iter: int = 100_000,
) -> FloatArray:
    """Stabilizing solution ``P`` of the discrete algebraic Riccati equation.

    Iterates ``P <- Q + A'P(A - BK)`` from ``P = Q`` until no entry moves by more than ``rtol``
    times the largest one.
    """
    p = q.astype(np.float64)
    for _ in range(max_iter):
        nxt: FloatArray = q + a.T @ p @ (a - b @ lqr_gain(a, b, r, p))
        if not np.isfinite(nxt).all():
            raise ValueError("Riccati iteration diverged: (A, B) is not stabilizable")
        if np.abs(nxt - p).max() <= rtol * max(1.0, float(np.abs(nxt).max())):
            return nxt
        p = nxt
    raise RuntimeError(f"Riccati iteration did not converge in {max_iter} iterations")


def _weights(q: Sequence[float], r: float) -> tuple[FloatArray, FloatArray]:
    values = [float(v) for v in q]
    if len(values) != N_STATES or not all(math.isfinite(v) and v >= 0.0 for v in values):
        raise ValueError(f"q must be {N_STATES} finite, non-negative state weights, got {q!r}")
    if not (math.isfinite(r) and r > 0.0):
        raise ValueError(f"r must be a finite, positive command weight, got {r!r}")
    return np.diag(values), np.array([[float(r)]])


def _matrix(m: FloatArray) -> Matrix:
    return tuple(tuple(float(v) for v in row) for row in m)


def design_lqr(
    challenge: str = "balance-v1",
    q: tuple[float, float, float, float] = DEFAULT_Q,
    r: float = DEFAULT_R,
) -> LQRDesign:
    """Identify ``challenge``'s robot about upright and design the discrete LQR for ``Q, R``.

    ``q`` weighs (pitch, pitch rate, wheel angle, wheel speed); ``r`` weighs the command.
    """
    q_mat, r_mat = _weights(q, r)
    a, b = _identify(_plant(challenge))
    k = lqr_gain(a, b, r_mat, solve_dare(a, b, q_mat, r_mat))
    radius = float(np.abs(np.linalg.eigvals(a - b @ k)).max())
    if radius >= 1.0:
        raise ValueError(f"weights q={q}, r={r} do not stabilize the robot (radius {radius:.4f})")
    k_pitch, k_pitch_rate, k_travel, k_speed = (float(v) for v in -k[0])
    gains = LQRGains(k_pitch=k_pitch, k_pitch_rate=k_pitch_rate, k_travel=k_travel, k_speed=k_speed)
    return LQRDesign(gains=gains, a=_matrix(a), b=_matrix(b), spectral_radius=radius)


def round_gains(gains: LQRGains, digits: int = 4) -> LQRGains:
    """``gains`` rounded to ``digits`` significant digits, the precision ``DEFAULT_GAINS`` ships."""
    return LQRGains(
        **{f.name: float(f"{getattr(gains, f.name):.{digits}g}") for f in fields(gains)}
    )


if __name__ == "__main__":
    result = design_lqr()
    print(f"closed-loop spectral radius {result.spectral_radius:.6f} per control period")
    print(f"DEFAULT_GAINS = {round_gains(result.gains)!r}")
