import json
import math
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest

from topple.contract import BalancerObs, Controller
from topple.controllers.lqr import DEFAULT_GAINS, LQRController, LQRGains
from topple.design.lqr import LQRDesign, design_lqr, lqr_gain, round_gains, solve_dare
from topple.envs.balance import BalanceEnv

CONTROL_DT = 0.01


@pytest.fixture(scope="module")
def design() -> LQRDesign:
    return design_lqr()


def obs(
    pitch: float = 0.0,
    pitch_rate: float = 0.0,
    yaw: float = 0.0,
    yaw_rate: float = 0.0,
    wheel_pos: tuple[float, float] = (0.0, 0.0),
    wheel_vel: tuple[float, float] = (0.0, 0.0),
) -> BalancerObs:
    return BalancerObs(0.0, pitch, pitch_rate, yaw, yaw_rate, wheel_pos, wheel_vel)


def survives(env: BalanceEnv, controller: LQRController, seed: int) -> bool:
    obs_arr, info = env.reset(seed=seed)
    controller.reset(seed)
    while True:
        action = controller.act(BalancerObs.from_array(obs_arr, info["t"]))
        obs_arr, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return not terminated


# ---------------------------------------------------------------- gains and controller


def test_gains_json_round_trip() -> None:
    gains = LQRGains(k_pitch=1.5, k_pitch_rate=0.25, k_travel=0.125, k_speed=0.0625)
    assert LQRGains.from_mapping(json.loads(gains.to_json())) == gains
    assert json.loads(gains.to_json())["yaw_kp"] == 0.5  # heading PD shared with the PID baseline


def test_gains_from_mapping_coerces_to_float() -> None:
    gains = LQRGains.from_mapping(
        {"k_pitch": "6", "k_pitch_rate": 1, "k_travel": 0.2, "k_speed": 0.1, "yaw_kd": "0.07"}
    )
    assert gains.k_pitch == 6.0
    assert isinstance(gains.k_pitch_rate, float)
    assert gains.yaw_kd == 0.07


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "6"])
def test_gains_reject_non_finite_or_non_numeric(bad: object) -> None:
    with pytest.raises(ValueError, match="k_pitch"):
        LQRGains(k_pitch=bad, k_pitch_rate=1.0, k_travel=0.2, k_speed=0.1)  # type: ignore[arg-type]


def test_act_is_the_linear_control_law() -> None:
    g = LQRGains(k_pitch=7.0, k_pitch_rate=0.9, k_travel=0.2, k_speed=0.15, yaw_kp=0.5, yaw_kd=0.05)
    o = obs(
        pitch=0.03,
        pitch_rate=-0.4,
        yaw=0.2,
        yaw_rate=0.1,
        wheel_pos=(1.0, 3.0),
        wheel_vel=(-2.0, 4.0),
    )
    drive = 7.0 * 0.03 + 0.9 * -0.4 + 0.2 * 2.0 + 0.15 * 1.0
    turn = -(0.5 * 0.2 + 0.05 * 0.1)
    left, right = LQRController(g).act(o)
    assert left == pytest.approx(drive - turn)
    assert right == pytest.approx(drive + turn)


def test_act_signs() -> None:
    ctrl = LQRController()
    left, right = ctrl.act(obs(pitch=0.05))
    assert left == right > 0  # leaning forward: drive the wheels under the fall
    left, right = ctrl.act(obs(yaw=0.3))
    assert left > 0 > right  # turned left (counter-clockwise): steer back to the right


def test_controller_is_stateless_and_satisfies_protocol() -> None:
    ctrl = LQRController()
    assert isinstance(ctrl, Controller)
    assert ctrl.gains == DEFAULT_GAINS
    o = obs(pitch=0.01, pitch_rate=0.2, wheel_pos=(0.5, 0.4), wheel_vel=(1.0, 1.1))
    before = ctrl.act(o)
    ctrl.reset(12345)
    assert ctrl.act(o) == before


def test_controller_module_imports_only_stdlib() -> None:
    """Portable to firmware and Pyodide: the controller must not pull in numpy or MuJoCo."""
    heavy = "('numpy', 'mujoco', 'gymnasium', 'pydantic')"
    code = (
        "import sys; import topple.controllers.lqr; "
        f"print(','.join(m for m in {heavy} if m in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


# ---------------------------------------------------------------- Riccati solver


def test_solve_dare_matches_scalar_closed_form() -> None:
    a, b, q, r = 1.1, 0.5, 2.0, 1.0
    # b^2 P^2 + (r - q b^2 - a^2 r) P - q r = 0, positive root
    lin = r - q * b * b - a * a * r
    p_exact = (-lin + math.sqrt(lin * lin + 4 * b * b * q * r)) / (2 * b * b)
    p = solve_dare(np.array([[a]]), np.array([[b]]), np.array([[q]]), np.array([[r]]))
    assert p[0, 0] == pytest.approx(p_exact, rel=1e-10)
    k = lqr_gain(np.array([[a]]), np.array([[b]]), np.array([[r]]), p)
    assert k[0, 0] == pytest.approx(a * b * p_exact / (r + b * b * p_exact), rel=1e-10)


def test_solve_dare_satisfies_riccati_equation() -> None:
    a = np.array([[1.0, 0.1], [0.0, 1.0]])  # discrete double integrator
    b = np.array([[0.005], [0.1]])
    q = np.diag([1.0, 0.1])
    r = np.array([[0.5]])
    p = solve_dare(a, b, q, r)
    k = lqr_gain(a, b, r, p)
    residual = q + a.T @ p @ a - a.T @ p @ b @ k - p
    assert np.abs(residual).max() < 1e-9 * np.abs(p).max()
    assert np.abs(np.linalg.eigvals(a - b @ k)).max() < 1.0


def test_solve_dare_rejects_unstabilizable_system() -> None:
    with pytest.raises(ValueError, match="diverged"):
        solve_dare(np.array([[2.0]]), np.array([[0.0]]), np.eye(1), np.eye(1))


def test_solve_dare_reports_non_convergence() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_dare(np.array([[1.1]]), np.array([[0.5]]), np.eye(1), np.eye(1), max_iter=2)


# ---------------------------------------------------------------- design


def test_design_is_deterministic(design: LQRDesign) -> None:
    assert design_lqr() == design


def test_identified_model_is_physical(design: LQRDesign) -> None:
    a, b = np.array(design.a), np.array(design.b)
    assert a.shape == (4, 4)
    assert b.shape == (4, 1)
    # pitch integrates pitch rate over exactly one control period (a stale read would see 8 ms)
    assert a[0, 1] == pytest.approx(CONTROL_DT, rel=0.01)
    # the wheel angle is a pure integrator: nothing depends on where the wheels are
    np.testing.assert_allclose(a[:, 2], [0.0, 0.0, 1.0, 0.0], atol=1e-9)
    assert a[1, 0] > 0  # gravity: leaning forward accelerates the fall forward
    assert b[1, 0] < 0  # a forward command tips the top backwards (motor reaction torque)
    assert b[3, 0] > 0  # ... and spins the wheels forward
    assert np.abs(np.linalg.eigvals(a)).max() > 1.0  # open loop: falls over


def test_design_closes_a_stable_loop(design: LQRDesign) -> None:
    a, b = np.array(design.a), np.array(design.b)
    g = design.gains
    k = -np.array([[g.k_pitch, g.k_pitch_rate, g.k_travel, g.k_speed]])
    radius = np.abs(np.linalg.eigvals(a - b @ k)).max()
    assert design.spectral_radius == pytest.approx(radius)
    assert design.spectral_radius < 1.0


def test_default_gains_match_fresh_design(design: LQRDesign) -> None:
    fresh, shipped = asdict(design.gains), asdict(DEFAULT_GAINS)
    stale = {
        name: (shipped[name], value)
        for name, value in fresh.items()
        if not math.isclose(shipped[name], value, rel_tol=1e-3)
    }
    assert not stale, (
        f"DEFAULT_GAINS no longer match the robot (shipped, designed): {stale}. Re-run "
        "`uv run python -m topple.design.lqr` and paste its DEFAULT_GAINS line into "
        "python/topple/controllers/lqr.py."
    )


def test_heavier_pitch_weight_stiffens_pitch_feedback(design: LQRDesign) -> None:
    stiff = design_lqr(q=(40.0, 0.1, 0.5, 0.05))
    assert stiff.gains.k_pitch > design.gains.k_pitch
    assert stiff.spectral_radius < 1.0


@pytest.mark.parametrize(
    ("q", "r", "message"),
    [
        ((10.0, 0.1, 0.5), 1.0, "q must be 4"),
        ((10.0, -0.1, 0.5, 0.05), 1.0, "q must be 4"),
        ((10.0, math.nan, 0.5, 0.05), 1.0, "q must be 4"),
        ((10.0, 0.1, 0.5, 0.05), 0.0, "r must be"),
        ((10.0, 0.1, 0.5, 0.05), math.inf, "r must be"),
    ],
)
def test_design_rejects_bad_weights(q: tuple[float, ...], r: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        design_lqr(q=q, r=r)  # type: ignore[arg-type]


def test_design_refuses_weights_that_do_not_stabilize() -> None:
    with pytest.raises(ValueError, match="do not stabilize"):
        design_lqr(q=(0.0, 0.0, 0.0, 0.0))


def test_round_gains_keeps_four_significant_digits() -> None:
    gains = LQRGains(k_pitch=5.80276, k_pitch_rate=0.763537, k_travel=0.205832, k_speed=0.0167124)
    assert round_gains(gains) == LQRGains(
        k_pitch=5.803, k_pitch_rate=0.7635, k_travel=0.2058, k_speed=0.01671
    )


# ---------------------------------------------------------------- closed loop in the env


@pytest.mark.parametrize("preset", ["ideal", "clean"])
def test_balances_every_episode_without_pushes(preset: str) -> None:
    env = BalanceEnv(preset)
    fell = [seed for seed in range(20) if not survives(env, LQRController(), seed)]
    assert fell == []


@pytest.mark.slow
def test_survives_most_standard_episodes() -> None:
    env = BalanceEnv("standard")
    fell = [seed for seed in range(100) if not survives(env, LQRController(), seed)]
    assert len(fell) <= 20, f"{len(fell)}/100 standard episodes fell: seeds {fell}"
