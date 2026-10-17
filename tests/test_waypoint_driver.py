"""The §8.7 reference driver and the waypoint anchors (docs/design/phase-2.md §8.7).

Synthetic ``balancer-target.obs.v1`` sequences only: no env, no episodes. ``test_driver_golden``
pins the law's exact output on a fixed sequence; the other tests check each step of the law and
each anchor's definition.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import math
import struct
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from topple.contract import (
    BALANCE_V1_ACTION,
    TARGET_FEATURES,
    TARGET_OBS_SCHEMA_ID,
    BalancerObs,
    BalancerTargetObs,
    Controller,
    RobotGeometry,
    TaskSpec,
    validate_action,
)
from topple.controllers import anchors
from topple.controllers.anchors import (
    OVERSHOOT_M,
    BalanceInPlace,
    GatesOnly,
    SpinInPlace,
    balancer_obs,
    rotate,
)
from topple.controllers.lqr import DEFAULT_GAINS, LQRController
from topple.controllers.waypoint_pid import (
    Driver,
    advance_position_ref,
    balance_drive,
    bearing,
    choose_direction,
    desired_speed,
    ramp,
    reverse_bearing,
    turn_command,
)

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"
ANCHOR_FILES = ("balance_in_place", "spin", "creeper", "gates_only", "reckless")
R = 0.034
DT = 0.01
G = DEFAULT_GAINS


def waypoint_spec(radius: float = R, dt: float = DT) -> TaskSpec:
    return TaskSpec(
        challenge="waypoint-v1",
        obs_schema=TARGET_OBS_SCHEMA_ID,
        obs_features=TARGET_FEATURES,
        action=BALANCE_V1_ACTION,
        control_dt=dt,
        episode_seconds=30.0,
        robot=RobotGeometry(wheel_radius_m=radius, track_width_m=0.14),
    )


def make_obs(
    gate: tuple[float, float] = (1.0, 0.0),
    next_: tuple[float, float] | None = None,
    final: float = 0.0,
    *,
    t: float = 0.0,
    pitch: float = 0.0,
    pitch_rate: float = 0.0,
    yaw: float = 0.0,
    yaw_rate: float = 0.0,
    wheel_pos: tuple[float, float] = (0.0, 0.0),
    wheel_vel: tuple[float, float] = (0.0, 0.0),
) -> BalancerTargetObs:
    return BalancerTargetObs(
        t,
        pitch,
        pitch_rate,
        yaw,
        yaw_rate,
        wheel_pos,
        wheel_vel,
        (0.0, 0.0),
        gate,
        gate if next_ is None else next_,
        final,
    )


def golden_gate(k: int) -> tuple[float, float]:
    """Ahead (speeds up to v_max), then behind (held forward until slow, then reverses), then
    ahead again (held reverse until slow), then within 3 cm (bearing ignored for turning)."""
    if k < 100:
        return (2.0 - 0.004 * k, 0.1 + 0.002 * k)
    if k < 220:
        return (-0.5, 0.3)
    if k < 270:
        return (1.0, -0.4)
    return (0.02, 0.01)


def golden_sequence() -> Iterator[BalancerTargetObs]:
    for k in range(300):
        gate = golden_gate(k)
        yield BalancerTargetObs(
            0.01 * k,
            0.01 * ((k % 7) - 3),
            0.02 * ((k % 5) - 2),
            0.0,
            0.1 * ((k % 3) - 1),
            (0.03 * k, 0.031 * k),
            (3.0 - 0.01 * k, 3.1 - 0.01 * k),
            (0.0, 0.0),
            gate,
            gate,
            0.0,
        )


def run(controller: Controller[Any], sequence: Sequence[BalancerTargetObs], seed: int = 0) -> list:
    controller.reset(seed)
    return [tuple(controller.act(o)) for o in sequence]


def load(path: Path) -> ModuleType:
    name = f"topple_waypoint_example_{path.parent.name}_{path.stem}"
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


# --- golden ------------------------------------------------------------------------------------

GOLDEN_SHA256 = "1cabd656e70d5981227ac85e8edd6314c8ff7ff367cf96969b295585fc7b721d"
GOLDEN_ACTIONS = {
    0: (0.22709953832440646, 0.31903297147951487),
    99: (-1.691941077847165, -1.3860351574469525),
    150: (0.5059843588235309, -0.08201564117646915),
    220: (1.5777146078431372, 2.177714607843137),
    270: (0.335418137254901, 0.347418137254901),
    299: (0.6955710278239375, 0.6835710278239375),
}


def test_driver_golden() -> None:
    """The shipped driver's exact output on a fixed synthetic sequence (pins the §8.7 law)."""
    driver = Driver(0.30, 0.8)
    driver.reset(0)
    actions, directions = [], []
    for o in golden_sequence():
        actions.append(driver.act(o))
        directions.append(driver.direction)
    for k, expected in GOLDEN_ACTIONS.items():
        assert actions[k] == expected, k
    packed = b"".join(struct.pack("<2d", *a) for a in actions)
    assert hashlib.sha256(packed).hexdigest() == GOLDEN_SHA256
    switches = [k for k in range(1, 300) if directions[k] != directions[k - 1]]
    assert switches == [132, 252]
    assert directions[0] == 1


def test_driver_golden_is_deterministic_and_resets() -> None:
    sequence = list(golden_sequence())
    a, b = Driver(0.30, 0.8), Driver(0.30, 0.8)
    first = run(a, sequence, seed=0)
    assert run(b, sequence, seed=12345) == first  # rng_seed is irrelevant
    assert run(a, sequence, seed=1) == first  # reset restores the initial state
    assert (a.direction, a.v_ref, a.x_ref) != (1, 0.0, 0.0)
    a.reset(0)
    assert (a.direction, a.v_ref, a.x_ref) == (1, 0.0, 0.0)


# --- step 1: direction -------------------------------------------------------------------------


def test_bearing() -> None:
    assert bearing((1.0, 0.0)) == (1.0, 0.0)
    assert bearing((0.0, 2.0)) == (2.0, math.pi / 2)
    assert bearing((-1.0, 0.0)) == (1.0, math.pi)
    assert bearing((3.0, -4.0)) == (5.0, math.atan2(-4.0, 3.0))


@pytest.mark.parametrize(
    ("direction", "herr", "v_ref", "expected"),
    [
        (1, math.pi / 2 + 1e-9, 0.0, -1),  # just over 90 deg: reverse
        (1, -(math.pi / 2 + 1e-9), 0.0, -1),
        (1, math.pi / 2, 0.0, 1),  # exactly 90 deg: forward
        (-1, 0.3, 0.0, 1),
        (1, 3.0, 0.0499, -1),  # |v_ref| < 0.05: may switch
        (1, 3.0, 0.05, 1),  # |v_ref| = 0.05: held
        (1, 3.0, 0.2, 1),
        (-1, 0.0, -0.05, -1),
        (-1, 0.0, -0.0499, 1),
        (-1, 3.0, 0.2, -1),  # already reversed and still wanted
    ],
)
def test_direction_switches_only_below_005(
    direction: int, herr: float, v_ref: float, expected: int
) -> None:
    assert choose_direction(direction, herr, v_ref) == expected


def test_reverse_bearing_wraps_by_pi() -> None:
    assert reverse_bearing(math.radians(170)) == pytest.approx(math.radians(-10))
    assert reverse_bearing(math.radians(-170)) == pytest.approx(math.radians(10))
    assert reverse_bearing(math.pi) == 0.0
    for herr in (-3.0, -1.6, 1.6, 3.0):
        wrapped = reverse_bearing(herr)
        assert -math.pi / 2 < wrapped < math.pi / 2
        assert math.cos(wrapped) == pytest.approx(-math.cos(herr))


def test_driver_holds_direction_until_slow() -> None:
    driver = Driver(0.30, 0.8)
    driver.reset(0)
    ahead, behind = make_obs((3.0, 0.0)), make_obs((-1.0, 0.0))
    for _ in range(100):
        driver.act(ahead)
    assert driver.v_ref == pytest.approx(0.30)
    speeds = []
    while driver.direction == 1:
        speeds.append(driver.v_ref)
        driver.act(behind)
    # it switched on the first step that began with |v_ref| < 0.05, and not before
    assert all(v >= 0.05 for v in speeds[:-1])
    assert speeds[-1] < 0.05
    assert all(b < a for a, b in itertools.pairwise(speeds))  # braking all the way
    for _ in range(100):
        driver.act(behind)
    assert driver.direction == -1
    assert driver.v_ref == pytest.approx(-0.30)


def test_driver_reverses_from_rest_toward_a_gate_behind() -> None:
    driver = Driver(0.30, 0.8)
    driver.reset(0)
    left, right = driver.act(make_obs((-1.0, 0.2)))
    assert driver.direction == -1
    assert driver.v_ref == pytest.approx(-0.8 * DT)
    # tail bearing: atan2(0.2, -1) - pi < 0, so it turns right (right wheel slower)
    herr = reverse_bearing(math.atan2(0.2, -1.0))
    assert herr < 0
    assert right - left == pytest.approx(2 * max(0.8 * herr, -0.30))


# --- step 2: speed -----------------------------------------------------------------------------


def test_desired_speed_law() -> None:
    assert desired_speed(1, 0.1, 0.0, 1.2, 0.30) == pytest.approx(0.12)
    assert desired_speed(1, 5.0, 0.0, 1.2, 0.30) == 0.30  # clipped at v_max
    assert desired_speed(-1, 5.0, 0.0, 1.2, 0.30) == -0.30
    assert desired_speed(1, 0.1, math.pi / 3, 1.2, 0.30) == pytest.approx(1.2 * 0.1 * 0.25)
    assert desired_speed(1, 1.0, 2.0, 1.2, 0.30) == 0.0  # cos < 0: along = 0
    assert desired_speed(-1, 1.0, -2.0, 1.2, 0.30) == 0.0
    assert desired_speed(1, 0.0, 0.0, 1.2, 0.30) == 0.0


def test_ramp_is_limited_by_a_max() -> None:
    step = 0.8 * DT
    assert ramp(0.0, 0.3, step) == step
    assert ramp(0.0, -0.3, step) == -step
    assert ramp(0.1, 0.1 + step / 2, step) == 0.1 + step / 2  # reached exactly within one step
    driver = Driver(0.30, 0.8)
    driver.reset(0)
    speeds = []
    for _ in range(50):
        driver.act(make_obs((5.0, 0.0)))
        speeds.append(driver.v_ref)
    deltas = [b - a for a, b in itertools.pairwise([0.0, *speeds])]
    assert all(0 <= d <= step + 1e-15 for d in deltas)
    assert speeds[36] == pytest.approx(37 * step)  # 0.30 / 0.008 = 37.5 steps
    assert speeds[37] == 0.30
    assert speeds[-1] == 0.30


# --- step 3: position reference ----------------------------------------------------------------


def test_position_reference_integrates_and_clamps() -> None:
    assert advance_position_ref(0.5, 0.034, R, DT, 0.0) == pytest.approx(0.51)
    assert advance_position_ref(2.995, 0.34, R, DT, 0.0) == 3.0  # clamped at travel + 3
    assert advance_position_ref(-2.995, -0.34, R, DT, 0.0) == -3.0
    assert advance_position_ref(0.0, 0.0, R, DT, 100.0) == 97.0  # far behind: pulled to travel-3


def test_driver_clamps_x_ref_to_the_travel_window() -> None:
    driver = Driver(0.30, 0.8)
    driver.reset(0)
    driver.act(make_obs((1.0, 0.0), wheel_pos=(100.0, 102.0)))
    assert driver.x_ref == 98.0  # travel 101, window +- 3 (wheel angle is non-stationary)
    for _ in range(300):
        driver.act(make_obs((5.0, 0.0)))  # the wheels never turn: x_ref runs ahead
    assert driver.x_ref == 3.0


# --- step 4: balance ---------------------------------------------------------------------------


def test_balance_drive_formula() -> None:
    o = make_obs(
        pitch=0.02, pitch_rate=-0.3, wheel_pos=(4.0, 4.2), wheel_vel=(5.0, 5.4), yaw_rate=0.7
    )
    v_ref, x_ref = 0.2, 3.9
    travel, speed, w = 4.1, 5.2, v_ref / R
    expected = (
        w / 30 + 5.803 * 0.02 + 0.7635 * -0.3 + 0.2058 * (travel - x_ref) + 0.1671 * (speed - w)
    )
    assert balance_drive(G, o, v_ref, x_ref, R) == pytest.approx(expected, rel=1e-12)


def test_balance_drive_at_rest_is_the_lqr() -> None:
    o = make_obs(pitch=0.05, pitch_rate=0.4, wheel_pos=(0.3, -0.1), wheel_vel=(1.0, 2.0), yaw=0.2)
    left, right = LQRController().act(balancer_obs(o))
    turn = -(G.yaw_kp * o.yaw + G.yaw_kd * o.yaw_rate)
    assert balance_drive(G, o, 0.0, 0.0, R) - turn == left
    assert balance_drive(G, o, 0.0, 0.0, R) + turn == right


# --- step 5 and 6: turn and output -------------------------------------------------------------


def test_turn_command() -> None:
    assert turn_command(0.2, 1.0, 0.0) == pytest.approx(0.16)
    assert turn_command(1.0, 1.0, 0.0) == 0.30  # clipped
    assert turn_command(-1.0, 1.0, 0.0) == -0.30
    assert turn_command(1.0, 1.0, 2.0) == pytest.approx(0.30 - 0.12)  # damping is not clipped
    assert turn_command(1.0, 0.0299, 0.0) == 0.0  # within 3 cm the bearing is ignored
    assert turn_command(1.0, 0.0299, -1.0) == pytest.approx(0.06)
    assert turn_command(1.0, 0.03, 0.0) == 0.30  # at 3 cm it is not


def test_driver_output_is_drive_minus_and_plus_turn() -> None:
    driver = Driver(0.30, 0.8)
    driver.reset(0)
    o = make_obs((1.0, 1.0), pitch=0.01, yaw_rate=0.5, wheel_vel=(1.0, 1.0))
    left, right = driver.act(o)
    v_ref = 0.8 * DT
    x_ref = v_ref / R * DT
    drive = balance_drive(G, o, v_ref, x_ref, R)
    turn = turn_command(math.pi / 4, math.sqrt(2), 0.5)
    assert (left, right) == (drive - turn, drive + turn)
    assert driver.v_ref == v_ref
    assert driver.x_ref == x_ref


# --- construction ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"v_max": 0.0, "a_max": 0.8},
        {"v_max": -0.3, "a_max": 0.8},
        {"v_max": math.nan, "a_max": 0.8},
        {"v_max": 0.3, "a_max": math.inf},
        {"v_max": 0.3, "a_max": 0.8, "k_v": 0.0},
        {"v_max": 0.3, "a_max": 0.8, "wheel_radius_m": 0.0},
        {"v_max": 0.3, "a_max": 0.8, "control_dt": -0.01},
    ],
)
def test_driver_rejects_bad_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        Driver(**kwargs)


def test_driver_rejects_non_numbers() -> None:
    with pytest.raises(TypeError):
        Driver(True, 0.8)
    with pytest.raises(TypeError):
        Driver("0.3", 0.8)  # type: ignore[arg-type]


def test_for_task_takes_radius_and_dt_from_the_spec() -> None:
    driver = Driver.for_task(waypoint_spec(radius=0.05, dt=0.02), v_max=0.30, a_max=0.8)
    assert (driver.wheel_radius_m, driver.control_dt, driver.k_v) == (0.05, 0.02, 1.2)
    driver.reset(0)
    driver.act(make_obs((5.0, 0.0)))
    assert driver.v_ref == pytest.approx(0.8 * 0.02)
    assert driver.x_ref == pytest.approx(0.8 * 0.02 / 0.05 * 0.02)


# --- anchors -----------------------------------------------------------------------------------


def varied_sequence() -> list[BalancerTargetObs]:
    return [
        make_obs(
            golden_gate(k),
            t=0.01 * k,
            pitch=0.01 * ((k % 7) - 3),
            pitch_rate=0.02 * ((k % 5) - 2),
            yaw=0.05 * ((k % 9) - 4),
            yaw_rate=0.1 * ((k % 3) - 1),
            wheel_pos=(0.03 * k, 0.031 * k),
            wheel_vel=(3.0 - 0.01 * k, 3.1 - 0.01 * k),
        )
        for k in range(300)
    ]


def test_balance_in_place_is_the_lqr() -> None:
    sequence = varied_sequence()
    expected = [LQRController().act(balancer_obs(o)) for o in sequence]
    assert run(BalanceInPlace(), sequence) == expected


def test_balancer_obs_keeps_the_balance_features() -> None:
    o = varied_sequence()[17]
    assert balancer_obs(o).features() == o.features()[:10]
    assert balancer_obs(o).t == o.t
    assert isinstance(balancer_obs(o), BalancerObs)


def test_spin_in_place_is_lqr_drive_with_constant_turn() -> None:
    sequence = varied_sequence()
    actions = run(SpinInPlace(), sequence)
    for o, (left, right) in zip(sequence, actions, strict=True):
        travel = 0.5 * (o.wheel_pos[0] + o.wheel_pos[1])
        speed = 0.5 * (o.wheel_vel[0] + o.wheel_vel[1])
        drive = (
            G.k_pitch * o.pitch
            + G.k_pitch_rate * o.pitch_rate
            + G.k_travel * travel
            + G.k_speed * speed
        )
        assert (left, right) == (drive - 0.30, drive + 0.30)


@pytest.mark.parametrize(("factory", "v_max"), [(anchors.creeper, 0.12), (anchors.reckless, 0.60)])
def test_creeper_and_reckless_are_the_driver(factory: Any, v_max: float) -> None:
    spec = waypoint_spec()
    controller = factory(spec)
    assert isinstance(controller, Driver)
    assert (controller.v_max, controller.a_max, controller.k_v) == (v_max, 0.8, 1.2)
    assert (controller.wheel_radius_m, controller.control_dt) == (R, DT)
    sequence = varied_sequence()
    assert run(controller, sequence) == run(Driver(v_max, 0.8), sequence)
    controller.reset(0)
    for _ in range(200):
        controller.act(make_obs((10.0, 0.0)))
    assert controller.v_ref == v_max


def test_gates_only_is_the_driver_before_the_dock() -> None:
    sequence = varied_sequence()  # final = 0 throughout
    controller = anchors.gates_only(waypoint_spec())
    assert run(controller, sequence) == run(Driver(0.30, 0.8), sequence)
    assert (controller.driver.v_max, controller.driver.a_max) == (0.30, 0.8)


def robot_frame(
    point: tuple[float, float], pose: tuple[float, float], heading: float
) -> tuple[float, float]:
    return rotate((point[0] - pose[0], point[1] - pose[1]), -heading)


def test_gates_only_aims_past_the_dock_along_the_last_leg() -> None:
    """A world with the last gate at (2, 0) and the dock at (2, 1.5): the last leg points +y.
    Seen from any later pose, it aims at (2, 1.8), i.e. 0.30 m past the dock."""
    last_gate, dock = (2.0, 0.0), (2.0, 1.5)
    controller = anchors.gates_only(waypoint_spec())
    controller.reset(0)
    pose, heading = (0.3, -0.2), 0.4  # heading as reported by yaw
    approach = make_obs(
        robot_frame(last_gate, pose, heading), robot_frame(dock, pose, heading), 0.0, yaw=heading
    )
    assert controller.aim(approach) == approach.gate
    assert controller.last_leg == pytest.approx((0.0, 1.0))
    for pose, heading in [((1.9, 0.1), 1.2), ((2.3, 1.0), 2.9), ((1.0, 2.0), -2.5)]:
        o = make_obs(robot_frame(dock, pose, heading), None, 1.0, yaw=heading)
        expected = robot_frame((2.0, 1.5 + OVERSHOOT_M), pose, heading)
        assert controller.aim(o) == pytest.approx(expected, abs=1e-12)
    assert controller.last_leg == pytest.approx((0.0, 1.0))  # frozen once final = 1


def test_gates_only_steers_the_driver_at_the_aim_point() -> None:
    gates_only = anchors.gates_only(waypoint_spec())
    gates_only.reset(0)
    gates_only.act(make_obs((1.0, 0.0), (1.0, 1.0), 0.0))  # leg +y in the reset frame
    reference = Driver(0.30, 0.8)
    reference.reset(0)
    reference.act(make_obs((1.0, 0.0)))
    final = make_obs((0.2, 0.5), None, 1.0, yaw=0.0, pitch=0.01)
    assert gates_only.act(final) == reference.steer(final, (0.2, 0.5 + OVERSHOOT_M))


def test_gates_only_keeps_the_latest_leg() -> None:
    controller = GatesOnly(Driver(0.30, 0.8))
    controller.reset(0)
    controller.aim(make_obs((1.0, 0.0), (2.0, 0.0), 0.0))  # leg +x
    controller.aim(make_obs((1.0, 0.0), (1.0, -3.0), 0.0))  # then -y: the last leg
    assert controller.last_leg == pytest.approx((0.0, -1.0))
    controller.aim(make_obs((1.0, 0.0), (1.0, 0.0), 0.0))  # zero-length leg: ignored
    assert controller.last_leg == pytest.approx((0.0, -1.0))


def test_gates_only_dock_first_uses_the_leg_from_the_start() -> None:
    controller = GatesOnly(Driver(0.30, 0.8))
    controller.reset(0)
    o = make_obs((0.0, 2.0), None, 1.0, yaw=0.5)  # no gate seen before the dock
    aim = controller.aim(o)
    assert aim == pytest.approx((0.0, 2.0 + OVERSHOOT_M))
    assert controller.last_leg == pytest.approx(rotate((0.0, 1.0), 0.5))


def test_gates_only_on_the_dock_with_no_leg_aims_at_the_dock() -> None:
    controller = GatesOnly(Driver(0.30, 0.8))
    controller.reset(0)
    o = make_obs((0.0, 0.0), None, 1.0)
    assert controller.aim(o) == (0.0, 0.0)
    assert controller.last_leg is None


def test_gates_only_reset_forgets_the_leg() -> None:
    controller = GatesOnly(Driver(0.30, 0.8))
    controller.reset(0)
    controller.aim(make_obs((1.0, 0.0), (1.0, 1.0), 0.0))
    assert controller.last_leg is not None
    controller.reset(1)
    assert controller.last_leg is None
    assert (controller.driver.v_ref, controller.driver.x_ref) == (0.0, 0.0)


# --- shipped submissions -----------------------------------------------------------------------


def anchor_reference(stem: str) -> Controller[Any]:
    return {
        "balance_in_place": BalanceInPlace(),
        "spin": SpinInPlace(),
        "creeper": Driver(0.12, 0.8),
        "gates_only": GatesOnly(Driver(0.30, 0.8)),
        "reckless": Driver(0.60, 0.8),
    }[stem]


def final_leg_sequence() -> list[BalancerTargetObs]:
    """Approach the last gate, then the dock (final = 1) for the rest."""
    head = [make_obs((1.0 - 0.01 * k, 0.1), (1.0, 1.0), 0.0, yaw=0.01 * k) for k in range(50)]
    tail = [make_obs((0.5, 0.5 - 0.005 * k), None, 1.0, yaw=0.5) for k in range(100)]
    return head + tail


@pytest.mark.parametrize("stem", ANCHOR_FILES)
def test_anchor_submissions_match_their_definition(stem: str) -> None:
    spec = waypoint_spec()
    controller = load(SUBMISSIONS / "anchors" / f"{stem}.py").make_controller(spec)
    assert isinstance(controller, Controller)
    sequence = varied_sequence() + final_leg_sequence()
    actions = run(controller, sequence)
    assert actions == run(anchor_reference(stem), sequence)
    assert all(validate_action(a, 2) == a for a in actions)


def test_waypoint_example_is_the_shipped_driver() -> None:
    spec = waypoint_spec()
    controller = load(SUBMISSIONS / "waypoint_pid_example.py").make_controller(spec)
    assert isinstance(controller, Driver)
    assert isinstance(controller, Controller)
    assert (controller.v_max, controller.a_max, controller.k_v) == (0.30, 0.8, 1.2)
    sequence = list(golden_sequence())
    assert run(controller, sequence) == run(Driver(0.30, 0.8), sequence)
