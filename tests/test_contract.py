import math
import struct
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from topple.contract import (
    BALANCE_V1_ACTION,
    FEATURES,
    OBS_SCHEMA_ID,
    ActionError,
    ActionSpec,
    BalancerObs,
    Controller,
    Observation,
    RobotGeometry,
    TaskSpec,
    clip_action,
    validate_action,
)


def make_obs(**overrides: object) -> BalancerObs:
    base: dict[str, object] = {
        "t": 0.5,
        "pitch": 0.01,
        "pitch_rate": -0.2,
        "yaw": 0.3,
        "yaw_rate": 0.05,
        "wheel_pos": (1.0, 1.1),
        "wheel_vel": (2.0, 2.2),
        "last_action": (0.1, -0.1),
    }
    base.update(overrides)
    return BalancerObs(**base)  # type: ignore[arg-type]


def test_observation_alias() -> None:
    assert Observation is BalancerObs
    assert OBS_SCHEMA_ID == "balancer.obs.v1"


def test_features_order_is_frozen() -> None:
    assert FEATURES == (
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


def test_obs_is_immutable() -> None:
    obs = make_obs()
    with pytest.raises(FrozenInstanceError):
        obs.pitch = 1.0  # type: ignore[misc]


def test_obs_coerces_numeric_types_to_float() -> None:
    obs = make_obs(pitch=np.float32(0.25), wheel_pos=(np.int64(1), 2), t=1)
    assert type(obs.pitch) is float
    assert type(obs.t) is float
    assert all(type(v) is float for v in obs.wheel_pos)


def test_obs_rejects_wrong_pair_length() -> None:
    with pytest.raises(ValueError, match="wheel_vel"):
        make_obs(wheel_vel=(1.0, 2.0, 3.0))


def test_to_array_layout() -> None:
    arr = make_obs().to_array()
    assert arr.dtype == np.float32
    assert arr.shape == (len(FEATURES),)
    np.testing.assert_allclose(
        arr, [0.01, -0.2, 0.3, 0.05, 1.0, 1.1, 2.0, 2.2, 0.1, -0.1], rtol=1e-6
    )


def test_from_array_round_trip_is_float32_canonical() -> None:
    obs = make_obs()
    back = BalancerObs.from_array(obs.to_array(), t=obs.t)
    assert back.t == obs.t
    np.testing.assert_array_equal(back.to_array(), obs.to_array())


def test_from_array_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="10"):
        BalancerObs.from_array(np.zeros(9, dtype=np.float32), t=0.0)


finite = st.floats(allow_nan=False, allow_infinity=False, width=64)
any_float = st.floats(width=64)


@given(st.lists(any_float, min_size=11, max_size=11))
def test_wire_round_trip_is_bitwise_exact(values: list[float]) -> None:
    t, p, pr, y, yr, wpl, wpr, wvl, wvr, lal, lar = values
    obs = BalancerObs(t, p, pr, y, yr, (wpl, wpr), (wvl, wvr), (lal, lar))
    back = BalancerObs.from_wire(obs.to_wire())
    for a, b in zip(
        (
            back.t,
            back.pitch,
            back.pitch_rate,
            back.yaw,
            back.yaw_rate,
            *back.wheel_pos,
            *back.wheel_vel,
            *back.last_action,
        ),
        values,
        strict=True,
    ):
        assert math.copysign(1.0, a) == math.copysign(1.0, b)
        assert (a == b) or (math.isnan(a) and math.isnan(b))


def test_wire_is_88_bytes_little_endian() -> None:
    wire = make_obs(t=1.0).to_wire()
    assert len(wire) == 11 * 8
    assert wire[:8] == struct.pack("<d", 1.0)


def test_from_wire_rejects_bad_length() -> None:
    with pytest.raises(ValueError, match="88"):
        BalancerObs.from_wire(b"\x00" * 80)


@pytest.mark.parametrize(
    "action",
    [
        [0.5, -0.5],
        (0.5, -0.5),
        np.array([0.5, -0.5]),
        np.array([0.5, -0.5], dtype=np.float32),
        [1, 0],
        (np.float32(0.5), 0),
    ],
)
def test_validate_action_accepts(action: object) -> None:
    out = validate_action(action, dim=2)
    assert isinstance(out, tuple)
    assert all(type(v) is float for v in out)
    assert len(out) == 2


@pytest.mark.parametrize(
    "action",
    [
        [0.5],
        [0.5, 0.5, 0.5],
        np.zeros((2, 1)),
        [float("nan"), 0.0],
        [float("inf"), 0.0],
        ["0.5", 0.0],
        [True, 0.0],
        [np.bool_(True), 0.0],
        None,
        0.5,
        {"left": 0.5, "right": 0.5},
        [complex(1, 0), 0.0],
    ],
)
def test_validate_action_rejects(action: object) -> None:
    with pytest.raises(ActionError):
        validate_action(action, dim=2)


def test_clip_action() -> None:
    clipped, saturated = clip_action((1.7, -0.2))
    assert clipped == (1.0, -0.2)
    assert saturated
    assert clip_action((0.5, -0.98)) == ((0.5, -0.98), False)
    assert clip_action((0.995, 0.0))[1]


def test_task_spec_round_trips_through_dict() -> None:
    spec = TaskSpec(
        challenge="balance-v1",
        obs_schema=OBS_SCHEMA_ID,
        obs_features=FEATURES,
        action=BALANCE_V1_ACTION,
        control_dt=0.01,
        episode_seconds=10.0,
        robot=RobotGeometry(wheel_radius_m=0.034, track_width_m=0.14),
    )
    assert TaskSpec.from_dict(spec.to_dict()) == spec
    assert spec.max_steps == 1000


def test_task_spec_rejects_non_integer_steps() -> None:
    with pytest.raises(ValueError, match="multiple"):
        TaskSpec(
            challenge="balance-v1",
            obs_schema=OBS_SCHEMA_ID,
            obs_features=FEATURES,
            action=BALANCE_V1_ACTION,
            control_dt=0.03,
            episode_seconds=10.0,
            robot=RobotGeometry(wheel_radius_m=0.034, track_width_m=0.14),
        )


def test_action_spec() -> None:
    assert ActionSpec(id="diffdrive.effort.v1", names=("left", "right")) == BALANCE_V1_ACTION
    assert BALANCE_V1_ACTION.dim == 2


def test_controller_protocol_runtime_check() -> None:
    class Good:
        def reset(self, rng_seed: int) -> None: ...
        def act(self, obs: BalancerObs) -> tuple[float, float]:
            return (0.0, 0.0)

    class Bad:
        def act(self, obs: BalancerObs) -> tuple[float, float]:
            return (0.0, 0.0)

    assert isinstance(Good(), Controller)
    assert not isinstance(Bad(), Controller)


def test_contract_import_is_lightweight() -> None:
    code = (
        "import sys, topple.contract; "
        "heavy = [m for m in ('numpy', 'mujoco', 'gymnasium', 'pydantic') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
