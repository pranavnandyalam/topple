import array
import copy
import math
import pickle
import struct
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from fractions import Fraction
from typing import Any

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
    controller_obs,
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


# ------------------------------------------------------------------ controller_obs (contract §6)


def features_of(obs: BalancerObs) -> list[float]:
    return [obs.pitch, obs.pitch_rate, obs.yaw, obs.yaw_rate, *obs.wheel_pos, *obs.wheel_vel,
            *obs.last_action]  # fmt: skip


def test_controller_obs_widens_the_env_vector_exactly() -> None:
    arr = np.array([0.1, -0.2, 0.3, 0.05, 1.0, 1.1, 2.0, 2.2, 0.1, -0.1], dtype=np.float32)
    obs = controller_obs(arr, 0.5)
    assert obs.t == 0.5
    assert features_of(obs) == [float(v) for v in arr]
    assert all(type(v) is float for v in features_of(obs))


def test_controller_obs_quantizes_wider_input_to_float32() -> None:
    wide = np.array([0.1, -0.2, 0.3, 0.05, 1.0, 1.1, 2.0, 2.2, 0.1, -0.1])  # float64
    obs = controller_obs(wide, 0.0)
    assert features_of(obs) == [float(np.float32(v)) for v in wide]
    assert obs.pitch != 0.1  # 0.1 is not a float32 value: the controller never sees it
    assert controller_obs(list(wide), 0.0) == obs


def test_controller_obs_keeps_t_unquantized() -> None:
    t = 0.01 * 3  # 0.030000000000000002, not a float32 value
    assert controller_obs(np.zeros(10, np.float32), t).t == t


@pytest.mark.parametrize("bad", [np.zeros(9, np.float32), np.zeros(11), np.zeros((2, 10)), []])
def test_controller_obs_rejects_wrong_shapes(bad: Any) -> None:
    with pytest.raises(ValueError, match="10"):
        controller_obs(bad, 0.0)


@given(
    st.lists(st.floats(width=64), min_size=10, max_size=10),
    st.floats(min_value=0.0, max_value=1e4, allow_nan=False),
)
def test_controller_obs_survives_the_wire_bit_for_bit(values: list[float], t: float) -> None:
    """In-process and out-of-process controllers receive identical bits (contract §6)."""
    obs = controller_obs(np.asarray(values), t)
    for got, raw in zip(features_of(obs), values, strict=True):
        with np.errstate(over="ignore"):
            want = np.float32(raw)  # overflow to inf is the float32 cast, as in the env
        assert struct.pack("<d", got) == struct.pack("<d", float(want))
    assert BalancerObs.from_wire(obs.to_wire()).to_wire() == obs.to_wire()


# ------------------------------------------------------------------ actions (contract §7)


@pytest.mark.parametrize(
    "action",
    [
        [0.5, -0.5],
        (0.5, -0.5),
        np.array([0.5, -0.5]),
        np.array([0.5, -0.5], dtype=np.float32),
        [1, 0],
        (np.float32(0.5), 0),
        range(2),
        array.array("d", [0.5, -0.5]),
        memoryview(np.array([0.5, -0.5])),
        [Fraction(1, 2), np.int64(-1)],
    ],
)
def test_validate_action_accepts(action: object) -> None:
    out = validate_action(action, dim=2)
    assert isinstance(out, tuple)
    assert all(type(v) is float for v in out)
    assert len(out) == 2


class Endless:
    """Re-iterable (not an iterator itself) and never runs out."""

    def __init__(self) -> None:
        self.served = 0

    def __iter__(self) -> Iterator[float]:
        while True:
            self.served += 1
            yield 0.0


class Raises:
    """A container whose own methods fail with an arbitrary exception."""

    def __init__(self, where: str) -> None:
        self.where = where

    def __getattr__(self, name: str) -> Any:
        if self.where == "ndim" and name == "ndim":
            raise RuntimeError("ndim exploded")
        raise AttributeError(name)

    def __iter__(self) -> Iterator[float]:
        if self.where == "iter":
            raise RuntimeError("iter exploded")
        yield 0.0
        raise LookupError("next exploded")


class BadReal(float):
    def __float__(self) -> float:
        raise OverflowError("cannot convert")


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
        [10**400, 0.0],
        (0.0, -(10**309)),
        np.array([10**400, 0], dtype=object),
        [Fraction(10**400), 0.0],
        (v for v in (0.1, 0.2)),
        iter([0.1, 0.2]),
        map(float, [1, 2]),
        {0.1, 0.2},
        frozenset({0.1, 0.2}),
        bytearray(b"\x01\x00"),
        b"\x01\x00",
        range(10**12),
        Endless(),
        Raises("ndim"),
        Raises("iter"),
        Raises("next"),
        [BadReal(0.5), 0.0],
    ],
    ids=lambda a: type(a).__name__,
)
def test_validate_action_rejects(action: object) -> None:
    with pytest.raises(ActionError):
        validate_action(action, dim=2)


def test_validate_action_reads_at_most_one_item_too_many() -> None:
    endless = Endless()
    with pytest.raises(ActionError, match="more than 2"):
        validate_action(endless, dim=2)
    assert endless.served == 3


def test_validate_action_never_runs_a_returned_generator() -> None:
    ran: list[bool] = []

    def lazy() -> Iterator[float]:
        ran.append(True)
        yield from (0.1, 0.2)

    with pytest.raises(ActionError, match=r"generator|iterator"):
        validate_action(lazy(), dim=2)
    assert ran == []


class Interrupts:
    """A container whose iteration raises a BaseException that is not an Exception."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def __iter__(self) -> Iterator[float]:
        raise self.exc


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(3), GeneratorExit()])
def test_validate_action_lets_base_exceptions_through(exc: BaseException) -> None:
    """Only Exceptions become ActionError; the harness decides what a BaseException means."""
    with pytest.raises(type(exc)):
        validate_action(Interrupts(exc), dim=2)


def test_validate_action_chains_the_original_error() -> None:
    with pytest.raises(ActionError) as caught:
        validate_action([10**400, 0.0], dim=2)
    assert isinstance(caught.value.__cause__, OverflowError)


scalars = st.one_of(
    st.integers(),
    st.floats(),
    st.booleans(),
    st.none(),
    st.text(max_size=3),
    st.complex_numbers(),
    st.fractions(),
)


@given(st.one_of(scalars, st.lists(scalars, max_size=4), st.tuples(scalars, scalars)))
def test_validate_action_fails_only_with_action_error(action: object) -> None:
    """Whatever a controller returns, the harness sees floats or an ActionError, nothing else."""
    try:
        out = validate_action(action, dim=2)
    except ActionError:
        return
    assert len(out) == 2
    assert all(type(v) is float and math.isfinite(v) for v in out)


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


def make_spec(**extras: Any) -> TaskSpec:
    return TaskSpec(
        challenge="balance-v1",
        obs_schema=OBS_SCHEMA_ID,
        obs_features=FEATURES,
        action=BALANCE_V1_ACTION,
        control_dt=0.01,
        episode_seconds=10.0,
        robot=RobotGeometry(wheel_radius_m=0.034, track_width_m=0.14),
        **extras,
    )


def test_task_spec_extras_are_an_immutable_copy() -> None:
    source = {"target_x": 1.5}
    spec = make_spec(extras=source)
    source["target_x"] = 99.0  # the caller's dict is not the spec's
    assert spec.extras == {"target_x": 1.5}
    with pytest.raises(TypeError):
        spec.extras["target_x"] = 2.0  # type: ignore[index]
    assert make_spec().extras == {}


def test_task_spec_extras_are_coerced_and_checked() -> None:
    spec = make_spec(extras={"gain": 2, "bias": np.float32(0.5)})
    assert spec.extras == {"gain": 2.0, "bias": 0.5}
    assert all(type(v) is float for v in spec.extras.values())
    with pytest.raises(TypeError, match="real number"):
        make_spec(extras={"gain": "2"})
    with pytest.raises(TypeError, match="real number"):
        make_spec(extras={"flag": True})
    with pytest.raises(TypeError, match="str"):
        make_spec(extras={3: 1.0})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_task_spec_extras_must_be_finite(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_spec(extras={"target_x": value})


def test_task_spec_extras_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        make_spec(extras=[("target_x", 1.0)])


@pytest.mark.parametrize("control_dt", [0.0, -0.01, math.nan, math.inf])
def test_task_spec_rejects_bad_control_dt(control_dt: float) -> None:
    with pytest.raises(ValueError, match="control_dt"):
        TaskSpec(
            challenge="balance-v1",
            obs_schema=OBS_SCHEMA_ID,
            obs_features=FEATURES,
            action=BALANCE_V1_ACTION,
            control_dt=control_dt,
            episode_seconds=10.0,
            robot=RobotGeometry(wheel_radius_m=0.034, track_width_m=0.14),
        )


def test_task_spec_is_hashable_and_round_trips() -> None:
    spec = make_spec(extras={"target_x": 1.5})
    twin = TaskSpec.from_dict(spec.to_dict())
    assert twin == spec
    assert hash(twin) == hash(spec)
    assert len({spec, twin, make_spec()}) == 2
    assert spec.to_dict()["extras"] == {"target_x": 1.5}


def test_task_spec_pickles_and_copies() -> None:
    """Vectorized training ships specs between processes; immutability must not break that."""
    spec = make_spec(extras={"target_x": 1.5})
    for clone in (pickle.loads(pickle.dumps(spec)), copy.deepcopy(spec), copy.copy(spec)):
        assert clone == spec
        with pytest.raises(TypeError):
            clone.extras["target_x"] = 2.0  # type: ignore[index]


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
