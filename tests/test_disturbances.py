import copy
import itertools
import json
import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from topple.disturbances import (
    MAX_PUSH_FORCE_N,
    PRESETS,
    PushConfig,
    Scenario,
    make_scenario,
    push_force_world,
    validate_scenario,
)

TIMESTEP = 0.002
CONTROL_DT = 0.01
EPISODE_S = 10.0


def scenario(seed: int, preset: str = "standard") -> Scenario:
    return make_scenario(
        seed, PRESETS[preset], timestep=TIMESTEP, control_dt=CONTROL_DT, episode_seconds=EPISODE_S
    )


def test_presets_exist() -> None:
    assert set(PRESETS) >= {"ideal", "clean", "standard"}
    assert PRESETS["ideal"].sensors == "ideal"
    assert not PRESETS["ideal"].randomize
    assert PRESETS["clean"].pushes is None
    assert PRESETS["standard"].pushes is not None


def test_deterministic() -> None:
    assert scenario(11) == scenario(11)
    assert scenario(11) != scenario(12)


def test_json_round_trip() -> None:
    s = scenario(5)
    text = json.dumps(s.to_dict())
    assert Scenario.from_dict(json.loads(text)) == s


def test_ideal_has_nominal_model_and_no_pushes() -> None:
    s = scenario(3, "ideal")
    assert s.pushes == ()
    assert s.model.chassis_mass_scale == 1.0
    assert s.model.motor_gain == (1.0, 1.0)
    assert s.model.actuator_delay_s == 0.0
    assert s.sensors.gyro_bias_rads == (0.0, 0.0, 0.0)
    assert abs(s.init_pitch_rad) <= math.radians(3) + 1e-12


def test_clean_randomizes_without_pushes() -> None:
    s = scenario(3, "clean")
    assert s.pushes == ()
    assert s.model.chassis_mass_scale != 1.0


@pytest.mark.parametrize("seed", range(40))
def test_standard_ranges(seed: int) -> None:
    s = scenario(seed)
    cfg = PRESETS["standard"]
    r = cfg.randomization
    assert r.mass_scale[0] <= s.model.chassis_mass_scale <= r.mass_scale[1]
    assert r.com_x_offset_m[0] <= s.model.com_x_offset_m <= r.com_x_offset_m[1]
    assert round(s.model.actuator_delay_s * 1000) in r.actuator_delay_ms
    lo = r.motor_gain_common[0] * r.motor_gain_per_wheel[0]
    hi = r.motor_gain_common[1] * r.motor_gain_per_wheel[1]
    assert all(lo <= g <= hi for g in s.model.motor_gain)
    assert all(r.gyro_bias_rads[0] <= b <= r.gyro_bias_rads[1] for b in s.sensors.gyro_bias_rads)

    pushes = cfg.pushes
    assert pushes is not None
    assert len(s.pushes) in pushes.count
    starts = [p.start_step * CONTROL_DT for p in s.pushes]
    assert starts == sorted(starts)
    assert all(pushes.earliest_s - 1e-9 <= t <= pushes.latest_s + 1e-9 for t in starts)
    assert all(b - a >= pushes.min_spacing_s - CONTROL_DT for a, b in itertools.pairwise(starts))
    for p in s.pushes:
        assert p.n_substeps == round(pushes.duration_s / TIMESTEP)
        impulse = p.magnitude_n * p.n_substeps * TIMESTEP
        assert pushes.impulse_ns[0] - 1e-9 <= impulse <= pushes.impulse_ns[1] + 1e-9
        assert abs(p.lateral_ratio) <= pushes.lateral_fraction_max
        assert p.forward_sign in (-1.0, 1.0)


def test_push_count_varies_across_seeds() -> None:
    counts = {len(scenario(seed).pushes) for seed in range(60)}
    assert counts == {1, 2, 3}


def test_force_cap() -> None:
    cfg = PushConfig(impulse_ns=(2.0, 4.0), duration_s=0.1, max_force_n=50.0)
    with pytest.raises(ValueError, match="max_force_n"):
        PushConfig.model_validate({**cfg.model_dump(), "impulse_ns": (9.0, 10.0)})


def test_push_force_world_follows_heading_at_onset() -> None:
    s = scenario(1)
    p = s.pushes[0]
    f0 = push_force_world(p, yaw=0.0)
    assert f0[2] == 0.0
    assert math.hypot(f0[0], f0[1]) == pytest.approx(p.magnitude_n)
    assert math.copysign(1, f0[0]) == p.forward_sign
    f90 = push_force_world(p, yaw=math.pi / 2)
    assert f90[0] == pytest.approx(-f0[1], abs=1e-12)
    assert f90[1] == pytest.approx(f0[0], abs=1e-12)


def test_scenario_rejects_unknown_preset_name() -> None:
    with pytest.raises(ValueError, match="preset"):
        Scenario.from_dict({**scenario(1).to_dict(), "preset": "nope"})


# ------------------------------------------------------------------ Scenario.from_dict validation

MAX_STEPS = round(EPISODE_S / CONTROL_DT)


@settings(max_examples=60, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**53 - 1),
    st.sampled_from(["ideal", "clean", "standard"]),
)
def test_every_drawn_scenario_round_trips_and_validates(seed: int, preset: str) -> None:
    s = scenario(seed, preset)
    assert Scenario.from_dict(json.loads(json.dumps(s.to_dict())), max_steps=MAX_STEPS) == s


def test_from_dict_accepts_json_integers_for_real_fields() -> None:
    data = scenario(2, "ideal").to_dict()
    data["init_pitch_rad"] = 0
    data["model"]["chassis_mass_scale"] = 1
    assert Scenario.from_dict(data).init_pitch_rad == 0.0


def with_two_pushes() -> dict[str, Any]:
    seed = next(seed for seed in itertools.count() if len(scenario(seed).pushes) >= 2)
    return scenario(seed).to_dict()


def setter(*path: Any, value: Any) -> Callable[[dict[str, Any]], None]:
    def apply(data: dict[str, Any]) -> None:
        target = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return apply


def deleter(*path: Any) -> Callable[[dict[str, Any]], None]:
    def apply(data: dict[str, Any]) -> None:
        target = data
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return apply


def swap_pushes(data: dict[str, Any]) -> None:
    data["pushes"][0], data["pushes"][1] = data["pushes"][1], data["pushes"][0]


def duplicate_push(data: dict[str, Any]) -> None:
    data["pushes"][1]["start_step"] = data["pushes"][0]["start_step"]


BAD_SCENARIOS: dict[str, Callable[[dict[str, Any]], None]] = {
    "seed negative": setter("seed", value=-1),
    "seed too big": setter("seed", value=2**53),
    "seed str": setter("seed", value="1"),
    "pitch nan": setter("init_pitch_rad", value=math.nan),
    "pitch inf": setter("init_pitch_rad", value=math.inf),
    "pitch str": setter("init_pitch_rad", value="0.1"),
    "pitch bool": setter("init_pitch_rad", value=True),
    "pitch rate nan": setter("init_pitch_rate_rads", value=math.nan),
    "preset unhashable": setter("preset", value=["standard"]),
    "seed unhashable": setter("seed", value=[1]),
    "unknown top-level key": setter("surprise", value=1),
    "missing model": deleter("model"),
    "pushes not a list": setter("pushes", value={"0": 1}),
    "pushes unsorted": swap_pushes,
    "pushes share a step": duplicate_push,
    "push before the episode": setter("pushes", 0, "start_step", value=-1),
    "push after the episode": setter("pushes", 1, "start_step", value=MAX_STEPS),
    "push step not int": setter("pushes", 0, "start_step", value=150.5),
    "push step bool": setter("pushes", 0, "start_step", value=True),
    "push without substeps": setter("pushes", 0, "n_substeps", value=0),
    "push above the force cap": setter("pushes", 0, "magnitude_n", value=MAX_PUSH_FORCE_N + 1),
    "push negative magnitude": setter("pushes", 0, "magnitude_n", value=-1.0),
    "push magnitude nan": setter("pushes", 0, "magnitude_n", value=math.nan),
    "push sign not unit": setter("pushes", 0, "forward_sign", value=0.5),
    "push lateral above 1": setter("pushes", 0, "lateral_ratio", value=1.5),
    "push unknown key": setter("pushes", 0, "torque", value=1.0),
    "push missing key": deleter("pushes", 0, "magnitude_n"),
    "mass scale zero": setter("model", "chassis_mass_scale", value=0.0),
    "mass scale huge": setter("model", "chassis_mass_scale", value=100.0),
    "mass scale nan": setter("model", "chassis_mass_scale", value=math.nan),
    "com offset huge": setter("model", "com_x_offset_m", value=1.0),
    "friction scale tiny": setter("model", "wheel_friction_scale", value=0.01),
    "motor gain one wheel": setter("model", "motor_gain", value=[1.0]),
    "motor gain huge": setter("model", "motor_gain", value=[1.0, 5.0]),
    "frictionloss negative": setter("model", "frictionloss_scale", value=-0.1),
    "delay negative": setter("model", "actuator_delay_s", value=-0.002),
    "delay beyond the buffer": setter("model", "actuator_delay_s", value=0.5),
    "model unknown key": setter("model", "gravity", value=9.81),
    "imu offset huge": setter("sensors", "imu_pitch_offset_rad", value=1.0),
    "gyro bias two axes": setter("sensors", "gyro_bias_rads", value=[0.0, 0.0]),
    "gyro bias huge": setter("sensors", "gyro_bias_rads", value=[0.0, 5.0, 0.0]),
    "gyro bias nan": setter("sensors", "gyro_bias_rads", value=[0.0, math.nan, 0.0]),
    "sensors unknown key": setter("sensors", "accel_bias", value=[0.0, 0.0, 0.0]),
}


@pytest.mark.parametrize("mutate", BAD_SCENARIOS.values(), ids=BAD_SCENARIOS.keys())
def test_from_dict_rejects_bad_scenarios(mutate: Callable[[dict[str, Any]], None]) -> None:
    data = copy.deepcopy(with_two_pushes())
    Scenario.from_dict(copy.deepcopy(data), max_steps=MAX_STEPS)  # the unmutated one is fine
    mutate(data)
    with pytest.raises(ValueError, match="scenario"):
        Scenario.from_dict(data, max_steps=MAX_STEPS)


@pytest.mark.parametrize("data", [None, [], "scenario", 3])
def test_from_dict_rejects_non_mappings(data: Any) -> None:
    with pytest.raises(ValueError, match="scenario"):
        Scenario.from_dict(data)


def test_push_steps_are_only_bounded_above_when_the_episode_length_is_known() -> None:
    data = with_two_pushes()
    data["pushes"][1]["start_step"] = MAX_STEPS + 5
    assert Scenario.from_dict(data).pushes[1].start_step == MAX_STEPS + 5
    with pytest.raises(ValueError, match="scenario"):
        Scenario.from_dict(data, max_steps=MAX_STEPS)


def test_presets_stay_inside_the_scenario_bounds() -> None:
    """The randomization ranges and push config can only produce scenarios from_dict accepts."""
    for seed in range(200):
        Scenario.from_dict(scenario(seed).to_dict(), max_steps=MAX_STEPS)
    assert PushConfig().max_force_n == MAX_PUSH_FORCE_N


@pytest.mark.parametrize("field", ["init_pitch_rad", "init_pitch_rate_rads"])
def test_validate_scenario_rejects_a_non_finite_initial_state(field: str) -> None:
    bad = replace(scenario(1), **{field: math.inf})
    with pytest.raises(ValueError, match="scenario"):
        validate_scenario(bad)
