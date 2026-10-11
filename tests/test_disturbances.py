import copy
import itertools
import json
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from topple.disturbances import (
    MAX_PUSH_FORCE_N,
    MAX_PUSH_IMPULSE_NS,
    MAX_PUSHES,
    PRESETS,
    EpisodeClock,
    Push,
    PushConfig,
    Scenario,
    check_push_schedule,
    make_scenario,
    push_force_world,
    validate_scenario,
)
from topple.rng import Stream, stream_rng

TIMESTEP = 0.002
CONTROL_DT = 0.01
EPISODE_S = 10.0
FRAME_SKIP = 5
MAX_STEPS = round(EPISODE_S / CONTROL_DT)
CLOCK = EpisodeClock(timestep=TIMESTEP, frame_skip=FRAME_SKIP, max_steps=MAX_STEPS)


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


def test_to_dict_hands_out_copies() -> None:
    """A Scenario is frozen: editing the dict it returns (trajectories embed one as
    meta.scenario) must never reach the scenario, its pushes included."""
    s = scenario(5)
    assert s.pushes
    before = copy.deepcopy(s)
    data = s.to_dict()
    data["pushes"][0]["magnitude_n"] = 49.0
    data["pushes"][0]["start_step"] = 0
    data["model"]["chassis_mass_scale"] = 1.9
    data["model"]["motor_gain"][0] = 1.9
    data["sensors"]["gyro_bias_rads"][0] = 0.09
    assert s == before
    assert s.to_dict() == before.to_dict() != data


def test_every_key_is_required() -> None:
    """Exact key sets (contract §9): to_dict always writes init_pitch_rate_rads, so a scenario
    without it is not one this code wrote."""
    data = scenario(5).to_dict()
    del data["init_pitch_rate_rads"]
    with pytest.raises(ValueError, match=r"missing keys \['init_pitch_rate_rads'\]"):
        Scenario.from_dict(data)


def test_ideal_has_nominal_model_and_no_pushes() -> None:
    s = scenario(3, "ideal")
    assert s.pushes == ()
    assert s.model.chassis_mass_scale == 1.0
    assert s.model.motor_gain == (1.0, 1.0)
    assert s.model.actuator_delay_s == 0.0
    assert s.sensors.gyro_bias_rads == (0.0, 0.0, 0.0)
    assert abs(s.init_pitch_rad) <= math.radians(3) + 1e-12


@pytest.mark.parametrize("seed", [0, 1000, 2**53 - 1])
def test_every_preset_draws_the_same_initial_pitch(seed: int) -> None:
    """§10: every preset, ``ideal`` included, draws its initial pitch U(-3°, 3°) from the INIT
    stream (pitch rate 0), so one seed starts every preset at the same pitch."""
    drawn = [scenario(seed, preset) for preset in PRESETS]
    assert len({s.init_pitch_rad for s in drawn}) == 1
    assert 0.0 < abs(drawn[0].init_pitch_rad) <= math.radians(3.0)
    assert {s.init_pitch_rate_rads for s in drawn} == {0.0}


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


def check_push_grid(seed: int) -> list[int]:
    """§10 on the 10 ms control grid: the first push at step >= 100 (1.0 s), starts >= 150
    steps (1.5 s) apart, the last at step <= 850: at or before 8.5 s."""
    steps = [p.start_step for p in scenario(seed).pushes]
    assert steps[0] >= 100
    assert steps[-1] <= 850
    assert all(b - a >= 150 for a, b in itertools.pairwise(steps))
    return steps


def test_push_starts_stay_on_the_documented_grid() -> None:
    """Start times are rounded to the control grid, so the last can land exactly on 8.50 s (it
    does for about 0.15% of seeds, public seed 1001 among them)."""
    last_on_bound = [seed for seed in range(5000) if check_push_grid(seed)[-1] == 850]
    assert last_on_bound
    assert check_push_grid(1001)[-1] == 850


@settings(max_examples=300, deadline=None)
@given(seed=st.integers(0, 2**53 - 1))
def test_push_starts_stay_on_the_documented_grid_for_any_seed(seed: int) -> None:
    check_push_grid(seed)


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


@settings(max_examples=60, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**53 - 1),
    st.sampled_from(["ideal", "clean", "standard"]),
)
def test_every_drawn_scenario_round_trips_and_validates(seed: int, preset: str) -> None:
    s = scenario(seed, preset)
    assert Scenario.from_dict(json.loads(json.dumps(s.to_dict())), clock=CLOCK) == s


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


def overlap_pushes(data: dict[str, Any]) -> None:
    """Start the second push 3 control steps (15 substeps) into the first one's 50 substeps."""
    data["pushes"][1]["start_step"] = data["pushes"][0]["start_step"] + 3


def push_fields(index: int, **values: Any) -> Callable[[dict[str, Any]], None]:
    def apply(data: dict[str, Any]) -> None:
        data["pushes"][index].update(values)

    return apply


def mixed_type_keys(*path: Any) -> Callable[[dict[str, Any]], None]:
    """Add unknown keys of mixed types, which cannot be sorted against each other."""

    def apply(data: dict[str, Any]) -> None:
        target = data
        for key in path:
            target = target[key]
        target[1] = 0
        target["zz"] = 0

    return apply


HUGE_INT = json.loads("1" + "0" * 400)  # valid JSON; does not fit a float

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
    "missing pitch rate": deleter("init_pitch_rate_rads"),
    "pushes not a list": setter("pushes", value={"0": 1}),
    "pushes unsorted": swap_pushes,
    "pushes share a step": duplicate_push,
    "pushes overlap": overlap_pushes,
    "push outlasts the episode": push_fields(  # the last push: no later one for it to overlap
        1, magnitude_n=0.0, n_substeps=MAX_STEPS * FRAME_SKIP + 1
    ),
    "push impulse above the cap": push_fields(0, magnitude_n=MAX_PUSH_FORCE_N, n_substeps=51),
    "push substeps not exact": setter("pushes", 0, "n_substeps", value=2**53 + 1),
    "push substeps huge int": setter("pushes", 0, "n_substeps", value=10**5000),
    "push step huge int": setter("pushes", 1, "start_step", value=10**5000),
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
    "pitch huge int": setter("init_pitch_rad", value=HUGE_INT),
    "pitch rate huge int": setter("init_pitch_rate_rads", value=HUGE_INT),
    "mass scale huge int": setter("model", "chassis_mass_scale", value=HUGE_INT),
    "motor gain huge int": setter("model", "motor_gain", value=[1.0, HUGE_INT]),
    "push lateral huge int": setter("pushes", 0, "lateral_ratio", value=HUGE_INT),
    "push magnitude huge int": setter("pushes", 0, "magnitude_n", value=-HUGE_INT),
    "gyro bias huge int": setter("sensors", "gyro_bias_rads", value=[0.0, HUGE_INT, 0.0]),
    "non-str top-level keys": mixed_type_keys(),
    "non-str push keys": mixed_type_keys("pushes", 0),
    "non-str model keys": mixed_type_keys("model"),
    "a single non-str key": setter(7, value=0),
}


@pytest.mark.parametrize("mutate", BAD_SCENARIOS.values(), ids=BAD_SCENARIOS.keys())
def test_from_dict_rejects_bad_scenarios(mutate: Callable[[dict[str, Any]], None]) -> None:
    data = copy.deepcopy(with_two_pushes())
    Scenario.from_dict(copy.deepcopy(data), clock=CLOCK)  # the unmutated one is fine
    mutate(data)
    with pytest.raises(ValueError, match="scenario"):
        Scenario.from_dict(data, clock=CLOCK)


JSON_LEAVES = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.integers(min_value=10**300, max_value=10**400)
    | st.floats()
    | st.text(max_size=5)
)
JSON_VALUES = st.recursive(
    JSON_LEAVES,
    lambda inner: (
        st.lists(inner, max_size=4)
        | st.dictionaries(st.text(max_size=3) | st.integers(), inner, max_size=4)
    ),
    max_leaves=12,
)


def json_paths(value: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Every key path inside a JSON-like value (the root excluded)."""
    if isinstance(value, dict):
        children = value.items()
    elif isinstance(value, list):
        children = enumerate(value)
    else:
        return []
    paths: list[tuple[Any, ...]] = []
    for key, child in children:
        paths += [(*prefix, key), *json_paths(child, (*prefix, key))]
    return paths


SCENARIO_PATHS = json_paths(with_two_pushes())


@settings(max_examples=300, deadline=None)
@given(st.sampled_from(SCENARIO_PATHS), JSON_VALUES)
def test_from_dict_raises_only_value_error(path: tuple[Any, ...], value: Any) -> None:
    """Whatever JSON-like value replaces any part of a valid scenario, the only failure is a
    ``ValueError`` (never ``TypeError``, ``OverflowError``, ``KeyError``...)."""
    data = with_two_pushes()
    setter(*path, value=value)(data)
    try:
        Scenario.from_dict(data, clock=CLOCK)
    except ValueError as exc:
        message = str(exc)
    else:
        return
    assert "scenario" in message


class LyingMapping(Mapping[str, Any]):
    """Has the right keys but fails with ``exc`` when a value is read."""

    def __init__(self, keys: list[str], exc: Exception) -> None:
        self.keys_, self.exc = keys, exc

    def __getitem__(self, key: str) -> Any:
        raise self.exc

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys_)

    def __len__(self) -> int:
        return len(self.keys_)


@pytest.mark.parametrize(
    "exc", [TypeError("nope"), OverflowError("too big")], ids=["type", "overflow"]
)
def test_from_dict_turns_any_type_or_overflow_error_into_value_error(exc: Exception) -> None:
    data = LyingMapping(list(with_two_pushes()), exc)
    with pytest.raises(ValueError, match="invalid scenario"):
        Scenario.from_dict(data)


@pytest.mark.parametrize("data", [None, [], "scenario", 3])
def test_from_dict_rejects_non_mappings(data: Any) -> None:
    with pytest.raises(ValueError, match="scenario"):
        Scenario.from_dict(data)


def test_push_steps_are_only_bounded_above_when_the_clock_is_known() -> None:
    data = with_two_pushes()
    data["pushes"][1]["start_step"] = MAX_STEPS + 5
    assert Scenario.from_dict(data).pushes[1].start_step == MAX_STEPS + 5
    with pytest.raises(ValueError, match="scenario"):
        Scenario.from_dict(data, clock=CLOCK)


# ------------------------------------------------------------------ push schedules (F1, F18)


def two_pushes(first: Push, second: Push) -> dict[str, Any]:
    data = with_two_pushes()
    data["pushes"] = [vars(first).copy(), vars(second).copy()]
    return data


def test_overlapping_pushes_are_rejected() -> None:
    """A push that starts while another is active would cut it short (the env runs one push at
    a time): (100, 50 substeps) then (103, 50) would deliver 65 of the 100 substeps stated."""
    data = two_pushes(Push(100, 50, 3.0, 1.0, 0.0), Push(103, 50, 3.0, 1.0, 0.0))
    with pytest.raises(ValueError, match=r"invalid scenario: .*overlap"):
        Scenario.from_dict(data, clock=CLOCK)
    with pytest.raises(ValueError, match="overlap"):
        check_push_schedule(Scenario.from_dict(data).pushes, frame_skip=FRAME_SKIP)


def test_back_to_back_pushes_are_accepted() -> None:
    """A push may start on the substep after the previous one ends: 15 substeps from step 100
    end exactly where step 103 begins."""
    first, second = Push(100, 15, 3.0, 1.0, 0.0), Push(103, 50, 3.0, -1.0, 0.0)
    parsed = Scenario.from_dict(two_pushes(first, second), clock=CLOCK)
    assert parsed.pushes == (first, second)
    check_push_schedule(parsed.pushes, frame_skip=FRAME_SKIP)
    with pytest.raises(ValueError, match="overlap"):
        check_push_schedule((replace(first, n_substeps=16), second), frame_skip=FRAME_SKIP)


def test_overlap_needs_the_frame_skip_to_be_seen() -> None:
    """Steps and substeps only compare through the frame skip: without it only the order and
    the one-push-per-step rule can be checked."""
    pushes = (Push(100, 50, 3.0, 1.0, 0.0), Push(103, 50, 3.0, 1.0, 0.0))
    check_push_schedule(pushes)
    with pytest.raises(ValueError, match="overlap"):
        check_push_schedule(pushes, frame_skip=FRAME_SKIP)
    with pytest.raises(ValueError, match="sorted by start_step"):
        check_push_schedule(pushes[::-1], frame_skip=FRAME_SKIP)


def test_the_impulse_cap_is_the_force_cap_held_for_a_standard_pulse() -> None:
    cfg = PushConfig()
    assert pytest.approx(MAX_PUSH_FORCE_N * cfg.duration_s) == MAX_PUSH_IMPULSE_NS
    assert 10 * cfg.impulse_ns[1] <= MAX_PUSH_IMPULSE_NS  # far above every calibrated push
    at_cap = push_fields(0, magnitude_n=MAX_PUSH_FORCE_N, n_substeps=50)
    data = with_two_pushes()
    at_cap(data)
    assert Scenario.from_dict(data, clock=CLOCK).pushes[0].n_substeps == 50


def test_a_sustained_push_is_bounded_by_its_impulse() -> None:
    """50 N held for the whole episode is 500 N·s, a thousand times a calibrated push."""
    data = with_two_pushes()
    data["pushes"] = data["pushes"][:1]
    push_fields(0, magnitude_n=MAX_PUSH_FORCE_N, n_substeps=MAX_STEPS * FRAME_SKIP)(data)
    with pytest.raises(ValueError, match="impulse"):
        Scenario.from_dict(data, clock=CLOCK)
    push_fields(0, magnitude_n=0.0)(data)  # a zero-force push may last the whole episode
    assert Scenario.from_dict(data, clock=CLOCK).pushes[0].n_substeps == MAX_STEPS * FRAME_SKIP


def test_a_push_lasts_no_longer_than_the_episode() -> None:
    """§9: with the clock known, ``n_substeps <= max_steps * frame_skip``. A zero-force push
    passes the impulse cap at any length and a lone push cannot overlap another, so only this
    bound refuses one that outlasts the episode."""
    limit = MAX_STEPS * FRAME_SKIP
    data = with_two_pushes()
    data["pushes"] = data["pushes"][:1]
    push_fields(0, start_step=0, magnitude_n=0.0, n_substeps=limit)(data)
    assert Scenario.from_dict(data, clock=CLOCK).pushes[0].n_substeps == limit
    push_fields(0, n_substeps=limit + 1)(data)
    with pytest.raises(
        ValueError, match=rf"pushes\[0\]\.n_substeps = {limit + 1} is outside \[1, {limit}\]"
    ):
        Scenario.from_dict(data, clock=CLOCK)
    assert Scenario.from_dict(data).pushes[0].n_substeps == limit + 1  # no clock, no such bound


@pytest.mark.parametrize("field", ["start_step", "n_substeps"])
def test_push_integers_must_be_exact_in_a_json_double(field: str) -> None:
    """Even without a clock: an integer JSON cannot carry exactly (or at all, beyond 4300
    digits) would make the scenario unwritable."""
    data = with_two_pushes()
    data["pushes"][1][field] = 2**53
    Scenario.from_dict(data)
    for value in (2**53 + 1, 10**5000):
        data["pushes"][1][field] = value
        with pytest.raises(ValueError, match=rf"pushes\[1\]\.{field} .*2\*\*53"):
            Scenario.from_dict(data)


def test_accepted_scenarios_are_writable() -> None:
    data = with_two_pushes()
    push_fields(1, n_substeps=MAX_STEPS * FRAME_SKIP, magnitude_n=0.0)(data)
    parsed = Scenario.from_dict(data, clock=CLOCK)
    assert Scenario.from_dict(json.loads(json.dumps(parsed.to_dict())), clock=CLOCK) == parsed


@pytest.mark.parametrize(
    "fields",
    [
        {"timestep": 0.0},
        {"timestep": math.nan},
        {"timestep": math.inf},
        {"frame_skip": 0},
        {"max_steps": 0},
        {"frame_skip": 5.0},
        {"max_steps": True},
    ],
)
def test_an_episode_clock_must_be_well_formed(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="clock"):
        EpisodeClock(**{**vars(CLOCK), **fields})


def test_presets_stay_inside_the_scenario_bounds() -> None:
    """The randomization ranges and push config can only produce scenarios from_dict accepts."""
    for seed in range(200):
        Scenario.from_dict(scenario(seed).to_dict(), clock=CLOCK)
    assert PushConfig().max_force_n == MAX_PUSH_FORCE_N


@pytest.mark.parametrize("field", ["init_pitch_rad", "init_pitch_rate_rads"])
def test_validate_scenario_rejects_a_non_finite_initial_state(field: str) -> None:
    bad = replace(scenario(1), **{field: math.inf})
    with pytest.raises(ValueError, match="scenario"):
        validate_scenario(bad)


# ------------------------------------------------------------------ portable draws (contract §9)


def _affine(low: float, high: float, u: float) -> float:
    """``low + (high - low)·u`` in plain IEEE double arithmetic: no fused multiply-add."""
    return low + (high - low) * u


@settings(max_examples=200, deadline=None)
@given(seed=st.integers(0, 2**53 - 1))
def test_scenario_draws_are_portable_affine_maps_of_raw_doubles(seed: int) -> None:
    """Every uniform draw is ``low + (high - low)·u`` with ``u`` the stream's next raw double,
    rounded step by step. numpy's ``Generator.uniform`` computes the same expression in C, where a
    compiler may contract it into one fused multiply-add, and its macOS arm64 and Linux x86_64
    builds disagree in the last bit; scenarios are trusted data compared bit for bit, so they must
    not (contract §9)."""
    preset = PRESETS["standard"]
    r = preset.randomization
    got = scenario(seed, "standard")

    init = stream_rng(seed, Stream.INIT).random()
    assert got.init_pitch_rad == _affine(*r.init_pitch_rad, init)

    u = iter(stream_rng(seed, Stream.MODEL).random(11).tolist())
    common = _affine(*r.motor_gain_common, next(u))
    assert got.model.chassis_mass_scale == _affine(*r.mass_scale, next(u))
    assert got.model.com_x_offset_m == _affine(*r.com_x_offset_m, next(u))
    assert got.model.wheel_friction_scale == _affine(*r.wheel_friction_scale, next(u))
    assert got.model.motor_gain == (
        common * _affine(*r.motor_gain_per_wheel, next(u)),
        common * _affine(*r.motor_gain_per_wheel, next(u)),
    )
    assert got.model.frictionloss_scale == _affine(*r.frictionloss_scale, next(u))
    assert got.sensors.imu_pitch_offset_rad == _affine(*r.imu_pitch_offset_rad, next(u))
    assert got.sensors.gyro_bias_rads == tuple(
        _affine(*r.gyro_bias_rads, next(u)) for _ in range(3)
    )


@settings(max_examples=200, deadline=None)
@given(seed=st.integers(0, 2**53 - 1))
def test_push_draws_are_portable_affine_maps_of_raw_doubles(seed: int) -> None:
    cfg = PRESETS["standard"].pushes
    assert cfg is not None
    pushes = scenario(seed, "standard").pushes
    rng = stream_rng(seed, Stream.PUSHES)
    rng.integers(len(cfg.count))
    n = MAX_PUSHES
    rng.random(n)  # start offsets: sorted and scaled, checked by the schedule tests
    impulses = [_affine(*cfg.impulse_ns, u) for u in rng.random(n).tolist()]
    rng.integers(0, 2, size=n)
    lim = cfg.lateral_fraction_max
    laterals = [_affine(-lim, lim, u) for u in rng.random(n).tolist()]
    for i, push in enumerate(pushes):
        assert push.magnitude_n == impulses[i] / (push.n_substeps * TIMESTEP)
        assert push.lateral_ratio == laterals[i]
