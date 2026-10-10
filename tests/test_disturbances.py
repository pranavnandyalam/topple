import itertools
import json
import math

import pytest

from topple.disturbances import (
    PRESETS,
    PushConfig,
    Scenario,
    make_scenario,
    push_force_world,
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
