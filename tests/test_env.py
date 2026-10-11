"""Adversarial tests for ``topple/Balance-v1`` (python/topple/envs/balance.py, contract §2-14)."""

from __future__ import annotations

import copy
import json
import math
import shutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import numpy.typing as npt
import pytest
from gymnasium.utils.env_checker import check_env
from pydantic import ValidationError

from topple.challenge import load_challenge_dir
from topple.contract import FEATURES, OBS_SCHEMA_ID, ActionError, BalancerObs
from topple.controllers.pid import PIDController
from topple.disturbances import (
    PRESETS,
    ModelOverrides,
    Push,
    Scenario,
    SensorOffsets,
    make_scenario,
    push_force_world,
)
from topple.envs import BALANCE_V1, BalanceEnv, RewardConfig
from topple.estimation import encoder_counts, odometry_yaw, wheel_angle_from_counts, wrap_angle
from topple.rng import MAX_SEED
from topple.robot import challenge_dir, robot_hash
from topple.traj import load_npz

Obs = npt.NDArray[np.float32]
Info = dict[str, Any]
Policy = Callable[[Obs, Info], Any]

ROBOT_XML = challenge_dir("balance-v1") / "robot" / "balancer.xml"
FALL_ANGLE = 0.70
INFO_KEYS = frozenset(
    {
        "episode_seed",
        "t",
        "step",
        "true_pitch",
        "tilt",
        "drift_m",
        "energy_j",
        "push_force",
        "reward_terms",
        "termination_reason",
    }
)
REWARD_TERMS = frozenset({"tilt", "rate", "drift", "yaw_rate", "effort", "smooth"})
# Everything _apply_model_overrides writes, plus constants mj_setConst derives from them.
MODEL_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "geom_friction",
    "actuator_gainprm",
    "actuator_biasprm",
    "dof_frictionloss",
    "actuator_delay",
    "body_subtreemass",
    "dof_invweight0",
)
_PID = PIDController()


# ---------------------------------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def _mujoco_log_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MuJoCo appends warnings to ./MUJOCO_LOG.TXT; keep that out of the repository."""
    monkeypatch.chdir(tmp_path)


def pid(obs: Obs, info: Info) -> tuple[float, float]:
    return _PID.act(BalancerObs.from_array(obs, t=info["t"]))


def idle(obs: Obs, info: Info) -> tuple[float, float]:
    return (0.0, 0.0)


def wobbly_pid(seed: int, std: float = 0.3) -> Policy:
    """PID plus independent per-wheel noise, so left and right commands always differ."""
    rng = np.random.default_rng(seed)

    def policy(obs: Obs, info: Info) -> tuple[float, float]:
        left, right = pid(obs, info)
        return (left + float(rng.normal(0.0, std)), right + float(rng.normal(0.0, std)))

    return policy


def spinning_pid(turn: float) -> Policy:
    """Balance with the PID but spin in place: its heading loop is blinded and replaced."""

    def policy(obs: Obs, info: Info) -> tuple[float, float]:
        blind = obs.copy()
        blind[2:4] = 0.0  # yaw, yaw_rate
        left, right = pid(blind, info)
        return (left - turn, right + turn)

    return policy


@dataclass(frozen=True)
class Rollout:
    """What one episode exposed. Index 0 comes from reset(); index k + 1 follows step k."""

    obs: list[Obs]
    infos: list[Info]
    rewards: list[float]
    actions: list[Any]
    qpos: list[npt.NDArray[np.float64]]
    qvel: list[npt.NDArray[np.float64]]
    terminated: bool
    truncated: bool

    @property
    def steps(self) -> int:
        return len(self.rewards)

    @property
    def reason(self) -> str | None:
        reason: str | None = self.infos[-1]["termination_reason"]
        return reason


def rollout(
    env: BalanceEnv,
    policy: Policy = pid,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
    max_steps: int | None = None,
) -> Rollout:
    obs, info = env.reset(seed=seed, options=options)
    observations, infos = [obs], [info]
    rewards: list[float] = []
    actions: list[Any] = []
    qpos, qvel = [env.data.qpos.copy()], [env.data.qvel.copy()]
    terminated = truncated = False
    while not (terminated or truncated) and (max_steps is None or len(rewards) < max_steps):
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        infos.append(info)
        rewards.append(reward)
        actions.append(action)
        qpos.append(env.data.qpos.copy())
        qvel.append(env.data.qvel.copy())
    return Rollout(observations, infos, rewards, actions, qpos, qvel, terminated, truncated)


def assert_bitwise_equal(actual: npt.ArrayLike, expected: npt.ArrayLike) -> None:
    a, b = np.asarray(actual), np.asarray(expected)
    np.testing.assert_array_equal(a, b)  # readable diff first
    assert a.dtype == b.dtype
    assert a.tobytes() == b.tobytes(), "equal values but different bits (signed zero?)"


def assert_same_episode(a: Rollout, b: Rollout) -> None:
    assert (a.steps, a.terminated, a.truncated, a.reason) == (
        b.steps,
        b.terminated,
        b.truncated,
        b.reason,
    )
    assert_bitwise_equal(np.array(a.obs), np.array(b.obs))
    assert a.rewards == b.rewards
    assert_bitwise_equal(np.array(a.qpos), np.array(b.qpos))
    assert_bitwise_equal(np.array(a.qvel), np.array(b.qvel))
    assert a.infos == b.infos


def scenario_for(seed: int, preset: str) -> Scenario:
    return make_scenario(
        seed, PRESETS[preset], timestep=0.002, control_dt=0.01, episode_seconds=10.0
    )


def nominal_scenario(**model: Any) -> Scenario:
    return Scenario(
        seed=0,
        preset="ideal",
        init_pitch_rad=0.0,
        pushes=(),
        model=ModelOverrides(**model),
        sensors=SensorOffsets(),
    )


def quat_pitch_yaw(quat: npt.NDArray[np.float64]) -> tuple[float, float]:
    """ZYX pitch and yaw of a wxyz quaternion (independent of the env's xmat formula)."""
    w, x, y, z = (float(v) for v in quat)
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return pitch, yaw


def robot_variant(tmp_path: Path, name: str, *edits: tuple[str, str]) -> Path:
    xml = ROBOT_XML.read_text()
    for old, new in edits:
        assert old in xml, old
        xml = xml.replace(old, new)
    path = tmp_path / f"{name}.xml"
    path.write_text(xml)
    return path


def wheel_axis(side: str, axis: str) -> str:
    return f'<joint name="{side}_wheel" type="hinge" axis="{axis}"/>'


# ------------------------------------------------------------------- spaces, registration, setup


@pytest.mark.parametrize("preset", ["ideal", "clean", "standard"])
def test_reset_returns_float32_observation_and_seeded_info(preset: str) -> None:
    env = BalanceEnv(preset)
    obs_space, act_space = env.observation_space, env.action_space
    assert isinstance(obs_space, gym.spaces.Box)
    assert obs_space.shape == (len(FEATURES),) == (10,)
    assert obs_space.dtype == np.float32
    assert np.isneginf(obs_space.low).all()
    assert np.isposinf(obs_space.high).all()
    assert isinstance(act_space, gym.spaces.Box)
    assert act_space.shape == (2,)
    assert act_space.dtype == np.float32
    assert act_space.low.tolist() == [-1.0, -1.0]
    assert act_space.high.tolist() == [1.0, 1.0]

    obs, info = env.reset(seed=42)
    assert obs.dtype == np.float32
    assert obs.shape == (10,)
    assert obs_space.contains(obs)
    assert info.keys() >= INFO_KEYS
    assert info["episode_seed"] == 42
    assert (info["step"], info["t"], info["energy_j"], info["drift_m"]) == (0, 0.0, 0.0, 0.0)
    assert info["reward_terms"] == {}
    assert info["termination_reason"] is None
    assert info["push_force"] == (0.0, 0.0, 0.0)
    first = BalancerObs.from_array(obs, t=0.0)
    assert first.yaw == 0.0
    assert first.wheel_pos == (0.0, 0.0)
    assert first.wheel_vel == (0.0, 0.0)
    assert first.last_action == (0.0, 0.0)
    assert abs(first.pitch) < math.radians(3.0 + 1.0) + 1e-6  # U(-3, 3) deg + mount offset


@pytest.mark.parametrize("preset", ["ideal", "clean", "standard"])
def test_gymnasium_env_checker_passes(preset: str) -> None:
    benign = ("minimum value is -infinity", "maximum value is infinity")  # unbounded by design
    env = BalanceEnv(preset)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env.unwrapped, skip_render_check=True)
    unexpected = [str(w.message) for w in caught if not any(b in str(w.message) for b in benign)]
    assert unexpected == []


def test_gym_make_builds_the_registered_env() -> None:
    assert BALANCE_V1 == "topple/Balance-v1"
    env = gym.make(f"topple.envs:{BALANCE_V1}", preset="clean")
    try:
        assert env.spec is not None
        assert env.spec.id == BALANCE_V1
        inner = env.unwrapped
        assert isinstance(inner, BalanceEnv)
        assert inner.preset.name == "clean"
        obs, info = env.reset(seed=0)
        assert info["episode_seed"] == 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the passive checker must stay silent
            for _ in range(20):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                assert isinstance(reward, float)
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
        assert env.observation_space.contains(obs)
        assert info["step"] == 20
    finally:
        env.close()
    default = gym.make(BALANCE_V1)
    assert default.unwrapped.preset.name == "standard"  # type: ignore[attr-defined]
    default.close()


def test_constructor_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="preset 'chaos'"):
        BalanceEnv("chaos")
    with pytest.raises(ValueError, match="render_mode"):
        BalanceEnv("ideal", render_mode="human")
    with pytest.raises(FileNotFoundError, match="nope-v1"):
        BalanceEnv("ideal", challenge="nope-v1")
    sideways = robot_variant(
        tmp_path, "sideways", (wheel_axis("right", "0 1 0"), wheel_axis("right", "1 0 0"))
    )
    with pytest.raises(ValueError, match=r"balancer\.robot\.v1.*axis"):
        BalanceEnv("ideal", robot_xml=sideways)


def test_timing_and_task_spec() -> None:
    env = BalanceEnv("ideal")
    assert env.model.opt.timestep == 0.002
    assert (env.control_dt, env.frame_skip, env.max_steps) == (0.01, 5, 1000)
    spec = env.task_spec
    assert spec.challenge == "balance-v1"
    assert spec.obs_schema == OBS_SCHEMA_ID
    assert spec.obs_features == FEATURES
    assert spec.robot.wheel_radius_m == pytest.approx(0.034)
    assert spec.robot.track_width_m == pytest.approx(0.14)
    assert env.metadata["render_fps"] * env.control_dt == pytest.approx(1.0)


# ----------------------------------------------------------------------------------- determinism


@pytest.mark.parametrize(
    ("preset", "seed"), [("clean", 0), ("clean", 5), ("standard", 3), ("standard", 4)]
)
def test_same_seed_and_actions_are_bit_identical_across_instances(preset: str, seed: int) -> None:
    first = rollout(BalanceEnv(preset), seed=seed)
    second = rollout(BalanceEnv(preset), seed=seed)
    assert_same_episode(first, second)
    if preset == "standard":  # the episodes really went through pushes
        assert sum(any(info["push_force"]) for info in first.infos) >= 10


def test_different_seeds_give_different_episodes() -> None:
    a = rollout(BalanceEnv("clean"), seed=0, max_steps=50)
    b = rollout(BalanceEnv("clean"), seed=1, max_steps=50)
    assert not np.array_equal(a.obs[0], b.obs[0])
    assert not np.array_equal(a.qpos[-1], b.qpos[-1])


def test_seedless_reset_draws_a_replayable_episode_seed() -> None:
    env = BalanceEnv("standard")
    first = rollout(env, max_steps=300)
    seed = first.infos[0]["episode_seed"]
    assert type(seed) is int
    assert 0 <= seed < MAX_SEED
    assert all(info["episode_seed"] == seed for info in first.infos)
    assert_same_episode(first, rollout(BalanceEnv("standard"), seed=seed, max_steps=300))
    assert env.reset()[1]["episode_seed"] != seed


def test_seeding_reset_makes_later_seedless_resets_reproducible() -> None:
    a, b = BalanceEnv("clean"), BalanceEnv("clean")
    a.reset(seed=123)
    b.reset(seed=123)
    assert a.reset()[1]["episode_seed"] == b.reset()[1]["episode_seed"]


def test_physics_does_not_depend_on_sensor_noise() -> None:
    """Verification replays commands open loop, so noise may change observations, never physics."""
    source = BalanceEnv("clean")
    source.reset(seed=5)
    other_noise = {**source.scenario.to_dict(), "seed": 6}
    commands = [(0.3 * math.sin(k / 7), 0.3 * math.cos(k / 5)) for k in range(80)]

    def open_loop(obs: Obs, info: Info) -> tuple[float, float]:
        return commands[info["step"]]

    a = rollout(BalanceEnv("clean"), open_loop, options={"scenario": source.scenario}, max_steps=80)
    b = rollout(BalanceEnv("clean"), open_loop, options={"scenario": other_noise}, max_steps=80)
    assert a.steps == b.steps
    assert_bitwise_equal(np.array(a.qpos), np.array(b.qpos))
    assert not np.array_equal(np.array(a.obs), np.array(b.obs))


def test_reset_mid_push_matches_a_fresh_env() -> None:
    env = BalanceEnv("standard", record=True)
    interrupted = rollout(env, seed=3, max_steps=212)  # scenario push starts at step 208
    assert any(interrupted.infos[-1]["push_force"])
    again = rollout(env, seed=4, max_steps=400)
    fresh_env = BalanceEnv("standard", record=True)
    assert_same_episode(again, rollout(fresh_env, seed=4, max_steps=400))
    assert_bitwise_equal(env.trajectory().qpos, fresh_env.trajectory().qpos)


def test_model_overrides_reset_to_nominal_before_each_episode() -> None:
    shared = BalanceEnv("clean")
    plan = [("clean", 3), ("clean", 1), ("clean", 3), ("ideal", 0), ("standard", 4), ("clean", 1)]
    for preset, seed in plan:
        shared.set_preset(preset)
        got = rollout(shared, seed=seed, max_steps=250)
        fresh = BalanceEnv(preset)
        assert_same_episode(got, rollout(fresh, seed=seed, max_steps=250))
        for field in MODEL_FIELDS:
            assert_bitwise_equal(getattr(shared.model, field), getattr(fresh.model, field))


def test_scenario_overrides_are_written_into_the_model() -> None:
    nominal, env = BalanceEnv("ideal"), BalanceEnv("clean")
    nominal.reset(seed=0)
    env.reset(seed=3)
    mo, m, m0, b = env.scenario.model, env.model, nominal.model, env.binding
    assert mo.chassis_mass_scale != 1.0  # seed 3 is genuinely randomized
    assert mo.actuator_delay_s == 0.010
    c = b.chassis_body
    assert m.body_mass[c] == m0.body_mass[c] * mo.chassis_mass_scale
    assert_bitwise_equal(m.body_inertia[c], m0.body_inertia[c] * mo.chassis_mass_scale)
    assert m.body_ipos[c, 0] == m0.body_ipos[c, 0] + mo.com_x_offset_m
    others = [i for i in range(m.nbody) if i != c]
    assert_bitwise_equal(m.body_mass[others], m0.body_mass[others])
    for g in b.wheel_geoms:
        assert m.geom_friction[g, 0] == m0.geom_friction[g, 0] * mo.wheel_friction_scale
    for i, motor in enumerate(b.motor_ids):
        assert m.actuator_gainprm[motor, 0] == m0.actuator_gainprm[motor, 0] * mo.motor_gain[i]
        assert m.actuator_delay[motor] == mo.actuator_delay_s
    assert_bitwise_equal(m.actuator_biasprm, m0.actuator_biasprm)  # back-EMF is never randomized
    for dof in b.wheel_dofadr:
        assert m.dof_frictionloss[dof] == m0.dof_frictionloss[dof] * mo.frictionloss_scale
    assert m.body_subtreemass[0] == pytest.approx(m.body_mass.sum())  # mj_setConst ran


@pytest.mark.parametrize(
    ("delay_s", "first_step_moves"), [(0.0, True), (0.008, True), (0.010, False)]
)
def test_actuator_delay_holds_commands_back(delay_s: float, first_step_moves: bool) -> None:
    def state_after(steps: int, action: tuple[float, float]) -> npt.NDArray[np.float64]:
        env = BalanceEnv("ideal")
        env.reset(options={"scenario": nominal_scenario(actuator_delay_s=delay_s)})
        for _ in range(steps):
            env.step(action)
        return np.concatenate([env.data.qpos, env.data.qvel])

    one_step_moves = not np.array_equal(state_after(1, (1.0, 1.0)), state_after(1, (0.0, 0.0)))
    assert one_step_moves == first_step_moves
    assert not np.array_equal(state_after(2, (1.0, 1.0)), state_after(2, (0.0, 0.0)))


# --------------------------------------------------------------------------------- reset options


def test_reset_accepts_scenario_object_and_json_dict() -> None:
    source = BalanceEnv("standard")
    want = rollout(source, seed=11, max_steps=300)
    as_json = json.loads(json.dumps(source.scenario.to_dict()))
    as_object = rollout(
        BalanceEnv("standard"), seed=999, options={"scenario": source.scenario}, max_steps=300
    )
    as_dict = rollout(BalanceEnv("standard"), options={"scenario": as_json}, max_steps=300)
    assert_same_episode(want, as_object)  # the scenario wins over the seed argument
    assert_same_episode(want, as_dict)
    assert as_object.infos[0]["episode_seed"] == 11


def test_scenario_preset_decides_the_sensor_model() -> None:
    want = rollout(BalanceEnv("ideal"), seed=2, max_steps=200)
    got = rollout(
        BalanceEnv("clean"), options={"scenario": scenario_for(2, "ideal")}, max_steps=200
    )
    assert_same_episode(want, got)


def test_initial_state_options_override_the_scenario() -> None:
    env = BalanceEnv("ideal")
    obs, info = env.reset(seed=0, options={"init_pitch_rad": 0.1, "init_pitch_rate": -0.5})
    seen = BalancerObs.from_array(obs, t=0.0)
    assert env.scenario.init_pitch_rad == 0.1
    assert env.scenario.init_pitch_rate_rads == -0.5
    assert info["true_pitch"] == pytest.approx(0.1, abs=1e-12)
    assert seen.pitch == float(np.float32(info["true_pitch"]))
    assert seen.pitch_rate == pytest.approx(-0.5, abs=1e-6)
    assert seen.wheel_vel == pytest.approx((0.5, 0.5), abs=1e-6)
    d, b = env.data, env.binding
    assert d.qvel[b.root_dofadr : b.root_dofadr + 3].tolist() == [0.0, 0.0, 0.0]
    for wheel in ("left_wheel_body", "right_wheel_body"):  # tires start without slipping
        vel = np.zeros(6)
        body = env.model.body(wheel).id
        mujoco.mj_objectVelocity(env.model, d, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
        np.testing.assert_allclose(vel, 0.0, atol=1e-12)

    base = scenario_for(9, "standard")
    env.reset(options={"scenario": base, "init_pitch_rad": -0.02, "pushes": []})
    assert env.scenario == replace(base, init_pitch_rad=-0.02, pushes=())


def test_reset_options_do_not_leak_into_the_next_episode() -> None:
    env = BalanceEnv("standard")
    env.reset(seed=3, options={"init_pitch_rad": 0.3, "init_pitch_rate": 1.0, "pushes": []})
    env.reset(seed=3)
    assert env.scenario == scenario_for(3, "standard")


def test_reset_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match=r"unknown reset options \['init_pitch'\]"):
        BalanceEnv("ideal").reset(seed=0, options={"init_pitch": 0.1})


@pytest.mark.parametrize(
    ("option", "field"),
    [("init_pitch_rad", "init_pitch_rad"), ("init_pitch_rate", "init_pitch_rate_rads")],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_reset_rejects_non_finite_initial_state(option: str, field: str, value: float) -> None:
    env = BalanceEnv("clean")
    with pytest.raises(ValueError, match="finite"):
        env.reset(seed=0, options={option: value})
    bad = {**scenario_for(0, "clean").to_dict(), field: value}
    with pytest.raises(ValueError, match="finite"):
        env.reset(options={"scenario": bad})


@pytest.mark.parametrize(
    "pushes",
    [
        [{"start_step": 5, "n_substeps": 50, "magnitude_n": 3.0, "forward_sign": 1.0}],
        [None],
        "push",
        [Push(5, 50, 3.0, 1.0, 0.0), (5, 50, 3.0, 1.0, 0.0)],
    ],
    ids=["dict", "none", "string", "tuple"],
)
def test_reset_rejects_pushes_that_are_not_push_objects(pushes: Any) -> None:
    with pytest.raises(ValueError, match="Push"):
        BalanceEnv("ideal").reset(seed=0, options={"pushes": pushes})


@pytest.mark.parametrize(
    "pushes",
    [
        [Push(5, 50, 3.0, 1.0, 0.0), Push(5, 50, 9.0, 1.0, 0.0)],
        [Push(40, 50, 3.0, 1.0, 0.0), Push(5, 50, 3.0, 1.0, 0.0)],
        [Push(100, 50, 3.0, 1.0, 0.0), Push(103, 50, 3.0, 1.0, 0.0)],
    ],
    ids=["same step", "unsorted", "overlapping"],
)
@pytest.mark.parametrize("via", ["pushes option", "scenario object", "scenario dict"])
def test_reset_rejects_push_schedules_that_cannot_run_as_written(
    pushes: list[Push], via: str
) -> None:
    """Two pushes at one step, or a push starting before the previous one has ended, would
    silently cut one short (the env runs one push at a time), and an unsorted list would not
    survive the scenario's JSON round trip: every reset path refuses all three."""
    env = BalanceEnv("ideal")
    scenario = replace(scenario_for(0, "ideal"), pushes=tuple(pushes))
    options: dict[str, Any] = {
        "pushes option": {"pushes": pushes},
        "scenario object": {"scenario": scenario},
        "scenario dict": {"scenario": scenario.to_dict()},
    }[via]
    with pytest.raises(ValueError, match="start_step"):
        env.reset(seed=0, options=options)


def test_back_to_back_pushes_both_run_in_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """15 substeps from step 100 end exactly where step 103 begins: both pushes deliver every
    substep they state, so the episode gets their full impulse, one after the other."""
    first = Push(100, 15, 3.0, forward_sign=1.0, lateral_ratio=0.0)
    second = Push(103, 50, 2.0, forward_sign=-1.0, lateral_ratio=0.0)
    env = BalanceEnv("ideal", record=True)
    applied: list[float] = []
    real_apply = mujoco.mj_applyFT

    def spy(m: Any, d: Any, force: Any, torque: Any, point: Any, body: int, target: Any) -> None:
        applied.append(float(np.linalg.norm(force)))
        real_apply(m, d, force, torque, point, body, target)

    monkeypatch.setattr(mujoco, "mj_applyFT", spy)
    pushed = rollout(env, seed=0, options={"pushes": [first, second]}, max_steps=130)
    assert pushed.steps == 130
    assert not pushed.terminated
    assert applied == pytest.approx([3.0] * 15 + [2.0] * 50)
    rows = env.trajectory().push_force
    active = np.flatnonzero(np.any(rows != 0.0, axis=1)).tolist()
    assert active == list(range(100, 113))
    assert np.hypot(*rows[100:103, :2].T).tolist() == pytest.approx([3.0] * 3)
    assert np.hypot(*rows[103:113, :2].T).tolist() == pytest.approx([2.0] * 10)


def test_seed_range_follows_the_contract() -> None:
    env = BalanceEnv("ideal")
    assert env.reset(seed=MAX_SEED - 1)[1]["episode_seed"] == MAX_SEED - 1
    with pytest.raises(ValueError, match=r"2\*\*53"):
        env.reset(seed=MAX_SEED)


@pytest.mark.parametrize(("n_substeps", "active_steps"), [(50, 10), (5, 1), (7, 2)])
def test_custom_push_is_reported_exactly_while_active(n_substeps: int, active_steps: int) -> None:
    start, magnitude = 20, 3.0
    push = Push(start, n_substeps, magnitude, forward_sign=1.0, lateral_ratio=0.0)
    env = BalanceEnv("ideal", record=True)
    pushed = rollout(env, seed=0, options={"pushes": [push]}, max_steps=60)
    window = list(range(start + 1, start + 1 + active_steps))  # info index k + 1 follows step k
    assert [k for k, info in enumerate(pushed.infos) if any(info["push_force"])] == window
    for k in window:
        fx, fy, fz = pushed.infos[k]["push_force"]
        assert math.hypot(fx, fy) == pytest.approx(magnitude)
        assert fz == 0.0
    rows = np.flatnonzero(np.any(env.trajectory().push_force != 0.0, axis=1))
    assert rows.tolist() == [k - 1 for k in window]

    calm = rollout(BalanceEnv("ideal"), seed=0, options={"pushes": []}, max_steps=60)
    assert_bitwise_equal(np.array(pushed.qpos[: start + 1]), np.array(calm.qpos[: start + 1]))
    assert not np.array_equal(pushed.qpos[start + 1], calm.qpos[start + 1])


@pytest.mark.parametrize("forward_sign", [1.0, -1.0])
def test_push_direction_is_frozen_at_the_onset_heading(forward_sign: float) -> None:
    onset, active = 300, 10
    push = Push(onset, active * 5, 4.0, forward_sign=forward_sign, lateral_ratio=0.0)
    policy = spinning_pid(0.3)
    env, twin = BalanceEnv("ideal"), BalanceEnv("ideal")
    obs, info = env.reset(seed=0, options={"init_pitch_rad": 0.0, "pushes": [push]})
    twin.reset(seed=0, options={"init_pitch_rad": 0.0, "pushes": []})
    for _ in range(onset):
        action = policy(obs, info)
        obs, _, _, _, info = env.step(action)
        twin.step(action)
    heading = quat_pitch_yaw(env.data.qpos[3:7])[1]
    assert abs(heading) > 1.0  # far from the world x axis, so a world-frame bug would show
    action = policy(obs, info)
    obs, _, _, _, info = env.step(action)
    twin.step(action)
    fx, fy, fz = info["push_force"]
    assert (math.hypot(fx, fy), fz) == (pytest.approx(push.magnitude_n), 0.0)
    along = heading if forward_sign > 0 else heading + math.pi
    # The heading at onset comes from the current qpos, not from MuJoCo's one-substep-old xmat.
    assert abs(wrap_angle(math.atan2(fy, fx) - along)) < 1e-9
    kick = env.data.qvel[0:2] - twin.data.qvel[0:2]  # what the push alone did to the axle
    assert abs(wrap_angle(math.atan2(kick[1], kick[0]) - math.atan2(fy, fx))) < 0.01

    forces = [info["push_force"]]
    for _ in range(active + 1):
        obs, _, terminated, _, info = env.step(policy(obs, info))
        assert not terminated
        forces.append(info["push_force"])
    assert forces[:active] == [forces[0]] * active  # fixed in the world while the robot turns
    assert forces[active:] == [(0.0, 0.0, 0.0)] * 2
    assert abs(wrap_angle(quat_pitch_yaw(env.data.qpos[3:7])[1] - heading)) > 0.1


class StepSpy:
    """Wraps ``mujoco.mj_step`` to see what the env hands MuJoCo on every substep."""

    def __init__(self, env: BalanceEnv, monkeypatch: pytest.MonkeyPatch) -> None:
        self.ctrl: list[npt.NDArray[np.float64]] = []
        self.qfrc: list[npt.NDArray[np.float64]] = []
        self.lever: list[npt.NDArray[np.float64]] = []  # push point relative to the root
        self.rotation: list[npt.NDArray[np.float64]] = []
        real_step, b = mujoco.mj_step, env.binding

        def spy(m: mujoco.MjModel, d: mujoco.MjData) -> None:
            self.ctrl.append(d.ctrl.copy())
            self.qfrc.append(d.qfrc_applied.copy())
            self.lever.append(d.site_xpos[b.push_site] - d.xpos[b.chassis_body])
            self.rotation.append(d.xmat[b.chassis_body].reshape(3, 3).copy())
            real_step(m, d)

        monkeypatch.setattr(mujoco, "mj_step", spy)


def test_push_force_acts_at_the_push_point_for_exactly_its_substeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = BalanceEnv("ideal")
    push = Push(3, n_substeps=7, magnitude_n=5.0, forward_sign=-1.0, lateral_ratio=0.2)
    env.reset(seed=0, options={"pushes": [push]})
    spy = StepSpy(env, monkeypatch)
    infos = [env.step([0.1, -0.1])[4] for _ in range(6)]
    assert len(spy.qfrc) == 6 * env.frame_skip
    window = range(3 * env.frame_skip, 3 * env.frame_skip + 7)
    assert [k for k, f in enumerate(spy.qfrc) if np.any(f)] == list(window)
    force = np.array(infos[3]["push_force"])
    for k in window:  # free joint: world force on dofs 0-2, body-frame torque on 3-5
        np.testing.assert_allclose(spy.qfrc[k][0:3], force, rtol=1e-12)
        torque = spy.rotation[k].T @ np.cross(spy.lever[k], force)
        np.testing.assert_allclose(spy.qfrc[k][3:6], torque, rtol=1e-9, atol=1e-12)
        assert spy.qfrc[k][list(env.binding.wheel_dofadr)].tolist() == [0.0, 0.0]
    assert not np.any(env.data.qfrc_applied)  # nothing lingers between steps


def test_commands_are_clipped_signed_and_held_for_every_substep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mirrored = robot_variant(
        tmp_path, "m", (wheel_axis("left", "0 1 0"), wheel_axis("left", "0 -1 0"))
    )
    env = BalanceEnv("ideal", robot_xml=mirrored)
    env.reset(seed=0)
    spy = StepSpy(env, monkeypatch)
    commands = [(2.0, -0.5), (0.25, -3.0), (0.0, 0.75)]
    for command in commands:
        env.step(command)
    clipped = np.clip(commands, -1.0, 1.0)
    expected = [[-left, right] for left, right in clipped.tolist()]  # left axis is mirrored
    motors = list(env.binding.motor_ids)
    held = [ctrl[motors].tolist() for ctrl in spy.ctrl]
    assert held == [c for c in expected for _ in range(env.frame_skip)]


def test_scheduled_pushes_hit_at_their_steps_along_the_heading() -> None:
    env = BalanceEnv("standard")
    ep = rollout(env, seed=9)  # PID survives all three pushes of this seed
    pushes = env.scenario.pushes
    assert len(pushes) == 3
    per_push = pushes[0].n_substeps // env.frame_skip
    window = [p.start_step + 1 + j for p in pushes for j in range(per_push)]
    assert [k for k, info in enumerate(ep.infos) if any(info["push_force"])] == window
    for p in pushes:
        heading = quat_pitch_yaw(ep.qpos[p.start_step][3:7])[1]
        reported = ep.infos[p.start_step + 1]["push_force"]
        assert reported == pytest.approx(push_force_world(p, heading), abs=0.01)  # newtons


# ------------------------------------------------------------------------------------- presets


def test_set_preset_waits_for_the_next_reset() -> None:
    env, twin = BalanceEnv("clean"), BalanceEnv("clean")
    obs, info = env.reset(seed=3)
    twin.reset(seed=3)
    env.set_preset("standard")
    assert env.preset.name == "clean"
    for _ in range(300):  # standard pushes for seed 3 start at step 208: none may leak in
        action = pid(obs, info)
        obs, reward, _, _, info = env.step(action)
        twin_obs, twin_reward, _, _, _ = twin.step(action)
        assert_bitwise_equal(obs, twin_obs)
        assert reward == twin_reward
        assert not any(info["push_force"])
    env.reset(seed=3)
    assert env.preset.name == "standard"
    assert env.scenario == scenario_for(3, "standard")
    env.reset(seed=3)
    assert env.preset.name == "standard"


def test_set_preset_rejects_unknown_names_and_keeps_the_pending_one() -> None:
    env = BalanceEnv("clean")
    env.set_preset("ideal")
    with pytest.raises(ValueError, match="'chaos' is not part of balance-v1"):
        env.set_preset("chaos")
    obs, info = env.reset(seed=1)
    assert env.preset.name == "ideal"
    assert obs[0] == np.float32(info["true_pitch"])  # ideal sensors took over


# --------------------------------------------------------------------------------------- sensors


def test_ideal_preset_observes_noise_free_ground_truth() -> None:
    """No noise, bias, mount offset, estimator or quantization: the exact current state.

    MuJoCo's derived data (``xmat``, sensors) lag one substep behind ``qpos`` after ``mj_step``,
    so the ideal observation is computed from ``qpos``/``qvel`` (contract §5, §11).
    """
    env = BalanceEnv("ideal")
    b, d = env.binding, env.data
    obs, info = env.reset(seed=4, options={"init_pitch_rate": 0.2})
    q0 = np.array([d.qpos[a] for a in b.wheel_qposadr])
    sign = np.array(b.wheel_sign)
    da = b.root_dofadr
    policy = wobbly_pid(4)
    for _ in range(300):
        action = policy(obs, info)
        obs, _, terminated, truncated, info = env.step(action)
        pitch, yaw = quat_pitch_yaw(d.qpos[b.root_qposadr + 3 : b.root_qposadr + 7])
        truth = [
            pitch,
            d.qvel[da + 4],  # free-joint angular velocity is in the body frame
            yaw,
            d.qvel[da + 5],
            *(sign * (np.array([d.qpos[a] for a in b.wheel_qposadr]) - q0)),
            *(sign * np.array([d.qvel[v] for v in b.wheel_dofadr])),
            *np.clip(action, -1.0, 1.0),
        ]
        np.testing.assert_allclose(obs, np.array(truth, dtype=np.float32), rtol=0, atol=1e-6)
        assert obs[1] == np.float32(d.qvel[da + 4])
        assert obs[3] == np.float32(d.qvel[da + 5])
        assert obs[0] == np.float32(info["true_pitch"])
        assert info["true_pitch"] == pytest.approx(pitch, abs=1e-12)
        assert not terminated
        assert not truncated


@pytest.mark.parametrize("preset", ["ideal", "clean", "standard"])
def test_info_and_reward_use_the_exact_current_state(preset: str) -> None:
    env = BalanceEnv(preset)
    b, d = env.binding, env.data
    obs, info = env.reset(seed=2)
    policy = wobbly_pid(2)
    for _ in range(200):
        obs, _, terminated, _, info = env.step(policy(obs, info))
        quat = d.qpos[b.root_qposadr + 3 : b.root_qposadr + 7]
        x, y = float(quat[1]), float(quat[2])
        assert info["tilt"] == pytest.approx(math.acos(1.0 - 2.0 * (x * x + y * y)), abs=1e-12)
        assert info["true_pitch"] == pytest.approx(quat_pitch_yaw(quat)[0], abs=1e-12)
        rate = float(d.qvel[b.root_dofadr + 4])
        assert info["reward_terms"]["rate"] == min((rate / 3.0) ** 2, 1.0)
        assert info["reward_terms"]["tilt"] == min((info["true_pitch"] / 0.2) ** 2, 1.0)
        if terminated:
            break


@pytest.mark.parametrize("preset", ["ideal", "clean"])
def test_fall_is_detected_on_the_substep_that_crosses_the_fall_angle(preset: str) -> None:
    """A robot a hair below the fall angle and tipping over ends after exactly one substep."""
    env = BalanceEnv(preset)
    env.reset(seed=0, options={"init_pitch_rad": FALL_ANGLE - 1e-4, "init_pitch_rate": 1.0})
    twin = copy.deepcopy(env.data)
    twin.ctrl[:] = 0.0
    mujoco.mj_step(env.model, twin)  # one substep of the same physics
    _, _, terminated, _, info = env.step([0.0, 0.0])
    assert terminated
    assert info["termination_reason"] == "fell"
    assert_bitwise_equal(env.data.qpos, twin.qpos)
    assert info["tilt"] > FALL_ANGLE


def test_energy_starts_from_the_reset_relative_wheel_angles(tmp_path: Path) -> None:
    """A wheel hinge with a ``ref`` starts at qpos = ref, which must not count as motion."""
    left = '<joint name="left_wheel" type="hinge" axis="0 1 0"/>'
    path = robot_variant(tmp_path, "ref", (left, left.replace("/>", ' ref="0.5"/>')))
    plain, offset = BalanceEnv("ideal"), BalanceEnv("ideal", robot_xml=path)
    for env in (plain, offset):
        env.reset(seed=0, options={"init_pitch_rad": 0.0})
    assert offset.data.qpos[offset.binding.wheel_qposadr[0]] == 0.5
    for _ in range(3):
        *_, info_plain = plain.step([0.5, 0.5])
        *_, info_offset = offset.step([0.5, 0.5])
        assert info_offset["energy_j"] == pytest.approx(info_plain["energy_j"], rel=1e-9)
    assert info_plain["energy_j"] > 0.0


def test_ideal_preset_records_noiseless_imu_rows_every_step() -> None:
    env = BalanceEnv("ideal", record=True)
    b, d = env.binding, env.data
    obs, info = env.reset(seed=1, options={"init_pitch_rad": 0.02})
    gyro, accel = [], []
    policy = wobbly_pid(1)
    for _ in range(50):
        obs, _, _, _, info = env.step(policy(obs, info))
        gyro.append(d.sensordata[b.gyro_adr : b.gyro_adr + 3].copy())
        accel.append(d.sensordata[b.accel_adr : b.accel_adr + 3].copy())
    traj = env.trajectory()
    assert_bitwise_equal(traj.gyro, np.array(gyro, dtype=np.float32))
    assert_bitwise_equal(traj.accel, np.array(accel, dtype=np.float32))
    assert len({tuple(row) for row in traj.gyro.tolist()}) == 50  # not a constant row


def test_encoders_quantize_and_differentiate_wheel_angles() -> None:
    env = BalanceEnv("clean")
    b, cpr = env.binding, env.params.sensing.encoder_cpr
    ep = rollout(env, wobbly_pid(7), seed=7, max_steps=300)
    obs = np.array(ep.obs)
    counts = np.rint(obs[:, 4:6].astype(np.float64) * cpr / (2.0 * math.pi)).astype(int)
    assert np.abs(counts).max() > 50  # the wheels really turned
    pos = np.array([[wheel_angle_from_counts(int(c), 0, cpr) for c in row] for row in counts])
    assert_bitwise_equal(obs[:, 4:6], pos.astype(np.float32))  # exact multiples of 2*pi/cpr

    def raw_counts(q: npt.NDArray[np.float64]) -> list[int]:
        return [
            encoder_counts(q[a], s, cpr) for a, s in zip(b.wheel_qposadr, b.wheel_sign, strict=True)
        ]

    origin = np.array(raw_counts(ep.qpos[0]))
    assert_bitwise_equal(counts, np.array([raw_counts(q) for q in ep.qpos]) - origin)

    vel = np.vstack([np.zeros((1, 2)), np.diff(pos, axis=0) / env.control_dt])
    assert_bitwise_equal(obs[:, 6:8], vel.astype(np.float32))
    yaw = [odometry_yaw(left, right, b.wheel_radius_m, b.track_width_m) for left, right in pos]
    assert_bitwise_equal(obs[:, 2], np.array(yaw, dtype=np.float32))
    assert np.abs(obs[:, 2]).max() > 0.01


@pytest.mark.parametrize("seed", [1, 2])
def test_full_sensor_pitch_estimate_tracks_the_truth(seed: int) -> None:
    env = BalanceEnv("clean")
    ep = rollout(env, seed=seed)
    assert ep.truncated
    sc = env.scenario
    assert ep.obs[0][0] == np.float32(sc.init_pitch_rad + sc.sensors.imu_pitch_offset_rad)
    error = np.array([o[0] - i["true_pitch"] for o, i in zip(ep.obs, ep.infos, strict=True)])
    assert np.abs(error).max() < math.radians(3.0)
    assert np.abs(error).mean() > 1e-4  # an estimate, not ground truth


@pytest.mark.parametrize(("preset", "saturates"), [("clean", True), ("ideal", False)])
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_gyro_saturates_at_full_scale(preset: str, saturates: bool, direction: float) -> None:
    env = BalanceEnv(preset)
    limit = env.params.sensing.gyro_range_rads
    rate = 2.0 * limit * direction
    obs, _ = env.reset(seed=0, options={"init_pitch_rate": rate})
    assert obs[1] == np.float32(math.copysign(limit, rate) if saturates else rate)


# ----------------------------------------------------------------------------------- termination


@pytest.mark.parametrize("preset", ["ideal", "clean"])
@pytest.mark.parametrize("lean", [0.05, -0.05])
def test_idle_robot_falls_within_one_second(preset: str, lean: float) -> None:
    env = BalanceEnv(preset)
    ep = rollout(env, idle, seed=1, options={"init_pitch_rad": lean})
    assert ep.terminated
    assert not ep.truncated
    assert ep.reason == "fell"
    assert ep.steps <= 100
    assert ep.infos[-1]["tilt"] > FALL_ANGLE >= ep.infos[-2]["tilt"]  # first crossing
    if preset == "ideal":  # no COM offset: it falls the way it leans
        assert math.copysign(1.0, ep.infos[-1]["true_pitch"]) == math.copysign(1.0, lean)
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0.0, 0.0])


@pytest.mark.parametrize("pitch", [0.8, -0.8, 1.5])
def test_initial_pitch_beyond_the_fall_angle_ends_the_first_step(pitch: float) -> None:
    ep = rollout(BalanceEnv("clean"), pid, seed=0, options={"init_pitch_rad": pitch})
    assert ep.steps == 1
    assert ep.terminated
    assert ep.reason == "fell"


def test_non_wheel_floor_contact_ends_the_episode(tmp_path: Path) -> None:
    tail = '<geom name="tail" type="sphere" size="0.01" pos="-0.06 0 -0.03" mass="0.001"/>'
    path = robot_variant(tmp_path, "tail", ('<site name="imu"', f'{tail}\n<site name="imu"'))
    ep = rollout(BalanceEnv("ideal", robot_xml=path), idle, options={"init_pitch_rad": 0.0})
    assert ep.steps == 1
    assert ep.reason == "floor_contact"
    assert ep.infos[-1]["tilt"] < 0.01


@pytest.mark.parametrize("magnitude", [1e4, 1e9, 1e300])
@pytest.mark.parametrize("preset", ["ideal", "clean"])
def test_absurd_push_ends_the_episode_without_nan(preset: str, magnitude: float) -> None:
    env = BalanceEnv(preset, record=True)
    push = Push(5, 50, magnitude, forward_sign=1.0, lateral_ratio=0.0)
    ep = rollout(env, seed=3, options={"pushes": [push]})
    assert ep.terminated
    assert not ep.truncated
    assert ep.reason in {"sim_unstable", "fell"}
    assert ep.steps <= push.start_step + 10  # it ends while the push is still on
    assert np.isfinite(np.array(ep.obs)).all()
    assert all(env.observation_space.contains(o) for o in ep.obs)
    assert np.isfinite(ep.rewards).all()
    for info in ep.infos:
        for key in ("true_pitch", "tilt", "drift_m", "energy_j"):
            assert math.isfinite(info[key]), (key, info)
    if ep.reason == "sim_unstable":  # rolled back to the last stable substep, MjData included
        for name in ("qacc", "qacc_warmstart", "sensordata", "actuator_force"):
            assert np.isfinite(getattr(env.data, name)).all(), name
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0.0, 0.0])
    env.simulate_epilogue(1.0)
    traj = env.trajectory()
    for name in ("qpos", "qvel", "obs", "ctrl", "gyro", "accel", "epilogue_qpos"):
        assert np.isfinite(getattr(traj, name)).all(), name
    if ep.reason == "sim_unstable":
        assert traj.epilogue_qpos.shape[0] == 100


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pid_on_ideal_is_truncated_exactly_at_max_steps(seed: int) -> None:
    env = BalanceEnv("ideal")
    ep = rollout(env, seed=seed)
    assert ep.steps == env.max_steps == 1000
    assert ep.truncated
    assert not ep.terminated
    assert ep.reason is None
    assert [info["step"] for info in ep.infos] == list(range(1001))
    assert all(info["t"] == info["step"] * env.control_dt for info in ep.infos)  # not data.time
    assert ep.infos[-1]["t"] == pytest.approx(10.0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0.0, 0.0])


def test_max_drift_option_terminates_with_drift() -> None:
    env = BalanceEnv("ideal", max_drift_m=0.005, record=True)
    ep = rollout(env, seed=1)
    assert ep.terminated
    assert ep.reason == "drift"
    assert ep.infos[-1]["drift_m"] > 0.005 >= ep.infos[-2]["drift_m"]
    assert env.trajectory().meta["termination"] == {"step": ep.steps, "reason": "drift"}


def test_step_before_any_reset_raises() -> None:
    with pytest.raises(RuntimeError, match="reset"):
        BalanceEnv("ideal").step([0.0, 0.0])


@pytest.mark.parametrize(
    "bad",
    [
        [math.nan, 0.0],
        [0.0, math.inf],
        np.array([0.0, -np.inf]),
        [0.5],
        [0.1, 0.2, 0.3],
        np.zeros((1, 2)),
        np.float64(0.5),
        0.5,
        "ab",
        None,
        {"left": 0.0, "right": 0.0},
        [True, False],
        [np.bool_(True), 0.0],
        [1j, 0.0],
        [None, 0.0],
        ["0.1", 0.0],
    ],
    ids=[
        "nan",
        "inf",
        "ndarray-inf",
        "too-short",
        "too-long",
        "2d",
        "numpy-scalar",
        "float",
        "string",
        "none",
        "mapping",
        "bools",
        "numpy-bool",
        "complex",
        "none-item",
        "string-item",
    ],
)
def test_invalid_action_raises_and_has_no_side_effects(bad: Any) -> None:
    env, twin = BalanceEnv("standard"), BalanceEnv("standard")
    env.reset(seed=3)
    twin.reset(seed=3)
    env.step([0.1, 0.1])
    twin.step([0.1, 0.1])
    with pytest.raises(ActionError):
        env.step(bad)
    got, want = env.step([0.2, -0.1]), twin.step([0.2, -0.1])
    assert got[4]["step"] == 2
    assert_bitwise_equal(got[0], want[0])
    assert got[1:4] == want[1:4]
    assert_bitwise_equal(env.data.qpos, twin.data.qpos)


def test_action_containers_are_interchangeable() -> None:
    finals = []
    for wrap in (list, tuple, np.array, lambda a: np.array(a, dtype=np.float32)):
        env = BalanceEnv("clean")
        env.reset(seed=2)
        for k in range(20):
            obs, *_ = env.step(wrap([0.5, -0.25] if k % 2 else [1, 0]))
        finals.append(obs)
    for obs in finals[1:]:
        assert_bitwise_equal(obs, finals[0])


# ---------------------------------------------------------------------------------------- reward


@pytest.mark.parametrize("preset", ["ideal", "standard"])
def test_reward_is_bounded_for_random_actions(preset: str) -> None:
    env = BalanceEnv(preset)
    weights = env.reward_config.model_dump()
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    for _ in range(1500):
        _, reward, terminated, truncated, info = env.step(rng.uniform(-3.0, 3.0, size=2))
        assert 0.0 <= reward <= 1.0
        terms = info["reward_terms"]
        assert set(terms) == REWARD_TERMS
        assert all(0.0 <= c <= 1.0 for c in terms.values()), terms
        assert reward == pytest.approx(1.0 - sum(weights[k] * c for k, c in terms.items()))
        if terminated or truncated:
            env.reset()


def test_reward_with_custom_weights() -> None:
    env = BalanceEnv(
        "ideal", reward=RewardConfig(tilt=1.0, rate=0, drift=0, yaw_rate=0, effort=0, smooth=0)
    )
    ep = rollout(env, idle, seed=0, options={"init_pitch_rad": 0.05})
    for reward, info in zip(ep.rewards, ep.infos[1:], strict=True):
        assert reward == 1.0 - min((info["true_pitch"] / 0.2) ** 2, 1.0)
    assert ep.rewards[0] > 0.9
    assert ep.rewards[-1] == 0.0  # fallen: tilt cost saturates


def test_reward_config_validation() -> None:
    with pytest.raises(ValidationError, match="sum to at most 1"):
        RewardConfig(tilt=0.9, rate=0.2)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        RewardConfig(effort=-0.1)
    with pytest.raises(ValidationError, match="Extra inputs"):
        RewardConfig.model_validate({"bonus": 0.1})
    config = RewardConfig()
    with pytest.raises(ValidationError, match="frozen"):
        config.tilt = 0.0  # type: ignore[misc]


def test_clipped_command_is_what_the_robot_and_the_next_observation_get() -> None:
    env = BalanceEnv("ideal", record=True)
    env.reset(seed=0)
    obs, _, _, _, info = env.step([5.0, -7.5])
    assert BalancerObs.from_array(obs, t=info["t"]).last_action == (1.0, -1.0)
    assert info["reward_terms"]["effort"] == 1.0
    assert info["reward_terms"]["smooth"] == 1.0
    traj = env.trajectory()
    assert traj.action.tolist() == [[5.0, -7.5]]
    assert traj.ctrl.tolist() == [[1.0, -1.0]]


# ------------------------------------------------------------------------------------- recording


def test_trajectory_records_exactly_what_the_controller_saw() -> None:
    env = BalanceEnv("standard", record=True)
    seen: list[Obs] = []

    def policy(obs: Obs, info: Info) -> tuple[float, float]:
        seen.append(obs.copy())
        left, right = pid(obs, info)
        if info["step"] % 50 == 25:
            left, right = left + 1.5, right + 1.5  # out of range: exercises clipping
        obs *= 0.0  # an agent normalizing in place must not corrupt the record
        return (left, right)

    ep = rollout(env, policy, seed=3, max_steps=260)
    traj = env.trajectory({"episode_index": 7})
    steps, nq, nv = ep.steps, env.model.nq, env.model.nv
    assert traj.steps == steps == 260
    assert (traj.qpos.shape, traj.qvel.shape) == ((steps + 1, nq), (steps + 1, nv))
    assert traj.obs.shape == (steps, 10)
    for name, width in (("action", 2), ("ctrl", 2), ("enc_counts", 2), ("gyro", 3)):
        assert getattr(traj, name).shape == (steps, width)
    assert (traj.accel.shape, traj.push_force.shape) == ((steps, 3), (steps, 3))
    assert traj.epilogue_qpos.shape == (0, nq)

    assert_bitwise_equal(traj.obs, np.array(seen))  # row k is what action k was computed from
    assert traj.action.tolist() == [list(a) for a in ep.actions]
    assert np.abs(traj.action).max() > 1.0
    assert_bitwise_equal(traj.ctrl, np.clip(traj.action, -1.0, 1.0))
    assert_bitwise_equal(traj.qpos, np.array(ep.qpos))
    assert_bitwise_equal(traj.qvel, np.array(ep.qvel))
    assert traj.push_force.tolist() == [list(i["push_force"]) for i in ep.infos[1:]]
    assert np.any(traj.push_force)

    assert_bitwise_equal(traj.obs[1:, 1], traj.gyro[:-1, 1])  # pitch_rate = last gyro sample
    assert_bitwise_equal(traj.obs[1:, 3], traj.gyro[:-1, 2])
    gravity = -float(env.model.opt.gravity[2])
    assert traj.accel[:, 2].mean() == pytest.approx(gravity, abs=0.3)  # specific force, not accel
    b, cpr = env.binding, env.params.sensing.encoder_cpr
    origin = [
        encoder_counts(traj.qpos[0, a], s, cpr)
        for a, s in zip(b.wheel_qposadr, b.wheel_sign, strict=True)
    ]
    from_counts = [
        [wheel_angle_from_counts(int(c), o, cpr) for c, o in zip(row, origin, strict=True)]
        for row in traj.enc_counts
    ]
    assert_bitwise_equal(traj.obs[1:, 4:6], np.array(from_counts, dtype=np.float32)[:-1])

    meta = traj.meta
    assert meta["episode_index"] == 7
    assert (meta["challenge"], meta["preset"]) == ("balance-v1", "standard")
    assert meta["scenario"] == env.scenario.to_dict()
    assert meta["termination"]["step"] == steps
    assert (meta["control_dt"], meta["timestep"], meta["frame_skip"]) == (0.01, 0.002, 5)
    assert meta["max_steps"] == 1000
    assert meta["obs_schema"] == OBS_SCHEMA_ID
    assert meta["obs_features"] == list(FEATURES)
    assert meta["wheel_sign"] == [1.0, 1.0]
    assert meta["joint_qpos"] == {"root": 0, "wheels": [7, 8]}
    assert meta["robot_hash"] == robot_hash(env.robot_xml, env.params)
    assert meta["versions"]["mujoco"] == mujoco.__version__


def test_a_robot_variant_records_its_own_hash(tmp_path: Path) -> None:
    """``meta.robot_hash`` is the hash of the robot actually simulated: videos refuse to draw a
    variant's states on the stock robot (fix spec I) and verification binds uploads by it."""
    from topple.video import render_trajectory

    left = wheel_axis("left", "0 1 0")
    path = robot_variant(tmp_path, "ref", (left, left.replace("/>", ' ref="0.5"/>')))
    env = BalanceEnv("ideal", robot_xml=path, record=True)
    rollout(env, lambda o, i: (0.0, 0.0), seed=0, max_steps=10)
    traj = env.trajectory({"episode_index": 0})
    stock = env.challenge
    assert traj.meta["robot_hash"] == robot_hash(path, env.params)
    assert traj.meta["robot_hash"] != robot_hash(stock.robot_xml, stock.params)
    with pytest.raises(ValueError, match=r"robot_hash .* refusing to draw the wrong robot"):
        render_trajectory(traj, tmp_path / "variant.gif")
    assert not (tmp_path / "variant.gif").exists()


def test_epilogue_keeps_simulating_with_zero_command() -> None:
    env = BalanceEnv("ideal", record=True)
    ep = rollout(env, lambda o, i: (0.4, 0.4), seed=0, options={"init_pitch_rad": 0.05})
    assert ep.terminated
    terminal = env.data.qpos.copy()
    env.simulate_epilogue(1.0)
    traj = env.trajectory()
    assert traj.epilogue_qpos.shape == (100, env.model.nq)
    assert np.isfinite(traj.epilogue_qpos).all()
    assert not np.array_equal(traj.epilogue_qpos[0], terminal)
    assert_bitwise_equal(traj.qpos[-1], terminal)  # the epilogue never rewrites the episode
    assert env.data.ctrl.tolist() == [0.0, 0.0]
    assert traj.meta["termination"] == {"step": ep.steps, "reason": "fell"}
    env.simulate_epilogue(0.25)
    assert env.trajectory().epilogue_qpos.shape == (25, env.model.nq)


def test_trajectory_round_trips_through_npz(tmp_path: Path) -> None:
    env = BalanceEnv("standard", record=True)
    rollout(env, seed=9)  # PID survives this seed: a full-length trajectory
    env.simulate_epilogue(1.0)
    traj = env.trajectory({"episode_index": 0})
    assert traj.meta["termination"] == {"step": 1000, "reason": "timeout"}
    path = tmp_path / "traj.npz"
    traj.save_npz(path)
    back = load_npz(path, max_steps=env.max_steps)
    for name in ("qpos", "qvel", "obs", "action", "ctrl", "enc_counts", "gyro", "accel"):
        assert_bitwise_equal(getattr(back, name), getattr(traj, name))
    assert_bitwise_equal(back.push_force, traj.push_force)
    assert_bitwise_equal(back.epilogue_qpos, traj.epilogue_qpos)
    assert back.meta == traj.meta
    assert Scenario.from_dict(back.meta["scenario"]) == env.scenario


def test_trajectory_replays_open_loop_from_scenario_and_commands() -> None:
    env = BalanceEnv("standard", record=True)
    rollout(env, seed=4)  # PID falls after the pushes of this seed
    traj = env.trajectory()
    replay = BalanceEnv("standard", record=True)
    replay.reset(options={"scenario": traj.meta["scenario"]})
    for command in traj.ctrl:
        replay.step(command)
    again = replay.trajectory()
    assert traj.meta["termination"]["reason"] == "fell"
    assert again.meta["termination"] == traj.meta["termination"]
    for name in ("qpos", "qvel", "obs", "enc_counts", "gyro", "accel", "push_force"):
        assert_bitwise_equal(getattr(again, name), getattr(traj, name))


def test_trajectory_meta_never_reaches_the_live_scenario() -> None:
    """Editing a trajectory's meta.scenario (a viewer, a fuzzer building its next candidate)
    must not change the pushes the env is still running."""
    env = BalanceEnv("standard", record=True)
    rollout(env, seed=9, max_steps=50)
    pushes = env.scenario.pushes
    expected = [vars(p).copy() for p in pushes]
    meta = env.trajectory().meta["scenario"]
    for push in meta["pushes"]:
        push["magnitude_n"] = 0.0
    assert [vars(p) for p in env.scenario.pushes] == expected
    assert [vars(p) for p in env._pushes_by_step.values()] == expected


def test_trajectory_needs_record_and_an_episode() -> None:
    plain = BalanceEnv("ideal")
    plain.reset(seed=0)
    with pytest.raises(RuntimeError, match="record=True"):
        plain.trajectory()
    with pytest.raises(RuntimeError, match="record=True"):
        BalanceEnv("ideal", record=True).trajectory()


def test_recording_restarts_with_each_episode() -> None:
    env = BalanceEnv("clean", record=True)
    rollout(env, seed=0, max_steps=40)
    second = rollout(env, seed=1, max_steps=15)
    traj = env.trajectory()
    assert traj.steps == 15
    assert_bitwise_equal(traj.qpos, np.array(second.qpos))
    assert_bitwise_equal(traj.obs, np.array(second.obs[:-1]))
    assert traj.meta["scenario"]["seed"] == 1


def test_energy_follows_the_battery_formula() -> None:
    env = BalanceEnv("clean", record=True)
    ep = rollout(env, wobbly_pid(3), seed=3, max_steps=400)
    traj, motor, b = env.trajectory(), env.params.motor, env.binding
    angles = (
        traj.qpos[:, list(b.wheel_qposadr)] - traj.qpos[0, list(b.wheel_qposadr)]
    ) * b.wheel_sign
    omega = np.diff(angles, axis=0) / env.control_dt
    power = (
        motor.v_nom
        * motor.stall_current_a
        * traj.ctrl
        * (traj.ctrl - omega / motor.omega_noload_rads)
    )
    expected = math.fsum(np.maximum(power, 0.0).ravel()) * env.control_dt
    energies = [info["energy_j"] for info in ep.infos]
    assert energies[-1] == pytest.approx(expected, rel=1e-9)
    assert energies == sorted(energies)
    assert energies[-1] > 0.5


# ------------------------------------------------------------------------------- mirrored robots


@pytest.mark.parametrize("flipped", [("right",), ("left",), ("left", "right")])
@pytest.mark.parametrize("preset", ["ideal", "clean"])
def test_mirrored_wheel_axes_are_invisible(
    preset: str, flipped: tuple[str, ...], tmp_path: Path
) -> None:
    edits = [(wheel_axis(side, "0 1 0"), wheel_axis(side, "0 -1 0")) for side in flipped]
    path = robot_variant(tmp_path, "mirrored", *edits)
    original = BalanceEnv(preset, record=True)
    mirrored = BalanceEnv(preset, robot_xml=path, record=True)
    sign = tuple(-1.0 if side in flipped else 1.0 for side in ("left", "right"))
    assert mirrored.binding.wheel_sign == sign

    options = {"init_pitch_rate": 0.3}  # also exercises the no-slip initial wheel speed
    a = rollout(original, wobbly_pid(1), seed=1, options=options)
    b = rollout(mirrored, wobbly_pid(1), seed=1, options=options)
    assert a.steps == b.steps
    assert a.steps > 500
    # Value-equal (ideal wheel channels can differ in the sign of zero: -1 * 0.0 == -0.0).
    np.testing.assert_array_equal(np.array(a.obs), np.array(b.obs))
    assert a.rewards == b.rewards
    for x, y in zip(a.infos, b.infos, strict=True):
        assert (x["energy_j"], x["drift_m"], x["tilt"]) == (y["energy_j"], y["drift_m"], y["tilt"])
    qpos_flip, qvel_flip = np.ones(original.model.nq), np.ones(original.model.nv)
    qpos_flip[list(original.binding.wheel_qposadr)] = sign
    qvel_flip[list(original.binding.wheel_dofadr)] = sign
    np.testing.assert_array_equal(np.array(a.qpos) * qpos_flip, np.array(b.qpos))
    np.testing.assert_array_equal(np.array(a.qvel) * qvel_flip, np.array(b.qvel))
    assert mirrored.trajectory().meta["wheel_sign"] == list(sign)
    assert_bitwise_equal(mirrored.trajectory().enc_counts, original.trajectory().enc_counts)


# ------------------------------------------------------------------------------------- rendering


@pytest.mark.render
def test_render_rgb_array_returns_the_current_frame() -> None:
    env = BalanceEnv("ideal", render_mode="rgb_array", render_size=(160, 120))
    try:
        env.reset(seed=0, options={"init_pitch_rad": 0.05})
        upright = env.render()
        assert upright is not None
        assert upright.shape == (120, 160, 3)
        assert upright.dtype == np.uint8
        assert upright.std() > 5.0  # a scene, not a blank buffer
        rollout(env, idle, options={"init_pitch_rad": 0.05})
        fallen = env.render()
        assert fallen is not None
        assert not np.array_equal(fallen, upright)
    finally:
        env.close()
    env.close()  # idempotent
    default = BalanceEnv("ideal", render_mode="rgb_array")
    try:
        default.reset(seed=0)
        frame = default.render()
        assert frame is not None
        assert frame.shape == (240, 320, 3)
    finally:
        default.close()


def test_render_without_render_mode_returns_none() -> None:
    env = BalanceEnv("ideal")
    env.reset(seed=0)
    assert env.render() is None


# ------------------------------------------------------------------------------------ regressions


def test_wheel_friction_randomization_below_one_reaches_the_contacts() -> None:
    env = BalanceEnv("clean")
    env.reset(seed=0)
    slippery = replace(env.scenario, model=replace(env.scenario.model, wheel_friction_scale=0.9))
    env.reset(options={"scenario": slippery, "init_pitch_rad": 0.0})
    env.step([0.0, 0.0])
    d, wheels = env.data, env.binding.wheel_geoms
    frictions = [
        float(d.contact[i].friction[0])
        for i in range(d.ncon)
        if {int(d.contact[i].geom1), int(d.contact[i].geom2)} & wheels
    ]
    assert frictions
    assert frictions == pytest.approx([0.9 * env.params.contact.wheel_friction] * len(frictions))


# ------------------------------------------------------------------------------ bundles, scenarios


def test_env_uses_the_challenge_object_it_is_given(tmp_path: Path) -> None:
    """Hashing and scoring stay on one bundle: the env never re-loads the challenge by id."""
    root = tmp_path / "balance-v1"
    shutil.copytree(challenge_dir("balance-v1"), root)
    yaml_path = root / "challenge.yaml"
    yaml_path.write_text(yaml_path.read_text().replace("e_ref_j: 13.0", "e_ref_j: 99.0"))
    bundle = load_challenge_dir(root)
    env = BalanceEnv("clean", challenge=bundle)
    assert env.challenge is bundle
    assert env.challenge.config.score.e_ref_j == 99.0
    assert env.robot_xml == root / "robot" / "balancer.xml"
    assert BalanceEnv("clean", challenge="balance-v1").challenge.config.score.e_ref_j == 13.0


def test_reset_validates_scenario_dicts() -> None:
    env = BalanceEnv("standard")
    data = scenario_for(5, "standard").to_dict()
    data["pushes"][0]["magnitude_n"] = 1e6
    with pytest.raises(ValueError, match="scenario"):
        env.reset(options={"scenario": data})
    late = scenario_for(5, "standard").to_dict()
    late["pushes"][-1]["start_step"] = env.max_steps
    with pytest.raises(ValueError, match="scenario"):
        env.reset(options={"scenario": late})


def test_termination_reason_is_exposed() -> None:
    env = BalanceEnv("ideal")
    env.reset(seed=0, options={"init_pitch_rad": 0.9})
    assert env.termination_reason is None
    env.step([0.0, 0.0])
    assert env.termination_reason == "fell"
    env.reset(seed=0)
    assert env.termination_reason is None
