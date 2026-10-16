"""Training wrappers (docs/design/phase-2.md §2.1, §2.3, §6 ``test_train_wrappers``).

``TrainingSeeds`` draws Python-int seeds in [2^52, 2^53) from the run's own stream, never a
public seed; ``ActionExcessPenalty`` widens the action space to ±box and charges the excess;
``make_train_env`` composes env -> Monitor -> penalty -> seeds (tested here with a stand-in for
SB3's ``Monitor``; the real one is the ``train``-marked test).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest

from topple.envs.balance import BalanceEnv
from topple.train.config import load_train_config
from topple.train.wrappers import (
    TRAIN_SEED_LO,
    ActionExcessPenalty,
    TrainingSeeds,
    make_train_env,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "train"
# §0's seed sets: eval_v1, wp_eval_v1, calibration (balance and waypoint), selection
PUBLIC_SETS = (range(1000, 1020), range(3000, 3050), range(400), range(2000, 2200))

Obs = npt.NDArray[np.float32]


class FakeEnv(gym.Env[Obs, Obs]):
    """Records what reaches it; reward 1 per step."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.seeds: list[int | None] = []
        self.options: list[dict[str, Any] | None] = []
        self.actions: list[Any] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        self.seeds.append(seed)
        self.options.append(options)
        return np.zeros(10, np.float32), {}

    def step(self, action: Any) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        return np.zeros(10, np.float32), 1.0, False, False, {"inner": True}


def draws(base: int, index: int, n: int) -> list[int]:
    env = FakeEnv()
    wrapped = TrainingSeeds(env, base, index)
    for _ in range(n):
        wrapped.reset()
    return [s for s in env.seeds if s is not None]


# --- TrainingSeeds ------------------------------------------------------------------------------


def test_seeds_are_python_ints_in_the_training_range() -> None:
    seeds = draws(0, 0, 1000)
    assert len(seeds) == 1000
    assert all(type(s) is int for s in seeds)  # not np.int64
    assert all(TRAIN_SEED_LO <= s < 2**53 for s in seeds)
    assert len(set(seeds)) == 1000


def test_seeds_are_deterministic_per_run_seed_and_env_index() -> None:
    assert draws(0, 0, 20) == draws(0, 0, 20)
    assert draws(0, 0, 20) != draws(0, 1, 20)
    assert draws(0, 0, 20) != draws(1, 0, 20)
    assert set(draws(0, 1, 50)).isdisjoint(draws(0, 0, 50))


def test_the_stream_is_pinned() -> None:
    """``default_rng(SeedSequence([base, index, 0x7709])).integers(2**52)`` + 2^52: integer
    generation is exact on every platform, so the first seeds of a run are known answers."""
    expected = [
        TRAIN_SEED_LO + int(v)
        for v in np.random.default_rng(np.random.SeedSequence([7, 3, 0x7709])).integers(
            2**52, size=3
        )
    ]
    assert draws(7, 3, 3) == expected


def test_10k_draws_never_hit_a_public_seed_set() -> None:
    seeds = draws(0, 0, 5000) + draws(2, 7, 5000)
    assert len(seeds) == 10_000
    for public in PUBLIC_SETS:
        assert not any(s in public for s in seeds)
    assert min(seeds) >= 2**52


def test_the_incoming_seed_is_ignored_and_options_pass_through() -> None:
    env = FakeEnv()
    wrapped = TrainingSeeds(env, 0, 0)
    wrapped.reset(seed=1000, options={"pushes": ()})
    assert env.seeds[0] != 1000
    assert env.seeds[0] == wrapped.last_seed
    assert env.options == [{"pushes": ()}]


@pytest.mark.parametrize(("base", "index"), [(-1, 0), (0, -1), (True, 0), (0, 1.0)])
def test_bad_seeding_arguments_are_refused(base: Any, index: Any) -> None:
    with pytest.raises(ValueError, match="int >= 0"):
        TrainingSeeds(FakeEnv(), base, index)


def test_the_balance_env_accepts_a_training_seed() -> None:
    env = TrainingSeeds(BalanceEnv("standard"), 0, 0)
    try:
        obs, info = env.reset()
        assert obs.shape == (10,)
        assert info["episode_seed"] == env.last_seed
        assert TRAIN_SEED_LO <= info["episode_seed"] < 2**53
    finally:
        env.close()


# --- ActionExcessPenalty ------------------------------------------------------------------------


def penalized(action: tuple[float, float], weight: float = 0.10) -> tuple[float, dict[str, Any]]:
    env = FakeEnv()
    wrapped = ActionExcessPenalty(env, box=2.0, weight=weight)
    wrapped.reset()
    sent = np.asarray(action, np.float32)
    _, reward, _, _, info = wrapped.step(sent)
    assert env.actions[-1] is sent  # unchanged: the env clips it
    return float(reward), info


@pytest.mark.parametrize(
    ("action", "penalty"),
    [
        ((0.5, 1.5), 0.10 * 0.125),  # mean((0, 0.5)^2) = 0.125
        ((2.0, -2.0), 0.10),  # mean((1, 1)^2) = 1
        ((-2.0, 2.0), 0.10),
        ((0.2, -0.9), 0.0),
        ((1.0, -1.0), 0.0),
    ],
)
def test_the_penalty_is_exact(action: tuple[float, float], penalty: float) -> None:
    reward, info = penalized(action)
    assert info["excess_penalty"] == pytest.approx(penalty, abs=1e-15)
    assert reward == pytest.approx(1.0 - penalty, abs=1e-15)
    assert info["inner"] is True


def test_the_penalty_is_capped_at_weight() -> None:
    wrapped = ActionExcessPenalty(FakeEnv(), box=10.0, weight=0.25)
    assert wrapped.penalty([10.0, 10.0]) == 0.25  # mean((9, 9)^2) = 81, capped at 1


def test_the_action_space_is_the_box() -> None:
    wrapped = ActionExcessPenalty(FakeEnv(), box=2.0, weight=0.1)
    space = wrapped.action_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (2,)
    assert space.dtype == np.float32
    assert np.all(space.low == -2.0)
    assert np.all(space.high == 2.0)


@pytest.mark.parametrize(("box", "weight"), [(0.5, 0.1), (2.0, -0.1), (np.nan, 0.1), (2.0, np.inf)])
def test_bad_penalty_arguments_are_refused(box: float, weight: float) -> None:
    with pytest.raises(ValueError, match=r"box|weight"):
        ActionExcessPenalty(FakeEnv(), box=box, weight=weight)


def test_the_penalty_wraps_only_a_unit_box_env() -> None:
    env = FakeEnv()
    env.action_space = gym.spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
    with pytest.raises(ValueError, match="±1"):
        ActionExcessPenalty(env, box=2.0, weight=0.1)


def test_the_balance_env_clips_what_the_box_allows() -> None:
    env = ActionExcessPenalty(BalanceEnv("ideal"), box=2.0, weight=0.1)
    try:
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.asarray([2.0, -2.0], np.float32))
        assert info["excess_penalty"] == pytest.approx(0.1)
    finally:
        env.close()


# --- make_train_env -----------------------------------------------------------------------------


class StubMonitor(gym.Wrapper[Obs, Obs, Obs, Obs]):
    """Stands in for ``stable_baselines3.common.monitor.Monitor`` (the train extra)."""


@pytest.fixture
def stub_sb3(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        name: ModuleType(name)
        for name in (
            "stable_baselines3",
            "stable_baselines3.common",
            "stable_baselines3.common.monitor",
        )
    }
    modules["stable_baselines3.common.monitor"].Monitor = StubMonitor  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def layers(env: gym.Env[Any, Any]) -> list[type]:
    found: list[type] = []
    while True:
        found.append(type(env))
        if not isinstance(env, gym.Wrapper):
            return found
        env = env.env


@pytest.mark.usefixtures("stub_sb3")
def test_make_train_env_composes_env_monitor_penalty_seeds() -> None:
    cfg = load_train_config(CONFIGS / "smoke_ppo.yaml")
    env = make_train_env(cfg, 1)
    try:
        assert layers(env) == [TrainingSeeds, ActionExcessPenalty, StubMonitor, BalanceEnv]
        assert isinstance(env, TrainingSeeds)
        inner = env.unwrapped
        assert isinstance(inner, BalanceEnv)
        assert inner.max_drift_m == 2.0
        assert inner.record is False
        assert inner.reward_config == cfg.env.reward.reward_config()
        assert env.action_space.shape == (2,)
        expected = draws(cfg.seed, 1, 1)[0]
        env.reset()
        assert env.last_seed == expected
        _, _, _, _, info = env.step(np.asarray([2.0, -2.0], np.float32))
        assert info["excess_penalty"] == pytest.approx(0.10)
    finally:
        env.close()


@pytest.mark.usefixtures("stub_sb3")
@pytest.mark.parametrize("name", ["smoke_sac", "ablate_nopenalty"])
def test_make_train_env_without_the_penalty(name: str) -> None:
    env = make_train_env(load_train_config(CONFIGS / f"{name}.yaml"), 0)
    try:
        assert layers(env) == [TrainingSeeds, StubMonitor, BalanceEnv]
        space = env.action_space
        assert isinstance(space, gym.spaces.Box)
        assert np.all(space.high == 1.0)
    finally:
        env.close()


@pytest.mark.train
def test_make_train_env_with_sb3s_monitor() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.monitor import Monitor

    env = make_train_env(load_train_config(CONFIGS / "smoke_ppo.yaml"), 0)
    try:
        assert layers(env) == [TrainingSeeds, ActionExcessPenalty, Monitor, BalanceEnv]
    finally:
        env.close()
