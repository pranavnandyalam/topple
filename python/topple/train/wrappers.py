"""Training env wrappers (docs/design/phase-2.md §2.1, §2.3). Gymnasium only: importing this
module never imports torch or Stable-Baselines3 (``make_train_env`` imports SB3's ``Monitor``
when it is called).

- ``TrainingSeeds``: every training episode's seed is drawn from the run's own stream,
  ``default_rng(SeedSequence([base_seed, env_index, 0x7709]))``, into ``[2^52, 2^53)``, which no
  public or private seed set touches (§0); the seed SB3's ``VecEnv`` passes is ignored.
- ``ActionExcessPenalty`` (PPO): actions live in ``Box(-box, box)``, so the Gaussian policy is
  not squeezed against a ±1 boundary, and the reward pays ``weight * min(mean_i(max(|a_i| -
  1, 0)^2), 1)`` for the part beyond ±1. The env still clips to its own bounds.

``make_train_env`` composes env -> ``Monitor`` -> penalty -> ``TrainingSeeds``: the Monitor
sits inside the penalty, so ``rollout/ep_rew_mean`` excludes it (§2.5). No wrapper changes
``max_steps`` or truncation: the env ends every episode itself (contract §8, §11).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from topple.train.config import PpoTrainConfig, SacTrainConfig

TRAIN_SEED_LO: Final = 2**52
TRAIN_SEED_SPAN: Final = 2**52  # seeds are TRAIN_SEED_LO + [0, 2^52): [2^52, 2^53)
SEED_STREAM_TAG: Final = 0x7709

Obs = npt.NDArray[np.float32]
Act = npt.NDArray[np.float32]
StepResult = tuple[Obs, SupportsFloat, bool, bool, dict[str, Any]]


class TrainingSeeds(gym.Wrapper[Obs, Act, Obs, Act]):
    """Reset every episode with ``TRAIN_SEED_LO + int(rng.integers(2^52))`` (a Python ``int``),
    ``rng = default_rng(SeedSequence([base_seed, env_index, 0x7709]))``: deterministic per
    (run seed, env index), different per env. The ``seed`` a caller passes is ignored.
    ``last_seed`` is the latest episode's seed (``None`` before the first reset)."""

    def __init__(self, env: gym.Env[Obs, Act], base_seed: int, env_index: int) -> None:
        super().__init__(env)
        for name, value in (("base_seed", base_seed), ("env_index", env_index)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an int >= 0, got {value!r}")
        self._rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, env_index, SEED_STREAM_TAG])
        )
        self.last_seed: int | None = None

    def next_seed(self) -> int:
        """The next episode's seed (advances the stream)."""
        return TRAIN_SEED_LO + int(self._rng.integers(TRAIN_SEED_SPAN))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        self.last_seed = self.next_seed()
        return self.env.reset(seed=self.last_seed, options=options)


class ActionExcessPenalty(gym.Wrapper[Obs, Act, Obs, Act]):
    """PPO only: the action space is ``Box(-box, box)`` of the env's action shape (the env's
    own space must be ``Box(-1, 1)``); each step's reward is reduced by ``weight *
    min(mean_i(max(|a_i| - 1, 0)^2), 1)``, also reported as ``info["excess_penalty"]``. The
    action reaches the env unchanged, and the env clips it."""

    def __init__(self, env: gym.Env[Obs, Act], box: float, weight: float) -> None:
        super().__init__(env)
        inner = env.action_space
        if not (
            isinstance(inner, gym.spaces.Box)
            and np.all(inner.low == -1.0)
            and np.all(inner.high == 1.0)
        ):
            raise ValueError("the action-excess penalty wraps an env whose actions are in ±1")
        if not (np.isfinite(box) and box >= 1.0):
            raise ValueError(f"box must be finite and at least 1, got {box!r}")
        if not (np.isfinite(weight) and weight >= 0.0):
            raise ValueError(f"weight must be finite and non-negative, got {weight!r}")
        self.box = float(box)
        self.weight = float(weight)
        self.action_space = gym.spaces.Box(-self.box, self.box, shape=inner.shape, dtype=np.float32)

    def penalty(self, action: npt.ArrayLike) -> float:
        """``weight * min(mean_i(max(|a_i| - 1, 0)^2), 1)``."""
        a = np.asarray(action, dtype=np.float64)
        excess = np.maximum(np.abs(a) - 1.0, 0.0)
        return self.weight * min(float(np.mean(excess**2)), 1.0)

    def step(self, action: Act) -> StepResult:
        penalty = self.penalty(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = {**info, "excess_penalty": penalty}
        return obs, float(reward) - penalty, terminated, truncated, info


def make_train_env(cfg: PpoTrainConfig | SacTrainConfig, env_index: int) -> gym.Env[Obs, Act]:
    """Training env ``env_index`` of ``cfg``: the challenge's env (balance: ``BalanceEnv`` on
    ``cfg.env.preset`` with the config's reward and ``max_drift_m``, not recording) ->
    SB3's ``Monitor`` -> ``ActionExcessPenalty`` (PPO, when enabled) -> ``TrainingSeeds(cfg.seed,
    env_index)``. Needs the ``train`` extra (SB3), imported here, not at module import."""
    from stable_baselines3.common.monitor import Monitor

    from topple.challenge import load_challenge
    from topple.envs.balance import BalanceEnv
    from topple.train.config import ActionExcessOn

    challenge = load_challenge(cfg.env.challenge)
    kwargs: dict[str, object] = {
        "challenge": challenge,
        "reward": cfg.env.reward.reward_config(),
        "max_drift_m": cfg.env.max_drift_m,
        "record": False,
    }
    if cfg.env.waypoint_reward is not None:
        from topple.envs.waypoint import WaypointEnv, WaypointRewardConfig

        reward = WaypointRewardConfig(**cfg.env.waypoint_reward.model_dump())
        env: gym.Env[Obs, Act] = WaypointEnv(cfg.env.preset, waypoint_reward=reward, **kwargs)  # type: ignore[arg-type]
    else:
        env = BalanceEnv(cfg.env.preset, **kwargs)  # type: ignore[arg-type]
    env = Monitor(env)
    excess = cfg.env.action_excess
    if cfg.algo == "ppo" and isinstance(excess, ActionExcessOn):
        env = ActionExcessPenalty(env, excess.box, excess.weight)
    return TrainingSeeds(env, cfg.seed, env_index)
