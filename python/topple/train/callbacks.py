"""SB3 callbacks for training (docs/design/phase-2.md §2.4, §2.5, §1; train extra).

``SelectionCallback`` runs §2.4's selection evals. PPO evaluates at rollout starts and SAC at
steps, both through ``EvalTrigger``, plus a final eval. Each eval goes through the real harness
on the in-memory ONNX bytes and then writes the snapshot in the same call. The callback also
checks §1's hard limits. ``HealthCallback`` (PPO) logs the per-rollout policy and penalty
numbers to TensorBoard.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from stable_baselines3.common.callbacks import BaseCallback

from topple.challenge import Challenge
from topple.evaluate import EvalProtocol
from topple.export import build_onnx
from topple.pool import PolicyRef
from topple.train.config import PpoTrainConfig, SacTrainConfig
from topple.train.resources import ResourceReading, ResourceSampler, abort_flag
from topple.train.selection import (
    EvalTrigger,
    GateBaseline,
    HistoryRow,
    choose_best,
    evaluate_policy_bytes,
    is_eligible,
    write_best_json,
)
from topple.train.snapshot import actor_layers, obs_norm, snapshot

LIMIT_CHECK_INTERVAL_S = 1.0  # the sampler's own period: checking more often reads the same data


@dataclass(frozen=True)
class PolicyStats:
    """§2.5: the policy's mean std and the fraction of pre-clip mean entries with |mean| > 1,
    on the eval's on-policy rows."""

    std_mean: float
    frac_mean_outside_1: float


def _features(module: Any, vecnorm: Any, rows: npt.NDArray[np.float32]) -> torch.Tensor:
    """Normalized rows through ``module``'s own features extractor (PPO: the policy; SAC: the
    actor, whose extractor is not shared with the critics)."""
    z = vecnorm.normalize_obs(rows.astype(np.float64))
    obs = torch.as_tensor(np.asarray(z, dtype=np.float32))
    features: torch.Tensor = module.extract_features(obs, module.features_extractor)
    return features


def policy_stats(model: Any, vecnorm: Any, onpolicy: npt.NDArray[np.float32]) -> PolicyStats | None:
    """``std_mean`` and ``frac_mean_outside_1`` on ``onpolicy`` (``None`` without rows). PPO:
    ``exp(log_std).mean()`` and the ``action_net`` mean; SAC: the mean over rows of
    ``exp(clamp(log_std, -20, 2))`` and the pre-tanh ``mu``."""
    if onpolicy.shape[0] == 0:
        return None
    with torch.no_grad():
        if hasattr(model.policy, "action_net"):  # PPO
            policy = model.policy
            latent = policy.mlp_extractor.forward_actor(_features(policy, vecnorm, onpolicy))
            mean = policy.action_net(latent)
            std = float(torch.exp(policy.log_std).mean())
        else:  # SAC
            actor = model.actor
            latent = actor.latent_pi(_features(actor, vecnorm, onpolicy))
            mean = actor.mu(latent)
            std = float(torch.exp(torch.clamp(actor.log_std(latent), -20.0, 2.0)).mean())
        outside = float((mean.abs() > 1.0).float().mean())
    return PolicyStats(std_mean=std, frac_mean_outside_1=outside)


@dataclass
class SelectionContext:
    """What the selection callback needs, built once by ``learn.train``."""

    cfg: PpoTrainConfig | SacTrainConfig
    out: Path
    vecnorm: Any
    pool: Executor
    protocol: EvalProtocol
    baseline: GateBaseline
    challenge: Challenge
    e_ref_j: float
    sampler: ResourceSampler
    started: float  # time.monotonic() at train start
    trigger: EvalTrigger
    history: list[HistoryRow] = field(default_factory=list)
    scores: list[tuple[float, ...]] = field(default_factory=list)
    abort: str | None = None
    final_reading: ResourceReading | None = None
    selection_s: float = 0.0


class SelectionCallback(BaseCallback):
    """§2.4's selection evals and snapshots, and §1's hard limits."""

    def __init__(self, ctx: SelectionContext) -> None:
        super().__init__(verbose=0)
        self.ctx = ctx
        self._next_limit_check = 0.0
        self._excess_sum = 0.0
        self._excess_n = 0
        self._best_key: tuple[int, bool] | None = None

    # --- SB3 hooks ---

    def _on_rollout_start(self) -> None:
        if self.ctx.cfg.algo == "ppo" and self.ctx.trigger.due(self.num_timesteps):
            self._evaluate(self.num_timesteps)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            if "excess_penalty" in info:
                self._excess_sum += float(info["excess_penalty"])
                self._excess_n += 1
        self.ctx.sampler.set_step(self.num_timesteps)
        if self.ctx.cfg.algo == "sac" and self.ctx.trigger.due(self.num_timesteps):
            self._evaluate(self.num_timesteps)
        return self._within_limits()

    def _on_training_end(self) -> None:
        n = self.model.num_timesteps
        if self.ctx.trigger.final_due(n):
            self._evaluate(n)
        # §1: the final reading comes before venv.close(), while the env workers are alive.
        self.ctx.final_reading = self.ctx.sampler.reading()
        self.logger.dump(step=n)

    # --- internals ---

    def _within_limits(self) -> bool:
        now = time.monotonic()
        if now < self._next_limit_check:
            return True
        self._next_limit_check = now + LIMIT_CHECK_INTERVAL_S
        flag = abort_flag(self.ctx.sampler.reading(), self.ctx.cfg.limits, now - self.ctx.started)
        if flag is None:
            return True
        self.ctx.abort = flag
        return False

    def _evaluate(self, step: int) -> None:
        ctx = self.ctx
        wall0 = time.monotonic()
        cpu0 = ctx.sampler.reading().cpu_s
        layers = actor_layers(self.model)
        norm = obs_norm(ctx.vecnorm)
        data = build_onnx(
            layers,
            norm,
            activation=ctx.cfg.policy.activation,
            challenge=ctx.challenge,
        )
        sha = hashlib.sha256(data).hexdigest()
        ref = PolicyRef("onnx", data, sha, ctx.challenge.id)
        result, onpolicy = evaluate_policy_bytes(ref, ctx.protocol, ctx.pool, step=step)
        eligible = is_eligible(result, ctx.baseline, ctx.cfg.selection, ctx.e_ref_j)
        stats = policy_stats(self.model, ctx.vecnorm, onpolicy)
        snapshot(
            ctx.cfg,
            self.model,
            ctx.vecnorm,
            step,
            ctx.out / "ckpt",
            layers=layers,
            norm=norm,
            onnx_sha256=sha,
            onpolicy=onpolicy,
        )
        excess = self._excess_sum / self._excess_n if self._excess_n else None
        self._excess_sum, self._excess_n = 0.0, 0
        wall = time.monotonic() - wall0
        row = HistoryRow(
            result=result,
            eligible=eligible,
            onnx_sha256=sha,
            std_mean=None if stats is None else stats.std_mean,
            frac_mean_outside_1=None if stats is None else stats.frac_mean_outside_1,
            excess_penalty_mean=excess if ctx.cfg.algo == "ppo" else None,
            cpu_s=ctx.sampler.reading().cpu_s - cpu0,
            wall_s=wall,
        )
        ctx.history.append(row)
        ctx.scores.append(result.scores)
        ctx.selection_s += wall
        best = choose_best(ctx.history, ctx.challenge.kind)  # type: ignore[arg-type]
        if best is not None and (best.step, best.eligible) != self._best_key:
            write_best_json(ctx.out / "best.json", best, ctx.baseline.rule)
            self._best_key = (best.step, best.eligible)
        self._log(row)

    def _log(self, row: HistoryRow) -> None:
        r = row.result
        self.logger.record("topple/selection/success", r.success_rate)
        self.logger.record("topple/selection/score", r.score)
        self.logger.record("topple/selection/median_energy", r.median_energy_j)
        self.logger.record("topple/selection/saturation", r.mean_saturation)
        if row.std_mean is not None:
            self.logger.record("topple/policy/std_mean", row.std_mean)
        if row.frac_mean_outside_1 is not None:
            self.logger.record("topple/policy/frac_mean_outside_1", row.frac_mean_outside_1)


class HealthCallback(BaseCallback):
    """PPO, per rollout (§2.5): ``topple/policy/frac_rollout_mean_outside_1`` from the rollout
    buffer (in ``_on_rollout_end``, before the next rollout resets it) and
    ``topple/train/excess_penalty_mean``."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._excess_sum = 0.0
        self._excess_n = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            if "excess_penalty" in info:
                self._excess_sum += float(info["excess_penalty"])
                self._excess_n += 1
        return True

    def _on_rollout_end(self) -> None:
        model: Any = self.model  # PPO: SB3 types the callback's model as BaseAlgorithm
        buffer = model.rollout_buffer
        obs = torch.as_tensor(buffer.observations.reshape(-1, buffer.observations.shape[-1]))
        with torch.no_grad():
            policy = model.policy
            features = policy.extract_features(obs.float(), policy.features_extractor)
            mean = policy.action_net(policy.mlp_extractor.forward_actor(features))
            outside = float((mean.abs() > 1.0).any(dim=1).float().mean())
        self.logger.record("topple/policy/frac_rollout_mean_outside_1", outside)
        if self._excess_n:
            self.logger.record(
                "topple/train/excess_penalty_mean", self._excess_sum / self._excess_n
            )
        self._excess_sum, self._excess_n = 0.0, 0
