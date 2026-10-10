"""Run controllers on a protocol and score them (docs/contract.md §8-12, §15).

``evaluate`` runs a fresh controller per episode, records every failure as a typed error instead of
crashing, scores each episode from its trajectory, and returns ``topple.results/1`` results.
"""

from __future__ import annotations

import math
import platform
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from topple import __version__
from topple.challenge import Challenge, load_challenge
from topple.contract import (
    BALANCE_V1_ACTION,
    OBS_SCHEMA_ID,
    ActionError,
    BalancerObs,
    Controller,
    TaskSpec,
)
from topple.envs.balance import BalanceEnv
from topple.ids import parse_protocol, resolve_protocol
from topple.results import (
    Aggregate,
    ControllerInfo,
    EpisodeResult,
    ErrorInfo,
    EvalResults,
    Isolation,
    Versions,
)
from topple.rng import controller_seed
from topple.robot import robot_hash
from topple.scoring import aggregate_score, episode_metrics, episode_score, episode_success
from topple.traj import Trajectory

ControllerFactory = Callable[[TaskSpec], Controller[Any]]
EPILOGUE_S = 1.0


@dataclass(frozen=True)
class EvalProtocol:
    challenge: Challenge
    preset: str
    seeds: tuple[int, ...]
    seed_set: Literal["public", "private"] = "public"
    salt: int = 0

    @property
    def id(self) -> str:
        return f"{self.challenge.id}/{self.preset}"

    @classmethod
    def from_id(cls, protocol_id: str) -> EvalProtocol:
        challenge_id, preset = parse_protocol(resolve_protocol(protocol_id))
        challenge = load_challenge(challenge_id)
        if preset not in challenge.config.presets:
            raise ValueError(f"{challenge_id} has no preset {preset!r}")
        return cls(challenge=challenge, preset=preset, seeds=challenge.config.seeds.public)

    def protocol_hash(self) -> str:
        return self.challenge.protocol_hash(self.preset)


@dataclass(frozen=True)
class EpisodeRun:
    result: EpisodeResult
    trajectory: Trajectory


def _error(code: str, step: int, exc: BaseException | str) -> ErrorInfo:
    detail = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return ErrorInfo.model_validate({"code": code, "step": step, "detail": detail})


def run_episode(
    env: BalanceEnv,
    factory: ControllerFactory,
    seed: int,
    index: int,
    *,
    salt: int = 0,
    report_seed: bool = True,
) -> EpisodeRun:
    """One episode: never raises for controller misbehavior; errors land in the result."""
    error: ErrorInfo | None = None
    obs_arr, info = env.reset(seed=seed)
    step = 0
    try:
        controller = factory(env.task_spec)
        controller.reset(controller_seed(seed, salt))
    except Exception as exc:
        error = _error("exception", 0, exc)
    else:
        obs = BalancerObs.from_array(obs_arr, info["t"])
        while True:
            try:
                action = controller.act(obs)
            except Exception as exc:
                error = _error("exception", step, exc)
                break
            try:
                obs_arr, _, terminated, truncated, info = env.step(action)
            except ActionError as exc:
                error = _error("invalid_action", step, exc)
                break
            step = info["step"]
            if info["termination_reason"] == "sim_unstable":
                error = _error("sim_unstable", step, "physics became unstable")
                break
            if terminated or truncated:
                break
            obs = BalancerObs.from_array(obs_arr, info["t"])
    env.simulate_epilogue(EPILOGUE_S)
    traj = env.trajectory({"episode_index": index, "error": error.model_dump() if error else None})
    metrics = episode_metrics(traj, env.params.motor)
    score_cfg = env.challenge.config.score
    score = 0.0 if error else episode_score(metrics, score_cfg)
    result = EpisodeResult(
        index=index,
        seed=seed if report_seed else None,
        success=error is None and episode_success(metrics, score_cfg),
        score=min(max(score, 0.0), 100.0),
        metrics=metrics.as_dict(),
        error=error,
    )
    return EpisodeRun(result=result, trajectory=traj)


def iter_episodes(protocol: EvalProtocol, factory: ControllerFactory) -> Iterator[EpisodeRun]:
    env = BalanceEnv(protocol.preset, challenge=protocol.challenge.id, record=True)
    try:
        for index, seed in enumerate(protocol.seeds):
            yield run_episode(
                env,
                factory,
                seed,
                index,
                salt=protocol.salt,
                report_seed=protocol.seed_set == "public",
            )
    finally:
        env.close()


def summarize(episodes: list[EpisodeResult], expected: int) -> Aggregate:
    times = [float(e.metrics["time_upright_s"]) for e in episodes if e.metrics is not None]
    return Aggregate(
        score=min(max(aggregate_score([e.score for e in episodes]), 0.0), 100.0),
        success_rate=(sum(e.success for e in episodes) / expected) if expected else 0.0,
        mean_time_upright_s=math.fsum(times) / len(times) if times else 0.0,
        episodes=len(episodes),
        errors=sum(e.error is not None for e in episodes),
    )


def platform_tag() -> str:
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def build_results(
    protocol: EvalProtocol,
    episodes: list[EpisodeResult],
    controller: ControllerInfo,
    isolation: Isolation,
) -> EvalResults:
    private = protocol.seed_set == "private"
    return EvalResults(
        challenge=protocol.challenge.id,
        protocol=protocol.id,
        protocol_hash=protocol.protocol_hash(),
        seed_set=protocol.seed_set,
        versions=Versions(
            topple=__version__,
            mujoco=mujoco.__version__,
            numpy=np.__version__,
            obs_schema=OBS_SCHEMA_ID,
            action_spec=BALANCE_V1_ACTION.id,
        ),
        platform=platform_tag(),
        robot_hash=robot_hash(protocol.challenge.robot_xml, protocol.challenge.params),
        controller=controller,
        isolation=isolation,
        complete=len(episodes) == len(protocol.seeds),
        episodes=None if private else episodes,
        aggregate=summarize(episodes, len(protocol.seeds)),
    )


def evaluate(
    protocol: EvalProtocol,
    factory: ControllerFactory,
    controller: ControllerInfo,
    *,
    isolation: Isolation = "inprocess",
    traj_dir: Path | None = None,
) -> EvalResults:
    """Run every seed of ``protocol``; optionally save ``traj_<index>.npz`` (public seeds only)."""
    episodes: list[EpisodeResult] = []
    for run in iter_episodes(protocol, factory):
        episodes.append(run.result)
        if traj_dir is not None and protocol.seed_set == "public":
            run.trajectory.save_npz(traj_dir / f"traj_{run.result.index:03d}.npz")
    return build_results(protocol, episodes, controller, isolation)
