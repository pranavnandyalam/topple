"""Run controllers on a protocol and score them (docs/contract.md §8-12, §15).

``evaluate`` runs a fresh controller per episode, records every failure as a typed error instead of
crashing, scores each episode from its trajectory, and returns ``topple.results/1`` results.

In-process evaluation runs the controller's code inside this interpreter: it is for trusted code
(your own controllers, the shipped examples). Misbehaviour is still contained per episode, but a
hostile controller could tamper with the scorer; ranked runs isolate it (Phase 3).
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
import numpy.typing as npt

from topple import __version__
from topple.challenge import Challenge, load_challenge
from topple.contract import (
    BALANCE_V1_ACTION,
    OBS_SCHEMA_ID,
    ActionError,
    Controller,
    TaskSpec,
    controller_obs,
    validate_action,
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
from topple.scoring import episode_metrics, episode_score, episode_success
from topple.traj import Trajectory

ControllerFactory = Callable[[TaskSpec], Controller[Any]]
EPILOGUE_S = 1.0
FALL_REASONS = frozenset({"fell", "floor_contact"})


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
        return self.challenge.protocol_hash(self.preset, seeds=self.seeds, salt=self.salt)


@dataclass(frozen=True)
class EpisodeRun:
    result: EpisodeResult
    trajectory: Trajectory


def wants_epilogue(termination_reason: str, has_error: bool) -> bool:
    """Whether to simulate the 1 s zero-command epilogue after an episode (contract §11).

    Only after a fall (``fell``, ``floor_contact``) or a controller error, so the replay shows what
    happened next. Never after truncation (survival: success videos end upright) and never after
    ``sim_unstable`` (there is no meaningful physics to continue).
    """
    if termination_reason == "sim_unstable":
        return False
    return termination_reason in FALL_REASONS or has_error


def _describe(exc: BaseException) -> str:
    """``Type: message`` for an exception from untrusted code, even if its ``__str__`` fails."""
    try:
        message = str(exc)
    except KeyboardInterrupt:
        raise
    except BaseException:
        message = "<str() failed>"
    return f"{type(exc).__name__}: {message}"


def _error(code: str, step: int, exc: BaseException | str) -> ErrorInfo:
    detail = exc if isinstance(exc, str) else _describe(exc)
    return ErrorInfo.model_validate({"code": code, "step": step, "detail": detail})


def _start_controller(
    env: BalanceEnv, factory: ControllerFactory, seed: int, salt: int
) -> Controller[Any] | ErrorInfo:
    """Build and reset a fresh controller, or describe why that failed."""
    try:
        controller = factory(env.task_spec)
        controller.reset(controller_seed(seed, salt))
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # SystemExit, GeneratorExit, ...: all the controller's fault
        return _error("exception", 0, exc)
    return controller


def _next_action(
    controller: Controller[Any], obs_arr: npt.NDArray[np.float32], t: float, step: int
) -> tuple[float, ...] | ErrorInfo:
    """The controller's validated action for this step, or the ``ErrorInfo`` it earned."""
    try:
        raw = controller.act(controller_obs(obs_arr, t))
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        return _error("exception", step, exc)
    try:
        return validate_action(raw, BALANCE_V1_ACTION.dim)
    except ActionError as exc:
        return _error("invalid_action", step, exc)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # the returned object's own code raised e.g. SystemExit
        return _error("exception", step, exc)


def _control_loop(
    env: BalanceEnv,
    controller: Controller[Any],
    obs_arr: npt.NDArray[np.float32],
    info: dict[str, Any],
) -> ErrorInfo | None:
    """Step until the episode ends; return the controller or physics error, if any."""
    while True:
        step = int(info["step"])
        action = _next_action(controller, obs_arr, float(info["t"]), step)
        if isinstance(action, ErrorInfo):
            return action
        obs_arr, _, terminated, truncated, info = env.step(action)
        if info["termination_reason"] == "sim_unstable":
            return _error("sim_unstable", int(info["step"]), "physics became unstable")
        if terminated or truncated:
            return None


def run_episode(
    env: BalanceEnv,
    factory: ControllerFactory,
    seed: int,
    index: int,
    *,
    salt: int = 0,
    report_seed: bool = True,
) -> EpisodeRun:
    """One episode. Never raises for controller misbehaviour: errors land in the result.

    Any exception from the controller (its factory, ``reset``, ``act``, or the code of the object
    it returns) becomes a typed error that scores 0; only ``KeyboardInterrupt`` propagates.
    """
    obs_arr, info = env.reset(seed=seed)
    started = _start_controller(env, factory, seed, salt)
    if isinstance(started, ErrorInfo):
        error: ErrorInfo | None = started
    else:
        error = _control_loop(env, started, obs_arr, info)
    if wants_epilogue(env.termination_reason or "timeout", error is not None):
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
    env = BalanceEnv(protocol.preset, challenge=protocol.challenge, record=True)
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
    """Aggregate over the ``expected`` episodes of a protocol (contract §12, §15).

    ``score`` is the exact sum of episode scores divided by ``expected`` and ``success_rate`` is
    successes over ``expected``: a missing or errored episode counts as 0. ``mean_time_upright_s``
    is the mean over the episodes that are present (it describes the runs that happened).
    """
    if len(episodes) > expected:
        raise ValueError(f"{len(episodes)} episodes, but only {expected} expected")
    times = [float(e.metrics["time_upright_s"]) for e in episodes if e.metrics is not None]
    score = math.fsum(e.score for e in episodes) / expected if expected else 0.0
    return Aggregate(
        score=min(max(score, 0.0), 100.0),
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
