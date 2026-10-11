"""Run controllers on a protocol and score them (docs/contract.md §8-12, §15).

``evaluate`` runs a fresh controller per episode, records every failure as a typed error instead of
crashing, scores each episode from its trajectory, and returns ``topple.results/1`` results.

In-process evaluation runs the controller's code inside this interpreter: it is for trusted code
(your own controllers, the shipped examples). Misbehaviour is still contained per episode, but a
hostile controller could tamper with the scorer; ranked runs isolate it (Phase 3).
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

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
    describe_exception,
    validate_action,
)
from topple.envs.balance import BalanceEnv
from topple.ids import parse_protocol, resolve_protocol
from topple.results import (
    ControllerInfo,
    EpisodeResult,
    ErrorInfo,
    EvalResults,
    Isolation,
    Versions,
)
from topple.results import summarize as summarize  # re-exported: §12's aggregate
from topple.rng import controller_seed
from topple.robot import robot_hash
from topple.scoring import episode_metrics, episode_score, episode_success
from topple.traj import Trajectory

ControllerFactory = Callable[[TaskSpec], Controller[Any]]
EPILOGUE_S = 1.0
SIM_UNSTABLE_DETAIL = "physics became unstable"
FALL_REASONS = frozenset({"fell", "floor_contact"})
MIN_PRIVATE_SALT = 2**64
"""Smallest salt of a private protocol: its salt is a 128-bit secret (contract §8, §9), so a
smaller one is almost certainly forgotten or hand-typed, and a controller could brute-force it."""
MAX_PRIVATE_SALT = 2**128 - 1
RunnerErrorCode = Literal["timeout", "crash", "protocol", "output_limit", "load_error"]
"""Run error codes (contract §15) that only the runner, not the harness, can know about."""
RUNNER_ERROR_CODES: frozenset[str] = frozenset(get_args(RunnerErrorCode))


class RunnerError(Exception):
    """Raised by a harness-owned controller proxy (the Phase 3 runner) for a failure of its own
    (no reply in time, a crashed subprocess, a malformed or oversized reply, a submission that
    would not load): the episode ends with ``code`` (contract §15) instead of ``exception``.

    Only this exact class, holding the code and detail it was built with, names a code; a
    subclass, or an instance whose attributes were changed, is an ordinary exception. An
    in-process controller could raise it too, which only picks another zero-scoring code.
    """

    def __init__(self, code: RunnerErrorCode, detail: str) -> None:
        _check_runner_error(code, detail)
        super().__init__(code, detail)
        self.code: str = code
        self.detail: str = detail


def _check_runner_error(code: object, detail: object) -> None:
    if type(code) is not str or code not in RUNNER_ERROR_CODES:
        raise ValueError(
            f"a runner error code is one of {sorted(RUNNER_ERROR_CODES)}, got {code!r}"
        )
    if type(detail) is not str:
        raise TypeError(f"a runner error detail must be a str, got {type(detail).__name__}")


@dataclass(frozen=True)
class EvalProtocol:
    """A protocol: challenge, preset, seeds and salt (contract §8, §9, §13).

    A public protocol's salt is 0; a private one's is a secret in
    ``[MIN_PRIVATE_SALT, MAX_PRIVATE_SALT]``. Anything else raises ``ValueError``.
    """

    challenge: Challenge
    preset: str
    seeds: tuple[int, ...]
    seed_set: Literal["public", "private"] = "public"
    salt: int = 0

    def __post_init__(self) -> None:
        if self.seed_set not in ("public", "private"):
            raise ValueError(f"seed_set must be 'public' or 'private', got {self.seed_set!r}")
        if type(self.salt) is not int:
            raise ValueError(f"the salt must be an int, got {type(self.salt).__name__}")
        if self.seed_set == "public" and self.salt != 0:
            raise ValueError("a public protocol's salt is 0: its seeds are public anyway")
        if self.seed_set == "private" and not (MIN_PRIVATE_SALT <= self.salt <= MAX_PRIVATE_SALT):
            raise ValueError(
                "a private protocol's salt is a secret in [2**64, 2**128) "
                "(e.g. 2**64 + secrets.randbelow(2**128 - 2**64))"
            )

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


def _error(code: str, step: int, detail: str) -> ErrorInfo:
    return ErrorInfo.model_validate({"code": code, "step": step, "detail": detail})


def _controller_error(exc: BaseException, step: int) -> ErrorInfo:
    """The error that ``exc``, raised by the controller's factory, ``reset`` or ``act``, earns:
    a ``RunnerError``'s own code, else ``exception``. Never raises (but ``KeyboardInterrupt``)."""
    reported = _runner_report(exc)
    if reported is None:
        return _error("exception", step, describe_exception(exc))
    code, detail = reported
    return _error(code, step, detail)


def _runner_report(exc: BaseException) -> tuple[str, str] | None:
    """``(code, detail)`` of a genuine ``RunnerError``, else None.

    ``type(exc)`` is the real class (``exc.__class__`` is never read), so a subclass is not one;
    the code and detail must still be exact ``str`` objects a runner error may hold, and reading
    them (an attribute deleted or set to anything) can fail without failing the harness.
    """
    if type(exc) is not RunnerError:
        return None
    try:
        code, detail = exc.code, exc.detail
    except KeyboardInterrupt:
        raise
    except BaseException:
        return None
    if type(code) is str and code in RUNNER_ERROR_CODES and type(detail) is str:
        return code, detail
    return None


def _start_controller(
    env: BalanceEnv, factory: ControllerFactory, seed: int, salt: int
) -> tuple[Controller[Any], None] | tuple[None, ErrorInfo]:
    """Build and reset a fresh controller: ``(controller, None)``, or ``(None, why it failed)``.

    A pair rather than a union so the caller never has to ask the untrusted object what it is
    (``isinstance`` would run the object's own ``__class__``).
    """
    try:
        controller = factory(env.task_spec)
        controller.reset(controller_seed(seed, salt))
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # SystemExit, GeneratorExit, ...: all the controller's fault
        return None, _controller_error(exc, 0)
    return controller, None


def _next_action(
    controller: Controller[Any], obs_arr: npt.NDArray[np.float32], t: float, step: int
) -> tuple[float, ...] | ErrorInfo:
    """The controller's validated action for this step, or the ``ErrorInfo`` it earned."""
    try:
        raw = controller.act(controller_obs(obs_arr, t))
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        return _controller_error(exc, step)
    try:
        return validate_action(raw, BALANCE_V1_ACTION.dim)
    except ActionError as exc:
        return _error("invalid_action", step, describe_exception(exc))
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # the returned object's own code raised e.g. SystemExit
        return _error("exception", step, describe_exception(exc))


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
        if isinstance(action, ErrorInfo):  # our own object either way: safe to inspect
            return action
        obs_arr, _, terminated, truncated, info = env.step(action)
        if info["termination_reason"] == "sim_unstable":
            return _error("sim_unstable", int(info["step"]), SIM_UNSTABLE_DETAIL)
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
    controller, error = _start_controller(env, factory, seed, salt)
    if controller is not None:
        error = _control_loop(env, controller, obs_arr, info)
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


def platform_tag() -> str:
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def build_results(
    protocol: EvalProtocol,
    episodes: list[EpisodeResult],
    controller: ControllerInfo,
    isolation: Isolation,
) -> EvalResults:
    """``topple.results/1`` for ``episodes`` of ``protocol``; ``complete`` when all of its N
    episodes are listed and none is ``missing`` (contract §15)."""
    private = protocol.seed_set == "private"
    missing = any(e.error is not None and e.error.code == "missing" for e in episodes)
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
        complete=len(episodes) == len(protocol.seeds) and not missing,
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
