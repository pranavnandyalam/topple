"""Run controllers on a protocol and score them (docs/contract.md §8-12, §15).

``evaluate`` runs a fresh controller per episode, records every failure as a typed error instead of
crashing, scores each episode from its trajectory, and returns ``topple.results/1`` results.

In-process evaluation runs the controller's code inside this interpreter: it is for trusted code
(your own controllers, the shipped examples). Misbehaviour is still contained per episode, but a
hostile controller could tamper with the scorer; ranked runs isolate it (Phase 3).
"""

from __future__ import annotations

import importlib.metadata
import itertools
import platform
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal, get_args

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
from topple.disturbances import Push
from topple.envs.balance import BalanceEnv
from topple.ids import parse_protocol, resolve_protocol
from topple.kinds import make_env, score_trajectory
from topple.results import (
    MAX_EPISODES,
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

    Only this exact class names a code, and only while ``code`` is a runner error code and
    ``detail`` a ``str``, both exact ``str`` objects, as the constructor requires (values
    reassigned since count as long as they qualify). A subclass, or an instance whose ``code`` or
    ``detail`` was deleted or set to anything else, is an ordinary exception. An in-process
    controller could raise it too, which only picks another zero-scoring code.
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
    ``[MIN_PRIVATE_SALT, MAX_PRIVATE_SALT]``. It has at most ``MAX_EPISODES`` seeds, the most
    episodes a results file lists (§15). Anything else raises ``ValueError``.
    """

    challenge: Challenge
    preset: str
    seeds: tuple[int, ...]
    seed_set: Literal["public", "private"] = "public"
    salt: int = 0

    def __post_init__(self) -> None:
        if len(self.seeds) > MAX_EPISODES:
            raise ValueError(
                f"a protocol has at most {MAX_EPISODES} seeds, got {len(self.seeds)}: a results "
                "file lists at most that many episodes"
            )
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

    def __repr__(self) -> str:
        """Never shows a private protocol's seeds or salt (contract §8): they are the secret
        that makes its runs rankable, and a repr ends up in logs and exception messages. (A
        field-by-field diff of two protocols, as pytest prints for ``==``, still shows them.)"""
        if self.seed_set == "private":
            seeds, salt = f"<{len(self.seeds)} secret seeds>", "<secret>"
        else:
            seeds, salt = repr(self.seeds), repr(self.salt)
        return (
            f"EvalProtocol(id={self.id!r}, seed_set={self.seed_set!r}, seeds={seeds}, salt={salt})"
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
    controller: Controller[Any],
    obs_arr: npt.NDArray[np.float32],
    t: float,
    step: int,
    schema: str = OBS_SCHEMA_ID,
) -> tuple[float, ...] | ErrorInfo:
    """The controller's validated action for this step, or the ``ErrorInfo`` it earned; the
    observation is built in the task's own schema (§8.4)."""
    try:
        raw = controller.act(controller_obs(obs_arr, t, schema=schema))
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


def _check_finite(obs_arr: npt.NDArray[Any], step: int) -> None:
    """``RuntimeError`` unless every feature the controller would see is finite.

    Non-finite observations are outside the policy domain (contract §17: runtimes disagree on
    NaN inputs), and the env never emits one (``sim_unstable`` rolls the bad substep back, a
    non-finite reset raises), so one here is a harness bug, like an episode that never ends. A
    real check, not an ``assert`` (``python -O`` strips those). The features are checked as the
    controller receives them, quantized to float32 (contract §6), where 1e39 is already inf.
    """
    with np.errstate(over="ignore"):
        quantized = np.asarray(obs_arr, dtype=np.float32)
    if not np.isfinite(quantized).all():
        raise RuntimeError(
            f"the env emitted a non-finite observation at step {step} (a harness bug, not the "
            "controller's: contract §17)"
        )


def _control_loop(
    env: BalanceEnv,
    controller: Controller[Any],
    obs_arr: npt.NDArray[np.float32],
    info: dict[str, Any],
) -> ErrorInfo | None:
    """Step until the episode ends; return the controller or physics error, if any.

    Bounded by the challenge's episode length (``task_spec.max_steps``), not by the env's own
    counter: an env that stopped ending its episodes is a harness bug, and evaluation fails
    loudly instead of recording (about 1 KB per step) without end. Every observation is checked
    to be finite before ``act`` sees it (``_check_finite``).
    """
    limit = env.task_spec.max_steps
    for _ in range(limit):
        step = int(info["step"])
        _check_finite(obs_arr, step)
        action = _next_action(controller, obs_arr, float(info["t"]), step, env.task_spec.obs_schema)
        if isinstance(action, ErrorInfo):  # our own object either way: safe to inspect
            return action
        obs_arr, _, terminated, truncated, info = env.step(action)
        if info["termination_reason"] == "sim_unstable":
            return _error("sim_unstable", int(info["step"]), SIM_UNSTABLE_DETAIL)
        if terminated or truncated:
            return None
    raise RuntimeError(
        f"the env did not end the episode within {env.challenge.id}'s {limit} steps "
        "(a harness bug, not the controller's)"
    )


def run_episode(
    env: BalanceEnv,
    factory: ControllerFactory,
    seed: int,
    index: int,
    *,
    salt: int = 0,
    report_seed: bool = True,
    pushes: Sequence[Push] | None = None,
) -> EpisodeRun:
    """One episode. Never raises for controller misbehaviour: errors land in the result.

    Any exception from the controller (its factory, ``reset``, ``act``, or the code of the object
    it returns) becomes a typed error that scores 0; only ``KeyboardInterrupt`` propagates.
    ``pushes`` replaces the seed's scenario pushes through the trusted reset hook (contract §9;
    benches such as ``topple shove``); ``None`` keeps them.
    """
    if pushes is None:
        obs_arr, info = env.reset(seed=seed)
    else:
        obs_arr, info = env.reset(seed=seed, options={"pushes": tuple(pushes)})
    controller, error = _start_controller(env, factory, seed, salt)
    if controller is not None:
        error = _control_loop(env, controller, obs_arr, info)
    if wants_epilogue(env.termination_reason or "timeout", error is not None):
        env.simulate_epilogue(EPILOGUE_S)
    traj = env.trajectory({"episode_index": index, "error": error.model_dump() if error else None})
    scored = score_trajectory(traj, env.challenge, error=error is not None)
    result = EpisodeResult(
        index=index,
        seed=seed if report_seed else None,
        success=scored.success,
        score=scored.score,
        metrics=scored.metrics,
        error=error,
    )
    return EpisodeRun(result=result, trajectory=traj)


def iter_episodes(protocol: EvalProtocol, factory: ControllerFactory) -> Iterator[EpisodeRun]:
    env = make_env(protocol.challenge, protocol.preset, record=True)
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


CPU_ISAS: Final = ("avx512f", "avx2", "sse4", "arm64-sme", "arm64-neon")
"""What ``cpu_isa`` reports: the widest vector extension that can change float results (numpy
and ONNX Runtime pick kernels by it), so per-episode results are compared per
(platform, ``cpu_isa``, onnxruntime version) (docs/design/phase-2.md §3.5)."""
_CPUINFO: Final = Path("/proc/cpuinfo")
_MAX_CPUINFO_LINES: Final = 4096  # the first processor's "flags" line comes within ~30
_SYSCTL: Final = "/usr/sbin/sysctl"


@lru_cache(maxsize=1)
def cpu_isa() -> str | None:
    """``avx512f`` / ``avx2`` / ``sse4`` from ``/proc/cpuinfo`` on Linux, ``arm64-sme`` /
    ``arm64-neon`` from ``sysctl hw.optional`` on macOS, ``None`` elsewhere or if unknown."""
    system = platform.system()
    if system == "Linux":
        return linux_cpu_isa(_CPUINFO)
    if system == "Darwin":
        return macos_cpu_isa(sysctl)
    return None


def linux_cpu_isa(cpuinfo: Path) -> str | None:
    """The ISA the first ``flags`` line of ``cpuinfo`` names (x86 only; ``None`` otherwise)."""
    try:
        with cpuinfo.open(encoding="ascii", errors="replace") as fh:
            for line in itertools.islice(fh, _MAX_CPUINFO_LINES):
                key, sep, value = line.partition(":")
                if sep and key.strip() == "flags":
                    return _x86_isa(set(value.split()))
    except OSError:
        return None
    return None


def _x86_isa(flags: set[str]) -> str | None:
    if "avx512f" in flags:
        return "avx512f"
    if "avx2" in flags:
        return "avx2"
    if flags & {"sse4_1", "sse4_2"}:
        return "sse4"
    return None


def macos_cpu_isa(read: Callable[[str], str | None]) -> str | None:
    """``arm64-sme`` if ``hw.optional.arm.FEAT_SME`` is 1, ``arm64-neon`` if NEON is, else
    ``None``; ``read`` returns a sysctl value (``sysctl`` in production)."""
    if read("hw.optional.arm.FEAT_SME") == "1":
        return "arm64-sme"
    if read("hw.optional.neon") == "1" or read("hw.optional.AdvSIMD") == "1":
        return "arm64-neon"
    return None


def sysctl(key: str) -> str | None:
    """``sysctl -n key``, or ``None`` if it fails (an unknown key on this machine)."""
    try:
        done = subprocess.run(
            [_SYSCTL, "-n", key], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


@lru_cache(maxsize=8)
def distribution_version(name: str) -> str | None:
    """The installed version of distribution ``name`` (never importing it), else ``None``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_versions(challenge: Challenge | None = None) -> Versions:
    """What this machine runs episodes with: Topple, MuJoCo, numpy, ``challenge``'s obs and
    action schemas (balance-v1's when None), and the installed ONNX Runtime, onnx and
    ``cpu_isa`` (recorded for every run, a Python controller's included: per-episode ONNX
    results are bit-exact only per platform, ``cpu_isa`` and onnxruntime version, contract
    §17)."""
    return Versions(
        topple=__version__,
        mujoco=mujoco.__version__,
        numpy=np.__version__,
        obs_schema=OBS_SCHEMA_ID if challenge is None else challenge.config.obs_schema,
        action_spec=BALANCE_V1_ACTION.id if challenge is None else challenge.config.action_spec,
        onnxruntime=distribution_version("onnxruntime"),
        onnx=distribution_version("onnx"),
        cpu_isa=cpu_isa(),
    )


def build_results(
    protocol: EvalProtocol,
    episodes: list[EpisodeResult],
    controller: ControllerInfo,
    isolation: Isolation,
) -> EvalResults:
    """``topple.results/1`` for ``episodes`` of ``protocol``; ``complete`` when all of its N
    episodes are listed and none is ``missing`` (contract §15). ``controller`` carries the
    ``policy_sha256`` of an ONNX submission; the versions record this machine's ONNX Runtime,
    onnx and ``cpu_isa`` for every run (a Python controller's included)."""
    private = protocol.seed_set == "private"
    missing = any(e.error is not None and e.error.code == "missing" for e in episodes)
    return EvalResults(
        challenge=protocol.challenge.id,
        protocol=protocol.id,
        protocol_hash=protocol.protocol_hash(),
        seed_set=protocol.seed_set,
        versions=run_versions(protocol.challenge),
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
