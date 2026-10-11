"""Trusted verification and re-scoring of uploaded trajectories (docs/contract.md §16).

The env is a pure function of (protocol, scenario, command sequence): physics never depends on
sensor noise or on controller internals. So a trusted stage verifies an untrusted ``traj_*.npz``
by replaying its commands open loop, without running the submission's code, and scores only the
replay. Nothing an untrusted party wrote is ever scored: the protocol, the episode index (hence
the seed and the scenario) and the verifier's own build are the trusted inputs; the upload only
supplies the commands to replay and the records the replay must reproduce bit for bit.
Bit-exactness holds on the same platform and MuJoCo version.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from topple.envs.balance import BalanceEnv
from topple.evaluate import (
    EPILOGUE_S,
    SIM_UNSTABLE_DETAIL,
    EvalProtocol,
    build_results,
    wants_epilogue,
)
from topple.results import (
    MAX_EPISODES,
    RUN_ERROR_CODES,
    VERIFY_ERROR_CODES,
    ControllerInfo,
    EpisodeResult,
    ErrorInfo,
    EvalResults,
    Isolation,
)
from topple.rng import validate_seed
from topple.robot import robot_hash
from topple.scoring import SURVIVED, episode_metrics, episode_score, episode_success
from topple.traj import Trajectory

VerifyCode = Literal["not_reproducible", "incomplete"]

INTEGRITY_ARRAYS = (
    "qpos",
    "qvel",
    "obs",
    "ctrl",
    "enc_counts",
    "gyro",
    "accel",
    "push_force",
    "epilogue_qpos",
)
"""Arrays an upload must reproduce bit for bit, in the order they are compared."""
ENV_META_FIELDS = (
    "termination",
    "max_steps",
    "control_dt",
    "timestep",
    "frame_skip",
    "joint_qpos",
    "wheel_sign",
    "obs_schema",
    "obs_features",
)
"""Env-produced metadata an upload must match exactly (``versions`` is informational)."""
MISSING_DETAIL = "no trajectory was uploaded for this episode"
_SHOW_CHARS = 40  # untrusted values quoted in a reason are cut to this many characters


@dataclass(frozen=True)
class EpisodeVerdict:
    """The trusted outcome of verifying one uploaded episode."""

    index: int
    ok: bool  # bound to (protocol, index) and reproduced bit-exactly
    code: VerifyCode | None
    reason: str  # human-readable; the first failing check
    replay: Trajectory | None  # trusted replay, as far as it ran (None if the upload failed
    # before replaying); its ``action`` is the upload's checked raw output, which a replay cannot
    # re-derive
    result: EpisodeResult  # scored only from ``replay`` and the env's own metadata


@dataclass(frozen=True)
class RunVerdict:
    """The trusted outcome of re-scoring a whole run (``verify_run``)."""

    results: EvalResults  # the trusted results; redacted (episodes null) for a private protocol
    failed: int  # episodes that did not verify (not_reproducible, incomplete or missing),
    # counted before redaction

    @property
    def ok(self) -> bool:
        """Every one of the protocol's episodes was uploaded and reproduced."""
        return self.failed == 0


@dataclass(frozen=True)
class _Target:
    """What the trusted side knows about the episode being verified."""

    protocol: EvalProtocol
    index: int
    seed: int
    bind_index: bool  # check meta.episode_index (verify_episode) or not (verify_with_seed)
    robot_hash: str

    @property
    def reported_seed(self) -> int | None:
        return self.seed if self.protocol.seed_set == "public" else None


@dataclass(frozen=True)
class _Replayed:
    trajectory: Trajectory
    error: ErrorInfo | None  # sim_unstable from the replay, the claimed error, or incomplete


class _RejectedError(Exception):
    def __init__(self, code: VerifyCode, reason: str, replay: Trajectory | None = None) -> None:
        super().__init__(reason)
        self.code: VerifyCode = code
        self.reason = reason
        self.replay = replay  # the replay as far as it ran, when it ran before the rejection


def _not_reproducible(reason: str) -> _RejectedError:
    return _RejectedError("not_reproducible", reason)


# ------------------------------------------------------------------ public API


def protocol_max_steps(protocol: EvalProtocol) -> int:
    """Episode length of ``protocol`` in control steps (contract §4): the cap for every upload."""
    cfg = protocol.challenge.config
    return round(cfg.episode_seconds / cfg.control_dt)


def verify_episode(upload: Trajectory, protocol: EvalProtocol, index: int) -> EpisodeVerdict:
    """Verify ``upload`` as episode ``index`` of ``protocol`` (seed ``protocol.seeds[index]``).

    Never raises for anything in the upload; ``ValueError`` only for an index outside the
    protocol (a caller bug).
    """
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(protocol.seeds)
    ):
        raise ValueError(
            f"episode index must be an int in [0, {len(protocol.seeds)}), got {index!r}"
        )
    with _replay_env(protocol) as env:
        return _verify(env, upload, _target(protocol, index, protocol.seeds[index], bind=True))


def verify_with_seed(
    upload: Trajectory, protocol: EvalProtocol, seed: int, *, index: int = 0
) -> EpisodeVerdict:
    """Verify ``upload`` against an explicit ``seed`` of ``protocol``'s challenge and preset.

    For local and debug runs whose seed is not in the protocol's list (``topple run --seed``).
    The scenario is still re-derived from ``seed``; ``index`` only labels the verdict and the
    upload's ``meta.episode_index`` is not checked. The verdict is not bound to the protocol.
    ``ValueError`` (a caller bug) for a seed outside ``[0, 2**53)`` or an index a results file
    cannot hold (outside ``[0, MAX_EPISODES)``, contract §15).
    """
    validate_seed(seed)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_EPISODES:
        raise ValueError(f"episode index must be an int in [0, {MAX_EPISODES}), got {index!r}")
    with _replay_env(protocol) as env:
        return _verify(env, upload, _target(protocol, index, seed, bind=False))


def rescore(
    protocol: EvalProtocol,
    uploads: Mapping[int, Trajectory],
    controller: ControllerInfo,
    isolation: Isolation,
) -> EvalResults:
    """Trusted results for ``protocol`` from uploaded trajectories, keyed by episode index.

    Exactly ``N = len(protocol.seeds)`` indices are expected: an index outside ``[0, N)``
    raises ``ValueError``; a missing one scores 0 with error ``missing``. The aggregate divides
    by N, and ``complete`` is true only when no episode is missing. ``controller`` and
    ``isolation`` come from the trusted caller; nothing is copied from the uploads.

    Each index is looked up once, in order, and no upload is kept once it is verified, so a
    mapping that loads on lookup holds one upload in memory at a time. Whether the run verified
    is ``verify_run``'s to say: a private protocol's results list no episodes.
    """
    return verify_run(protocol, uploads, controller, isolation).results


def verify_run(
    protocol: EvalProtocol,
    uploads: Mapping[int, Trajectory],
    controller: ControllerInfo,
    isolation: Isolation,
) -> RunVerdict:
    """``rescore``'s results, plus how many episodes did not verify (counted from the verdicts
    before a private protocol's results are redacted)."""
    n = len(protocol.seeds)
    bad = [k for k in uploads if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < n]
    if bad:
        shown = ", ".join(_show(k) for k in bad[:5])
        raise ValueError(f"episode index values outside [0, {n}): {shown}")
    episodes: list[EpisodeResult] = []
    with _replay_env(protocol) as env:
        for index, seed in enumerate(protocol.seeds):
            target = _target(protocol, index, seed, bind=True)
            upload = uploads.get(index)
            episodes.append(
                _missing(target) if upload is None else _verify(env, upload, target).result
            )
    failed = sum(e.error is not None and e.error.code in VERIFY_ERROR_CODES for e in episodes)
    return RunVerdict(build_results(protocol, episodes, controller, isolation), failed)


# ------------------------------------------------------------------ the checks, in order


def _verify(env: BalanceEnv, upload: Trajectory, target: _Target) -> EpisodeVerdict:
    replay: Trajectory | None = None
    try:
        claimed = _check_meta(upload, target)
        _check_scenario(env, upload.meta, target.seed)
        _check_commands(upload, env.max_steps)
        replayed = _replay(env, upload, claimed, target.index)
        replay = replayed.trajectory
        if replayed.error is not None and replayed.error.code == "incomplete":
            raise _RejectedError("incomplete", replayed.error.detail)
        _check_integrity(upload, replay)
    except _RejectedError as rejection:
        ran = rejection.replay if rejection.replay is not None else replay
        return _rejected(env, target, upload.steps, rejection, ran)
    return _accepted(env, target, replayed)


def _check_meta(upload: Trajectory, target: _Target) -> ErrorInfo | None:
    """1. Meta binding; also parses the upload's own error report (returned if well formed)."""
    meta = upload.meta
    protocol = target.protocol
    for key, expected in (
        ("challenge", protocol.challenge.id),
        ("preset", protocol.preset),
        ("robot_hash", target.robot_hash),
    ):
        value = meta.get(key)
        if not isinstance(value, str) or value != expected:
            raise _not_reproducible(f"meta.{key} is {_show(value)}, expected {expected!r}")
    if target.bind_index:
        value = meta.get("episode_index")
        if type(value) is not int or value != target.index:
            raise _not_reproducible(
                f"meta.episode_index is {_show(value)}, expected {target.index}"
            )
    return _claimed_error(meta.get("error"), upload.steps)


def _claimed_error(raw: object, steps: int) -> ErrorInfo | None:
    """The upload's error report, sanitized; a malformed report is a forgery."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"code", "step", "detail"}:
        raise _not_reproducible("meta.error must be null or {code, step, detail}")
    code, step, detail = raw["code"], raw["step"], raw["detail"]
    if not isinstance(code, str) or code not in RUN_ERROR_CODES:
        raise _not_reproducible(f"meta.error.code {_show(code)} is not a run error code")
    if type(step) is not int or step != steps:
        raise _not_reproducible(
            f"meta.error.step is {_show(step)}, but the trajectory stops at step {steps}"
        )
    if not isinstance(detail, str):
        raise _not_reproducible("meta.error.detail must be a string")
    return ErrorInfo.model_validate({"code": code, "step": steps, "detail": detail})


def _check_scenario(env: BalanceEnv, meta: Mapping[str, Any], seed: int) -> None:
    """2. Reset the trusted env on the trusted seed; the upload must carry that very scenario."""
    env.reset(seed=seed)
    if _canonical(meta.get("scenario")) != _canonical(env.scenario.to_dict()):
        raise _not_reproducible("meta.scenario differs from the scenario of the protocol seed")


def _check_commands(upload: Trajectory, max_steps: int) -> None:
    """3. The commands are replayable: finite, in [-1, 1], and the clipped raw actions."""
    if upload.steps > max_steps:
        raise _not_reproducible(
            f"the trajectory has {upload.steps} steps, more than the protocol's {max_steps}"
        )
    ctrl, action = upload.ctrl, upload.action
    with np.errstate(invalid="ignore"):
        row = _first_row(~np.isfinite(ctrl).all(axis=1) | (np.abs(ctrl) > 1.0).any(axis=1))
    if row is not None:
        raise _not_reproducible(f"ctrl row {row} is not a finite command in [-1, 1]")
    row = _first_row(~np.isfinite(action).all(axis=1))
    if row is not None:
        raise _not_reproducible(f"action row {row} is not finite")
    row = _first_row((np.clip(action, -1.0, 1.0) != ctrl).any(axis=1))
    if row is not None:
        raise _not_reproducible(f"ctrl row {row} is not the clipped action")


def _replay(
    env: BalanceEnv, upload: Trajectory, claimed: ErrorInfo | None, index: int
) -> _Replayed:
    """4. Step the trusted env with the uploaded commands, then decide the episode's error."""
    steps = upload.steps
    done = False
    for k in range(steps):
        if done:
            raise _ended_early(env, upload, k, index)
        _, _, terminated, truncated, _ = env.step(upload.ctrl[k])
        done = terminated or truncated
    reason = env.termination_reason
    impossible = _impossible_claim(reason, done, steps, claimed)
    if impossible is not None:
        raise _RejectedError(
            "not_reproducible", impossible, _bare_replay(env, upload, steps, index)
        )
    error = _replay_error(reason, done, steps, env.max_steps, claimed)
    if wants_epilogue(reason or SURVIVED, has_error=error is not None):
        env.simulate_epilogue(EPILOGUE_S)
    meta = {"episode_index": index, "error": error.model_dump() if error else None}
    # The env records the commands it was stepped with as ``action``; the raw controller output
    # cannot be re-derived open loop, so the replay keeps the upload's (checked in step 3).
    return _Replayed(replace(env.trajectory(meta), action=upload.action), error)


def _bare_replay(env: BalanceEnv, upload: Trajectory, steps: int, index: int) -> Trajectory:
    """The replay of the upload's first ``steps`` rows, without an error or an epilogue, for a
    rejection's metrics."""
    return replace(
        env.trajectory({"episode_index": index, "error": None}), action=upload.action[:steps]
    )


def _ended_early(env: BalanceEnv, upload: Trajectory, k: int, index: int) -> _RejectedError:
    """The env terminated or truncated after ``k`` of the upload's rows: the upload does not
    reproduce, but the replay ran, so the rejection carries it (its first ``k`` steps, no error
    and no epilogue) for the verdict's metrics."""
    reason = f"the replay ended after {k} steps, the trajectory has {upload.steps}"
    return _RejectedError("not_reproducible", reason, _bare_replay(env, upload, k, index))


def _impossible_claim(
    reason: str | None, done: bool, steps: int, claimed: ErrorInfo | None
) -> str | None:
    """Why the upload's error report cannot come from an honest run, if it cannot.

    The replay's own blow-up overrides any report. Otherwise: a controller is never called again
    once the env terminates or truncates, so no honest run records an error at that step, and
    only the replay itself assigns ``sim_unstable``. Either claim would put a code that nothing
    reported (a runner's, or the physics') into the trusted results.
    """
    if claimed is None or reason == "sim_unstable":
        return None
    if done:
        return (
            f"meta.error claims {claimed.code} at step {steps}, where the replay ended "
            f"({reason or SURVIVED}) without an error"
        )
    if claimed.code == "sim_unstable":
        return "meta.error claims sim_unstable, but the replay stayed stable"
    return None


def _replay_error(
    reason: str | None, done: bool, steps: int, max_steps: int, claimed: ErrorInfo | None
) -> ErrorInfo | None:
    """The trusted error: physics blow-up first, then the upload's own report, else whether the
    commands stopped early without any explanation (``incomplete``)."""
    if reason == "sim_unstable":
        return ErrorInfo(code="sim_unstable", step=steps, detail=SIM_UNSTABLE_DETAIL)
    if claimed is not None:
        return claimed
    if not done:
        return ErrorInfo(
            code="incomplete",
            step=steps,
            detail=(
                f"the trajectory stops after {steps} of {max_steps} steps without terminating "
                "and without an error"
            ),
        )
    return None


def _check_integrity(upload: Trajectory, replay: Trajectory) -> None:
    """5. Every recorded array bit-equal to the replay's, then the env-produced metadata."""
    for name in INTEGRITY_ARRAYS:
        mine, trusted = getattr(upload, name), getattr(replay, name)
        if mine.shape != trusted.shape:
            raise _not_reproducible(
                f"{name} has shape {mine.shape}, the replay has {trusted.shape}"
            )
        row = _first_differing_row(mine, trusted)
        if row is not None:
            raise _not_reproducible(f"{name} differs from the replay at row {row}")
    for field in ENV_META_FIELDS:
        if _canonical(upload.meta.get(field)) != _canonical(replay.meta[field]):
            raise _not_reproducible(f"meta.{field} differs from the replay")


# ------------------------------------------------------------------ verdicts


def _accepted(env: BalanceEnv, target: _Target, replayed: _Replayed) -> EpisodeVerdict:
    """6. Score the replay exactly as ``evaluate.run_episode`` scores a live episode."""
    replay, error = replayed.trajectory, replayed.error
    cfg = env.challenge.config.score
    metrics = episode_metrics(replay, env.params.motor)
    score = 0.0 if error else episode_score(metrics, cfg)
    result = EpisodeResult(
        index=target.index,
        seed=target.reported_seed,
        success=error is None and episode_success(metrics, cfg),
        score=min(max(score, 0.0), 100.0),
        metrics=metrics.as_dict(),
        error=error,
    )
    reason = "reproduced bit-exactly" + (f"; the episode ended with {error.code}" if error else "")
    return EpisodeVerdict(target.index, True, None, reason, replay, result)


def _rejected(
    env: BalanceEnv,
    target: _Target,
    steps: int,
    rejection: _RejectedError,
    replay: Trajectory | None,
) -> EpisodeVerdict:
    metrics = episode_metrics(replay, env.params.motor).as_dict() if replay is not None else None
    result = EpisodeResult(
        index=target.index,
        seed=target.reported_seed,
        success=False,
        score=0.0,
        metrics=metrics,
        error=ErrorInfo(code=rejection.code, step=steps, detail=rejection.reason),
    )
    return EpisodeVerdict(target.index, False, rejection.code, rejection.reason, replay, result)


def _missing(target: _Target) -> EpisodeResult:
    return EpisodeResult(
        index=target.index,
        seed=target.reported_seed,
        success=False,
        score=0.0,
        metrics=None,
        error=ErrorInfo(code="missing", step=0, detail=MISSING_DETAIL),
    )


# ------------------------------------------------------------------ helpers


def _target(protocol: EvalProtocol, index: int, seed: int, *, bind: bool) -> _Target:
    challenge = protocol.challenge
    return _Target(protocol, index, seed, bind, robot_hash(challenge.robot_xml, challenge.params))


@contextmanager
def _replay_env(protocol: EvalProtocol) -> Iterator[BalanceEnv]:
    """The trusted env: the protocol's own challenge object and preset, recording."""
    env = BalanceEnv(protocol.preset, challenge=protocol.challenge, record=True)
    try:
        yield env
    finally:
        env.close()


def _canonical(value: object) -> str | None:
    """Canonical JSON for exact comparison (``1`` != ``1.0`` != ``true``); None if unencodable."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError):
        return None


def _first_row(mask: npt.NDArray[np.bool_]) -> int | None:
    rows = np.flatnonzero(mask)
    return int(rows[0]) if rows.size else None


def _first_differing_row(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> int | None:
    """First row whose bytes differ (bitwise, so ``-0.0`` != ``0.0`` and NaN payloads count)."""
    if a.shape[0] == 0:
        return None
    rows_a = a.reshape(a.shape[0], -1).view(np.uint8)
    rows_b = b.reshape(b.shape[0], -1).view(np.uint8)
    return _first_row((rows_a != rows_b).any(axis=1))


def _show(value: object) -> str:
    """A short repr of an untrusted value for an error message."""
    try:
        text = repr(value)
    except Exception:  # an exotic in-process object; uploads only hold JSON values
        return "<unprintable>"
    return text if len(text) <= _SHOW_CHARS else text[: _SHOW_CHARS - 3] + "..."
