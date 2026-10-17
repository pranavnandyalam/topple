"""Evaluation pools and the policy references they carry (docs/design/phase-2.md §1, §3.5).

Selection, compare, shove and report evaluations run episodes in a spawn ``ProcessPoolExecutor``
of at most ``min(workers, 4, os.cpu_count())`` processes (``eval_pool``), created once and shut
down in ``finally`` (or a ``with`` block) by its caller.

What crosses the process boundary is a ``PolicyRef``, never a session, a ``LoadedPolicy`` or a
factory: the parent reads and validates a submission once and sends either the canonical ONNX
bytes with their ``policy_sha256`` or a Python submission's path with its submission sha256. A
worker rebuilds the controller (``worker_factory``) and re-checks the sha256 before its first
episode. ONNX bytes travel with the reference, so a policy changed on disk after the parent's
load is never what a worker scores. A Python submission is read once by the worker and that
read's hash is compared before its bytes are compiled, so a changed entry file (or any changed
file of a directory) makes the worker raise before any of it runs. What a Python submission
reads itself after loading (a sibling module it imports, an asset) is an ordinary read of
whatever is on disk then: in-process Python cannot pin it, and Phase 3's runner isolates it.
Results come back by pickle, which is acceptable locally only (a local eval is not a security
boundary).

Workers receive public protocols only: submitting a task pickles its protocol, salt included
(only its ``repr`` is redacted), so ``submit_episodes`` refuses a private one in the parent,
before anything is sent, and ``run_episodes`` checks again in the worker. A task's indices (and
pushes) are read once, bounded, into a tuple before they are checked (decision log B16): a
one-shot iterator checked first and sent afterwards would run nothing, and an endless one is
refused after ``MAX_TASK_EPISODES + 1`` items instead of being read until memory runs out.

A task may replace its episodes' pushes (``pushes``): the trusted ``reset(options={"pushes":
...})`` hook (contract §9) that robustness benches such as ``topple shove`` use; ``None`` keeps
each seed's own scenario.

A selection eval's task (``submit_episodes_with_rows``) also returns, per episode, up to
``ONPOLICY_ROWS`` of the observations the policy saw (docs/design/phase-2.md §2.4 parity rows
(a)): ``traj.obs`` at ``np.linspace(0, T - 1, 5).round()`` for an episode of T >= 1 steps, none
for T = 0. The trajectory itself never leaves the worker.
"""

from __future__ import annotations

import itertools
import math
import multiprocessing
import os
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np
import numpy.typing as npt

from topple.disturbances import Push, check_push_schedule
from topple.evaluate import ControllerFactory, EpisodeRun, EvalProtocol, run_episode
from topple.ids import validate_challenge_id
from topple.kinds import make_env
from topple.loader import Submission, load_submission
from topple.policy.manifest import HEX64
from topple.policy.validate import MAX_ONNX_BYTES
from topple.results import MAX_EPISODES

if TYPE_CHECKING:
    from topple.challenge import Challenge
    from topple.results import EpisodeResult

POOL_MAX_WORKERS: Final = 4
START_METHOD: Final = "spawn"  # a fresh interpreter: no inherited sessions, threads or locks
MAX_TASK_EPISODES: Final = MAX_EPISODES  # a results file lists at most this many (contract §15)
MAX_TASK_PUSHES: Final = 16  # an episode of 1,000 steps holds a handful; a bench uses one
ONPOLICY_ROWS: Final = 5
"""On-policy parity rows per episode of at least one step (§2.4 (a))."""


@dataclass(frozen=True)
class PolicyRef:
    """What a worker needs to rebuild a controller: ``kind`` ``"onnx"`` with the canonical
    bytes (``payload``, at most 1 MiB) and their ``policy_sha256``, or ``"python"`` with the
    submission's resolved path and its submission sha256; plus the challenge id the policy was
    validated for. Small and picklable by construction."""

    kind: Literal["onnx", "python"]
    payload: bytes | str = field(repr=False)
    sha256: str
    challenge_id: str

    def __post_init__(self) -> None:
        if self.kind not in ("onnx", "python"):
            raise ValueError(f"a PolicyRef is 'onnx' or 'python', not {self.kind!r}")
        if self.kind == "onnx" and type(self.payload) is not bytes:
            raise TypeError("an ONNX PolicyRef carries the policy's bytes")
        if self.kind == "python" and type(self.payload) is not str:
            raise TypeError("a Python PolicyRef carries the submission's path")
        if isinstance(self.payload, bytes) and len(self.payload) > MAX_ONNX_BYTES:
            raise ValueError(f"an ONNX policy has at most {MAX_ONNX_BYTES} bytes")
        if type(self.sha256) is not str or not HEX64.fullmatch(self.sha256):
            raise ValueError("a PolicyRef's sha256 is 64 lowercase hex digits")
        validate_challenge_id(self.challenge_id)


def policy_ref(sub: Path, *, challenge: Challenge) -> PolicyRef:
    """The parent's side: load ``sub`` once (``load_submission``: a policy is fully validated
    and smoke-tested for ``challenge``, a Python submission imported) and reference it."""
    return submission_ref(sub, challenge=challenge)[1]


def submission_ref(sub: Path, *, challenge: Challenge) -> tuple[Submission, PolicyRef]:
    """``policy_ref`` with the load it came from: a report records the submission's id, its
    submission sha256 and its ``policy_sha256`` from the same single read the workers' sha256
    checks are bound to (``topple compare``, ``topple shove``)."""
    loaded = load_submission(sub, challenge=challenge)
    policy = loaded.policy  # set exactly when ``loaded.kind == "onnx"`` (Submission checks)
    if policy is not None:
        ref = PolicyRef("onnx", policy.validated.canonical, policy.policy_sha256, challenge.id)
        return loaded, ref
    return loaded, PolicyRef("python", str(sub.resolve()), loaded.sha256, challenge.id)


def worker_factory(ref: PolicyRef) -> ControllerFactory:
    """The worker's side: the controller factory of ``ref``, after checking that what is
    loaded is what the parent loaded (``LoadError`` otherwise, before any episode).

    ONNX: ``policy_from_bytes`` with ``expected_sha256`` (validation and smoke test again, in
    this process). Python: ``load_submission`` with ``expected_sha256``, which compares the hash
    of its single read before compiling it, so changed bytes are refused, never run.
    """
    from topple.challenge import load_challenge

    challenge = load_challenge(ref.challenge_id)
    if ref.kind == "onnx":
        from topple.controllers.onnx_policy import policy_factory
        from topple.policy.runtime import policy_from_bytes

        data = cast(bytes, ref.payload)  # PolicyRef ties the payload's type to its kind
        loaded = policy_from_bytes(data, challenge=challenge, expected_sha256=ref.sha256)
        return policy_factory(loaded)
    path = Path(cast(str, ref.payload))
    return load_submission(path, challenge=challenge, expected_sha256=ref.sha256).factory


def submit_episodes(
    pool: Executor,
    ref: PolicyRef,
    protocol: EvalProtocol,
    indices: Iterable[int],
    *,
    pushes: Iterable[Push] | None = None,
) -> Future[list[EpisodeResult]]:
    """Submit ``run_episodes(ref, protocol, indices, pushes)`` to ``pool``, after checking its
    arguments here in the parent: submitting pickles them to a worker, so a private protocol
    (its salt) must be refused before ``submit``, not by the worker that already received it.
    ``indices`` and ``pushes`` are read once into tuples (B16), and those tuples are sent."""
    checked, injected = _check_task(ref, protocol, indices, pushes)
    return pool.submit(run_episodes, ref, protocol, checked, injected)


def _check_task(
    ref: PolicyRef,
    protocol: EvalProtocol,
    indices: Iterable[int],
    pushes: Iterable[Push] | None,
) -> tuple[tuple[int, ...], tuple[Push, ...] | None]:
    """The task's indices and pushes, each read once (at most one item past its cap) into a
    tuple and checked; ``ValueError`` for anything a worker must not receive."""
    if protocol.seed_set != "public":
        raise ValueError("pool workers run public protocols only: a private salt stays home")
    if protocol.challenge.id != ref.challenge_id:
        raise ValueError(
            f"the policy was validated for {ref.challenge_id}, not {protocol.challenge.id}"
        )
    n = len(protocol.seeds)
    checked = tuple(itertools.islice(indices, MAX_TASK_EPISODES + 1))
    if len(checked) > MAX_TASK_EPISODES:
        raise ValueError(f"a task runs at most {MAX_TASK_EPISODES} episodes")
    if not all(type(i) is int and 0 <= i < n for i in checked):
        raise ValueError(f"episode indices must be ints in [0, {n})")
    return checked, None if pushes is None else _checked_pushes(pushes)


def _checked_pushes(pushes: Iterable[Push]) -> tuple[Push, ...]:
    """``pushes`` as a tuple of ``Push`` (exactly), each one that can fire (an int
    ``start_step >= 0``, an int ``n_substeps >= 1``) with a finite magnitude, direction and
    lateral ratio, sorted by start step with no repeats; the env checks the rest of the
    schedule against its clock at reset (contract §9)."""
    injected = tuple(itertools.islice(pushes, MAX_TASK_PUSHES + 1))
    if len(injected) > MAX_TASK_PUSHES:
        raise ValueError(f"a task injects at most {MAX_TASK_PUSHES} pushes")
    if not all(type(p) is Push for p in injected):
        raise ValueError("pushes must be topple.disturbances.Push objects")
    for p in injected:  # a trusted hook skips the bounds, never the arithmetic (contract §9)
        if not (_whole(p.start_step, 0) and _whole(p.n_substeps, 1)):
            raise ValueError(
                "injected pushes need an int start_step >= 0 and an int n_substeps >= 1"
            )
        if not all(map(_finite, (p.magnitude_n, p.forward_sign, p.lateral_ratio))):
            raise ValueError(
                "injected pushes must have a finite magnitude, direction and lateral ratio"
            )
    check_push_schedule(injected)
    return injected


def _whole(value: object, least: int) -> bool:
    return type(value) is int and value >= least


def _finite(value: object) -> bool:
    """A real number (not a bool) that is a finite float: an int too large for a float is not."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def run_episodes(
    ref: PolicyRef,
    protocol: EvalProtocol,
    indices: Iterable[int],
    pushes: Iterable[Push] | None = None,
) -> list[EpisodeResult]:
    """A pool task (submit it with ``submit_episodes``): episodes ``indices`` of the public
    ``protocol`` for ``ref``, in order, on one env, each with ``pushes`` in place of its
    scenario's when given. The arguments are checked again here, and the controller is rebuilt
    (its sha256 re-checked) before the first episode."""
    checked, injected = _check_task(ref, protocol, indices, pushes)
    return _run(ref, protocol, checked, injected, lambda run: run.result)


@dataclass(frozen=True)
class EpisodeRows:
    """One episode's result and its on-policy parity rows (``onpolicy_rows``)."""

    result: EpisodeResult
    obs_rows: npt.NDArray[np.float32]


def onpolicy_rows(obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """§2.4 (a): the rows ``np.linspace(0, T - 1, 5).round().astype(int)`` of an episode's
    ``obs`` (T x D float32, row t is what the policy saw at step t): ``ONPOLICY_ROWS`` rows for
    T >= 1, repeats allowed for T < 5; a (0, D) array for T = 0 (an episode that ended before
    its first action, e.g. ``invalid_action`` at step 0)."""
    if obs.ndim != 2 or obs.dtype != np.float32:
        raise ValueError("obs must be a float32 array of rows [T, D]")
    steps = obs.shape[0]
    if steps == 0:
        return np.empty((0, obs.shape[1]), dtype=np.float32)
    picked = np.linspace(0, steps - 1, ONPOLICY_ROWS).round().astype(int)
    return np.ascontiguousarray(obs[picked], dtype=np.float32)


def submit_episodes_with_rows(
    pool: Executor, ref: PolicyRef, protocol: EvalProtocol, indices: Iterable[int]
) -> Future[list[EpisodeRows]]:
    """``submit_episodes`` for a selection eval: the task (``run_episodes_with_rows``) also
    returns each episode's on-policy parity rows. Checked in the parent as ``submit_episodes``
    checks (public protocol, its challenge, indices read once)."""
    checked, _ = _check_task(ref, protocol, indices, None)
    return pool.submit(run_episodes_with_rows, ref, protocol, checked)


def run_episodes_with_rows(
    ref: PolicyRef, protocol: EvalProtocol, indices: Iterable[int]
) -> list[EpisodeRows]:
    """A pool task (``submit_episodes_with_rows``): ``run_episodes`` (each seed's own
    scenario) with each episode's ``onpolicy_rows`` of its trajectory, in seed order."""
    checked, _ = _check_task(ref, protocol, indices, None)
    return _run(
        ref,
        protocol,
        checked,
        None,
        lambda run: EpisodeRows(run.result, onpolicy_rows(run.trajectory.obs)),
    )


def _run[T](
    ref: PolicyRef,
    protocol: EvalProtocol,
    checked: tuple[int, ...],
    injected: tuple[Push, ...] | None,
    keep: Callable[[EpisodeRun], T],
) -> list[T]:
    """``keep`` of each of the checked episodes, in order, on one env; the controller is rebuilt
    (its sha256 re-checked) before the first episode."""
    factory = worker_factory(ref)
    env = make_env(protocol.challenge, protocol.preset, record=True)
    try:
        return [
            keep(
                run_episode(env, factory, protocol.seeds[i], i, salt=protocol.salt, pushes=injected)
            )
            for i in checked
        ]
    finally:
        env.close()


def pool_size(workers: int) -> int:
    """``min(workers, POOL_MAX_WORKERS, os.cpu_count())`` for a positive int ``workers``."""
    if type(workers) is not int or workers < 1:
        raise ValueError(f"workers must be a positive int, got {workers!r}")
    return min(workers, POOL_MAX_WORKERS, os.cpu_count() or 1)


def eval_pool(workers: int) -> ProcessPoolExecutor:
    """A spawn ``ProcessPoolExecutor`` of ``pool_size(workers)`` processes. The caller creates
    it once and shuts it down in ``finally`` (``with eval_pool(n) as pool: ...``)."""
    return ProcessPoolExecutor(
        max_workers=pool_size(workers), mp_context=multiprocessing.get_context(START_METHOD)
    )
