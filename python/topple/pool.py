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
before anything is sent, and ``run_episodes`` checks again in the worker.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from topple.envs.balance import BalanceEnv
from topple.evaluate import ControllerFactory, EvalProtocol, run_episode
from topple.ids import validate_challenge_id
from topple.loader import load_submission
from topple.policy.manifest import HEX64
from topple.policy.validate import MAX_ONNX_BYTES

if TYPE_CHECKING:
    from topple.challenge import Challenge
    from topple.results import EpisodeResult

POOL_MAX_WORKERS: Final = 4
START_METHOD: Final = "spawn"  # a fresh interpreter: no inherited sessions, threads or locks


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
    loaded = load_submission(sub, challenge=challenge)
    policy = loaded.policy  # set exactly when ``loaded.kind == "onnx"`` (Submission checks)
    if policy is not None:
        return PolicyRef("onnx", policy.validated.canonical, policy.policy_sha256, challenge.id)
    return PolicyRef("python", str(sub.resolve()), loaded.sha256, challenge.id)


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
    pool: Executor, ref: PolicyRef, protocol: EvalProtocol, indices: Sequence[int]
) -> Future[list[EpisodeResult]]:
    """Submit ``run_episodes(ref, protocol, indices)`` to ``pool``, after checking its
    arguments here in the parent: submitting pickles them to a worker, so a private protocol
    (its salt) must be refused before ``submit``, not by the worker that already received it."""
    _check_task(ref, protocol, indices)
    return pool.submit(run_episodes, ref, protocol, tuple(indices))


def _check_task(ref: PolicyRef, protocol: EvalProtocol, indices: Sequence[int]) -> None:
    if protocol.seed_set != "public":
        raise ValueError("pool workers run public protocols only: a private salt stays home")
    if protocol.challenge.id != ref.challenge_id:
        raise ValueError(
            f"the policy was validated for {ref.challenge_id}, not {protocol.challenge.id}"
        )
    n = len(protocol.seeds)
    if not all(type(i) is int and 0 <= i < n for i in indices):
        raise ValueError(f"episode indices must be ints in [0, {n})")


def run_episodes(
    ref: PolicyRef, protocol: EvalProtocol, indices: Sequence[int]
) -> list[EpisodeResult]:
    """A pool task (submit it with ``submit_episodes``): episodes ``indices`` of the public
    ``protocol`` for ``ref``, in order, on one env. The arguments are checked again here, and
    the controller is rebuilt (its sha256 re-checked) before the first episode."""
    _check_task(ref, protocol, indices)
    factory = worker_factory(ref)
    env = BalanceEnv(protocol.preset, challenge=protocol.challenge, record=True)
    try:
        return [
            run_episode(env, factory, protocol.seeds[i], i, salt=protocol.salt).result
            for i in indices
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
