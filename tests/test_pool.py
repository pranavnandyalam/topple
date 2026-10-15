"""Evaluation pools hand policies over by reference (docs/design/phase-2.md §1, §3.5, §6
``test_pool``).

The parent reads and validates a submission once and sends a ``PolicyRef``: canonical ONNX
bytes with their sha256, or a Python submission's path with its submission sha256. A worker
rebuilds the controller from that and re-checks the sha256 before its first episode: ONNX
bytes travel with the reference, and a changed Python submission is refused before any of it
runs. Sessions and factories never cross a process boundary, a private protocol never reaches
a worker, and pools are spawn-based and clamped to ``min(workers, 4, cpus)``.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import pickle
import shutil
from concurrent.futures import Executor, Future
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import topple.pool as pool_module
from policy_support import first_party_manifest, sac_bytes, write_policy_dir
from topple.challenge import load_challenge
from topple.contract import Controller
from topple.envs.balance import BalanceEnv
from topple.evaluate import EvalProtocol, run_episode
from topple.loader import LoadError, load_submission, submission_sha256
from topple.pool import (
    POOL_MAX_WORKERS,
    PolicyRef,
    eval_pool,
    policy_ref,
    pool_size,
    run_episodes,
    submit_episodes,
    worker_factory,
)

BALANCE = load_challenge("balance-v1")
REPO = Path(__file__).resolve().parents[1]
PID = REPO / "submissions" / "pid_example.py"
SPEC = BalanceEnv("ideal").task_spec
CONTROLLER = """
class C:
    def reset(self, rng_seed): pass
    def act(self, obs): return (0.0, 0.0)

def make_controller(spec):
    return C()
"""


def ideal(n: int = 1) -> EvalProtocol:
    protocol = EvalProtocol.from_id("balance-v1/ideal")
    return replace(protocol, seeds=protocol.seeds[:n])


# --- PolicyRef --------------------------------------------------------------------------------


def test_policy_ref_for_onnx(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot")
    data = (root / "policy.onnx").read_bytes()
    ref = policy_ref(root, challenge=BALANCE)
    assert ref.kind == "onnx"
    assert ref.payload == data
    assert ref.sha256 == hashlib.sha256(data).hexdigest()
    assert ref.challenge_id == "balance-v1"
    assert "payload" not in repr(ref)  # a megabyte of bytes never reaches a log line


def test_policy_ref_for_python(tmp_path: Path) -> None:
    sub = tmp_path / "pid.py"
    shutil.copyfile(PID, sub)
    ref = policy_ref(sub, challenge=BALANCE)
    assert ref.kind == "python"
    assert ref.payload == str(sub.resolve())
    assert ref.sha256 == submission_sha256(sub)


def test_policy_ref_refuses_a_bad_submission(tmp_path: Path) -> None:
    m = first_party_manifest(sac_bytes())  # says clip(mean); the SAC graph is tanh(mean)
    root = write_policy_dir(tmp_path / "bot", sac_bytes(), manifest=m)
    with pytest.raises(LoadError, match="action_rule_mismatch"):
        policy_ref(root, challenge=BALANCE)


def test_policy_refs_pickle_small(tmp_path: Path) -> None:
    ref = policy_ref(write_policy_dir(tmp_path / "bot"), challenge=BALANCE)
    blob = pickle.dumps(ref)
    assert pickle.loads(blob) == ref
    assert len(blob) < len(ref.payload) + 512  # the bytes, a sha and an id: no session


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"kind": "torch"}, ValueError),
        ({"kind": "onnx", "payload": "path"}, TypeError),
        ({"kind": "python", "payload": b"bytes"}, TypeError),
        ({"sha256": "A" * 64}, ValueError),
        ({"sha256": "a" * 64 + "\n"}, ValueError),
        ({"challenge_id": "Balance"}, ValueError),
        ({"payload": b"x" * ((1 << 20) + 1)}, ValueError),
    ],
)
def test_policy_ref_fields_are_checked(fields: dict[str, object], error: type[Exception]) -> None:
    base: dict[str, object] = {
        "kind": "onnx",
        "payload": b"onnx",
        "sha256": "a" * 64,
        "challenge_id": "balance-v1",
    }
    with pytest.raises(error):
        PolicyRef(**(base | fields))  # type: ignore[arg-type]


# --- worker_factory ---------------------------------------------------------------------------


def test_worker_factory_rebuilds_an_onnx_policy(tmp_path: Path) -> None:
    ref = policy_ref(write_policy_dir(tmp_path / "bot"), challenge=BALANCE)
    controller = worker_factory(ref)(SPEC)
    assert isinstance(controller, Controller)


def test_a_worker_given_a_wrong_onnx_sha_raises(tmp_path: Path) -> None:
    ref = policy_ref(write_policy_dir(tmp_path / "bot"), challenge=BALANCE)
    forged = replace(ref, sha256=hashlib.sha256(sac_bytes()).hexdigest())
    with pytest.raises(LoadError, match="policy_sha256_mismatch"):
        worker_factory(forged)


def test_a_worker_given_a_wrong_python_sha_raises_before_running_it(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    sub = tmp_path / "bot.py"
    sub.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + CONTROLLER)
    ref = policy_ref(sub, challenge=BALANCE)
    marker.unlink()
    sub.write_text(sub.read_text() + "\n# changed after the parent loaded it\n")
    with pytest.raises(LoadError, match="changed since"):
        worker_factory(ref)
    assert not marker.exists()  # the changed code never ran


def test_a_python_file_swapped_while_the_worker_loads_it_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is swapped just before the worker reads it: the worker compares the hash of
    its single read with the reference before compiling those bytes, so the swapped code never
    runs (there is no separate pre-hash whose window a swap could hit)."""
    marker = tmp_path / "ran"
    sub = tmp_path / "pid.py"
    shutil.copyfile(PID, sub)
    ref = policy_ref(sub, challenge=BALANCE)
    real = load_submission

    def swapped(path: Path, **kwargs: object) -> object:
        sub.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + CONTROLLER)
        return real(path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pool_module, "load_submission", swapped)
    with pytest.raises(LoadError, match="changed since"):
        worker_factory(ref)
    assert not marker.exists()  # the swapped code never ran


def test_a_python_ref_whose_directory_became_a_policy_fails_the_sha_check(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bot"
    root.mkdir()
    shutil.copyfile(PID, root / "controller.py")
    ref = policy_ref(root, challenge=BALANCE)
    (root / "controller.py").unlink()
    write_policy_dir(root)
    with pytest.raises(LoadError, match="changed since"):
        worker_factory(ref)


# --- run_episodes -----------------------------------------------------------------------------


def test_run_episodes_refuses_a_private_protocol(tmp_path: Path) -> None:
    ref = policy_ref(write_policy_dir(tmp_path / "bot"), challenge=BALANCE)
    private = replace(ideal(), seed_set="private", salt=2**64 + 5)
    with pytest.raises(ValueError, match="public"):
        run_episodes(ref, private, [0])


class _RecordingPool(Executor):
    """An executor that records what would be pickled to a worker, and runs nothing."""

    def __init__(self) -> None:
        self.submitted: list[tuple[object, ...]] = []

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        self.submitted.append((fn, *args))
        future: Future[Any] = Future()
        future.set_result([])
        return future


def test_submit_episodes_refuses_a_private_protocol_before_anything_is_pickled(
    tmp_path: Path,
) -> None:
    """Submitting pickles the protocol to a worker, salt included (only its repr is redacted),
    so the parent refuses a private protocol before ``submit``, not the worker after it."""
    ref = policy_ref(write_policy_dir(tmp_path / "bot"), challenge=BALANCE)
    private = replace(ideal(), seed_set="private", salt=2**64 + 5)
    pool = _RecordingPool()
    with pytest.raises(ValueError, match="public"):
        submit_episodes(pool, ref, private, [0])
    with pytest.raises(ValueError, match="other-v1"):
        submit_episodes(pool, replace(ref, challenge_id="other-v1"), ideal(), [0])
    with pytest.raises(ValueError, match="indices"):
        submit_episodes(pool, ref, ideal(1), [1])
    assert pool.submitted == []
    submit_episodes(pool, ref, ideal(1), [0])
    assert pool.submitted == [(run_episodes, ref, ideal(1), (0,))]


def test_run_episodes_refuses_another_challenge(tmp_path: Path) -> None:
    ref = replace(
        policy_ref(write_policy_dir(tmp_path / "bot"), challenge=BALANCE), challenge_id="other-v1"
    )
    with pytest.raises(ValueError, match="other-v1"):
        run_episodes(ref, ideal(), [0])


@pytest.mark.parametrize("indices", [[-1], [1], [True]], ids=["negative", "outside", "bool"])
def test_run_episodes_refuses_indices_outside_the_protocol(
    tmp_path: Path, indices: list[int]
) -> None:
    ref = policy_ref(write_policy_dir(tmp_path / "bot"), challenge=BALANCE)
    with pytest.raises(ValueError, match="indices"):
        run_episodes(ref, ideal(1), indices)


def test_run_episodes_in_process_matches_run_episode(tmp_path: Path) -> None:
    sub = tmp_path / "pid.py"
    shutil.copyfile(PID, sub)
    ref = policy_ref(sub, challenge=BALANCE)
    protocol = ideal(2)
    (second,) = run_episodes(ref, protocol, [1])
    env = BalanceEnv("ideal", challenge=BALANCE, record=True)
    try:
        expected = run_episode(env, load_submission(sub).factory, protocol.seeds[1], 1).result
    finally:
        env.close()
    assert second == expected


def test_a_spawn_pool_scores_the_bytes_the_parent_loaded(tmp_path: Path) -> None:
    """Mutating the files after the parent's load changes nothing a worker scores (ONNX), and a
    Python submission changed on disk makes its worker raise instead of scoring other code."""
    root = write_policy_dir(tmp_path / "bot")
    original = (root / "policy.onnx").read_bytes()
    onnx_ref = policy_ref(root, challenge=BALANCE)
    sub = tmp_path / "pid.py"
    shutil.copyfile(PID, sub)
    py_ref = policy_ref(sub, challenge=BALANCE)
    expected = run_episodes(onnx_ref, ideal(), [0])

    write_policy_dir(
        root, sac_bytes(), manifest=first_party_manifest(sac_bytes(), action_rule="tanh(mean)")
    )
    sub.write_text(PID.read_text() + "\n# changed\n")

    with eval_pool(2) as pool:
        onnx_future = submit_episodes(pool, onnx_ref, ideal(), [0])
        py_future = submit_episodes(pool, py_ref, ideal(), [0])
        assert onnx_future.result(timeout=120) == expected
        with pytest.raises(LoadError, match="changed since"):
            py_future.result(timeout=120)
    assert onnx_ref.sha256 == hashlib.sha256(original).hexdigest()


# --- eval_pool --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("workers", "cpus", "size"),
    [(1, 8, 1), (2, 8, 2), (4, 8, 4), (16, 8, 4), (4, 2, 2), (3, None, 1)],
)
def test_pool_size_is_clamped(
    monkeypatch: pytest.MonkeyPatch, workers: int, cpus: int | None, size: int
) -> None:
    monkeypatch.setattr(pool_module.os, "cpu_count", lambda: cpus)
    assert pool_size(workers) == size
    assert POOL_MAX_WORKERS == 4


@pytest.mark.parametrize("workers", [0, -1, True, 2.0])
def test_pool_size_needs_a_positive_int(workers: object) -> None:
    with pytest.raises(ValueError, match="workers"):
        pool_size(workers)  # type: ignore[arg-type]


def test_eval_pool_is_spawn_based_and_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pool_module.os, "cpu_count", lambda: 64)
    pool = eval_pool(9)
    try:
        assert pool._max_workers == 4
        context = pool._mp_context
        assert isinstance(context, multiprocessing.context.SpawnContext)
    finally:
        pool.shutdown()
