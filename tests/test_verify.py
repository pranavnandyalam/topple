"""Trusted verification and re-scoring of honest runs (docs/contract.md §16)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from topple.envs.balance import BalanceEnv
from topple.evaluate import EPILOGUE_S, EvalProtocol, evaluate, run_episode
from topple.results import EpisodeResult
from topple.traj import Trajectory
from topple.verify import (
    EpisodeVerdict,
    protocol_max_steps,
    rescore,
    verify_episode,
    verify_with_seed,
)
from verify_support import (
    INFO,
    Honest,
    explodes,
    lqr,
    pid,
    public,
    run_honest,
    wide_random,
    with_meta,
    without_meta,
    zero,
)

FACTORIES = {"pid": pid, "lqr": lqr, "random": wide_random}


@pytest.fixture(scope="module")
def honest_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Honest]:
    """PID, LQR and a clipping random controller on the first 3 public standard seeds."""
    return {
        name: run_honest(public("standard", 3), factory, tmp_path_factory.mktemp(name))
        for name, factory in FACTORIES.items()
    }


@pytest.fixture(scope="module")
def fallen(tmp_path_factory: pytest.TempPathFactory) -> Honest:
    """Zero torque on the clean preset: every episode falls and has an epilogue."""
    return run_honest(public("clean", 2), zero, tmp_path_factory.mktemp("zero"))


def assert_reproduces(verdict: EpisodeVerdict, expected: EpisodeResult) -> None:
    assert verdict.ok, verdict.reason
    assert verdict.code is None
    assert verdict.result == expected


# ------------------------------------------------------------------ honest runs verify


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_honest_runs_verify_and_reproduce_evaluate(
    honest_runs: dict[str, Honest], name: str
) -> None:
    run = honest_runs[name]
    assert run.results.episodes is not None
    for index, upload in run.uploads.items():
        assert_reproduces(verify_episode(upload, run.protocol, index), run.results.episodes[index])


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_rescore_reproduces_evaluate_exactly(honest_runs: dict[str, Honest], name: str) -> None:
    run = honest_runs[name]
    again = rescore(run.protocol, run.uploads, INFO, "inprocess")
    assert again.episodes == run.results.episodes  # score, success, metrics and errors
    assert again.aggregate == run.results.aggregate
    assert again.complete
    assert again.protocol_hash == run.results.protocol_hash
    assert again.robot_hash == run.results.robot_hash
    assert again.model_dump(exclude={"versions", "platform"}) == run.results.model_dump(
        exclude={"versions", "platform"}
    )


def test_the_random_controller_really_clips(honest_runs: dict[str, Honest]) -> None:
    upload = honest_runs["random"].uploads[0]
    assert (np.abs(upload.action) > 1.0).any()
    assert not np.array_equal(upload.action, upload.ctrl)


def test_a_verdict_carries_the_trusted_replay(honest_runs: dict[str, Honest]) -> None:
    run = honest_runs["pid"]
    verdict = verify_episode(run.uploads[0], run.protocol, 0)
    assert verdict.index == 0
    assert verdict.replay is not None
    np.testing.assert_array_equal(verdict.replay.qpos, run.uploads[0].qpos)
    assert verdict.replay.meta["episode_index"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("factory", [pid, lqr], ids=["pid", "lqr"])
def test_rescore_reproduces_evaluate_on_all_public_standard_seeds(
    tmp_path: Path, factory: Any
) -> None:
    run = run_honest(EvalProtocol.from_id("balance-v1/standard"), factory, tmp_path)
    assert len(run.uploads) == 20
    again = rescore(run.protocol, run.uploads, INFO, "inprocess")
    assert again.episodes == run.results.episodes
    assert again.aggregate == run.results.aggregate
    assert again.complete


# ------------------------------------------------------------------ errors and epilogues


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (RuntimeError("boom \x00 \ud800"), "exception"),
        (SystemExit(3), "exception"),
        ((float("nan"), 0.0), "invalid_action"),
        ((0.1,), "invalid_action"),
    ],
)
def test_honest_controller_errors_verify_and_keep_the_error(
    tmp_path: Path, payload: Any, code: str
) -> None:
    run = run_honest(public("standard", 1), explodes(37, payload), tmp_path)
    assert run.results.episodes is not None
    expected = run.results.episodes[0]
    assert expected.error is not None
    assert (expected.error.code, expected.error.step) == (code, 37)
    verdict = verify_episode(run.uploads[0], run.protocol, 0)
    assert_reproduces(verdict, expected)
    assert verdict.result.score == 0.0
    assert verdict.replay is not None
    assert verdict.replay.epilogue_qpos.shape[0] == round(EPILOGUE_S / 0.01)


def test_factory_error_verifies_with_zero_steps(tmp_path: Path) -> None:
    def broken(_spec: Any) -> Any:
        raise ImportError("no numpy for you")

    run = run_honest(public("clean", 1), broken, tmp_path)
    assert run.uploads[0].steps == 0
    assert run.results.episodes is not None
    assert_reproduces(verify_episode(run.uploads[0], run.protocol, 0), run.results.episodes[0])


def test_sim_unstable_replays_as_sim_unstable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    real_reset = BalanceEnv.reset

    def reset(self: BalanceEnv, **kwargs: Any) -> Any:
        calls["n"] = 0
        return real_reset(self, **kwargs)

    def blows_up(self: BalanceEnv) -> bool:
        calls["n"] += 1
        return calls["n"] > 37

    monkeypatch.setattr(BalanceEnv, "reset", reset)
    monkeypatch.setattr(BalanceEnv, "_unstable", blows_up)
    run = run_honest(public("clean", 1), pid, tmp_path)
    assert run.results.episodes is not None
    expected = run.results.episodes[0]
    assert expected.error is not None
    assert expected.error.code == "sim_unstable"
    assert_reproduces(verify_episode(run.uploads[0], run.protocol, 0), expected)
    # Stripping the error does not help: the replay itself goes unstable.
    stripped = with_meta(run.uploads[0], error=None)
    verdict = verify_episode(stripped, run.protocol, 0)
    assert verdict.result.error is not None
    assert verdict.result.error.code == "sim_unstable"
    assert verdict.result.score == 0.0


def test_survivors_have_no_epilogue_and_falls_do(
    honest_runs: dict[str, Honest], fallen: Honest
) -> None:
    survivor = honest_runs["lqr"].uploads[0]
    assert survivor.meta["termination"]["reason"] == "timeout"
    verdict = verify_episode(survivor, honest_runs["lqr"].protocol, 0)
    assert verdict.replay is not None
    assert verdict.replay.epilogue_qpos.shape[0] == 0
    fell = verify_episode(fallen.uploads[0], fallen.protocol, 0)
    assert fell.ok, fell.reason
    assert fell.replay is not None
    assert fell.replay.epilogue_qpos.shape[0] == round(EPILOGUE_S / 0.01)


def test_an_honest_error_report_only_lowers_the_score(honest_runs: dict[str, Honest]) -> None:
    """Claiming an error on a good episode can only cost its author: the episode scores 0."""
    run = honest_runs["lqr"]
    upload = run.uploads[0]
    claimed = with_meta(upload, error={"code": "timeout", "step": upload.steps, "detail": "my bad"})
    verdict = verify_episode(claimed, run.protocol, 0)
    assert verdict.result.score == 0.0
    assert not verdict.result.success


# ------------------------------------------------------------------ local runs (verify_with_seed)


def test_a_local_seed_verifies_with_its_own_seed(tmp_path: Path) -> None:
    protocol = replace(public("standard", 1), seeds=(7,))
    run = run_honest(protocol, pid, tmp_path)
    upload = run.uploads[0]
    assert run.results.episodes is not None
    verdict = verify_with_seed(upload, public("standard", 20), 7)
    assert_reproduces(verdict, run.results.episodes[0])
    bound = verify_episode(upload, public("standard", 20), 0)
    assert not bound.ok
    assert bound.code == "not_reproducible"
    assert "scenario" in bound.reason


def test_verify_with_seed_takes_the_label_index(honest_runs: dict[str, Honest]) -> None:
    run = honest_runs["pid"]
    verdict = verify_with_seed(
        with_meta(run.uploads[1], episode_index=5), run.protocol, run.protocol.seeds[1], index=5
    )
    assert verdict.ok, verdict.reason
    assert verdict.index == verdict.result.index == 5


@pytest.mark.parametrize("seed", [-1, 2**53, True, 1.0])
def test_verify_with_seed_rejects_bad_seeds(honest_runs: dict[str, Honest], seed: Any) -> None:
    run = honest_runs["pid"]
    with pytest.raises(ValueError, match="seed"):
        verify_with_seed(run.uploads[0], run.protocol, seed)


@pytest.mark.parametrize("index", [-1, 3, True])
def test_verify_episode_rejects_indices_outside_the_protocol(
    honest_runs: dict[str, Honest], index: Any
) -> None:
    run = honest_runs["pid"]
    with pytest.raises(ValueError, match="index"):
        verify_episode(run.uploads[0], run.protocol, index)


# ------------------------------------------------------------------ rescore API


@pytest.mark.parametrize("keys", [[0, 1, 2, 3], [-1], [True], ["0"]])
def test_rescore_rejects_extra_and_out_of_range_indices(
    honest_runs: dict[str, Honest], keys: list[Any]
) -> None:
    run = honest_runs["pid"]
    uploads: dict[Any, Trajectory] = dict.fromkeys(keys, run.uploads[0])
    with pytest.raises(ValueError, match="index"):
        rescore(run.protocol, uploads, INFO, "inprocess")


@pytest.mark.parametrize(
    "present", [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)], ids=str
)
def test_rescore_counts_missing_episodes_as_zero(
    honest_runs: dict[str, Honest], present: tuple[int, ...]
) -> None:
    run = honest_runs["lqr"]
    assert run.results.episodes is not None
    uploads = {i: run.uploads[i] for i in present}
    again = rescore(run.protocol, uploads, INFO, "container")
    assert again.isolation == "container"
    assert again.complete == (len(present) == 3)
    assert again.episodes is not None
    assert len(again.episodes) == 3
    for i, e in enumerate(again.episodes):
        if i in present:
            assert e == run.results.episodes[i]
        else:
            assert e.error is not None
            assert e.error.code == "missing"
            assert (e.score, e.success, e.metrics, e.seed) == (
                0.0,
                False,
                None,
                run.protocol.seeds[i],
            )
    honest_scores = [run.results.episodes[i].score for i in present]
    assert again.aggregate.score == pytest.approx(sum(honest_scores) / 3, abs=1e-12)
    successes = sum(run.results.episodes[i].success for i in present)
    assert again.aggregate.success_rate == successes / 3
    assert again.aggregate.errors == 3 - len(present)


def test_rescore_of_a_private_protocol_is_redacted(honest_runs: dict[str, Honest]) -> None:
    run = honest_runs["pid"]
    private = replace(run.protocol, seed_set="private", salt=99)
    again = rescore(private, run.uploads, INFO, "container")
    assert again.episodes is None
    assert again.seed_set == "private"


# ------------------------------------------------------------------ robustness


JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=8),
    lambda inner: (
        st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=5), inner, max_size=3)
    ),
    max_leaves=8,
)
META_KEYS = (
    "challenge",
    "preset",
    "robot_hash",
    "episode_index",
    "scenario",
    "error",
    "termination",
)


@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(key=st.sampled_from(META_KEYS), value=JSON)
def test_arbitrary_meta_values_never_crash_verification(
    honest_runs: dict[str, Honest], key: str, value: Any
) -> None:
    run = honest_runs["random"]
    upload = run.uploads[0]
    edited = with_meta(upload, **{key: value})
    verdict = verify_episode(edited, run.protocol, 0)
    if verdict.ok:  # only an edit that changes nothing (or only lowers the score) can pass
        assert verdict.result.score <= run.results.episodes[0].score  # type: ignore[index]
    else:
        assert verdict.result.score == 0.0
        assert verdict.code in ("not_reproducible", "incomplete")


@pytest.mark.parametrize("key", META_KEYS)
def test_missing_meta_keys_fail_verification(honest_runs: dict[str, Honest], key: str) -> None:
    run = honest_runs["pid"]
    verdict = verify_episode(without_meta(run.uploads[0], key), run.protocol, 0)
    if key == "error":  # absent and null mean the same: no error
        assert verdict.ok, verdict.reason
    else:
        assert not verdict.ok
        assert verdict.result.score == 0.0


def test_protocol_max_steps() -> None:
    assert protocol_max_steps(public("standard", 1)) == 1000


def test_run_episode_is_what_verification_replays(honest_runs: dict[str, Honest]) -> None:
    """The replay env is the evaluation env: a fresh env reproduces a reused one."""
    run = honest_runs["pid"]
    env = BalanceEnv("standard", record=True)
    fresh = run_episode(env, pid, run.protocol.seeds[2], 2).trajectory
    np.testing.assert_array_equal(fresh.qpos, run.uploads[2].qpos)
    results = evaluate(run.protocol, pid, INFO)
    assert results.episodes == run.results.episodes


def test_verify_with_seed_rejects_bad_label_indices(honest_runs: dict[str, Honest]) -> None:
    run = honest_runs["pid"]
    with pytest.raises(ValueError, match="index"):
        verify_with_seed(run.uploads[0], run.protocol, run.protocol.seeds[0], index=-1)


class Unprintable:
    def __repr__(self) -> str:
        raise RuntimeError("no repr")


@pytest.mark.parametrize("value", [object(), Unprintable(), {"a": {1, 2}}])
def test_in_process_meta_that_is_not_json_fails_cleanly(
    honest_runs: dict[str, Honest], value: object
) -> None:
    """Uploads come from JSON, but an in-process Trajectory may hold anything: still no crash."""
    run = honest_runs["pid"]
    for key in ("challenge", "scenario", "termination"):
        verdict = verify_episode(with_meta(run.uploads[0], **{key: value}), run.protocol, 0)
        assert not verdict.ok
        assert verdict.result.score == 0.0
