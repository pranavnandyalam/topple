"""Trusted verification and re-scoring of honest runs (docs/contract.md §16)."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
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
    ENV_META_FIELDS,
    EpisodeVerdict,
    RunVerdict,
    protocol_max_steps,
    rescore,
    verify_episode,
    verify_run,
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


class RecordingUploads(Mapping[int, Trajectory]):
    """A mapping that records every lookup, as a lazily loading caller would see them."""

    def __init__(self, uploads: dict[int, Trajectory]) -> None:
        self.uploads = uploads
        self.lookups: list[int] = []

    def __getitem__(self, index: int) -> Trajectory:
        self.lookups.append(index)
        return self.uploads[index]

    def __iter__(self) -> Iterator[int]:
        return iter(self.uploads)

    def __len__(self) -> int:
        return len(self.uploads)


def test_rescore_looks_each_upload_up_once_in_order(honest_runs: dict[str, Honest]) -> None:
    """The order a lazily loading mapping relies on to hold one upload at a time (§16)."""
    run = honest_runs["pid"]
    uploads = RecordingUploads({2: run.uploads[2], 0: run.uploads[0]})
    again = rescore(run.protocol, uploads, INFO, "inprocess")
    assert uploads.lookups == [0, 1, 2]  # index 1 is looked up and found missing
    assert again.episodes is not None
    assert run.results.episodes is not None
    assert again.episodes[1].error is not None
    assert again.episodes[1].error.code == "missing"
    assert [again.episodes[i] for i in (0, 2)] == [run.results.episodes[i] for i in (0, 2)]


def test_the_random_controller_really_clips(honest_runs: dict[str, Honest]) -> None:
    upload = honest_runs["random"].uploads[0]
    assert (np.abs(upload.action) > 1.0).any()
    assert not np.array_equal(upload.action, upload.ctrl)


def test_the_replay_keeps_the_raw_actions(honest_runs: dict[str, Honest]) -> None:
    """``action`` is the controller's raw output (§14), which an open-loop replay cannot
    re-derive: the replay carries the upload's checked actions, not a copy of ``ctrl``."""
    run = honest_runs["random"]
    for index, upload in run.uploads.items():
        verdict = verify_episode(upload, run.protocol, index)
        assert verdict.ok, verdict.reason
        assert verdict.replay is not None
        np.testing.assert_array_equal(verdict.replay.action, upload.action)
        np.testing.assert_array_equal(verdict.replay.ctrl, upload.ctrl)
        assert (np.abs(verdict.replay.action) > 1.0).any()


@pytest.mark.parametrize("saturated", [1e300, 1.0], ids=["huge", "unit"])
def test_rewriting_saturated_actions_changes_no_score(
    honest_runs: dict[str, Honest], saturated: float
) -> None:
    """Only ``clip(action) == ctrl`` ties a saturated row's raw action down, so an upload may put
    any finite value of the right sign there. ``replay.action`` is that value, and nothing may be
    scored from it (§16): every result and the aggregate stay exactly the honest ones."""
    run = honest_runs["random"]
    assert run.results.episodes is not None
    forged: dict[int, Trajectory] = {}
    for index, upload in run.uploads.items():
        rows = np.abs(upload.ctrl) == 1.0
        action = np.where(rows, np.sign(upload.ctrl) * saturated, upload.action)
        assert not np.array_equal(action, upload.action)
        forged[index] = replace(upload, action=action)
        verdict = verify_episode(forged[index], run.protocol, index)
        assert_reproduces(verdict, run.results.episodes[index])
        assert verdict.replay is not None
        np.testing.assert_array_equal(verdict.replay.action, action)
    again = rescore(run.protocol, forged, INFO, "inprocess")
    assert again.episodes == run.results.episodes
    assert again.aggregate == run.results.aggregate


def test_a_verdict_carries_the_trusted_replay(honest_runs: dict[str, Honest]) -> None:
    run = honest_runs["pid"]
    verdict = verify_episode(run.uploads[0], run.protocol, 0)
    assert verdict.index == 0
    assert verdict.replay is not None
    np.testing.assert_array_equal(verdict.replay.qpos, run.uploads[0].qpos)
    assert verdict.replay.meta["episode_index"] == 0


@pytest.mark.parametrize("preset", ["ideal", "clean", "standard"])
@pytest.mark.parametrize("factory", [pid, lqr], ids=["pid", "lqr"])
def test_honest_runs_reproduce_on_every_preset(tmp_path: Path, preset: str, factory: Any) -> None:
    """Each preset has its own sensor path (``ideal`` records noiseless IMU readings)."""
    run = run_honest(public(preset, 2), factory, tmp_path)
    assert run.results.episodes is not None
    for index, upload in run.uploads.items():
        assert_reproduces(verify_episode(upload, run.protocol, index), run.results.episodes[index])
    again = rescore(run.protocol, run.uploads, INFO, "inprocess")
    assert again.episodes == run.results.episodes
    assert again.aggregate == run.results.aggregate


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
    # The replay's blow-up takes precedence over whatever error the upload claims (§16 step 4).
    steps = run.uploads[0].steps
    other = with_meta(run.uploads[0], error={"code": "exception", "step": steps, "detail": "x"})
    verdict = verify_episode(other, run.protocol, 0)
    assert verdict.ok, verdict.reason
    assert verdict.result.error == expected.error


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


def claim_error(run: Honest, index: int, code: str = "timeout") -> Trajectory:
    """A well-formed error report added to an honest episode, with the epilogue the replay
    then simulates (an error means a 1 s epilogue, §11), so the claim reaches the accepted path."""
    upload = run.uploads[index]
    claimed = with_meta(upload, error={"code": code, "step": upload.steps, "detail": "my bad"})
    replay = verify_episode(claimed, run.protocol, index).replay
    assert replay is not None
    return replace(claimed, epilogue_qpos=replay.epilogue_qpos)


def test_a_claimed_error_without_its_epilogue_does_not_reproduce(
    honest_runs: dict[str, Honest],
) -> None:
    run = honest_runs["lqr"]
    upload = run.uploads[0]
    claimed = with_meta(upload, error={"code": "timeout", "step": upload.steps, "detail": "x"})
    verdict = verify_episode(claimed, run.protocol, 0)
    assert not verdict.ok
    assert verdict.code == "not_reproducible"
    assert "epilogue_qpos" in verdict.reason
    assert (verdict.result.score, verdict.result.success) == (0.0, False)


def test_an_honest_error_report_only_lowers_the_score(honest_runs: dict[str, Honest]) -> None:
    """Claiming an error on a good episode can only cost its author: it reproduces, keeps the
    error, scores 0 and is not a success, although the replay's own metrics are a success."""
    run = honest_runs["lqr"]
    assert run.results.episodes is not None
    honest = run.results.episodes[0]
    assert honest.success
    assert honest.score > 0
    verdict = verify_episode(claim_error(run, 0), run.protocol, 0)
    assert verdict.ok, verdict.reason
    assert verdict.code is None
    assert verdict.result.error is not None
    assert (verdict.result.error.code, verdict.result.error.step) == ("timeout", 1000)
    assert verdict.result.error.detail == "my bad"
    assert verdict.result.score == 0.0
    assert verdict.result.success is False
    assert verdict.result.metrics == honest.metrics  # the replay is scored, then zeroed
    assert "ended with timeout" in verdict.reason


def test_rescore_counts_a_claimed_error_as_a_failure(honest_runs: dict[str, Honest]) -> None:
    run = honest_runs["lqr"]
    assert run.results.episodes is not None
    uploads = {**run.uploads, 0: claim_error(run, 0, "exception")}
    again = rescore(run.protocol, uploads, INFO, "inprocess")
    assert again.complete
    assert again.episodes is not None
    assert again.episodes[0].error is not None
    assert again.episodes[0].error.code == "exception"
    assert again.episodes[1:] == run.results.episodes[1:]
    successes = sum(e.success for e in run.results.episodes)
    assert again.aggregate.success_rate == (successes - 1) / 3
    assert again.aggregate.errors == run.results.aggregate.errors + 1
    assert again.aggregate.score == pytest.approx(
        (run.results.aggregate.score * 3 - run.results.episodes[0].score) / 3, abs=1e-12
    )


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
    # Redaction hides the episodes, never changes the aggregate.
    public_again = rescore(run.protocol, run.uploads, INFO, "container")
    assert again.aggregate == public_again.aggregate == run.results.aggregate
    assert again.complete


def test_a_private_rescore_counts_forged_and_missing_episodes(
    honest_runs: dict[str, Honest],
) -> None:
    run = honest_runs["lqr"]
    assert run.results.episodes is not None
    private = replace(run.protocol, seed_set="private", salt=99)
    upload = run.uploads[0]
    forged = replace(upload, push_force=upload.push_force + 1.0)
    again = rescore(private, {0: forged, 1: run.uploads[1]}, INFO, "container")
    assert again.episodes is None
    assert not again.complete
    assert again.aggregate.errors == 2
    assert again.aggregate.episodes == 3
    assert again.aggregate.score == pytest.approx(run.results.episodes[1].score / 3, abs=1e-12)
    assert again.aggregate.success_rate == run.results.episodes[1].success / 3


PRIVATE_SALT = 2**127 + 1


def test_verify_run_counts_failures_before_the_results_are_redacted(
    honest_runs: dict[str, Honest],
) -> None:
    """§16: whether a run verified comes from the verdicts. A private protocol's results list no
    episodes, so a count taken from them would call every forged run reproduced."""
    run = honest_runs["lqr"]
    private = replace(run.protocol, seed_set="private", salt=PRIVATE_SALT)
    upload = run.uploads[0]
    forged = replace(upload, push_force=upload.push_force + 1.0)
    verdict = verify_run(private, {0: forged, 1: run.uploads[1]}, INFO, "container")
    assert isinstance(verdict, RunVerdict)
    assert verdict.results.episodes is None
    assert verdict.failed == 2  # episode 0 does not reproduce, episode 2 is missing
    assert not verdict.ok
    assert verdict.results == rescore(private, {0: forged, 1: run.uploads[1]}, INFO, "container")
    honest = verify_run(private, run.uploads, INFO, "container")
    assert (honest.ok, honest.failed) == (True, 0)
    assert honest.results == rescore(private, run.uploads, INFO, "container")


def test_errors_of_the_run_itself_are_not_verification_failures(tmp_path: Path) -> None:
    """An honest error report reproduces: the episode scores 0 but the run verified."""
    errored = run_honest(public("clean", 2), explodes(30, RuntimeError("bug")), tmp_path)
    verdict = verify_run(errored.protocol, errored.uploads, INFO, "inprocess")
    assert verdict.results.aggregate.errors == 2
    assert (verdict.ok, verdict.failed) == (True, 0)


# ------------------------------------------------------------------ robustness


JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=8),
    lambda inner: (
        st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=5), inner, max_size=3)
    ),
    max_leaves=8,
)
CONTRACT_ENV_META_FIELDS = (
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
"""Contract §16 step 5, in order: written out here so dropping one from the code fails a test."""
META_KEYS = (
    "challenge",
    "preset",
    "robot_hash",
    "episode_index",
    "scenario",
    "error",
    *CONTRACT_ENV_META_FIELDS,
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


def other_reason(termination: dict[str, Any]) -> dict[str, Any]:
    return {**termination, "reason": "fell" if termination["reason"] == "timeout" else "timeout"}


PLAUSIBLE_LIES: dict[str, Any] = {
    "termination": other_reason,
    "max_steps": lambda v: v // 2,
    "control_dt": lambda v: v * 2,
    "timestep": lambda v: v / 2,
    "frame_skip": lambda v: v + 2,
    "joint_qpos": lambda v: {**v, "wheels": v["wheels"][::-1]},
    "wheel_sign": lambda v: [v[0], -v[1]],
    "obs_schema": lambda v: v + ".forged",
    "obs_features": lambda v: v[::-1],
}
"""A wrong but plausible value for each env-produced meta field, from the honest one."""
SAME_VALUE_OTHER_TYPE: dict[str, Any] = {
    "termination": lambda v: {**v, "step": float(v["step"])},
    "max_steps": float,
    "control_dt": lambda v: [v],
    "timestep": str,
    "frame_skip": float,
    "joint_qpos": lambda v: {**v, "root": float(v["root"])},
    "wheel_sign": lambda v: [int(x) for x in v],
    "obs_schema": lambda v: [v],
    "obs_features": ",".join,
}
"""The honest value re-typed (``1000`` -> ``1000.0``, ``[1.0, 1.0]`` -> ``[1, 1]``, ...)."""


def test_verify_checks_exactly_the_contract_env_meta_fields() -> None:
    assert ENV_META_FIELDS == CONTRACT_ENV_META_FIELDS
    assert set(PLAUSIBLE_LIES) == set(SAME_VALUE_OTHER_TYPE) == set(CONTRACT_ENV_META_FIELDS)


@pytest.mark.parametrize("edit", ["lie", "retyped"])
@pytest.mark.parametrize("field", CONTRACT_ENV_META_FIELDS)
def test_every_env_meta_field_must_equal_the_replay(
    honest_runs: dict[str, Honest], field: str, edit: str
) -> None:
    run = honest_runs["pid"]
    upload = run.uploads[0]
    honest = upload.meta[field]
    forge = (PLAUSIBLE_LIES if edit == "lie" else SAME_VALUE_OTHER_TYPE)[field]
    forged = forge(honest)
    assert json.dumps(forged, sort_keys=True) != json.dumps(honest, sort_keys=True)
    verdict = verify_episode(with_meta(upload, **{field: forged}), run.protocol, 0)
    assert not verdict.ok
    assert verdict.code == "not_reproducible"
    assert verdict.reason == f"meta.{field} differs from the replay"
    assert (verdict.result.score, verdict.result.success) == (0.0, False)


@pytest.mark.parametrize("retype", [float, bool], ids=["float", "bool"])
@pytest.mark.parametrize("index", [0, 1])
def test_episode_index_must_be_an_int_not_an_equal_float_or_bool(
    honest_runs: dict[str, Honest], index: int, retype: Any
) -> None:
    """§16 step 1: ``meta.episode_index == index`` as an integer (``1.0 == True == 1``)."""
    run = honest_runs["pid"]
    forged = retype(index)
    assert forged == index
    verdict = verify_episode(
        with_meta(run.uploads[index], episode_index=forged), run.protocol, index
    )
    assert not verdict.ok
    assert verdict.code == "not_reproducible"
    assert verdict.reason == f"meta.episode_index is {forged!r}, expected {index}"
    assert (verdict.result.score, verdict.result.success) == (0.0, False)


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
