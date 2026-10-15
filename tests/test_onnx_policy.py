"""The ONNX runtime: ``load_policy_dir``, ``policy_from_bytes`` and ``ONNXPolicy``
(docs/design/phase-2.md §3.5, §6 ``test_onnx_policy``).

A policy directory loads only if every check passes; any failure is one fixed-code
``LoadError``. ``ONNXPolicy`` implements the Controller protocol: it feeds the float32
observation row, checks the output and returns Python floats.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import topple.policy.runtime as runtime
import topple.policy.validate as validate
from policy_support import (
    A,
    D,
    build,
    first_party_manifest,
    fixture_rows,
    graph_model,
    manifest_bytes,
    node,
    parse,
    ppo_bytes,
    sac_bytes,
    serialize,
    tensor,
    tiny_norm,
    tiny_weights,
    write_policy_dir,
)
from topple.challenge import load_challenge
from topple.contract import BalancerObs, Controller, TaskSpec
from topple.controllers.onnx_policy import ONNXPolicy, policy_factory
from topple.envs.balance import BalanceEnv
from topple.evaluate import EvalProtocol, evaluate
from topple.loader import LoadError, load_submission
from topple.policy.build import policy_metadata, reference_actions
from topple.policy.runtime import LoadedPolicy, load_policy_dir, policy_from_bytes
from topple.policy.validate import canonicalize
from topple.results import ControllerInfo

PPO = ppo_bytes()
SAC = sac_bytes()
BALANCE = load_challenge("balance-v1")
SPEC: TaskSpec = BalanceEnv("ideal").task_spec
# one load error: "<file>: <code>: <fixed text>" (several joined by "; ")
FIXED = re.compile(r"policy\.(onnx|json): [a-z][a-z0-9_ ]*(: [\x20-\x7e]*)?")


def load_error(root: Path) -> str:
    with pytest.raises(LoadError) as info:
        load_policy_dir(root, challenge=BALANCE)
    text = str(info.value)
    assert FIXED.fullmatch(text), text
    return text


def obs_of(row: np.ndarray[Any, Any]) -> BalancerObs:
    return BalancerObs.from_array([float(v) for v in row], 0.0)


# --- load_policy_dir --------------------------------------------------------------------------


@pytest.mark.parametrize(("data", "rule"), [(PPO, "clip(mean)"), (SAC, "tanh(mean)")])
def test_first_party_policies_load(tmp_path: Path, data: bytes, rule: str) -> None:
    root = write_policy_dir(
        tmp_path / "bot", data, manifest=first_party_manifest(data, action_rule=rule)
    )
    loaded = load_policy_dir(root, challenge=BALANCE)
    assert isinstance(loaded, LoadedPolicy)
    assert loaded.manifest is not None
    assert loaded.manifest.action_rule == rule
    assert loaded.validated.canonical == data
    assert loaded.validated.policy_sha256 == hashlib.sha256(data).hexdigest()
    assert (loaded.validated.obs_dim, loaded.validated.action_dim) == (D, A)


def test_sessions_run_on_the_cpu_provider_only(tmp_path: Path) -> None:
    loaded = load_policy_dir(write_policy_dir(tmp_path / "bot"), challenge=BALANCE)
    assert loaded.session.get_providers() == ["CPUExecutionProvider"]


def test_sha_mismatch(tmp_path: Path) -> None:
    m = first_party_manifest(PPO)
    m["onnx"]["sha256"] = hashlib.sha256(SAC).hexdigest()
    text = load_error(write_policy_dir(tmp_path / "bot", PPO, manifest=m))
    assert "policy_sha256_mismatch" in text


def test_metadata_mismatch(tmp_path: Path) -> None:
    """The graph's ``topple.*`` metadata must say what policy.json says (§3.2)."""
    meta = policy_metadata(SPEC, "clip(mean)")
    meta["topple.control_dt"] = json.dumps(0.02)
    data = build(metadata=meta)
    text = load_error(write_policy_dir(tmp_path / "bot", data))
    assert text == "policy.onnx: metadata_mismatch: topple.control_dt differs from policy.json"


def test_metadata_compares_json_types_exactly(tmp_path: Path) -> None:
    """``1.0`` and ``true`` are equal in Python but not in JSON: the comparison is type-exact,
    so the metadata check fails here (a Python ``==`` would pass it on to the challenge)."""
    proto = parse(PPO)
    for entry in proto.metadata_props:
        if entry.key == "topple.control_dt":
            entry.value = "true"
    data = serialize(proto)
    m = first_party_manifest(data)
    m["control_dt"] = 1.0
    text = load_error(write_policy_dir(tmp_path / "bot", data, manifest=m))
    assert text == "policy.onnx: metadata_mismatch: topple.control_dt differs from policy.json"


def test_metadata_that_is_not_json_is_a_mismatch(tmp_path: Path) -> None:
    proto = parse(PPO)
    for entry in proto.metadata_props:
        if entry.key == "topple.challenge":
            entry.value = "NaN"
    data = serialize(proto)
    text = load_error(write_policy_dir(tmp_path / "bot", data))
    assert "metadata_mismatch: topple.challenge" in text


def test_challenge_mismatch(tmp_path: Path) -> None:
    """Graph and manifest agree with each other on control_dt 0.02, not with balance-v1."""
    meta = policy_metadata(SPEC, "clip(mean)")
    meta["topple.control_dt"] = json.dumps(0.02)
    data = build(metadata=meta)
    m = first_party_manifest(data)
    m["control_dt"] = 0.02
    text = load_error(write_policy_dir(tmp_path / "bot", data, manifest=m))
    assert text == "policy.json: challenge_mismatch: control_dt must be 0.01"


def test_model_mismatch(tmp_path: Path) -> None:
    m = first_party_manifest(PPO)
    m["onnx"]["opset"] = 18
    text = load_error(write_policy_dir(tmp_path / "bot", PPO, manifest=m))
    assert text == "policy.json: manifest_model_mismatch: onnx.opset differs from policy.onnx"


def test_action_rule_must_be_the_structure(tmp_path: Path) -> None:
    m = first_party_manifest(PPO, action_rule="tanh(mean)")
    text = load_error(write_policy_dir(tmp_path / "bot", PPO, manifest=m))
    assert text == (
        "policy.json: action_rule_mismatch: policy.json says 'tanh(mean)'; "
        "the graph is 'clip(mean)'"
    )


def test_other_structures_must_say_other(tmp_path: Path) -> None:
    """A valid graph that is no §3.2 sequence loads only as ``other``."""
    w = tiny_weights()
    nodes = [
        node("Gemm", ["obs", "w0", "b0"], "h", transB=1),
        node("Tanh", ["h"], "t"),
        node("Gemm", ["t", "w1"], "action", transB=1),
    ]
    inits = [
        tensor("w0", w.layers[0][0]),
        tensor("b0", w.layers[0][1]),
        tensor("w1", w.layers[2][0]),
    ]
    data = canonicalize(graph_model(nodes, inits, metadata=policy_metadata(SPEC, "other")))
    root = write_policy_dir(
        tmp_path / "bot", data, manifest=first_party_manifest(data, action_rule="other")
    )
    assert load_policy_dir(root, challenge=BALANCE).manifest is not None
    m = first_party_manifest(data, action_rule="clip(mean)")
    text = load_error(write_policy_dir(tmp_path / "bot2", data, manifest=m))
    assert "action_rule_mismatch" in text


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d[:-1],  # truncated
        lambda d: d + b"\x00",
        lambda d: b"not onnx",
        lambda d: serialize(_with_producer_version(d)),
    ],
    ids=["truncated", "trailing", "garbage", "producer_version"],
)
def test_invalid_onnx_is_a_fixed_code_load_error(tmp_path: Path, mutate: Any) -> None:
    data = mutate(PPO)
    m = first_party_manifest(PPO)
    m["onnx"]["sha256"] = hashlib.sha256(data).hexdigest()
    text = load_error(write_policy_dir(tmp_path / "bot", data, manifest=m))
    assert text.startswith("policy.onnx: ")
    assert "Traceback" not in text


def _with_producer_version(data: bytes) -> Any:
    proto = parse(data)
    proto.producer_version = ""
    return proto


def test_wrong_width_for_the_challenge_is_rejected(tmp_path: Path) -> None:
    norm = tiny_norm(d=11)
    meta = policy_metadata(SPEC, "clip(mean)")
    meta["topple.obs_features"] = json.dumps([*SPEC.obs_features, "extra"])
    data = build(tiny_weights(d=11), norm, metadata=meta)
    text = load_error(write_policy_dir(tmp_path / "bot", data))
    assert text.startswith("policy.onnx: bad_input")


def test_smoke_failures_are_load_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = write_policy_dir(tmp_path / "bot")

    def terminated(v: object) -> list[str]:
        return ["smoke call exceeded 1 s"]

    monkeypatch.setattr(runtime, "smoke_test", terminated)
    assert load_error(root) == "policy.onnx: smoke call exceeded 1 s"


def test_smoke_timers_are_cancelled_after_a_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """32 calls, each with its own timer over a fresh RunOptions, every timer cancelled and
    finished when the load returns, none of them fired."""
    timers: list[threading.Timer] = []
    options: list[Any] = []

    class RecordingTimer(threading.Timer):
        def __init__(self, interval: float, function: Any, args: Any = None, **kw: Any) -> None:
            super().__init__(interval, function, args, **kw)
            timers.append(self)
            options.append(args[0])

    monkeypatch.setattr(validate.threading, "Timer", RecordingTimer)
    load_policy_dir(write_policy_dir(tmp_path / "bot"), challenge=BALANCE)
    assert len(timers) == 32
    assert all(t.finished.is_set() and not t.is_alive() for t in timers)
    assert len({id(o) for o in options}) == 32  # a fresh RunOptions per call
    assert not any(o.terminate for o in options)  # cancelled before it could fire


def test_an_exception_inside_the_load_is_a_fixed_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(proto: object) -> object:
        raise RuntimeError("raw \x1b text")

    monkeypatch.setattr(runtime, "structure_match", broken)
    text = load_error(write_policy_dir(tmp_path / "bot"))
    assert text == "policy.onnx: policy_load_failed: raised while loading"


def test_load_policy_dir_needs_a_challenge(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Challenge"):
        load_policy_dir(write_policy_dir(tmp_path / "bot"), challenge="balance-v1")  # type: ignore[arg-type]


# --- policy_from_bytes ------------------------------------------------------------------------


def test_policy_from_bytes() -> None:
    loaded = policy_from_bytes(PPO, challenge=BALANCE)
    assert loaded.manifest is None
    assert loaded.validated.canonical == PPO
    expected = hashlib.sha256(PPO).hexdigest()
    assert policy_from_bytes(PPO, challenge=BALANCE, expected_sha256=expected).manifest is None


def test_policy_from_bytes_checks_the_expected_sha() -> None:
    with pytest.raises(LoadError, match="policy_sha256_mismatch"):
        policy_from_bytes(PPO, challenge=BALANCE, expected_sha256=hashlib.sha256(SAC).hexdigest())


def test_policy_from_bytes_checks_metadata_against_the_challenge() -> None:
    meta = policy_metadata(SPEC, "clip(mean)")
    meta["topple.challenge"] = json.dumps("other-v1")
    with pytest.raises(LoadError) as info:
        policy_from_bytes(build(metadata=meta), challenge=BALANCE)
    assert (
        str(info.value)
        == "policy.onnx: metadata_mismatch: topple.challenge differs from balance-v1"
    )


def test_policy_from_bytes_needs_the_rule_the_structure_has() -> None:
    meta = policy_metadata(SPEC, "clip(mean)")
    meta["topple.action_rule"] = json.dumps("other")
    proto = parse(PPO)
    for entry in proto.metadata_props:
        if entry.key == "topple.action_rule":
            entry.value = meta["topple.action_rule"]
    with pytest.raises(LoadError, match=r"metadata_mismatch: topple\.action_rule"):
        policy_from_bytes(serialize(proto), challenge=BALANCE)


@pytest.mark.parametrize("data", [bytearray(PPO), "policy"], ids=["bytearray", "str"])
def test_policy_from_bytes_takes_exact_bytes(data: Any) -> None:
    with pytest.raises(LoadError, match=r"policy\.onnx must be bytes"):
        policy_from_bytes(data, challenge=BALANCE)


# --- ONNXPolicy -------------------------------------------------------------------------------


def test_act_returns_a_tuple_of_python_floats() -> None:
    loaded = policy_from_bytes(PPO, challenge=BALANCE)
    policy = policy_factory(loaded)(SPEC)
    assert isinstance(policy, ONNXPolicy)
    assert isinstance(policy, Controller)
    rows = fixture_rows()
    expected = reference_actions(
        tiny_weights(), tiny_norm(), "clip(mean)", rows, action=SPEC.action
    )
    policy.reset(7)
    for row, want in zip(rows, expected, strict=True):
        got = policy.act(obs_of(row))
        assert type(got) is tuple
        assert len(got) == A
        assert all(type(x) is float for x in got)
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=0)


def test_reset_is_a_no_op() -> None:
    policy = policy_factory(policy_from_bytes(PPO, challenge=BALANCE))(SPEC)
    obs = obs_of(fixture_rows()[3])
    first = policy.act(obs)
    assert policy.reset(123) is None
    assert policy.act(obs) == first


class FakeSession:
    def __init__(self, out: Any) -> None:
        self.out = out
        self.fed: list[Any] = []

    def run(self, names: Any, feeds: dict[str, Any]) -> Any:
        self.fed.append(feeds)
        return self.out


def test_act_feeds_one_float32_row() -> None:
    session = FakeSession([np.zeros((1, A), np.float32)])
    policy = ONNXPolicy(session, D, A)
    policy.act(obs_of(fixture_rows()[0]))
    (feed,) = session.fed
    assert set(feed) == {"obs"}
    assert feed["obs"].dtype == np.float32
    assert feed["obs"].shape == (1, D)


@pytest.mark.parametrize(
    "out",
    [
        [np.zeros((1, A), np.float64)],
        [np.zeros((1, A + 1), np.float32)],
        [np.zeros((A,), np.float32)],
        [],
        "nonsense",
    ],
    ids=["float64", "width", "rank1", "empty", "str"],
)
def test_act_checks_the_output(out: Any) -> None:
    with pytest.raises(RuntimeError, match=r"float32 \(1, 2\)"):
        ONNXPolicy(FakeSession(out), D, A).act(obs_of(fixture_rows()[0]))


def test_act_checks_the_observation_width() -> None:
    with pytest.raises(ValueError, match="11 features"):
        ONNXPolicy(FakeSession([np.zeros((1, A), np.float32)]), D + 1, A).act(
            obs_of(fixture_rows()[0])
        )


def test_the_factory_refuses_another_task() -> None:
    factory = policy_factory(policy_from_bytes(PPO, challenge=BALANCE))
    other = TaskSpec.from_dict({**SPEC.to_dict(), "obs_features": [*SPEC.obs_features, "x"]})
    with pytest.raises(ValueError, match="10 features"):
        factory(other)


def test_policies_and_sessions_are_never_pickled() -> None:
    loaded = policy_from_bytes(PPO, challenge=BALANCE)
    policy = policy_factory(loaded)(SPEC)
    for obj in (loaded, policy):
        with pytest.raises(TypeError, match="never pickled"):
            pickle.dumps(obj)
        with pytest.raises(TypeError, match="never pickled"):
            copy.copy(obj)


# --- the harness ------------------------------------------------------------------------------


def test_one_ideal_episode_through_evaluate(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "tiny_ppo")
    sub = load_submission(root, challenge=BALANCE)
    assert sub.kind == "onnx"
    protocol = EvalProtocol.from_id("balance-v1/ideal")
    protocol = EvalProtocol(protocol.challenge, protocol.preset, seeds=protocol.seeds[:1])
    info = ControllerInfo(id=sub.id, sha256=sub.sha256, policy_sha256=sub.policy_sha256)
    results = evaluate(protocol, sub.factory, info)
    assert results.episodes is not None
    (episode,) = results.episodes
    assert episode.error is None, episode.error
    assert episode.metrics is not None
    assert episode.metrics["steps"] > 0
    assert results.controller.policy_sha256 == hashlib.sha256(PPO).hexdigest()
    assert results.versions.onnxruntime is not None


def test_cli_run_evaluates_a_policy_directory(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from topple.cli import app
    from topple.results import read_results

    root = write_policy_dir(tmp_path / "tiny_ppo")
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app, ["run", str(root), "-p", "balance-v1/ideal", "--out", str(out), "--no-video"]
    )
    assert result.exit_code == 0, result.output
    results = read_results(out / "results.json")
    assert results.controller.id == "tiny_ppo"
    assert results.controller.policy_sha256 == hashlib.sha256(PPO).hexdigest()


def test_cli_eval_reports_a_bad_policy_in_one_line(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from topple.cli import app

    m = first_party_manifest(PPO)
    m["challenge"] = "<b>evil</b>"
    root = write_policy_dir(tmp_path / "bot", manifest=manifest_bytes(m))
    result = CliRunner().invoke(
        app, ["eval", str(root), "-p", "balance-v1/ideal", "--out", str(tmp_path / "o")]
    )
    assert result.exit_code == 2
    assert result.output.count("\n") == 1, result.output
    assert "<" not in result.output
    assert "manifest_invalid: challenge" in result.output
