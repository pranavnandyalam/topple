import json
import math
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

import topple.evaluate as evaluate_module
from topple.challenge import load_challenge_dir
from topple.contract import BalancerObs, TaskSpec, controller_obs
from topple.controllers.pid import PIDController
from topple.envs.balance import BalanceEnv
from topple.evaluate import (
    EvalProtocol,
    evaluate,
    run_episode,
    summarize,
    wants_epilogue,
)
from topple.results import (
    ControllerInfo,
    EpisodeResult,
    ErrorInfo,
    EvalResults,
    read_results,
    write_results,
)
from topple.rng import controller_seed
from topple.robot import challenge_dir
from topple.traj import load_npz
from topple.verify import verify_episode

INFO = ControllerInfo(id="test", sha256="0" * 64)


def pid(_spec: TaskSpec) -> PIDController:
    return PIDController()


class Zero:
    def reset(self, rng_seed: int) -> None:
        pass

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        return (0.0, 0.0)


class Recorder:
    """Records what the harness hands the controller."""

    seeds: list[int] = []  # noqa: RUF012
    specs: list[TaskSpec] = []  # noqa: RUF012

    def __init__(self, spec: TaskSpec) -> None:
        Recorder.specs.append(spec)

    def reset(self, rng_seed: int) -> None:
        Recorder.seeds.append(rng_seed)

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        return (0.0, 0.0)


class Explodes:
    def __init__(self, at_step: int, payload: Any) -> None:
        self.at_step, self.payload, self.k = at_step, payload, 0

    def reset(self, rng_seed: int) -> None:
        self.k = 0

    def act(self, obs: BalancerObs) -> Any:
        self.k += 1
        if self.k > self.at_step:
            if isinstance(self.payload, BaseException):
                raise self.payload
            return self.payload
        return (0.0, 0.0)


def protocol(preset: str = "clean", n: int = 3) -> EvalProtocol:
    base = EvalProtocol.from_id(f"balance-v1/{preset}")
    return replace(base, seeds=tuple(range(n)))


def test_protocol_from_id_and_alias() -> None:
    p = EvalProtocol.from_id("eval_v1")
    assert p.id == "balance-v1/standard"
    assert len(p.seeds) == 20
    assert EvalProtocol.from_id("eval_v1_clean").preset == "clean"
    with pytest.raises(ValueError, match="no preset"):
        EvalProtocol.from_id("balance-v1/nope")


def test_protocol_hash_depends_on_preset() -> None:
    assert (
        EvalProtocol.from_id("balance-v1/clean").protocol_hash()
        != EvalProtocol.from_id("balance-v1/standard").protocol_hash()
    )


def test_pid_on_clean_scores_high(tmp_path: Path) -> None:
    results = evaluate(protocol("clean", 3), pid, INFO, traj_dir=tmp_path)
    assert results.complete
    assert results.aggregate.success_rate == 1.0
    assert results.aggregate.score > 85
    assert results.episodes is not None
    assert all(e.error is None for e in results.episodes)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "traj_000.npz",
        "traj_001.npz",
        "traj_002.npz",
    ]


def test_zero_controller_scores_below_10() -> None:
    results = evaluate(protocol("clean", 3), lambda spec: Zero(), INFO)
    assert results.aggregate.success_rate == 0.0
    assert results.aggregate.score < 10


def test_controller_never_sees_episode_seed() -> None:
    Recorder.seeds.clear()
    Recorder.specs.clear()
    evaluate(protocol("ideal", 2), Recorder, INFO)
    assert len(Recorder.seeds) == 2
    assert not set(Recorder.seeds) & {0, 1}
    assert Recorder.specs[0].challenge == "balance-v1"
    assert Recorder.specs[0].robot.wheel_radius_m == pytest.approx(0.034)


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (RuntimeError("boom"), "exception"),
        ((float("nan"), 0.0), "invalid_action"),
        ((0.1,), "invalid_action"),
        ("left", "invalid_action"),
    ],
)
def test_controller_errors_are_recorded_not_raised(payload: Any, code: str) -> None:
    env = BalanceEnv("ideal", record=True)
    run = run_episode(env, lambda spec: Explodes(5, payload), seed=0, index=0)
    assert run.result.error is not None
    assert run.result.error.code == code
    assert run.result.error.step == 5
    assert run.result.score == 0.0
    assert not run.result.success
    assert run.trajectory.steps == 5


def test_factory_errors_are_recorded() -> None:
    def bad_factory(spec: TaskSpec) -> Zero:
        raise ImportError("missing dependency")

    env = BalanceEnv("ideal", record=True)
    run = run_episode(env, bad_factory, seed=0, index=0)
    assert run.result.error is not None
    assert run.result.error.code == "exception"
    assert run.trajectory.steps == 0


def test_long_error_detail_is_truncated() -> None:
    env = BalanceEnv("ideal", record=True)
    run = run_episode(env, lambda spec: Explodes(0, RuntimeError("x" * 5000)), seed=0, index=0)
    assert run.result.error is not None
    assert len(run.result.error.detail) == 300


def test_results_round_trip_and_atomic_write(tmp_path: Path) -> None:
    results = evaluate(protocol("ideal", 2), pid, INFO)
    path = tmp_path / "out" / "results.json"
    write_results(results, path)
    assert read_results(path) == results
    data = json.loads(path.read_text())
    assert data["format"] == "topple.results/1"
    assert data["isolation"] == "inprocess"
    assert not results.rankable
    assert [p.name for p in path.parent.iterdir()] == ["results.json"]


def test_results_reject_nan_and_extra_fields() -> None:
    results = evaluate(protocol("ideal", 1), pid, INFO)
    data = results.model_dump()
    data["aggregate"]["score"] = math.nan
    with pytest.raises(ValidationError):
        EvalResults.model_validate(data)
    data = results.model_dump()
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        EvalResults.model_validate(data)


def test_private_seed_set_redacts_episodes() -> None:
    p = replace(protocol("ideal", 2), seed_set="private", salt=12345)
    results = evaluate(p, pid, INFO)
    assert results.seed_set == "private"
    assert results.episodes is None
    assert results.aggregate.episodes == 2


def test_error_info_codes_are_closed() -> None:
    with pytest.raises(ValidationError):
        ErrorInfo.model_validate({"code": "teleported", "step": 0})


def test_saved_trajectory_verifies_and_rescores(tmp_path: Path) -> None:
    results = evaluate(protocol("standard", 2), pid, INFO, traj_dir=tmp_path)
    assert results.episodes is not None
    for i in range(len(results.episodes)):
        traj = load_npz(tmp_path / f"traj_{i:03d}.npz")
        verification = verify_episode(traj)
        assert verification.ok, verification.reason
        assert traj.meta["episode_index"] == i


def test_tampered_trajectory_fails_verification(tmp_path: Path) -> None:
    evaluate(protocol("clean", 1), pid, INFO, traj_dir=tmp_path)
    traj = load_npz(tmp_path / "traj_000.npz")
    qpos = traj.qpos.copy()
    qpos[-1, 0] += 1e-9
    forged = type(traj)(**{**traj.__dict__, "qpos": qpos})
    verification = verify_episode(forged)
    assert not verification.ok
    assert "qpos" in verification.reason


def test_truncated_trajectory_fails_verification(tmp_path: Path) -> None:
    """Claiming survival with fewer commands than the real episode needed is caught."""
    evaluate(protocol("clean", 1), lambda spec: Zero(), INFO, traj_dir=tmp_path)
    traj = load_npz(tmp_path / "traj_000.npz")
    ctrl = np.vstack([traj.ctrl, np.zeros((5, 2))])
    longer = type(traj)(
        **{
            **traj.__dict__,
            "ctrl": ctrl,
            "action": np.vstack([traj.action, np.zeros((5, 2))]),
            "obs": np.vstack([traj.obs, np.zeros((5, 10), np.float32)]),
            "enc_counts": np.vstack([traj.enc_counts, np.zeros((5, 2), np.int32)]),
            "gyro": np.vstack([traj.gyro, np.zeros((5, 3), np.float32)]),
            "accel": np.vstack([traj.accel, np.zeros((5, 3), np.float32)]),
            "push_force": np.vstack([traj.push_force, np.zeros((5, 3))]),
            "qpos": np.vstack([traj.qpos, np.repeat(traj.qpos[-1:], 5, axis=0)]),
            "qvel": np.vstack([traj.qvel, np.repeat(traj.qvel[-1:], 5, axis=0)]),
        }
    )
    assert not verify_episode(longer).ok


# ------------------------------------------------------------------ controller misbehaviour (B)


class Endless:
    def __iter__(self) -> Iterator[float]:
        while True:
            yield 0.0


class IterRaises:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def __iter__(self) -> Iterator[float]:
        raise self.exc


class BadStrError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("cannot describe myself")


def lazy_action() -> Iterator[float]:
    yield from (0.1, 0.2)


MISBEHAVIOUR: dict[str, tuple[Any, str]] = {
    "act raises SystemExit": (SystemExit(3), "exception"),
    "act raises GeneratorExit": (GeneratorExit(), "exception"),
    "act raises StopIteration": (StopIteration(), "exception"),
    "exception with a broken __str__": (BadStrError(), "exception"),
    "action iteration raises SystemExit": (IterRaises(SystemExit(1)), "exception"),
    "action iteration raises ValueError": (IterRaises(ValueError("bad")), "invalid_action"),
    "generator": (lazy_action(), "invalid_action"),
    "huge int": ([10**400, 0.0], "invalid_action"),
    "endless iterable": (Endless(), "invalid_action"),
    "huge range": (range(10**12), "invalid_action"),
    "set": ({0.1, 0.2}, "invalid_action"),
}


@pytest.mark.parametrize(("payload", "code"), MISBEHAVIOUR.values(), ids=MISBEHAVIOUR.keys())
def test_misbehaving_controllers_never_crash_the_harness(payload: Any, code: str) -> None:
    env = BalanceEnv("ideal", record=True)
    run = run_episode(env, lambda spec: Explodes(5, payload), seed=0, index=0)
    assert run.result.error is not None
    assert run.result.error.code == code
    assert run.result.error.step == 5
    assert run.result.score == 0.0
    assert not run.result.success
    assert run.trajectory.steps == 5


@pytest.mark.parametrize("where", ["factory", "reset"])
@pytest.mark.parametrize("exc", [SystemExit(2), GeneratorExit(), BadStrError()])
def test_setup_failures_are_exceptions(where: str, exc: BaseException) -> None:
    class Fails(Zero):
        def reset(self, rng_seed: int) -> None:
            raise exc

    def factory(spec: TaskSpec) -> Zero:
        if where == "factory":
            raise exc
        return Fails()

    run = run_episode(BalanceEnv("ideal", record=True), factory, seed=0, index=0)
    assert run.result.error is not None
    assert (run.result.error.code, run.result.error.step) == ("exception", 0)
    assert run.trajectory.steps == 0


@pytest.mark.parametrize("where", ["factory", "reset", "act", "action"])
def test_keyboard_interrupt_still_stops_the_run(where: str) -> None:
    class Interrupts(Zero):
        def reset(self, rng_seed: int) -> None:
            if where == "reset":
                raise KeyboardInterrupt

        def act(self, obs: BalancerObs) -> Any:
            if where == "act":
                raise KeyboardInterrupt
            return IterRaises(KeyboardInterrupt()) if where == "action" else (0.0, 0.0)

    def factory(spec: TaskSpec) -> Zero:
        if where == "factory":
            raise KeyboardInterrupt
        return Interrupts()

    with pytest.raises(KeyboardInterrupt):
        run_episode(BalanceEnv("ideal", record=True), factory, seed=0, index=0)


def test_sim_unstable_scores_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def blows_up(self: BalanceEnv) -> bool:
        calls["n"] += 1
        return calls["n"] > 37  # the 38th substep: inside control step 7

    monkeypatch.setattr(BalanceEnv, "_unstable", blows_up)
    run = run_episode(BalanceEnv("ideal", record=True), pid, seed=0, index=0)
    assert run.result.error is not None
    assert (run.result.error.code, run.result.error.step) == ("sim_unstable", 8)
    assert run.result.score == 0.0
    assert not run.result.success
    assert run.trajectory.meta["termination"]["reason"] == "sim_unstable"
    assert run.trajectory.epilogue_qpos.shape[0] == 0  # nothing to show after a blow-up


# ------------------------------------------------------------------ private seeds (B, §8)


def test_private_seeds_use_the_salt() -> None:
    Recorder.seeds.clear()
    Recorder.specs.clear()
    salt = 2**127 + 12345
    p = replace(protocol("ideal", 2), seed_set="private", salt=salt)
    evaluate(p, Recorder, INFO)
    assert Recorder.seeds == [controller_seed(s, salt) for s in p.seeds]
    assert not set(Recorder.seeds) & {controller_seed(s, 0) for s in p.seeds}


def test_private_seeds_never_save_trajectories(tmp_path: Path) -> None:
    p = replace(protocol("ideal", 2), seed_set="private", salt=99)
    results = evaluate(p, pid, INFO, traj_dir=tmp_path / "traj")
    assert not (tmp_path / "traj").exists() or not any((tmp_path / "traj").iterdir())
    assert results.episodes is None
    assert results.aggregate.episodes == 2
    text = results.model_dump_json()
    assert all(f'"seed":{s}' not in text.replace(" ", "") for s in p.seeds)


def test_run_episode_can_hide_the_seed() -> None:
    run = run_episode(BalanceEnv("ideal", record=True), pid, seed=7, index=0, report_seed=False)
    assert run.result.seed is None
    assert run_episode(BalanceEnv("ideal", record=True), pid, seed=7, index=0).result.seed == 7


# ------------------------------------------------------------------ epilogue (C)


@pytest.mark.parametrize(
    ("reason", "has_error", "expected"),
    [
        ("fell", False, True),
        ("floor_contact", False, True),
        ("timeout", False, False),
        ("timeout", True, True),
        ("fell", True, True),
        ("sim_unstable", True, False),
        ("sim_unstable", False, False),
        ("drift", False, False),
    ],
)
def test_wants_epilogue(reason: str, has_error: bool, expected: bool) -> None:
    assert wants_epilogue(reason, has_error) is expected


def test_surviving_episode_has_no_epilogue() -> None:
    run = run_episode(BalanceEnv("ideal", record=True), pid, seed=0, index=0)
    assert run.trajectory.meta["termination"]["reason"] == "timeout"
    assert run.trajectory.epilogue_qpos.shape == (0, run.trajectory.qpos.shape[1])


def test_fallen_and_errored_episodes_get_a_one_second_epilogue() -> None:
    fell = run_episode(BalanceEnv("ideal", record=True), lambda spec: Zero(), seed=0, index=0)
    assert fell.trajectory.meta["termination"]["reason"] == "fell"
    assert fell.trajectory.epilogue_qpos.shape[0] == 100
    errored = run_episode(
        BalanceEnv("ideal", record=True),
        lambda spec: Explodes(5, RuntimeError("x")),
        seed=0,
        index=0,
    )
    assert errored.trajectory.epilogue_qpos.shape[0] == 100


# ------------------------------------------------------------------ aggregates (E)


def episode(index: int, score: float, success: bool, upright_s: float) -> EpisodeResult:
    return EpisodeResult(
        index=index,
        seed=index,
        success=success,
        score=score,
        metrics={"time_upright_s": upright_s},
        error=None,
    )


def test_summarize_divides_by_the_expected_episode_count() -> None:
    present = [episode(0, 90.0, True, 10.0), episode(1, 60.0, False, 4.0)]
    agg = summarize(present, expected=4)
    assert agg.score == 37.5  # missing episodes count as 0
    assert agg.success_rate == 0.25
    assert agg.mean_time_upright_s == 7.0  # over the episodes that are present
    assert (agg.episodes, agg.errors) == (2, 0)


def test_summarize_uses_exact_summation() -> None:
    present = [episode(i, v, False, 1.0) for i, v in enumerate([1e-16, 1.0, -1e-16 + 1e-16])]
    assert summarize(present, expected=3).score == math.fsum([1e-16, 1.0, 0.0]) / 3


def test_summarize_rejects_more_episodes_than_expected() -> None:
    with pytest.raises(ValueError, match="expected"):
        summarize([episode(0, 1.0, True, 1.0), episode(1, 1.0, True, 1.0)], expected=1)


def test_summarize_of_nothing_is_zero() -> None:
    agg = summarize([], expected=3)
    assert (agg.score, agg.success_rate, agg.mean_time_upright_s, agg.episodes) == (0, 0, 0, 0)


# ------------------------------------------------------------------ observation path (F)


def test_controllers_receive_controller_obs_of_the_recorded_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def spy(obs_arr: Any, t: float) -> BalancerObs:
        calls.append(t)
        return controller_obs(obs_arr, t)

    seen: list[BalancerObs] = []

    class Sees(Zero):
        def act(self, obs: BalancerObs) -> tuple[float, float]:
            seen.append(obs)
            return PIDController().act(obs)

    monkeypatch.setattr(evaluate_module, "controller_obs", spy)
    run = run_episode(BalanceEnv("clean", record=True), lambda spec: Sees(), seed=1, index=0)
    traj = run.trajectory
    assert len(calls) == len(seen) == traj.steps
    for k, obs in enumerate(seen):
        assert obs.to_wire() == controller_obs(traj.obs[k], k * 0.01).to_wire()
        assert BalancerObs.from_wire(obs.to_wire()) == obs


# ------------------------------------------------------------------ one bundle per protocol (I)


def test_evaluate_runs_on_the_protocols_challenge_object(tmp_path: Path) -> None:
    root = tmp_path / "balance-v1"
    shutil.copytree(challenge_dir("balance-v1"), root)
    yaml_path = root / "challenge.yaml"
    yaml_path.write_text(yaml_path.read_text().replace("e_ref_j: 13.3", "e_ref_j: 0.001"))
    base = protocol("clean", 1)
    custom = replace(base, challenge=load_challenge_dir(root))
    a = evaluate(base, pid, INFO)
    b = evaluate(custom, pid, INFO)
    assert a.episodes is not None
    assert b.episodes is not None
    assert b.episodes[0].score < a.episodes[0].score  # scored with the custom E_ref
    assert b.protocol_hash != a.protocol_hash
