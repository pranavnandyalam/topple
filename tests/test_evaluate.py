import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from topple.contract import BalancerObs, TaskSpec
from topple.controllers.pid import PIDController
from topple.envs.balance import BalanceEnv
from topple.evaluate import EvalProtocol, evaluate, run_episode
from topple.results import ControllerInfo, ErrorInfo, EvalResults, read_results, write_results
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
            if isinstance(self.payload, Exception):
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
    for i, episode in enumerate(results.episodes):
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
