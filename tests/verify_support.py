"""Shared helpers for the verification tests: honest runs saved to disk and edits of them."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from topple.contract import BalancerObs, TaskSpec
from topple.controllers.lqr import LQRController
from topple.controllers.pid import PIDController
from topple.evaluate import ControllerFactory, EvalProtocol, evaluate
from topple.results import ControllerInfo, EvalResults
from topple.traj import Trajectory, load_npz
from topple.verify import protocol_max_steps

INFO = ControllerInfo(id="test", sha256="0" * 64)


def public(preset: str, n: int) -> EvalProtocol:
    """The first ``n`` public seeds of ``balance-v1/<preset>``."""
    base = EvalProtocol.from_id(f"balance-v1/{preset}")
    return replace(base, seeds=base.seeds[:n])


def pid(_spec: TaskSpec) -> PIDController:
    return PIDController()


def lqr(_spec: TaskSpec) -> LQRController:
    return LQRController()


class WideRandom:
    """Uniform commands in [-1.3, 1.3]: exercises clipping (action != ctrl) and falls quickly."""

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, rng_seed: int) -> None:
        self._rng = random.Random(rng_seed)

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        return (self._rng.uniform(-1.3, 1.3), self._rng.uniform(-1.3, 1.3))


def wide_random(_spec: TaskSpec) -> WideRandom:
    return WideRandom()


class Zero:
    def reset(self, rng_seed: int) -> None:
        pass

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        return (0.0, 0.0)


def zero(_spec: TaskSpec) -> Zero:
    return Zero()


class Explodes(PIDController):
    """A PID controller whose ``act`` returns ``payload`` (or raises it) from step ``at`` on."""

    def __init__(self, at: int, payload: Any) -> None:
        super().__init__()
        self.at, self.payload, self.k = at, payload, 0

    def reset(self, rng_seed: int) -> None:
        super().reset(rng_seed)
        self.k = 0

    def act(self, obs: BalancerObs) -> Any:
        self.k += 1
        if self.k > self.at:
            if isinstance(self.payload, BaseException):
                raise self.payload
            return self.payload
        return super().act(obs)


def explodes(at: int, payload: Any) -> ControllerFactory:
    return lambda _spec: Explodes(at, payload)


@dataclass(frozen=True)
class Honest:
    protocol: EvalProtocol
    results: EvalResults
    uploads: dict[int, Trajectory]


def run_honest(protocol: EvalProtocol, factory: ControllerFactory, out: Path) -> Honest:
    """Evaluate in process, save every trajectory, and load the files back as untrusted uploads."""
    results = evaluate(protocol, factory, INFO, traj_dir=out)
    steps = protocol_max_steps(protocol)
    uploads = {
        i: load_npz(out / f"traj_{i:03d}.npz", max_steps=steps) for i in range(len(protocol.seeds))
    }
    return Honest(protocol, results, uploads)


def with_meta(traj: Trajectory, **changes: Any) -> Trajectory:
    return replace(traj, meta={**traj.meta, **changes})


def without_meta(traj: Trajectory, key: str) -> Trajectory:
    return replace(traj, meta={k: v for k, v in traj.meta.items() if k != key})


def truncated(traj: Trajectory, steps: int) -> Trajectory:
    """The first ``steps`` rows of every per-step array (a self-consistent shorter record)."""
    return replace(
        traj,
        qpos=traj.qpos[: steps + 1],
        qvel=traj.qvel[: steps + 1],
        obs=traj.obs[:steps],
        action=traj.action[:steps],
        ctrl=traj.ctrl[:steps],
        enc_counts=traj.enc_counts[:steps],
        gyro=traj.gyro[:steps],
        accel=traj.accel[:steps],
        push_force=traj.push_force[:steps],
        epilogue_qpos=traj.epilogue_qpos[:0],
    )
