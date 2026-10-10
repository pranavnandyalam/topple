"""Re-simulate an episode from its trajectory and check it reproduces exactly.

The env is a pure function of (challenge, scenario, command sequence): physics never depends on
sensor noise or on controller internals. So a trusted stage can verify an untrusted trajectory by
replaying its commands open loop, without running the submission's code, and re-score it.
Bit-exactness holds on the same platform and MuJoCo version (docs/contract.md §14).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topple.envs.balance import BalanceEnv
from topple.traj import Trajectory


@dataclass(frozen=True)
class Verification:
    ok: bool
    steps_match: bool
    max_abs_qpos_diff: float
    reason: str


def replay(traj: Trajectory) -> Trajectory:
    meta = traj.meta
    env = BalanceEnv(meta["preset"], challenge=meta["challenge"], record=True)
    try:
        env.reset(options={"scenario": meta["scenario"]})
        for k in range(traj.steps):
            _, _, terminated, truncated, _ = env.step(traj.ctrl[k])
            if terminated or truncated:
                break
        return env.trajectory()
    finally:
        env.close()


def verify_episode(traj: Trajectory) -> Verification:
    replayed = replay(traj)
    if replayed.steps != traj.steps:
        return Verification(
            False,
            False,
            float("inf"),
            f"replay ran {replayed.steps} steps, trajectory has {traj.steps}",
        )
    diff = float(np.max(np.abs(replayed.qpos - traj.qpos))) if traj.qpos.size else 0.0
    if not np.array_equal(replayed.qpos, traj.qpos):
        return Verification(False, True, diff, "qpos differs from the re-simulation")
    return Verification(True, True, 0.0, "ok")
