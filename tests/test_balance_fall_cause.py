"""balance-v1 records ``fall_cause`` (docs/design/phase-2.md §8.1, §8.5: an additive metric;
Pranav's decision 2026-10-17). It is ``none`` unless the episode fell, then ``push`` or
``self`` by the 1.5 s window; every other metric, the score and the success are unchanged."""

from __future__ import annotations

from dataclasses import replace

from topple.challenge import load_challenge
from topple.controllers.pid import PIDController
from topple.evaluate import EvalProtocol, iter_episodes
from topple.kinds import score_trajectory
from topple.scoring import episode_metrics
from topple.waypoint import fall_cause

BALANCE = load_challenge("balance-v1")


def test_every_balance_episode_records_its_fall_cause() -> None:
    proto = replace(EvalProtocol.from_id("balance-v1/standard"), seeds=tuple(range(1000, 1012)))
    causes = []
    for run in iter_episodes(proto, lambda _spec: PIDController()):
        m = run.result.metrics
        assert m is not None
        meta = run.trajectory.meta
        pushes = [int(p["start_step"]) for p in meta["scenario"]["pushes"]]
        fell = m["termination"] in ("fell", "floor_contact")
        want = fall_cause(str(m["termination"]), int(m["steps"]) if fell else None, pushes,
                          float(meta["control_dt"]))  # fmt: skip
        assert m["fall_cause"] == want
        base = episode_metrics(run.trajectory, BALANCE.params.motor).as_dict()
        assert {k: v for k, v in m.items() if k != "fall_cause"} == base
        scored = score_trajectory(run.trajectory, BALANCE, error=False)
        assert (scored.score, scored.success) == (run.result.score, run.result.success)
        causes.append(m["fall_cause"])
    assert "none" in causes
    assert "push" in causes  # PID falls to pushes on standard (seed 1002 is a golden fall)
