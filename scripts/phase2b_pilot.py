"""The waypoint pilot's checks (docs/design/phase-2.md §8.7 step 4): item 6, the CPU budget and
the budget rule, from one from-scratch run (``ppo_waypoint.yaml``, seed 0).

1. **CPU budget.** waypoint ``max_cpu_s`` = 3 x the pilot's ``cpu_s`` (§1), rounded up.
2. **Budget rule** (§8.8). From ``selection_scores.npz``: ``paired_bootstrap(mean of the evals
   labelled nearest 5.5M and the final, mean of the evals nearest 3.5M and 4.0M)`` per seed; a
   lower bound > 0 raises ``total_timesteps`` to 8,000,000 for both arms.
3. **Item 6** (§8.9). The run's best checkpoint, exported ``--report-only``, against the
   reference driver and the creeper on ``standard`` 0..399: (a) docked >= 60%; (b)
   ``paired[ppo, driver]`` lower bound > -5; (c) score > creeper's; (d) fell <= driver's fell +
   0.20. Falls split by ``fall_cause``. (e) Report only: PPO and driver falls on ``clean``
   0..399.

``heavy.sh uv run python scripts/phase2b_pilot.py runs/ppo_wp_pilot_s0 --out out/wp_pilot.json``
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SUBMISSIONS = REPO / "submissions"
DRIVER = SUBMISSIONS / "waypoint_pid_example.py"
CREEPER = SUBMISSIONS / "anchors" / "creeper.py"
SEEDS = tuple(range(400))
CPU_FACTOR = 3.0
LATE_LABELS = (5_500_000,)  # with the final eval
MID_LABELS = (3_500_000, 4_000_000)
RAISED_TIMESTEPS = 8_000_000
MIN_DOCKED = 0.60
MIN_PAIRED_LO = -5.0
FALL_MARGIN = 0.20
CHUNK = 25


def nearest(steps: Sequence[int], label: int) -> int:
    """The index of the eval whose step is nearest ``label`` (ties to the earlier)."""
    return min(range(len(steps)), key=lambda i: (abs(steps[i] - label), i))


def budget_rule(scores: np.ndarray, steps: Sequence[int]) -> dict[str, Any]:
    from topple.stats import paired_bootstrap

    late = sorted({nearest(steps, label) for label in LATE_LABELS} | {len(steps) - 1})
    mid = sorted({nearest(steps, label) for label in MID_LABELS})
    mean, lo, hi = paired_bootstrap(scores[late].mean(axis=0), scores[mid].mean(axis=0))
    return {
        "late_steps": [int(steps[i]) for i in late],
        "mid_steps": [int(steps[i]) for i in mid],
        "paired": {"mean": mean, "lo": lo, "hi": hi},
        "total_timesteps": RAISED_TIMESTEPS if lo > 0 else None,
    }


def episodes(ref: Any, protocol: Any, pool: Executor) -> list[Any]:
    """Every episode of ``protocol`` for ``ref``, in seed order."""
    from topple.pool import submit_episodes_with_rows

    n = len(protocol.seeds)
    chunks = [range(s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]
    futures = [submit_episodes_with_rows(pool, ref, protocol, c) for c in chunks]
    rows = list(itertools.chain.from_iterable(f.result() for f in futures))
    if [r.result.index for r in rows] != list(range(n)):
        raise RuntimeError("the pool returned episodes out of seed order")
    return [r.result for r in rows]


def summary(eps: Sequence[Any]) -> dict[str, Any]:
    n = len(eps)
    errors = sum(e.error is not None for e in eps)
    metrics = [e.metrics or {} for e in eps]
    falls = Counter(str(m.get("fall_cause")) for m in metrics if m.get("fell") is True)
    return {
        "n": n,
        "errors": errors,
        "score": math.fsum(e.score for e in eps) / n,
        "scores": [e.score for e in eps],
        "docked": sum(m.get("docked") is True for m in metrics) / n,
        "fell": sum(m.get("fell") is True for m in metrics) / n,
        "falls_by_cause": dict(sorted(falls.items())),
    }


def item6(ppo: dict[str, Any], driver: dict[str, Any], creeper: dict[str, Any]) -> dict[str, Any]:
    from topple.stats import paired_bootstrap

    mean, lo, hi = paired_bootstrap(ppo["scores"], driver["scores"])
    checks = {
        "a_docked": ppo["docked"] >= MIN_DOCKED,
        "b_paired_lo": lo > MIN_PAIRED_LO,
        "c_above_creeper": ppo["score"] > creeper["score"],
        "d_falls": ppo["fell"] <= driver["fell"] + FALL_MARGIN + 1e-9,
        "no_errors": ppo["errors"] == 0 and driver["errors"] == 0 and creeper["errors"] == 0,
    }
    return {"paired_ppo_driver": {"mean": mean, "lo": lo, "hi": hi}, "checks": checks,
            "pass": all(checks.values())}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    from topple.challenge import load_challenge
    from topple.evaluate import EvalProtocol, cpu_isa, platform_tag
    from topple.export import export_policy, read_run_json
    from topple.pool import eval_pool, policy_ref

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    run: Path = args.run
    meta = read_run_json(run / "train_meta.json")
    assert isinstance(meta, dict)
    with np.load(run / "selection_scores.npz") as npz:
        scores, steps = np.asarray(npz["scores"]), [int(s) for s in npz["steps"]]
    report_dir = run / "report_policy"
    if not report_dir.exists():
        export_policy(run, report_dir, report_only=True)
    challenge = load_challenge("waypoint-v1")
    standard = replace(EvalProtocol.from_id("waypoint-v1/standard"), seeds=SEEDS)
    clean = replace(EvalProtocol.from_id("waypoint-v1/clean"), seeds=SEEDS)
    refs = {
        "ppo": policy_ref(report_dir, challenge=challenge),
        "driver": policy_ref(DRIVER, challenge=challenge),
        "creeper": policy_ref(CREEPER, challenge=challenge),
    }
    with eval_pool(args.workers) as pool:
        std = {name: summary(episodes(ref, standard, pool)) for name, ref in refs.items()}
        cln = {name: summary(episodes(refs[name], clean, pool)) for name in ("ppo", "driver")}
    cpu_s = float(meta["cpu_s"])
    report: dict[str, Any] = {
        "format": "topple.wp_pilot/1",
        "run": str(run),
        "git_commit": meta.get("git_commit"),
        "git_dirty": meta.get("git_dirty"),
        "platform": platform_tag(),
        "cpu_isa": cpu_isa(),
        "best": meta.get("best"),
        "flags": meta.get("flags"),
        "num_timesteps": meta.get("num_timesteps"),
        "cpu_s": cpu_s,
        "wall_clock_s": meta.get("wall_clock_s"),
        "max_cpu_s": math.ceil(CPU_FACTOR * cpu_s),
        "budget_rule": budget_rule(scores, steps),
        "standard": {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in std.items()},
        "clean_report_only": {
            k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in cln.items()
        },
        "item6": item6(std["ppo"], std["driver"], std["creeper"]),
        "selection_history": [
            {k: row.get(k) for k in ("step", "score", "eligible", "extras")}
            for row in meta.get("selection_history", [])
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    s = report["standard"]
    print(f"cpu_s {cpu_s:.0f} -> max_cpu_s {report['max_cpu_s']}")
    print(f"budget rule: {report['budget_rule']}")
    for name in ("ppo", "driver", "creeper"):
        r = s[name]
        print(f"{name}: score {r['score']:.2f} docked {r['docked']:.3f} fell {r['fell']:.3f} "
              f"errors {r['errors']} falls {r['falls_by_cause']}")  # fmt: skip
    print(f"item 6: {report['item6']}")
    print(f"clean: {report['clean_report_only']}")
    return 0 if report["item6"]["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
