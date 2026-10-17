"""waypoint-v1 calibration (docs/design/phase-2.md §8.7 steps 1-2).

Step 1 measures the reference driver on ``standard`` seeds 0..399 at each push ceiling
``impulse_hi`` in 0.300:0.500:0.025 and chooses, among the values with falls in [3%, 10%] and
docked >= 88%, the one closest to 6% falls (ties to the lower value); if none qualifies, 2b
stops for an owner decision. Step 2 sets E_ref = 2 x the driver's median energy at the chosen
value, rounded to 0.1 J. Every value goes into the JSON written to ``--out`` with Wilson 95%
CIs, the platform and the clean - standard driver gap.

Episodes run in a spawn process pool; each worker builds the challenge with the swept ceiling
in its standard preset (the bundle on disk is not changed), so results do not depend on the
pool. ``heavy.sh uv run python scripts/phase2b_calibrate.py --out out/wp_calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SEEDS = range(400)
IMPULSES = tuple(float(Decimal("0.300") + Decimal("0.025") * i) for i in range(9))
FALLS_BAND = (0.03, 0.10)
FALLS_TARGET = 0.06
MIN_DOCKED = 0.88
CHUNK = 25


def challenge_with(impulse_hi: float | None) -> Any:
    """waypoint-v1 with ``impulse_hi`` as its standard preset's push ceiling (``None``: as on
    disk)."""
    from topple.challenge import Challenge, load_challenge

    base: Challenge = load_challenge("waypoint-v1")
    if impulse_hi is None:
        return base
    std = base.preset("standard")
    assert std.pushes is not None
    pushes = std.pushes.model_copy(update={"impulse_ns": (std.pushes.impulse_ns[0], impulse_hi)})
    presets = tuple(
        std.model_copy(update={"pushes": pushes}) if p.name == "standard" else p
        for p in base.config.preset_defs
    )
    return replace(base, config=base.config.model_copy(update={"preset_defs": presets}))


def run_chunk(task: tuple[float | None, str, tuple[int, ...]]) -> list[dict[str, Any]]:
    """Driver episodes of ``seeds`` on ``preset`` at ``impulse_hi``: per-episode metrics."""
    from topple.evaluate import run_episode
    from topple.kinds import make_env
    from topple.loader import load_submission

    impulse_hi, preset, seeds = task
    challenge = challenge_with(impulse_hi)
    sub = load_submission(REPO / "submissions" / "waypoint_pid_example.py", challenge=challenge)
    env = make_env(challenge, preset, record=True)
    out = []
    try:
        for seed in seeds:
            run = run_episode(env, sub.factory, seed, 0)
            m = run.result.metrics or {}
            out.append(
                {
                    "seed": seed,
                    "score": run.result.score,
                    "docked": bool(m.get("docked")),
                    "fell": bool(m.get("fell")),
                    "energy_j": float(m.get("energy_j", 0.0)),
                    "error": run.result.error is not None,
                }
            )
    finally:
        env.close()
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    fell = sum(e["fell"] for e in episodes)
    docked = sum(e["docked"] for e in episodes)
    return {
        "n": n,
        "falls": fell / n,
        "falls_ci": wilson(fell, n),
        "docked": docked / n,
        "docked_ci": wilson(docked, n),
        "errors": sum(e["error"] for e in episodes),
        "mean_score": math.fsum(e["score"] for e in episodes) / n,
        "median_energy_j": statistics.median(e["energy_j"] for e in episodes),
    }


def choose(curve: dict[float, dict[str, Any]]) -> float | None:
    qualified = [
        hi
        for hi, row in curve.items()
        if FALLS_BAND[0] <= row["falls"] <= FALLS_BAND[1] and row["docked"] >= MIN_DOCKED
    ]
    if not qualified:
        return None
    return min(qualified, key=lambda hi: (abs(curve[hi]["falls"] - FALLS_TARGET), hi))


def main(argv: list[str] | None = None) -> int:
    from topple.evaluate import cpu_isa, platform_tag

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    tasks: list[tuple[float | None, str, tuple[int, ...]]] = []
    seeds = tuple(SEEDS)
    chunks = [seeds[i : i + CHUNK] for i in range(0, len(seeds), CHUNK)]
    for hi in IMPULSES:
        tasks += [(hi, "standard", c) for c in chunks]
    tasks += [(None, "clean", c) for c in chunks]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        results = list(pool.map(run_chunk, tasks))
    by: dict[tuple[float | None, str], list[dict[str, Any]]] = {}
    for (key_hi, preset, _), eps in zip(tasks, results, strict=True):
        by.setdefault((key_hi, preset), []).extend(eps)
    curve = {hi: summarize(by[(hi, "standard")]) for hi in IMPULSES}
    clean = summarize(by[(None, "clean")])
    chosen = choose(curve)
    report: dict[str, Any] = {
        "format": "topple.wp_calibration/1",
        "platform": platform_tag(),
        "cpu_isa": cpu_isa(),
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "rule": {"falls_band": FALLS_BAND, "falls_target": FALLS_TARGET, "min_docked": MIN_DOCKED},
        "curve": {f"{hi:.3f}": row for hi, row in curve.items()},
        "clean": clean,
        "impulse_hi": chosen,
    }
    if chosen is not None:
        row = curve[chosen]
        report["e_ref_j"] = round(2 * row["median_energy_j"], 1)
        report["clean_minus_standard"] = clean["mean_score"] - row["mean_score"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for hi, row in curve.items():
        print(f"{hi:.3f}: falls {row['falls']:.3f} docked {row['docked']:.3f} "
              f"score {row['mean_score']:.2f} E_med {row['median_energy_j']:.2f}")  # fmt: skip
    e_ref = report.get("e_ref_j")
    print(f"chosen impulse_hi {chosen}; E_ref {e_ref}; clean {clean['mean_score']:.2f}")
    return 0 if chosen is not None else 3


if __name__ == "__main__":
    sys.exit(main())
