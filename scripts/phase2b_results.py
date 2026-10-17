"""The Phase 2b results package (docs/design/phase-2.md §8.8, §8.9 items 5-7, 11).

``heavy.sh uv run python scripts/phase2b_results.py --scratch runs/ppo_wp_s{0,1,2} \\
    --finetune runs/ppo_wpft_s{0,1,2} --out docs/results``

1. Reads each run's ``train_meta.json`` into its committed summary (2a's ``RunSummary``) and
   refuses a run whose ``selection_protocol_hash`` or waypoint ``env_algo_version`` is not the
   current one (item 11).
2. Exports every run's best checkpoint ``--report-only`` into a copy named after the run, and
   compares the driver, the creeper and those six on ``waypoint-v1/standard`` 0..399, on
   ``waypoint-v1/clean`` 0..399 (report only) and on ``wp_eval_v1``.
3. Ships per §8.8: the candidates in waypoint ``rank_key`` order; each is judged on item 6 from
   the standard comparison, exported to ``submissions/ppo_waypoint`` with its parity fixture and
   judged on item 9's parity half; the first that passes both ships.
4. Runs ``surgery.sanity_check`` on each fine-tune source (item 7).
5. Writes ``phase2b.json`` (``topple.phase2b/1``, validated by ``phase2b_common.parse_report``),
   the three comparisons under ``phase2b/`` and ``phase2b.md``.

Every number comes from a run, an evaluation or the pilot file; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any, Final

import phase2b_common as wp
from phase2_common import REPO, candidates
from phase2_results import ResultsError, core, export_report_copy, run_summary, topple
from phase2b_render import render_markdown
from topple.challenge import env_algo_version, load_challenge
from topple.compare import Comparison, compare, to_json, to_markdown
from topple.evaluate import EvalProtocol
from topple.export import read_run_json

SHIPPED: Final = "submissions/ppo_waypoint"
FIXTURE_NAME: Final = "ppo_waypoint"
FIXTURE: Final = f"tests/data/{FIXTURE_NAME}_parity.npz"
DRIVER: Final = "submissions/waypoint_pid_example.py"
CREEPER: Final = "submissions/anchors/creeper.py"
ANCHORS: Final = tuple(f"submissions/anchors/{a}.py" for a in wp.ANCHOR_IDS if a != "creeper")
FRESH_SEEDS: Final = tuple(range(400))
SANITY_SEEDS: Final = range(20)
PILOT: Final = "docs/results/phase2b/pilot/wp_pilot_darwin.json"
SUBDIR: Final = "phase2b"
PROTOCOLS: Final = {
    "standard": "waypoint-v1/standard",
    "clean": "waypoint-v1/clean",
    "wp_eval_v1": "wp_eval_v1",
}


def protocol(name: str) -> EvalProtocol:
    p = EvalProtocol.from_id(PROTOCOLS[name])
    return p if name == "wp_eval_v1" else replace(p, seeds=FRESH_SEEDS)


# --- runs -----------------------------------------------------------------------------------------


def read_runs(scratch: Sequence[Path], finetune: Sequence[Path]) -> list[dict[str, Any]]:
    """Each run's summary with its arm; item 11's binding is checked here."""
    want_hash = wp.selection_protocol().protocol_hash()
    want_algo = env_algo_version("waypoint")
    out = []
    for arm, dirs in (("scratch", scratch), ("finetune", finetune)):
        for run_dir in dirs:
            s = run_summary(run_dir)
            if s["selection_protocol_hash"] != want_hash:
                raise ResultsError(f"{run_dir.name}: trained on another selection protocol")
            if s["env_algo_version"] != want_algo:
                raise ResultsError(
                    f"{run_dir.name}: trained on waypoint env {s['env_algo_version']}"
                )
            if (arm == "finetune") != ("init_source" in s):
                raise ResultsError(f"{run_dir.name}: the {arm} arm and its init_source disagree")
            out.append({"arm": arm, "summary": s})
    return out


def first_eligible(summary: Mapping[str, Any]) -> int | None:
    return next((r["step"] for r in summary["selection_history"] if r["eligible"]), None)


def pretraining_steps(summary: Mapping[str, Any]) -> int:
    """The fine-tune source run's whole budget (its best is known only after it ends)."""
    meta = read_run_json(REPO / summary["init_source"]["run"] / "train_meta.json")
    if not isinstance(meta, dict) or not isinstance(meta.get("num_timesteps"), int):
        raise ResultsError(f"{summary['init_source']['run']}: no num_timesteps")
    return int(meta["num_timesteps"])


def headline(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """§8.8: env steps to the first eligible selection eval, median per arm; the fine-tune arm
    also with its source's pretraining steps added."""
    out: dict[str, Any] = {}
    for arm in ("scratch", "finetune"):
        rows = [r["summary"] for r in runs if r["arm"] == arm]
        steps = [first_eligible(s) for s in rows]
        row: dict[str, Any] = {"first_eligible_steps": steps}
        known = [s for s in steps if s is not None]
        row["median"] = median(known) if len(known) == len(steps) else None
        if arm == "finetune":
            pre = [pretraining_steps(s) for s in rows]
            row["pretraining_steps"] = pre
            row["median_with_pretraining"] = (
                median(s + p for s, p in zip(known, pre, strict=True))
                if len(known) == len(steps)
                else None
            )
        out[arm] = row
    return out


# --- evaluations ----------------------------------------------------------------------------------


def evaluate_all(
    runs: Sequence[Mapping[str, Any]], run_dirs: Mapping[str, Path], work: Path, workers: int
) -> dict[str, Comparison]:
    policies = work / "policies"
    policies.mkdir(parents=True)
    copies = [export_report_copy(run_dirs[r["summary"]["run"]], policies) for r in runs]
    driver, creeper = REPO / DRIVER, REPO / CREEPER
    return {
        "standard": compare(
            [driver, creeper, *(REPO / a for a in ANCHORS), *copies],
            protocol("standard"),
            workers=workers,
        ),
        "clean": compare([driver, *copies], protocol("clean"), workers=workers),
        "wp_eval_v1": compare([driver, creeper, *copies], protocol("wp_eval_v1"), workers=workers),
    }


def ship(
    runs: Sequence[Mapping[str, Any]], run_dirs: Mapping[str, Path], standard: Comparison
) -> dict[str, Any]:
    """§8.8 shipping: the first candidate passing item 6 and item 9's parity half."""
    ranked = candidates([core(r["summary"]) for r in runs], kind="waypoint")
    arm = {r["summary"]["run"]: r["arm"] for r in runs}
    attempts = []
    for run in ranked:
        failed = [k for k, ok in wp.item6(standard, run)["checks"].items() if not ok]
        if not failed:
            topple("export", str(run_dirs[run]), "--out", str(REPO / SHIPPED),
                   "--test-fixture", FIXTURE_NAME, "--replace")  # fmt: skip
            failed = wp.parity_failures(REPO / FIXTURE, REPO / SHIPPED)
        attempts.append({"run": run, "failed": failed})
        if not failed:
            best = next(r["summary"]["best"] for r in runs if r["summary"]["run"] == run)
            return {"run": run, "arm": arm[run], "step": best["step"],
                    "onnx_sha256": best["onnx_sha256"], "attempts": attempts}  # fmt: skip
    return {"run": None, "arm": None, "step": None, "onnx_sha256": None, "attempts": attempts}


def sanity(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Item 7 for each fine-tune run's source, rebuilt exactly as training built it."""
    from topple.train.config import PpoTrainConfig, load_train_config
    from topple.train.surgery import extended_init, sanity_check

    out = []
    for r in runs:
        if r["arm"] != "finetune":
            continue
        s = r["summary"]
        cfg = load_train_config(REPO / "configs/train/ppo_waypoint_ft.yaml", seed=s["seed"])
        assert isinstance(cfg, PpoTrainConfig)
        ext = extended_init(cfg, cwd=REPO)
        source = ext.source.to_json()
        if source != s["init_source"]:
            raise ResultsError(f"{s['run']}: its init source no longer resolves to {source}")
        out.append({"run": s["run"], "source": source,
                    "survival": sanity_check(ext, SANITY_SEEDS)})  # fmt: skip
    return out


# --- the report -----------------------------------------------------------------------------------


def arms(runs: Sequence[Mapping[str, Any]], cmps: Mapping[str, Comparison]) -> dict[str, Any]:
    """Per arm, mean and sd over its seeds of the fresh standard score and fell rate, and the
    median of its seeds' median energies over the driver's (§8.8)."""
    std = cmps["standard"]
    drv_energy = wp.rates(wp.entry(std, wp.DRIVER_ID)).median_energy_j
    out: dict[str, Any] = {}
    for arm in ("scratch", "finetune"):
        rs = [wp.rates(wp.entry(std, r["summary"]["run"])) for r in runs if r["arm"] == arm]
        score, score_sd = wp.mean_sd([r.score for r in rs])
        fell, fell_sd = wp.mean_sd([r.fell for r in rs])
        energy = median(r.median_energy_j for r in rs) / drv_energy
        out[arm] = {"score_mean": score, "score_sd": score_sd, "fell_mean": fell,
                    "fell_sd": fell_sd, "energy_vs_driver": energy}  # fmt: skip
    out["better_arm"] = max(("scratch", "finetune"), key=lambda a: out[a]["score_mean"])
    return out


def per_run(runs: Sequence[Mapping[str, Any]], cmps: Mapping[str, Comparison]) -> list[Any]:
    rows = []
    for r in runs:
        name = r["summary"]["run"]
        rows.append({
            "run": name,
            "arm": r["arm"],
            "first_eligible_step": first_eligible(r["summary"]),
            "standard": wp.rates(wp.entry(cmps["standard"], name)).to_json(),
            "clean": wp.rates(wp.entry(cmps["clean"], name)).to_json(),
            "wp_eval_v1": wp.rates(wp.entry(cmps["wp_eval_v1"], name)).to_json(),
            "item6": wp.item6(cmps["standard"], name),
        })  # fmt: skip
    return rows


def build_report(
    runs: Sequence[Mapping[str, Any]],
    cmps: Mapping[str, Comparison],
    shipped: Mapping[str, Any],
    sanity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    std = cmps["standard"]
    pilot = json.loads((REPO / PILOT).read_text())
    rows = per_run(runs, cmps)
    names = {r["summary"]["run"] for r in runs}
    exchange = (
        wp.exchange_rate(wp.entry(std, shipped["run"])) if shipped["run"] is not None else None
    )
    return {
        "format": wp.PHASE2B_FORMAT,
        "e_ref_j": load_challenge("waypoint-v1").config.score.e_ref_j,
        "runs": [{"arm": r["arm"], "summary": r["summary"]} for r in runs],
        "fresh": rows,
        "baselines": {
            p: {e.id: wp.rates(e).to_json() for e in cmp.entries if e.id not in names}
            for p, cmp in cmps.items()
        },
        "item6_passes": sum(r["item6"]["pass"] for r in rows),
        "headline": headline(runs),
        "arms": arms(runs, cmps),
        "shipped": dict(shipped),
        "sanity": list(sanity_rows),
        "exchange_rate": None if exchange is None or math.isnan(exchange) else exchange,
        "pilot": {
            k: pilot[k]
            for k in (
                "run",
                "git_commit",
                "best",
                "flags",
                "num_timesteps",
                "cpu_s",
                "max_cpu_s",
                "budget_rule",
                "item6",
                "clean_report_only",
                "standard",
            )
        },
    }


def write(out: Path, report: Mapping[str, Any], cmps: Mapping[str, Comparison]) -> None:
    sub = out / SUBDIR
    sub.mkdir(parents=True, exist_ok=True)
    for name, cmp in cmps.items():
        (sub / f"runs_{name}.json").write_text(to_json(cmp), encoding="utf-8")
        (sub / f"runs_{name}.md").write_text(to_markdown(cmp), encoding="utf-8")
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    wp.parse_report(data.encode())
    (out / wp.PHASE2B_JSON).write_text(data, encoding="utf-8")
    (out / wp.PHASE2B_MD).write_text(render_markdown(wp.parse_report(data.encode())), "utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scratch", type=Path, nargs=3, required=True)
    p.add_argument("--finetune", type=Path, nargs=3, required=True)
    p.add_argument("--out", type=Path, default=REPO / "docs/results")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    try:
        runs = read_runs(args.scratch, args.finetune)
        run_dirs = {d.name: d for d in (*args.scratch, *args.finetune)}
        with tempfile.TemporaryDirectory() as tmp:
            cmps = evaluate_all(runs, run_dirs, Path(tmp), args.workers)
        shipped = ship(runs, run_dirs, cmps["standard"])
        report = build_report(runs, cmps, shipped, sanity(runs))
        write(args.out, report, cmps)
    except (ResultsError, wp.Phase2bError) as exc:
        print(f"phase2b_results: {exc}", file=sys.stderr)
        return 2
    print(f"shipped {shipped['run']} ({shipped['arm']}); attempts {shipped['attempts']}")
    print(f"item 6: {report['item6_passes']} of {len(report['fresh'])} runs pass")
    return 0 if shipped["run"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
