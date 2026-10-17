"""The Phase 2b done-when gate (docs/design/phase-2.md §8.9), torch-free.

``uv run --no-sync python scripts/phase2b_gate.py out/wp_cal.json out/wp_clean.json
out/wp_eval.json docs/results/phase2b.json --submission submissions/ppo_waypoint
--fixture tests/data/ppo_waypoint_parity.npz --results-dir docs/results/phase2``

It judges the items it can from these files; ``phase2b_done.sh`` runs the rest (items 2 and 3
and item 9's parity test by pytest, item 9's re-export by ``cmp``, item 10 by
``phase2_ci_check.py``). Every comparison is re-read with ``read_comparison``; nothing in
``phase2b.json`` is trusted that the fresh comparisons can recompute:

1. ``wp_cal.json`` (standard 0..399: the driver, the five anchors and ``ppo_waypoint``): the
   driver's band and the five anchor requirements, at the current protocol hash; every 2b run
   trained on the current bundle.
4. ``docs/contract.md`` names each 2b term.
5. ``phase2b.json`` parses, holds 3 runs per arm and ``phase2b.md`` is exactly its rendering.
6. ``ppo_waypoint`` passes item 6 in ``wp_cal.json``; ``wp_clean.json`` (report only) and
   ``wp_eval.json`` are complete and hold it.
7. Every fine-tune run has a sanity row for its own source with survival ≥ 95%.
8. The video, its GIF and three gallery GIFs exist within their size caps and GIF sizes.
9. The submission is the shipped run's checkpoint (``onnx_sha256``) and its fixture holds.
11. Every 2b run's ``selection_protocol_hash`` and waypoint ``env_algo_version`` are current.

Exit 0 iff every item passes; 1 if one fails; 2 if an input cannot be read.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import phase2b_common as wp
from phase2b_render import render_markdown
from topple.challenge import env_algo_version, load_challenge
from topple.compare import Comparison, read_comparison
from topple.evaluate import EvalProtocol
from topple.policy.runtime import load_policy_dir

REPO: Final = Path(__file__).resolve().parents[1]
SHIPPED_ID: Final = "ppo_waypoint"
FRESH_SEEDS: Final = tuple(range(400))
MIN_SURVIVAL: Final = 0.95
CONTRACT_TERMS: Final = (
    "topple/Waypoint-v1",
    "wp_eval_v1",
    "topple.traj/2",
    "balancer-target.obs.v1",
    "TARGETS = 6",
    "fall_cause",
    "course_pts",
    "ENV_ALGO_VERSION",
    "max_steps",
)
MB: Final = 1 << 20
MEDIA: Final = {  # name: (cap in bytes, GIF side in pixels or None)
    "waypoint.mp4": (8 * MB, None),
    "waypoint.gif": (5 * MB, 480),
    "wp_gallery_1.gif": (3 * MB, 480),
    "wp_gallery_2.gif": (3 * MB, 480),
    "wp_gallery_3.gif": (3 * MB, 480),
}


class GateInputError(Exception):
    """An input the gate cannot read (exit 2)."""


def fresh_hash(preset: str) -> str:
    return replace(EvalProtocol.from_id(f"waypoint-v1/{preset}"), seeds=FRESH_SEEDS).protocol_hash()


def standard_hash() -> str:
    return fresh_hash("standard")


EXPECTED_HASHES: Final = {
    "wp_cal": lambda: fresh_hash("standard"),
    "wp_clean": lambda: fresh_hash("clean"),
    "wp_eval": lambda: EvalProtocol.from_id("wp_eval_v1").protocol_hash(),
}
RUNS_FILES: Final = {"standard": "wp_cal", "clean": "wp_clean", "wp_eval_v1": "wp_eval"}


def gif_size(path: Path) -> tuple[int, int] | None:
    head = path.read_bytes()[:10]
    if head[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    return int.from_bytes(head[6:8], "little"), int.from_bytes(head[8:10], "little")


def item1(cal: Comparison, rep: wp.Phase2bReport) -> list[str]:
    bundle = load_challenge("waypoint-v1").bundle_hash()
    out = [k for k, ok in wp.band_and_anchors(cal).items() if not ok]
    if cal.protocol_hash != standard_hash():
        out.append("wp_cal is not standard 0..399 at the current protocol hash")
    out += [f"{r.summary.run} trained on another bundle" for r in rep.runs
            if r.summary.bundle_hash != bundle]  # fmt: skip
    return out


def item4(contract: str) -> list[str]:
    return [f"docs/contract.md does not name {t!r}" for t in CONTRACT_TERMS if t not in contract]


def item5(rep: wp.Phase2bReport, md: str | None) -> list[str]:
    out = []
    for arm in ("scratch", "finetune"):
        if sum(r.arm == arm for r in rep.runs) != 3:
            out.append(f"not 3 {arm} runs")
    if [r.summary.run for r in rep.runs] != [f.run for f in rep.fresh]:
        out.append("the fresh rows are not the runs, in order")
    if md is None:
        out.append("phase2b.md is missing")
    elif md != render_markdown(rep):
        out.append("phase2b.md is not the rendering of phase2b.json")
    return out


def item6(cal: Comparison, clean: Comparison, ev: Comparison, policy_sha256: str) -> list[str]:
    """Item 6 on ``wp_cal``; every comparison must be its own protocol (standard 0..399, clean
    0..399, wp_eval_v1) and its ``ppo_waypoint`` entry the submission's policy bytes."""
    result = wp.item6(cal, SHIPPED_ID)
    out = [k for k, ok in result["checks"].items() if not ok]
    for name, cmp in (("wp_cal", cal), ("wp_clean", clean), ("wp_eval", ev)):
        if cmp.protocol_hash != EXPECTED_HASHES[name]():
            out.append(f"{name} is not its protocol at the current hash")
        try:
            if wp.entry(cmp, SHIPPED_ID).policy_sha256 != policy_sha256:
                out.append(f"{name}: {SHIPPED_ID} is not the submission's policy")
            wp.entry(cmp, wp.DRIVER_ID)
        except wp.Phase2bError as exc:
            out.append(f"{name}: {exc}")
    return out


def item5_rows(rep: wp.Phase2bReport, runs_dir: Path) -> list[str]:
    """The report's per-run rows equal their recomputation from the committed comparisons
    (``phase2b/runs_*.json``), and its shipping attempts replay §8.8: the candidates in waypoint
    ``rank_key`` order, each failing item 6 exactly as recomputed, the shipped run first to pass."""
    from phase2_common import candidates
    from phase2_results import core

    cmps = {p: read_comparison(runs_dir / f"runs_{p}.json") for p in RUNS_FILES}
    out = []
    for p, cmp in cmps.items():
        if cmp.protocol_hash != EXPECTED_HASHES[RUNS_FILES[p]]():
            out.append(f"runs_{p}.json is not its protocol at the current hash")
    for row in rep.fresh:
        for p, cmp in cmps.items():
            got = wp.rates(wp.entry(cmp, row.run)).to_json()
            if got != getattr(row, p).model_dump(mode="json"):
                out.append(f"{row.run}: its {p} row differs from runs_{p}.json")
        if wp.item6(cmps["standard"], row.run) != row.item6.model_dump(mode="json", by_alias=True):
            out.append(f"{row.run}: its item 6 differs from runs_standard.json")
    ranked = candidates(
        [core(r.summary.model_dump(mode="json")) for r in rep.runs], kind="waypoint"
    )
    tried = [a.run for a in rep.shipped.attempts]
    if tried != ranked[: len(tried)]:
        out.append("the shipping attempts are not the candidates in rank order")
    for a in rep.shipped.attempts:
        failed6 = [k for k, ok in wp.item6(cmps["standard"], a.run)["checks"].items() if not ok]
        if failed6 and list(a.failed) != failed6:
            out.append(f"{a.run}: its recorded item 6 failures are not the recomputed ones")
    passed = [a.run for a in rep.shipped.attempts if not a.failed]
    if passed != ([rep.shipped.run] if rep.shipped.run else []) or (
        passed and tried[-1] != passed[0]
    ):
        out.append("the shipped run is not the first attempt that passed")
    return out


def item7(rep: wp.Phase2bReport) -> list[str]:
    out = []
    rows = {s.run: s for s in rep.sanity}
    for r in rep.runs:
        if r.arm != "finetune":
            continue
        s = rows.get(r.summary.run)
        src = r.summary.init_source
        if s is None or src is None:
            out.append(f"{r.summary.run}: no sanity row or no init_source")
        elif s.source != src.model_dump(mode="json"):
            out.append(f"{r.summary.run}: its sanity row checks another source")
        elif s.survival < MIN_SURVIVAL:
            out.append(f"{r.summary.run}: survival {s.survival:.2f} < {MIN_SURVIVAL}")
    return out


def item8(results_dir: Path) -> list[str]:
    out = []
    for name, (cap, side) in MEDIA.items():
        path = results_dir / name
        if not path.is_file():
            out.append(f"{name} is missing")
            continue
        if path.stat().st_size > cap:
            out.append(f"{name} is over {cap // MB} MB")
        if side is not None and gif_size(path) != (side, side):
            out.append(f"{name} is not a {side}x{side} GIF")
        if side is None and path.read_bytes()[4:8] != b"ftyp":
            out.append(f"{name} is not an MP4 (no ftyp box)")
    return out


def item9(rep: wp.Phase2bReport, submission: Path, fixture: Path) -> list[str]:
    out = wp.parity_failures(fixture, submission)
    policy = load_policy_dir(submission, challenge=load_challenge("waypoint-v1"))
    if rep.shipped.run is None or policy.policy_sha256 != rep.shipped.onnx_sha256:
        out.append("the submission is not the shipped run's checkpoint")
    return out


def item11(rep: wp.Phase2bReport) -> list[str]:
    want_hash = wp.selection_protocol().protocol_hash()
    want_algo = env_algo_version("waypoint")
    out = []
    for r in rep.runs:
        s = r.summary
        if s.selection_protocol_hash != want_hash:
            out.append(f"{s.run}: another selection protocol")
        if s.env_algo_version != want_algo:
            out.append(f"{s.run}: waypoint env {s.env_algo_version}, now {want_algo}")
    return out


def read_inputs(args: argparse.Namespace) -> tuple[list[Comparison], wp.Phase2bReport, str]:
    try:
        cmps = [read_comparison(p) for p in (args.cal, args.clean, args.eval)]
        rep = wp.parse_report(args.report.read_bytes())
        contract = (REPO / "docs/contract.md").read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise GateInputError(str(exc).splitlines()[0]) from exc
    return cmps, rep, contract


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cal", type=Path)
    p.add_argument("clean", type=Path)
    p.add_argument("eval", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        (cal, clean, ev), rep, contract = read_inputs(args)
    except GateInputError as exc:
        print(f"phase2b_gate: {exc}", file=sys.stderr)
        return 2
    md_path = args.report.with_name(wp.PHASE2B_MD)
    md = md_path.read_text(encoding="utf-8") if md_path.is_file() else None
    items: dict[str, Callable[[], list[str]]] = {
        "1 challenge frozen": lambda: item1(cal, rep),
        "4 contract": lambda: item4(contract),
        "5 results": lambda: item5(rep, md) + item5_rows(rep, args.report.with_name("phase2b")),
        "6 learnability": lambda: item6(cal, clean, ev, rep.shipped.onnx_sha256 or ""),
        "7 fine-tune sanity": lambda: item7(rep),
        "8 video": lambda: item8(args.results_dir),
        "9 export": lambda: item9(rep, args.submission, args.fixture),
        "11 binding": lambda: item11(rep),
    }
    failed = 0
    for name, check in items.items():
        try:
            problems = check()
        except (wp.Phase2bError, OSError, ValueError) as exc:
            problems = [str(exc).splitlines()[0]]
        failed += bool(problems)
        print(f"item {name}: " + ("pass" if not problems else "FAIL: " + "; ".join(problems)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
