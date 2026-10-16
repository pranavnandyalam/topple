"""The Phase 2a results package (docs/design/phase-2.md §5.1, §2.5, §4 "Ubuntu evidence").

Build (the default)::

    heavy.sh uv run python scripts/phase2_results.py --ppo runs/ppo_s{0,1,2} \\
        --sac runs/sac_s{0,1,2} --ablation runs/ablate_nopenalty --out docs/results

1. Reads each run's ``train_meta.json`` into its committed summary (exactly the §5.1 fields,
   validated as ``RunSummary``). The ``attempts`` field comes from ``runs/<name>.attempt*``.
2. Exports every PPO and SAC run's best checkpoint ``--report-only``, into a copy named after
   the run, so each is its own comparison entry.
3. Compares PID, LQR and those policies on eval_v1 and on the calibration seeds 0..199.
4. Ships per §2.5. It tries the candidates in rank order. Each one is exported to
   ``submissions/ppo_balance`` with its parity fixture and evaluated on eval_v1. It is judged
   by ``phase2_gate``'s own ``SHIP_GATES`` checks, so the choice is exactly what G7 replays.
5. Compares again with the shipped directory in place. The policy bytes are the same, so the
   numbers are too.
6. Writes ``phase2.json`` (validated by ``parse_phase2``) and renders the media and
   ``phase2.md`` with ``phase2_render``.

``--fetch-ci`` (after pushing) downloads the ubuntu evidence of the newest successful ``push``
run of ``ci.yml`` on this branch. That run's commit must be an ancestor of HEAD, with no
``EVIDENCE_PATHS`` changed since, and its ``results.json`` must hold the shipped policy. The
evidence goes to ``phase2/ci_ubuntu/``, with ``source.json``, and ``phase2.md`` is re-rendered.

Nothing here writes a number by hand: every value comes from a run, an evaluation or the spec's
pilot table (``PILOT``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from phase2_common import (
    CI_SOURCE_FILE,
    EVIDENCE_PATHS,
    FIXTURE_DIR,
    LQR_SUBMISSION,
    MAX_ATTEMPTS,
    PHASE2_FORMAT,
    PID_SUBMISSION,
    REPO,
    SHIP_GATES,
    SHIPPED_SUBMISSION,
    UBUNTU_GATES,
    CommandError,
    Fresh,
    FreshRow,
    G11Row,
    Phase2Report,
    RetryAttempt,
    RunCore,
    RunEntry,
    RunSummary,
    ShipAttempt,
    Shipped,
    calibration_protocol,
    candidates,
    choose_shipped,
    eval_protocol,
    git_env,
    parse_ci_source,
    parse_phase2,
    run_command,
    selection_protocol,
)
from phase2_gate import (
    GateInputs,
    Subject,
    baselines,
    calibration_checks,
    evaluate,
    fresh_local_failures,
    read_eval_results,
    read_fixture,
    read_policy,
    read_subject,
)
from topple.compare import Comparison, compare, read_comparison, to_json
from topple.export import read_run_json
from topple.results import read_results

PHASE2_JSON: Final = "phase2.json"
PHASE2_MD: Final = "phase2.md"
RESULTS_SUBDIR: Final = "phase2"
CI_DIR: Final = "ci_ubuntu"
ARTIFACT: Final = "ubuntu-evidence"
WORKFLOW: Final = "ci.yml"
SUMMARY_FIELDS: Final = tuple(RunSummary.model_fields)

PILOT: Final[Mapping[str, Any]] = {
    "energy_cost_variant": {
        "score": 64.9,
        "success": 0.505,
        "saturated_share": 0.82,
        "source": "docs/design/phase-2.md §2.2 (pilot measurement)",
    },
    "log_std_init": {
        "shipped": 0.0,
        "worse": [-0.5, -1.0],
        "source": "docs/design/phase-2.md §2.2 (pilot measurement)",
    },
    "constant_lr": {
        "seeds_regressed": 1,
        "of": 3,
        "after_steps": 2_300_000,
        "source": "docs/design/phase-2.md §0 (pilot measurement)",
    },
}
"""§5.1's pilot constants, each with its source, rendered as "pilot measurement"."""


class ResultsError(Exception):
    """A refusal: one line naming what is wrong (exit 2)."""


# --- run summaries ------------------------------------------------------------------------------


def run_summary(run_dir: Path) -> dict[str, Any]:
    """``run_dir``'s committed summary (§5.1): exactly ``RunSummary``'s fields, every value
    from its ``train_meta.json`` (and the attempts kept beside it), validated."""
    meta = read_run_json(run_dir / "train_meta.json")
    if not isinstance(meta, dict):
        raise ResultsError(f"{run_dir}/train_meta.json is not an object")
    cfg = meta.get("config")
    if not isinstance(cfg, dict):
        raise ResultsError(f"{run_dir}/train_meta.json has no config")
    summary: dict[str, Any] = {
        "run": run_dir.name,
        "algo": cfg["algo"],
        "seed": cfg["seed"],
        "total_timesteps": cfg["total_timesteps"],
        "attempts": attempts(run_dir),
        "best": {**meta["best"], "rule": cfg["selection"]["gate"]},
    }
    for name in SUMMARY_FIELDS:
        if name not in summary and name in meta:
            summary[name] = meta[name]
    missing = [f for f in SUMMARY_FIELDS if f not in summary and f != "init_source"]
    if missing:
        raise ResultsError(f"{run_dir.name}: train_meta.json lacks {missing}")
    RunSummary.model_validate_json(json.dumps(summary, allow_nan=False), strict=True)
    return summary


def attempts(run_dir: Path) -> list[dict[str, Any]]:
    """The ``wall_clock_abort`` attempts kept as ``<run>.attempt<k>`` (§1), oldest first."""
    found = []
    for k in range(1, MAX_ATTEMPTS + 1):
        path = run_dir.with_name(f"{run_dir.name}.attempt{k}")
        if not path.is_dir():
            continue
        meta = read_run_json(path / "train_meta.json")
        if not isinstance(meta, dict):
            raise ResultsError(f"{path.name}/train_meta.json is not an object")
        found.append(
            RetryAttempt(
                dir=path.name,
                flags=tuple(meta["flags"]),
                wall_clock_s=meta["wall_clock_s"],
                load_avg_start=tuple(meta["load_avg_start"]),
                cpu_s=meta["cpu_s"],
            ).model_dump(mode="json")
        )
    return found


def core(summary: Mapping[str, Any]) -> RunCore:
    return RunCore.model_validate_json(json.dumps(summary, allow_nan=False), strict=True)


# --- exports and evaluations --------------------------------------------------------------------


def topple(*args: str) -> None:
    """Run the ``topple`` CLI of this environment; ``ResultsError`` with its last line if it
    fails."""
    exe = Path(sys.executable).with_name("topple")
    result = run_command(
        [str(exe), *args], timeout_s=3600.0, cwd=REPO
    )  # fixtures are repo-relative
    if result.returncode != 0:
        tail = (result.stderr or result.stdout.decode("utf-8", "replace")).strip()[-400:]
        raise ResultsError(f"topple {args[0]} failed (exit {result.returncode}): {tail}")


def export_report_copy(run_dir: Path, policies: Path) -> Path:
    """Export ``run_dir``'s best checkpoint ``--report-only`` (§3.3) and copy the policy
    directory to ``policies/<run>``, whose name is the comparison entry's id."""
    topple("export", str(run_dir), "--out", str(run_dir / "report_policy"), "--report-only",
           "--replace")  # fmt: skip
    target = policies / run_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run_dir / "report_policy", target)
    return target


def export_shipped(run_dir: Path) -> None:
    """Export ``run_dir`` as the shipped submission with its parity fixture (§5.1)."""
    topple("export", str(run_dir), "--out", str(REPO / SHIPPED_SUBMISSION), "--test-fixture",
           "ppo_balance", "--replace")  # fmt: skip


def fresh_compares(entries: Sequence[Path], workers: int) -> tuple[Comparison, Comparison]:
    """PID (0), LQR (1) and ``entries`` on eval_v1 and on the calibration seeds 0..199."""
    subs = [REPO / PID_SUBMISSION, REPO / LQR_SUBMISSION, *entries]
    return (
        compare(subs, eval_protocol(), workers=workers),
        compare(subs, calibration_protocol(), workers=workers),
    )


@dataclass(frozen=True)
class Evaluated:
    """What one run contributes to the fresh evaluation."""

    run: str
    entry: Path
    report_only: bool
    below_pid_gate: bool


def entry_index(cmp: Comparison, entry: Path) -> int:
    ids = [e.id for e in cmp.entries]
    return ids.index(entry.name)


# --- §2.5 shipping ------------------------------------------------------------------------------


def ship_failures(run_dir: Path, first: Fresh, copy: Path, work: Path) -> tuple[str, ...]:
    """The ``SHIP_GATES`` the run in ``run_dir`` fails, judged by ``phase2_gate``:
    - G1-G4 with ``fresh_local_failures`` on the fresh comparisons, by policy sha256. Its
      report-only copy ``copy`` has the same policy bytes, and this is exactly the check G7
      replays on every recorded attempt.
    - G5 and G6 with ``evaluate``, on the shipping export (``submissions/ppo_balance`` and its
      parity fixture) and its eval_v1 ``results.json``. An export that refuses the checkpoint
      (parity or coverage) fails G5."""
    k = entry_index(first.calibration, copy)
    sha = first.calibration.entries[k].policy_sha256 or ""
    row = FreshRow(run=run_dir.name, entry=copy.name, policy_sha256=sha, report_only=True,
                   below_pid_gate=False)  # fmt: skip
    local, problems = fresh_local_failures(first, row)
    if problems:
        raise ResultsError(f"{run_dir.name}: {problems[0]}")
    try:
        export_shipped(run_dir)
    except ResultsError as exc:
        if "parity" not in str(exc):  # only a parity or coverage refusal is a G5 failure
            raise
        return tuple(g for g in SHIP_GATES if g in (*local, "G5"))
    out = work / f"eval_{run_dir.name}"
    if out.exists():
        shutil.rmtree(out)
    sub = REPO / SHIPPED_SUBMISSION
    topple("eval", str(sub), "-p", "eval_v1", "--out", str(out), "--videos", "none")
    inp = GateInputs(
        cmp_cal=first.calibration,
        cmp_eval=first.eval_v1,
        results=read_eval_results(out / "results.json"),
        report=None,
        ci_results=None,
        ci_cmp=None,
        ci_source=None,
        subject=read_subject(sub),
        policy=read_policy(sub),
        fixture=read_fixture(REPO / FIXTURE_DIR / "ppo_balance_parity.npz"),
        tree=None,
        baselines=baselines(),
    )
    verdict = evaluate(inp, ("G5", "G6"))
    failed = {*local, *(g for g in ("G5", "G6") if verdict[g])}
    return tuple(g for g in SHIP_GATES if g in failed)


def ship(
    ppo_cores: Sequence[RunCore],
    failures: Callable[[str], Sequence[str]],
    ubuntu_failed: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, tuple[ShipAttempt, ...]]:
    """§2.5's choice and its attempts. ``failures(run)`` is a candidate's local verdict
    (``failed_on: "local"``). A candidate that passes locally but is listed in
    ``ubuntu_failed`` (G2-G4 measured by CI after an earlier push, ``--ubuntu-failed``) fails
    those gates ``failed_on: "ubuntu"``, and the next candidate is tried."""
    ubuntu = dict(ubuntu_failed or {})
    where: dict[str, str] = {}

    def judged(run: str) -> Sequence[str]:
        local = tuple(failures(run))
        if any(g in SHIP_GATES for g in local):
            where[run] = "local"
            return local
        if run in ubuntu:
            where[run] = "ubuntu"
            return tuple(ubuntu[run])
        return local

    choice = choose_shipped(candidates(ppo_cores), judged)
    attempts_ = tuple(
        ShipAttempt(
            run=run,
            failed_gates=tuple(g for g in SHIP_GATES if g in failed),  # typed as ShipGate
            failed_on=where[run] if failed else None,  # type: ignore[arg-type]
        )
        for run, failed in choice.attempts
    )
    if choice.run is None:
        tried = [(a.run, list(a.failed_gates)) for a in attempts_]
        raise ResultsError(f"no PPO candidate passes the ship gates: {tried} (2a fails)")
    return choice.run, attempts_


def g11_table(
    cmp_cal: Comparison, ppo: Sequence[RunCore], entries: Mapping[str, Path]
) -> list[G11Row]:
    rows = []
    for c in ppo:
        k = entry_index(cmp_cal, entries[c.run])
        checks = calibration_checks(cmp_cal, k, f"fresh calibration, {c.run}")
        rows.append(
            G11Row(
                run=c.run,
                eligible=c.best.eligible,
                g2=not checks["G2"],
                g2b=not checks["G2b"],
                g3=not checks["G3"],
                g4=not checks["G4"],
            )
        )
    return rows


# --- writing ------------------------------------------------------------------------------------


SHIPPED_FILES: Final = (
    SHIPPED_SUBMISSION,
    FIXTURE_DIR / "ppo_balance_parity.npz",
    FIXTURE_DIR / "ppo_balance_parity.bin",
    FIXTURE_DIR / "ppo_balance_parity.json",
)


@contextlib.contextmanager
def restored_on_failure(backup: Path) -> Iterator[None]:
    """Keep a copy of the shipped submission and its fixtures; put them back if shipping
    fails, so a failed build never leaves another candidate in their place."""
    if backup.exists():
        shutil.rmtree(backup)
    backup.mkdir(parents=True)
    for rel in SHIPPED_FILES:
        src = REPO / rel
        if src.is_dir():
            shutil.copytree(src, backup / rel)
        elif src.is_file():
            (backup / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, backup / rel)
    try:
        yield
    except BaseException:
        for rel in SHIPPED_FILES:
            dst, saved = REPO / rel, backup / rel
            if dst.is_dir():
                shutil.rmtree(dst)
            elif dst.exists():
                dst.unlink()
            if saved.is_dir():
                shutil.copytree(saved, dst)
            elif saved.is_file():
                shutil.copy2(saved, dst)
        raise


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_compare(path: Path, cmp: Comparison) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(cmp), encoding="utf-8")


def report_json(report: Phase2Report) -> bytes:
    """``phase2.json``'s bytes, re-read with ``parse_phase2`` (the gates' reader) first."""
    data = (report.model_dump_json(indent=2) + "\n").encode("utf-8")
    parse_phase2(data)
    return data


def selection_baselines(workers: int) -> dict[str, dict[str, float]]:
    """PID and LQR on the selection seeds 2000..2199: the learning curves' reference lines."""
    cmp = compare([REPO / PID_SUBMISSION, REPO / LQR_SUBMISSION], selection_protocol(),
                  workers=workers)  # fmt: skip
    out = {}
    for name, entry in zip(("pid", "lqr"), cmp.entries, strict=True):
        n = len(cmp.seeds)
        out[name] = {
            "score": sum(entry.scores) / n,
            "success_rate": sum(entry.success) / n,
        }
    return out


def render(report: Phase2Report, results_dir: Path, work: Path, workers: int) -> Phase2Report:
    """Media and ``phase2.md`` (``phase2_render``); the returned report holds the values the
    markdown renders."""
    import phase2_render

    values = phase2_render.make_media(
        report, shipped=REPO / SHIPPED_SUBMISSION, out_dir=results_dir, work=work, workers=workers
    )
    merged = report.model_copy(update={"render": {**report.render, **values}})
    (results_dir.parent / PHASE2_MD).write_text(
        phase2_render.render_markdown(merged), encoding="utf-8"
    )
    return merged


def ci_render(results_dir: Path, shipped_policy: str) -> dict[str, Any]:
    """``render.ci`` from committed ubuntu evidence, only if it is the shipped policy's (older
    evidence is ignored here and replaced by the next ``--fetch-ci``)."""
    ci = results_dir / CI_DIR
    if not (ci / CI_SOURCE_FILE).is_file():
        return {}
    if read_results(ci / "results.json").controller.policy_sha256 != shipped_policy:
        print("note: ci_ubuntu/ holds another policy's evidence; run --fetch-ci after pushing")
        return {}
    source = parse_ci_source((ci / CI_SOURCE_FILE).read_bytes())
    return {
        "ci": {
            "run_id": source.run_id,
            "head_sha": source.head_sha,
            "results": json.loads((ci / "results.json").read_text()),
            "cmp_cal": json.loads((ci / "cmp_cal.json").read_text()),
        }
    }


# --- build --------------------------------------------------------------------------------------


def build(args: argparse.Namespace) -> int:
    runs = {"ppo": args.ppo, "sac": args.sac, "ablation": [args.ablation]}
    if len(args.ppo) != 3 or len(args.sac) != 3:
        raise ResultsError("--ppo and --sac each take exactly 3 run directories")
    summaries = {d.name: run_summary(d) for role in runs.values() for d in role}
    run_dirs = {d.name: d for d in [*args.ppo, *args.sac, args.ablation]}
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    policies = work / "policies"
    policies.mkdir(exist_ok=True)

    ppo_cores = [core(summaries[d.name]) for d in args.ppo]
    reported = [*args.ppo, *args.sac]
    copies = {d.name: export_report_copy(d, policies) for d in reported}
    first_eval, first_cal = fresh_compares([copies[d.name] for d in reported], args.workers)
    first = Fresh(eval_v1=first_eval, calibration=first_cal, rows=())
    with restored_on_failure(work / "previous_submission"):
        shipped_run, ship_attempts = ship(
            ppo_cores,
            lambda run: ship_failures(run_dirs[run], first, copies[run], work),
            args.ubuntu_failed,
        )
    # §2.5 stops at the first passing candidate, so its export is the one left in
    # submissions/ppo_balance (with its parity fixture)
    entries = {d.name: copies[d.name] for d in reported}
    entries[shipped_run] = REPO / SHIPPED_SUBMISSION
    cmp_eval, cmp_cal = fresh_compares([entries[d.name] for d in reported], args.workers)
    subject = read_subject(REPO / SHIPPED_SUBMISSION)
    if not isinstance(subject, Subject):
        raise ResultsError(f"the shipped submission: {subject}")

    rows = []
    for d in reported:
        c = core(summaries[d.name])
        k = entry_index(cmp_cal, entries[d.name])
        rows.append(
            FreshRow(
                run=d.name,
                entry=cmp_cal.entries[k].id,
                policy_sha256=cmp_cal.entries[k].policy_sha256 or "",
                report_only=d.name != shipped_run,
                below_pid_gate=not c.best.eligible,
            )
        )
    report = Phase2Report(
        format=PHASE2_FORMAT,
        runs=tuple(
            RunEntry(role=role, summary=summaries[d.name])  # type: ignore[arg-type]
            for role, dirs in runs.items()
            for d in dirs
        ),
        fresh=Fresh(eval_v1=cmp_eval, calibration=cmp_cal, rows=tuple(rows)),
        shipped=Shipped(
            run=shipped_run, policy_sha256=subject.policy_sha256, attempts=ship_attempts
        ),
        g11=tuple(g11_table(cmp_cal, ppo_cores, entries)),
        pilot=dict(PILOT),
        render={"selection_baselines": selection_baselines(args.workers)},  # type: ignore[dict-item]
    )
    results_dir = Path(args.out) / RESULTS_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    ci = ci_render(results_dir, subject.policy_sha256)
    report = report.model_copy(update={"render": {**report.render, **ci}})
    report_json(report)  # validate before anything is rendered
    if not args.skip_media:
        report = render(report, results_dir, work / "media", args.workers)
    write_atomic(Path(args.out) / PHASE2_JSON, report_json(report))
    print(f"shipped {shipped_run} after {[(a.run, list(a.failed_gates)) for a in ship_attempts]}")
    print(f"wrote {Path(args.out) / PHASE2_JSON}")
    return 0


# --- --fetch-ci ---------------------------------------------------------------------------------


def _json_output(cmd: Sequence[str]) -> Any:
    result = run_command(list(cmd), timeout_s=120.0)
    if result.returncode != 0:
        raise ResultsError(f"{cmd[0]} {cmd[1]} failed: {result.stderr[:300]}")
    return json.loads(result.stdout)


def evidence_unchanged(head_sha: str, git: Callable[[Sequence[str]], int]) -> bool:
    """``head_sha`` is an ancestor of HEAD and no ``EVIDENCE_PATHS`` changed since (§4)."""
    if git(["merge-base", "--is-ancestor", head_sha, "HEAD"]) != 0:
        return False
    return git(["diff", "--quiet", head_sha, "HEAD", "--", *EVIDENCE_PATHS]) == 0


def fetch_ci(args: argparse.Namespace) -> int:
    report = parse_phase2((Path(args.out) / PHASE2_JSON).read_bytes())
    results_dir = Path(args.out) / RESULTS_SUBDIR
    env = git_env()

    def git(cmd: Sequence[str]) -> int:
        return run_command(["git", "-C", str(REPO), *cmd], env=env).returncode

    branch = args.branch or (
        run_command(["git", "-C", str(REPO), "branch", "--show-current"], env=env)
        .stdout.decode()
        .strip()
    )
    runs = _json_output(
        ["gh", "run", "list", "--branch", branch, "--workflow", WORKFLOW, "--event", "push",
         "--limit", "100", "--json", "databaseId,status,conclusion,headSha"]
    )  # fmt: skip
    protocol_hash = calibration_protocol().protocol_hash()
    for run in sorted(runs, key=lambda r: r["databaseId"], reverse=True):
        if (run["status"], run["conclusion"]) != ("completed", "success"):
            continue
        if not evidence_unchanged(run["headSha"], git):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            got = run_command(
                ["gh", "run", "download", str(run["databaseId"]), "--name", ARTIFACT, "--dir", tmp],
                timeout_s=300.0,
            )
            if got.returncode != 0:
                continue
            found = {p.name: p for p in Path(tmp).rglob("*.json")}
            if "results.json" not in found or "cmp_cal_ci.json" not in found:
                continue
            results = read_results(found["results.json"])
            cmp = read_comparison(found["cmp_cal_ci.json"])
            if results.controller.policy_sha256 != report.shipped.policy_sha256:
                continue
            if cmp.protocol_hash != protocol_hash:
                continue
            # Everything is built and validated before anything is written.
            source = {
                "format": "topple.ci_source/1",
                "run_id": run["databaseId"],
                "head_sha": run["headSha"],
                "event": "push",
                "branch": branch,
            }
            source_bytes = (json.dumps(source, indent=2) + "\n").encode()
            parse_ci_source(source_bytes)
            results_bytes = found["results.json"].read_bytes()
            cmp_bytes = found["cmp_cal_ci.json"].read_bytes()
            ci_values = {
                "ci": {
                    "run_id": run["databaseId"],
                    "head_sha": run["headSha"],
                    "results": json.loads(results_bytes),
                    "cmp_cal": json.loads(cmp_bytes),
                }
            }
            updated = report.model_copy(update={"render": {**report.render, **ci_values}})
            import phase2_render

            markdown = phase2_render.render_markdown(updated)
            report_bytes = report_json(updated)
            ci = results_dir / CI_DIR
            ci.mkdir(parents=True, exist_ok=True)
            write_atomic(ci / "results.json", results_bytes)
            write_atomic(ci / "cmp_cal.json", cmp_bytes)
            write_atomic(ci / CI_SOURCE_FILE, source_bytes)
            write_atomic(Path(args.out) / PHASE2_MD, markdown.encode("utf-8"))
            write_atomic(Path(args.out) / PHASE2_JSON, report_bytes)
        print(f"ubuntu evidence from CI run {run['databaseId']} ({run['headSha'][:12]})")
        return 0
    raise ResultsError(
        "no successful push run of ci.yml on this branch has ubuntu evidence for the shipped "
        "policy with HEAD's evidence paths unchanged (push, wait for CI, then retry)"
    )


# --- cli ----------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ppo", type=Path, nargs="*", default=[])
    p.add_argument("--sac", type=Path, nargs="*", default=[])
    p.add_argument("--ablation", type=Path)
    p.add_argument("--out", type=Path, default=REPO / "docs" / "results")
    p.add_argument("--work", type=Path, default=REPO / "out" / "phase2_work")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-media", action="store_true", help="phase2.json only (tests)")
    p.add_argument("--fetch-ci", action="store_true")
    p.add_argument("--branch")
    p.add_argument(
        "--ubuntu-failed",
        type=_ubuntu_failure,
        action="append",
        default=[],
        metavar="RUN=G2,G2b",
        help="a candidate whose pushed CI evidence failed these of G2-G4 (§2.5): ship the next",
    )
    return p.parse_args(argv)


def _ubuntu_failure(text: str) -> tuple[str, tuple[str, ...]]:
    run, sep, gates = text.partition("=")
    found = tuple(g for g in gates.split(",") if g)
    if not sep or not run or not found or not set(found) <= set(UBUNTU_GATES):
        raise argparse.ArgumentTypeError(f"expected RUN=G2,G2b,G3,G4 (a subset), not {text!r}")
    return run, found


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.ubuntu_failed = dict(args.ubuntu_failed)
    try:
        if args.fetch_ci:
            return fetch_ci(args)
        if args.ablation is None:
            raise ResultsError("--ablation is required")
        return build(args)
    except (ResultsError, CommandError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    sys.exit(main())
