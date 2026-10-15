"""The Phase 2a done-when gates (docs/design/phase-2.md §4): exit 0 iff every ``PASS_GATES`` gate
passes, naming each failure.

    uv run python scripts/phase2_gate.py out/cmp_cal.json out/cmp_eval_v1.json \\
        out/ppo_eval_v1/results.json docs/results/phase2.json \\
        --ci-dir docs/results/phase2/ci_ubuntu --submission submissions/ppo_balance \\
        --results-dir docs/results/phase2 [--fixture tests/data/ppo_balance_parity.npz]

Every gate is a pure function of parsed inputs (``GateInputs``), so ``phase2_results.py`` (§2.5:
``SHIP_GATES`` on a candidate's own fresh evaluations, before the ubuntu half exists) and the
tests run them without real runs; ``main`` only reads the files. Inputs are read as contract §15
reads ``results.json`` (a regular file of at most 16 MiB, never blocking, exact JSON types,
``format`` required); one that cannot be read fails every gate that reads it, with the reason.

- **Comparisons.** Entry 0 must be ``submissions/pid_example.py`` and entry 1
  ``submissions/lqr_example.py`` (by submission sha256); the shipped policy is the one entry
  whose ``policy_sha256`` is the submission's ``onnx.sha256`` (and whose ``sha256`` is its
  submission sha256). The protocol hash must be the current calibration's (balance-v1/standard,
  seeds 0..199) or eval_v1's, and every episode present. A paired row used by a gate must be
  ``paired_bootstrap``'s (10,000 resamples, seed 0) and equal its recomputation from the
  per-seed scores (within ``PAIRED_ABS_TOL``); the gate decides on the recomputed bound.
- **G2, G2b, G3 (0..199), G4** hold on the local compare and on the ubuntu one
  (``--ci-dir``), which must record a linux platform, ``cpu_isa`` and onnxruntime.
- **G5** re-runs batch-1 parity against the fixture (``topple.export``'s parity code), **G6**
  loads the submission (``load_policy_dir``), **G7** also requires the ablation run (so G10
  judges it), **G8** checks the results package's files (and that ``phase2.json`` and
  ``phase2.md`` are regular files, not symlinks) with local git commands, **G10** reads only
  ``phase2.json`` and recomputes the selection protocol hash, ``ENV_ALGO_VERSION`` and the
  bundle hash from code. G9 is ``phase2_ci_check.py``.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Final

from phase2_common import (
    CHALLENGE_ID,
    FIXTURE_DIR,
    LQR_SUBMISSION,
    MAX_MEAN_SATURATION,
    MAX_PEAK_TREE_RSS_MIB,
    MAX_RESULTS_ENTRIES,
    MAX_RESULTS_TOTAL_BYTES,
    MIN_G11_RUNS,
    N_ABLATION_RUNS,
    N_PPO_RUNS,
    N_SAC_RUNS,
    PAIRED_ABS_TOL,
    PASS_GATES,
    PHASE2_MD,
    PID_SUBMISSION,
    REPO,
    RESULT_FILES,
    SHIP_GATES,
    SUCCESS_MARGIN,
    CommandError,
    FreshRow,
    G11Row,
    Phase2Report,
    RunCore,
    calibration_protocol,
    candidates,
    choose_shipped,
    describe_error,
    eval_protocol,
    git_env,
    printable,
    read_input,
    read_phase2,
    run_command,
    run_core,
    run_label,
    run_summary,
    selection_protocol,
)
from topple.challenge import ENV_ALGO_VERSION, load_challenge
from topple.compare import Comparison, ordered_pairs, read_comparison
from topple.evaluate import EvalProtocol
from topple.export import (
    MAX_NPZ_BYTES,
    MIN_SATURATED,
    MIN_UNSATURATED,
    PARITY_TOLERANCE,
    batch1_max_abs_diff,
    coverage_counts,
    read_parity_fixture,
)
from topple.loader import LoadError, submission_hashes, submission_sha256
from topple.policy.manifest import policy_task
from topple.policy.runtime import LoadedPolicy, load_policy_dir
from topple.results import EvalResults, read_results
from topple.stats import N_RESAMPLES, SEED, paired_bootstrap
from topple.train.selection import NOT_SHIPPABLE

CALIBRATION_GATES: Final = ("G2", "G2b", "G3", "G4")
LOCAL: Final = "local"
UBUNTU: Final = "ubuntu"


# --- inputs ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InputError:
    """An input that could not be read: every gate that needs it fails with ``message``."""

    message: str


@dataclass(frozen=True)
class Subject:
    """The shipped submission: its submission sha256 (contract §8) and ``policy_sha256``
    (sha256 of ``policy.onnx``, its ``onnx.sha256``)."""

    sha256: str
    policy_sha256: str


@dataclass(frozen=True)
class Baselines:
    """The submission sha256 of ``submissions/pid_example.py`` and ``lqr_example.py``."""

    pid_sha256: str
    lqr_sha256: str


@dataclass(frozen=True)
class ResultsTree:
    """What G8 reads from the file system and git: the size of each ``RESULT_FILES`` entry
    (``None``: missing or not a regular file; symlinks are not followed), the total size of
    ``docs/results/phase2/``, and the paths (relative to the checkout) that are required but not
    tracked, untracked (not ignored) under the results, modified since ``HEAD``, or (of
    ``phase2.json`` and ``phase2.md``) missing or not regular files (a committed symlink would
    let the gate read, and the report show, bytes git never sees)."""

    sizes: Mapping[str, int | None]
    total_bytes: int
    not_tracked: tuple[str, ...]
    untracked: tuple[str, ...]
    modified: tuple[str, ...]
    not_regular: tuple[str, ...]


@dataclass(frozen=True)
class GateInputs:
    """Every input, parsed. ``None`` marks an input that is not part of this evaluation (a §2.5
    candidate is judged on its local evaluations before the ubuntu half exists): the ubuntu
    compare is then skipped; a gate that needs any other ``None`` input fails."""

    cmp_cal: Comparison | InputError
    cmp_eval: Comparison | InputError
    results: EvalResults | InputError
    report: Phase2Report | InputError | None
    ci_results: EvalResults | InputError | None
    ci_cmp: Comparison | InputError | None
    subject: Subject | InputError
    policy: LoadedPolicy | InputError
    fixture: bytes | InputError
    tree: ResultsTree | InputError | None
    baselines: Baselines
    labels: Mapping[str, str] = field(default_factory=dict)
    """Display names of the input files (``cmp_cal`` -> ``out/cmp_cal.json``)."""


type Problems = list[str]


def _errors(*values: object) -> Problems:
    """The messages of the inputs that could not be read (or were not given)."""
    problems = []
    for value in values:
        if isinstance(value, InputError):
            problems.append(value.message)
        elif value is None:
            problems.append("an input this gate needs was not given")
    return problems


# --- comparisons ----------------------------------------------------------------------------------


def _frame(
    cmp: Comparison, protocol: EvalProtocol, name: str, base: Baselines, where: str
) -> Problems:
    """The comparison is of ``protocol`` (its hash covers the seeds), complete, with PID as
    entry 0 and LQR as entry 1."""
    problems = []
    if cmp.protocol_hash != protocol.protocol_hash():
        problems.append(f"{where}: protocol_hash is not the current {name} protocol's")
    if not cmp.complete:
        problems.append(f"{where}: incomplete (an episode is missing)")
    if len(cmp.entries) < 3:
        problems.append(f"{where}: {len(cmp.entries)} entries; PID, LQR and a policy are needed")
        return problems
    for k, (sub, sha) in enumerate(
        ((PID_SUBMISSION, base.pid_sha256), (LQR_SUBMISSION, base.lqr_sha256))
    ):
        if cmp.entries[k].sha256 != sha:
            problems.append(f"{where}: entry {k + 1} is not {sub} ({sub.stem})")
    return problems


def _find(
    cmp: Comparison, policy_sha256: str, sha256: str | None, where: str
) -> tuple[int, Problems]:
    """The index (>= 2) of the one entry for ``policy_sha256``, which must also have the
    submission sha256 ``sha256`` when given."""
    matches = [
        k for k in range(2, len(cmp.entries)) if cmp.entries[k].policy_sha256 == policy_sha256
    ]
    if len(matches) != 1:
        return -1, [f"{where}: {len(matches)} entries are policy {policy_sha256[:12]}, not 1"]
    k = matches[0]
    if sha256 is not None and cmp.entries[k].sha256 != sha256:
        return -1, [
            f"{where}: the policy's entry has another submission_sha256 than the submission"
        ]
    return k, []


def _bound(cmp: Comparison, i: int, j: int, name: str, where: str) -> tuple[float, Problems]:
    """The recomputed lower bound of ``paired[entry i, entry j]``, once the stored row is
    checked to be ``paired_bootstrap``'s and equal to the recomputation."""
    row = cmp.paired[ordered_pairs(len(cmp.entries)).index((i, j))]
    if (row.n_resamples, row.seed) != (N_RESAMPLES, SEED):
        return math.nan, [
            f"{where}: {name} uses {row.n_resamples} resamples and seed {row.seed}, "
            f"not {N_RESAMPLES} and {SEED}"
        ]
    mean, lo, hi = _recompute(cmp.entries[i].scores, cmp.entries[j].scores)
    stored = (row.mean, row.lo, row.hi)
    if any(abs(a - b) > PAIRED_ABS_TOL for a, b in zip(stored, (mean, lo, hi), strict=True)):
        return lo, [f"{where}: {name} differs from its recomputation with paired_bootstrap"]
    return lo, []


@functools.lru_cache(maxsize=64)
def _recompute(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    """``paired_bootstrap(a, b)`` with §5.3's defaults (pure, so the G11 runs and the two
    compares, which share PID and LQR, are bootstrapped once each)."""
    return paired_bootstrap(a, b, n_resamples=N_RESAMPLES, seed=SEED)


def _rate(successes: int, n: int) -> str:
    return f"{successes / n:.3f} ({successes}/{n})"


def calibration_checks(cmp: Comparison, k: int, where: str) -> dict[str, Problems]:
    """G2, G2b, G3's 0..199 part and G4 for entry ``k`` of a framed calibration comparison
    (PID entry 0, LQR entry 1)."""
    pid, lqr, ppo = cmp.entries[0], cmp.entries[1], cmp.entries[k]
    n = len(cmp.seeds)
    out: dict[str, Problems] = {gate: [] for gate in CALIBRATION_GATES}
    for gate, j, name in (("G2", 0, "paired[ppo, pid]"), ("G2b", 1, "paired[ppo, lqr]")):
        lo, problems = _bound(cmp, k, j, name, where)
        out[gate] += problems
        if not problems and not lo > 0.0:
            out[gate].append(f"{where}: {name} lower bound {lo:+.4f} is not > 0")
    s_ppo, s_pid, s_lqr = sum(ppo.success), sum(pid.success), sum(lqr.success)
    if Fraction(s_ppo, n) < Fraction(s_lqr, n) - SUCCESS_MARGIN:
        out["G2b"].append(
            f"{where}: PPO success {_rate(s_ppo, n)} is below LQR's {_rate(s_lqr, n)} - "
            f"{float(SUCCESS_MARGIN):g}"
        )
    if s_ppo < s_pid:
        out["G3"].append(f"{where}: PPO success {_rate(s_ppo, n)} is below PID's {_rate(s_pid, n)}")
    e_ref = calibration_protocol().challenge.config.score.e_ref_j
    energy = median(ppo.energy_j)
    if not energy <= e_ref:
        out["G4"].append(f"{where}: PPO median energy {energy:.4f} J is above E_ref {e_ref:g} J")
    saturation = math.fsum(ppo.saturation_fraction) / n
    if not saturation <= MAX_MEAN_SATURATION:
        out["G4"].append(
            f"{where}: PPO mean saturation {saturation:.4f} is above {MAX_MEAN_SATURATION:g}"
        )
    return out


def _ubuntu_provenance(cmp: Comparison, where: str) -> Problems:
    """The ubuntu evidence records what produced it (§4): a linux platform, ``cpu_isa`` and
    the onnxruntime version."""
    problems = []
    if not cmp.platform.startswith("linux-"):
        problems.append(f"{where}: platform {printable(cmp.platform, 64)!r} is not linux")
    for name in ("cpu_isa", "onnxruntime"):
        if getattr(cmp.versions, name) is None:
            problems.append(f"{where}: versions.{name} is not recorded")
    return problems


def _calibration_gates(inp: GateInputs) -> dict[str, Problems]:
    """G2, G2b, G3 (0..199 part) and G4 on the local compare and, when given, the ubuntu one."""
    out: dict[str, Problems] = {gate: [] for gate in CALIBRATION_GATES}
    sources: list[tuple[str, Comparison | InputError]] = [(LOCAL, inp.cmp_cal)]
    if inp.ci_cmp is not None:
        sources.append((UBUNTU, inp.ci_cmp))
    for name, cmp in sources:
        label = inp.labels.get("ci_cmp" if name == UBUNTU else "cmp_cal", "cmp_cal.json")
        where = f"{name} 0..199 ({label})"
        problems = _errors(cmp, inp.subject)
        if not problems:
            assert isinstance(cmp, Comparison)
            assert isinstance(inp.subject, Subject)
            problems = _frame(cmp, calibration_protocol(), "calibration", inp.baselines, where)
            if name == UBUNTU:
                problems += _ubuntu_provenance(cmp, where)
        if not problems:
            assert isinstance(cmp, Comparison)
            assert isinstance(inp.subject, Subject)
            k, problems = _find(cmp, inp.subject.policy_sha256, inp.subject.sha256, where)
        if problems:
            for gate in CALIBRATION_GATES:
                out[gate] += problems
            continue
        assert isinstance(cmp, Comparison)
        for gate, found in calibration_checks(cmp, k, where).items():
            out[gate] += found
    return out


def _eval_v1(inp: GateInputs) -> tuple[Comparison | None, int, Problems]:
    """The framed eval_v1 comparison and the shipped policy's entry in it."""
    where = f"eval_v1 ({inp.labels.get('cmp_eval', 'cmp_eval_v1.json')})"
    problems = _errors(inp.cmp_eval, inp.subject)
    if problems:
        return None, -1, problems
    assert isinstance(inp.cmp_eval, Comparison)
    assert isinstance(inp.subject, Subject)
    problems = _frame(inp.cmp_eval, eval_protocol(), "eval_v1", inp.baselines, where)
    if problems:
        return None, -1, problems
    k, problems = _find(inp.cmp_eval, inp.subject.policy_sha256, inp.subject.sha256, where)
    return inp.cmp_eval, k, problems


# --- the gates ------------------------------------------------------------------------------------


def gate_g1(inp: GateInputs) -> Problems:
    """eval_v1 protocol score (fsum/N, contract §12): PPO > PID."""
    cmp, k, problems = _eval_v1(inp)
    if problems or cmp is None:
        return problems
    n = len(cmp.seeds)
    ppo, pid = math.fsum(cmp.entries[k].scores) / n, math.fsum(cmp.entries[0].scores) / n
    if not ppo > pid:
        return [f"eval_v1: PPO score {ppo:.4f} is not above PID's {pid:.4f}"]
    return []


def gate_g3(inp: GateInputs, calibration: Mapping[str, Problems]) -> Problems:
    """success(PPO) >= success(PID) on 0..199 (local and ubuntu) and on eval_v1."""
    problems = list(calibration["G3"])
    cmp, k, found = _eval_v1(inp)
    problems += found
    if not found and cmp is not None:
        n = len(cmp.seeds)
        s_ppo, s_pid = sum(cmp.entries[k].success), sum(cmp.entries[0].success)
        if s_ppo < s_pid:
            problems.append(
                f"eval_v1: PPO success {_rate(s_ppo, n)} is below PID's {_rate(s_pid, n)}"
            )
    return problems


def gate_g5(inp: GateInputs) -> Problems:
    """Batch-1 parity of the submission against its fixture: max |a_onnx - a_ref| <= 1e-5, at
    least 300 unsaturated and 32 saturated rows, fixture ``onnx_sha256`` == ``onnx.sha256``."""
    problems = _errors(inp.policy, inp.fixture)
    if problems:
        return problems
    assert isinstance(inp.policy, LoadedPolicy)
    assert isinstance(inp.fixture, bytes)
    task = policy_task(load_challenge(CHALLENGE_ID))
    try:
        fx = read_parity_fixture(inp.fixture, obs_dim=task.obs_dim, action_dim=task.action_dim)
    except ValueError as exc:
        return [f"fixture: {printable(str(exc))}"]
    if fx.onnx_sha256 != inp.policy.policy_sha256:
        problems.append(
            f"fixture onnx_sha256 {fx.onnx_sha256[:12]} is not onnx.sha256 "
            f"{inp.policy.policy_sha256[:12]}"
        )
    try:
        worst = batch1_max_abs_diff(inp.policy.validated, fx.obs, fx.action_ref)
    except ValueError as exc:
        problems.append(f"batch-1 parity: {exc}")
    else:
        if not worst <= PARITY_TOLERANCE:
            problems.append(
                f"batch-1 parity: max |a_onnx - a_ref| is {worst:.3g}, above {PARITY_TOLERANCE:g}"
            )
    unsaturated, saturated = coverage_counts(fx.action_ref)
    if unsaturated < MIN_UNSATURATED:
        problems.append(f"{unsaturated} unsaturated rows (< {MIN_UNSATURATED})")
    if saturated < MIN_SATURATED:
        problems.append(f"{saturated} saturated rows (< {MIN_SATURATED})")
    return problems


def gate_g6(inp: GateInputs) -> Problems:
    """``results.json`` is the submission's eval_v1 run: ``controller.policy_sha256`` ==
    ``onnx.sha256``, ``controller.sha256`` == ``submission_sha256``, ``versions.onnxruntime``
    set; and ``load_policy_dir`` accepts the submission."""
    problems = _errors(inp.results, inp.subject, inp.policy)
    if problems:
        return problems
    assert isinstance(inp.results, EvalResults)
    assert isinstance(inp.subject, Subject)
    assert isinstance(inp.policy, LoadedPolicy)
    r, s = inp.results, inp.subject
    name = inp.labels.get("results", "results.json")
    if r.controller.policy_sha256 != s.policy_sha256:
        problems.append(f"{name}: controller.policy_sha256 is not the submission's onnx.sha256")
    if r.controller.sha256 != s.sha256:
        problems.append(f"{name}: controller.sha256 is not the submission's submission_sha256")
    if r.versions.onnxruntime is None:
        problems.append(f"{name}: versions.onnxruntime is not set")
    if r.protocol_hash != eval_protocol().protocol_hash():
        problems.append(f"{name}: protocol_hash is not eval_v1's")
    if not r.complete:
        problems.append(f"{name}: incomplete")
    if inp.policy.policy_sha256 != s.policy_sha256:
        problems.append("the loaded policy is not the one hashed (did the submission change?)")
    return problems


@dataclass(frozen=True)
class _Runs:
    """phase2.json's runs: how many each role lists, every name listed, the valid table parts
    (``RunCore``) and their names by role, and the problems of the invalid ones, by role."""

    counts: dict[str, int]
    names: set[str]
    roles: dict[str, list[str]]
    cores: dict[str, RunCore]
    problems: dict[str, Problems]

    def all_problems(self) -> Problems:
        return [p for problems in self.problems.values() for p in problems]


def _runs(report: Phase2Report) -> _Runs:
    roles: dict[str, list[str]] = {"ppo": [], "sac": [], "ablation": []}
    counts = dict.fromkeys(roles, 0)
    names: set[str] = set()
    cores: dict[str, RunCore] = {}
    problems: dict[str, Problems] = {"ppo": [], "sac": [], "ablation": []}
    for index, entry in enumerate(report.runs):
        counts[entry.role] += 1
        name = entry.summary.get("run")
        if isinstance(name, str):
            names.add(name)
        try:
            core = run_core(entry)
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            problems[entry.role].append(f"run {run_label(entry, index)}: {describe_error(exc)}")
            continue
        if core.run in cores:
            problems[entry.role].append(f"run {core.run!r} is listed twice")
            continue
        expected_algo = "sac" if entry.role == "sac" else "ppo"
        if core.algo != expected_algo:
            problems[entry.role].append(f"run {core.run!r} ({entry.role}) has algo {core.algo}")
        cores[core.run] = core
        roles[entry.role].append(core.run)
    return _Runs(counts, names, roles, cores, problems)


def gate_g7(inp: GateInputs) -> Problems:
    """phase2.json has 3 PPO and 3 SAC runs, each with a fresh-eval row: its exported policy
    is its best checkpoint's, it is an entry of both fresh comparisons, and a run below the PID
    gate was exported ``--report-only`` and is marked ``below_pid_gate``; and the ablation run
    (§5.1), which G10 judges with the others (§2.3). The shipped run is a PPO run exported for
    shipping, its policy is the submission, and it and its attempts are §2.5's choice among the
    candidates."""
    problems = _errors(inp.report, inp.subject)
    if problems:
        return problems
    assert isinstance(inp.report, Phase2Report)
    assert isinstance(inp.subject, Subject)
    report = inp.report
    runs = _runs(report)
    problems += runs.all_problems()
    expected_runs = (
        ("ppo", N_PPO_RUNS, "PPO"),
        ("sac", N_SAC_RUNS, "SAC"),
        ("ablation", N_ABLATION_RUNS, "ablation"),
    )
    for role, count, label in expected_runs:
        if runs.counts[role] != count:
            problems.append(f"phase2.json has {runs.counts[role]} {label} runs, not {count}")
    fresh = report.fresh
    frames = (
        ("eval_v1", fresh.eval_v1, eval_protocol()),
        ("calibration", fresh.calibration, calibration_protocol()),
    )
    for name, cmp, protocol in frames:
        problems += _frame(cmp, protocol, name, inp.baselines, f"fresh {name}")
    rows: dict[str, FreshRow] = {}
    for fresh_row in fresh.rows:
        if fresh_row.run in rows:
            problems.append(f"run {fresh_row.run!r} has two fresh-eval rows")
        rows[fresh_row.run] = fresh_row
    for name in runs.roles["ppo"] + runs.roles["sac"]:
        core, row = runs.cores[name], rows.get(name)
        if row is None:
            problems.append(f"run {name!r} has no fresh-eval row")
            continue
        if row.policy_sha256 != core.best.onnx_sha256:
            problems.append(f"run {name!r}: the fresh row's policy is not its best checkpoint's")
        if core.best.rule != "pid":
            problems.append(f"run {name!r}: best.rule is {core.best.rule}, not pid")
        if row.below_pid_gate != (not core.best.eligible):
            problems.append(
                f"run {name!r}: below_pid_gate is {str(row.below_pid_gate).lower()}, but its "
                f"best is {'' if core.best.eligible else 'not '}eligible"
            )
        if not core.best.eligible and not row.report_only:
            problems.append(f"run {name!r} is below the PID gate but was not exported report-only")
        for cmp_name, cmp, _ in frames:
            ids = [e.id for e in cmp.entries]
            k = ids.index(row.entry) if row.entry in ids else -1
            if k < 2 or cmp.entries[k].policy_sha256 != row.policy_sha256:
                problems.append(
                    f"run {name!r}: fresh {cmp_name} entry {row.entry!r} is not its policy"
                )
    extra = sorted(set(rows) - runs.names)
    if extra:
        problems.append(f"fresh-eval rows for runs that are not reported: {extra}")
    problems += _shipped(report, runs, rows, inp.subject)
    return problems


def _shipped(
    report: Phase2Report, runs: _Runs, rows: Mapping[str, FreshRow], subject: Subject
) -> Problems:
    shipped = report.shipped
    problems: Problems = []
    if shipped.run not in runs.roles["ppo"]:
        return [f"shipped run {shipped.run!r} is not one of the PPO runs"]
    row = rows.get(shipped.run)
    if row is not None:
        if row.report_only:
            problems.append(f"shipped run {shipped.run!r} was exported report-only")
        if row.policy_sha256 != shipped.policy_sha256:
            problems.append("shipped.policy_sha256 is not its fresh row's policy")
    if shipped.policy_sha256 != subject.policy_sha256:
        problems.append("shipped.policy_sha256 is not the submission's onnx.sha256")
    if runs.problems["ppo"]:
        return [*problems, "cannot replay the section 2.5 choice: a PPO run's table is invalid"]
    ranked = candidates([runs.cores[r] for r in runs.roles["ppo"]])
    recorded = [
        (a.run, tuple(g for g in SHIP_GATES if g in a.failed_gates)) for a in shipped.attempts
    ]
    failures = dict(recorded)
    if len(failures) != len(recorded):
        problems.append("shipped.attempts names a run twice")
    choice = choose_shipped(ranked, lambda run: failures.get(run, ()))
    if choice.run != shipped.run or choice.attempts != tuple(recorded):
        problems.append(
            f"shipped {shipped.run!r} after attempts {recorded} is not the section 2.5 choice: "
            f"candidates {ranked} give {choice.run!r} after {list(choice.attempts)}"
        )
    return problems


def gate_g8(inp: GateInputs) -> Problems:
    """Every required ``docs/results/phase2/`` file exists within its §5.2 cap (and the total
    within 20 MB), is tracked and unmodified, nothing under the results is untracked, and
    ``ci_ubuntu/results.json`` is the submission's."""
    problems = _errors(inp.tree, inp.ci_results, inp.subject)
    if problems:
        return problems
    assert isinstance(inp.tree, ResultsTree)
    assert isinstance(inp.ci_results, EvalResults)
    assert isinstance(inp.subject, Subject)
    tree = inp.tree
    for rel, cap in RESULT_FILES.items():
        size = tree.sizes.get(rel)
        if size is None:
            problems.append(f"{rel}: missing (or not a regular file)")
        elif cap is not None and size > cap:
            problems.append(f"{rel}: {size} bytes, over its {cap}-byte cap")
    if tree.total_bytes > MAX_RESULTS_TOTAL_BYTES:
        problems.append(
            f"the results package holds {tree.total_bytes} bytes, over {MAX_RESULTS_TOTAL_BYTES}"
        )
    problems += [f"{printable(p)}: not tracked by git" for p in tree.not_tracked]
    problems += [f"{printable(p)}: untracked (commit or remove it)" for p in tree.untracked]
    problems += [f"{printable(p)}: modified since HEAD" for p in tree.modified]
    problems += [
        f"{printable(p)}: not a regular file (a symlink counts as missing)"
        for p in tree.not_regular
    ]
    if inp.ci_results.controller.policy_sha256 != inp.subject.policy_sha256:
        problems.append("ci_ubuntu/results.json: controller.policy_sha256 is not onnx.sha256")
    return problems


def gate_g10(inp: GateInputs) -> Problems:
    """Every reported run's summary has exactly the §5.1 fields, no abort flag and no
    ``nondeterministic``, ``num_timesteps >= total_timesteps``, ``peak_tree_rss_mib`` < 6144,
    and the current selection protocol hash, ``ENV_ALGO_VERSION`` and bundle hash (recomputed
    here; decision log 11); the shipped run has ``git_dirty: false``."""
    problems = _errors(inp.report)
    if problems:
        return problems
    assert isinstance(inp.report, Phase2Report)
    selection_hash = selection_protocol().protocol_hash()
    bundle_hash = load_challenge(CHALLENGE_ID).bundle_hash()
    names = []
    for index, entry in enumerate(inp.report.runs):
        label = run_label(entry, index)
        try:
            s = run_summary(entry)
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            problems.append(f"run {label}: {describe_error(exc)}")
            continue
        names.append(s.run)
        blocked = [f for f in s.flags if f in NOT_SHIPPABLE]
        if blocked:
            problems.append(f"run {label}: flags {blocked}")
        if s.num_timesteps < s.total_timesteps:
            problems.append(
                f"run {label}: num_timesteps {s.num_timesteps} < total_timesteps "
                f"{s.total_timesteps}"
            )
        if not s.peak_tree_rss_mib < MAX_PEAK_TREE_RSS_MIB:
            problems.append(
                f"run {label}: peak_tree_rss_mib {s.peak_tree_rss_mib:g} is not below "
                f"{MAX_PEAK_TREE_RSS_MIB}"
            )
        if s.selection_protocol_hash != selection_hash:
            problems.append(
                f"run {label}: selection_protocol_hash {s.selection_protocol_hash[:12]} is not "
                f"the current {selection_hash[:12]}"
            )
        if s.env_algo_version != ENV_ALGO_VERSION:
            problems.append(
                f"run {label}: env_algo_version {s.env_algo_version} is not the current "
                f"{ENV_ALGO_VERSION}"
            )
        if s.bundle_hash != bundle_hash:
            problems.append(
                f"run {label}: bundle_hash {s.bundle_hash[:12]} is not the current "
                f"{bundle_hash[:12]}"
            )
        if s.run == inp.report.shipped.run and s.git_dirty:
            problems.append(f"run {label} (shipped): git_dirty is true")
    if inp.report.shipped.run not in names and not problems:
        problems.append(f"the shipped run {inp.report.shipped.run!r} has no summary")
    return problems


def gate_g11(inp: GateInputs) -> Problems:
    """At least 2 of the 3 PPO runs have an eligible best and their exported policies pass G2,
    G3 and G4 on 0..199 in the fresh evaluation; the G11 table (G2b reported, not gating)
    equals the recomputation."""
    problems = _errors(inp.report)
    if problems:
        return problems
    assert isinstance(inp.report, Phase2Report)
    report = inp.report
    runs = _runs(report)
    ppo = runs.roles["ppo"]
    cal = report.fresh.calibration
    frame = _frame(cal, calibration_protocol(), "calibration", inp.baselines, "fresh calibration")
    if frame:
        return frame
    table: dict[str, G11Row] = {}
    for g11_row in report.g11:
        if g11_row.run in table:
            problems.append(f"g11 lists {g11_row.run!r} twice")
        table[g11_row.run] = g11_row
    if sorted(table) != sorted(ppo) or runs.counts["ppo"] != N_PPO_RUNS:
        problems.append(
            f"the g11 table's runs {sorted(table)} are not the 3 PPO runs {sorted(ppo)}"
        )
    fresh = {row.run: row for row in report.fresh.rows}
    passed = 0
    for name in ppo:
        row = fresh.get(name)
        if row is None:
            problems.append(f"PPO run {name!r} has no fresh-eval row")
            continue
        where = f"fresh calibration, {name}"
        k, found = _find(cal, row.policy_sha256, None, where)
        if found:
            problems += found
            continue
        checks = calibration_checks(cal, k, where)
        computed = G11Row(
            run=name,
            eligible=runs.cores[name].best.eligible,
            g2=not checks["G2"],
            g2b=not checks["G2b"],
            g3=not checks["G3"],
            g4=not checks["G4"],
        )
        if name in table and table[name] != computed:
            wrong = [
                f
                for f in ("eligible", "g2", "g2b", "g3", "g4")
                if getattr(table[name], f) != getattr(computed, f)
            ]
            problems.append(f"g11 row {name!r}: {wrong} differ from the recomputation")
        if computed.eligible and computed.g2 and computed.g3 and computed.g4:
            passed += 1
    if passed < MIN_G11_RUNS:
        problems.insert(
            0,
            f"only {passed} of {N_PPO_RUNS} PPO runs are eligible and pass G2, G3 and G4 on "
            f"0..199 (at least {MIN_G11_RUNS} must)",
        )
    return problems + runs.problems["ppo"]


GateFn = Callable[[GateInputs, Mapping[str, Problems]], Problems]
GATES: Final[Mapping[str, GateFn]] = {
    "G1": lambda inp, _cal: gate_g1(inp),
    "G2": lambda _inp, cal: list(cal["G2"]),
    "G2b": lambda _inp, cal: list(cal["G2b"]),
    "G3": gate_g3,
    "G4": lambda _inp, cal: list(cal["G4"]),
    "G5": lambda inp, _cal: gate_g5(inp),
    "G6": lambda inp, _cal: gate_g6(inp),
    "G7": lambda inp, _cal: gate_g7(inp),
    "G8": lambda inp, _cal: gate_g8(inp),
    "G10": lambda inp, _cal: gate_g10(inp),
    "G11": lambda inp, _cal: gate_g11(inp),
}
"""Each gate: its problems (none: it passes), given the inputs and the shared G2-G4 work."""


def evaluate(inp: GateInputs, gates: Sequence[str] = PASS_GATES) -> dict[str, tuple[str, ...]]:
    """The problems of each of ``gates``, in order; a gate passes iff it has none."""
    unknown = [g for g in gates if g not in GATES]
    if unknown:
        raise ValueError(f"unknown gates {unknown}")
    calibration = (
        _calibration_gates(inp)
        if set(gates) & set(CALIBRATION_GATES)
        else {g: [] for g in CALIBRATION_GATES}
    )
    # A problem shared by two sources (the submission unreadable for both compares) shows once.
    return {g: tuple(dict.fromkeys(GATES[g](inp, calibration))) for g in gates}


# --- reading the inputs ---------------------------------------------------------------------------


def _read[T](path: Path, read: Callable[[Path], T]) -> T | InputError:
    try:
        return read(path)
    except (OSError, ValueError) as exc:  # pydantic's ValidationError is a ValueError
        return InputError(f"{printable(str(path))}: {describe_error(exc)}")


def read_compare(path: Path) -> Comparison | InputError:
    return _read(path, read_comparison)


def read_report(path: Path) -> Phase2Report | InputError:
    return _read(path, read_phase2)


def read_eval_results(path: Path) -> EvalResults | InputError:
    return _read(path, read_results)


def read_fixture(path: Path) -> bytes | InputError:
    return _read(path, lambda p: read_input(p, MAX_NPZ_BYTES))


def read_subject(sub: Path) -> Subject | InputError:
    try:
        sha, policy_sha = submission_hashes(sub)
    except (LoadError, OSError) as exc:
        return InputError(f"{printable(str(sub))}: {printable(str(exc))}")
    if policy_sha is None:
        return InputError(f"{printable(str(sub))}: not a topple.policy/1 directory")
    return Subject(sha256=sha, policy_sha256=policy_sha)


def read_policy(sub: Path) -> LoadedPolicy | InputError:
    try:
        return load_policy_dir(sub, challenge=load_challenge(CHALLENGE_ID))
    except LoadError as exc:
        return InputError(f"load_policy_dir refuses {printable(str(sub))}: {printable(str(exc))}")


def _git(cwd: Path, args: Sequence[str], env: Mapping[str, str]) -> str:
    result = run_command(["git", "-C", str(cwd), *args], env=env)
    if result.returncode != 0:
        raise CommandError(f"git {args[0]} failed: {result.stderr.strip()[:200]}")
    return result.stdout.decode("utf-8", "replace")


def _paths(output: str) -> list[str]:
    return [p for p in output.split("\0") if p]


def _sizes(results_dir: Path) -> tuple[dict[str, int | None], int]:
    """Each required file's size (lstat: a symlink is not a regular file) and the total size
    of the regular files under ``results_dir`` (a bounded walk that follows no symlink)."""
    sizes: dict[str, int | None] = {}
    for rel in RESULT_FILES:
        try:
            info = os.lstat(results_dir / rel)
        except OSError:
            sizes[rel] = None
            continue
        sizes[rel] = info.st_size if stat.S_ISREG(info.st_mode) else None
    total, seen, pending = 0, 0, [results_dir]
    while pending:
        with os.scandir(pending.pop()) as listing:
            for item in listing:
                seen += 1
                if seen > MAX_RESULTS_ENTRIES:
                    raise ValueError(f"more than {MAX_RESULTS_ENTRIES} entries under the results")
                if item.is_dir(follow_symlinks=False):
                    pending.append(Path(item.path))
                elif item.is_file(follow_symlinks=False):
                    total += item.stat(follow_symlinks=False).st_size
    return sizes, total


def _is_regular(path: Path) -> bool:
    """``path`` itself (``lstat``: a symlink is not followed) is a regular file."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def scan_results(
    results_dir: Path, report: Path, *, extra_env: Mapping[str, str] | None = None
) -> ResultsTree | InputError:
    """G8's facts about ``results_dir`` (``docs/results/phase2``), ``report``
    (``phase2.json``) and the ``phase2.md`` beside it, from the file system and local git
    commands (``ls-files``, ``diff HEAD``); ``InputError`` outside a git checkout."""
    env = {**git_env(), **(extra_env or {})}
    try:
        sizes, total = _sizes(results_dir)
        root = Path(_git(results_dir, ["rev-parse", "--show-toplevel"], env).rstrip("\n"))
        root = root.resolve()
        # Directories are resolved (the checkout may sit behind a symlink such as macOS's
        # /tmp); file names are not, so a symlinked required file stays itself.
        top = results_dir.resolve()
        doc = report.parent.resolve() / report.name
        # Unmodified and nothing untracked: all of docs/results (phase2.json's directory), as
        # §4's `git diff --quiet -- docs/results`, and the results package wherever it is.
        rel = [p.relative_to(root).as_posix() for p in (top, doc.parent)]
        required = [(top / r).relative_to(root).as_posix() for r in RESULT_FILES]
        docs = (doc, doc.with_name(PHASE2_MD))
        required += [p.relative_to(root).as_posix() for p in docs]
        not_regular = tuple(p.relative_to(root).as_posix() for p in docs if not _is_regular(p))
        tracked = set(_paths(_git(root, ["ls-files", "-z", "--", *required], env)))
        untracked = _paths(
            _git(root, ["ls-files", "-z", "--others", "--exclude-standard", "--", *rel], env)
        )
        modified = _paths(_git(root, ["diff", "--name-only", "-z", "HEAD", "--", *rel], env))
    except (OSError, ValueError, CommandError) as exc:
        return InputError(f"{printable(str(results_dir))}: {describe_error(exc)}")
    return ResultsTree(
        sizes=sizes,
        total_bytes=total,
        not_tracked=tuple(p for p in required if p not in tracked),
        untracked=tuple(sorted(untracked)),
        modified=tuple(sorted(modified)),
        not_regular=not_regular,
    )


def baselines() -> Baselines:
    return Baselines(
        pid_sha256=submission_sha256(REPO / PID_SUBMISSION),
        lqr_sha256=submission_sha256(REPO / LQR_SUBMISSION),
    )


def load_inputs(args: argparse.Namespace, base: Baselines) -> GateInputs:
    fixture = args.fixture or REPO / FIXTURE_DIR / f"{args.submission.name}_parity.npz"
    ci_dir: Path = args.ci_dir
    return GateInputs(
        cmp_cal=read_compare(args.cmp_cal),
        cmp_eval=read_compare(args.cmp_eval_v1),
        results=read_eval_results(args.results),
        report=read_report(args.phase2),
        ci_results=read_eval_results(ci_dir / "results.json"),
        ci_cmp=read_compare(ci_dir / "cmp_cal.json"),
        subject=read_subject(args.submission),
        policy=read_policy(args.submission),
        fixture=read_fixture(fixture),
        tree=scan_results(args.results_dir, args.phase2),
        baselines=base,
        labels={
            "cmp_cal": printable(str(args.cmp_cal)),
            "cmp_eval": printable(str(args.cmp_eval_v1)),
            "results": printable(str(args.results)),
            "ci_cmp": printable(str(ci_dir / "cmp_cal.json")),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase2_gate.py", description="Evaluate the Phase 2a done-when gates (§4)."
    )
    parser.add_argument("cmp_cal", type=Path, help="local compare on 0..199 (out/cmp_cal.json)")
    parser.add_argument("cmp_eval_v1", type=Path, help="local compare on eval_v1")
    parser.add_argument("results", type=Path, help="the submission's eval_v1 results.json")
    parser.add_argument("phase2", type=Path, help="docs/results/phase2.json")
    parser.add_argument("--ci-dir", type=Path, required=True, help="the ubuntu evidence")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--fixture", type=Path, default=None, help="default: tests/data/<name>_parity.npz"
    )
    args = parser.parse_args(argv)
    try:
        base = baselines()
    except (LoadError, OSError) as exc:
        print(f"phase2_gate: cannot hash the baselines: {printable(str(exc))}", file=sys.stderr)
        return 2
    outcome = evaluate(load_inputs(args, base))
    failed = [g for g, problems in outcome.items() if problems]
    for gate, problems in outcome.items():
        if not problems:
            print(f"{gate:<4} PASS")
            continue
        print(f"{gate:<4} FAIL {printable(problems[0])}")
        for problem in problems[1:]:
            print(f"{'':<4}      {printable(problem)}")
    if failed:
        print(f"{len(failed)} of {len(outcome)} gates fail: {', '.join(failed)}")
        return 1
    print(f"all {len(outcome)} gates pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
