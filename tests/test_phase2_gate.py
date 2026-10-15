"""The Phase 2a done-when gates on synthetic inputs (docs/design/phase-2.md §4, §2.5, §5.1, §6
``test_phase2_gate``).

``scripts/phase2_gate.py`` evaluates every ``PASS_GATES`` gate from parsed inputs, so these tests
build the inputs here, with no training run: comparisons (``topple.compare/1``) of PID, LQR and
PPO with chosen per-seed results, a ``results.json``, a ``phase2.json``, and a tiny exported
policy (from ``tests/run_support.py``) standing in for ``submissions/ppo_balance`` and its parity
fixture. A good set passes every gate; each mutation names exactly the gates it breaks. The CI
check (``scripts/phase2_ci_check.py``) runs against a fake ``gh`` on ``PATH`` and never calls
GitHub. ``scripts/phase2_done.sh`` is checked against §4's block (its ``export && cmp`` line as
two commands, so errexit sees a failing re-export), with shellcheck when it is installed, and
run with stand-in ``uv`` and ``git`` on ``PATH``.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest

import phase2_ci_check as ci_check
import phase2_common as common
import phase2_gate as gate
from phase2_common import G10_FIELDS, PASS_GATES, SHIP_GATES
from run_support import make_run
from topple.challenge import ENV_ALGO_VERSION, load_challenge
from topple.compare import Comparison, Entry, PairedRow, ordered_pairs, to_json
from topple.evaluate import EvalProtocol
from topple.export import export_policy
from topple.loader import submission_hashes, submission_sha256
from topple.policy.runtime import load_policy_dir
from topple.results import (
    ControllerInfo,
    EpisodeResult,
    EvalResults,
    Versions,
    summarize,
    write_results,
)
from topple.stats import N_RESAMPLES, SEED, paired_bootstrap
from topple.train.selection import FLAGS

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
CAL = common.calibration_protocol()
EVAL = common.eval_protocol()
BASE = gate.Baselines(
    pid_sha256=submission_sha256(REPO / common.PID_SUBMISSION),
    lqr_sha256=submission_sha256(REPO / common.LQR_SUBMISSION),
)
VERSIONS = Versions(
    topple="0.1.0",
    mujoco="3.15.0",
    numpy="2.3.0",
    obs_schema="balancer.obs.v1",
    action_spec="diffdrive.effort.v1",
    onnxruntime="1.31.0",
    onnx="1.23.2",
    cpu_isa="avx512f",
)
SELECTION_HASH = "f30ef19048989c9b6edf34e5711b5c116ecd386568a5a1e6673360fc4fcf6287"
CALIBRATION_HASH = "9d3630d321a70821c754a9c64ae121edb8e6847f28a9862f6b4139c5bde59d5e"
BUNDLE_HASH = "89a62007e2f56e61b3a87f3e30034dafc5e6bde4d6c675c07cd79c8d0bbaedd0"
PPO_RUNS = ("ppo_s0", "ppo_s1", "ppo_s2")
SAC_RUNS = ("sac_s0", "sac_s1", "sac_s2")


def fake_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- per-seed results -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Perf:
    """A controller's synthetic episodes: ``successes`` of the seeds survive (scattered by a
    permutation), scoring about ``won``; the rest fall, scoring about ``fell``."""

    won: float
    fell: float
    successes: int
    energy: float
    saturation: float


PID_CAL = Perf(70.0, 45.0, 130, 6.5, 0.074)
LQR_CAL = Perf(81.0, 55.0, 175, 3.3, 0.046)
PPO_CAL = Perf(87.0, 60.0, 188, 3.0, 0.030)
SAC_CAL = Perf(66.0, 40.0, 90, 8.0, 0.120)
PID_EVAL = Perf(76.0, 50.0, 15, 7.4, 0.074)
LQR_EVAL = Perf(86.0, 60.0, 19, 3.0, 0.046)
PPO_EVAL = Perf(88.0, 60.0, 19, 3.0, 0.030)


@dataclass(frozen=True)
class Arrays:
    scores: tuple[float, ...]
    success: tuple[bool, ...]
    energy_j: tuple[float, ...]
    saturation_fraction: tuple[float, ...]

    @property
    def termination(self) -> tuple[str, ...]:
        return tuple("timeout" if ok else "fell" for ok in self.success)


def arrays(perf: Perf, n: int, salt: int) -> Arrays:
    wiggle = [((i * 7919 + salt * 104729) % 1001) / 1000.0 - 0.5 for i in range(n)]
    success = tuple((i * 37 + salt * 11) % n < perf.successes for i in range(n))
    scores = tuple(
        round((perf.won if ok else perf.fell) + 4.0 * w, 6)
        for ok, w in zip(success, wiggle, strict=True)
    )
    energy = tuple(round(perf.energy * (1.0 + 0.2 * w), 6) for w in wiggle)
    return Arrays(scores, success, energy, (perf.saturation,) * n)


def entry(name: str, a: Arrays, *, sha256: str, policy_sha256: str | None) -> Entry:
    return Entry(
        id=name,
        policy_sha256=policy_sha256,
        sha256=sha256,
        scores=a.scores,
        success=a.success,
        energy_j=a.energy_j,
        saturation_fraction=a.saturation_fraction,
        termination=a.termination,  # type: ignore[arg-type]
        extras={},
    )


@functools.lru_cache(maxsize=256)
def bootstrap(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    """``paired_bootstrap`` (what ``topple compare`` stores), once per pair of arrays."""
    return paired_bootstrap(a, b)


def comparison(
    protocol: EvalProtocol, entries: Sequence[Entry], *, versions: Versions = VERSIONS
) -> Comparison:
    rows = []
    for i, j in ordered_pairs(len(entries)):
        a, b = entries[i], entries[j]
        mean, lo, hi = bootstrap(a.scores, b.scores)
        rows.append(
            PairedRow(a=a.id, b=b.id, mean=mean, lo=lo, hi=hi, n_resamples=N_RESAMPLES, seed=SEED)
        )
    return Comparison(
        protocol=protocol.id,
        protocol_hash=protocol.protocol_hash(),
        complete=not any("missing" in e.termination for e in entries),
        seeds=protocol.seeds,
        platform="linux-x86_64",
        versions=versions,
        entries=tuple(entries),
        paired=tuple(rows),
    )


def baselines(protocol: EvalProtocol, pid: Perf, lqr: Perf) -> list[Entry]:
    n = len(protocol.seeds)
    return [
        entry("pid_example", arrays(pid, n, 1), sha256=BASE.pid_sha256, policy_sha256=None),
        entry("lqr_example", arrays(lqr, n, 2), sha256=BASE.lqr_sha256, policy_sha256=None),
    ]


def three_way(
    protocol: EvalProtocol,
    subject: gate.Subject,
    *,
    pid: Perf,
    lqr: Perf,
    ppo: Perf | Arrays,
    versions: Versions = VERSIONS,
) -> Comparison:
    """PID, LQR and the shipped policy, as ``phase2_done.sh`` runs ``topple compare``."""
    n = len(protocol.seeds)
    ppo_arrays = ppo if isinstance(ppo, Arrays) else arrays(ppo, n, 3)
    shipped = entry(
        "ppo_balance", ppo_arrays, sha256=subject.sha256, policy_sha256=subject.policy_sha256
    )
    return comparison(protocol, [*baselines(protocol, pid, lqr), shipped], versions=versions)


# --- the shipped policy, its results and phase2.json ----------------------------------------------


@pytest.fixture(scope="module")
def shipped(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A tiny PPO export standing in for ``submissions/ppo_balance``, with its parity fixture,
    and a second (SAC) export whose fixture belongs to another policy."""
    root = tmp_path_factory.mktemp("shipped")
    run = make_run(root / "runs", name="ppo_s0")
    sub = root / "submissions" / "ppo_balance"
    sub.parent.mkdir()
    export_policy(run.path, sub, test_fixture="ppo_balance", fixture_dir=root / "data")
    other = make_run(root / "runs", algo="sac", name="sac_s0")
    export_policy(other.path, root / "sac_sub", test_fixture="other", fixture_dir=root / "data")
    sha, policy_sha = submission_hashes(sub)
    assert policy_sha is not None
    return {
        "dir": sub,
        "subject": gate.Subject(sha256=sha, policy_sha256=policy_sha),
        "policy": load_policy_dir(sub, challenge=load_challenge("balance-v1")),
        "fixture": (root / "data" / "ppo_balance_parity.npz").read_bytes(),
        "other_fixture": (root / "data" / "other_parity.npz").read_bytes(),
        "fixture_path": root / "data" / "ppo_balance_parity.npz",
    }


def eval_results(subject: gate.Subject, a: Arrays, *, versions: Versions = VERSIONS) -> EvalResults:
    episodes = [
        EpisodeResult(
            index=i,
            seed=seed,
            success=ok,
            score=score,
            metrics={
                "time_upright_s": 10.0 if ok else 4.0,
                "termination": "timeout" if ok else "fell",
                "energy_j": energy,
            },
        )
        for i, (seed, ok, score, energy) in enumerate(
            zip(EVAL.seeds, a.success, a.scores, a.energy_j, strict=True)
        )
    ]
    return EvalResults(
        challenge="balance-v1",
        protocol=EVAL.id,
        protocol_hash=EVAL.protocol_hash(),
        seed_set="public",
        versions=versions,
        platform="linux-x86_64",
        robot_hash="0" * 64,
        controller=ControllerInfo(
            id="ppo_balance", sha256=subject.sha256, policy_sha256=subject.policy_sha256
        ),
        isolation="inprocess",
        complete=True,
        episodes=episodes,
        aggregate=summarize(episodes, len(episodes)),
    )


def history_row(step: int, *, eligible: bool, score: float, sha: str) -> dict[str, Any]:
    return {
        "step": step,
        "success_rate": 0.9 if eligible else 0.5,
        "score": score,
        "median_energy_j": 3.0,
        "mean_saturation": 0.03,
        "extras": {"fell": 0.1},
        "eligible": eligible,
        "std_mean": 0.3,
        "frac_mean_outside_1": 0.02,
        "excess_penalty_mean": 0.001,
        "cpu_s": 20.0,
        "wall_s": 5.2,
        "onnx_sha256": sha,
    }


def summary(
    name: str, *, algo: str, seed: int, eligible: bool, score: float, sha: str
) -> dict[str, Any]:
    """Every §5.1 field of a finished, clean run whose best checkpoint is the last."""
    total = 5_000_000 if algo == "ppo" else 1_000_000
    final = 5_013_504 if algo == "ppo" else 1_000_000
    return {
        "run": name,
        "algo": algo,
        "seed": seed,
        "config_sha256": fake_sha(f"config {name}"),
        "config_file_sha256": fake_sha(f"file {name}"),
        "total_timesteps": total,
        "num_timesteps": final,
        "selection_protocol_hash": SELECTION_HASH,
        "env_algo_version": ENV_ALGO_VERSION,
        "bundle_hash": BUNDLE_HASH,
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "git_dirty": False,
        "versions": {"python": "3.12.3", "torch": "2.14.1", "psutil": None},
        "mp_start_method": "forkserver",
        "torch_threads": 1,
        "wall_clock_s": 512.5,
        "train_s": 400.0,
        "selection_s": 100.0,
        "cpu_s": 1450.0,
        "load_avg_start": [3.1, 2.9, 2.5],
        "load_avg_end": [4.0, 3.2, 2.7],
        "peak_tree_rss_mib": 3224.0,
        "peak_procs": 16,
        "selection_history": [
            history_row(262_144, eligible=False, score=60.0, sha=fake_sha(f"early {name}")),
            history_row(final, eligible=eligible, score=score, sha=sha),
        ],
        "best": {"step": final, "eligible": eligible, "rule": "pid", "onnx_sha256": sha},
        "flags": [] if eligible else ["no_eligible"],
        "e_ref_j": 13.1,
        "attempts": [],
    }


def phase2_doc(
    subject: gate.Subject,
    *,
    shipped_run: str = "ppo_s0",
    attempts: Sequence[tuple[str, Sequence[str]]] = (("ppo_s0", ()),),
) -> dict[str, Any]:
    """A good ``phase2.json``: 3 PPO runs (ppo_s0 ranked first), 3 SAC runs below the PID gate
    and the ablation; fresh comparisons of all six with PID and LQR. ``shipped_run``'s policy
    is the submission's, shipped after ``attempts``."""

    def policy_sha(run: str, subject: gate.Subject) -> str:
        return subject.policy_sha256 if run == shipped_run else fake_sha(f"policy {run}")

    ppo_scores = {"ppo_s0": 84.0, "ppo_s1": 83.5, "ppo_s2": 82.0}
    runs = [
        {
            "role": "ppo",
            "summary": summary(
                r,
                algo="ppo",
                seed=k,
                eligible=True,
                score=ppo_scores[r],
                sha=policy_sha(r, subject),
            ),
        }
        for k, r in enumerate(PPO_RUNS)
    ]
    runs += [
        {
            "role": "sac",
            "summary": summary(
                r, algo="sac", seed=k, eligible=False, score=60.0, sha=policy_sha(r, subject)
            ),
        }
        for k, r in enumerate(SAC_RUNS)
    ]
    ablation = summary(
        "ablate_nopenalty", algo="ppo", seed=1, eligible=True, score=80.0, sha=fake_sha("ablation")
    )
    ablation["total_timesteps"] = ablation["num_timesteps"] = 1_000_000
    for row in ablation["selection_history"]:
        row["step"] = min(row["step"], 1_000_000)
    ablation["best"]["step"] = 1_000_000
    runs.append({"role": "ablation", "summary": ablation})

    def fresh(protocol: EvalProtocol, ppo: Perf, sac: Perf, pid: Perf, lqr: Perf) -> dict[str, Any]:
        n = len(protocol.seeds)
        entries = baselines(protocol, pid, lqr)
        entries += [
            entry(
                r,
                arrays(ppo, n, 10 + k),
                sha256=fake_sha(f"sub {r}"),
                policy_sha256=policy_sha(r, subject),
            )
            for k, r in enumerate(PPO_RUNS)
        ]
        entries += [
            entry(
                r,
                arrays(sac, n, 20 + k),
                sha256=fake_sha(f"sub {r}"),
                policy_sha256=policy_sha(r, subject),
            )
            for k, r in enumerate(SAC_RUNS)
        ]
        return json.loads(to_json(comparison(protocol, entries)))

    rows = [
        {
            "run": r,
            "entry": r,
            "policy_sha256": policy_sha(r, subject),
            "report_only": r != shipped_run,
            "below_pid_gate": False,
        }
        for r in PPO_RUNS
    ]
    rows += [
        {
            "run": r,
            "entry": r,
            "policy_sha256": policy_sha(r, subject),
            "report_only": True,
            "below_pid_gate": True,
        }
        for r in SAC_RUNS
    ]
    return {
        "format": "topple.phase2/1",
        "runs": runs,
        "fresh": {
            "eval_v1": fresh(EVAL, PPO_EVAL, Perf(60.0, 40.0, 8, 8.0, 0.12), PID_EVAL, LQR_EVAL),
            "calibration": fresh(CAL, PPO_CAL, SAC_CAL, PID_CAL, LQR_CAL),
            "rows": rows,
        },
        "shipped": {
            "run": shipped_run,
            "policy_sha256": subject.policy_sha256,
            "attempts": [{"run": r, "failed_gates": list(g)} for r, g in attempts],
        },
        "g11": [
            {"run": r, "eligible": True, "g2": True, "g2b": True, "g3": True, "g4": True}
            for r in PPO_RUNS
        ],
        "pilot": {"energy_cost_variant": {"score": 64.9, "success": 0.505, "source": "pilot"}},
        "render": {"ci_run": 123},
    }


def report_of(doc: Mapping[str, Any]) -> common.Phase2Report:
    return common.parse_phase2(json.dumps(doc).encode())


def good_tree() -> gate.ResultsTree:
    return gate.ResultsTree(
        sizes=dict.fromkeys(common.RESULT_FILES, 1000),
        total_bytes=11_000,
        not_tracked=(),
        untracked=(),
        modified=(),
        not_regular=(),
    )


@dataclass(frozen=True)
class GoodSet:
    """Every gate input, all passing; ``inputs(**changes)`` swaps any of them."""

    shipped: dict[str, Any]

    @property
    def subject(self) -> gate.Subject:
        return self.shipped["subject"]  # type: ignore[no-any-return]

    def cal(self, **kw: Any) -> Comparison:
        return three_way(
            CAL, self.subject, **{"pid": PID_CAL, "lqr": LQR_CAL, "ppo": PPO_CAL, **kw}
        )

    def eval_v1(self, **kw: Any) -> Comparison:
        return three_way(
            EVAL, self.subject, **{"pid": PID_EVAL, "lqr": LQR_EVAL, "ppo": PPO_EVAL, **kw}
        )

    def results(self, **kw: Any) -> EvalResults:
        return eval_results(self.subject, arrays(PPO_EVAL, 20, 3), **kw)

    def doc(self) -> dict[str, Any]:
        return phase2_doc(self.subject)

    def inputs(self, **changes: Any) -> gate.GateInputs:
        base = gate.GateInputs(
            cmp_cal=self.cal(),
            cmp_eval=self.eval_v1(),
            results=self.results(),
            report=report_of(self.doc()),
            ci_results=self.results(),
            ci_cmp=self.cal(),
            subject=self.subject,
            policy=self.shipped["policy"],
            fixture=self.shipped["fixture"],
            tree=good_tree(),
            baselines=BASE,
        )
        return replace(base, **changes)


@pytest.fixture(scope="module")
def good(shipped: dict[str, Any]) -> GoodSet:
    return GoodSet(shipped)


def failing(
    inputs: gate.GateInputs, gates: Sequence[str] = PASS_GATES
) -> dict[str, tuple[str, ...]]:
    return {g: problems for g, problems in gate.evaluate(inputs, gates).items() if problems}


# --- constants ------------------------------------------------------------------------------------


def test_the_gate_constants() -> None:
    assert SHIP_GATES == ("G1", "G2", "G2b", "G3", "G4", "G5", "G6")
    assert (*SHIP_GATES, "G7", "G8", "G10", "G11") == PASS_GATES
    assert "G9" not in PASS_GATES  # phase2_ci_check.py
    assert set(gate.GATES) == set(PASS_GATES)
    assert set(G10_FIELDS) <= set(common.RunSummary.model_fields)


def test_the_recomputed_values_are_the_spec_known_answers() -> None:
    """§2.3's known answers at v5.3 (they move only with a reviewed re-pin)."""
    assert common.selection_protocol().protocol_hash() == SELECTION_HASH
    assert common.calibration_protocol().protocol_hash() == CALIBRATION_HASH
    assert load_challenge("balance-v1").bundle_hash() == BUNDLE_HASH
    assert ENV_ALGO_VERSION == 1
    assert EVAL.seeds == tuple(range(1000, 1020))


def test_the_flag_literal_is_trainings_flag_list() -> None:
    assert get_args(common.Flag) == FLAGS


def test_the_run_summary_has_exactly_the_section_5_1_fields() -> None:
    fields = [
        "run",
        "algo",
        "seed",
        "config_sha256",
        "config_file_sha256",
        "total_timesteps",
        "num_timesteps",
        "selection_protocol_hash",
        "env_algo_version",
        "bundle_hash",
        "git_commit",
        "git_dirty",
        "versions",
        "mp_start_method",
        "torch_threads",
        "wall_clock_s",
        "train_s",
        "selection_s",
        "cpu_s",
        "load_avg_start",
        "load_avg_end",
        "peak_tree_rss_mib",
        "peak_procs",
        "selection_history",
        "best",
        "flags",
        "e_ref_j",
        "attempts",
        "init_source",
    ]
    assert list(common.RunSummary.model_fields) == fields


# --- a good set -----------------------------------------------------------------------------------


def test_a_good_set_passes_every_gate(good: GoodSet) -> None:
    assert failing(good.inputs()) == {}


def test_the_good_set_is_not_trivially_good(good: GoodSet) -> None:
    """The bars are close enough that the mutations below are meaningful: PPO beats LQR by a
    few points, not by everything."""
    cal = good.cal()
    _, lo, _ = paired_bootstrap(cal.entries[2].scores, cal.entries[1].scores)
    assert 0.0 < lo < 10.0


# --- each gate, failing alone ---------------------------------------------------------------------


def g2_only(good: GoodSet) -> gate.GateInputs:
    """PID scores exactly what PPO scores (it only survives less): PPO - PID's interval sits
    on 0, while PPO still beats LQR and PID's success."""
    ppo = arrays(PPO_CAL, 200, 3)
    pid_success = tuple(ok and i % 4 != 0 for i, ok in enumerate(ppo.success))
    pid = Arrays(ppo.scores, pid_success, ppo.energy_j, ppo.saturation_fraction)
    cal = good.cal()
    entries = (replace_arrays(cal.entries[0], pid), *cal.entries[1:])
    return good.inputs(cmp_cal=comparison(CAL, entries))


def replace_arrays(e: Entry, a: Arrays) -> Entry:
    return entry(e.id, a, sha256=e.sha256, policy_sha256=e.policy_sha256)


def g2b_only(good: GoodSet) -> gate.GateInputs:
    """PPO plays exactly as LQR does: the PPO - LQR interval is [0, 0]."""
    return good.inputs(cmp_cal=good.cal(ppo=arrays(LQR_CAL, 200, 2)))


def g2b_success_only(good: GoodSet) -> gate.GateInputs:
    """PPO scores far above LQR but survives 0.80 against LQR's 0.875 (- 0.03 = 0.845)."""
    return good.inputs(cmp_cal=good.cal(ppo=Perf(97.0, 92.0, 160, 3.0, 0.03)))


def g3_local_only(good: GoodSet) -> gate.GateInputs:
    """PID survives more than PPO (190 vs 188) but scores far lower."""
    return good.inputs(cmp_cal=good.cal(pid=Perf(55.0, 30.0, 190, 6.5, 0.07)))


def g3_eval_only(good: GoodSet) -> gate.GateInputs:
    """eval_v1: PID survives all 20, PPO 19, but PPO scores higher (G1 holds)."""
    return good.inputs(cmp_eval=good.eval_v1(pid=Perf(60.0, 40.0, 20, 7.4, 0.07)))


MUTATIONS: dict[str, tuple[Callable[[GoodSet], gate.GateInputs], set[str], str]] = {
    "g1": (
        lambda g: g.inputs(cmp_eval=g.eval_v1(ppo=Perf(70.0, 50.0, 19, 3.0, 0.03))),
        {"G1"},
        "eval_v1",
    ),
    "g2_local": (g2_only, {"G2"}, "local"),
    "g2b_interval": (g2b_only, {"G2b"}, "paired[ppo, lqr]"),
    "g2b_success": (g2b_success_only, {"G2b"}, "success"),
    "g3_local": (g3_local_only, {"G3"}, "local"),
    "g3_eval_v1": (g3_eval_only, {"G3"}, "eval_v1"),
    "g4_energy": (
        lambda g: g.inputs(cmp_cal=g.cal(ppo=Perf(87.0, 60.0, 188, 14.0, 0.03))),
        {"G4"},
        "E_ref",
    ),
    "g4_saturation": (
        lambda g: g.inputs(cmp_cal=g.cal(ppo=Perf(87.0, 60.0, 188, 3.0, 0.11))),
        {"G4"},
        "saturation",
    ),
    "g4_ubuntu_only": (
        lambda g: g.inputs(ci_cmp=g.cal(ppo=Perf(87.0, 60.0, 188, 13.2, 0.03))),
        {"G4"},
        "ubuntu",
    ),
    "g2b_ubuntu_only": (
        lambda g: g.inputs(ci_cmp=g.cal(ppo=arrays(LQR_CAL, 200, 2))),
        {"G2b"},
        "ubuntu",
    ),
    "g5_other_fixture": (
        lambda g: g.inputs(fixture=g.shipped["other_fixture"]),
        {"G5"},
        "onnx_sha256",
    ),
    "g5_no_fixture": (
        lambda g: g.inputs(fixture=gate.InputError("ppo_balance_parity.npz: No such file")),
        {"G5"},
        "No such file",
    ),
    "g6_policy_sha": (
        lambda g: g.inputs(
            results=g.results().model_copy(
                update={
                    "controller": ControllerInfo(
                        id="x", sha256=g.subject.sha256, policy_sha256="1" * 64
                    )
                }
            )
        ),
        {"G6"},
        "policy_sha256",
    ),
    "g6_submission_sha": (
        lambda g: g.inputs(
            results=g.results().model_copy(
                update={
                    "controller": ControllerInfo(
                        id="x", sha256="1" * 64, policy_sha256=g.subject.policy_sha256
                    )
                }
            )
        ),
        {"G6"},
        "submission_sha256",
    ),
    "g6_no_onnxruntime": (
        lambda g: g.inputs(
            results=g.results(versions=VERSIONS.model_copy(update={"onnxruntime": None}))
        ),
        {"G6"},
        "onnxruntime",
    ),
    "g8_ci_results_policy": (
        lambda g: g.inputs(
            ci_results=g.results().model_copy(
                update={
                    "controller": ControllerInfo(
                        id="x", sha256=g.subject.sha256, policy_sha256="2" * 64
                    )
                }
            )
        ),
        {"G8"},
        "ci_ubuntu/results.json",
    ),
}


@pytest.mark.parametrize("name", MUTATIONS)
def test_each_failing_gate_is_named(name: str, good: GoodSet) -> None:
    mutate, expected, words = MUTATIONS[name]
    failed = failing(mutate(good))
    assert set(failed) == expected, failed
    assert any(words in problem for problems in failed.values() for problem in problems), failed


@pytest.mark.parametrize(
    ("lqr", "ppo", "passes"), [(175, 169, True), (175, 168, False), (91, 85, True)]
)
def test_g2b_success_margin_is_exact(lqr: int, ppo: int, passes: bool, good: GoodSet) -> None:
    """success(PPO) >= success(LQR) - 0.03 at the boundary: 175/200 - 0.03 is exactly
    169/200 (passes) and 168/200 fails G2b alone; 91/200 - 0.03 is exactly 85/200, which
    floats get wrong (85/200 < 91/200 - 0.03 in binary), so the gate decides on fractions.
    PPO's scores stay far above LQR's and PID survives less than PPO."""
    cal = good.cal(
        pid=Perf(70.0, 45.0, 80, 6.5, 0.074),
        lqr=Perf(81.0, 55.0, lqr, 3.3, 0.046),
        ppo=Perf(97.0, 92.0, ppo, 3.0, 0.03),
    )
    assert [sum(e.success) for e in cal.entries] == [80, lqr, ppo]
    failed = failing(good.inputs(cmp_cal=cal))
    if passes:
        assert failed == {}
    else:
        assert set(failed) == {"G2b"}, failed
        assert f"{ppo / 200:.3f} ({ppo}/200)" in failed["G2b"][0]


def test_g1_compares_ppo_with_pid_not_lqr(good: GoodSet) -> None:
    """eval_v1: PPO scores above PID but below LQR, which passes G1."""
    cmp = good.eval_v1(ppo=Perf(80.0, 55.0, 19, 3.0, 0.03))
    n = len(cmp.seeds)
    pid, lqr, ppo = (sum(e.scores) / n for e in cmp.entries)
    assert pid < ppo < lqr
    assert failing(good.inputs(cmp_eval=cmp)) == {}


def test_g5_needs_32_saturated_rows(good: GoodSet) -> None:
    """A fixture keeping every unsaturated row but only 31 saturated ones fails G5 alone."""
    with np.load(io.BytesIO(good.shipped["fixture"])) as npz:
        members = {k: npz[k] for k in npz.files}
    saturated = np.abs(members["action_ref"]).max(axis=1) >= 0.999
    keep = ~saturated
    keep[np.flatnonzero(saturated)[:31]] = True
    for key in ("obs", "action_ref", "set"):
        members[key] = members[key][keep]
    buf = io.BytesIO()
    np.savez(buf, **members)
    failed = failing(good.inputs(fixture=buf.getvalue()))
    assert set(failed) == {"G5"}
    assert failed["G5"] == ("31 saturated rows (< 32)",)


def test_g6_needs_a_complete_run(good: GoodSet) -> None:
    results = good.results().model_copy(update={"complete": False})
    failed = failing(good.inputs(results=results))
    assert set(failed) == {"G6"}
    assert "incomplete" in failed["G6"][0]


def test_the_policy_entry_must_be_the_submission(good: GoodSet) -> None:
    """An entry with the shipped ONNX but another submission sha256 (another policy.json, say)
    is not the submission: every gate reading that compare fails."""
    other = gate.Subject(sha256="7" * 64, policy_sha256=good.subject.policy_sha256)
    cal = three_way(CAL, other, pid=PID_CAL, lqr=LQR_CAL, ppo=PPO_CAL)
    failed = failing(good.inputs(cmp_cal=cal))
    assert set(failed) == {"G2", "G2b", "G3", "G4"}
    assert "submission_sha256" in failed["G2"][0]
    ev = three_way(EVAL, other, pid=PID_EVAL, lqr=LQR_EVAL, ppo=PPO_EVAL)
    failed = failing(good.inputs(cmp_eval=ev))
    assert set(failed) == {"G1", "G3"}
    assert "submission_sha256" in failed["G1"][0]


def test_a_missing_input_fails_every_gate_that_reads_it(good: GoodSet) -> None:
    missing = gate.InputError("out/cmp_cal.json: No such file or directory")
    failed = failing(good.inputs(cmp_cal=missing))
    assert set(failed) == {"G2", "G2b", "G3", "G4"}
    assert all("No such file" in p[0] for p in failed.values())
    assert set(failing(good.inputs(report=gate.InputError("phase2.json: no format")))) == {
        "G7",
        "G10",
        "G11",
    }
    assert set(failing(good.inputs(policy=gate.InputError("LoadError")))) == {"G5", "G6"}


def test_a_compare_of_another_protocol_or_policy_fails(good: GoodSet) -> None:
    other = EvalProtocol.from_id("balance-v1/clean")
    clean = replace(other, seeds=CAL.seeds)
    wrong_protocol = three_way(clean, good.subject, pid=PID_CAL, lqr=LQR_CAL, ppo=PPO_CAL)
    failed = failing(good.inputs(cmp_cal=wrong_protocol))
    assert set(failed) == {"G2", "G2b", "G3", "G4"}
    assert "protocol_hash" in failed["G2"][0]
    stranger = gate.Subject(sha256=good.subject.sha256, policy_sha256="3" * 64)
    failed = failing(
        good.inputs(cmp_eval=three_way(EVAL, stranger, pid=PID_EVAL, lqr=LQR_EVAL, ppo=PPO_EVAL))
    )
    assert set(failed) == {"G1", "G3"}
    swapped = good.cal()
    swapped = comparison(CAL, (swapped.entries[1], swapped.entries[0], swapped.entries[2]))
    assert "pid_example" in failing(good.inputs(cmp_cal=swapped))["G2"][0]


def test_an_incomplete_compare_fails(good: GoodSet) -> None:
    cal = good.cal()
    ppo = cal.entries[2]
    lost = ppo.model_copy(
        update={
            "termination": ("missing", *ppo.termination[1:]),
            "scores": (0.0, *ppo.scores[1:]),
            "success": (False, *ppo.success[1:]),
        }
    )
    incomplete = comparison(CAL, (*cal.entries[:2], lost))
    assert not incomplete.complete
    failed = failing(good.inputs(ci_cmp=incomplete))
    assert set(failed) == {"G2", "G2b", "G3", "G4"}
    assert "incomplete" in failed["G4"][0]


def test_a_stored_paired_row_must_equal_its_recomputation(good: GoodSet) -> None:
    cal = good.cal()
    rows = list(cal.paired)
    k = ordered_pairs(3).index((2, 0))
    rows[k] = rows[k].model_copy(update={"lo": rows[k].lo + 1e-9})
    failed = failing(good.inputs(cmp_cal=cal.model_copy(update={"paired": tuple(rows)})))
    assert set(failed) == {"G2"}
    assert "recomputation" in failed["G2"][0]
    rows = list(cal.paired)
    rows[k] = rows[k].model_copy(update={"n_resamples": 100})
    failed = failing(good.inputs(cmp_cal=cal.model_copy(update={"paired": tuple(rows)})))
    assert set(failed) == {"G2"}


def test_a_last_bit_difference_in_a_stored_row_is_tolerated(good: GoodSet) -> None:
    """A row computed on ubuntu may differ from a Mac recomputation in the last bits."""
    cal = good.cal()
    rows = list(cal.paired)
    k = ordered_pairs(3).index((2, 0))
    rows[k] = rows[k].model_copy(update={"lo": float(np.nextafter(rows[k].lo, 0.0))})
    assert failing(good.inputs(ci_cmp=cal.model_copy(update={"paired": tuple(rows)}))) == {}


def test_the_ubuntu_compare_must_record_its_runtime(good: GoodSet) -> None:
    for missing in ("cpu_isa", "onnxruntime"):
        versions = VERSIONS.model_copy(update={missing: None})
        failed = failing(good.inputs(ci_cmp=good.cal(versions=versions)))
        assert set(failed) == {"G2", "G2b", "G3", "G4"}
        assert missing in failed["G2"][0]
    darwin = good.cal().model_copy(update={"platform": "darwin-arm64"})
    assert set(failing(good.inputs(ci_cmp=darwin))) == {"G2", "G2b", "G3", "G4"}


def test_without_the_ubuntu_half_only_the_local_compare_counts(good: GoodSet) -> None:
    """A §2.5 candidate is judged on its fresh local evaluations before push."""
    inputs = good.inputs(ci_cmp=None, ci_results=None, tree=None, report=None)
    assert failing(inputs, SHIP_GATES) == {}


def test_g5_needs_coverage(good: GoodSet) -> None:
    """A fixture cut to its first 100 rows has fewer than 300 unsaturated rows."""
    with np.load(io.BytesIO(good.shipped["fixture"])) as npz:
        members = {k: npz[k] for k in npz.files}
    for key in ("obs", "action_ref", "set"):
        members[key] = members[key][:100]
    buf = io.BytesIO()
    np.savez(buf, **members)
    failed = failing(good.inputs(fixture=buf.getvalue()))
    assert set(failed) == {"G5"}
    assert "unsaturated" in failed["G5"][0]


def test_g5_refuses_a_fixture_that_is_not_an_npz(good: GoodSet) -> None:
    failed = failing(good.inputs(fixture=b"PK\x05\x06" + b"\0" * 18))
    assert set(failed) == {"G5"}


def test_g6_needs_the_eval_v1_protocol(good: GoodSet) -> None:
    results = good.results().model_copy(update={"protocol_hash": "4" * 64})
    assert set(failing(good.inputs(results=results))) == {"G6"}


# --- G7 -------------------------------------------------------------------------------------------


def with_doc(good: GoodSet, edit: Callable[[dict[str, Any]], None]) -> gate.GateInputs:
    doc = good.doc()
    edit(doc)
    return good.inputs(report=report_of(doc))


def runs_of(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["summary"]["run"]: r["summary"] for r in doc["runs"]}


def drop_sac(doc: dict[str, Any]) -> None:
    doc["runs"] = [r for r in doc["runs"] if r["summary"]["run"] != "sac_s2"]
    doc["fresh"]["rows"] = [r for r in doc["fresh"]["rows"] if r["run"] != "sac_s2"]


def unmarked(doc: dict[str, Any]) -> None:
    for row in doc["fresh"]["rows"]:
        if row["run"] == "sac_s1":
            row["below_pid_gate"] = False


def no_fresh_row(doc: dict[str, Any]) -> None:
    doc["fresh"]["rows"] = [r for r in doc["fresh"]["rows"] if r["run"] != "sac_s0"]


def shipped_not_first(doc: dict[str, Any]) -> None:
    """ppo_s1 ranks below ppo_s0 on the selection seeds, and ppo_s0 has no failed attempt."""
    runs_of(doc)["ppo_s1"]["selection_history"][-1]["score"] = 83.0
    doc["shipped"]["attempts"] = [{"run": "ppo_s1", "failed_gates": []}]
    doc["shipped"]["run"] = "ppo_s1"


def wrong_entry(doc: dict[str, Any]) -> None:
    doc["fresh"]["rows"][1]["entry"] = "ppo_s2"


def ineligible_not_report_only(doc: dict[str, Any]) -> None:
    for row in doc["fresh"]["rows"]:
        if row["run"] == "sac_s0":
            row["report_only"] = False


def shipped_is_another_policy(doc: dict[str, Any]) -> None:
    doc["shipped"]["policy_sha256"] = "5" * 64


def best_not_by_the_pid_rule(doc: dict[str, Any]) -> None:
    runs_of(doc)["ppo_s1"]["best"]["rule"] = "none"


def no_ablation(doc: dict[str, Any]) -> None:
    """Without the ablation, G10 could not judge it (§2.3: PPO, SAC and ablation)."""
    doc["runs"] = [r for r in doc["runs"] if r["role"] != "ablation"]


G7_EDITS = {
    "two_sac_runs": (drop_sac, "SAC"),
    "below_pid_gate_unmarked": (unmarked, "below_pid_gate"),
    "no_fresh_row": (no_fresh_row, "fresh"),
    "shipped_is_not_the_first_candidate": (shipped_not_first, "section 2.5"),
    "row_names_another_entry": (wrong_entry, "entry"),
    "ineligible_exported_without_report_only": (ineligible_not_report_only, "report-only"),
    "shipped_is_another_policy": (shipped_is_another_policy, "submission"),
    "best_not_by_the_pid_rule": (best_not_by_the_pid_rule, "best.rule"),
    "no_ablation": (no_ablation, "0 ablation runs, not 1"),
}


@pytest.mark.parametrize("name", G7_EDITS)
def test_g7_names_a_bad_run_table(name: str, good: GoodSet) -> None:
    edit, words = G7_EDITS[name]
    failed = failing(with_doc(good, edit))
    assert set(failed) == {"G7"}, failed
    assert any(words in p for p in failed["G7"]), failed["G7"]


def test_g7_accepts_a_recorded_fallback(good: GoodSet) -> None:
    """ppo_s0 failed G2b; the next candidate, ppo_s1, shipped: §2.5's record."""
    doc = phase2_doc(
        good.subject, shipped_run="ppo_s1", attempts=[("ppo_s0", ["G2b"]), ("ppo_s1", [])]
    )
    assert failing(good.inputs(report=report_of(doc))) == {}
    unrecorded = phase2_doc(good.subject, shipped_run="ppo_s1", attempts=[("ppo_s1", [])])
    failed = failing(good.inputs(report=report_of(unrecorded)))
    assert set(failed) == {"G7"}
    assert any("section 2.5" in p and "'ppo_s0'" in p for p in failed["G7"]), failed


# --- G8 -------------------------------------------------------------------------------------------

G8_TREES = {
    "missing": {"sizes": {**dict.fromkeys(common.RESULT_FILES, 1000), "hero.mp4": None}},
    "over_its_cap": {"sizes": {**dict.fromkeys(common.RESULT_FILES, 1000), "hero.gif": 6_000_001}},
    "over_the_total": {"total_bytes": 20_000_001},
    "not_tracked": {"not_tracked": ("phase2/hero.mp4",)},
    "untracked_mp4": {"untracked": ("phase2/extra.mp4",)},
    "modified": {"modified": ("phase2.md",)},
    "symlinked_report": {"not_regular": ("docs/results/phase2.json",)},
}


@pytest.mark.parametrize("name", G8_TREES)
def test_g8_names_a_bad_results_package(name: str, good: GoodSet) -> None:
    tree = replace(good_tree(), **G8_TREES[name])
    failed = failing(good.inputs(tree=tree))
    assert set(failed) == {"G8"}


def test_g8_caps_are_the_section_5_2_caps() -> None:
    caps = common.RESULT_FILES
    assert caps["hero.mp4"] == 8_000_000
    assert caps["hero.gif"] == 6_000_000
    assert caps["gallery_1.gif"] == 2_000_000
    assert {"ci_ubuntu/results.json", "ci_ubuntu/cmp_cal.json"} <= set(caps)
    assert common.MAX_RESULTS_TOTAL_BYTES == 20_000_000


# --- G10 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("field", G10_FIELDS)
def test_g10_names_each_missing_field(field: str, good: GoodSet) -> None:
    failed = failing(with_doc(good, lambda doc: runs_of(doc)["sac_s1"].pop(field)))
    assert "G10" in failed
    assert set(failed) <= {"G7", "G10"}  # flags are also part of G7's run table
    assert any("sac_s1" in p and f"missing field {field!r}" in p for p in failed["G10"]), failed


def test_g10_names_any_other_missing_section_5_1_field(good: GoodSet) -> None:
    failed = failing(with_doc(good, lambda doc: runs_of(doc)["ppo_s2"].pop("bundle_hash")))
    assert set(failed) == {"G10"}
    assert "missing field 'bundle_hash'" in failed["G10"][0]


def test_g10_refuses_a_field_outside_section_5_1(good: GoodSet) -> None:
    failed = failing(with_doc(good, lambda doc: runs_of(doc)["ppo_s1"].update(rank=1)))
    assert set(failed) == {"G10"}


def set_field(run: str, **values: Any) -> Callable[[dict[str, Any]], None]:
    return lambda doc: runs_of(doc)[run].update(values)


G10_EDITS = {
    "stale_selection_hash_on_sac": (
        set_field("sac_s2", selection_protocol_hash="6" * 64),
        "sac_s2",
    ),
    "old_env_algo_version": (set_field("ablate_nopenalty", env_algo_version=0), "env_algo_version"),
    "abort_flag": (set_field("ablate_nopenalty", flags=["cpu_abort"]), "cpu_abort"),
    "nondeterministic": (
        set_field("sac_s0", flags=["no_eligible", "nondeterministic"]),
        "nondeterministic",
    ),
    "short_run": (set_field("sac_s1", num_timesteps=999_999), "num_timesteps"),
    "rss_at_the_bar": (set_field("ppo_s2", peak_tree_rss_mib=6144.0), "peak_tree_rss_mib"),
    "shipped_dirty": (set_field("ppo_s0", git_dirty=True), "git_dirty"),
    "stale_bundle_on_the_ablation": (
        set_field("ablate_nopenalty", bundle_hash="8" * 64),
        "bundle_hash",
    ),
}


@pytest.mark.parametrize("name", G10_EDITS)
def test_g10_names_a_bad_run(name: str, good: GoodSet) -> None:
    edit, words = G10_EDITS[name]
    failed = failing(with_doc(good, edit))
    assert set(failed) == {"G10"}, failed
    assert any(words in p for p in failed["G10"]), failed["G10"]


def test_g10_allows_a_dirty_tree_on_a_run_that_was_not_shipped(good: GoodSet) -> None:
    assert failing(with_doc(good, set_field("ppo_s1", git_dirty=True))) == {}


# --- G11 ------------------------------------------------------------------------------------------


def weak_ppo(doc: dict[str, Any], runs: Sequence[str]) -> None:
    """``runs`` play like PID in the fresh calibration: G2 fails for them."""
    cal = Comparison.model_validate_json(json.dumps(doc["fresh"]["calibration"]), strict=True)
    pid = cal.entries[0]
    entries = [
        e.model_copy(
            update={"scores": pid.scores, "success": pid.success, "termination": pid.termination}
        )
        if e.id in runs
        else e
        for e in cal.entries
    ]
    doc["fresh"]["calibration"] = json.loads(to_json(comparison(CAL, entries)))
    for row in doc["g11"]:
        if row["run"] in runs:
            row.update(g2=False, g2b=False)


def test_g11_needs_two_of_three_ppo_runs(good: GoodSet) -> None:
    assert failing(with_doc(good, lambda doc: weak_ppo(doc, ["ppo_s2"]))) == {}
    failed = failing(with_doc(good, lambda doc: weak_ppo(doc, ["ppo_s1", "ppo_s2"])))
    assert set(failed) == {"G11"}
    assert "1 of 3" in failed["G11"][0]


def make_ineligible(doc: dict[str, Any], run: str, *, score: float) -> None:
    """``run``'s best (its last checkpoint, now scoring ``score`` on the selection seeds) is
    below the PID gate, recorded consistently: flag, fresh row and G11 row."""
    s = runs_of(doc)[run]
    s["selection_history"][-1].update(eligible=False, score=score)
    s["best"]["eligible"] = False
    s["flags"] = ["no_eligible"]
    for row in doc["fresh"]["rows"]:
        if row["run"] == run:
            row.update(report_only=True, below_pid_gate=True)
    for row in doc["g11"]:
        if row["run"] == run:
            row["eligible"] = False


def test_g11_counts_only_runs_with_an_eligible_best(good: GoodSet) -> None:
    """ppo_s1 and ppo_s2 pass G2, G3 and G4 on 0..199 but their bests are not eligible: only
    ppo_s0 counts. ppo_s1 ranks first on the selection seeds, so a §2.5 replay that ignored
    eligibility would ship it and G7 would fail too."""

    def two_ineligible(doc: dict[str, Any]) -> None:
        make_ineligible(doc, "ppo_s1", score=99.0)
        make_ineligible(doc, "ppo_s2", score=83.0)

    failed = failing(with_doc(good, two_ineligible))
    assert set(failed) == {"G11"}, failed
    assert "1 of 3" in failed["G11"][0]


def test_g11_table_must_match_the_recomputation(good: GoodSet) -> None:
    def lie(doc: dict[str, Any]) -> None:
        weak_ppo(doc, ["ppo_s2"])
        doc["g11"][2]["g2"] = True

    failed = failing(with_doc(good, lie))
    assert set(failed) == {"G11"}
    assert "ppo_s2" in failed["G11"][0]


def test_g11_g2b_is_reported_not_gating(good: GoodSet) -> None:
    def lqr_level(doc: dict[str, Any]) -> None:
        for row in doc["g11"]:
            row["g2b"] = False
        cal = Comparison.model_validate_json(json.dumps(doc["fresh"]["calibration"]), strict=True)
        lqr = cal.entries[1]
        entries = [
            e.model_copy(
                update={
                    "scores": lqr.scores,
                    "success": lqr.success,
                    "termination": lqr.termination,
                    "energy_j": lqr.energy_j,
                    "saturation_fraction": lqr.saturation_fraction,
                }
            )
            if e.id in PPO_RUNS
            else e
            for e in cal.entries
        ]
        doc["fresh"]["calibration"] = json.loads(to_json(comparison(CAL, entries)))

    assert failing(with_doc(good, lqr_level)) == {}


# --- §2.5: candidates and the fallback ------------------------------------------------------------


def cores(doc: Mapping[str, Any]) -> list[common.RunCore]:
    return [common.run_core(entry) for entry in report_of(doc).runs]


def test_candidates_are_eligible_clean_ppo_runs_by_selection_rank(good: GoodSet) -> None:
    doc = good.doc()
    ppo_only = [c for c in cores(doc) if c.run in PPO_RUNS]
    assert common.candidates(ppo_only) == ["ppo_s0", "ppo_s1", "ppo_s2"]
    runs_of(doc)["ppo_s2"]["selection_history"][-1]["score"] = 99.0
    runs_of(doc)["ppo_s0"]["flags"] = ["wall_clock_abort"]
    ppo_only = [c for c in cores(doc) if c.run in PPO_RUNS]
    assert common.candidates(ppo_only) == ["ppo_s2", "ppo_s1"]
    doc = good.doc()
    make_ineligible(doc, "ppo_s1", score=99.0)  # ranks first, but its best is not eligible
    ppo_only = [c for c in cores(doc) if c.run in PPO_RUNS]
    assert common.candidates(ppo_only) == ["ppo_s0", "ppo_s2"]


def test_the_section_2_5_fallback_ships_the_next_candidate_on_a_g2b_only_failure(
    good: GoodSet,
) -> None:
    """ppo_s0's fresh evaluation fails G2b alone; ppo_s1 passes every ship gate and ships.
    Failures outside ``SHIP_GATES`` (G8 here) never block shipping."""
    evaluated: list[str] = []

    def ship_failures(run: str) -> set[str]:
        evaluated.append(run)
        inputs = g2b_only(good) if run == "ppo_s0" else good.inputs()
        return set(failing(inputs, SHIP_GATES)) | {"G8"}

    choice = common.choose_shipped(["ppo_s0", "ppo_s1", "ppo_s2"], ship_failures)
    assert choice.run == "ppo_s1"
    assert choice.attempts == (("ppo_s0", ("G2b",)), ("ppo_s1", ()))
    assert evaluated == ["ppo_s0", "ppo_s1"]  # ppo_s2 is never evaluated
    nothing = common.choose_shipped(["ppo_s0"], lambda _run: set(SHIP_GATES))
    assert nothing.run is None
    assert nothing.attempts == (("ppo_s0", SHIP_GATES),)


# --- reading the inputs from files ----------------------------------------------------------------


def git(root: Path, *args: str) -> None:
    """``git`` in ``root`` with neither the developer's global or system config nor their hooks
    (commit signing or a ``core.hooksPath`` pre-commit hook would fail or run foreign code)."""
    env = {**common.git_env(), "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            f"core.hooksPath={os.devnull}",
            *args,
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        timeout=60,
    )


@dataclass(frozen=True)
class Files:
    root: Path
    argv: list[str]
    results_dir: Path


def write_files(good: GoodSet, root: Path) -> Files:
    """A checkout-like tree: out/ with the local evidence and docs/results/ committed to git."""
    out = root / "out"
    (out / "ppo_eval_v1").mkdir(parents=True)
    (out / "cmp_cal.json").write_text(to_json(good.cal()))
    (out / "cmp_eval_v1.json").write_text(to_json(good.eval_v1()))
    write_results(good.results(), out / "ppo_eval_v1" / "results.json")
    docs = root / "docs" / "results"
    results_dir = docs / "phase2"
    for rel in common.RESULT_FILES:
        (results_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (results_dir / rel).write_bytes(b"x" * 100)
    write_results(good.results(), results_dir / "ci_ubuntu" / "results.json")
    (results_dir / "ci_ubuntu" / "cmp_cal.json").write_text(to_json(good.cal()))
    (docs / "phase2.json").write_text(json.dumps(good.doc(), indent=1))
    (docs / "phase2.md").write_text("# Phase 2\n")
    (root / ".gitignore").write_text("*.mp4\n!docs/results/**/*.mp4\n/out/\n")
    git(root, "init", "-q")
    git(root, "add", ".gitignore", "docs")
    git(root, "commit", "-q", "-m", "results")
    argv = [
        str(out / "cmp_cal.json"),
        str(out / "cmp_eval_v1.json"),
        str(out / "ppo_eval_v1" / "results.json"),
        str(docs / "phase2.json"),
        "--ci-dir",
        str(results_dir / "ci_ubuntu"),
        "--submission",
        str(good.shipped["dir"]),
        "--results-dir",
        str(results_dir),
        "--fixture",
        str(good.shipped["fixture_path"]),
    ]
    return Files(root, argv, results_dir)


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@needs_git
def test_the_cli_passes_a_good_set_from_files(
    good: GoodSet, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = write_files(good, tmp_path)
    assert gate.main(files.argv) == 0
    out = capsys.readouterr().out
    for g in PASS_GATES:
        assert re.search(rf"^{g}\s+PASS$", out, re.MULTILINE), out
    assert "all 11 gates pass" in out


@needs_git
def test_the_cli_names_each_failing_gate_and_exits_1(
    good: GoodSet, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = write_files(good, tmp_path)
    (files.results_dir / "ci_ubuntu" / "cmp_cal.json").write_text(
        to_json(good.cal(ppo=Perf(87.0, 60.0, 188, 13.2, 0.03)))
    )
    git(files.root, "commit", "-q", "-am", "ubuntu evidence")
    (files.root / "out" / "cmp_eval_v1.json").unlink()
    assert gate.main(files.argv) == 1
    out = capsys.readouterr().out
    assert re.search(r"^G4\s+FAIL\s+ubuntu", out, re.MULTILINE), out
    assert re.search(r"^G1\s+FAIL\s+.*cmp_eval_v1\.json", out, re.MULTILINE), out
    assert "3 of 11 gates fail: G1, G3, G4" in out


@needs_git
def test_g8_sees_an_untracked_mp4_from_git(good: GoodSet, tmp_path: Path) -> None:
    files = write_files(good, tmp_path)
    (files.results_dir / "pid_vs_ppo_seed7.mp4").write_bytes(b"x" * 10)
    tree = gate.scan_results(files.results_dir, files.root / "docs" / "results" / "phase2.json")
    assert isinstance(tree, gate.ResultsTree)
    assert tree.untracked == ("docs/results/phase2/pid_vs_ppo_seed7.mp4",)
    failed = failing(good.inputs(tree=tree))
    assert set(failed) == {"G8"}
    assert "untracked" in failed["G8"][0]


@needs_git
def test_g8_sees_an_untracked_required_file_and_a_modified_one(
    good: GoodSet, tmp_path: Path
) -> None:
    files = write_files(good, tmp_path)
    git(files.root, "rm", "-q", "--cached", "docs/results/phase2/hero.mp4")
    git(files.root, "commit", "-q", "-m", "untrack hero")
    (files.root / "docs" / "results" / "phase2.md").write_text("# edited\n")
    tree = gate.scan_results(files.results_dir, files.root / "docs" / "results" / "phase2.json")
    assert isinstance(tree, gate.ResultsTree)
    assert "docs/results/phase2/hero.mp4" in tree.not_tracked
    assert tree.modified == ("docs/results/phase2.md",)
    problems = failing(good.inputs(tree=tree))["G8"]
    assert any("hero.mp4" in p and "not tracked" in p for p in problems)
    assert any("phase2.md" in p and "modified" in p for p in problems)


@needs_git
def test_g8_measures_sizes_without_following_symlinks(good: GoodSet, tmp_path: Path) -> None:
    files = write_files(good, tmp_path)
    hero = files.results_dir / "hero.gif"
    hero.unlink()
    hero.symlink_to(files.results_dir / "hero.mp4")
    tree = gate.scan_results(files.results_dir, files.root / "docs" / "results" / "phase2.json")
    assert isinstance(tree, gate.ResultsTree)
    assert tree.sizes["hero.gif"] is None


@needs_git
@pytest.mark.parametrize("name", ["phase2.json", "phase2.md"])
def test_g8_refuses_a_committed_symlink_for_the_report(
    name: str, good: GoodSet, tmp_path: Path
) -> None:
    """A committed symlink into the ignored out/ would let the gate read (and phase2.md show)
    bytes git never sees, however they change: like the results files, it counts as missing."""
    files = write_files(good, tmp_path)
    docs = files.root / "docs" / "results"
    (files.root / "out" / name).write_bytes((docs / name).read_bytes())
    (docs / name).unlink()
    (docs / name).symlink_to(files.root / "out" / name)
    git(files.root, "add", f"docs/results/{name}")
    git(files.root, "commit", "-q", "-m", "symlink")
    tree = gate.scan_results(files.results_dir, docs / "phase2.json")
    assert isinstance(tree, gate.ResultsTree)
    assert tree.not_regular == (f"docs/results/{name}",)
    assert (tree.not_tracked, tree.untracked, tree.modified) == ((), (), ())
    failed = failing(good.inputs(tree=tree))
    assert set(failed) == {"G8"}
    message = f"docs/results/{name}: not a regular file (a symlink counts as missing)"
    assert failed["G8"] == (message,)


def test_scanning_outside_a_git_checkout_is_an_input_error(tmp_path: Path) -> None:
    env_ceiling = {"GIT_CEILING_DIRECTORIES": str(tmp_path)}
    (tmp_path / "phase2").mkdir()
    result = gate.scan_results(tmp_path / "phase2", tmp_path / "phase2.json", extra_env=env_ceiling)
    assert isinstance(result, gate.InputError)


def test_inputs_are_read_strictly(
    good: GoodSet, tmp_path: Path, make_fifo: Callable[[Path], Path]
) -> None:
    """``format`` required, exact JSON types, never blocking on a FIFO."""
    doc = good.doc()
    del doc["format"]
    path = tmp_path / "phase2.json"
    path.write_text(json.dumps(doc))
    assert isinstance(gate.read_report(path), gate.InputError)
    doc = good.doc()
    doc["g11"][0]["g2"] = 1  # an int is not a bool
    path.write_text(json.dumps(doc))
    error = gate.read_report(path)
    assert isinstance(error, gate.InputError)
    assert "g11.0.g2" in error.message
    doc = good.doc()
    del doc["fresh"]["calibration"]["format"]
    path.write_text(json.dumps(doc))
    assert "format" in gate.read_report(path).message  # type: ignore[union-attr]
    fifo = make_fifo(tmp_path / "fifo.json")
    assert isinstance(gate.read_report(fifo), gate.InputError)
    assert isinstance(gate.read_compare(fifo), gate.InputError)


def test_error_text_never_echoes_input_values(good: GoodSet, tmp_path: Path) -> None:
    doc = good.doc()
    doc["shipped"]["run"] = "\x1b[31mred‮"
    path = tmp_path / "phase2.json"
    path.write_text(json.dumps(doc))
    error = gate.read_report(path)
    assert isinstance(error, gate.InputError)
    assert "\x1b" not in error.message
    assert "‮" not in error.message
    assert "red" not in error.message


def put(path: Sequence[str | int], value: object) -> Callable[[dict[str, Any]], None]:
    def edit(doc: dict[str, Any]) -> None:
        target: Any = runs_of(doc) if path[0] == "runs" else doc
        for key in path[1:-1] if path[0] == "runs" else path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


@pytest.mark.parametrize(
    ("where", "value", "location"),
    [
        (("runs", "sac_s1", "peak_tree_rss_mib"), math.nan, "runs.4.summary"),
        (("runs", "ppo_s0", "selection_history", 0, "score"), math.inf, "runs.0.summary"),
        (("pilot", "energy_cost_variant", "score"), -math.inf, "pilot"),
        (("render", "curve"), [1.0, math.nan], "render"),
        (("render", "ci_run"), 2**53, "render"),
    ],
    ids=["nan_in_a_summary", "inf_in_a_history_row", "inf_in_pilot", "nan_in_render", "big_int"],
)
def test_phase2_json_refuses_non_finite_and_inexact_numbers(
    where: Sequence[str | int], value: object, location: str, good: GoodSet, tmp_path: Path
) -> None:
    """Contract §15: no NaN or Infinity, also in the free-form objects (Python's json writes
    NaN by default); integers must be exact in a JSON double. The file is an input error that
    names where, so G7, G10 and G11 fail by name, never with a traceback."""
    doc = good.doc()
    put(where, value)(doc)
    path = tmp_path / "phase2.json"
    path.write_text(json.dumps(doc))
    error = gate.read_report(path)
    assert isinstance(error, gate.InputError), error
    assert location in error.message
    assert "not finite" in error.message or "2**53" in error.message
    assert set(failing(good.inputs(report=error))) == {"G7", "G10", "G11"}
    path.write_text(json.dumps(good.doc()).replace('"score": 64.9', '"score": 1e999'))
    assert isinstance(gate.read_report(path), gate.InputError)


def test_a_summary_that_cannot_be_reencoded_names_its_run(good: GoodSet) -> None:
    """Past the parser (a report built in code, say), a NaN in a summary still fails G7 and
    G10 by name instead of raising."""
    report = report_of(good.doc())
    runs = list(report.runs)
    runs[5] = runs[5].model_copy(
        update={"summary": {**runs[5].summary, "peak_tree_rss_mib": math.nan}}
    )
    failed = failing(good.inputs(report=report.model_copy(update={"runs": tuple(runs)})))
    assert set(failed) == {"G7", "G10"}
    assert all(any("'sac_s2'" in p for p in failed[g]) for g in failed), failed


def test_a_summary_error_says_what_is_wrong(good: GoodSet) -> None:
    """The summary validators' own reasons (fixed text and a number) are named, not just
    ``value_error``."""
    failed = failing(with_doc(good, lambda doc: runs_of(doc)["ppo_s1"]["best"].update(step=12)))
    assert set(failed) == {"G7", "G10", "G11"}
    for problems in failed.values():
        assert any(
            "'ppo_s1'" in p and "best.step 12 must be exactly one selection_history step" in p
            for p in problems
        ), problems
    failed = failing(with_doc(good, set_field("sac_s0", flags=["no_eligible", "no_eligible"])))
    assert set(failed) == {"G10"}
    assert "flags must be unique" in failed["G10"][0]


# --- phase2_ci_check.py ---------------------------------------------------------------------------

HEAD = "a" * 40
FAKE_GH = """#!/bin/sh
# A fake gh: records its arguments, prints a canned answer, never touches the network.
printf '%s\\n' "$*" >> "$FAKE_GH_LOG"
case "$1 $2" in
  "run list") cat "$FAKE_GH_RUNS"; exit "${FAKE_GH_STATUS:-0}" ;;
  "run view") cat "$FAKE_GH_JOBS"; exit 0 ;;
esac
exit 64
"""
JOBS_OK = {
    "jobs": [
        {"name": n, "status": "completed", "conclusion": "success"} for n in ci_check.REQUIRED_JOBS
    ]
}


@pytest.fixture
def fake_gh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., subprocess.CompletedProcess[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GH_LOG", str(tmp_path / "gh.log"))

    def run(
        runs: object, jobs: object = JOBS_OK, *, status: int = 0, raw: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        (tmp_path / "runs.json").write_text(raw if raw is not None else json.dumps(runs))
        (tmp_path / "jobs.json").write_text(json.dumps(jobs))
        env = {
            **os.environ,
            "FAKE_GH_RUNS": str(tmp_path / "runs.json"),
            "FAKE_GH_JOBS": str(tmp_path / "jobs.json"),
            "FAKE_GH_STATUS": str(status),
        }
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "phase2_ci_check.py"),
                "--head",
                HEAD,
                "--workflow",
                "ci.yml",
                "--branch",
                "phase-2-learn",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )

    return run


def ci_run(
    run_id: int, sha: str = HEAD, status: str = "completed", conclusion: str = "success"
) -> dict[str, Any]:
    return {"databaseId": run_id, "status": status, "conclusion": conclusion, "headSha": sha}


needs_posix_sh = pytest.mark.skipif(os.name == "nt", reason="the fake gh is a POSIX shell script")


@needs_posix_sh
def test_the_ci_check_passes_a_successful_run_for_head(
    fake_gh: Callable[..., Any], tmp_path: Path
) -> None:
    result = fake_gh([ci_run(7, sha="b" * 40, conclusion="failure"), ci_run(9)])
    assert result.returncode == 0, result.stderr
    assert "run 9" in result.stdout
    calls = (tmp_path / "gh.log").read_text().splitlines()
    assert (
        calls[0] == "run list --branch phase-2-learn --workflow ci.yml "
        "--json databaseId,status,conclusion,headSha --limit 100"
    )
    assert calls[1] == "run view 9 --json jobs"


CI_FAILURES = {
    "no_run": ([ci_run(5, sha="c" * 40)], ci_check.NO_RUN),
    "in_progress": ([ci_run(5, status="in_progress", conclusion="")], ci_check.IN_PROGRESS),
    "queued": ([ci_run(5, status="queued", conclusion="")], ci_check.IN_PROGRESS),
    "cancelled": ([ci_run(5, conclusion="cancelled")], ci_check.CANCELLED),
    "failure": ([ci_run(5, conclusion="failure")], ci_check.FAILED),
    "timed_out": ([ci_run(5, conclusion="timed_out")], ci_check.FAILED),
    "newest_failed": ([ci_run(5), ci_run(6, conclusion="failure")], ci_check.FAILED),
}


@needs_posix_sh
@pytest.mark.parametrize("name", CI_FAILURES)
def test_the_ci_check_fails_with_distinct_messages(name: str, fake_gh: Callable[..., Any]) -> None:
    runs, message = CI_FAILURES[name]
    result = fake_gh(runs)
    assert result.returncode == 1
    assert message in result.stderr


def test_the_four_ci_messages_are_distinct() -> None:
    messages = (ci_check.NO_RUN, ci_check.IN_PROGRESS, ci_check.CANCELLED, ci_check.FAILED)
    assert len(set(messages)) == 4
    assert not any(a in b for a in messages for b in messages if a != b)


@needs_posix_sh
def test_the_ci_check_needs_all_three_jobs(fake_gh: Callable[..., Any]) -> None:
    jobs = copy.deepcopy(JOBS_OK)
    jobs["jobs"] = [j for j in jobs["jobs"] if j["name"] != "python-train"]
    result = fake_gh([ci_run(9)], jobs)
    assert result.returncode == 1
    assert "python-train" in result.stderr


@needs_posix_sh
@pytest.mark.parametrize(
    ("raw", "status"),
    [("not json", 0), ("[]", 1), (json.dumps([{"databaseId": "x"}]), 0), ("\x1b[2J" * 10, 0)],
    ids=["not_json", "gh_failed", "bad_rows", "escape_codes"],
)
def test_the_ci_check_refuses_what_gh_cannot_answer(
    raw: str, status: int, fake_gh: Callable[..., Any]
) -> None:
    result = fake_gh(None, raw=raw, status=status)
    assert result.returncode == 2
    assert "\x1b" not in result.stderr


def test_the_ci_check_bounds_gh_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(args: Sequence[str], **kw: Any) -> common.CommandResult:
        calls.append({"args": list(args), **kw})
        return common.CommandResult(0, json.dumps([ci_run(3)]).encode(), "")

    monkeypatch.setattr(ci_check, "run_command", fake_run)
    monkeypatch.setattr(ci_check, "jobs_problems", lambda *_a, **_k: [])
    assert ci_check.main(["--head", HEAD, "--workflow", "ci.yml", "--branch", "main"]) == 0
    assert calls[0]["timeout_s"] <= 120
    assert calls[0]["max_bytes"] <= 1 << 20


def test_run_command_kills_a_command_that_prints_too_much() -> None:
    with pytest.raises(common.CommandError, match="printed more than"):
        common.run_command(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 100000)"], max_bytes=1000
        )
    with pytest.raises(common.CommandError, match="longer than"):
        common.run_command([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=0.5)
    with pytest.raises(common.CommandError, match="cannot run"):
        common.run_command(["/nonexistent/gh"])
    ok = common.run_command(
        [sys.executable, "-c", "print('hi'); import sys; print('e', file=sys.stderr)"]
    )
    assert (ok.returncode, ok.stdout.strip(), ok.stderr.strip()) == (0, b"hi", "e")


# --- phase2_done.sh -------------------------------------------------------------------------------


DONE_SH = SCRIPTS / "phase2_done.sh"
DONE_SH_MARKER = "# docs/design/phase-2.md section 4 (test_phase2_gate checks these lines):"
SHELLCHECK_DIRECTIVE = "# shellcheck disable=SC2086  # section 4 splits $S into three submissions"
REEXPORT = 'uv run topple export "runs/$K" --out "$D"'
REBUILD_CMP = 'cmp "$D/policy.onnx" submissions/ppo_balance/policy.onnx'


def spec_block() -> list[str]:
    """The bash block of docs/design/phase-2.md §4, line by line."""
    text = (REPO / "docs" / "design" / "phase-2.md").read_text()
    section = text[text.index("## 4. Done-when for 2a") :]
    block = section[section.index("```bash\n") + len("```bash\n") :]
    return block[: block.index("```")].splitlines()


def section_4_commands() -> list[str]:
    """§4's block, with its ``export ... && cmp ...`` line as two commands: errexit ignores a
    failure on the left of ``&&``, so a failing re-export would skip the ``cmp`` and the script
    would carry on (the spec block is being changed to two lines; then this is the identity)."""
    commands: list[str] = []
    for line in spec_block():
        if line == f"{REEXPORT} && {REBUILD_CMP}":
            commands += [REEXPORT, REBUILD_CMP]
        elif line.strip():
            commands.append(line)
    return commands


def done_sh_body() -> list[str]:
    lines = DONE_SH.read_text().splitlines()
    return [line for line in lines[lines.index(DONE_SH_MARKER) + 1 :] if line.strip()]


def test_phase2_done_runs_exactly_the_section_4_commands() -> None:
    lines = DONE_SH.read_text().splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert "set -euo pipefail" in lines
    assert done_sh_body() == section_4_commands()
    commands = "\n".join(section_4_commands())
    for needed in (
        "UV_PROJECT_ENVIRONMENT=.venv-eval",
        'find_spec("torch")',
        "topple eval submissions/ppo_balance -p eval_v1",
        "--json out/cmp_cal.json",
        REEXPORT,
        REBUILD_CMP,
        "scripts/phase2_ci_check.py",
        "scripts/phase2_gate.py",
    ):
        assert needed in commands


def test_phase2_done_never_hides_a_failure_from_errexit() -> None:
    """``set -e`` does not stop on a failing command left of ``&&`` or ``||``, nor (without
    ``pipefail``) inside a pipeline, so the done-when runs only simple commands."""
    for line in done_sh_body():
        assert "&&" not in line, line
        assert "||" not in line, line
    assert done_sh_body().index(REEXPORT) + 1 == done_sh_body().index(REBUILD_CMP)


def test_phase2_done_is_valid_bash() -> None:
    assert os.access(DONE_SH, os.X_OK)
    # §4's block splits $S into three submissions on purpose; shellcheck (preinstalled on CI's
    # ubuntu-24.04, 0.9.0) reports that as SC2086 and exits 1 unless the file disables it. The
    # directive must directly follow the shebang to apply to the whole file.
    assert DONE_SH.read_text().splitlines()[1] == SHELLCHECK_DIRECTIVE
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    result = subprocess.run(
        [bash, "-n", str(DONE_SH)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    shellcheck = shutil.which("shellcheck")
    if shellcheck is not None:
        checked = subprocess.run(
            [shellcheck, str(DONE_SH)], capture_output=True, text=True, timeout=60, check=False
        )
        assert checked.returncode == 0, checked.stdout


STUB_UV = """#!/bin/sh
# A stand-in uv: logs its arguments, runs nothing. `topple export` follows $STUB_EXPORT.
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *'["shipped"]["run"]'*) echo ppo_s0 ;;
  "run topple export "*)
    case "$STUB_EXPORT" in
      fail) echo "error: runs/ppo_s0 is not a run directory" >&2; exit 2 ;;
      same) mkdir -p "$6" && cp submissions/ppo_balance/policy.onnx "$6/policy.onnx" ;;
      other) mkdir -p "$6" && printf 'other bytes' > "$6/policy.onnx" ;;
    esac ;;
esac
exit 0
"""
STUB_GIT = """#!/bin/sh
# A stand-in git: the checkout is $STUB_TOP, clean, at a fixed HEAD.
case "$*" in
  "rev-parse --show-toplevel") echo "$STUB_TOP" ;;
  "rev-parse HEAD") echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;
esac
exit 0
"""


@needs_posix_sh
@pytest.mark.parametrize(
    ("export", "passes"),
    [("same", True), ("fail", False), ("other", False)],
    ids=["rebuilds_the_shipped_bytes", "export_fails", "export_rebuilds_other_bytes"],
)
def test_phase2_done_stops_unless_the_re_export_rebuilds_the_shipped_bytes(
    export: str, passes: bool, tmp_path: Path
) -> None:
    """With stand-in ``uv`` and ``git`` on ``PATH`` (nothing is installed or evaluated): a
    re-export that fails (``runs/`` is gitignored, so another checkout has none) or rebuilds
    other bytes stops the script before G9 and the gates; the same bytes let it through."""
    bash, cmp = shutil.which("bash"), shutil.which("cmp")
    if bash is None or cmp is None:
        pytest.skip("needs bash and cmp")
    bin_dir, top, scratch = tmp_path / "bin", tmp_path / "checkout", tmp_path / "tmp"
    for d in (bin_dir, top / "submissions" / "ppo_balance", scratch):
        d.mkdir(parents=True)
    (top / "submissions" / "ppo_balance" / "policy.onnx").write_bytes(b"shipped onnx bytes")
    for name, text in (("uv", STUB_UV), ("git", STUB_GIT)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "calls.log"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "TMPDIR": str(scratch),
        "STUB_LOG": str(log),
        "STUB_TOP": str(top),
        "STUB_EXPORT": export,
    }
    result = subprocess.run(
        [bash, str(DONE_SH)], capture_output=True, text=True, timeout=60, env=env, check=False
    )
    calls = log.read_text()
    assert "run topple export runs/ppo_s0 --out " in calls
    if passes:
        assert result.returncode == 0, result.stderr
        assert "scripts/phase2_ci_check.py" in calls
        assert "scripts/phase2_gate.py" in calls
    else:
        assert result.returncode != 0
        assert "phase2_ci_check.py" not in calls
        assert "phase2_gate.py" not in calls
