"""The Phase 2b checks both ``phase2b_results.py`` and ``phase2b_gate.py`` compute from a
``topple.compare/1`` file (docs/design/phase-2.md §8.7, §8.9 items 1 and 6), so the choice the
results make is exactly what the gate replays.

Rates come from an entry's waypoint extras (``compare.WAYPOINT_EXTRAS``): ``docked`` and
``fell`` per seed, ``fall_cause`` for the split. An episode without metrics (an error) is neither
docked nor fallen; every check that judges a policy also requires that it has no errors.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from phase2_common import RunSummary
from topple.challenge import load_challenge
from topple.compare import SCORELESS, Comparison, Entry
from topple.evaluate import FALL_REASONS, EvalProtocol
from topple.export import (
    MIN_SATURATED,
    MIN_UNSATURATED,
    PARITY_TOLERANCE,
    batch1_max_abs_diff,
    coverage_counts,
    read_parity_fixture,
)
from topple.policy.manifest import policy_task
from topple.policy.runtime import load_policy_dir
from topple.results import MAX_RESULTS_BYTES
from topple.stats import paired_bootstrap
from topple.waypoint import SCORE_PARTS

PHASE2B_FORMAT: Final = "topple.phase2b/1"
PHASE2B_JSON: Final = "phase2b.json"
PHASE2B_MD: Final = "phase2b.md"
SELECTION_SEEDS: Final = tuple(range(2000, 2200))
"""``ppo_waypoint.yaml``'s selection seeds (§8.8)."""
EPISODE_S: Final = load_challenge("waypoint-v1").config.episode_seconds

DRIVER_ID: Final = "waypoint_pid_example"
CREEPER_ID: Final = "creeper"
ANCHOR_IDS: Final = ("balance_in_place", "spin", "creeper", "gates_only", "reckless")

MIN_DOCKED: Final = 0.60
"""§8.9 item 6 (a)."""
MIN_PAIRED_LO: Final = -5.0
"""§8.9 item 6 (b)."""
FALL_MARGIN: Final = 0.20
"""§8.9 item 6 (d)."""
FLOAT_SLACK: Final = 1e-9
"""Absorbs the rounding of a rate sum, as ``topple.train.selection.FLOAT_SLACK`` does."""

BAND_FELL: Final = (0.03, 0.10)
"""§8.7 step 1 / §8.9 item 1: the driver's fall rate on standard 0..399."""
BAND_MIN_DOCKED: Final = 0.88
FLAT_ANCHOR_MAX: Final = 25.0
"""Balance in place and spin in place score at most this (§8.7 table)."""
CREEPER_GAP: Final = 10.0
GATES_ONLY_GAP: Final = 5.0


class Phase2bError(Exception):
    """A comparison the 2b checks cannot judge: one line naming what is wrong."""


@dataclass(frozen=True)
class Rates:
    """One entry's rates on a comparison's seeds."""

    n: int
    errors: int
    score: float
    docked: float
    fell: float
    falls_by_cause: dict[str, int]
    median_energy_j: float
    median_t_finish_s: float
    parts: dict[str, float]
    """The mean of each score part (§8.6) over the episodes that have one."""

    def to_json(self) -> dict[str, object]:
        return {
            "n": self.n,
            "errors": self.errors,
            "score": self.score,
            "docked": self.docked,
            "fell": self.fell,
            "falls_by_cause": dict(self.falls_by_cause),
            "median_energy_j": self.median_energy_j,
            "median_t_finish_s": self.median_t_finish_s,
            "parts": dict(self.parts),
        }


def entry(cmp: Comparison, entry_id: str) -> Entry:
    """The entry named ``entry_id``; ``Phase2bError`` if there is none or the run is not a
    complete waypoint comparison."""
    if not cmp.complete:
        raise Phase2bError(f"{cmp.protocol}: the comparison is incomplete")
    found = [e for e in cmp.entries if e.id == entry_id]
    if len(found) != 1:
        raise Phase2bError(f"{cmp.protocol}: no single entry {entry_id!r}")
    for name in ("docked", "fell", "fall_cause", "t_finish_s"):
        if name not in found[0].extras:
            raise Phase2bError(f"{cmp.protocol}: {entry_id} records no {name!r} per seed")
    return found[0]


def rates(e: Entry) -> Rates:
    """``e``'s rates. Falls come from the validated ``termination`` array (contract §11); the
    extras must agree with it (``fell`` per seed, and every success docked), so a forged or
    inconsistent extras column is refused, never counted. An error is any scoreless episode
    (``error``, ``sim_unstable``, ``missing``)."""
    n = len(e.scores)
    if n == 0:
        raise Phase2bError(f"{e.id}: no episodes")
    docked = e.extras["docked"]
    fell = [t in FALL_REASONS for t in e.termination]
    for k, (f, x, d, ok) in enumerate(zip(fell, e.extras["fell"], docked, e.success, strict=True)):
        if (x is not None and x is not f) or (ok and d is not True):
            raise Phase2bError(f"{e.id}: seed index {k}: extras disagree with the termination")
    causes = Counter(str(c) for c, f in zip(e.extras["fall_cause"], fell, strict=True) if f)
    return Rates(
        n=n,
        errors=sum(t in SCORELESS for t in e.termination),
        score=math.fsum(e.scores) / n,
        docked=sum(d is True for d in docked) / n,
        fell=sum(fell) / n,
        falls_by_cause=dict(sorted(causes.items())),
        median_energy_j=median(e.energy_j),
        median_t_finish_s=median(  # an errored episode is a non-finisher (§8.6 ranking)
            float(t) if isinstance(t, (int, float)) else EPISODE_S for t in e.extras["t_finish_s"]
        ),
        parts={k: _mean_present(e.extras.get(k, ())) for k in SCORE_PARTS},
    )


def _mean_present(values: Sequence[object]) -> float:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return math.fsum(xs) / len(xs) if xs else 0.0


def item6(cmp: Comparison, ppo_id: str) -> dict[str, Any]:
    """§8.9 item 6 (a)-(d) for ``ppo_id`` against the driver and the creeper in ``cmp``, plus
    the requirement that none of the three erred."""
    ppo, driver, creeper = (rates(entry(cmp, i)) for i in (ppo_id, DRIVER_ID, CREEPER_ID))
    mean, lo, hi = paired_bootstrap(entry(cmp, ppo_id).scores, entry(cmp, DRIVER_ID).scores)
    checks = {
        "a_docked": ppo.docked >= MIN_DOCKED,
        "b_paired_lo": lo > MIN_PAIRED_LO,
        "c_above_creeper": ppo.score > creeper.score,
        "d_falls": ppo.fell <= driver.fell + FALL_MARGIN + FLOAT_SLACK,
        "no_errors": ppo.errors == driver.errors == creeper.errors == 0,
    }
    return {
        "paired_ppo_driver": {"mean": mean, "lo": lo, "hi": hi},
        "checks": checks,
        "pass": all(checks.values()),
    }


def band_and_anchors(cmp: Comparison) -> dict[str, bool]:
    """§8.9 item 1's measured half on ``standard`` 0..399: the driver's band and the five
    anchor requirements of the §8.7 table, and no errors among them."""
    rs: Mapping[str, Rates] = {i: rates(entry(cmp, i)) for i in (DRIVER_ID, *ANCHOR_IDS)}
    d = rs[DRIVER_ID]
    lo, hi = BAND_FELL
    return {
        "band_fell": lo <= d.fell <= hi,
        "band_docked": d.docked >= BAND_MIN_DOCKED,
        "balance_in_place": rs["balance_in_place"].score <= FLAT_ANCHOR_MAX,
        "spin": rs["spin"].score <= FLAT_ANCHOR_MAX,
        "creeper": rs["creeper"].score <= d.score - CREEPER_GAP,
        "gates_only": rs["gates_only"].score <= d.score - GATES_ONLY_GAP,
        "reckless": rs["reckless"].score < d.score,
        "no_errors": all(r.errors == 0 for r in rs.values()),
    }


def exchange_rate(e: Entry) -> float:
    """§8.6: the mean score of docked survivors minus the mean score of fallers (``nan`` when
    either group is empty)."""
    docked = e.extras["docked"]
    fell = e.extras["fell"]
    kept = [s for s, d, f in zip(e.scores, docked, fell, strict=True) if d is True and f is False]
    lost = [s for s, f in zip(e.scores, fell, strict=True) if f is True]
    if not kept or not lost:
        return math.nan
    return math.fsum(kept) / len(kept) - math.fsum(lost) / len(lost)


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for one value)."""
    m = math.fsum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    return m, math.sqrt(math.fsum((v - m) ** 2 for v in values) / (len(values) - 1))


def parity_failures(fixture: Path, submission: Path) -> list[str]:
    """Item 9's parity half: the fixture is bound to the policy and holds (§8.9)."""
    challenge = load_challenge("waypoint-v1")
    policy = load_policy_dir(submission, challenge=challenge)
    task = policy_task(challenge)
    fx = read_parity_fixture(fixture.read_bytes(), obs_dim=task.obs_dim, action_dim=task.action_dim)
    failed = []
    if task.obs_dim != 15:
        failed.append("obs_dim")
    if fx.onnx_sha256 != policy.policy_sha256:
        failed.append("fixture_sha256")
    if batch1_max_abs_diff(policy.validated, fx.obs, fx.action_ref) > PARITY_TOLERANCE:
        failed.append("parity")
    unsat, sat = coverage_counts(fx.action_ref)
    if unsat < MIN_UNSATURATED or sat < MIN_SATURATED:
        failed.append("coverage")
    return failed


def selection_protocol() -> EvalProtocol:
    """The 2b selection protocol, built as ``topple.train.selection.selection_protocol`` builds
    it, so its hash is every 2b run's ``selection_protocol_hash`` (item 11)."""
    return replace(EvalProtocol.from_id("waypoint-v1/standard"), seeds=SELECTION_SEEDS)


# --- phase2b.json ---------------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


Rate = Annotated[float, Field(ge=0.0, le=1.0)]
Arm = Literal["scratch", "finetune"]


class RatesModel(_Strict):
    n: Annotated[int, Field(ge=1)]
    errors: Annotated[int, Field(ge=0)]
    score: Annotated[float, Field(ge=0.0, le=100.0)]
    docked: Rate
    fell: Rate
    falls_by_cause: dict[Literal["push", "self"], Annotated[int, Field(ge=1)]]
    median_energy_j: Annotated[float, Field(ge=0.0)]
    median_t_finish_s: Annotated[float, Field(ge=0.0)]
    parts: dict[str, float]


class Paired(_Strict):
    mean: float
    lo: float
    hi: float


class Item6(_Strict):
    paired_ppo_driver: Paired
    checks: dict[str, bool]
    passed: bool = Field(alias="pass")


class RunRow(_Strict):
    arm: Arm
    summary: RunSummary


class FreshRun(_Strict):
    run: str
    arm: Arm
    first_eligible_step: int | None
    standard: RatesModel
    clean: RatesModel
    wp_eval_v1: RatesModel
    item6: Item6


class Attempt(_Strict):
    run: str
    failed: tuple[str, ...]


class Shipped(_Strict):
    run: str | None
    arm: Arm | None
    step: int | None
    onnx_sha256: str | None
    attempts: tuple[Attempt, ...]


class Sanity(_Strict):
    run: str
    source: dict[str, str | int]
    survival: Rate


class Phase2bReport(_Strict):
    """``phase2b.json`` (``topple.phase2b/1``)."""

    format: Literal["topple.phase2b/1"]
    e_ref_j: Annotated[float, Field(gt=0.0)]
    runs: Annotated[tuple[RunRow, ...], Field(min_length=6, max_length=6)]
    fresh: Annotated[tuple[FreshRun, ...], Field(min_length=6, max_length=6)]
    baselines: dict[str, dict[str, RatesModel]]
    """Per protocol, the rates of each non-run entry (the driver and the anchors)."""
    item6_passes: Annotated[int, Field(ge=0, le=6)]
    headline: dict[Arm, dict[str, Any]]
    arms: dict[str, Any]
    shipped: Shipped
    sanity: Annotated[tuple[Sanity, ...], Field(max_length=3)]
    exchange_rate: float | None
    pilot: dict[str, Any]


def parse_report(data: bytes) -> Phase2bReport:
    """``data`` as ``phase2b.json``; ``ValueError`` (pydantic's included) else."""
    if len(data) > MAX_RESULTS_BYTES:
        raise ValueError(f"phase2b.json has more than {MAX_RESULTS_BYTES} bytes")
    return Phase2bReport.model_validate_json(data, strict=True)
