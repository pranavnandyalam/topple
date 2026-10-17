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
from dataclasses import dataclass
from statistics import median
from typing import Any, Final

from topple.compare import SCORELESS, Comparison, Entry
from topple.stats import paired_bootstrap

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
    n = len(e.scores)
    if n == 0:
        raise Phase2bError(f"{e.id}: no episodes")
    docked = e.extras["docked"]
    fell = e.extras["fell"]
    causes = Counter(str(c) for c, f in zip(e.extras["fall_cause"], fell, strict=True) if f is True)
    return Rates(
        n=n,
        errors=sum(t in SCORELESS - {"sim_unstable"} for t in e.termination),
        score=math.fsum(e.scores) / n,
        docked=sum(d is True for d in docked) / n,
        fell=sum(f is True for f in fell) / n,
        falls_by_cause=dict(sorted(causes.items())),
        median_energy_j=median(e.energy_j),
        median_t_finish_s=median(
            float(t) if isinstance(t, (int, float)) else math.inf for t in e.extras["t_finish_s"]
        ),
    )


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
