"""``scripts/phase2b_common.py``: item 6, the item-1 band and anchors, rates and the exchange
rate, computed from a waypoint ``topple.compare/1`` file (docs/design/phase-2.md §8.7, §8.9)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import pytest

import phase2b_common as wp
from topple.compare import (
    WAYPOINT_EXTRAS,
    Comparison,
    Entry,
    ExtraValue,
    PairedRow,
    ordered_pairs,
)
from topple.evaluate import EvalProtocol, platform_tag, run_versions
from topple.stats import N_RESAMPLES, SEED, paired_bootstrap

N = 50
SHA = "0" * 64


def ent(name: str, scores: Sequence[float], *, docked: int, fell: int,
        push: int = 0, errors: int = 0) -> Entry:  # fmt: skip
    """``docked`` docked survivors first, then ``fell`` fallers (the first ``push`` of them
    push-caused), then ``errors`` errored episodes, then survivors that did not dock."""
    n = len(scores)
    kinds = ["dock"] * docked + ["fall"] * fell + ["error"] * errors
    kinds += ["slow"] * (n - len(kinds))
    term = ["fell" if k == "fall" else "error" if k == "error" else "timeout" for k in kinds]
    extras: dict[str, list[ExtraValue]] = {m: [] for m in WAYPOINT_EXTRAS}
    for i, k in enumerate(kinds):
        row: dict[str, ExtraValue] = {
            "waypoints": 5 if k == "dock" else 2,
            "progress": 0.5,
            "docked": k == "dock",
            "fell": k == "fall",
            "t_finish_s": 9.0 + i * 0.01 if k == "dock" else 20.0,
            "max_hold_steps": 100 if k == "dock" else 10,
            "fall_cause": ("push" if i - docked < push else "self") if k == "fall" else "none",
            "course_pts": 1.0,
            "speed_pts": 1.0,
            "alive_pts": 1.0,
            "battery_pts": 1.0,
        }
        for m in WAYPOINT_EXTRAS:
            extras[m].append(None if k == "error" else row[m])
    return Entry(
        id=name,
        policy_sha256=None,
        sha256=SHA,
        scores=tuple(0.0 if k == "error" else s for k, s in zip(kinds, scores, strict=True)),
        success=tuple(k == "dock" for k in kinds),
        energy_j=tuple(10.0 + 0.1 * i for i in range(n)),
        saturation_fraction=(0.01,) * n,
        termination=tuple(term),  # type: ignore[arg-type]
        extras={k: tuple(v) for k, v in extras.items()},
    )


def comparison(entries: Sequence[Entry]) -> Comparison:
    proto = EvalProtocol.from_id("waypoint-v1/standard")
    seeds = tuple(range(len(entries[0].scores)))
    proto = replace(proto, seeds=seeds)
    rows = []
    for i, j in ordered_pairs(len(entries)):
        mean, lo, hi = paired_bootstrap(entries[i].scores, entries[j].scores)
        rows.append(PairedRow(a=entries[i].id, b=entries[j].id, mean=mean, lo=lo, hi=hi,
                              n_resamples=N_RESAMPLES, seed=SEED))  # fmt: skip
    return Comparison(
        protocol=proto.id,
        protocol_hash=proto.protocol_hash(),
        complete=True,
        seeds=seeds,
        platform=platform_tag(),
        versions=run_versions(),
        entries=tuple(entries),
        paired=tuple(rows),
    )


def flat(value: float) -> list[float]:
    return [value + 0.01 * (k % 3) for k in range(N)]


def driver(**kw: int) -> Entry:
    return ent(wp.DRIVER_ID, flat(80.0), docked=kw.get("docked", 45), fell=kw.get("fell", 2))


def creeper() -> Entry:
    return ent(wp.CREEPER_ID, flat(60.0), docked=0, fell=0)


# --- rates ----------------------------------------------------------------------------------------


def test_rates_count_docked_falls_by_cause_and_errors() -> None:
    r = wp.rates(ent("p", flat(70.0), docked=24, fell=4, push=3, errors=2))
    assert (r.n, r.errors) == (N, 2)
    assert r.docked == pytest.approx(0.48)
    assert r.fell == pytest.approx(0.08)
    assert r.falls_by_cause == {"push": 3, "self": 1}
    assert r.median_t_finish_s == 20.0  # 24 of 50 finish: the median is a non-finisher's 20 s


def test_entry_refuses_a_missing_id_a_duplicate_or_absent_extras() -> None:
    cmp = comparison([driver(), creeper()])
    with pytest.raises(wp.Phase2bError, match="no single entry 'ppo'"):
        wp.entry(cmp, "ppo")
    bare = driver().model_copy(update={"extras": {}})
    with pytest.raises(wp.Phase2bError, match="records no 'docked'"):
        wp.entry(comparison([bare]), wp.DRIVER_ID)
    with pytest.raises(wp.Phase2bError, match="incomplete"):
        wp.entry(cmp.model_copy(update={"complete": False}), wp.DRIVER_ID)


# --- item 6 ---------------------------------------------------------------------------------------


def test_item6_passes_a_policy_that_matches_the_driver() -> None:
    ppo = ent("ppo", flat(79.0), docked=40, fell=10)
    out = wp.item6(comparison([driver(), creeper(), ppo]), "ppo")
    assert out["pass"] is True
    assert set(out["checks"]) == {
        "a_docked", "b_paired_lo", "c_above_creeper", "d_falls", "no_errors"
    }  # fmt: skip


@pytest.mark.parametrize(
    ("ppo", "creeper_score", "failed"),
    [
        (ent("ppo", flat(79.0), docked=27, fell=1), 60.0, "a_docked"),  # 54% < 60%
        (ent("ppo", flat(70.0), docked=40, fell=1), 60.0, "b_paired_lo"),  # 10 points below
        (ent("ppo", flat(79.0), docked=40, fell=1), 79.5, "c_above_creeper"),
        (ent("ppo", flat(79.0), docked=35, fell=13), 60.0, "d_falls"),  # 26% > 4% + 20%
        (ent("ppo", flat(79.0), docked=40, fell=1), -1.0, "no_errors"),  # the creeper errs
    ],
)
def test_item6_fails_each_sub_gate_alone(ppo: Entry, creeper_score: float, failed: str) -> None:
    slow = ent(
        wp.CREEPER_ID, flat(abs(creeper_score)), docked=0, fell=0, errors=int(creeper_score < 0)
    )
    out = wp.item6(comparison([driver(), slow, ppo]), "ppo")
    checks = out["checks"]
    assert isinstance(checks, dict)
    assert [k for k, v in checks.items() if not v] == [failed]
    assert out["pass"] is False


def test_the_fall_margin_is_inclusive() -> None:
    """fell == driver_fell + 0.20 passes (the float sum is absorbed)."""
    ppo = ent("ppo", flat(79.0), docked=38, fell=12)  # 24% = 4% + 20%
    out = wp.item6(comparison([driver(), creeper(), ppo]), "ppo")
    assert out["checks"]["d_falls"] is True


# --- item 1 ---------------------------------------------------------------------------------------


def anchors(**scores: float) -> list[Entry]:
    base = {"balance_in_place": 20.0, "spin": 17.0, "creeper": 60.0, "gates_only": 69.0,
            "reckless": 71.0} | scores  # fmt: skip
    return [ent(k, flat(v), docked=0, fell=0) for k, v in base.items()]


def test_band_and_anchors_hold_for_the_calibrated_numbers() -> None:
    out = wp.band_and_anchors(comparison([driver(), *anchors()]))
    assert all(out.values()), out


@pytest.mark.parametrize(
    ("drv", "scores", "failed"),
    [
        ({"fell": 1}, {}, "band_fell"),  # 2% < 3%
        ({"docked": 44, "fell": 6}, {}, "band_fell"),  # 12% > 10%
        ({"docked": 43}, {}, "band_docked"),  # 86% < 88%
        ({}, {"balance_in_place": 25.5}, "balance_in_place"),
        ({}, {"spin": 26.0}, "spin"),
        ({}, {"creeper": 70.5}, "creeper"),
        ({}, {"gates_only": 75.5}, "gates_only"),
        ({}, {"reckless": 80.5}, "reckless"),
    ],
)
def test_band_and_anchors_fail_each_requirement_alone(
    drv: dict[str, int], scores: dict[str, float], failed: str
) -> None:
    out = wp.band_and_anchors(comparison([driver(**drv), *anchors(**scores)]))
    assert [k for k, v in out.items() if not v] == [failed]


# --- reporting helpers ----------------------------------------------------------------------------


def test_exchange_rate_is_docked_survivors_minus_fallers() -> None:
    scores = [90.0] * 10 + [30.0] * 5 + [70.0] * 5
    e = ent("ppo", scores, docked=10, fell=5)
    assert wp.exchange_rate(e) == pytest.approx(60.0)
    assert math.isnan(wp.exchange_rate(ent("ppo", flat(80.0), docked=20, fell=0)))


def test_mean_sd() -> None:
    assert wp.mean_sd([1.0, 2.0, 3.0]) == pytest.approx((2.0, 1.0))
    assert wp.mean_sd([5.0]) == (5.0, 0.0)
