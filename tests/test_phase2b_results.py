"""``scripts/phase2b_results.py``'s statistics (docs/design/phase-2.md §8.8): steps to the first
eligible eval, the headline medians with and without pretraining, and the per-arm summary. The
shipping replay is ``phase2b_gate.item5_rows``' (``test_phase2b_gate.py``)."""

from __future__ import annotations

from typing import Any

import pytest

import phase2b_common as wp
import phase2b_results as results
from test_phase2b_common import comparison, driver, ent, flat


def summary(run: str, eligible_at: int | None, init: bool = False) -> dict[str, Any]:
    steps = (1_000_000, 2_000_000, 3_000_000)
    rows = [{"step": s, "eligible": eligible_at is not None and s >= eligible_at} for s in steps]
    s: dict[str, Any] = {"run": run, "selection_history": rows}
    if init:
        s["init_source"] = {"run": "runs/ppo_s0", "step": 1, "onnx_sha256": "0" * 64}
    return s


def test_first_eligible_is_the_first_eligible_eval() -> None:
    assert results.first_eligible(summary("a", 2_000_000)) == 2_000_000
    assert results.first_eligible(summary("a", None)) is None


def test_headline_medians_with_and_without_pretraining(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "pretraining_steps", lambda s: 5_000_000)
    runs = [
        {"arm": "scratch", "summary": summary("s0", 1_000_000)},
        {"arm": "scratch", "summary": summary("s1", 3_000_000)},
        {"arm": "scratch", "summary": summary("s2", 2_000_000)},
        {"arm": "finetune", "summary": summary("f0", 1_000_000, True)},
        {"arm": "finetune", "summary": summary("f1", 1_000_000, True)},
        {"arm": "finetune", "summary": summary("f2", None, True)},
    ]
    h = results.headline(runs)
    assert h["scratch"] == {"first_eligible_steps": [1_000_000, 3_000_000, 2_000_000],
                            "median": 2_000_000}  # fmt: skip
    assert h["finetune"]["median"] is None  # a run that never got there leaves no median
    assert h["finetune"]["median_with_pretraining"] is None
    assert h["finetune"]["pretraining_steps"] == [5_000_000] * 3
    runs[-1] = {"arm": "finetune", "summary": summary("f2", 3_000_000, True)}
    h = results.headline(runs)
    assert h["finetune"]["median"] == 1_000_000
    assert h["finetune"]["median_with_pretraining"] == 6_000_000


def test_arms_summarise_each_arm_against_the_driver() -> None:
    names = {"s0": 80.0, "s1": 82.0, "s2": 84.0, "f0": 70.0, "f1": 71.0, "f2": 72.0}
    entries = [driver(), ent(wp.CREEPER_ID, flat(60.0), docked=0, fell=0)]
    entries += [ent(n, flat(v), docked=40, fell=5) for n, v in names.items()]
    runs = [{"arm": "scratch" if n[0] == "s" else "finetune", "summary": {"run": n}} for n in names]
    out = results.arms(runs, {"standard": comparison(entries)})
    assert out["scratch"]["score_mean"] == pytest.approx(82.01, abs=0.01)
    assert out["scratch"]["score_sd"] == pytest.approx(2.0, abs=1e-6)
    assert out["finetune"]["fell_mean"] == pytest.approx(0.1)
    assert out["scratch"]["energy_vs_driver"] == pytest.approx(1.0)
    assert out["better_arm"] == "scratch"
