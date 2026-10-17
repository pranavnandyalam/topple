"""``scripts/phase2_results.py``'s pure parts (docs/design/phase-2.md §5.1, §2.5, §4).

The end-to-end build runs on the real training runs (slice 3b); here: summaries come from
``train_meta.json`` and validate, retry attempts are read, the §2.5 choice and its attempts,
the ``--fetch-ci`` evidence binding, and ``phase2.json`` is re-read with the gates' parser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from phase2_common import InitSource, RunCore, ShipAttempt
from phase2_results import (
    ResultsError,
    attempts,
    evidence_unchanged,
    run_summary,
    ship,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "train"
SHA = "a" * 64


def _row(step: int, eligible: bool, score: float, success: float) -> dict[str, Any]:
    return {
        "step": step,
        "success_rate": success,
        "score": score,
        "median_energy_j": 2.0,
        "mean_saturation": 0.02,
        "extras": {"fell": 1 - success},
        "eligible": eligible,
        "std_mean": 0.2,
        "frac_mean_outside_1": 0.01,
        "excess_penalty_mean": 0.001,
        "cpu_s": 10.0,
        "wall_s": 5.0,
        "onnx_sha256": f"{step % 10}" * 64,
    }


def _core(name: str, score: float, *, eligible: bool = True, algo: str = "ppo") -> RunCore:
    row = _row(1024, eligible, score, 0.9)
    return RunCore.model_validate_json(
        json.dumps(
            {
                "run": name,
                "algo": algo,
                "selection_history": [row],
                "best": {
                    "step": 1024,
                    "eligible": eligible,
                    "rule": "pid",
                    "onnx_sha256": row["onnx_sha256"],
                },
                "flags": [],
            }
        ),
        strict=True,
    )


# --- §2.5 --------------------------------------------------------------------------------------


def test_ship_takes_the_best_ranked_candidate_that_passes() -> None:
    cores = [_core("ppo_s0", 86.0), _core("ppo_s1", 80.0), _core("ppo_s2", 87.0)]
    failing = {"ppo_s2": ("G2b",)}
    run, tried = ship(cores, lambda r: failing.get(r, ()))
    assert run == "ppo_s0"
    assert tried == (
        ShipAttempt(run="ppo_s2", failed_gates=("G2b",), failed_on="local"),
        ShipAttempt(run="ppo_s0", failed_gates=(), failed_on=None),
    )


def test_ship_skips_ineligible_runs_and_fails_when_none_passes() -> None:
    cores = [_core("ppo_s0", 90.0, eligible=False), _core("ppo_s1", 80.0)]
    with pytest.raises(ResultsError, match="2a fails"):
        ship(cores, lambda _r: ("G5",))


def test_ship_ignores_failures_of_non_ship_gates() -> None:
    run, tried = ship([_core("ppo_s0", 80.0)], lambda _r: ("G8", "G11"))
    assert run == "ppo_s0"
    assert tried[0].failed_gates == ()


# --- --fetch-ci --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ancestor", "diff", "ok"),
    [(0, 0, True), (1, 0, False), (0, 1, False)],
    ids=["unchanged", "not-ancestor", "evidence-changed"],
)
def test_evidence_binds_to_an_ancestor_with_unchanged_evidence_paths(
    ancestor: int, diff: int, ok: bool
) -> None:
    calls: list[list[str]] = []

    def git(cmd: Any) -> int:
        calls.append(list(cmd))
        return ancestor if cmd[0] == "merge-base" else diff

    assert evidence_unchanged("b" * 40, git) is ok
    assert calls[0][:3] == ["merge-base", "--is-ancestor", "b" * 40]
    if ancestor == 0:
        assert calls[1][:4] == ["diff", "--quiet", "b" * 40, "HEAD"]
        assert "python" in calls[1]
        assert "uv.lock" in calls[1]


# --- run summaries -----------------------------------------------------------------------------


def test_attempts_are_read_oldest_first(tmp_path: Path) -> None:
    run = tmp_path / "ppo_s0"
    run.mkdir()
    for k, wall in ((1, 2400.5), (2, 2401.0)):
        d = tmp_path / f"ppo_s0.attempt{k}"
        d.mkdir()
        meta = {
            "flags": ["wall_clock_abort"],
            "wall_clock_s": wall,
            "load_avg_start": [1.0, 2.0, 3.0],
            "cpu_s": 900.0,
        }
        (d / "train_meta.json").write_text(json.dumps(meta))
    got = attempts(run)
    assert [a["dir"] for a in got] == ["ppo_s0.attempt1", "ppo_s0.attempt2"]
    assert got[0]["load_avg_start"] == [1.0, 2.0, 3.0]
    assert attempts(tmp_path / "other") == []


@pytest.mark.train
def test_a_trained_run_gives_a_valid_summary(tmp_path: Path) -> None:
    from topple.train.config import read_train_config
    from topple.train.learn import train

    loaded = read_train_config(CONFIGS / "smoke_ppo.yaml")
    run = train(loaded.config, tmp_path / "ppo_s0", config_file_sha256=loaded.file_sha256)
    summary = run_summary(run)
    assert summary["run"] == "ppo_s0"
    assert summary["attempts"] == []
    assert summary["best"]["rule"] == "none"
    assert len(summary["load_avg_start"]) == 3


def test_a_summary_missing_a_field_is_refused(tmp_path: Path) -> None:
    run = tmp_path / "ppo_s0"
    run.mkdir()
    cfg = {"algo": "ppo", "seed": 0, "total_timesteps": 2048, "selection": {"gate": "pid"}}
    best = {"step": 1024, "eligible": True, "onnx_sha256": SHA}
    (run / "train_meta.json").write_text(json.dumps({"config": cfg, "best": best}))
    with pytest.raises(ResultsError, match="lacks"):
        run_summary(run)


# --- review fixes: the ubuntu fallback, restoring, stale evidence ------------------------------


def test_an_ubuntu_failure_ships_the_next_candidate() -> None:
    cores = [_core("ppo_s0", 86.0), _core("ppo_s2", 87.0)]
    run, tried = ship(cores, lambda _r: (), {"ppo_s2": ("G2b",)})
    assert run == "ppo_s0"
    assert tried == (
        ShipAttempt(run="ppo_s2", failed_gates=("G2b",), failed_on="ubuntu"),
        ShipAttempt(run="ppo_s0", failed_gates=(), failed_on=None),
    )


def test_a_local_failure_wins_over_a_listed_ubuntu_one() -> None:
    cores = [_core("ppo_s2", 87.0), _core("ppo_s0", 86.0)]
    run, tried = ship(cores, lambda r: ("G5",) if r == "ppo_s2" else (), {"ppo_s2": ("G2",)})
    assert run == "ppo_s0"
    assert tried[0] == ShipAttempt(run="ppo_s2", failed_gates=("G5",), failed_on="local")


@pytest.mark.parametrize("text", ["ppo_s2", "ppo_s2=", "=G2", "ppo_s2=G5", "ppo_s2=G2,G9"])
def test_ubuntu_failed_takes_only_g2_to_g4(text: str) -> None:
    import argparse

    from phase2_results import _ubuntu_failure

    with pytest.raises(argparse.ArgumentTypeError):
        _ubuntu_failure(text)
    assert _ubuntu_failure("ppo_s2=G2b,G4") == ("ppo_s2", ("G2b", "G4"))


def test_a_failed_ship_restores_the_previous_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phase2_results

    monkeypatch.setattr(phase2_results, "REPO", tmp_path)
    sub = tmp_path / "submissions" / "ppo_balance"
    sub.mkdir(parents=True)
    (sub / "policy.onnx").write_bytes(b"old")
    fixture = tmp_path / "tests" / "data" / "ppo_balance_parity.npz"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"old fixture")

    def failing_ship() -> None:
        with phase2_results.restored_on_failure(tmp_path / "bak"):
            (sub / "policy.onnx").write_bytes(b"candidate")
            fixture.write_bytes(b"candidate fixture")
            (tmp_path / "tests" / "data" / "ppo_balance_parity.bin").write_bytes(b"new")
            raise ResultsError("no candidate")

    with pytest.raises(ResultsError):
        failing_ship()
    assert (sub / "policy.onnx").read_bytes() == b"old"
    assert fixture.read_bytes() == b"old fixture"
    assert not (tmp_path / "tests" / "data" / "ppo_balance_parity.bin").exists()


def test_stale_ci_evidence_is_not_rendered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import phase2_results

    class Stub:
        class controller:  # noqa: N801
            policy_sha256 = "c" * 64

    ci = tmp_path / "ci_ubuntu"
    ci.mkdir()
    (ci / "source.json").write_text("{}")
    monkeypatch.setattr(phase2_results, "read_results", lambda _p: Stub())
    assert phase2_results.ci_render(tmp_path, "d" * 64) == {}


@pytest.mark.parametrize(
    ("run", "ok"),
    [
        ("runs/ppo_s0", True),
        ("ppo_s0", True),
        ("../ppo_s0", False),
        ("/abs/run", False),
        ("runs/.x", False),
        ("runs//x", False),
    ],
)
def test_init_source_names_the_run_as_the_config_does(run: str, ok: bool) -> None:
    """§8.8: ``train_meta.init_source.run`` is the config's ``init.from_runs`` entry."""
    doc = {"run": run, "step": 5013504, "onnx_sha256": "a" * 64}
    if ok:
        assert InitSource.model_validate(doc).run == run
    else:
        with pytest.raises(ValueError, match="should match pattern"):
            InitSource.model_validate(doc)
