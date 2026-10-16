"""Checkpoint selection (docs/design/phase-2.md §2.4, §2.5, §6 ``test_selection``).

The eligibility truth table, rank keys, the best-checkpoint fallback, on-policy parity rows from
real episodes (5 per seed, a 3-step episode, none for a T = 0 episode), the edge and random
parity sets, the run flags on synthetic histories and the eval schedule (labels unique,
increasing, the last equal to SB3's final ``num_timesteps``) for the smoke and shipped configs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from policy_support import ppo_bytes, tiny_norm
from pool_support import InlinePool
from topple.evaluate import EvalProtocol, iter_episodes
from topple.loader import LoadError, load_submission
from topple.policy.build import ObsNorm
from topple.pool import (
    ONPOLICY_ROWS,
    EpisodeRows,
    PolicyRef,
    eval_pool,
    onpolicy_rows,
    policy_ref,
    run_episodes_with_rows,
    submit_episodes_with_rows,
)
from topple.results import EpisodeResult, ErrorInfo, summarize
from topple.train.config import (
    HealthConfig,
    SelectionConfig,
    load_train_config,
    validate_train_config,
)
from topple.train.selection import (
    ABORT_FLAGS,
    FLAGS,
    N_RANDOM_ROWS,
    NOT_SHIPPABLE,
    SET_EDGE,
    SET_ONPOLICY,
    SET_RANDOM,
    EvalTrigger,
    GateBaseline,
    HistoryRow,
    SelectionResult,
    best_json,
    choose_best,
    edge_row_count,
    edge_rows,
    eval_labels,
    evaluate_policy_bytes,
    final_timesteps,
    is_eligible,
    late_regression,
    measure_gate_baseline,
    onpolicy_seed_count,
    parity_obs,
    random_rows,
    rank_key,
    run_flags,
    selection_protocol,
    std_collapse,
    summarize_episodes,
    write_best_json,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "train"
PID = REPO / "submissions" / "pid_example.py"
E_REF = 13.1
SHA = "ab" * 32
# the v5.3 known answer for balance-v1/standard on seeds 2000..2199 (§2.3)
SELECTION_HASH = "f30ef19048989c9b6edf34e5711b5c116ecd386568a5a1e6673360fc4fcf6287"


def result(
    step: int = 0,
    success: float = 0.7,
    score: float = 70.0,
    energy: float = 6.0,
    saturation: float = 0.05,
    **extras: float,
) -> SelectionResult:
    return SelectionResult(
        step=step,
        success_rate=success,
        score=score,
        median_energy_j=energy,
        mean_saturation=saturation,
        scores=(score,),
        extras=extras,
    )


def selection(gate: str = "pid", max_saturation: float | None = 0.08) -> SelectionConfig:
    return SelectionConfig(
        protocol="balance-v1/standard",
        seeds={"first": 2000, "last": 2199},  # type: ignore[arg-type]
        every_steps=250_000,
        workers=4,
        gate=gate,  # type: ignore[arg-type]
        max_saturation=max_saturation,
    )


PID_BASE = GateBaseline("pid", reference=result(success=0.65, score=70.9))


# --- results ------------------------------------------------------------------------------------


def test_a_result_is_finite_and_its_json_omits_the_scores() -> None:
    r = result(step=262_144, fell=0.3)
    assert r.to_json() == {
        "step": 262_144,
        "success_rate": 0.7,
        "score": 70.0,
        "median_energy_j": 6.0,
        "mean_saturation": 0.05,
        "extras": {"fell": 0.3},
    }
    with pytest.raises(ValueError, match="finite"):
        result(score=float("nan"))
    with pytest.raises(ValueError, match="step"):
        result(step=-1)


def episode(
    index: int, *, score: float, success: bool, termination: str, energy: float, saturation: float
) -> EpisodeResult:
    return EpisodeResult(
        index=index,
        seed=2000 + index,
        success=success,
        score=score,
        metrics={
            "termination": termination,
            "energy_j": energy,
            "saturation_fraction": saturation,
            "time_upright_s": 10.0,
        },
    )


def test_summarize_follows_the_aggregate_rules() -> None:
    episodes = [
        episode(0, score=90.0, success=True, termination="timeout", energy=3.0, saturation=0.1),
        episode(1, score=40.0, success=False, termination="fell", energy=9.0, saturation=0.0),
        episode(
            2, score=60.0, success=False, termination="floor_contact", energy=5.0, saturation=0.2
        ),
        EpisodeResult(
            index=3,
            seed=2003,
            success=False,
            score=0.0,
            metrics=None,
            error=ErrorInfo(code="missing", step=0, detail=""),
        ),
    ]
    r = summarize_episodes(episodes, step=7)
    assert r.step == 7
    assert r.success_rate == 0.25
    assert r.score == 47.5  # fsum / N: the missing episode counts 0
    assert r.median_energy_j == 4.0  # median of (3, 9, 5, 0)
    assert r.mean_saturation == pytest.approx(0.075)
    assert r.scores == (90.0, 40.0, 60.0, 0.0)
    assert r.extras == {"fell": 0.5}
    # the leaderboard's own aggregate (results.summarize), so rank keys match it exactly
    aggregate = summarize(episodes, len(episodes))
    assert (r.score, r.success_rate) == (aggregate.score, aggregate.success_rate)
    with pytest.raises(ValueError, match="at least one episode"):
        summarize_episodes([], step=0)


# --- the gate -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("r", "eligible"),
    [
        (result(success=0.70, energy=6.0, saturation=0.05), True),
        (result(success=0.65, energy=E_REF, saturation=0.08), True),  # every bound inclusive
        (result(success=0.64, energy=6.0, saturation=0.05), False),  # below PID's success
        (result(success=0.90, energy=13.2, saturation=0.05), False),  # above E_ref
        (result(success=0.90, energy=6.0, saturation=0.081), False),  # saturated
        (result(success=1.00, score=99.0, energy=20.0, saturation=0.5), False),
    ],
)
def test_the_pid_gate(r: SelectionResult, eligible: bool) -> None:
    assert is_eligible(r, PID_BASE, selection(), E_REF) is eligible


def test_the_none_gate_takes_every_checkpoint() -> None:
    bad = result(success=0.0, energy=99.0, saturation=1.0)
    assert is_eligible(bad, GateBaseline("none"), selection("none", None), E_REF)


def test_the_waypoint_gate_is_phase_2b() -> None:
    base = GateBaseline("waypoint", reference=result(), anchor=result(score=10.0))
    with pytest.raises(NotImplementedError):
        is_eligible(result(), base, selection("waypoint", None), E_REF)


def test_the_baseline_must_match_the_configs_gate() -> None:
    with pytest.raises(ValueError, match="gate"):
        is_eligible(result(), GateBaseline("none"), selection("pid"), E_REF)
    with pytest.raises(ValueError, match="reference"):
        GateBaseline("pid")
    with pytest.raises(ValueError, match="unknown"):
        GateBaseline("lqr")  # type: ignore[arg-type]


def test_a_baseline_records_its_per_seed_scores_and_fall_rate() -> None:
    base = GateBaseline("pid", reference=replace(result(fell=0.35), scores=(70.0, 71.0)))
    doc = base.to_json()
    assert doc["rule"] == "pid"
    assert doc["anchor"] is None
    assert doc["reference"] == {**result(fell=0.35).to_json(), "scores": [70.0, 71.0]}
    assert json.loads(json.dumps(doc)) == doc


def test_measure_the_pid_baseline_on_the_selection_seeds() -> None:
    protocol = replace(EvalProtocol.from_id("balance-v1/standard"), seeds=(2000, 2001))
    base = measure_gate_baseline("pid", protocol, InlinePool(2), pid_submission=PID)
    assert base.rule == "pid"
    assert base.reference is not None
    expected = [run.result for run in iter_episodes(protocol, load_submission(PID).factory)]
    assert base.reference == summarize_episodes(expected, step=0)
    assert measure_gate_baseline("none", protocol, InlinePool(2), pid_submission=PID) == (
        GateBaseline("none")
    )
    with pytest.raises(NotImplementedError):
        measure_gate_baseline("waypoint", protocol, InlinePool(2), pid_submission=PID)


# --- rank and best ------------------------------------------------------------------------------


def test_rank_key_per_kind() -> None:
    assert rank_key(result(score=85.004, success=0.9), "balance") == (85.0, 0.9)
    # score first (rounded to 2 places), then success
    assert rank_key(result(score=85.004, success=0.9), "balance") > rank_key(
        result(score=85.001, success=0.8), "balance"
    )
    assert rank_key(result(score=85.1, success=0.5), "balance") > rank_key(
        result(score=85.0, success=1.0), "balance"
    )
    fast = result(score=80.0, success=0.9, median_t_finish_s=9.0)
    slow = result(score=80.0, success=0.9, median_t_finish_s=12.0)
    assert rank_key(fast, "waypoint") == (80.0, 0.9, -9.0)
    assert rank_key(fast, "waypoint") > rank_key(slow, "waypoint")
    with pytest.raises(ValueError, match="median_t_finish_s"):
        rank_key(result(), "waypoint")
    with pytest.raises(ValueError, match="kind"):
        rank_key(result(), "sumo")  # type: ignore[arg-type]


def row(step: int, score: float, eligible: bool, success: float = 0.7, **kw: Any) -> HistoryRow:
    return HistoryRow(
        result=result(step=step, score=score, success=success),
        eligible=eligible,
        onnx_sha256=SHA,
        **kw,
    )


def test_best_is_the_top_eligible_checkpoint() -> None:
    history = [row(1, 90.0, False), row(2, 80.0, True), row(3, 85.0, True), row(4, 70.0, True)]
    best = choose_best(history, "balance")
    assert best is not None
    assert best.step == 3
    assert best_json(best, "pid") == {"step": 3, "eligible": True, "rule": "pid"}


def test_best_falls_back_to_the_top_ineligible_checkpoint() -> None:
    history = [row(1, 60.0, False), row(2, 75.0, False), row(3, 70.0, False)]
    best = choose_best(history, "balance")
    assert best is not None
    assert (best.step, best.eligible) == (2, False)
    assert best_json(best, "pid") == {"step": 2, "eligible": False, "rule": "pid"}


def test_best_ties_go_to_the_earlier_step() -> None:
    history = [row(1, 80.0, True), row(2, 80.001, True), row(3, 79.0, True)]
    best = choose_best(history, "balance")
    assert best is not None
    assert best.step == 1


def test_best_of_nothing_and_of_a_bad_history() -> None:
    assert choose_best([], "balance") is None
    with pytest.raises(ValueError, match="unique"):
        choose_best([row(2, 80.0, True), row(2, 81.0, True)], "balance")
    with pytest.raises(ValueError, match="unique"):
        choose_best([row(3, 80.0, True), row(2, 81.0, True)], "balance")


def test_best_json_is_rewritten_atomically(tmp_path: Path) -> None:
    path = tmp_path / "best.json"
    write_best_json(path, row(5, 80.0, True), "pid")
    assert json.loads(path.read_text()) == {"step": 5, "eligible": True, "rule": "pid"}
    victim = tmp_path / "victim"
    victim.write_text("keep")
    path.unlink()
    path.symlink_to(victim)  # a planted symlink is replaced, never written through
    write_best_json(path, row(6, 81.0, False), "none")
    assert victim.read_text() == "keep"
    assert not path.is_symlink()
    assert json.loads(path.read_text()) == {"step": 6, "eligible": False, "rule": "none"}


def test_a_history_row_json() -> None:
    r = row(
        9,
        80.0,
        True,
        std_mean=0.5,
        frac_mean_outside_1=0.01,
        excess_penalty_mean=0.002,
        cpu_s=12.5,
        wall_s=5.0,
    )
    doc = r.to_json()
    assert doc["step"] == 9
    assert doc["eligible"] is True
    assert doc["onnx_sha256"] == SHA
    assert "scores" not in doc
    assert set(doc) == {
        "step",
        "success_rate",
        "score",
        "median_energy_j",
        "mean_saturation",
        "extras",
        "eligible",
        "std_mean",
        "frac_mean_outside_1",
        "excess_penalty_mean",
        "cpu_s",
        "wall_s",
        "onnx_sha256",
    }


# --- flags --------------------------------------------------------------------------------------

HEALTH = HealthConfig(
    min_std=0.03,
    min_std_before_step=2_000_000,
    regression_after_step=2_000_000,
    regression_drop=0.10,
)


def test_late_regression_on_a_synthetic_history() -> None:
    peak = [row(1_000_000, 80.0, True, success=0.85)]
    assert late_regression([*peak, row(2_500_000, 70.0, True, success=0.75)], HEALTH)
    assert not late_regression([*peak, row(2_500_000, 70.0, True, success=0.76)], HEALTH)
    # a drop before regression_after_step does not count, a later recovery does not undo one
    early = [
        *peak,
        row(1_500_000, 60.0, True, success=0.70),
        row(2_500_000, 80.0, True, success=0.80),
    ]
    assert not late_regression(early, HEALTH)
    late = [
        *peak,
        row(2_250_000, 60.0, True, success=0.70),
        row(2_500_000, 80.0, True, success=0.85),
    ]
    assert late_regression(late, HEALTH)
    assert not late_regression([], HEALTH)


def test_std_collapse_counts_only_before_its_step() -> None:
    assert std_collapse([row(1_500_000, 80.0, True, std_mean=0.02)], HEALTH)
    assert not std_collapse([row(2_000_000, 80.0, True, std_mean=0.02)], HEALTH)
    assert not std_collapse([row(1_000_000, 80.0, True, std_mean=0.03)], HEALTH)
    assert not std_collapse([row(1_000_000, 80.0, True)], HEALTH)  # SAC rows carry no std here


def test_run_flags() -> None:
    hot = HistoryRow(result=result(step=1, energy=14.0), eligible=False, onnx_sha256=SHA)
    history = [hot, row(2, 80.0, False, std_mean=0.01)]
    assert run_flags(history, health=HEALTH, e_ref_j=E_REF) == (
        "std_collapse",
        "high_energy",
        "no_eligible",
    )
    assert run_flags(history, health=None, e_ref_j=E_REF) == ("high_energy", "no_eligible")
    assert run_flags([row(1, 80.0, True)], health=HEALTH, e_ref_j=E_REF) == ()
    assert set(FLAGS) >= NOT_SHIPPABLE >= {"rss_abort", "cpu_abort", "wall_clock_abort"}


EXPECTED_FLAGS = (
    "std_collapse",
    "late_regression",
    "high_energy",
    "no_eligible",
    "rss_abort",
    "cpu_abort",
    "wall_clock_abort",
    "nondeterministic",
)


def test_flags_have_one_order_in_every_process() -> None:
    """``train_meta.flags`` is written in ``FLAGS`` order: it must not depend on the string hash
    seed (a frozenset unpacked into the tuple did)."""
    assert FLAGS == EXPECTED_FLAGS
    assert ABORT_FLAGS == ("rss_abort", "cpu_abort", "wall_clock_abort")
    code = "from topple.train.selection import FLAGS; print(','.join(FLAGS))"
    for hash_seed in ("1", "2", "3"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        assert tuple(out.strip().split(",")) == EXPECTED_FLAGS


# --- the eval schedule --------------------------------------------------------------------------


def simulate(cfg: Any) -> tuple[int, ...]:
    """SB3's loops, step by step: PPO's trigger at each rollout start, SAC's at each step."""
    trigger = EvalTrigger(
        cfg.selection.every_steps, None if cfg.algo == "ppo" else cfg.total_timesteps
    )
    labels: list[int] = []
    num = 0
    while num < cfg.total_timesteps:
        if cfg.algo == "ppo":
            if trigger.due(num):
                labels.append(num)
            num += cfg.ppo.n_steps * cfg.n_envs
        else:
            for _ in range(cfg.sac.train_freq):
                num += cfg.n_envs
                if trigger.due(num):
                    labels.append(num)
    return (*labels, num)


@pytest.mark.parametrize(
    ("name", "periodic", "first", "last_periodic", "final"),
    [
        ("smoke_ppo", 1, 1024, 1024, 2048),
        ("smoke_sac", 1, 1024, 1024, 2048),
        ("ppo_balance", 19, 262_144, 4_751_360, 5_013_504),
        ("sac_balance", 19, 50_000, 950_000, 1_000_000),
        ("ablate_nopenalty", 3, 262_144, 753_664, 1_015_808),
    ],
)
def test_eval_labels_of_the_shipped_configs(
    name: str, periodic: int, first: int, last_periodic: int, final: int
) -> None:
    cfg = load_train_config(CONFIGS / f"{name}.yaml")
    labels = eval_labels(cfg)
    assert len(labels) == periodic + 1
    assert (labels[0], labels[-2], labels[-1]) == (first, last_periodic, final)
    assert all(b > a for a, b in itertools.pairwise(labels))  # unique, increasing
    assert labels[-1] == final_timesteps(cfg) >= cfg.total_timesteps
    if name.startswith("smoke") or cfg.algo == "sac":
        assert labels[-1] == cfg.total_timesteps
    assert labels == simulate(cfg)


def test_ppo_labels_for_5m_are_the_measured_ones() -> None:
    labels = eval_labels(load_train_config(CONFIGS / "ppo_balance.yaml"))
    assert labels[:3] == (262_144, 507_904, 753_664)
    assert all(label % 16_384 == 0 for label in labels)


@pytest.mark.parametrize("algo", ["ppo", "sac"])
@pytest.mark.parametrize("n_envs", [1, 3, 8])
@pytest.mark.parametrize("unit", [7, 64, 512])
@pytest.mark.parametrize("every", [500, 1000, 4096])
def test_eval_labels_match_the_step_by_step_trigger(
    algo: str, n_envs: int, unit: int, every: int
) -> None:
    base = load_train_config(CONFIGS / f"smoke_{algo}.yaml").model_dump(mode="json")
    base.update(n_envs=n_envs, total_timesteps=16_000 + unit)
    base["selection"]["every_steps"] = every
    if algo == "ppo":
        base["ppo"].update(n_steps=unit, batch_size=2)
    else:
        base["sac"]["train_freq"] = max(1, unit // 64)
    cfg = validate_train_config(base)
    labels = eval_labels(cfg)
    assert labels == simulate(cfg)
    assert len(set(labels)) == len(labels)


def test_the_final_eval_is_skipped_where_an_abort_stopped_at_a_periodic_label() -> None:
    """SAC: a hard limit seen in the ``_on_step`` that just ran the periodic eval at 1,024 stops
    training at 1,024; ``_on_training_end`` must not evaluate (and snapshot) that label again."""
    trigger = EvalTrigger(1024, 2048)
    assert trigger.final_due(0)  # nothing evaluated yet
    assert not trigger.due(1022)
    assert trigger.final_due(1022)
    assert trigger.due(1024)
    assert trigger.last_label == 1024
    assert not trigger.final_due(1024)  # the abort path: the periodic eval was the final one
    assert trigger.final_due(1026)  # a later stop still gets its own final eval
    assert not trigger.due(2048)
    assert trigger.final_due(2048)


@pytest.mark.parametrize("name", ["ppo_balance", "sac_balance", "ablate_nopenalty"])
@pytest.mark.parametrize("k", [0, 2])
def test_a_resumed_trigger_continues_the_schedule(name: str, k: int) -> None:
    """A run resumed from the snapshot at the k-th label (SB3 keeps ``num_timesteps``) evaluates
    exactly the labels after it, never the resume point itself."""
    cfg = load_train_config(CONFIGS / f"{name}.yaml")
    labels = eval_labels(cfg)
    start = labels[k]
    sac = cfg.algo == "sac"
    trigger = EvalTrigger.resumed(
        cfg.selection.every_steps, start, cfg.total_timesteps if sac else None
    )
    assert not trigger.final_due(start)
    seen: list[int] = []
    num = start
    unit = cfg.n_envs if sac else cfg.ppo.n_steps * cfg.n_envs
    while num < cfg.total_timesteps:
        if not sac and trigger.due(num):
            seen.append(num)
        num += unit
        if sac and trigger.due(num):
            seen.append(num)
    assert trigger.final_due(num)
    assert (*seen, num) == labels[k + 1 :]
    assert EvalTrigger(cfg.selection.every_steps).due(start)  # a fresh trigger would re-fire


def test_a_resumed_trigger_is_checked() -> None:
    with pytest.raises(ValueError, match="num_timesteps"):
        EvalTrigger.resumed(1024, -1)
    with pytest.raises(ValueError, match="every_steps"):
        EvalTrigger(0)


# --- on-policy rows (a) -------------------------------------------------------------------------


def obs_of(t: int, d: int = 10) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.arange(t * d, dtype=np.float32).reshape(t, d)


@pytest.mark.parametrize(
    ("steps", "picked"),
    [
        (1, [0, 0, 0, 0, 0]),
        (3, [0, 0, 1, 2, 2]),  # np.round: half to even
        (5, [0, 1, 2, 3, 4]),
        (1000, [0, 250, 500, 749, 999]),
    ],
)
def test_onpolicy_rows_pick_five_spread_steps(steps: int, picked: list[int]) -> None:
    obs = obs_of(steps)
    rows = onpolicy_rows(obs)
    assert rows.dtype == np.float32
    assert rows.shape == (ONPOLICY_ROWS, 10)
    assert np.array_equal(rows, obs[picked])


def test_a_zero_step_episode_has_no_rows() -> None:
    assert onpolicy_rows(obs_of(0)).shape == (0, 10)
    with pytest.raises(ValueError, match="float32"):
        onpolicy_rows(obs_of(3).astype(np.float64))  # type: ignore[arg-type]


STOPPER = """
from topple.contract import BalancerObs, TaskSpec

class Stopper:
    def reset(self, rng_seed: int) -> None:
        self.n = 0

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        self.n += 1
        if self.n > {steps}:
            raise RuntimeError("stop")
        return (0.0, 0.0)

def make_controller(spec: TaskSpec | None = None) -> Stopper:
    return Stopper()
"""


def stopper(tmp_path: Path, steps: int) -> Path:
    path = tmp_path / f"stop_after_{steps}.py"
    path.write_text(STOPPER.format(steps=steps))
    return path


def protocol_of(*seeds: int, preset: str = "standard") -> EvalProtocol:
    return replace(EvalProtocol.from_id(f"balance-v1/{preset}"), seeds=seeds)


@pytest.mark.parametrize(("steps", "expected"), [(3, 3), (0, 0)])
def test_rows_come_from_the_episodes_trajectory(tmp_path: Path, steps: int, expected: int) -> None:
    sub = stopper(tmp_path, steps)
    protocol = protocol_of(2000)
    ref = policy_ref(sub, challenge=protocol.challenge)
    (got,) = run_episodes_with_rows(ref, protocol, [0])
    (run,) = list(iter_episodes(protocol, load_submission(sub).factory))
    assert run.trajectory.steps == expected
    assert got.result == run.result
    assert got.result.error is not None
    assert got.result.error.step == expected
    assert np.array_equal(got.obs_rows, onpolicy_rows(run.trajectory.obs))
    assert got.obs_rows.shape == ((ONPOLICY_ROWS, 10) if expected else (0, 10))


def onnx_ref() -> PolicyRef:
    data = ppo_bytes()
    return PolicyRef("onnx", data, hashlib.sha256(data).hexdigest(), "balance-v1")


def test_a_selection_eval_returns_five_rows_per_seed_with_steps(tmp_path: Path) -> None:
    """A T = 0 episode (an error before the first action) adds no rows and no crash."""
    seeds = tuple(range(2000, 2030))  # 30 seeds: two pool tasks of 25 and 5
    pool = InlinePool(2)
    r, rows = evaluate_policy_bytes(onnx_ref(), protocol_of(*seeds), pool, step=1024)
    assert len(pool.calls) == 2
    assert [call[3] for call in pool.calls] == [tuple(range(25)), tuple(range(25, 30))]
    assert r.step == 1024
    assert len(r.scores) == 30
    assert rows.dtype == np.float32
    assert rows.shape == (5 * 30, 10)
    assert onpolicy_seed_count(rows) == 30

    zero = policy_ref(stopper(tmp_path, 0), challenge=protocol_of(2000).challenge)
    r0, rows0 = evaluate_policy_bytes(zero, protocol_of(2000, 2001), InlinePool(1))
    assert rows0.shape == (0, 10)
    assert onpolicy_seed_count(rows0) == 0
    assert r0.score == 0.0
    assert r0.success_rate == 0.0

    three = policy_ref(stopper(tmp_path, 3), challenge=protocol_of(2000).challenge)
    _, rows3 = evaluate_policy_bytes(three, protocol_of(2000, 2001, 2002), InlinePool(1))
    assert rows3.shape == (15, 10)
    assert onpolicy_seed_count(rows3) == 3


def test_the_rows_are_in_seed_order() -> None:
    seeds = (2003, 2000, 2001)
    _, rows = evaluate_policy_bytes(onnx_ref(), protocol_of(*seeds), InlinePool(1))
    for k, seed in enumerate(seeds):
        _, alone = evaluate_policy_bytes(onnx_ref(), protocol_of(seed), InlinePool(1))
        assert np.array_equal(rows[5 * k : 5 * k + 5], alone)


def test_a_failing_task_cancels_and_raises() -> None:
    bad = PolicyRef("onnx", b"not onnx", hashlib.sha256(b"not onnx").hexdigest(), "balance-v1")
    with pytest.raises(LoadError, match=r"policy\.onnx"):
        evaluate_policy_bytes(bad, protocol_of(2000), InlinePool(1))


def test_submit_with_rows_checks_in_the_parent() -> None:
    private = replace(protocol_of(2000), seed_set="private", salt=2**64)
    with pytest.raises(ValueError, match="public"):
        submit_episodes_with_rows(InlinePool(1), onnx_ref(), private, [0])
    with pytest.raises(ValueError, match="indices"):
        submit_episodes_with_rows(InlinePool(1), onnx_ref(), protocol_of(2000), [1])


def test_a_real_pool_gives_the_same_rows() -> None:
    protocol = protocol_of(2000, 2001)
    expected = evaluate_policy_bytes(onnx_ref(), protocol, InlinePool(1))
    with eval_pool(2) as pool:
        got = evaluate_policy_bytes(onnx_ref(), protocol, pool)
    assert got[0] == expected[0]
    assert np.array_equal(got[1], expected[1])


def test_episode_rows_pickle_small() -> None:
    import pickle

    (got,) = run_episodes_with_rows(onnx_ref(), protocol_of(2000), [0])
    again = pickle.loads(pickle.dumps(got))
    assert isinstance(again, EpisodeRows)
    assert np.array_equal(again.obs_rows, got.obs_rows)
    assert len(pickle.dumps(got)) < 8 * 1024  # never a trajectory


# --- the selection protocol ---------------------------------------------------------------------


def test_the_selection_protocol_hash_is_the_known_answer() -> None:
    cfg = load_train_config(CONFIGS / "ppo_balance.yaml")
    protocol = selection_protocol(cfg.selection)
    assert protocol.seeds == tuple(range(2000, 2200))
    assert protocol.protocol_hash() == SELECTION_HASH


# --- parity rows (b) and (c) --------------------------------------------------------------------


@pytest.mark.parametrize("d", [10, 15])
def test_the_edge_set_has_6d_plus_4_rows(d: int) -> None:
    norm = tiny_norm(d)
    rows = edge_rows(norm)
    assert rows.shape == (edge_row_count(d), d) == (6 * d + 4, d)
    assert rows.dtype == np.float32
    mean = norm.mean
    std = np.sqrt(norm.var + norm.epsilon)
    for j in (0, d - 1):
        block = rows[6 * j : 6 * j + 6]
        others = np.delete(block, j, axis=1)
        assert np.array_equal(others, np.tile(np.delete(mean, j).astype(np.float32), (6, 1)))
        expected = [
            mean[j] + 10 * std[j],
            mean[j] - 10 * std[j],
            mean[j] + 10.001 * std[j],
            mean[j] - 10.001 * std[j],
            1e6,
            -1e6,
        ]
        assert np.array_equal(block[:, j], np.asarray(expected).astype(np.float32))
    tail = rows[-4:]
    assert np.array_equal(tail[0], mean.astype(np.float32))
    assert not tail[1].any()
    assert (tail[2] == 1e6).all()
    assert (tail[3] == -1e6).all()


def test_random_rows_are_fixed_and_within_12_std() -> None:
    norm = tiny_norm()
    rows = random_rows(norm)
    assert rows.shape == (N_RANDOM_ROWS, 10)
    assert rows.dtype == np.float32
    assert np.array_equal(rows, random_rows(norm))
    k = (rows.astype(np.float64) - norm.mean) / np.sqrt(norm.var + norm.epsilon)
    assert np.abs(k).max() <= 12.0 + 1e-5
    assert np.abs(k).max() > 11.0  # spread over the whole range
    u = np.random.default_rng(np.random.SeedSequence(0)).random((N_RANDOM_ROWS, 10))
    expected = norm.mean + np.sqrt(norm.var + norm.epsilon) * (-12.0 + 24.0 * u)
    assert np.array_equal(rows, expected.astype(np.float32))


def test_parity_obs_concatenates_the_three_sets_in_order() -> None:
    norm = tiny_norm()
    onpolicy = obs_of(10)
    obs, labels = parity_obs(onpolicy, norm)
    assert obs.dtype == np.float32
    assert labels.dtype == np.uint8
    assert obs.shape == (10 + N_RANDOM_ROWS + 64, 10)
    assert np.array_equal(obs[:10], onpolicy)
    assert np.array_equal(obs[10 : 10 + N_RANDOM_ROWS], random_rows(norm))
    assert np.array_equal(obs[10 + N_RANDOM_ROWS :], edge_rows(norm))
    assert labels.tolist() == [SET_ONPOLICY] * 10 + [SET_RANDOM] * N_RANDOM_ROWS + [SET_EDGE] * 64
    with pytest.raises(ValueError, match="float32"):
        parity_obs(obs_of(5, d=9), norm)
    with pytest.raises(ValueError, match="groups of 5"):
        onpolicy_seed_count(obs_of(4))


@pytest.mark.parametrize(
    "norm",
    [
        ObsNorm(mean=np.zeros(3), var=np.ones(2), epsilon=1e-8, clip=10.0),
        ObsNorm(mean=np.zeros(3), var=-np.ones(3), epsilon=1e-8, clip=10.0),
        ObsNorm(mean=np.full(3, np.nan), var=np.ones(3), epsilon=1e-8, clip=10.0),
        ObsNorm(mean=np.zeros(3), var=np.ones(3), epsilon=0.0, clip=10.0),
    ],
)
def test_parity_rows_refuse_bad_statistics(norm: ObsNorm) -> None:
    with pytest.raises(ValueError, match=r"mean|var|epsilon"):
        edge_rows(norm)
    with pytest.raises(ValueError, match=r"mean|var|epsilon"):
        random_rows(norm)
