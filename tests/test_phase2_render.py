"""The Phase 2a media and ``phase2.md`` (``scripts/phase2_render.py``, docs/design/phase-2.md §5.2).

A small synthetic ``phase2.json`` (6 eval_v1 seeds, 8 calibration seeds) stands in for the real
one. ``make_media`` runs with its expensive parts replaced: re-run episodes return one short
real trajectory with the stored score, ``side_by_side_frames`` yields two noise frames (no GL)
and the shove bench returns chosen curves. What is checked: the §5.2 seed rules (ties, the
no-PID-fall fallback, the lower median), the learning-curve values, the reward-design x/y, the
i50s passed through, the GIF rung recorded, the size caps, the bit-exact re-run check, and that
``phase2.md`` has no unfilled field and fails on a missing one. One ``render`` test draws a real
gallery GIF; one ``slow`` test checks that an in-process re-run equals ``topple compare``'s
pool episode for PID and an ONNX policy.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import phase2_common as common
import phase2_render as render
from run_support import make_run
from topple import video
from topple.bench import ShoveCurve
from topple.compare import Comparison, Entry, PairedRow, compare, ordered_pairs, to_json
from topple.envs import BalanceEnv
from topple.evaluate import platform_tag, run_episode, run_versions
from topple.export import export_policy
from topple.loader import submission_hashes, submission_sha256
from topple.stats import N_RESAMPLES, SEED, paired_bootstrap
from topple.traj import Trajectory

REPO = Path(__file__).resolve().parents[1]
PID_SHA = submission_sha256(REPO / common.PID_SUBMISSION)
LQR_SHA = submission_sha256(REPO / common.LQR_SUBMISSION)
EVAL_SEEDS = common.eval_protocol().seeds[:6]
CAL_SEEDS = tuple(range(8))
FAKE = "f" * 64
Frame = npt.NDArray[np.uint8]


def sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


# --- comparisons ----------------------------------------------------------------------------------


def ent(name: str, scores: Sequence[float], terms: Sequence[str], *, sha256: str,
        policy: str | None = None) -> Entry:  # fmt: skip
    return Entry(
        id=name,
        policy_sha256=policy,
        sha256=sha256,
        scores=tuple(scores),
        success=tuple(t == "timeout" for t in terms),
        energy_j=tuple(3.0 + 0.1 * k for k in range(len(scores))),
        saturation_fraction=(0.03,) * len(scores),
        termination=tuple(terms),  # type: ignore[arg-type]
        extras={},
    )


def comparison(seeds: Sequence[int], entries: Sequence[Entry]) -> Comparison:
    proto = common.protocol("eval_v1", seeds)
    rows = []
    for i, j in ordered_pairs(len(entries)):
        mean, lo, hi = paired_bootstrap(entries[i].scores, entries[j].scores)
        rows.append(PairedRow(a=entries[i].id, b=entries[j].id, mean=mean, lo=lo, hi=hi,
                              n_resamples=N_RESAMPLES, seed=SEED))  # fmt: skip
    return Comparison(
        protocol=proto.id,
        protocol_hash=proto.protocol_hash(),
        complete=True,
        seeds=tuple(seeds),
        platform=platform_tag(),
        versions=run_versions(),
        entries=tuple(entries),
        paired=tuple(rows),
    )


def pair(pid: Sequence[tuple[float, str]], ppo: Sequence[tuple[float, str]]) -> Comparison:
    seeds = tuple(range(len(pid)))
    return comparison(seeds, [
        ent("pid_example", [s for s, _ in pid], [t for _, t in pid], sha256=PID_SHA),
        ent("ppo", [s for s, _ in ppo], [t for _, t in ppo], sha256=FAKE, policy=FAKE),
    ])  # fmt: skip


# --- §5.2 seed rules ------------------------------------------------------------------------------


def test_hero_takes_the_largest_gap_where_pid_fell_and_ppo_survived() -> None:
    c = pair(
        [(40.0, "fell"), (10.0, "timeout"), (50.0, "floor_contact"), (45.0, "fell")],
        [(90.0, "timeout"), (99.0, "timeout"), (95.0, "timeout"), (80.0, "fell")],
    )
    pick = render.hero_seed(c, c.entries[1], c.entries[0])
    # seed 1 has the largest gap overall (89) but PID survived there; seed 3's PPO fell
    assert (pick.index, pick.gap, pick.rule) == (0, 50.0, "pid_fell")


def test_hero_ties_go_to_the_lowest_index() -> None:
    c = pair([(40.0, "fell"), (30.0, "fell"), (40.0, "fell")],
             [(90.0, "timeout"), (80.0, "timeout"), (90.0, "timeout")])  # fmt: skip
    assert render.hero_seed(c, c.entries[1], c.entries[0]).index == 0


def test_hero_falls_back_to_the_largest_gap_overall() -> None:
    c = pair([(70.0, "timeout"), (40.0, "fell"), (60.0, "timeout")],
             [(90.0, "timeout"), (50.0, "fell"), (90.0, "timeout")])  # fmt: skip
    pick = render.hero_seed(c, c.entries[1], c.entries[0])
    assert (pick.index, pick.gap, pick.rule) == (2, 30.0, "largest_gap")


def test_the_median_seed_is_the_lower_median() -> None:
    c = pair(
        [(50.0, "timeout")] * 4,
        [(60.0, "timeout"), (51.0, "timeout"), (58.0, "timeout"), (53.0, "timeout")],
    )
    pick = render.median_seed(c, c.entries[1], c.entries[0])
    assert (pick.index, pick.gap) == (3, 3.0)  # gaps 10, 1, 8, 3: lower median 3
    tied = pair([(50.0, "timeout")] * 3, [(52.0, "timeout")] * 3)
    assert render.median_seed(tied, tied.entries[1], tied.entries[0]).index == 1


def test_the_gallery_is_the_three_lowest_ties_to_the_lowest_index() -> None:
    c = pair(
        [(0.0, "timeout")] * 5,
        [(70.0, "fell"), (20.0, "fell"), (90.0, "timeout"), (20.0, "fell"), (30.0, "fell")],
    )
    assert render.lowest_seeds(c.entries[1]) == [1, 3, 4]
    with pytest.raises(ValueError, match="at least 3"):
        render.lowest_seeds(pair([(0.0, "timeout")], [(0.0, "timeout")]).entries[1])


# --- the synthetic phase2.json --------------------------------------------------------------------


def history_row(step: int, *, eligible: bool, score: float, success: float, frac: float | None,
                cpu: float | None, sha256: str) -> dict[str, Any]:  # fmt: skip
    return {
        "step": step, "success_rate": success, "score": score, "median_energy_j": 3.0,
        "mean_saturation": 0.03, "extras": {}, "eligible": eligible, "std_mean": 0.3,
        "frac_mean_outside_1": frac, "excess_penalty_mean": 0.001, "cpu_s": cpu,
        "wall_s": 5.0, "onnx_sha256": sha256,
    }  # fmt: skip


def summary(name: str, algo: str, seed: int, rows: Sequence[dict[str, Any]], *,
            cpu_s: float, best: int | None = None) -> dict[str, Any]:  # fmt: skip
    best_row = rows[-1] if best is None else rows[best]
    final = rows[-1]["step"]
    return {
        "run": name, "algo": algo, "seed": seed, "config_sha256": sha(f"c {name}"),
        "config_file_sha256": sha(f"f {name}"), "total_timesteps": final,
        "num_timesteps": final, "selection_protocol_hash": FAKE, "env_algo_version": 1,
        "bundle_hash": FAKE, "git_commit": "0" * 40, "git_dirty": False,
        "versions": {"python": "3.12.3"}, "mp_start_method": "forkserver", "torch_threads": 1,
        "wall_clock_s": 600.0, "train_s": 500.0, "selection_s": 100.0, "cpu_s": cpu_s,
        "load_avg_start": [3.0, 2.0, 1.0], "load_avg_end": [4.0, 3.0, 2.0],
        "peak_tree_rss_mib": 3000.0, "peak_procs": 16, "selection_history": list(rows),
        "best": {"step": best_row["step"], "eligible": best_row["eligible"], "rule": "pid",
                 "onnx_sha256": best_row["onnx_sha256"]},
        "flags": [] if best_row["eligible"] else ["no_eligible"], "e_ref_j": 13.1,
        "attempts": [],
    }  # fmt: skip


def ppo_summary(seed: int, policy: str) -> dict[str, Any]:
    name = f"ppo_s{seed}"
    rows = [
        history_row(500_000, eligible=False, score=60.0, success=0.5, frac=0.30 - 0.1 * seed,
                    cpu=20.0, sha256=sha(f"{name} 1")),
        history_row(1_000_000, eligible=seed != 2, score=80.0, success=0.8, frac=0.02 * seed,
                    cpu=20.0, sha256=sha(f"{name} 2")),
        history_row(2_000_000, eligible=True, score=85.0 + seed, success=0.9, frac=0.01,
                    cpu=20.0, sha256=policy),
    ]  # fmt: skip
    return summary(name, "ppo", seed, rows, cpu_s=1060.0)


def sac_summary(seed: int) -> dict[str, Any]:
    name = f"sac_s{seed}"
    rows = [
        history_row(500_000, eligible=False, score=65.0 + seed, success=0.4, frac=0.1,
                    cpu=10.0, sha256=sha(f"{name} 1")),
        history_row(1_000_000, eligible=False, score=60.0, success=0.35, frac=0.1, cpu=10.0,
                    sha256=sha(f"{name} 2")),
    ]  # fmt: skip
    return summary(name, "sac", seed, rows, cpu_s=1220.0, best=0)


def ablation_summary() -> dict[str, Any]:
    rows = [
        history_row(900_000, eligible=False, score=70.0, success=0.5, frac=0.40, cpu=None,
                    sha256=sha("abl 1")),
        history_row(1_015_808, eligible=False, score=71.0, success=0.52, frac=0.35, cpu=None,
                    sha256=sha("abl 2")),
    ]  # fmt: skip
    return summary("ablate_nopenalty", "ppo", 1, rows, cpu_s=600.0)


TERMS_EVAL = {
    "pid": ["timeout", "fell", "timeout", "fell", "timeout", "timeout"],
    "lqr": ["timeout"] * 6,
}
SCORES_EVAL = {
    "pid": [72.0, 40.0, 75.0, 45.0, 70.0, 71.0],
    "lqr": [85.0, 84.0, 86.0, 83.0, 82.0, 81.0],
}


def runs_and_fresh(policy: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs = [{"role": "ppo", "summary": ppo_summary(k, policy if k == 0 else sha(f"p{k}"))}
            for k in range(3)]  # fmt: skip
    runs += [{"role": "sac", "summary": sac_summary(k)} for k in range(3)]
    runs.append({"role": "ablation", "summary": ablation_summary()})

    def fresh(seeds: Sequence[int], cal: bool) -> dict[str, Any]:
        n = len(seeds)
        if cal:
            pid = ent("pid_example", [60.0 + k for k in range(n)], ["fell", "timeout"] * (n // 2),
                      sha256=PID_SHA)  # fmt: skip
            lqr = ent("lqr_example", [80.0] * n, ["timeout"] * n, sha256=LQR_SHA)
        else:
            pid = ent("pid_example", SCORES_EVAL["pid"], TERMS_EVAL["pid"], sha256=PID_SHA)
            lqr = ent("lqr_example", SCORES_EVAL["lqr"], TERMS_EVAL["lqr"], sha256=LQR_SHA)
        entries = [pid, lqr]
        for k in range(3):
            scores = [90.0 - (5 * ((i * 3 + k) % n) if cal else i) for i in range(n)]
            terms = ["timeout" if s >= 75 else "fell" for s in scores]
            entries.append(ent(f"ppo_s{k}", scores, terms, sha256=sha(f"sub {k}"),
                               policy=policy if k == 0 else sha(f"p{k}")))  # fmt: skip
        for k in range(3):
            entries.append(ent(f"sac_s{k}", [55.0 + k] * n, ["fell"] * n,
                               sha256=sha(f"sac {k}"), policy=sha(f"s{k}")))  # fmt: skip
        return json.loads(to_json(comparison(seeds, entries)))

    rows = [{"run": f"ppo_s{k}", "entry": f"ppo_s{k}", "policy_sha256":
             policy if k == 0 else sha(f"p{k}"), "report_only": k != 0,
             "below_pid_gate": False} for k in range(3)]  # fmt: skip
    rows += [{"run": f"sac_s{k}", "entry": f"sac_s{k}", "policy_sha256": sha(f"s{k}"),
              "report_only": True, "below_pid_gate": True} for k in range(3)]  # fmt: skip
    return runs, {"eval_v1": fresh(EVAL_SEEDS, False), "calibration": fresh(CAL_SEEDS, True),
                  "rows": rows}  # fmt: skip


PILOT = {
    "energy_cost_variant": {"score": 64.9, "success": 0.505, "saturated_share": 0.82,
                            "source": "pilot §2.2"},
    "log_std_init": {"shipped": 0.0, "worse": [-0.5, -1.0], "source": "pilot §2.2"},
    "constant_lr": {"seeds_regressed": 1, "of": 3, "after_steps": 2_300_000,
                    "source": "pilot §0"},
}  # fmt: skip
BASELINES = {"pid": {"score": 70.6, "success_rate": 0.655},
             "lqr": {"score": 82.5, "success_rate": 0.9}}  # fmt: skip


def report_doc(policy: str, render_values: dict[str, Any] | None = None) -> dict[str, Any]:
    runs, fresh = runs_and_fresh(policy)
    return {
        "format": "topple.phase2/1",
        "runs": runs,
        "fresh": fresh,
        "shipped": {"run": "ppo_s0", "policy_sha256": policy,
                    "attempts": [{"run": "ppo_s0", "failed_gates": [], "failed_on": None}]},
        "g11": [{"run": f"ppo_s{k}", "eligible": True, "g2": True, "g2b": k != 1,
                 "g3": True, "g4": k != 2} for k in range(3)],
        "pilot": PILOT,
        "render": {"selection_baselines": BASELINES, **(render_values or {})},
    }  # fmt: skip


def report_of(doc: dict[str, Any]) -> common.Phase2Report:
    return common.parse_phase2(json.dumps(doc).encode("utf-8"))


@pytest.fixture(scope="module")
def policy_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny exported policy standing in for ``submissions/ppo_balance``."""
    root = tmp_path_factory.mktemp("policy")
    run = make_run(root / "runs", name="ppo_s0")
    sub = root / "ppo_balance"
    export_policy(run.path, sub, test_fixture="ppo_balance", fixture_dir=root / "data")
    return sub


@pytest.fixture(scope="module")
def policy_sha(policy_dir: Path) -> str:
    _, policy = submission_hashes(policy_dir)
    assert policy is not None
    return policy


@pytest.fixture(scope="module")
def fallen() -> Trajectory:
    """A short real episode (zero command: it falls), with one scenario push added."""

    class Zero:
        def reset(self, rng_seed: int) -> None:
            pass

        def act(self, obs: object) -> tuple[float, float]:
            return (0.0, 0.0)

    env = BalanceEnv("ideal", record=True)
    try:
        traj = run_episode(env, lambda spec: Zero(), 1, 0).trajectory
    finally:
        env.close()
    scenario = dict(traj.meta["scenario"])
    push = {"start_step": 10, "n_substeps": 50, "magnitude_n": 4.0, "forward_sign": 1.0,
            "lateral_ratio": 0.0}  # fmt: skip
    late = {**push, "start_step": traj.steps + 5}  # after the end: not the last push
    scenario["pushes"] = [push, late]
    return replace(traj, meta={**traj.meta, "scenario": scenario})


# --- make_media with its expensive parts replaced ------------------------------------------------


def curve(sign: int, survived: Sequence[Sequence[bool]], ident: str) -> ShoveCurve:
    return ShoveCurve(id=ident, sha256=FAKE, policy_sha256=None, protocol="balance-v1/clean",
                      protocol_hash=FAKE, forward_sign=sign, impulses=(0.3, 0.325, 0.35),
                      seeds=(0, 1), survived=tuple(tuple(r) for r in survived))  # fmt: skip


T, F = True, False
SHOVES = [
    (curve(1, [[T, T], [T, F], [F, F]], "pid"), curve(-1, [[T, T], [T, T], [F, F]], "pid")),
    (curve(1, [[T, T], [T, T], [T, T]], "lqr"), curve(-1, [[T, T], [T, F], [F, F]], "lqr")),
    (curve(1, [[T, T], [T, T], [F, F]], "ppo"), curve(-1, [[T, T], [T, T], [T, T]], "ppo")),
]


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, fallen: Trajectory) -> dict[str, Any]:
    """Re-runs return ``fallen`` with the stored score (``wrong`` set: off by 1e-9); frames
    are noise (no GL); the shove bench returns ``SHOVES``."""
    state: dict[str, Any] = {"calls": [], "wrong": False, "frames": [], "fallen": fallen}

    def rerun(sub: Path, proto: Any, indices: Sequence[int], work: Path, name: str
              ) -> dict[int, tuple[Trajectory, float]]:  # fmt: skip
        state["calls"].append((name, sorted(set(indices))))
        fresh = state["report"].fresh
        cmp = fresh.calibration if name == "ppo_cal" else fresh.eval_v1
        entry = {"pid": 0, "lqr": 1}.get(name)
        e = cmp.entries[entry] if entry is not None else render.shipped_entry(state["report"], cmp)
        bump = 1e-9 if state["wrong"] else 0.0
        return {i: (fallen, e.scores[i] + bump) for i in indices}

    def frames(trajs: Any, labels: Any, *, pane: int, fps: int) -> Iterator[Frame]:
        state["frames"].append((pane, fps, tuple(labels)))
        rng = np.random.default_rng(0)
        for _ in range(fps // 5 + 1):  # sizes still order by fps
            yield rng.integers(0, 256, size=(pane, pane * len(trajs), 3), dtype=np.uint8)

    monkeypatch.setattr(render, "rerun", rerun)
    monkeypatch.setattr(video, "side_by_side_frames", frames)
    monkeypatch.setattr(render, "run_shoves", lambda subs, *a: SHOVES)
    return state


def media(fakes: dict[str, Any], doc: dict[str, Any], policy_dir: Path, tmp_path: Path
          ) -> dict[str, Any]:  # fmt: skip
    fakes["report"] = report_of(doc)
    return render.make_media(fakes["report"], shipped=policy_dir, out_dir=tmp_path / "out",
                             work=tmp_path / "work", workers=1)  # fmt: skip


def test_make_media_writes_every_file_and_returns_the_values(
    fakes: dict[str, Any], policy_dir: Path, policy_sha: str, tmp_path: Path
) -> None:
    values = media(fakes, report_doc(policy_sha), policy_dir, tmp_path)
    out = tmp_path / "out"
    wanted = {name for name in common.RESULT_FILES if not name.startswith("ci_ubuntu/")}
    assert {p.name for p in out.iterdir()} == wanted
    for name in wanted:
        cap = common.RESULT_FILES[name]
        assert cap is None or (out / name).stat().st_size <= cap
    assert json.loads(json.dumps(values, allow_nan=False)) == values
    common.check_json_numbers(values)
    # eval: PID fell at 1 and 3; shipped (ppo_s0) scores 90 - i, so gaps 49 (i=1), 42 (i=3)
    hero = values["hero"]
    assert (hero["index"], hero["seed"], hero["gap"], hero["rule"]) == (
        1, EVAL_SEEDS[1], 49.0, "pid_fell"
    )  # fmt: skip
    assert hero["gif_rung"] == {"width": 480, "fps": 20}
    assert hero["mp4_frames"] == 11
    assert (480, 50, ("PID", "PPO (shipped)")) in fakes["frames"]  # 960 x 480 at 50 fps
    # PPO - PID: 18, 49, 13, 42, 16, 14 -> lower median 16 (index 4); PPO - LQR: 5, 5, 2,
    # 4, 4, 4 -> sorted (2, 4, 4, 4, 5, 5) by value then index: the third is index 4
    assert values["median"]["pid_ppo"]["index"] == 4
    assert values["median"]["lqr_ppo"]["index"] == 4
    assert dict(fakes["calls"]) == {
        "ppo": [1, 4], "pid": [1, 4], "lqr": [4], "ppo_cal": [2, 5, 7]
    }  # fmt: skip
    gallery = values["gallery"]
    assert [g["index"] for g in gallery] == [5, 2, 7]  # cal scores 90 - 5 * (3i % 8)
    assert [g["score"] for g in gallery] == [55.0, 60.0, 65.0]
    assert gallery[0]["gif_rung"] == {"width": 320, "fps": 10}
    dt = fakes["fallen"].meta["timestep"]
    assert gallery[0]["last_push"] == pytest.approx({"impulse_ns": 4.0 * 50 * dt, "time_s": 0.1})
    assert gallery[0]["termination"] in ("fell", "floor_contact")
    assert gallery[0]["fall_cause"] is None
    shove = {c["label"]: c for c in values["shove"]["controllers"]}
    assert shove["PID"]["i50_forward"] == pytest.approx(0.325)
    assert shove["PID"]["i50_backward"] == pytest.approx(0.3375)
    assert shove["PID"]["i50_min"] == pytest.approx(0.325)
    assert shove["LQR"]["i50_forward"] is None
    assert shove["LQR"]["i50_min"] is None
    assert shove["PPO (shipped)"]["i50_forward"] == pytest.approx(0.3375)
    assert shove["PPO (shipped)"]["survival_forward"] == [1.0, 1.0, 0.0]
    assert values["learning"]["env_steps_to_pid"] == {
        "ppo_s0": [500_000, 1_000_000], "ppo_s1": [500_000, 1_000_000],
        "ppo_s2": [1_000_000, 2_000_000],
    }  # fmt: skip
    rd = values["reward_design"]
    assert (rd["ablation"]["step"], rd["ablation"]["frac_mean_outside_1"]) == (1_015_808, 0.35)
    assert (rd["ppo"]["run"], rd["ppo"]["frac_mean_outside_1"]) == ("ppo_s1", 0.02)
    assert len(values["sac_vs_ppo"]) == 3


def test_a_rerun_that_is_not_bit_exact_fails(
    fakes: dict[str, Any], policy_dir: Path, policy_sha: str, tmp_path: Path
) -> None:
    fakes["wrong"] = True
    with pytest.raises(RuntimeError, match="re-ran to"):
        media(fakes, report_doc(policy_sha), policy_dir, tmp_path)


def test_the_selection_baselines_are_required_before_any_work(
    fakes: dict[str, Any], policy_dir: Path, policy_sha: str, tmp_path: Path
) -> None:
    doc = report_doc(policy_sha)
    doc["render"] = {}
    with pytest.raises(KeyError, match="selection_baselines"):
        media(fakes, doc, policy_dir, tmp_path)
    assert fakes["calls"] == []


def test_another_policy_dir_is_refused(
    fakes: dict[str, Any], policy_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="not the shipped policy"):
        media(fakes, report_doc(sha("other policy")), policy_dir, tmp_path)


def test_another_platform_is_refused(
    fakes: dict[str, Any], policy_dir: Path, policy_sha: str, tmp_path: Path
) -> None:
    doc = report_doc(policy_sha)
    doc["fresh"]["eval_v1"]["platform"] = "plan9-mips"
    with pytest.raises(ValueError, match="bit-exact only there"):
        media(fakes, doc, policy_dir, tmp_path)


def test_an_mp4_over_its_cap_raises_and_is_not_written(
    fakes: dict[str, Any], policy_dir: Path, policy_sha: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    monkeypatch.setattr(render, "RESULT_FILES", {**common.RESULT_FILES, "hero.mp4": 100})
    with pytest.raises(ValueError, match=r"hero\.mp4 is \d+ bytes, over its 100-byte cap"):
        media(fakes, report_doc(policy_sha), policy_dir, tmp_path)
    assert not (tmp_path / "out" / "hero.mp4").exists()


def test_a_gif_over_its_last_rung_raises(
    fakes: dict[str, Any], policy_dir: Path, policy_sha: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    monkeypatch.setattr(render, "RESULT_FILES", {**common.RESULT_FILES, "hero.gif": 1000})
    with pytest.raises(video.GifTooLargeError):
        media(fakes, report_doc(policy_sha), policy_dir, tmp_path)
    assert not (tmp_path / "out" / "hero.gif").exists()


def test_the_rung_used_is_recorded(
    fakes: dict[str, Any], policy_dir: Path, policy_sha: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, fallen: Trajectory,
) -> None:  # fmt: skip
    """With a cap only the smaller rungs fit, the ladder steps down and says which rung."""
    sizes = {}
    for rung in video.GALLERY_LADDER[:2]:
        out = tmp_path / f"probe_{rung.width}_{rung.fps}.gif"
        video.render_gif_ladder([fallen], ["x"], out, (rung,), max_bytes=10**9)
        sizes[rung] = out.stat().st_size
    cap = sizes[video.GALLERY_LADDER[1]]
    assert sizes[video.GALLERY_LADDER[0]] > cap
    caps = {**common.RESULT_FILES, **{f"gallery_{k}.gif": cap for k in (1, 2, 3)}}
    monkeypatch.setattr(render, "RESULT_FILES", caps)
    values = media(fakes, report_doc(policy_sha), policy_dir, tmp_path / "run")
    assert [g["gif_rung"] for g in values["gallery"]] == [{"width": 320, "fps": 5}] * 3


# --- pure values ----------------------------------------------------------------------------------


def core(doc: dict[str, Any]) -> common.RunSummary:
    return common.run_summary(common.RunEntry(role="ppo", summary=doc))


def test_cpu_minutes_spread_training_and_cumulate_selection() -> None:
    s = core(ppo_summary(0, FAKE))  # cpu_s 1060, three evals of 20 s: training 1000 s
    assert render.cpu_minutes(s) == pytest.approx(
        [(1000 * 0.25 + 20) / 60, (1000 * 0.5 + 40) / 60, (1000 + 60) / 60]
    )
    ab = core(ablation_summary())  # no per-eval cpu_s: the whole 600 s is spread
    assert render.cpu_minutes(ab) == pytest.approx([600 * 900_000 / 1_015_808 / 60, 600 / 60])


def test_env_steps_to_pid_and_the_nearest_eval() -> None:
    s = core(ppo_summary(2, FAKE))
    assert render.steps_to_pid(s) == [1_000_000, 2_000_000]
    first = core(summary("x", "ppo", 0, [history_row(10, eligible=True, score=80.0,
             success=0.9, frac=None, cpu=None, sha256=FAKE)], cpu_s=1.0))  # fmt: skip
    assert render.steps_to_pid(first) == [0, 10]
    never = core(sac_summary(0) | {"algo": "ppo"})
    assert render.steps_to_pid(never) is None
    rows = core(ppo_summary(0, FAKE)).selection_history
    assert render.nearest_row(rows, 750_000).step == 500_000  # a tie: the earlier eval
    assert render.nearest_row(rows, 1_400_000).step == 1_000_000


def test_reward_design_needs_frac_mean_outside_1(policy_sha: str) -> None:
    doc = report_doc(policy_sha)
    for row in doc["runs"][1]["summary"]["selection_history"]:
        row["frac_mean_outside_1"] = None
    with pytest.raises(ValueError, match="frac_mean_outside_1"):
        render.reward_design(report_of(doc))


def test_sac_vs_ppo_pairs_by_seed(policy_sha: str) -> None:
    (first, *_) = render.sac_vs_ppo(report_of(report_doc(policy_sha)))
    assert (first["sac_run"], first["ppo_run"]) == ("sac_s0", "ppo_s0")
    assert first["sac_best"]["step"] == 500_000
    assert first["equal_steps_ppo"]["step"] == 500_000
    assert first["ppo_best"]["step"] == 2_000_000
    assert first["ppo_final"]["score"] == 85.0
    # SAC's best at 500k: (1200 * 0.5 + 10) / 60 CPU-min; PPO's evals: 4.5, 9.0, 17.7
    assert first["sac_best"]["cpu_min"] == pytest.approx(610 / 60)
    assert first["equal_cpu_ppo"]["step"] == 1_000_000


# --- phase2.md ------------------------------------------------------------------------------------


@pytest.fixture
def rendered(fakes: dict[str, Any], policy_dir: Path, policy_sha: str, tmp_path: Path
             ) -> dict[str, Any]:  # fmt: skip
    values = media(fakes, report_doc(policy_sha), policy_dir, tmp_path)
    return report_doc(policy_sha, values)


def test_markdown_fills_every_field(rendered: dict[str, Any]) -> None:
    text = render.render_markdown(report_of(rendered))
    assert "{" not in text
    assert "}" not in text
    for section in ("## 1. Hero", "## 2. Scoreboard", "## 3. Shove test", "## 4. Learning",
                    "## 5. Failure gallery", "## 6. Reward-design", "## 7. SAC vs PPO",
                    "## 8. README snippet"):  # fmt: skip
        assert section in text
    assert render.HERO_CAPTION in text
    assert "2 of 3 PPO seeds reached the PID gate" in text
    assert "2 of 3 at LQR level (G2b)" in text
    assert "(below PID gate)" in text
    assert "Pending CI evidence" in text
    assert "from 35.0% to 2.0% at 1M steps" in text
    assert "log_std 0 beat \u22120.5 and \u22121 (pilot measurement; pilot §2.2)" in text
    assert "| PID | 0.325 | " in text
    assert "| LQR | none | 0.325 | none |" in text
    assert "ppo_s2: env steps to PID level in [1,000,000, 2,000,000]" in text
    assert "learner-throughput-limited (UTD 0.25, serial updates)" in text
    assert f"policy {rendered['shipped']['policy_sha256'][:12]}" in text
    assert "phase2/gallery_3.gif" in text


def test_markdown_shows_the_ubuntu_block_when_ci_evidence_exists(rendered: dict[str, Any]) -> None:
    cal = rendered["fresh"]["calibration"]
    ubuntu = {**cal, "entries": cal["entries"][:3]}
    ubuntu["paired"] = [r for r in cal["paired"] if r["a"] in ("lqr_example", "ppo_s0")
                        and r["b"] in ("pid_example", "lqr_example")]  # fmt: skip
    rendered["render"]["ci"] = {"run_id": 4242, "head_sha": "0" * 40, "results": {},
                                "cmp_cal": ubuntu}  # fmt: skip
    text = render.render_markdown(report_of(rendered))
    assert "Pending CI evidence" not in text
    assert "CI run 4242" in text
    block = text.split("### Seeds 0..199 (ubuntu CI)")[1].split("## 3.")[0]
    assert block.count("\n| PPO (shipped") == 1


@pytest.mark.parametrize(
    "path",
    [("hero", "gap"), ("hero", "gif_rung"), ("median", "lqr_ppo"), ("gallery",),
     ("shove", "controllers"), ("learning", "env_steps_to_pid"), ("reward_design", "ppo"),
     ("sac_vs_ppo",)],
)  # fmt: skip
def test_markdown_fails_on_a_missing_field(rendered: dict[str, Any], path: tuple[str, ...]) -> None:
    holder = rendered["render"]
    for key in path[:-1]:
        holder = holder[key]
    del holder[path[-1]]
    with pytest.raises(KeyError, match="is missing"):
        render.render_markdown(report_of(rendered))


def test_markdown_fails_on_a_missing_pilot_constant(rendered: dict[str, Any]) -> None:
    del rendered["pilot"]["constant_lr"]["after_steps"]
    with pytest.raises(KeyError):
        render.render_markdown(report_of(rendered))


def test_markdown_fails_on_an_ill_typed_field(rendered: dict[str, Any]) -> None:
    rendered["render"]["hero"]["gap"] = "big"
    with pytest.raises(ValueError, match=r"hero\.gap"):
        render.render_markdown(report_of(rendered))


# --- the real harness ---------------------------------------------------------------------------


@pytest.mark.render
def test_a_real_gallery_gif(fallen: Trajectory, tmp_path: Path) -> None:
    rung = render._gif([fallen], ["PPO seed 1"], "gallery_1.gif", tmp_path,
                       (video.GifRung(64, 10),))  # fmt: skip
    assert rung == {"width": 64, "fps": 10}
    assert 0 < (tmp_path / "gallery_1.gif").stat().st_size <= common.RESULT_FILES["gallery_1.gif"]


@pytest.mark.slow
def test_a_rerun_equals_the_pool_episode(policy_dir: Path, tmp_path: Path) -> None:
    """The in-process re-run is bit-exact with ``topple compare``'s pool episode, for a
    Python controller and an ONNX policy."""
    proto = common.protocol("eval_v1", EVAL_SEEDS[:2])
    cmp = compare([REPO / common.PID_SUBMISSION, policy_dir], proto, workers=1)
    for sub, e, name in ((REPO / common.PID_SUBMISSION, cmp.entries[0], "pid"),
                         (policy_dir, cmp.entries[1], "ppo")):  # fmt: skip
        trajs = render.episodes(sub, cmp, e, [0, 1], tmp_path, name)
        assert sorted(trajs) == [0, 1]
        assert (tmp_path / f"{name}_{EVAL_SEEDS[1]}.npz").is_file()
