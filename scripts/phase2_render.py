"""The Phase 2a results package's media and ``phase2.md`` (docs/design/phase-2.md §5.2).

``phase2_results.py`` calls two functions:

- ``make_media(report, shipped=..., out_dir=..., work=..., workers=...)`` renders every file of
  ``RESULT_FILES`` but the ubuntu evidence into ``out_dir`` (``docs/results/phase2/``) and
  returns the values ``phase2.md`` needs; the caller merges them into ``report.render``.
- ``render_markdown(report)`` writes ``phase2.md`` from ``report`` alone (``report.render``
  included). A missing field raises ``KeyError`` or ``ValueError``: no number is hand-written.

What ``make_media`` reads from ``report``:

- **Seeds.** The per-seed scores of ``report.fresh``: entry 0 is PID, entry 1 LQR, and the
  shipped policy is the entry of the ``FreshRow`` whose run is ``report.shipped.run``. The hero
  seed (§5.2 item 1) is the eval_v1 seed with the largest score(PPO) - score(PID) among those
  where PID fell (``fell`` or ``floor_contact``) and PPO survived (``timeout``), ties to the
  lowest index; if there is none, the largest difference overall. Each median video's seed is
  the lower median of that pair's differences (PPO - PID, PPO - LQR), ties in value to the lower
  index. The failure gallery is the shipped policy's 3 lowest scores on 0..199.
- **Trajectories.** Those seeds are run again with the real harness (``run_episode`` on the
  protocol of the fresh comparison, with the same submissions), saved into ``work`` and read
  back with ``load_npz``. Episodes are bit-exact per platform, ``cpu_isa`` and onnxruntime
  version, so the platform must be the comparisons' and every re-run score must equal the
  stored one exactly (``RuntimeError`` otherwise).
- **Learning curves** from each run summary's ``selection_history``, with the PID and LQR
  selection-seed lines from ``report.render["selection_baselines"]`` (the caller fills it).
  CPU-minutes at an eval: the summary's ``cpu_s`` minus the rows' selection-eval ``cpu_s`` is
  spread over env steps in proportion (training CPU), plus the rows' ``cpu_s`` cumulated up to
  that eval (selection CPU); when a row has no ``cpu_s``, the whole ``cpu_s`` is spread.
- **Shove test** (§5.2 item 3) on PID, LQR and the shipped policy with ``topple.bench``.

Size caps (``RESULT_FILES``, §5.2) are enforced: an MP4 is rendered into ``work`` and copied
into ``out_dir`` only within its cap, a GIF ladder raises ``GifTooLargeError`` past its last
rung, and the whole of ``out_dir`` must stay within ``MAX_RESULTS_TOTAL_BYTES``.

Torch-free, like every ``scripts/phase2_*.py``.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median, stdev
from typing import TYPE_CHECKING, Final, Literal

from pydantic import JsonValue

from phase2_common import (
    LQR_SUBMISSION,
    MAX_RESULTS_TOTAL_BYTES,
    PID_SUBMISSION,
    REPO,
    RESULT_FILES,
    FreshRow,
    HistoryRowModel,
    Phase2Report,
    RunSummary,
    check_json_numbers,
    protocol,
    run_summary,
)
from topple import video
from topple._atomic import replacement
from topple.bench import (
    DEFAULT_IMPULSES,
    SHOVE_PROTOCOL,
    ShoveCurve,
    impulse_at_50,
    parse_impulses,
    run_shoves,
)
from topple.compare import Comparison, Entry, PairedRow, parse_comparison
from topple.evaluate import FALL_REASONS, EvalProtocol, platform_tag, run_episode, run_versions
from topple.loader import load_submission, submission_sha256
from topple.stats import bootstrap_ci
from topple.traj import Trajectory, load_npz
from topple.verify import protocol_max_steps

if TYPE_CHECKING:
    from matplotlib.figure import Figure

MINUS: Final = chr(0x2212)  # a typographic minus sign
SURVIVED: Final = "timeout"
HERO_PANE: Final = 480  # MP4 960 x 480: two square 480 px panes
HERO_FPS: Final = 50
GALLERY_SIZE: Final = 3
SHOVE_SEEDS: Final = tuple(range(100))
SHOVE_SIGNS: Final = (1, -1)
REWARD_STEP: Final = 1_000_000
"""§5.2 item 6: the evals nearest 1M steps."""
REWARD_SEED: Final = 1
"""§5.2 item 6: the PPO run compared with the ablation (``runs/ppo_s1``)."""
LABEL_PID: Final = "PID"
LABEL_LQR: Final = "LQR"
LABEL_PPO: Final = "PPO (shipped)"
CPU_AXIS: Final = (
    "train CPU-minutes: the run's process-tree cpu_s minus its selection evals' cpu_s, spread "
    "over env steps, plus the selection evals' cpu_s cumulated up to each eval"
)
HERO_CAPTION: Final = (
    f"Seed chosen as the largest PPO{MINUS}PID gap among eval_v1 seeds where PID fell; a median "
    "seed is below."
)
HERO_FALLBACK_CAPTION: Final = (
    f"No eval_v1 seed had PID fall while PPO survived, so the seed is the largest PPO{MINUS}PID "
    "gap overall; a median seed is below."
)
_UNFILLED: Final = re.compile(r"\{[A-Za-z_][A-Za-z0-9_.\[\]'\"]*\}")

# --- seed choice (§5.2 items 1 and 5) -------------------------------------------------------------


@dataclass(frozen=True)
class SeedPick:
    """Index ``index`` (seed ``seed``) of a comparison: ``gap`` = score(PPO) - score(other)."""

    index: int
    seed: int
    gap: float
    rule: Literal["pid_fell", "largest_gap", "median"]


def _gaps(ppo: Entry, other: Entry) -> list[float]:
    if len(ppo.scores) != len(other.scores) or not ppo.scores:
        raise ValueError("both entries need one score per seed")
    return [p - q for p, q in zip(ppo.scores, other.scores, strict=True)]


def hero_seed(cmp: Comparison, ppo: Entry, pid: Entry) -> SeedPick:
    """§5.2 item 1: the largest PPO - PID gap among the seeds where PID fell and PPO survived,
    ties to the lowest index; if no seed qualifies, the largest gap overall."""
    gaps = _gaps(ppo, pid)
    fell = [
        k
        for k in range(len(gaps))
        if pid.termination[k] in FALL_REASONS and ppo.termination[k] == SURVIVED
    ]
    pool = fell or list(range(len(gaps)))
    k = max(pool, key=lambda i: (gaps[i], -i))
    return SeedPick(k, cmp.seeds[k], gaps[k], "pid_fell" if fell else "largest_gap")


def median_seed(cmp: Comparison, ppo: Entry, other: Entry) -> SeedPick:
    """The lower median of the PPO - other differences (equal values ordered by index)."""
    gaps = _gaps(ppo, other)
    order = sorted(range(len(gaps)), key=lambda i: (gaps[i], i))
    k = order[(len(order) - 1) // 2]
    return SeedPick(k, cmp.seeds[k], gaps[k], "median")


def lowest_seeds(entry: Entry, count: int = GALLERY_SIZE) -> list[int]:
    """The indices of ``entry``'s ``count`` lowest scores, ties to the lowest index."""
    if len(entry.scores) < count:
        raise ValueError(f"the failure gallery needs at least {count} seeds")
    return sorted(range(len(entry.scores)), key=lambda i: (entry.scores[i], i))[:count]


# --- the report's entries -----------------------------------------------------------------------


def shipped_row(report: Phase2Report) -> FreshRow:
    rows = [row for row in report.fresh.rows if row.run == report.shipped.run]
    if len(rows) != 1:
        raise ValueError("exactly one fresh row must belong to the shipped run")
    if rows[0].policy_sha256 != report.shipped.policy_sha256:
        raise ValueError("the shipped run's fresh row is not the shipped policy")
    return rows[0]


def entry_named(cmp: Comparison, name: str) -> Entry:
    found = [e for e in cmp.entries if e.id == name]
    if len(found) != 1:
        raise ValueError("a fresh row's entry must be exactly one comparison entry")
    return found[0]


def shipped_entry(report: Phase2Report, cmp: Comparison) -> Entry:
    entry = entry_named(cmp, shipped_row(report).entry)
    if entry.policy_sha256 != report.shipped.policy_sha256:
        raise ValueError("the shipped entry's policy_sha256 is not the shipped policy's")
    return entry


def summaries(report: Phase2Report) -> dict[str, tuple[str, RunSummary]]:
    """Every reported run's role and validated summary, by run name."""
    out: dict[str, tuple[str, RunSummary]] = {}
    for entry in report.runs:
        summary = run_summary(entry)
        if summary.run in out:
            raise ValueError("two reported runs have one name")
        out[summary.run] = (entry.role, summary)
    return out


def runs_of(report: Phase2Report, role: str) -> list[RunSummary]:
    """The runs of ``role``, by seed (then name)."""
    found = [s for r, s in summaries(report).values() if r == role]
    return sorted(found, key=lambda s: (s.seed, s.run))


# --- learning curves (§5.2 item 4) --------------------------------------------------------------


def cpu_minutes(summary: RunSummary) -> list[float]:
    """CPU-minutes at each selection eval (``CPU_AXIS``)."""
    rows = summary.selection_history
    total = summary.num_timesteps or max(row.step for row in rows) or 1
    evals = [row.cpu_s for row in rows]
    if any(c is None for c in evals):
        return [summary.cpu_s * row.step / total / 60.0 for row in rows]
    cumulated, out = 0.0, []
    train = max(summary.cpu_s - math.fsum(c for c in evals if c is not None), 0.0)
    for row, cpu in zip(rows, evals, strict=True):
        cumulated += cpu or 0.0
        out.append((train * row.step / total + cumulated) / 60.0)
    return out


def steps_to_pid(summary: RunSummary) -> list[int] | None:
    """§5.2 item 4: [previous eval's step (0 for the first), first eligible eval's step]."""
    previous = 0
    for row in summary.selection_history:
        if row.eligible:
            return [previous, row.step]
        previous = row.step
    return None


def nearest_row(rows: Sequence[HistoryRowModel], step: float) -> HistoryRowModel:
    """The row nearest ``step``; a tie goes to the earlier eval."""
    return min(rows, key=lambda r: (abs(r.step - step), r.step))


def _baseline_lines(report: Phase2Report) -> dict[str, tuple[float, float]]:
    """``render["selection_baselines"]``: {"pid"|"lqr": (score, success_rate)}."""
    raw = _get(report.render, "selection_baselines")
    return {
        name: (_num(raw, name, "score"), _num(raw, name, "success_rate")) for name in ("pid", "lqr")
    }


def plot_learning_curves(report: Phase2Report, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    lines = _baseline_lines(report)
    fig = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    styles = ("-", "--", ":")
    colors = {"ppo": "#2a78d6", "sac": "#eb6834"}
    for algo in ("ppo", "sac"):
        for k, s in enumerate(runs_of(report, algo)):
            steps = [r.step / 1e6 for r in s.selection_history]
            cpu = cpu_minutes(s)
            label = f"{algo.upper()} seed {s.seed}"
            style = {"color": colors[algo], "linestyle": styles[k % 3], "linewidth": 2}
            for col, x in enumerate((steps, cpu)):
                axes[0][col].plot(x, [r.success_rate for r in s.selection_history], **style,
                                  marker="o", markersize=3, label=label)  # fmt: skip
                axes[1][col].plot(x, [r.score for r in s.selection_history], **style,
                                  marker="o", markersize=3, label=label)  # fmt: skip
    for row, (metric, unit) in enumerate((("success", 1), ("score", 0))):
        for col, xlabel in enumerate(("env steps (millions)", "train CPU-minutes")):
            ax = axes[row][col]
            for name, color, dash in (("pid", "#52514e", "--"), ("lqr", "#0b0b0b", ":")):
                ax.axhline(lines[name][unit], color=color, linestyle=dash, linewidth=1.5,
                           label=f"{name.upper()} (selection seeds)")  # fmt: skip
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"selection {metric}")
            ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Selection evals (seeds 2000..2199) per run")
    _save_png(fig, path)


# --- shove test (§5.2 item 3) -------------------------------------------------------------------


def shove_values(labelled: Sequence[tuple[str, tuple[ShoveCurve, ...]]]) -> dict[str, JsonValue]:
    """Each controller's i50 per direction and their minimum (``None`` unless both exist)."""
    controllers: list[JsonValue] = []
    impulses: list[JsonValue] = []
    for label, curves in labelled:
        by_sign = {c.forward_sign: c for c in curves}
        if set(by_sign) != set(SHOVE_SIGNS):
            raise ValueError("a shove result needs both directions")
        fwd, bwd = impulse_at_50(by_sign[1]), impulse_at_50(by_sign[-1])
        impulses = list(by_sign[1].impulses)
        controllers.append(
            {
                "label": label,
                "id": by_sign[1].id,
                "i50_forward": fwd,
                "i50_backward": bwd,
                "i50_min": None if fwd is None or bwd is None else min(fwd, bwd),
                "survival_forward": list(by_sign[1].survival),
                "survival_backward": list(by_sign[-1].survival),
            }
        )
    seeds = labelled[0][1][0].seeds if labelled else ()
    return {
        "protocol": SHOVE_PROTOCOL,
        "seeds": [min(seeds), max(seeds)] if seeds else [],
        "n_seeds": len(seeds),
        "impulses": impulses,
        "controllers": controllers,
    }


def plot_shove(values: Mapping[str, JsonValue], path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, sharey=True)
    x = [_num(v) for v in _list(values, "impulses")]
    palette = (("#52514e", "--"), ("#0b0b0b", ":"), ("#2a78d6", "-"))  # PID, LQR, PPO
    for k, c in enumerate(_list(values, "controllers")):
        color, dash = palette[k % 3]
        for ax, key in zip(axes, ("survival_forward", "survival_backward"), strict=True):
            ys = [_num(v) for v in _list(c, key)]
            ax.plot(x, ys, color=color, linestyle=dash, linewidth=2, marker="o", markersize=4,
                    label=_str(c, "label"))  # fmt: skip
    for ax, title in zip(axes, ("forward push", "backward push"), strict=True):
        ax.axhline(0.5, color="#a0a0a0", linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("impulse (N·s)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("survival (seeds 0..99, preset clean)")
    axes[0].legend(fontsize=9)
    _save_png(fig, path)


def _save_png(fig: Figure, path: Path) -> None:
    with replacement(path, suffix=".partial.png") as partial:
        fig.savefig(partial, format="png", dpi=100, metadata={"Software": None})


# --- reward design and SAC vs PPO (§5.2 items 6 and 7) ------------------------------------------


def _row_json(row: HistoryRowModel, cpu_min: float | None = None) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {
        "step": row.step,
        "score": row.score,
        "success_rate": row.success_rate,
        "median_energy_j": row.median_energy_j,
        "mean_saturation": row.mean_saturation,
        "frac_mean_outside_1": row.frac_mean_outside_1,
    }
    if cpu_min is not None:
        out["cpu_min"] = cpu_min
    return out


def reward_design(report: Phase2Report) -> dict[str, JsonValue]:
    """x and y of §5.2 item 6: ``frac_mean_outside_1`` of the ablation and of the PPO seed-1
    run at their evals nearest 1M steps (the ablation's row also gives the failure mode)."""
    ablation = runs_of(report, "ablation")
    ppo = [s for s in runs_of(report, "ppo") if s.seed == REWARD_SEED]
    if len(ablation) != 1 or len(ppo) != 1:
        raise ValueError("reward design needs exactly one ablation run and one PPO seed-1 run")
    out: dict[str, JsonValue] = {}
    for key, s in (("ablation", ablation[0]), ("ppo", ppo[0])):
        row = nearest_row(s.selection_history, REWARD_STEP)
        if row.frac_mean_outside_1 is None:
            raise ValueError(f"the {key} run's eval nearest 1M steps has no frac_mean_outside_1")
        out[key] = {"run": s.run, **_row_json(row)}
    return out


def sac_vs_ppo(report: Phase2Report) -> list[JsonValue]:
    """§5.2 item 7, per seed pair (PPO and SAC runs paired in seed order): (a) PPO's eval
    nearest SAC's best step vs SAC's best, (b) PPO's eval nearest SAC's best's CPU-minutes vs
    SAC's best, (c) each run's best and final over its full budget."""
    ppo_runs, sac_runs = runs_of(report, "ppo"), runs_of(report, "sac")
    if not sac_runs or len(ppo_runs) < len(sac_runs):
        raise ValueError("SAC vs PPO needs a PPO run for every SAC run")
    out: list[JsonValue] = []
    for ppo, sac in zip(ppo_runs, sac_runs, strict=False):
        p_cpu, s_cpu = cpu_minutes(ppo), cpu_minutes(sac)
        p_rows, s_rows = ppo.selection_history, sac.selection_history
        s_best = s_rows.index(next(r for r in s_rows if r.step == sac.best.step))
        p_best = p_rows.index(next(r for r in p_rows if r.step == ppo.best.step))
        at_step = p_rows.index(nearest_row(p_rows, s_rows[s_best].step))
        at_cpu = min(range(len(p_rows)), key=lambda i: (abs(p_cpu[i] - s_cpu[s_best]), i))
        out.append(
            {
                "ppo_run": ppo.run,
                "sac_run": sac.run,
                "sac_best": _row_json(s_rows[s_best], s_cpu[s_best]),
                "equal_steps_ppo": _row_json(p_rows[at_step], p_cpu[at_step]),
                "equal_cpu_ppo": _row_json(p_rows[at_cpu], p_cpu[at_cpu]),
                "ppo_best": _row_json(p_rows[p_best], p_cpu[p_best]),
                "ppo_final": _row_json(p_rows[-1], p_cpu[-1]),
                "sac_final": _row_json(s_rows[-1], s_cpu[-1]),
            }
        )
    return out


# --- episodes -----------------------------------------------------------------------------------


def comparison_protocol(cmp: Comparison) -> EvalProtocol:
    """The protocol a fresh comparison ran, rebuilt from its id and seeds and checked against
    its ``protocol_hash``."""
    proto = protocol(cmp.protocol, cmp.seeds)
    if proto.protocol_hash() != cmp.protocol_hash:
        raise ValueError(f"{cmp.protocol}: the comparison's protocol_hash is not the current one")
    return proto


def rerun(
    sub: Path, proto: EvalProtocol, indices: Sequence[int], work: Path, name: str
) -> dict[int, tuple[Trajectory, float]]:
    """Episodes ``indices`` of ``proto`` for ``sub``, run again with the real harness on one env
    (as a pool task runs them), each saved to ``work`` and read back with ``load_npz``: index ->
    (trajectory, score)."""
    from topple.envs import BalanceEnv

    loaded = load_submission(sub, challenge=proto.challenge)
    env = BalanceEnv(proto.preset, challenge=proto.challenge, record=True)
    out: dict[int, tuple[Trajectory, float]] = {}
    try:
        for i in sorted(set(indices)):
            run = run_episode(env, loaded.factory, proto.seeds[i], i, salt=proto.salt)
            path = work / f"{name}_{proto.seeds[i]}.npz"
            run.trajectory.save_npz(path)
            out[i] = (load_npz(path, max_steps=protocol_max_steps(proto)), float(run.result.score))
    finally:
        env.close()
    return out


def episodes(
    sub: Path, cmp: Comparison, entry: Entry, indices: Sequence[int], work: Path, name: str
) -> dict[int, Trajectory]:
    """``rerun`` on ``cmp``'s protocol; each score must equal ``entry``'s stored one exactly."""
    runs = rerun(sub, comparison_protocol(cmp), indices, work, name)
    for i, (_, score) in runs.items():
        if score != entry.scores[i]:
            raise RuntimeError(
                f"{name} seed {cmp.seeds[i]} re-ran to {score!r}, not the stored "
                f"{entry.scores[i]!r}: render on the fresh comparisons' platform"
            )
    return {i: traj for i, (traj, _) in runs.items()}


def episode_facts(traj: Trajectory) -> dict[str, JsonValue]:
    """A gallery caption's facts: termination reason, end time, ``fall_cause`` if recorded,
    and the last push that started before the end (impulse N·s, start s)."""
    meta = traj.meta
    control_dt, timestep = float(meta["control_dt"]), float(meta["timestep"])
    term = meta.get("termination")
    reason = term.get("reason") if isinstance(term, dict) else None
    error = meta.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    cause = meta.get("fall_cause", term.get("fall_cause") if isinstance(term, dict) else None)
    scenario = meta.get("scenario")
    pushes = scenario.get("pushes", []) if isinstance(scenario, dict) else []
    started = [p for p in pushes if isinstance(p, dict) and int(p["start_step"]) < traj.steps]
    last: JsonValue = None
    if started:
        p = max(started, key=lambda q: int(q["start_step"]))
        last = {
            "impulse_ns": float(p["magnitude_n"]) * int(p["n_substeps"]) * timestep,
            "time_s": int(p["start_step"]) * control_dt,
        }
    return {
        "termination": reason if isinstance(reason, str) else "unknown",
        "error": code if isinstance(code, str) else None,
        "time_s": traj.steps * control_dt,
        "fall_cause": cause if isinstance(cause, str) else None,
        "last_push": last,
    }


# --- media --------------------------------------------------------------------------------------


def _cap(name: str) -> int | None:
    return RESULT_FILES[name]


def _mp4(
    trajs: Sequence[Trajectory], labels: Sequence[str], name: str, out: Path, work: Path
) -> int:
    """Render into ``work``, then copy to ``out / name`` only within its cap."""
    staged = work / name
    frames = video.render_side_by_side(trajs, labels, staged, pane=HERO_PANE, fps=HERO_FPS)
    size, cap = staged.stat().st_size, _cap(name)
    if cap is not None and size > cap:
        raise ValueError(f"{name} is {size} bytes, over its {cap}-byte cap")
    with replacement(out / name, suffix=".partial.mp4") as partial:
        shutil.copyfile(staged, partial)
    return frames


def _gif(trajs: Sequence[Trajectory], labels: Sequence[str], name: str, out: Path,
         rungs: Sequence[video.GifRung]) -> dict[str, JsonValue]:  # fmt: skip
    cap = _cap(name)
    if cap is None:
        raise ValueError(f"{name} has no size cap")
    rung = video.render_gif_ladder(trajs, labels, out / name, rungs, max_bytes=cap)
    return {"width": rung.width, "fps": rung.fps}


def _check_inputs(report: Phase2Report, shipped: Path) -> None:
    """The fresh comparisons ran here (same platform, ``cpu_isa``, onnxruntime), PID and LQR are
    entries 0 and 1, and ``shipped`` holds the shipped policy."""
    here = run_versions()
    for cmp in (report.fresh.eval_v1, report.fresh.calibration):
        there = cmp.versions
        if (cmp.platform, there.cpu_isa, there.onnxruntime) != (
            platform_tag(),
            here.cpu_isa,
            here.onnxruntime,
        ):
            raise ValueError(
                f"the fresh comparisons ran on {cmp.platform} {there.cpu_isa} onnxruntime "
                f"{there.onnxruntime}; episodes re-run bit-exact only there"
            )
        for k, sub in enumerate((PID_SUBMISSION, LQR_SUBMISSION)):
            if cmp.entries[k].sha256 != submission_sha256(REPO / sub):
                raise ValueError(f"entry {k} of {cmp.protocol} is not {sub}")
    loaded = load_submission(shipped, challenge=comparison_protocol(report.fresh.eval_v1).challenge)
    if loaded.policy_sha256 != report.shipped.policy_sha256:
        raise ValueError(f"{shipped} is not the shipped policy")


def _pick_json(pick: SeedPick, ppo: Entry, other: Entry) -> dict[str, JsonValue]:
    k = pick.index
    return {
        "index": k,
        "seed": pick.seed,
        "gap": pick.gap,
        "rule": pick.rule,
        "ppo_score": ppo.scores[k],
        "other_score": other.scores[k],
        "ppo_termination": ppo.termination[k],
        "other_termination": other.termination[k],
    }


def _span(span: list[int] | None) -> JsonValue:
    return None if span is None else [*span]


def _total_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def make_media(
    report: Phase2Report, *, shipped: Path, out_dir: Path, work: Path, workers: int = 4
) -> dict[str, JsonValue]:
    """Render every ``docs/results/phase2/`` media file of ``RESULT_FILES`` except
    ``ci_ubuntu/*`` into ``out_dir`` and return the values ``phase2.md`` needs (merged by the
    caller into ``report.render``). ``shipped`` is the shipped ``topple.policy/1`` directory;
    ``work`` is a scratch directory for trajectories."""
    _baseline_lines(report)  # fail before any work when the caller did not provide them
    _check_inputs(report, shipped)
    ev, cal = report.fresh.eval_v1, report.fresh.calibration
    pid, lqr, ppo = ev.entries[0], ev.entries[1], shipped_entry(report, ev)
    ppo_cal = shipped_entry(report, cal)
    hero = hero_seed(ev, ppo, pid)
    med_pid, med_lqr = median_seed(ev, ppo, pid), median_seed(ev, ppo, lqr)
    worst = lowest_seeds(ppo_cal)
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    ppo_ev = episodes(shipped, ev, ppo, [hero.index, med_pid.index, med_lqr.index], work, "ppo")
    pid_ev = episodes(REPO / PID_SUBMISSION, ev, pid, [hero.index, med_pid.index], work, "pid")
    lqr_ev = episodes(REPO / LQR_SUBMISSION, ev, lqr, [med_lqr.index], work, "lqr")
    ppo_worst = episodes(shipped, cal, ppo_cal, worst, work, "ppo_cal")

    pair = [pid_ev[hero.index], ppo_ev[hero.index]]
    hero_json = _pick_json(hero, ppo, pid)
    hero_json["mp4_frames"] = _mp4(pair, [LABEL_PID, LABEL_PPO], "hero.mp4", out_dir, work)
    hero_json["gif_rung"] = _gif(pair, [LABEL_PID, LABEL_PPO], "hero.gif", out_dir,
                                 video.HERO_LADDER)  # fmt: skip
    _mp4([pid_ev[med_pid.index], ppo_ev[med_pid.index]], [LABEL_PID, LABEL_PPO],
         "median_pid_ppo.mp4", out_dir, work)  # fmt: skip
    _mp4([lqr_ev[med_lqr.index], ppo_ev[med_lqr.index]], [LABEL_LQR, LABEL_PPO],
         "median_lqr_ppo.mp4", out_dir, work)  # fmt: skip

    gallery: list[JsonValue] = []
    for n, k in enumerate(worst, start=1):
        name = f"gallery_{n}.gif"
        label = f"PPO seed {cal.seeds[k]}"
        rung = _gif([ppo_worst[k]], [label], name, out_dir, video.GALLERY_LADDER)
        facts = episode_facts(ppo_worst[k])
        gallery.append({"file": name, "index": k, "seed": cal.seeds[k],
                        "score": ppo_cal.scores[k], "gif_rung": rung, **facts})  # fmt: skip

    plot_learning_curves(report, out_dir / "learning_curves.png")
    subs = [REPO / PID_SUBMISSION, REPO / LQR_SUBMISSION, shipped]
    curves = run_shoves(subs, parse_impulses(DEFAULT_IMPULSES), SHOVE_SEEDS, SHOVE_SIGNS, workers)
    shove = shove_values(list(zip((LABEL_PID, LABEL_LQR, LABEL_PPO), curves, strict=True)))
    plot_shove(shove, out_dir / "shove.png")

    total = _total_bytes(out_dir)
    if total > MAX_RESULTS_TOTAL_BYTES:
        raise ValueError(f"{out_dir} holds {total} bytes, over {MAX_RESULTS_TOTAL_BYTES}")
    here = run_versions()
    values: dict[str, JsonValue] = {
        "media_platform": {
            "platform": platform_tag(),
            "cpu_isa": here.cpu_isa,
            "onnxruntime": here.onnxruntime,
        },
        "hero": hero_json,
        "median": {
            "pid_ppo": _pick_json(med_pid, ppo, pid),
            "lqr_ppo": _pick_json(med_lqr, ppo, lqr),
        },
        "gallery": gallery,
        "shove": shove,
        "learning": {
            "cpu_axis": CPU_AXIS,
            "env_steps_to_pid": {s.run: _span(steps_to_pid(s)) for s in runs_of(report, "ppo")},
        },
        "reward_design": reward_design(report),
        "sac_vs_ppo": sac_vs_ppo(report),
    }
    check_json_numbers(values)
    return values


# --- markdown -----------------------------------------------------------------------------------


def _get(obj: JsonValue | Mapping[str, JsonValue], *path: str | int) -> JsonValue:
    """``obj[path...]``; ``KeyError`` naming the path when a field is missing."""
    cur: JsonValue | Mapping[str, JsonValue] = obj
    for key in path:
        if isinstance(key, int) and isinstance(cur, list) and -len(cur) <= key < len(cur):
            cur = cur[key]
            continue
        if isinstance(key, str) and isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            raise KeyError("render field " + ".".join(str(p) for p in path) + " is missing")
    return cur  # type: ignore[return-value]


def _num(obj: JsonValue | Mapping[str, JsonValue], *path: str | int) -> float:
    value = _get(obj, *path)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("render field " + ".".join(map(str, path)) + " is not a finite number")
    return float(value)


def _opt(obj: JsonValue | Mapping[str, JsonValue], *path: str | int) -> float | None:
    """A number or ``null`` (the field itself must exist)."""
    return None if _get(obj, *path) is None else _num(obj, *path)


def _str(obj: JsonValue | Mapping[str, JsonValue], *path: str | int) -> str:
    value = _get(obj, *path)
    if not isinstance(value, str):
        raise ValueError("render field " + ".".join(map(str, path)) + " is not a string")
    return value


def _list(obj: JsonValue | Mapping[str, JsonValue], *path: str | int) -> list[JsonValue]:
    value = _get(obj, *path)
    if not isinstance(value, list):
        raise ValueError("render field " + ".".join(map(str, path)) + " is not a list")
    return value


def _ns(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}"


def _steps(n: float) -> str:
    return f"{round(n):,}"


@dataclass(frozen=True)
class _Row:
    label: str
    entry: Entry
    summary: RunSummary | None = None
    paired: bool = False


def _paired(cmp: Comparison, a: Entry, b: Entry) -> PairedRow:
    rows = [r for r in cmp.paired if (r.a, r.b) == (a.id, b.id)]
    if len(rows) != 1:
        raise ValueError(f"{cmp.protocol}: no paired row for {a.id} vs {b.id}")
    return rows[0]


def _ci(row: PairedRow) -> str:
    return f"{row.mean:+.2f} [{row.lo:+.2f}, {row.hi:+.2f}]"


def _cells(cmp: Comparison, row: _Row) -> list[str]:
    e, n = row.entry, len(row.entry.scores)
    mean, lo, hi = bootstrap_ci(e.scores)
    cells = [
        row.label,
        f"{mean:.2f} [{lo:.2f}, {hi:.2f}]",
        str(sum(t in FALL_REASONS for t in e.termination)),
        f"{sum(t == SURVIVED for t in e.termination) / n:.3f}",
        f"{sum(e.success) / n:.3f}",
        f"{median(e.energy_j):.2f}",
        f"{math.fsum(e.saturation_fraction) / n:.3f}",
        "" if row.summary is None else _steps(row.summary.best.step),
        "" if row.summary is None else f"{row.summary.cpu_s / 60:.1f}",
    ]
    if row.paired:
        cells += [_ci(_paired(cmp, e, cmp.entries[0])), _ci(_paired(cmp, e, cmp.entries[1]))]
    else:
        cells += ["", ""]
    return cells


HEADER: Final = (
    "| Controller | Score [95% CI] | Falls | Survival | Success | Median energy (J) "
    f"| Mean saturation | Env steps (best) | Train CPU-min | PPO {MINUS} PID [95% CI] "
    f"| PPO {MINUS} LQR [95% CI] |\n|{'---|' * 11}"
)


def _recipe(entries: Sequence[Entry]) -> list[str]:
    scores = [math.fsum(e.scores) / len(e.scores) for e in entries]
    success = [sum(e.success) / len(e.success) for e in entries]
    sd = (stdev(scores), stdev(success)) if len(entries) > 1 else (0.0, 0.0)
    return [
        f"PPO recipe ({len(entries)} seeds)",
        f"{math.fsum(scores) / len(scores):.2f} ± {sd[0]:.2f} (min {min(scores):.2f})",
        "",
        "",
        f"{math.fsum(success) / len(success):.3f} ± {sd[1]:.3f} (min {min(success):.3f})",
        *[""] * 6,
    ]


def _scoreboard(report: Phase2Report, cmp: Comparison) -> list[str]:
    runs = summaries(report)
    ship = shipped_row(report)
    rows = [_Row(LABEL_PID, cmp.entries[0]), _Row(LABEL_LQR, cmp.entries[1])]
    rows.append(_Row("PPO (shipped: best of 3 by selection seeds)", shipped_entry(report, cmp),
                     runs[ship.run][1], paired=True))  # fmt: skip
    ppo_entries: list[Entry] = []
    seeds: list[_Row] = []
    fresh = sorted(report.fresh.rows, key=lambda r: (runs[r.run][0], runs[r.run][1].seed))
    for fr in fresh:
        role, s = runs[fr.run]
        mark = " (below PID gate)" if fr.below_pid_gate else ""
        e = entry_named(cmp, fr.entry)
        if role == "ppo":
            ppo_entries.append(e)
            seeds.append(_Row(f"PPO seed {s.seed} ({s.run}){mark}", e, s, paired=True))
        elif role == "sac":
            budget = f"{s.total_timesteps / 1e6:.1f}M"
            seeds.append(_Row(f"SAC ({budget} steps, UTD 0.25) seed {s.seed}{mark}", e, s))
    lines = [HEADER]
    lines += ["| " + " | ".join(_cells(cmp, r)) + " |" for r in rows]
    if ppo_entries:
        lines.append("| " + " | ".join(_recipe(ppo_entries)) + " |")
    lines += ["| " + " | ".join(_cells(cmp, r)) + " |" for r in seeds]
    return lines


def _ubuntu_block(report: Phase2Report) -> list[str]:
    ci = report.render.get("ci")
    if ci is None:
        return [
            "> **Pending CI evidence.** The ubuntu block is filled from "
            "`docs/results/phase2/ci_ubuntu/` when `scripts/phase2_results.py --fetch-ci` "
            "regenerates this file after CI's push run."
        ]
    cmp = parse_comparison(json.dumps(_get(ci, "cmp_cal")).encode("utf-8"))
    ship = [e for e in cmp.entries if e.policy_sha256 == report.shipped.policy_sha256]
    if len(ship) != 1:
        raise ValueError("the ubuntu compare must hold the shipped policy exactly once")
    summary = summaries(report)[report.shipped.run][1]
    rows = [
        _Row(LABEL_PID, cmp.entries[0]),
        _Row(LABEL_LQR, cmp.entries[1]),
        _Row("PPO (shipped: best of 3 by selection seeds)", ship[0], summary, paired=True),
    ]
    head = (
        f"CI run {round(_num(ci, 'run_id'))}, policy {report.shipped.policy_sha256[:12]}; "
        f"{cmp.platform} {cmp.versions.cpu_isa}, onnxruntime {cmp.versions.onnxruntime}."
    )
    return [head, "", HEADER, *("| " + " | ".join(_cells(cmp, r)) + " |" for r in rows)]


def _hero_section(r: Mapping[str, JsonValue]) -> list[str]:
    caption = HERO_CAPTION if _str(r, "hero", "rule") == "pid_fell" else HERO_FALLBACK_CAPTION
    lines = [
        "## 1. Hero: PID vs PPO",
        "",
        f"![PID vs PPO on eval_v1 seed {round(_num(r, 'hero', 'seed'))}](phase2/hero.gif)",
        "",
        caption,
        "",
        f"eval_v1 seed {round(_num(r, 'hero', 'seed'))}: PID {_num(r, 'hero', 'other_score'):.2f}"
        f" ({_str(r, 'hero', 'other_termination')}), PPO {_num(r, 'hero', 'ppo_score'):.2f} "
        f"({_str(r, 'hero', 'ppo_termination')}), gap {_num(r, 'hero', 'gap'):+.2f}. "
        f"[MP4 960x480 at 50 fps](phase2/hero.mp4); the GIF is "
        f"{round(_num(r, 'hero', 'gif_rung', 'width'))} px wide at "
        f"{round(_num(r, 'hero', 'gif_rung', 'fps'))} fps.",
        "",
    ]
    medians = (("pid_ppo", "PID", "median_pid_ppo"), ("lqr_ppo", "LQR", "median_lqr_ppo"))
    for key, other, name in medians:
        lines.append(
            f"- Median-difference seed, {other} vs PPO: [eval_v1 seed "
            f"{round(_num(r, 'median', key, 'seed'))}](phase2/{name}.mp4), PPO {MINUS} {other} "
            f"{_num(r, 'median', key, 'gap'):+.2f}."
        )
    return lines


def _shove_section(r: Mapping[str, JsonValue]) -> list[str]:
    seeds = _list(r, "shove", "seeds")
    lines = [
        "## 3. Shove test",
        "",
        f"Preset `clean`, seeds {round(_num(seeds, 0))}..{round(_num(seeds, 1))}, one push at "
        "2.0 s (0.1 s pulse, no lateral part) of each impulse in "
        f"{_num(r, 'shove', 'impulses', 0):.3f}..{_num(r, 'shove', 'impulses', -1):.3f} N·s, "
        "forward and backward. i50: where survival first falls through 50% (N·s).",
        "",
        "![Survival vs impulse](phase2/shove.png)",
        "",
        "| Controller | i50 forward | i50 backward | min |",
        "|---|---|---|---|",
    ]
    for c in _list(r, "shove", "controllers"):
        lines.append(
            f"| {_str(c, 'label')} | {_ns(_opt(c, 'i50_forward'))} | "
            f"{_ns(_opt(c, 'i50_backward'))} | {_ns(_opt(c, 'i50_min'))} |"
        )
    return lines


def _learning_section(report: Phase2Report, r: Mapping[str, JsonValue]) -> list[str]:
    lines = [
        "## 4. Learning curves",
        "",
        "![Selection success and score vs env steps and CPU-minutes](phase2/learning_curves.png)",
        "",
        f"CPU axis: {_str(r, 'learning', 'cpu_axis')}.",
        "",
    ]
    for s in runs_of(report, "ppo"):
        span = _get(r, "learning", "env_steps_to_pid", s.run)
        if span is None:
            lines.append(f"- {s.run}: never reached PID level on the selection seeds.")
        else:
            lo, hi = _num(span, 0), _num(span, 1)
            lines.append(f"- {s.run}: env steps to PID level in [{_steps(lo)}, {_steps(hi)}].")
    walls = "; ".join(
        f"{s.run} {s.wall_clock_s / 60:.1f} min (load {s.load_avg_start[0]:.1f} to "
        f"{s.load_avg_end[0]:.1f})"
        for s in (*runs_of(report, "ppo"), *runs_of(report, "sac"))
    )
    return [
        *lines,
        "",
        f"Wall-clock, on a shared machine (1-minute load at start and end): {walls}.",
    ]


def _gallery_section(r: Mapping[str, JsonValue]) -> list[str]:
    lines = ["## 5. Failure gallery", "", "The shipped PPO's 3 lowest-scoring seeds in 0..199.", ""]
    for g in _list(r, "gallery"):
        reason = _str(g, "termination")
        err = _get(g, "error")
        cause = _get(g, "fall_cause")
        push = _get(g, "last_push")
        parts = [f"{reason}" + (f" (error {err})" if isinstance(err, str) else "")
                 + f" at {_num(g, 'time_s'):.2f} s"]  # fmt: skip
        if isinstance(cause, str):
            parts.append(f"fall cause {cause}")
        parts.append(
            "no push before the end"
            if push is None
            else f"last push {_num(push, 'impulse_ns'):.3f} N·s at {_num(push, 'time_s'):.2f} s"
        )
        seed = round(_num(g, "seed"))
        lines += [
            f"![PPO seed {seed}](phase2/{_str(g, 'file')})",
            "",
            f"Seed {seed}, score {_num(g, 'score'):.2f}: " + "; ".join(parts) + ". "
            f"GIF {round(_num(g, 'gif_rung', 'width'))} px at "
            f"{round(_num(g, 'gif_rung', 'fps'))} fps.",
            "",
        ]
    return lines


def _signed(value: float) -> str:
    """``value`` (``%g``) with a typographic minus."""
    return f"{value:g}".replace("-", MINUS)


def _pilot(pilot: Mapping[str, JsonValue], key: str) -> str:
    """A pilot constant's citation: "pilot measurement" and its ``source``."""
    source = _str(pilot, key, "source")
    return source if "pilot measurement" in source else f"pilot measurement; {source}"


def _reward_section(report: Phase2Report, r: Mapping[str, JsonValue]) -> list[str]:
    p = report.pilot
    abl, ppo = _get(r, "reward_design", "ablation"), _get(r, "reward_design", "ppo")
    worse = " and ".join(_signed(_num(w)) for w in _list(p, "log_std_init", "worse"))
    return [
        "## 6. Reward-design notes",
        "",
        f"- u² kept: the energy-cost variant reached {_num(p, 'energy_cost_variant', 'score'):.1f}"
        f" / {_num(p, 'energy_cost_variant', 'success'):.3f} with "
        f"{_num(p, 'energy_cost_variant', 'saturated_share'):.0%} of steps saturated "
        f"({_pilot(p, 'energy_cost_variant')}).",
        f"- log_std {_signed(_num(p, 'log_std_init', 'shipped'))} beat {worse} "
        f"({_pilot(p, 'log_std_init')}).",
        "- The action-excess penalty cut the share of steps with the pre-clip action mean outside "
        f"±1 from {_num(abl, 'frac_mean_outside_1'):.1%} to {_num(ppo, 'frac_mean_outside_1'):.1%}"
        f" at 1M steps (seed {REWARD_SEED}; {_str(abl, 'run')} at step {_steps(_num(abl, 'step'))}"
        f", {_str(ppo, 'run')} at step {_steps(_num(ppo, 'step'))}).",
        f"- The 1M-step failure mode without the penalty ({_str(abl, 'run')}): selection success "
        f"{_num(abl, 'success_rate'):.3f}, score {_num(abl, 'score'):.2f}, median energy "
        f"{_num(abl, 'median_energy_j'):.2f} J, mean saturation "
        f"{_num(abl, 'mean_saturation'):.3f}.",
        f"- With a constant learning rate, {round(_num(p, 'constant_lr', 'seeds_regressed'))} of "
        f"{round(_num(p, 'constant_lr', 'of'))} seeds regressed after "
        f"{_num(p, 'constant_lr', 'after_steps') / 1e6:.1f}M steps "
        f"({_pilot(p, 'constant_lr')}).",
    ]


def _sac_section(r: Mapping[str, JsonValue]) -> list[str]:
    def cell(row: JsonValue) -> str:
        return (
            f"{_num(row, 'score'):.2f} / {_num(row, 'success_rate'):.3f} at "
            f"{_steps(_num(row, 'step'))} steps, {_num(row, 'cpu_min'):.1f} CPU-min"
        )

    lines = [
        "## 7. SAC vs PPO",
        "",
        'SAC is "learner-throughput-limited (UTD 0.25, serial updates)". UTD 0.25 was chosen for '
        "the shared-machine budget (SAC's default UTD 1 measured 2.6 to 4.2x slower); "
        f"target_entropy {MINUS}4 and UTD 1 were tested and worse. Selection score / success "
        "from each run's logged evals:",
        "",
        "| Pair | (a) equal env steps: SAC best | PPO | (b) equal CPU: PPO | (c) full budget: "
        f"SAC best | SAC final {MINUS} best | PPO best | PPO final {MINUS} best |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in _list(r, "sac_vs_ppo"):
        sac_drop = _num(p, "sac_final", "score") - _num(p, "sac_best", "score")
        ppo_drop = _num(p, "ppo_final", "score") - _num(p, "ppo_best", "score")
        lines.append(
            f"| {_str(p, 'sac_run')} / {_str(p, 'ppo_run')} | {cell(_get(p, 'sac_best'))} | "
            f"{cell(_get(p, 'equal_steps_ppo'))} | {cell(_get(p, 'equal_cpu_ppo'))} | "
            f"peak at {_steps(_num(p, 'sac_best', 'step'))} | {sac_drop:+.2f} | "
            f"{cell(_get(p, 'ppo_best'))} | {ppo_drop:+.2f} |"
        )
    return lines


def _readme_section(report: Phase2Report) -> list[str]:
    cal = report.fresh.calibration
    runs = summaries(report)
    ppo = [entry_named(cal, fr.entry) for fr in report.fresh.rows if runs[fr.run][0] == "ppo"]
    sac = [entry_named(cal, fr.entry) for fr in report.fresh.rows if runs[fr.run][0] == "sac"]
    if not ppo or not sac:
        raise ValueError("the README table needs PPO and SAC fresh rows")

    def score(e: Entry) -> float:
        return math.fsum(e.scores) / len(e.scores)

    def success(e: Entry) -> float:
        return sum(e.success) / len(e.success)

    ship = shipped_entry(report, cal)
    rows = [
        (LABEL_PID, score(cal.entries[0]), success(cal.entries[0])),
        (LABEL_LQR, score(cal.entries[1]), success(cal.entries[1])),
        (LABEL_PPO, score(ship), success(ship)),
        (f"SAC (mean of {len(sac)} seeds)", math.fsum(map(score, sac)) / len(sac),
         math.fsum(map(success, sac)) / len(sac)),
    ]  # fmt: skip
    return [
        "## 8. README snippet",
        "",
        "```markdown",
        "![PID vs PPO](docs/results/phase2/hero.gif)",
        "",
        "| Controller | Score (seeds 0..199) | Success |",
        "|---|---|---|",
        *(f"| {name} | {s:.2f} | {k:.3f} |" for name, s, k in rows),
        "",
        f"PPO recipe over {len(ppo)} seeds: mean score {math.fsum(map(score, ppo)) / len(ppo):.2f}"
        f", min {min(map(score, ppo)):.2f}; mean success "
        f"{math.fsum(map(success, ppo)) / len(ppo):.3f}, min {min(map(success, ppo)):.3f}.",
        "```",
    ]


def render_markdown(report: Phase2Report) -> str:
    """``phase2.md``, every number filled from ``report`` (``report.render`` included).
    ``KeyError`` or ``ValueError`` on a missing or ill-typed template field."""
    r = report.render
    cal = report.fresh.calibration
    g11 = report.g11
    k = sum(row.eligible and row.g2 and row.g3 and row.g4 for row in g11)
    j = sum(row.g2b for row in g11)
    ci = r.get("ci")
    ci_note = "pending" if ci is None else f"{round(_num(ci, 'run_id'))}"
    lines = [
        "# Phase 2a results: learning to balance",
        "",
        "Generated by `scripts/phase2_results.py` from `docs/results/phase2.json`; do not edit.",
        "",
        *_hero_section(r),
        "",
        "## 2. Scoreboard",
        "",
        f"{k} of {len(g11)} PPO seeds reached the PID gate (G11: eligible, G2, G3 and G4 on "
        f"0..199); {j} of {len(g11)} at LQR level (G2b).",
        "",
        "### eval_v1",
        "",
        *_scoreboard(report, report.fresh.eval_v1),
        "",
        "### Seeds 0..199 (Mac)",
        "",
        *_scoreboard(report, cal),
        "",
        f"Measured on {cal.platform} {cal.versions.cpu_isa}, onnxruntime "
        f"{cal.versions.onnxruntime}. CI's ubuntu numbers differ per episode (§3.5) and are shown "
        "in the second block, from `docs/results/phase2/ci_ubuntu/` (CI run "
        f"{ci_note}, policy {report.shipped.policy_sha256[:12]}).",
        "",
        "### Seeds 0..199 (ubuntu CI)",
        "",
        *_ubuntu_block(report),
        "",
        *_shove_section(r),
        "",
        *_learning_section(report, r),
        "",
        *_gallery_section(r),
        *_reward_section(report, r),
        "",
        *_sac_section(r),
        "",
        *_readme_section(report),
        "",
    ]
    text = "\n".join(lines)
    if _UNFILLED.search(text):
        raise ValueError("phase2.md has an unfilled template field")
    return text
