"""``phase2b.md`` from ``phase2b.json`` (docs/design/phase-2.md §8.9 item 5): every number is
templated from the report, nothing is typed in by hand. Run ids are the report's own (validated
``RunName`` values), so they need no escaping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from phase2b_common import DRIVER_ID, Phase2bReport, RatesModel
from topple.waypoint import SCORE_PARTS

VIDEO: Final = "phase2/waypoint.gif"
MINUS: Final = "\u2212"
TIMES: Final = "\u00d7"
ANCHOR_ORDER: Final = ("waypoint_pid_example", "creeper", "gates_only", "reckless",
                       "balance_in_place", "spin")  # fmt: skip


def _steps(n: float | int | None) -> str:
    return "never" if n is None else f"{n / 1e6:.2f}M"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _falls(r: RatesModel) -> str:
    split = ", ".join(f"{k} {v}" for k, v in r.falls_by_cause.items())
    return f"{_pct(r.fell)} ({split})" if split else _pct(r.fell)


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


def _headline(rep: Phase2bReport) -> list[str]:
    s, f = rep.headline["scratch"], rep.headline["finetune"]
    rows = [
        ["From scratch", ", ".join(_steps(x) for x in s["first_eligible_steps"]),
         _steps(s["median"]), "—"],
        ["Fine-tune", ", ".join(_steps(x) for x in f["first_eligible_steps"]),
         _steps(f["median"]), _steps(f["median_with_pretraining"])],
    ]  # fmt: skip
    return [
        "## Headline: steps to the first eligible checkpoint",
        "",
        "Env steps until a selection eval first passed the `waypoint` gate (seeds 0, 1, 2). "
        "The fine-tune arm also pays for its balance pretraining: "
        + ", ".join(_steps(x) for x in f["pretraining_steps"])
        + " steps.",
        "",
        *_table(["Arm", "First eligible", "Median", "Median incl. pretraining"], rows),
        "",
    ]


def _fresh(rep: Phase2bReport) -> list[str]:
    rows = []
    for r in rep.fresh:
        st, i6 = r.standard, r.item6
        p = i6.paired_ppo_driver
        rows.append(
            [
                f"`{r.run}`",
                r.arm,
                f"{st.score:.2f}",
                _pct(st.docked),
                _falls(st),
                f"{p.mean:+.2f} [{p.lo:+.2f}, {p.hi:+.2f}]",
                "pass" if i6.passed else "FAIL",
            ]
        )
    base = rep.baselines["standard"]
    for name in (DRIVER_ID, "creeper"):
        b = base[name]
        rows.append([f"`{name}`", "baseline", f"{b.score:.2f}", _pct(b.docked), _falls(b), "", ""])
    a = rep.arms
    lines = [
        "## Fresh evaluation: `waypoint-v1/standard` seeds 0..399",
        "",
        f"Item 6 (docked ≥ 60%, paired PPO {MINUS} driver lower bound > {MINUS}5, above the "
        f"creeper, fell ≤ driver + 20 pp): **{rep.item6_passes} of {len(rep.fresh)} runs pass.**",
        "",
        *_table(
            [
                "Run",
                "Arm",
                "Score",
                "Docked",
                "Fell (by cause)",
                f"PPO {MINUS} driver [95% CI]",
                "Item 6",
            ],
            rows,
        ),
        "",
    ]
    for arm, label in (("scratch", "From scratch"), ("finetune", "Fine-tune")):
        x = a[arm]
        lines.append(
            f"- {label}: score {x['score_mean']:.2f} ± {x['score_sd']:.2f}, fell "
            f"{_pct(x['fell_mean'])} ± {_pct(x['fell_sd'])}; median energy "
            f"{x['energy_vs_driver']:.2f}{TIMES} the driver's."
        )
    lines += [f"- Better arm (higher mean fresh score): {a['better_arm']}.", ""]
    return lines


def _other(
    rep: Phase2bReport, proto: Literal["clean", "wp_eval_v1"], title: str, note: str
) -> list[str]:
    rows = [[f"`{r.run}`", f"{getattr(r, proto).score:.2f}", _pct(getattr(r, proto).docked),
             _falls(getattr(r, proto))] for r in rep.fresh]  # fmt: skip
    rows += [[f"`{k}`", f"{b.score:.2f}", _pct(b.docked), _falls(b)]
             for k, b in rep.baselines[proto].items()]  # fmt: skip
    return [f"## {title}", "", note, "", *_table(["Entry", "Score", "Docked", "Fell"], rows), ""]


def _parts(rep: Phase2bReport) -> list[str]:
    base = rep.baselines["standard"]
    shipped = next((r for r in rep.fresh if r.run == rep.shipped.run), None)
    entries: list[tuple[str, RatesModel]] = [(k, base[k]) for k in ANCHOR_ORDER if k in base]
    if shipped is not None:
        entries.insert(1, (f"{shipped.run} (shipped)", shipped.standard))
    rows = [[f"`{k}`", f"{r.score:.2f}", *(f"{r.parts[p]:.2f}" for p in SCORE_PARTS),
             _pct(r.docked), _pct(r.fell), f"{r.median_t_finish_s:.2f}"]
            for k, r in entries]  # fmt: skip
    head = ["Entry", "Score", "Course /55", "Speed /25", "Alive /15", "Battery /5", "Docked",
            "Fell", "Median t_finish (s)"]  # fmt: skip
    lines = ["## Anchors and PPO: score parts on standard 0..399", "", *_table(head, rows), ""]
    if rep.exchange_rate is not None:
        d = rep.exchange_rate
        lines += [
            f"Exchange rate: a fall costs about {d:.1f} points (the shipped policy's mean score "
            f"of docked survivors minus fallers), so each extra 1% of falls costs about "
            f"{d / 100:.2f}; finishing 1 s faster on a docked run earns 25/(20 {MINUS} 5) = 1.67.",
            "",
        ]
    return lines


def _selection(rep: Phase2bReport) -> list[str]:
    lines = ["## Selection evals (seeds 2000..2199)", ""]
    for row in rep.runs:
        s = row.summary
        rows = []
        for h in s.selection_history:
            ex: Mapping[str, Any] = h.extras
            rows.append([f"{h.step / 1e6:.2f}M", f"{h.score:.2f}", f"{ex['mean_gates']:.2f}",
                         _pct(ex["docked"]), _pct(ex["fell"]), f"{ex['median_t_finish_s']:.2f}",
                         "yes" if h.eligible else "no"])  # fmt: skip
        lines += [f"### `{s.run}` ({row.arm}, best {s.best.step / 1e6:.2f}M)", "",
                  *_table(["Step", "Score", "Gates", "Docked", "Fell", "Median t_finish",
                           "Eligible"], rows), ""]  # fmt: skip
    return lines


def _shipping(rep: Phase2bReport) -> list[str]:
    sh = rep.shipped
    lines = ["## Shipping (§8.8)", ""]
    for k, a in enumerate(sh.attempts, 1):
        verdict = "ships" if not a.failed else "fails " + ", ".join(a.failed)
        lines.append(f"{k}. `{a.run}`: {verdict}")
    lines.append("")
    if sh.run is not None:
        lines += [f"`submissions/ppo_waypoint` = `{sh.run}` ({sh.arm}), step {sh.step:,}, "
                  f"onnx sha256 `{sh.onnx_sha256}`.", ""]  # fmt: skip
    return lines


def _sanity(rep: Phase2bReport) -> list[str]:
    lines = ["## Fine-tune sanity (item 7)", "",
             "Each fine-tune source after `extend_obs`, on `balance-v1/clean` seeds 0..19 with "
             "5 zero features appended and +100 rad on wheel_pos (features 4, 5); full-10 s "
             "survival:", ""]  # fmt: skip
    for s in rep.sanity:
        lines.append(f"- `{s.run}` from `{s.source['run']}` step {s.source['step']:,}: "
                     f"{_pct(s.survival)}")  # fmt: skip
    return [*lines, ""]


def _pilot(rep: Phase2bReport) -> list[str]:
    p = rep.pilot
    b = p["budget_rule"]["paired"]
    return [
        "## Pilot and budget",
        "",
        f"Pilot `ppo_waypoint.yaml` seed 0 at `{p['git_commit'][:7]}`: {p['num_timesteps']:,} "
        f"steps, cpu {p['cpu_s']:.0f} s, so waypoint `max_cpu_s` = {p['max_cpu_s']}. Budget "
        f"rule: late {MINUS} mid {b['mean']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}], so "
        "total_timesteps "
        + ("stays 6M." if p["budget_rule"]["total_timesteps"] is None else "rises to 8M."),
        "",
    ]


def render_markdown(rep: Phase2bReport) -> str:
    sh = rep.shipped
    lines = [
        "# Phase 2b results: `waypoint-v1`",
        "",
        (f"Shipped `{sh.run}` ({sh.arm}) as `submissions/ppo_waypoint`. "
         if sh.run else "No candidate passed; nothing shipped. ")
        + f"E_ref {rep.e_ref_j} J.",
        "",
        f"![driver and ppo_waypoint on one course]({VIDEO})",
        "",
        *_headline(rep),
        *_fresh(rep),
        *_parts(rep),
        *_other(rep, "clean", "Report only: `waypoint-v1/clean` seeds 0..399",
                "No pushes: every fall here is self-caused."),
        *_other(rep, "wp_eval_v1", "`wp_eval_v1` (50 public seeds)", "The leaderboard protocol."),
        *_shipping(rep),
        *_sanity(rep),
        *_pilot(rep),
        *_selection(rep),
    ]  # fmt: skip
    return "\n".join(lines).rstrip() + "\n"
