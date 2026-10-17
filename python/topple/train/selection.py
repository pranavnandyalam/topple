"""Checkpoint selection, eval schedule, parity rows and run flags (docs/design/phase-2.md §2.4,
§2.5). Torch-free: the SB3 callbacks (``callbacks.py``, the ``train`` extra) only glue these
pure pieces together.

- **Selection eval** (``evaluate_policy_bytes``): a checkpoint's in-memory ONNX bytes, as a
  ``PolicyRef``, on the selection seeds through the real harness in an evaluation pool; it
  returns a ``SelectionResult`` and the on-policy parity rows (a) the workers sent back.
- **Gate** (``measure_gate_baseline``, ``is_eligible``): ``pid`` compares with PID measured once
  at train start on the same seeds (success at least PID's, median energy at most E_ref, mean
  saturation at most ``selection.max_saturation``); ``waypoint`` (§8.8) compares with the
  reference driver and the creeper (score above the creeper's, docked >= 0.65, falls at most
  the driver's + 0.20, paired bootstrap lower bound vs the driver above -5); ``none`` makes
  every checkpoint eligible.
- **Rank** (``rank_key``), **best** (``choose_best``, ``write_best_json``): the top eligible
  checkpoint by rank key, or the top one while none is eligible; ties go to the earlier step.
- **Schedule** (``EvalTrigger``, ``eval_labels``, ``final_timesteps``): exactly SB3's step
  counts at which the callbacks evaluate, for PPO (rollout starts) and SAC (steps).
- **Parity rows** (``random_rows``, ``edge_rows``, ``parity_obs``): sets (b) and (c), and the
  ``set`` labels of ``parity_ref.npz``.
- **Flags** (``run_flags``): ``std_collapse``, ``late_regression``, ``high_energy``,
  ``no_eligible``; the abort flags and ``nondeterministic`` are set by training itself.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import numpy.typing as npt

from topple.evaluate import FALL_REASONS, EvalProtocol
from topple.policy.manifest import policy_task
from topple.pool import ONPOLICY_ROWS, EpisodeRows, PolicyRef, policy_ref, submit_episodes_with_rows
from topple.results import summarize
from topple.stats import paired_bootstrap

if TYPE_CHECKING:
    from topple.policy.build import ObsNorm
    from topple.results import EpisodeResult
    from topple.train.config import (
        HealthConfig,
        PpoTrainConfig,
        SacTrainConfig,
        SelectionConfig,
    )

GateRule = Literal["pid", "waypoint", "none"]
Kind = Literal["balance", "waypoint"]
CHUNK_EPISODES: Final = 25
"""Episodes per pool task (as ``topple.compare``); results never depend on it."""
N_RANDOM_ROWS: Final = 1000
RANDOM_SPREAD: Final = 12.0
"""Random rows (b) are ``mean + sqrt(var + eps) * U(-12, 12)``."""
EDGE_SIGMAS: Final = (10.0, 10.001)
"""Edge rows (c) put one feature at ``mean ± 10 std`` (the normalization clip) and just beyond."""
EDGE_VALUE: Final = 1e6
SET_ONPOLICY: Final = 0
SET_RANDOM: Final = 1
SET_EDGE: Final = 2
ABORT_FLAGS: Final = ("rss_abort", "cpu_abort", "wall_clock_abort")
NONDETERMINISTIC: Final = "nondeterministic"
NOT_SHIPPABLE: Final = frozenset({*ABORT_FLAGS, NONDETERMINISTIC})
"""A run with any of these flags is never shipped or reported as a result (§1)."""
FLAGS: Final = (
    "std_collapse",
    "late_regression",
    "high_energy",
    "no_eligible",
    *ABORT_FLAGS,
    NONDETERMINISTIC,
)
"""Every flag, in the order ``train_meta.flags`` lists them (a tuple: never a set's order, which
follows the string hash seed and so changes from process to process)."""
FLOAT_SLACK: Final = 1e-9
"""Success rates are k/n: a drop of 0.85 - 0.75 is 0.0999999... in floating point."""
WAYPOINT_DRIVER: Final = "waypoint_pid_example.py"
WAYPOINT_CREEPER: Final = "anchors/creeper.py"
WAYPOINT_MIN_DOCKED: Final = 0.65
"""§8.8: item 6's docked >= 0.60 with a 0.05 margin, like balance's saturation 0.08 vs 0.10."""
WAYPOINT_FALL_MARGIN: Final = 0.20
WAYPOINT_MIN_PAIRED_LO: Final = -5.0


# --- results --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionResult:
    """One selection eval of the checkpoint at env step ``step``: success rate, mean score
    (fsum/N, contract §12), median battery energy and mean per-episode saturation over the
    selection seeds, the per-seed scores in seed order (for paired comparisons) and ``extras``
    (balance: ``fell``, the fall rate; waypoint, 2b: mean_gates, docked, fell,
    median_t_finish_s)."""

    step: int
    success_rate: float
    score: float
    median_energy_j: float
    mean_saturation: float
    scores: tuple[float, ...]
    extras: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError(f"step must be an int >= 0, got {self.step!r}")
        object.__setattr__(self, "scores", tuple(float(s) for s in self.scores))
        object.__setattr__(self, "extras", {str(k): float(v) for k, v in self.extras.items()})
        numbers = (self.success_rate, self.score, self.median_energy_j, self.mean_saturation)
        if not all(math.isfinite(x) for x in (*numbers, *self.scores, *self.extras.values())):
            raise ValueError("a selection result is finite")

    def to_json(self) -> dict[str, object]:
        """Every field but ``scores`` (a ``selection_history`` row's result part, §2.3; the
        per-seed scores go to ``selection_scores.npz``)."""
        return {
            "step": self.step,
            "success_rate": self.success_rate,
            "score": self.score,
            "median_energy_j": self.median_energy_j,
            "mean_saturation": self.mean_saturation,
            "extras": dict(self.extras),
        }


def summarize_episodes(
    episodes: Sequence[EpisodeResult],
    *,
    step: int,
    kind: Kind = "balance",
    episode_s: float | None = None,
) -> SelectionResult:
    """The ``SelectionResult`` of ``episodes`` (one per selection seed, in seed order): score and
    success rate are the leaderboard's own aggregate (``results.summarize``, contract §12). An
    episode without metrics (``missing``) counts 0 energy, 0 saturation and no fall. A waypoint
    eval (``kind``) adds §8.8's extras, which need ``episode_s``."""
    n = len(episodes)
    if n == 0:
        raise ValueError("a selection eval has at least one episode")
    aggregate = summarize(episodes, n)
    energy = [_metric(e, "energy_j") for e in episodes]
    saturation = [_metric(e, "saturation_fraction") for e in episodes]
    fell = sum(
        e.metrics is not None and e.metrics.get("termination") in FALL_REASONS for e in episodes
    )
    return SelectionResult(
        step=step,
        success_rate=aggregate.success_rate,
        score=aggregate.score,
        median_energy_j=float(median(energy)),
        mean_saturation=math.fsum(saturation) / n,
        scores=tuple(float(e.score) for e in episodes),
        extras={
            "fell": fell / n,
            **(_waypoint_extras(episodes, episode_s) if kind == "waypoint" else {}),
        },
    )


def _waypoint_extras(
    episodes: Sequence[EpisodeResult], episode_s: float | None
) -> dict[str, float]:
    """§8.8: a waypoint eval also records mean gates, the docked rate and the median finish
    time (a non-finisher, and a missing episode, counts the whole episode, as ``t_finish_s``
    does)."""
    if episode_s is None:
        raise ValueError("a waypoint selection eval needs the episode length")
    n = len(episodes)
    gates = [0.0 if e.metrics is None else float(e.metrics["waypoints"]) for e in episodes]
    docked = sum(e.metrics is not None and e.metrics.get("docked") is True for e in episodes)
    finish = [episode_s if e.metrics is None else float(e.metrics["t_finish_s"]) for e in episodes]
    return {
        "mean_gates": math.fsum(gates) / n,
        "docked": docked / n,
        "median_t_finish_s": float(median(finish)),
    }


def _metric(e: EpisodeResult, name: str) -> float:
    return 0.0 if e.metrics is None else float(e.metrics[name])


def selection_protocol(sel: SelectionConfig) -> EvalProtocol:
    """``sel.protocol`` with the selection seeds in place of its own, built as ``topple run``
    builds its one-seed protocol (``dataclasses.replace``), so ``protocol_hash`` covers them
    (``selection_protocol_hash``, §2.3)."""
    return replace(EvalProtocol.from_id(sel.protocol), seeds=sel.seeds.seeds)


def evaluate_policy_bytes(
    ref: PolicyRef,
    protocol: EvalProtocol,
    pool: Executor,
    *,
    step: int = 0,
    baseline: bool = False,
) -> tuple[SelectionResult, npt.NDArray[np.float32]]:
    """Run ``ref`` (a checkpoint's in-memory ONNX bytes, or a Python baseline) on every seed of
    the public ``protocol`` in ``pool``: its ``SelectionResult`` at ``step`` and the on-policy
    parity rows (§2.4 (a)), ``ONPOLICY_ROWS`` per episode of at least one step, in seed order.
    A failing task cancels the rest and raises; the caller owns (and shuts down) ``pool``. A
    ``baseline`` (the gate's reference or anchor) raises ``RuntimeError`` if any episode ended
    in a controller error: scored 0, it would silently weaken the gate."""
    n = len(protocol.seeds)
    chunks = [range(start, min(start + CHUNK_EPISODES, n)) for start in range(0, n, CHUNK_EPISODES)]
    futures: list[Future[list[EpisodeRows]]] = []
    try:
        futures = [submit_episodes_with_rows(pool, ref, protocol, chunk) for chunk in chunks]
        episodes = list(itertools.chain.from_iterable(f.result() for f in futures))
    except BaseException:
        for f in futures:
            f.cancel()
        raise
    if [e.result.index for e in episodes] != list(range(n)):
        raise RuntimeError("the pool returned episodes out of seed order")
    errors = [e.result for e in episodes if e.result.error is not None]
    if baseline and errors:
        first = errors[0]
        raise RuntimeError(
            f"the gate baseline erred on {len(errors)} of {n} episodes "
            f"(seed {first.seed}: {first.error})"
        )
    width = policy_task(protocol.challenge).obs_dim
    rows = [e.obs_rows for e in episodes if e.obs_rows.shape[0]]
    onpolicy = np.concatenate(rows) if rows else np.empty((0, width), dtype=np.float32)
    episode_s = float(protocol.challenge.config.episode_seconds)
    result = summarize_episodes(
        [e.result for e in episodes],
        step=step,
        kind=protocol.challenge.kind,  # type: ignore[arg-type]
        episode_s=episode_s,
    )
    return result, onpolicy


def onpolicy_seed_count(onpolicy: npt.NDArray[np.float32]) -> int:
    """``n_onpolicy_seeds``: the episodes with T >= 1, each of which sent exactly
    ``ONPOLICY_ROWS`` rows."""
    if onpolicy.shape[0] % ONPOLICY_ROWS:
        raise ValueError(f"on-policy rows come in groups of {ONPOLICY_ROWS}")
    return int(onpolicy.shape[0]) // ONPOLICY_ROWS


# --- the gate -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateBaseline:
    """What the gate compares with, measured once at train start on the selection seeds with
    the same harness: ``pid`` -> ``reference`` is PID (``submissions/pid_example.py``);
    ``waypoint`` (2b) -> ``reference`` the reference driver and ``anchor`` the creeper;
    ``none`` -> nothing."""

    rule: GateRule
    reference: SelectionResult | None = None
    anchor: SelectionResult | None = None

    def __post_init__(self) -> None:
        if self.rule not in ("pid", "waypoint", "none"):
            raise ValueError(f"unknown gate {self.rule!r}")
        if self.rule != "none" and self.reference is None:
            raise ValueError(f"the {self.rule} gate needs its reference result")
        if (self.rule == "waypoint") != (self.anchor is not None):
            raise ValueError("the waypoint gate, and only it, has an anchor (the creeper)")

    def to_json(self) -> dict[str, object]:
        """``train_meta.gate_baseline``: the rule and each measured result with its per-seed
        scores and fall rate (§2.3)."""

        def full(r: SelectionResult | None) -> dict[str, object] | None:
            return None if r is None else {**r.to_json(), "scores": list(r.scores)}

        return {"rule": self.rule, "reference": full(self.reference), "anchor": full(self.anchor)}


def measure_gate_baseline(
    rule: GateRule, protocol: EvalProtocol, pool: Executor, *, pid_submission: Path
) -> GateBaseline:
    """The ``rule`` baseline on ``protocol`` (the selection seeds): for ``pid``, the PID
    submission at ``pid_submission`` loaded once here and run by reference in ``pool``."""
    if rule == "none":
        return GateBaseline("none")
    if rule == "waypoint":
        # §8.8: the reference driver and the creeper anchor, beside the PID example
        root = pid_submission.parent
        driver = policy_ref(root / WAYPOINT_DRIVER, challenge=protocol.challenge)
        creeper = policy_ref(root / WAYPOINT_CREEPER, challenge=protocol.challenge)
        reference, _ = evaluate_policy_bytes(driver, protocol, pool, baseline=True)
        anchor, _ = evaluate_policy_bytes(creeper, protocol, pool, baseline=True)
        return GateBaseline("waypoint", reference=reference, anchor=anchor)
    ref = policy_ref(pid_submission, challenge=protocol.challenge)
    reference, _ = evaluate_policy_bytes(ref, protocol, pool, baseline=True)
    return GateBaseline("pid", reference=reference)


def is_eligible(
    r: SelectionResult, base: GateBaseline, sel: SelectionConfig, e_ref_j: float
) -> bool:
    """§2.4 step 3. ``pid``: ``success_rate >= PID's`` and ``median_energy_j <= E_ref`` and
    ``mean_saturation <= sel.max_saturation``; ``none``: always; ``waypoint``: §8.8's four
    conditions (``_waypoint_eligible``)."""
    if base.rule != sel.gate:
        raise ValueError(f"the baseline is for the {base.rule} gate, the config's is {sel.gate}")
    if base.rule == "none":
        return True
    if base.rule == "waypoint":
        return _waypoint_eligible(r, base)
    reference = base.reference
    if reference is None or sel.max_saturation is None:  # GateBaseline and the config check
        raise ValueError("the pid gate needs PID's result and selection.max_saturation")
    return (
        r.success_rate >= reference.success_rate
        and r.median_energy_j <= e_ref_j
        and r.mean_saturation <= sel.max_saturation
    )


def _waypoint_eligible(r: SelectionResult, base: GateBaseline) -> bool:
    """§8.8: score above the creeper's, docked >= 0.65, falls at most the driver's + 0.20, and
    the paired bootstrap lower bound of (scores - driver's) above -5 (every item-6 sub-gate)."""
    driver, creeper = base.reference, base.anchor
    if driver is None or creeper is None:
        raise ValueError("the waypoint gate needs the driver's and the creeper's results")
    if len(r.scores) != len(driver.scores):
        raise ValueError("a waypoint eval and the driver's must cover the same seeds")
    _, lo, _ = paired_bootstrap(r.scores, driver.scores)
    return (
        r.score > creeper.score
        and r.extras["docked"] >= WAYPOINT_MIN_DOCKED
        and r.extras["fell"] <= driver.extras["fell"] + WAYPOINT_FALL_MARGIN + FLOAT_SLACK
        and lo > WAYPOINT_MIN_PAIRED_LO
    )


def rank_key(r: SelectionResult, kind: Kind) -> tuple[float, ...]:
    """§2.4 step 4, matching the leaderboard: balance ``(round(score, 2), success_rate)``;
    waypoint ``(round(score, 2), success_rate, -median_t_finish_s)``. Larger is better."""
    if kind == "balance":
        return (round(r.score, 2), r.success_rate)
    if kind == "waypoint":
        if "median_t_finish_s" not in r.extras:
            raise ValueError("a waypoint result has extras['median_t_finish_s']")
        return (round(r.score, 2), r.success_rate, -r.extras["median_t_finish_s"])
    raise ValueError(f"unknown kind {kind!r}")


# --- history, best and flags --------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryRow:
    """One ``selection_history`` row (§2.3): the eval's result (all fields but ``scores``),
    its eligibility, the policy stats (§2.5; ``None`` where they do not apply), the eval's
    CPU and wall seconds and the sha256 of the ONNX bytes it scored."""

    result: SelectionResult
    eligible: bool
    onnx_sha256: str
    std_mean: float | None = None
    frac_mean_outside_1: float | None = None
    excess_penalty_mean: float | None = None
    cpu_s: float | None = None
    wall_s: float | None = None

    @property
    def step(self) -> int:
        return self.result.step

    def to_json(self) -> dict[str, object]:
        return {
            **self.result.to_json(),
            "eligible": self.eligible,
            "std_mean": self.std_mean,
            "frac_mean_outside_1": self.frac_mean_outside_1,
            "excess_penalty_mean": self.excess_penalty_mean,
            "cpu_s": self.cpu_s,
            "wall_s": self.wall_s,
            "onnx_sha256": self.onnx_sha256,
        }


def choose_best(history: Sequence[HistoryRow], kind: Kind) -> HistoryRow | None:
    """§2.4 step 5: the eligible row with the largest ``rank_key``, or, while none is eligible,
    the row with the largest one; the earlier step on a tie; ``None`` for no rows."""
    _check_unique(history)
    candidates = [row for row in history if row.eligible] or list(history)
    if not candidates:
        return None
    return max(candidates, key=lambda row: (rank_key(row.result, kind), -row.step))


def best_json(row: HistoryRow, rule: GateRule) -> dict[str, object]:
    """``best.json``: ``{"step": N, "eligible": bool, "rule": "<gate>"}``."""
    return {"step": row.step, "eligible": row.eligible, "rule": rule}


def write_best_json(path: Path, row: HistoryRow, rule: GateRule) -> None:
    """Rewrite ``best.json`` atomically (a new file renamed over it, never written through a
    symlink at ``path``)."""
    from topple._atomic import replacement

    with replacement(path, suffix=".tmp") as tmp:
        tmp.write_text(json.dumps(best_json(row, rule), indent=2) + "\n", encoding="utf-8")


def std_collapse(history: Sequence[HistoryRow], health: HealthConfig) -> bool:
    """PPO: a policy std below ``health.min_std`` at an eval before
    ``health.min_std_before_step`` (a warning, §2.5)."""
    return any(
        row.std_mean is not None
        and row.step < health.min_std_before_step
        and row.std_mean < health.min_std
        for row in history
    )


def late_regression(history: Sequence[HistoryRow], health: HealthConfig) -> bool:
    """PPO: a selection success after ``health.regression_after_step`` that is at least
    ``health.regression_drop`` below the running best success of the evals before it
    (a warning, reported, §2.5)."""
    best = -math.inf
    for row in history:
        success = row.result.success_rate
        if (
            row.step > health.regression_after_step
            and best - success >= health.regression_drop - FLOAT_SLACK
        ):
            return True
        best = max(best, success)
    return False


def high_energy(history: Sequence[HistoryRow], e_ref_j: float) -> bool:
    """Any eval's median energy above E_ref."""
    return any(row.result.median_energy_j > e_ref_j for row in history)


def no_eligible(history: Sequence[HistoryRow]) -> bool:
    return not any(row.eligible for row in history)


def run_flags(
    history: Sequence[HistoryRow], *, health: HealthConfig | None, e_ref_j: float
) -> tuple[str, ...]:
    """The flags a finished run's history earns, in ``FLAGS`` order (the health flags only
    with a ``health`` block, i.e. PPO)."""
    _check_unique(history)
    raised = {
        "high_energy": high_energy(history, e_ref_j),
        "no_eligible": no_eligible(history),
    }
    if health is not None:
        raised["std_collapse"] = std_collapse(history, health)
        raised["late_regression"] = late_regression(history, health)
    return tuple(flag for flag in FLAGS if raised.get(flag, False))


def reproduces(new_rows: Sequence[object], old_rows: Sequence[object]) -> bool:
    """§1: a ``--retry`` re-run reproduces the aborted attempt when both ``selection_history``
    lists (rows as written to ``train_meta.json``) give the same ``onnx_sha256`` at every label
    both reached. A row without an integer step or a string sha never matches."""

    def by_step(rows: Sequence[object]) -> dict[int, object] | None:
        found: dict[int, object] = {}
        for row in rows:
            if not isinstance(row, Mapping) or type(row.get("step")) is not int:
                return None
            sha = row.get("onnx_sha256")
            if not isinstance(sha, str):
                return None
            found[row["step"]] = sha
        return found

    new, old = by_step(new_rows), by_step(old_rows)
    if new is None or old is None:
        return False
    return all(new[step] == old[step] for step in new.keys() & old.keys())


def _check_unique(history: Sequence[HistoryRow]) -> None:
    steps = [row.step for row in history]
    if any(b <= a for a, b in itertools.pairwise(steps)):
        raise ValueError("selection_history steps must be unique and increasing")


# --- the eval schedule --------------------------------------------------------------------------


@dataclass
class EvalTrigger:
    """§2.4's trigger, fed SB3's ``num_timesteps``: due when ``num_timesteps // every_steps``
    exceeds its last value. PPO calls it at each rollout start (``total_timesteps=None``); SAC
    at each step, where it is never due at or past ``total_timesteps`` (the final eval in
    ``_on_training_end`` covers that). ``last_label`` is the label of the last eval it allowed
    (or the resume point): ``final_due`` keeps the final eval from repeating it."""

    every_steps: int
    total_timesteps: int | None = None
    _bucket: int = field(default=0, repr=False)
    last_label: int | None = None

    def __post_init__(self) -> None:
        if type(self.every_steps) is not int or self.every_steps < 1:
            raise ValueError(f"every_steps must be an int >= 1, got {self.every_steps!r}")

    @classmethod
    def resumed(
        cls, every_steps: int, num_timesteps: int, total_timesteps: int | None = None
    ) -> EvalTrigger:
        """The trigger of a run resumed from its snapshot at ``num_timesteps`` (SB3 keeps the
        count): due at the next label of the schedule, never at the resume point, whose eval
        and snapshot exist already."""
        if type(num_timesteps) is not int or num_timesteps < 0:
            raise ValueError(f"num_timesteps must be an int >= 0, got {num_timesteps!r}")
        trigger = cls(every_steps, total_timesteps, last_label=num_timesteps)
        trigger._bucket = num_timesteps // every_steps
        return trigger

    def due(self, num_timesteps: int) -> bool:
        bucket = num_timesteps // self.every_steps
        if bucket <= self._bucket:
            return False
        if self.total_timesteps is not None and num_timesteps >= self.total_timesteps:
            return False
        self._bucket = bucket
        self.last_label = num_timesteps
        return True

    def final_due(self, num_timesteps: int) -> bool:
        """Whether ``_on_training_end`` evaluates (and snapshots) at ``num_timesteps``: always,
        unless that label was evaluated already. That happens only when a hard limit stopped
        training in the same callback call that ran a periodic eval (SAC's ``_on_step``), or
        right at a resume point; the eval there is then the final one (§1, §2.4)."""
        return num_timesteps != self.last_label


def final_timesteps(cfg: PpoTrainConfig | SacTrainConfig) -> int:
    """SB3's ``num_timesteps`` when ``learn(total_timesteps)`` returns: PPO collects whole
    rollouts of ``n_steps * n_envs`` (5M -> 5,013,504); SAC whole collections of ``train_freq``
    steps of ``n_envs`` (1M -> 1,000,000)."""
    per = cfg.ppo.n_steps if cfg.algo == "ppo" else cfg.sac.train_freq
    chunk = per * cfg.n_envs
    return -(-cfg.total_timesteps // chunk) * chunk


def eval_labels(cfg: PpoTrainConfig | SacTrainConfig) -> tuple[int, ...]:
    """Every selection eval's label (``num_timesteps``) in order: the periodic ones, then the
    final at ``final_timesteps(cfg)``. Unique and strictly increasing by construction; the same
    numbers ``EvalTrigger`` yields when fed SB3's step counts (computed arithmetically: at most
    ``total_timesteps // every_steps`` iterations).

    PPO: at the rollout starts ``k * R`` (``R = n_steps * n_envs``, k = 1 .. K - 1) where
    ``num_timesteps // every`` increases. SAC: at the step counts ``j * n_envs`` below
    ``total_timesteps`` where it increases."""
    every = cfg.selection.every_steps
    final = final_timesteps(cfg)
    if cfg.algo == "ppo":
        step = cfg.ppo.n_steps * cfg.n_envs
        last = final - step  # the last rollout starts here
    else:
        step = cfg.n_envs
        last = cfg.total_timesteps - 1  # strictly below total_timesteps
    labels: list[int] = []
    for m in range(1, last // every + 1):
        label = -(-m * every // step) * step  # the first count reaching m * every
        if label <= last and (not labels or label != labels[-1]):
            labels.append(label)
    return (*labels, final)


# --- parity rows ------------------------------------------------------------------------------


def _norm_arrays(norm: ObsNorm) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    mean = np.asarray(norm.mean, dtype=np.float64)
    var = np.asarray(norm.var, dtype=np.float64)
    if mean.ndim != 1 or mean.shape != var.shape or mean.size == 0:
        raise ValueError("mean and var must be 1-D arrays of one length")
    if not (np.isfinite(mean).all() and np.isfinite(var).all() and (var >= 0).all()):
        raise ValueError("mean and var must be finite, var non-negative")
    if not (math.isfinite(norm.epsilon) and norm.epsilon > 0):
        raise ValueError("epsilon must be finite and positive")
    return mean, np.sqrt(var + norm.epsilon)


def random_rows(norm: ObsNorm, n: int = N_RANDOM_ROWS) -> npt.NDArray[np.float32]:
    """Parity rows (b): ``f32(mean + sqrt(var + eps) * U(-12, 12))`` from
    ``default_rng(SeedSequence(0))``. ``U`` is drawn as ``-12 + 24 * random()`` with numpy's
    separately rounded ``*`` and ``+`` (never ``Generator.uniform``, whose last bit differs
    between platforms, contract §9), so the rows are the same everywhere."""
    mean, std = _norm_arrays(norm)
    u = np.random.default_rng(np.random.SeedSequence(0)).random((n, mean.size))
    k = -RANDOM_SPREAD + (2.0 * RANDOM_SPREAD) * u
    rows: npt.NDArray[np.float32] = (mean + std * k).astype(np.float32)
    return rows


def edge_rows(norm: ObsNorm) -> npt.NDArray[np.float32]:
    """Parity rows (c), ``6 * D + 4``: for each feature j, ``mean`` with ``x_j`` at ``mean_j +
    10 std_j``, ``mean_j - 10 std_j``, ``mean_j + 10.001 std_j``, ``mean_j - 10.001 std_j``,
    ``+1e6``, ``-1e6`` (std = sqrt(var + eps)); then ``mean``, zeros, all ``+1e6``, all ``-1e6``.
    Computed in float64 and rounded to float32 once."""
    mean, std = _norm_arrays(norm)
    d = mean.size
    rows: list[npt.NDArray[np.float64]] = []
    for j in range(d):
        values = [mean[j] + s * k * std[j] for k in EDGE_SIGMAS for s in (1.0, -1.0)]
        for value in (*values, EDGE_VALUE, -EDGE_VALUE):
            row = mean.copy()
            row[j] = value
            rows.append(row)
    rows += [mean.copy(), np.zeros(d), np.full(d, EDGE_VALUE), np.full(d, -EDGE_VALUE)]
    out: npt.NDArray[np.float32] = np.asarray(rows, dtype=np.float64).astype(np.float32)
    return out


def edge_row_count(obs_dim: int) -> int:
    return 6 * obs_dim + 4


def parity_obs(
    onpolicy: npt.NDArray[np.float32], norm: ObsNorm
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """``parity_ref.npz``'s ``obs`` and ``set``: the on-policy rows (a), then the random rows
    (b), then the edge rows (c), labelled 0, 1 and 2."""
    b = random_rows(norm)
    c = edge_rows(norm)
    if onpolicy.dtype != np.float32 or onpolicy.ndim != 2 or onpolicy.shape[1] != b.shape[1]:
        raise ValueError(f"on-policy rows must be float32 [N, {b.shape[1]}]")
    obs = np.concatenate([onpolicy, b, c]).astype(np.float32)
    labels = np.concatenate(
        [
            np.full(onpolicy.shape[0], SET_ONPOLICY, np.uint8),
            np.full(b.shape[0], SET_RANDOM, np.uint8),
            np.full(c.shape[0], SET_EDGE, np.uint8),
        ]
    )
    return obs, labels
