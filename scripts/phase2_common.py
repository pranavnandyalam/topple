"""What the Phase 2a scripts share (docs/design/phase-2.md §2.5, §4, §5.1).

- **Gate names.** ``SHIP_GATES`` (§2.5): the gates a shipping candidate must pass on its own
  fresh evaluations before it is shipped, used by ``phase2_results.py`` (``choose_shipped``),
  ``phase2_gate.py`` and ``test_phase2_gate``. ``PASS_GATES`` (§4): every gate
  ``phase2_gate.py`` evaluates; G9 is ``phase2_ci_check.py``.
- **Protocols.** The calibration (balance-v1/standard, seeds 0..199), selection (seeds
  2000..2199) and eval_v1 protocols, built as ``topple run`` builds a protocol with other seeds
  (``dataclasses.replace`` of ``EvalProtocol.from_id``), so ``protocol_hash`` covers the seeds.
- **``phase2.json``** (``topple.phase2/1``, written by ``phase2_results.py``): the reported runs,
  each a role and its committed per-run summary of **exactly** the §5.1 fields (``RunSummary``);
  the fresh evaluations (two ``topple.compare/1`` objects with PID, LQR and every exported
  policy, and one row per run binding it to its entry); the shipped run with every attempt; the
  G11 table; the pilot constants; and the values only the report renders. It is read like every
  gate input, as contract §15 reads ``results.json``: a regular file of at most 16 MiB
  (``open_regular``, never blocking), exact JSON types, ``format`` required, no NaN or Infinity
  (also in the free-form objects, whose integers must be exact in a double: |n| < 2**53).
- **The results package's files** (``RESULT_FILES``) and their §5.2 size caps (G8).
- **``run_command``**: a subprocess with a timeout and a cap on what it may print (git, gh).

Torch-free: nothing here imports torch, stable_baselines3 or psutil.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import IO, Annotated, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    model_validator,
)

from topple._inputs import open_regular
from topple.compare import Comparison
from topple.evaluate import EvalProtocol
from topple.results import MAX_RESULTS_BYTES, Identifier, Sha256, VersionStr
from topple.rng import MAX_SEED
from topple.train.selection import NOT_SHIPPABLE, SelectionResult, rank_key

# --- gates ----------------------------------------------------------------------------------------

SHIP_GATES: Final = ("G1", "G2", "G2b", "G3", "G4", "G5", "G6")
"""§2.5: a candidate that fails any of these on its own fresh evaluations is not shipped; the
next candidate is tried."""
PASS_GATES: Final = ("G1", "G2", "G2b", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11")
"""§4: ``phase2_gate.py`` exits 0 iff every one of these passes (G9 is ``phase2_ci_check.py``)."""
ShipGate = Literal["G1", "G2", "G2b", "G3", "G4", "G5", "G6"]
LOCAL_SHIP_GATES: Final = ("G1", "G2", "G2b", "G3", "G4")
"""The ship gates G7 recomputes for every §2.5 attempt from ``phase2.json``'s own fresh
evaluations (G5 and G6 need the candidate's fixture and results, which it does not hold)."""
UBUNTU_GATES: Final = ("G2", "G2b", "G3", "G4")
"""§4: the gates evaluated on the ubuntu compare too (G3 on its 0..199 part)."""
G10_FIELDS: Final = (
    "flags",
    "num_timesteps",
    "total_timesteps",
    "peak_tree_rss_mib",
    "selection_protocol_hash",
    "env_algo_version",
    "git_dirty",
)
"""The summary fields G10 judges (every other §5.1 field must be present too)."""

# --- bars (§4) ------------------------------------------------------------------------------------

MAX_MEAN_SATURATION: Final = 0.10
"""G4: PPO's mean per-episode saturation on 0..199 (selection uses 0.08, a margin under it)."""
SUCCESS_MARGIN: Final = Fraction(3, 100)
"""G2b: success(PPO) >= success(LQR) - 0.03, decided on exact fractions (success rates are k/n)."""
MAX_PEAK_TREE_RSS_MIB: Final = 6144
"""G10: every reported run's peak tree RSS stays below this (the hard limit is 8,192)."""
MIN_G11_RUNS: Final = 2
"""G11: at least this many of the 3 PPO runs pass the PID gate, G2, G3 and G4."""
N_PPO_RUNS: Final = 3
N_SAC_RUNS: Final = 3
N_ABLATION_RUNS: Final = 1
PAIRED_ABS_TOL: Final = 1e-12
"""A stored paired row must equal its recomputation within this (decision log B19: the bootstrap
sums in a platform's SIMD order, so a row computed on ubuntu can differ from a Mac recomputation
in the last bits; the known answers are pinned at the same 1e-12)."""

# --- protocols ------------------------------------------------------------------------------------

CHALLENGE_ID: Final = "balance-v1"
CALIBRATION_PROTOCOL: Final = "balance-v1/standard"
CALIBRATION_SEEDS: Final = tuple(range(200))
SELECTION_PROTOCOL: Final = "balance-v1/standard"
SELECTION_SEEDS: Final = tuple(range(2000, 2200))
EVAL_PROTOCOL: Final = "eval_v1"


def protocol(protocol_id: str, seeds: Sequence[int] | None = None) -> EvalProtocol:
    """``EvalProtocol.from_id(protocol_id)``, with ``seeds`` in place of its own when given, as
    ``topple run --seed`` and ``topple compare --seeds`` build it."""
    base = EvalProtocol.from_id(protocol_id)
    return base if seeds is None else dataclasses.replace(base, seeds=tuple(seeds))


def calibration_protocol() -> EvalProtocol:
    return protocol(CALIBRATION_PROTOCOL, CALIBRATION_SEEDS)


def selection_protocol() -> EvalProtocol:
    return protocol(SELECTION_PROTOCOL, SELECTION_SEEDS)


def eval_protocol() -> EvalProtocol:
    return protocol(EVAL_PROTOCOL)


# --- the repository -------------------------------------------------------------------------------

REPO: Final = Path(__file__).resolve().parents[1]
PID_SUBMISSION: Final = Path("submissions/pid_example.py")
LQR_SUBMISSION: Final = Path("submissions/lqr_example.py")
SHIPPED_SUBMISSION: Final = Path("submissions/ppo_balance")
FIXTURE_DIR: Final = Path("tests/data")
PHASE2_MD: Final = "phase2.md"
"""Rendered beside ``phase2.json`` (``docs/results/``)."""
MB: Final = 1_000_000
RESULT_FILES: Final[Mapping[str, int | None]] = {
    "hero.mp4": 8 * MB,
    "hero.gif": 6 * MB,
    "median_pid_ppo.mp4": 8 * MB,
    "median_lqr_ppo.mp4": 8 * MB,
    "gallery_1.gif": 2 * MB,
    "gallery_2.gif": 2 * MB,
    "gallery_3.gif": 2 * MB,
    "learning_curves.png": None,
    "shove.png": None,
    "ci_ubuntu/results.json": MAX_RESULTS_BYTES,
    "ci_ubuntu/cmp_cal.json": MAX_RESULTS_BYTES,
    "ci_ubuntu/source.json": 4096,
}
"""Every required file of ``docs/results/phase2/`` (§5.2 items 1, 3, 4, 5 and the ubuntu
evidence with its source) and its size cap in bytes (``None``: only the total cap applies).
``MB`` is 10^6."""
CI_SOURCE_FILE: Final = "source.json"
"""Beside the ubuntu evidence (``--ci-dir``): the CI run it came from (``CiSource``)."""
MAX_CI_SOURCE_BYTES: Final = 4096
EVIDENCE_PATHS: Final = (
    "python",
    "submissions",
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/ci.yml",
    "tests/test_shipped_policy.py",
)
"""What produces the ubuntu evidence's numbers: G8 fails if any of these changed between the CI
run's commit and HEAD (§4), so the evidence is what HEAD's code computes."""
MAX_RESULTS_TOTAL_BYTES: Final = 20 * MB
"""§5.2: all of ``docs/results/phase2/``."""
MAX_RESULTS_ENTRIES: Final = 256
"""The walk of ``docs/results/phase2/`` stops past this many entries (G8 then fails)."""

# --- phase2.json ----------------------------------------------------------------------------------

PHASE2_FORMAT: Final = "topple.phase2/1"
MAX_RUNS: Final = 32
MAX_HISTORY: Final = 10_000
MAX_VERSIONS: Final = 16
MAX_ATTEMPTS: Final = 2
"""§1: at most 2 re-runs of a ``wall_clock_abort`` run."""
MAX_FREE_KEYS: Final = 64

RunName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
RunPath = Annotated[
    str,
    StringConstraints(
        max_length=200, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*$"
    ),
]
"""A run directory relative to the checkout, as ``init.from_runs`` names it (``runs/ppo_s0``):
no absolute path and no component starting with a dot."""
GitCommit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
VersionKey = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_.-]{1,32}$")]
Flag = Literal[  # topple.train.selection.FLAGS, in order (a test keeps them equal)
    "std_collapse",
    "late_regression",
    "high_energy",
    "no_eligible",
    "rss_abort",
    "cpu_abort",
    "wall_clock_abort",
    "nondeterministic",
]
NonNegInt = Annotated[int, Field(ge=0, lt=MAX_SEED)]
NonNegFloat = Annotated[float, Field(ge=0.0)]
Fraction01 = Annotated[float, Field(ge=0.0, le=1.0)]
LoadAvg = tuple[NonNegFloat, NonNegFloat, NonNegFloat]
"""``os.getloadavg()``: the 1, 5 and 15 minute load averages."""
Role = Literal["ppo", "sac", "ablation"]


class SummaryError(ValueError):
    """A rule of this module's models broken; its text is fixed words and numbers, never an
    input string, so ``describe_error`` may show it."""


def check_json_numbers(value: JsonValue) -> None:
    """Contract §15 for a free-form JSON value: no NaN or Infinity (pydantic's parser accepts
    them, and ``1e999`` reads as Infinity), and every integer exact in a JSON double
    (|n| < 2**53, as seeds are). ``SummaryError`` otherwise. Iterative: the parser bounds the
    nesting, this walk never recurses."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise SummaryError("a number is not finite (NaN or Infinity)")
        elif isinstance(item, int) and not isinstance(item, bool) and abs(item) >= MAX_SEED:
            raise SummaryError("an integer is not exact in a JSON double (|n| >= 2**53)")


def _json_numbers(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    check_json_numbers(value)
    return value


FreeObject = Annotated[
    dict[str, JsonValue], Field(max_length=MAX_FREE_KEYS), AfterValidator(_json_numbers)
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class HistoryRowModel(_Strict):
    """One ``selection_history`` row (§2.3, ``topple.train.selection.HistoryRow.to_json``)."""

    step: NonNegInt
    success_rate: Fraction01
    score: Annotated[float, Field(ge=0.0, le=100.0)]
    median_energy_j: NonNegFloat
    mean_saturation: Fraction01
    extras: Annotated[dict[str, float], Field(max_length=MAX_FREE_KEYS)]
    eligible: bool
    std_mean: NonNegFloat | None
    frac_mean_outside_1: Fraction01 | None
    excess_penalty_mean: NonNegFloat | None
    cpu_s: NonNegFloat | None
    wall_s: NonNegFloat | None
    onnx_sha256: Sha256

    def result(self) -> SelectionResult:
        return SelectionResult(
            step=self.step,
            success_rate=self.success_rate,
            score=self.score,
            median_energy_j=self.median_energy_j,
            mean_saturation=self.mean_saturation,
            scores=(),
            extras=self.extras,
        )


class Best(_Strict):
    """``best.json`` (§2.4 step 5) with its checkpoint's ``onnx_sha256``."""

    step: NonNegInt
    eligible: bool
    rule: Literal["pid", "waypoint", "none"]
    onnx_sha256: Sha256


class RetryAttempt(_Strict):
    """A ``wall_clock_abort`` attempt kept as ``runs/<name>.attempt<k>`` (§1)."""

    dir: RunName
    flags: tuple[Flag, ...]
    wall_clock_s: NonNegFloat
    load_avg_start: LoadAvg
    cpu_s: NonNegFloat


class InitSource(_Strict):
    """A fine-tune's starting checkpoint (2b): ``run`` as the config's ``init.from_runs``
    names it."""

    run: RunPath
    step: NonNegInt
    onnx_sha256: Sha256


def _best_in_history(best: Best, history: Sequence[HistoryRowModel]) -> None:
    """``best`` names one history row, with that row's eligibility and ONNX sha256."""
    rows = [row for row in history if row.step == best.step]
    if len(rows) != 1:
        raise SummaryError(f"best.step {best.step} must be exactly one selection_history step")
    if (rows[0].eligible, rows[0].onnx_sha256) != (best.eligible, best.onnx_sha256):
        raise SummaryError("best.eligible and best.onnx_sha256 must be its history row's")
    steps = [row.step for row in history]
    if len(set(steps)) != len(steps):
        raise SummaryError("selection_history steps must be unique")


class RunSummary(_Strict):
    """A reported run's committed summary in ``phase2.json``: exactly the §5.1 fields
    (normative; G10 and the report read only these). ``init_source`` is a fine-tune's only."""

    run: RunName
    algo: Literal["ppo", "sac"]
    seed: NonNegInt
    config_sha256: Sha256
    config_file_sha256: Sha256
    total_timesteps: Annotated[int, Field(ge=1, lt=MAX_SEED)]
    num_timesteps: NonNegInt
    selection_protocol_hash: Sha256
    env_algo_version: Annotated[int, Field(ge=0, le=2**31)]
    bundle_hash: Sha256
    git_commit: GitCommit
    git_dirty: bool
    versions: Annotated[dict[VersionKey, VersionStr | None], Field(max_length=MAX_VERSIONS)]
    mp_start_method: Literal["fork", "forkserver", "spawn"] | None
    torch_threads: Annotated[int, Field(ge=1, le=1024)]
    wall_clock_s: NonNegFloat
    train_s: NonNegFloat
    selection_s: NonNegFloat
    cpu_s: NonNegFloat
    load_avg_start: LoadAvg
    load_avg_end: LoadAvg
    peak_tree_rss_mib: NonNegFloat
    peak_procs: Annotated[int, Field(ge=1, le=1_000_000)]
    selection_history: Annotated[
        tuple[HistoryRowModel, ...], Field(min_length=1, max_length=MAX_HISTORY)
    ]
    best: Best
    flags: tuple[Flag, ...]
    e_ref_j: Annotated[float, Field(gt=0.0)]
    attempts: Annotated[tuple[RetryAttempt, ...], Field(max_length=MAX_ATTEMPTS)]
    init_source: InitSource | None = None

    @model_validator(mode="after")
    def _consistent(self) -> RunSummary:
        _best_in_history(self.best, self.selection_history)
        if len(set(self.flags)) != len(self.flags):
            raise SummaryError("flags must be unique")
        return self


class RunCore(_Strict):
    """The part of a summary the run table, G7, G11 and the §2.5 candidate order read; the
    rest of the summary is G10's (``RunSummary``), so a summary missing a field only G10 judges
    fails G10 alone."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True, allow_inf_nan=False)

    run: RunName
    algo: Literal["ppo", "sac"]
    selection_history: Annotated[
        tuple[HistoryRowModel, ...], Field(min_length=1, max_length=MAX_HISTORY)
    ]
    best: Best
    flags: tuple[Flag, ...]

    @model_validator(mode="after")
    def _consistent(self) -> RunCore:
        _best_in_history(self.best, self.selection_history)
        return self

    def best_row(self) -> HistoryRowModel:
        return next(row for row in self.selection_history if row.step == self.best.step)


class RunEntry(_Strict):
    """A reported run: its role (one of the 3 PPO seeds, the 3 SAC seeds or the ablation) and
    its summary, validated by the gates that read it (``run_summary``, ``run_core``)."""

    role: Role
    summary: FreeObject


class FreshRow(_Strict):
    """A run's fresh evaluation: its exported policy (``policy_sha256``) is entry ``entry`` of
    both fresh comparisons; ``report_only`` for a ``--report-only`` export (§3.3);
    ``below_pid_gate`` marks a run whose best checkpoint is not eligible."""

    run: RunName
    entry: Identifier
    policy_sha256: Sha256
    report_only: bool
    below_pid_gate: bool


class Fresh(_Strict):
    """PID (entry 0), LQR (entry 1) and every exported policy, evaluated fresh on eval_v1 and on
    the calibration seeds 0..199 (§5.1)."""

    eval_v1: Comparison
    calibration: Comparison
    rows: Annotated[tuple[FreshRow, ...], Field(max_length=MAX_RUNS)]


class ShipAttempt(_Strict):
    """One §2.5 candidate tried for shipping, the ship gates it failed (none: shipped) and
    where they were measured: ``local`` (its own fresh evaluations, which ``phase2.json``
    holds, so G7 recomputes their G1-G4 part) or ``ubuntu`` (CI's compare after push, G2-G4
    only, for a candidate that passed every ship gate locally); ``null`` when it failed none."""

    run: RunName
    failed_gates: tuple[ShipGate, ...]
    failed_on: Literal["local", "ubuntu"] | None

    @model_validator(mode="after")
    def _consistent(self) -> ShipAttempt:
        if (self.failed_on is None) != (not self.failed_gates):
            raise SummaryError("failed_on is null exactly when no gate failed")
        if self.failed_on == "ubuntu" and not set(self.failed_gates) <= set(UBUNTU_GATES):
            raise SummaryError("only G2, G2b, G3 and G4 are measured on ubuntu")
        return self


class Shipped(_Strict):
    run: RunName
    policy_sha256: Sha256
    attempts: Annotated[tuple[ShipAttempt, ...], Field(min_length=1, max_length=N_PPO_RUNS)]


class G11Row(_Strict):
    """One PPO run of the G11 table: its best's eligibility and G2, G2b (reported, not gating),
    G3 and G4 on its fresh 0..199 evaluation."""

    run: RunName
    eligible: bool
    g2: bool
    g2b: bool
    g3: bool
    g4: bool


class Phase2Report(_Strict):
    """``phase2.json`` (``topple.phase2/1``)."""

    format: Literal["topple.phase2/1"] = PHASE2_FORMAT
    runs: Annotated[tuple[RunEntry, ...], Field(max_length=MAX_RUNS)]
    fresh: Fresh
    shipped: Shipped
    g11: Annotated[tuple[G11Row, ...], Field(max_length=N_PPO_RUNS)]
    pilot: FreeObject
    """Pilot constants (§5.1), each with its ``source``."""
    render: FreeObject
    """Values only ``phase2.md`` renders (shove curves, GIF rungs, the CI run id, ...)."""


class CiSource(_Strict):
    """``ci_ubuntu/source.json`` (``topple.ci_source/1``, written by ``phase2_results.py
    --fetch-ci``): the CI run the ubuntu evidence was downloaded from, a ``push`` run (never
    ``pull_request``) of ``branch`` at ``head_sha``."""

    format: Literal["topple.ci_source/1"]
    run_id: Annotated[int, Field(ge=1, lt=MAX_SEED)]
    head_sha: GitCommit
    event: Literal["push"]
    branch: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.][A-Za-z0-9_./-]{0,254}$")]


def parse_ci_source(data: bytes) -> CiSource:
    """``data`` as ``ci_source/source.json``: exact JSON types, every field required."""
    return CiSource.model_validate_json(data, strict=True)


def parse_phase2(data: bytes) -> Phase2Report:
    """``data`` as ``phase2.json``: exact JSON types, ``format`` required (also on the two
    embedded comparisons). ``ValueError`` (pydantic's ``ValidationError`` included) else."""
    if len(data) > MAX_RESULTS_BYTES:
        raise ValueError(f"phase2.json has more than {MAX_RESULTS_BYTES} bytes")
    report = Phase2Report.model_validate_json(data, strict=True)
    if "format" not in report.model_fields_set:
        raise ValueError(f"no format: not a {PHASE2_FORMAT} file")
    for name in ("eval_v1", "calibration"):
        if "format" not in getattr(report.fresh, name).model_fields_set:
            raise ValueError(f"fresh.{name}: no format: not a topple.compare/1 object")
    return report


def read_input(path: Path, cap: int = MAX_RESULTS_BYTES) -> bytes:
    """``path``'s bytes, read as contract §15 reads ``results.json``: a regular file (symlinks
    followed; a FIFO, device or directory refused without blocking) of at most ``cap`` bytes.
    ``ValueError`` or ``OSError`` otherwise."""
    fh, _ = open_regular(path, cap)
    with fh:
        data = fh.read(cap + 1)  # it may have grown since it was examined
    if len(data) > cap:
        raise ValueError(f"{path.name} has more than {cap} bytes")
    return data


def read_phase2(path: Path) -> Phase2Report:
    return parse_phase2(read_input(path))


def _validate_summary[M: BaseModel](model: type[M], summary: Mapping[str, JsonValue]) -> M:
    """``summary`` (already parsed from JSON) validated in JSON mode, so arrays are tuples and
    scalars keep their exact JSON types. ``ValueError`` (``ValidationError`` included) if it is
    invalid, also when it never went through the parser (a non-finite number is a
    ``SummaryError`` here, not a failure to re-encode it)."""
    check_json_numbers(dict(summary))  # a Mapping, not necessarily a JSON object
    return model.model_validate_json(json.dumps(summary, allow_nan=False), strict=True)


def run_summary(entry: RunEntry) -> RunSummary:
    return _validate_summary(RunSummary, entry.summary)


def run_core(entry: RunEntry) -> RunCore:
    return _validate_summary(RunCore, entry.summary)


def run_label(entry: RunEntry, index: int) -> str:
    """The run's name if its summary has a valid one, else its position."""
    name = entry.summary.get("run")
    if isinstance(name, str) and len(name) <= 64 and name.isascii() and name.isprintable():
        return repr(name)
    return f"#{index}"


def describe_error(exc: BaseException) -> str:
    """A one-line, printable description of ``exc`` that never echoes input values: for a
    pydantic error, the location and type of its first errors (``missing field 'x'``), and the
    reason when one of this module's rules (``SummaryError``: fixed text) raised it."""
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors()[:4]:
            loc = ".".join(printable(str(p), 64) for p in err["loc"]) or "(root)"
            reason = err.get("ctx", {}).get("error")
            if err["type"] == "missing":
                parts.append(f"missing field {loc!r}")
            elif isinstance(reason, SummaryError):
                parts.append(f"{loc}: {printable(str(reason))}")
            else:
                parts.append(f"{loc}: {err['type']}")
        more = exc.error_count() - len(parts)
        return "; ".join(parts) + (f" (and {more} more)" if more > 0 else "")
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    return printable(str(exc))


def printable(text: str, limit: int = 300) -> str:
    """``text`` on one line, safe for a terminal or a CI log: tabs and line breaks become
    spaces, every other character that is not printable (controls, escapes, bidi overrides,
    line separators) ``?``; at most ``limit`` characters."""
    return "".join(
        ch if ch.isprintable() else " " if ch in "\t\n\r" else "?" for ch in text[:limit]
    )


# --- §2.5 shipping --------------------------------------------------------------------------------


def candidates(runs: Sequence[RunCore]) -> list[str]:
    """§2.5's shipping candidates: PPO runs whose best checkpoint is eligible and that carry no
    abort flag and no ``nondeterministic``, best first by their best's ``rank_key`` on the
    selection seeds (never on 0..199); a tie keeps the listed order."""
    eligible = [
        r for r in runs if r.algo == "ppo" and r.best.eligible and not set(r.flags) & NOT_SHIPPABLE
    ]
    ranked = sorted(
        eligible, key=lambda r: rank_key(r.best_row().result(), "balance"), reverse=True
    )
    return [r.run for r in ranked]


@dataclass(frozen=True)
class ShipChoice:
    """The shipped run (``None``: no candidate passed, so 2a fails) and every attempt: each
    candidate tried, in order, with the ship gates it failed."""

    run: str | None
    attempts: tuple[tuple[str, tuple[str, ...]], ...]


def choose_shipped(ranked: Sequence[str], failures: Callable[[str], Collection[str]]) -> ShipChoice:
    """§2.5: try the candidates in order; ``failures(run)`` evaluates one (the gates it fails,
    any of them) and is called only until one passes every ``SHIP_GATES`` gate. Failures of
    other gates never block shipping."""
    attempts: list[tuple[str, tuple[str, ...]]] = []
    for run in ranked:
        failed = set(failures(run))
        ship_failed = tuple(gate for gate in SHIP_GATES if gate in failed)
        attempts.append((run, ship_failed))
        if not ship_failed:
            return ShipChoice(run, tuple(attempts))
    return ShipChoice(None, tuple(attempts))


# --- subprocesses ---------------------------------------------------------------------------------

DEFAULT_TIMEOUT_S: Final = 60.0
JOIN_TIMEOUT_S: Final = 5.0
MAX_OUTPUT_BYTES: Final = 1 << 20


class CommandError(Exception):
    """A command that could not run, ran too long or printed too much."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: str
    """At most the first 4 KiB, decoded with replacement and made printable."""


def run_command(
    args: Sequence[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = MAX_OUTPUT_BYTES,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run ``args`` (no shell) with stdin closed. Its output is read as it comes, at most
    ``max_bytes`` of stdout and 4 KiB of stderr: printing more, or running past ``timeout_s``,
    kills it and raises ``CommandError``, as does a command that cannot be started."""
    try:
        proc = subprocess.Popen(
            list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=None if env is None else dict(env),
        )
    except OSError as exc:
        raise CommandError(f"cannot run {args[0]}: {exc.strerror or type(exc).__name__}") from None
    assert proc.stdout is not None  # both are pipes
    assert proc.stderr is not None
    out = _Reader(proc.stdout, max_bytes, proc)
    err = _Reader(proc.stderr, 4096, None)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise CommandError(f"{args[0]} ran longer than {timeout_s:g} s") from None
    finally:
        out.join()
        err.join()
    if out.overflow:
        raise CommandError(f"{args[0]} printed more than {max_bytes} bytes")
    stderr = printable(err.data.decode("utf-8", "replace"), 4096)
    return CommandResult(proc.returncode, out.data, stderr)


class _Reader:
    """Drain a pipe on a thread, keeping at most ``cap`` bytes; past that, kill ``proc`` (if
    given) and keep draining, so the child never blocks on a full pipe."""

    def __init__(self, pipe: IO[bytes], cap: int, proc: subprocess.Popen[bytes] | None) -> None:
        self.data = b""
        self.overflow = False
        self._chunks: list[bytes] = []
        self._size = 0
        self._thread = threading.Thread(target=self._run, args=(pipe, cap, proc), daemon=True)
        self._thread.start()

    def _run(self, pipe: IO[bytes], cap: int, proc: subprocess.Popen[bytes] | None) -> None:
        with pipe:
            while chunk := pipe.read(65536) if not hasattr(pipe, "read1") else pipe.read1(65536):
                room = cap - self._size
                if len(chunk) > room:
                    self.overflow = True
                    chunk = chunk[: max(room, 0)]
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                if chunk:
                    self._chunks.append(chunk)
                    self._size += len(chunk)

    def join(self) -> None:
        """Wait (a bounded time: a grandchild may hold the pipe open) and collect the data."""
        self._thread.join(timeout=JOIN_TIMEOUT_S)
        self.data = b"".join(self._chunks)


def git_env() -> dict[str, str]:
    """The environment for local git commands: ``os.environ`` without ``GIT_*`` variables (a
    ``GIT_DIR`` or ``GIT_WORK_TREE`` from a hook would point git elsewhere), never a pager or a
    prompt."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"})
    return env
