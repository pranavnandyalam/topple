"""``topple compare``: several submissions on one protocol, paired by seed
(docs/design/phase-2.md §5.3).

``compare`` loads and validates every submission once in this process (``submission_ref``),
runs their episodes in one evaluation pool (at most ``min(workers, 4, cpus)`` spawn workers,
§1) and records, per submission, its per-seed arrays in seed order: score, success, energy,
saturation and the recorded termination reason (contract §11; ``error`` for an episode the
harness stopped). Every ordered pair i > j of submissions gets a paired bootstrap row of score
differences (``a`` = entry i, ``b`` = entry j) from ``topple.stats.paired_bootstrap``, the one
algorithm the Phase 2 gates recompute from those arrays.

The ``topple.compare/1`` file is read like contract §15's ``results.json``: a regular file of at
most 16 MiB (``read_comparison``), strict JSON types, no NaN or infinity, every object closed,
``format`` required, ids sanitized and every string bounded. Each episode's termination is a
recorded reason or ``missing`` and agrees with its score and success: ``error``,
``sim_unstable`` and ``missing`` score 0 with no success (contract §15), and only a ``timeout``
can be a success (§12). Its numbers are as trustworthy as whoever ran the comparison: a ranked
score never comes from one (contract §16).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Annotated, Final, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from topple._inputs import open_regular
from topple.evaluate import FALL_REASONS, platform_tag, run_versions
from topple.pool import eval_pool, pool_size, submission_ref, submit_episodes
from topple.results import (
    MAX_EPISODES,
    MAX_ID_CHARS,
    MAX_METRIC_CHARS,
    MAX_METRICS,
    MAX_RESULTS_BYTES,
    Identifier,
    MetricName,
    PlatformStr,
    Printable,
    ProtocolId,
    Sha256,
    Versions,
    sanitize_id,
)
from topple.rng import MAX_SEED
from topple.stats import N_RESAMPLES, SEED, bootstrap_ci, paired_bootstrap
from topple.waypoint import SCORE_PARTS

if TYPE_CHECKING:
    from topple.evaluate import EvalProtocol
    from topple.loader import Submission
    from topple.results import EpisodeResult

FORMAT: Final = "topple.compare/1"
MAX_ENTRIES: Final = 32
"""Most submissions one comparison holds (496 paired rows)."""
MAX_RESAMPLES: Final = 1_000_000
CHUNK_EPISODES = 25
"""Episodes per pool task: enough tasks to keep every worker busy, few enough that rebuilding
the controller per task stays cheap. Results never depend on it."""
MISSING: Final = "missing"
UNKNOWN_ID: Final = "unknown"
MINUS: Final = "\u2212"  # a typographic minus: ids may hold "-", which is escaped
WAYPOINT_EXTRAS: Final = (
    "waypoints",
    "progress",
    "docked",
    "fell",
    "t_finish_s",
    "max_hold_steps",
    "fall_cause",
    *SCORE_PARTS,
)
"""The waypoint-v1 metrics (§8.6) an entry records per seed beyond the common arrays, so the 2b
gate recomputes docked and fell rates, fall causes and score parts from the file alone; ``None``
for an episode without metrics. balance-v1 entries record none."""
EXTRAS_OF_KIND: Final[dict[str, tuple[str, ...]]] = {"balance": (), "waypoint": WAYPOINT_EXTRAS}
_ASCII_PUNCTUATION: Final = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


Seed = Annotated[int, Field(ge=0, lt=MAX_SEED)]
Score = Annotated[float, Field(ge=0.0, le=100.0)]
Difference = Annotated[float, Field(ge=-100.0, le=100.0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Energy = Annotated[float, Field(ge=0.0)]
Termination = Literal["fell", "floor_contact", "sim_unstable", "timeout", "error", "missing"]
"""A recorded termination reason (contract §11: ``fell``, ``floor_contact``, ``sim_unstable``,
``timeout``, ``error``), or ``missing``."""
SCORELESS: Final = frozenset({"error", "sim_unstable", MISSING})
"""Reasons of an episode that scores 0 and is no success: an error (contract §15;
``sim_unstable`` is one) or no episode at all."""
SURVIVED: Final = "timeout"
"""The only reason a success can have (contract §12: success implies survived)."""
ExtraValue = (
    bool
    | int
    | float
    | Annotated[str, StringConstraints(max_length=MAX_METRIC_CHARS), Printable]
    | None
)
PerSeed = Field(max_length=MAX_EPISODES)


class Entry(_Strict):
    """One submission's episodes, every array in seed order. ``id`` is the submission's name
    (untrusted: sanitized as contract §15 sanitizes ``controller.id``; escape it before
    rendering); ``sha256`` its submission sha256 and ``policy_sha256`` sha256(policy.onnx) for a
    ``topple.policy/1`` submission (``None`` for Python). ``extras`` maps a challenge's further
    metric names to their per-seed values (waypoint-v1, §8.6; none for balance-v1)."""

    id: Identifier
    policy_sha256: Sha256 | None
    sha256: Sha256
    scores: Annotated[tuple[Score, ...], PerSeed]
    success: Annotated[tuple[bool, ...], PerSeed]
    energy_j: Annotated[tuple[Energy, ...], PerSeed]
    saturation_fraction: Annotated[tuple[Fraction, ...], PerSeed]
    termination: Annotated[tuple[Termination, ...], PerSeed]
    extras: Annotated[
        dict[MetricName, Annotated[tuple[ExtraValue, ...], PerSeed]],
        Field(max_length=MAX_METRICS),
    ]

    @field_validator("id", mode="before")
    @classmethod
    def _sanitize(cls, value: object) -> object:
        """Before the type check, so a lone surrogate is replaced rather than refused."""
        return sanitize_id(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _consistent(self) -> Entry:
        n = len(self.scores)
        arrays = (self.success, self.energy_j, self.saturation_fraction, self.termination)
        if any(len(arr) != n for arr in (*arrays, *self.extras.values())):
            raise ValueError(f"entry {self.id!r}: every per-seed array must have {n} values")
        for k, reason in enumerate(self.termination):
            if reason in SCORELESS and (self.scores[k] != 0.0 or self.success[k]):
                raise ValueError(
                    f"entry {self.id!r}: seed {k} ({reason}) must score 0 with no success"
                )
            if self.success[k] and reason != SURVIVED:
                raise ValueError(
                    f"entry {self.id!r}: seed {k} ({reason}) cannot be a success, "
                    f"only a {SURVIVED} can"
                )
        return self


class PairedRow(_Strict):
    """``paired_bootstrap(entry a's scores, entry b's scores, n_resamples, seed)``: the mean
    score difference a - b over the seeds and its 95% interval."""

    a: Identifier
    b: Identifier
    mean: Difference
    lo: Difference
    hi: Difference
    n_resamples: Annotated[int, Field(ge=1, le=MAX_RESAMPLES)]
    seed: Seed

    @field_validator("a", "b", mode="before")
    @classmethod
    def _sanitize(cls, value: object) -> object:
        return sanitize_id(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _ordered(self) -> PairedRow:
        if self.lo > self.hi:
            raise ValueError("a paired row's lo must not exceed its hi")
        return self


class Comparison(_Strict):
    """``topple.compare/1``: submissions on the seeds of one protocol (``protocol_hash`` covers
    the seeds), with the machine's platform and versions (per-episode ONNX results are
    bit-exact only per platform, ``cpu_isa`` and onnxruntime version, contract §17).
    ``complete`` is true iff no episode is ``missing``."""

    format: Literal["topple.compare/1"] = FORMAT
    protocol: ProtocolId
    protocol_hash: Sha256
    complete: bool
    seeds: Annotated[tuple[Seed, ...], Field(min_length=1, max_length=MAX_EPISODES)]
    platform: PlatformStr
    versions: Versions
    entries: Annotated[tuple[Entry, ...], Field(min_length=1, max_length=MAX_ENTRIES)]
    paired: Annotated[tuple[PairedRow, ...], Field(max_length=MAX_ENTRIES * MAX_ENTRIES)]

    @model_validator(mode="after")
    def _consistent(self) -> Comparison:
        n = len(self.seeds)
        ids = [e.id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("entry ids must be unique")
        if any(len(e.scores) != n for e in self.entries):
            raise ValueError(f"every entry must hold one episode per seed ({n})")
        pairs = [(r.a, r.b) for r in self.paired]
        if pairs != [(ids[i], ids[j]) for i, j in ordered_pairs(len(ids))]:
            raise ValueError("paired rows must be every ordered pair i > j of entries, in order")
        missing = any(MISSING in e.termination for e in self.entries)
        if self.complete == missing:
            raise ValueError("complete must be true exactly when no episode is missing")
        return self


def ordered_pairs(n: int) -> list[tuple[int, int]]:
    """(i, j) for every i > j of ``n`` entries: (1, 0), (2, 0), (2, 1), ..."""
    return [(i, j) for i in range(n) for j in range(i)]


# --- running ----------------------------------------------------------------------------------


def compare(subs: Sequence[Path], protocol: EvalProtocol, *, workers: int = 4) -> Comparison:
    """Run every submission of ``subs`` on every seed of the public ``protocol``.

    Each submission is loaded once here (a policy fully validated and smoke-tested for the
    protocol's challenge; ``LoadError`` otherwise) and run by reference in an evaluation pool
    of ``pool_size(workers)`` processes. ``ValueError`` for no submissions, more than
    ``MAX_ENTRIES``, two with one name (ids must tell the rows apart), a private protocol or a
    bad ``workers``; a failing pool task cancels the rest and raises.
    """
    if not subs:
        raise ValueError("compare needs at least one submission")
    if len(subs) > MAX_ENTRIES:
        raise ValueError(f"compare takes at most {MAX_ENTRIES} submissions")
    if protocol.seed_set != "public":
        raise ValueError("compare runs public protocols only (pool workers never see a salt)")
    pool_size(workers)  # a bad value fails before anything is loaded
    loaded = [submission_ref(sub, challenge=protocol.challenge) for sub in subs]
    ids = [_entry_id(sub) for sub, _ in loaded]
    for name, count in Counter(ids).items():
        if count > 1:
            raise ValueError(f"two submissions are named {name!r}; rename one")
    chunks = _chunks(len(protocol.seeds), CHUNK_EPISODES)
    pool = eval_pool(workers)
    try:
        futures = [
            [submit_episodes(pool, ref, protocol, chunk) for chunk in chunks] for _, ref in loaded
        ]
        runs = [_collect(fs, len(protocol.seeds)) for fs in futures]
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    entries = tuple(
        _entry(name, sub, episodes, protocol.challenge.kind)
        for name, (sub, _), episodes in zip(ids, loaded, runs, strict=True)
    )
    return Comparison(
        protocol=protocol.id,
        protocol_hash=protocol.protocol_hash(),
        complete=not any(MISSING in e.termination for e in entries),
        seeds=tuple(protocol.seeds),
        platform=platform_tag(),
        versions=run_versions(),
        entries=entries,
        paired=tuple(_paired(entries[i], entries[j]) for i, j in ordered_pairs(len(entries))),
    )


def _entry_id(sub: Submission) -> str:
    """The submission's name as a results file records it (contract §15)."""
    return sanitize_id(sub.id or UNKNOWN_ID)[:MAX_ID_CHARS]


def _chunks(n: int, size: int) -> list[range]:
    """``range(n)`` cut into consecutive pieces of at most ``size``."""
    return [range(start, min(start + size, n)) for start in range(0, n, size)]


def _collect(futures: Sequence[Future[list[EpisodeResult]]], n: int) -> list[EpisodeResult]:
    """The results of one submission's tasks, which cover the indices 0..n-1 in order."""
    episodes = list(itertools.chain.from_iterable(f.result() for f in futures))
    if [e.index for e in episodes] != list(range(n)):
        raise RuntimeError("the pool returned episodes out of seed order")
    return episodes


def _entry(name: str, sub: Submission, episodes: Sequence[EpisodeResult], kind: str) -> Entry:
    return Entry(
        id=name,
        policy_sha256=sub.policy_sha256,
        sha256=sub.sha256,
        scores=tuple(float(e.score) for e in episodes),
        success=tuple(bool(e.success) for e in episodes),
        energy_j=tuple(_metric(e, "energy_j") for e in episodes),
        saturation_fraction=tuple(_metric(e, "saturation_fraction") for e in episodes),
        termination=tuple(_termination(e) for e in episodes),
        extras=_extras(episodes, kind),
    )


def _extras(episodes: Sequence[EpisodeResult], kind: str) -> dict[str, tuple[ExtraValue, ...]]:
    """``kind``'s extra metrics per seed (``None`` where an episode has no metrics)."""
    return {
        name: tuple(None if e.metrics is None else e.metrics[name] for e in episodes)
        for name in EXTRAS_OF_KIND[kind]
    }


def _metric(e: EpisodeResult, name: str) -> float:
    """A numeric metric of ``e`` (0.0 for an episode without metrics: a missing one)."""
    return 0.0 if e.metrics is None else float(e.metrics[name])


def _termination(e: EpisodeResult) -> Termination:
    """The recorded reason (``Comparison`` refuses any other value), or ``missing``."""
    return cast(Termination, MISSING if e.metrics is None else str(e.metrics["termination"]))


def _paired(a: Entry, b: Entry) -> PairedRow:
    mean, lo, hi = paired_bootstrap(a.scores, b.scores, n_resamples=N_RESAMPLES, seed=SEED)
    return PairedRow(a=a.id, b=b.id, mean=mean, lo=lo, hi=hi, n_resamples=N_RESAMPLES, seed=SEED)


# --- writing and reading ----------------------------------------------------------------------


def to_json(c: Comparison) -> str:
    """The ``topple.compare/1`` text: indented JSON in field order, then a newline."""
    return c.model_dump_json(indent=2) + "\n"


def parse_comparison(data: bytes) -> Comparison:
    """``data`` as a ``Comparison`` with exact JSON types (nothing coerced); ``ValueError``
    (pydantic's ``ValidationError``) for anything else, a file without ``format`` included."""
    if len(data) > MAX_RESULTS_BYTES:
        raise ValueError(f"a comparison has at most {MAX_RESULTS_BYTES} bytes")
    c = Comparison.model_validate_json(data, strict=True)
    if "format" not in c.model_fields_set:
        raise ValueError(f"no format: not a {FORMAT} file")
    return c


def read_comparison(path: Path) -> Comparison:
    """Read and validate ``path`` as ``read_results`` reads ``results.json`` (contract §15): a
    regular file (symlinks are followed; a FIFO, device or directory is refused without
    blocking on it or reading from it) of at most 16 MiB."""
    try:
        fh, _ = open_regular(path, MAX_RESULTS_BYTES)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    with fh:
        data = fh.read(MAX_RESULTS_BYTES + 1)  # it may have grown since it was examined
    return parse_comparison(data)


# --- Markdown ---------------------------------------------------------------------------------


def to_markdown(c: Comparison) -> str:
    """A Markdown report: one row per submission (score [95% CI], success, falls, errors,
    median energy, mean saturation) and one per paired row. Every id is escaped (every ASCII
    punctuation character backslash-escaped, CommonMark's rule), so a name cannot add markup,
    links or table cells."""
    md = escape_markdown
    n = len(c.seeds)
    isa = c.versions.cpu_isa or "unknown cpu"
    ort = c.versions.onnxruntime or "none"
    lines = [
        f"## {c.protocol}: {n} seeds ({_seed_span(c.seeds)})",  # its grammar has no markup
        "",
        f"protocol_hash {c.protocol_hash[:12]} · {'complete' if c.complete else 'INCOMPLETE'} · "
        f"{md(c.platform)} {md(isa)} · onnxruntime {md(ort)}",
        "",
        "| # | Submission | Score [95% CI] | Success | Falls | Errors | Median energy (J) "
        "| Mean saturation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, e in enumerate(c.entries, start=1):
        score = math.fsum(e.scores) / n  # §12's aggregate
        _, lo, hi = bootstrap_ci(e.scores)
        falls = sum(t in FALL_REASONS for t in e.termination)
        errors = sum(t in ("error", "sim_unstable", MISSING) for t in e.termination)
        lines.append(
            f"| {k} | {md(e.id)} | {score:.2f} [{lo:.2f}, {hi:.2f}] | "
            f"{sum(e.success) / n:.3f} | {falls} | {errors} | {median(e.energy_j):.2f} | "
            f"{math.fsum(e.saturation_fraction) / n:.3f} |"
        )
    if c.paired:
        lines += [
            "",
            "Paired score differences (seed-paired bootstrap, 95% CI):",
            "",
            f"| A {MINUS} B | Mean | 95% CI | Resamples |",
            "|---|---|---|---|",
        ]
        for r in c.paired:
            lines.append(
                f"| {md(r.a)} {MINUS} {md(r.b)} | {r.mean:+.2f} | [{r.lo:+.2f}, {r.hi:+.2f}] | "
                f"{r.n_resamples} (seed {r.seed}) |"
            )
    return "\n".join(lines) + "\n"


def _seed_span(seeds: Sequence[int]) -> str:
    first, last = seeds[0], seeds[-1]
    if contiguous(seeds):
        return f"{first}..{last}"
    return f"{first}, ..., {last}" if len(seeds) > 2 else ", ".join(map(str, seeds))


def contiguous(seeds: Sequence[int]) -> bool:
    """Whether ``seeds`` is ``first..last`` with no gap, decided without building the range: the
    lengths are compared first, so seeds read from a file (``[0, 2**53 - 1]``) cost nothing."""
    if not seeds or seeds[-1] - seeds[0] + 1 != len(seeds):
        return False
    return all(b == a + 1 for a, b in itertools.pairwise(seeds))


def escape_markdown(text: str) -> str:
    """``text`` with every ASCII punctuation character backslash-escaped (CommonMark lets any
    of them be escaped), so untrusted text renders literally inside a table cell."""
    return "".join(f"\\{ch}" if ch in _ASCII_PUNCTUATION else ch for ch in text)
