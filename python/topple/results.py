"""``results.json`` schema ``topple.results/1`` (docs/contract.md §15)."""

from __future__ import annotations

import errno
import itertools
import math
import os
import tempfile
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from topple.ids import parse_protocol, validate_challenge_id

FORMAT: Literal["topple.results/1"] = "topple.results/1"
MAX_DETAIL_CHARS = 300
MAX_ID_CHARS = 200
MAX_PLATFORM_CHARS = 64
MAX_VERSION_CHARS = 64
MAX_EPISODES = 10_000
"""Most episodes one results file lists (a protocol has 20 public seeds)."""
MAX_METRICS = 32
"""Most metrics per episode (balance-v1 reports 11)."""
MAX_METRIC_CHARS = 64
"""Longest metric name and string value."""
MAX_TIME_UPRIGHT_S = 1e9
"""Largest ``time_upright_s`` a complete run may report: far beyond any episode, and small
enough that the sum over ``MAX_EPISODES`` episodes stays far from overflowing."""
MAX_RESULTS_BYTES = 16 * 1024 * 1024
"""Most bytes ``read_results`` reads."""
RESULTS_FILE_MODE = 0o644
REPLACEMENT_CHAR = "�"
_KEPT_CONTROLS = frozenset("\n\t")
_UNSAFE_IN_IDS = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
_NO_DIRECTORY_FSYNC = frozenset({errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP})
_NO_DIRECTORY_OPEN = frozenset({errno.EACCES, errno.EPERM})  # e.g. a write-only drop directory

RunErrorCode = Literal[
    "exception",
    "invalid_action",
    "timeout",
    "crash",
    "protocol",
    "output_limit",
    "load_error",
    "sim_unstable",
]
"""Errors a run can record: the controller's (and ``sim_unstable``, the physics')."""
VerifyErrorCode = Literal["incomplete", "not_reproducible", "missing"]
"""Errors only trusted verification assigns (contract §16)."""
ErrorCode = RunErrorCode | VerifyErrorCode
RUN_ERROR_CODES: frozenset[str] = frozenset(get_args(RunErrorCode))
VERIFY_ERROR_CODES: frozenset[str] = frozenset(get_args(VerifyErrorCode))
Isolation = Literal["inprocess", "subprocess", "container"]


def _printable(value: str) -> str:
    """Refuse what a terminal, a CI log or a page would act on: control, format (bidi overrides,
    zero-width characters) and separator characters but the space (``str.isprintable``)."""
    if not value.isprintable():
        raise ValueError(f"must hold printable characters only, got {value!r}")
    return value


def _protocol_id(value: str) -> str:
    parse_protocol(value)
    return value


Printable = AfterValidator(_printable)
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=MAX_ID_CHARS)]
ChallengeId = Annotated[Identifier, AfterValidator(validate_challenge_id)]
ProtocolId = Annotated[Identifier, AfterValidator(_protocol_id)]
PlatformStr = Annotated[str, StringConstraints(max_length=MAX_PLATFORM_CHARS), Printable]
VersionStr = Annotated[str, StringConstraints(max_length=MAX_VERSION_CHARS), Printable]
MetricName = Annotated[str, StringConstraints(min_length=1, max_length=MAX_METRIC_CHARS), Printable]
MetricValue = (
    float | int | bool | Annotated[str, StringConstraints(max_length=MAX_METRIC_CHARS), Printable]
)
Metrics = Annotated[dict[MetricName, MetricValue], Field(max_length=MAX_METRICS)]


def sanitize_detail(text: str) -> str:
    """Make untrusted error text safe to store: at most ``MAX_DETAIL_CHARS`` characters, with
    lone surrogates and control characters (other than newline and tab) replaced by U+FFFD.

    One character in, one character out, so the result is idempotent and keeps positions.
    """
    return "".join(
        ch
        if ch in _KEPT_CONTROLS or unicodedata.category(ch) not in ("Cc", "Cs")
        else REPLACEMENT_CHAR
        for ch in text[:MAX_DETAIL_CHARS]
    )


def sanitize_id(text: str) -> str:
    """Make an untrusted identifier (a submission's file name) safe to store and show on one
    line: control, format (bidi overrides, zero-width characters) and line-separator characters
    and lone surrogates become U+FFFD, one for one, so the result is idempotent."""
    return "".join(
        REPLACEMENT_CHAR if unicodedata.category(ch) in _UNSAFE_IN_IDS else ch for ch in text
    )


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ErrorInfo(_Strict):
    code: ErrorCode
    step: int = Field(ge=0)
    detail: str = ""  # untrusted: produced by submitted code; escape before rendering

    @field_validator("detail")
    @classmethod
    def _sanitize(cls, value: str) -> str:
        return sanitize_detail(value)


class EpisodeResult(_Strict):
    index: int = Field(ge=0)
    seed: int | None  # None for private seed sets
    success: bool
    score: float = Field(ge=0.0, le=100.0)
    metrics: Metrics | None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _errors_score_zero(self) -> EpisodeResult:
        if self.error is not None and (self.score != 0.0 or self.success):
            raise ValueError("an episode with an error scores 0 and is not a success")
        return self


class Aggregate(_Strict):
    score: float = Field(ge=0.0, le=100.0)
    success_rate: float = Field(ge=0.0, le=1.0)
    mean_time_upright_s: float = Field(ge=0.0)
    episodes: int = Field(ge=0)
    errors: int = Field(ge=0)


class Versions(_Strict):
    topple: VersionStr
    mujoco: VersionStr
    numpy: VersionStr
    obs_schema: VersionStr
    action_spec: VersionStr


class ControllerInfo(_Strict):
    id: Identifier  # untrusted: a submission's name; sanitized, escape before rendering
    sha256: Sha256

    @field_validator("id", mode="before")
    @classmethod
    def _sanitize(cls, value: object) -> object:
        """Before the type check, so a lone surrogate is replaced rather than refused."""
        return sanitize_id(value) if isinstance(value, str) else value


class EvalResults(_Strict):
    """One evaluation's results. ``isolation`` records how the controller ran; it does not make a
    run rankable (Phase 3 decides that from which trusted job produced the file)."""

    format: Literal["topple.results/1"] = FORMAT
    challenge: ChallengeId
    protocol: ProtocolId
    protocol_hash: Sha256
    seed_set: Literal["public", "private"]
    versions: Versions
    platform: PlatformStr
    robot_hash: Sha256
    controller: ControllerInfo
    isolation: Isolation
    complete: bool
    episodes: Annotated[list[EpisodeResult], Field(max_length=MAX_EPISODES)] | None
    aggregate: Aggregate

    @model_validator(mode="after")
    def _consistent(self) -> EvalResults:
        if self.seed_set == "private":
            if self.episodes is not None:
                raise ValueError("private results are redacted: episodes must be null")
            return self
        if self.episodes is None:
            raise ValueError("public results must list their episodes")
        _check_public_episodes(self.episodes, self.aggregate)
        if self.complete:
            _check_complete(self.episodes, self.aggregate)
        return self


def _check_public_episodes(episodes: list[EpisodeResult], aggregate: Aggregate) -> None:
    if aggregate.episodes != len(episodes):
        raise ValueError(
            f"aggregate.episodes is {aggregate.episodes} but {len(episodes)} episodes are listed"
        )
    indices = [e.index for e in episodes]
    if any(b <= a for a, b in itertools.pairwise(indices)):
        raise ValueError("episode index values must be strictly increasing")
    if any(e.seed is None for e in episodes):
        raise ValueError("every public episode must carry its seed")
    errors = sum(e.error is not None for e in episodes)
    if aggregate.errors != errors:
        raise ValueError(f"aggregate.errors is {aggregate.errors} but {errors} episodes errored")


def _check_complete(episodes: list[EpisodeResult], aggregate: Aggregate) -> None:
    """A complete run lists all N of its episodes, none missing, so §12's aggregate of them is
    known exactly and must be the one recorded."""
    if any(e.error is not None and e.error.code == "missing" for e in episodes):
        raise ValueError("a complete run has no missing episode")
    for e in episodes:
        if e.metrics is not None and not _is_time(e.metrics.get("time_upright_s")):
            raise ValueError(
                f"episode {e.index}: metrics.time_upright_s must be a number of seconds in "
                f"[0, {MAX_TIME_UPRIGHT_S:g}]"
            )
    if summarize(episodes, len(episodes)) != aggregate:
        raise ValueError("aggregate does not match the episodes (contract §12)")


def _is_time(value: object) -> bool:
    """A number (not a bool) in ``[0, MAX_TIME_UPRIGHT_S]``, so ``summarize`` can sum any
    ``MAX_EPISODES`` of them: no NaN, no infinity, and no integer too large for a float (the
    comparisons are exact)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return 0 <= value <= MAX_TIME_UPRIGHT_S


def summarize(episodes: Sequence[EpisodeResult], expected: int) -> Aggregate:
    """Aggregate over the ``expected`` episodes of a protocol (contract §12, §15).

    ``score`` is the exact sum of episode scores divided by ``expected`` and ``success_rate`` is
    successes over ``expected``: a missing or errored episode counts as 0. ``mean_time_upright_s``
    is the mean over the episodes that are present (it describes the runs that happened).
    """
    if len(episodes) > expected:
        raise ValueError(f"{len(episodes)} episodes, but only {expected} expected")
    times = [float(e.metrics["time_upright_s"]) for e in episodes if e.metrics is not None]
    score = math.fsum(e.score for e in episodes) / expected if expected else 0.0
    return Aggregate(
        score=min(max(score, 0.0), 100.0),
        success_rate=(sum(e.success for e in episodes) / expected) if expected else 0.0,
        mean_time_upright_s=math.fsum(times) / len(times) if times else 0.0,
        episodes=len(episodes),
        errors=sum(e.error is not None for e in episodes),
    )


def write_results(results: EvalResults, path: Path) -> None:
    """Atomic, durable write: readers never see a half-written file, and once this returns the
    file is on disk (the file is fsynced before the rename, its directory after, where the
    directory can be opened and synced), mode 0644 whatever the umask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            _set_mode(fh.fileno(), tmp)
            fh.write(results.model_dump_json(indent=2))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _set_mode(fd: int, path: str) -> None:
    """``RESULTS_FILE_MODE`` on the open temporary file, whatever the umask. Through the path
    where ``os.fchmod`` does not exist (Windows before Python 3.13)."""
    if hasattr(os, "fchmod"):
        os.fchmod(fd, RESULTS_FILE_MODE)
    else:
        os.chmod(path, RESULTS_FILE_MODE)


def _fsync_directory(directory: Path) -> None:
    """Make a rename in ``directory`` durable. Skipped where the directory cannot be opened
    (Windows; no read permission, as in a write-only drop directory) or its file system does not
    support syncing it; any other failure raises, after the rename has already happened."""
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        if exc.errno in _NO_DIRECTORY_OPEN:
            return
        raise
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _NO_DIRECTORY_FSYNC:
            raise
    finally:
        os.close(fd)


def read_results(path: Path) -> EvalResults:
    """Read and validate ``path``, at most ``MAX_RESULTS_BYTES`` of it.

    Validation proves a file well formed, never its numbers true: whoever publishes a score takes
    it from trusted verification (``topple.verify``, contract §16), never from an uploaded file.
    """
    with path.open("rb") as fh:
        data = fh.read(MAX_RESULTS_BYTES + 1)
    if len(data) > MAX_RESULTS_BYTES:
        raise ValueError(f"{path.name} has more than {MAX_RESULTS_BYTES} bytes")
    return EvalResults.model_validate_json(data)
