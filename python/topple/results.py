"""``results.json`` schema ``topple.results/1`` (docs/contract.md §15)."""

from __future__ import annotations

import errno
import itertools
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

FORMAT: Literal["topple.results/1"] = "topple.results/1"
MAX_DETAIL_CHARS = 300
MAX_ID_CHARS = 200
MAX_PLATFORM_CHARS = 64
MAX_VERSION_CHARS = 64
RESULTS_FILE_MODE = 0o644
REPLACEMENT_CHAR = "�"
_KEPT_CONTROLS = frozenset("\n\t")
_NO_DIRECTORY_FSYNC = frozenset({errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP})

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

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=MAX_ID_CHARS)]
VersionStr = Annotated[str, StringConstraints(max_length=MAX_VERSION_CHARS)]


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
    metrics: dict[str, float | int | bool | str] | None
    error: ErrorInfo | None = None


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
    id: Identifier
    sha256: Sha256


class EvalResults(_Strict):
    """One evaluation's results. ``isolation`` records how the controller ran; it does not make a
    run rankable (Phase 3 decides that from which trusted job produced the file)."""

    format: Literal["topple.results/1"] = FORMAT
    challenge: Identifier
    protocol: Identifier
    protocol_hash: Sha256
    seed_set: Literal["public", "private"]
    versions: Versions
    platform: Annotated[str, StringConstraints(max_length=MAX_PLATFORM_CHARS)]
    robot_hash: Sha256
    controller: ControllerInfo
    isolation: Isolation
    complete: bool
    episodes: list[EpisodeResult] | None
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


def write_results(results: EvalResults, path: Path) -> None:
    """Atomic, durable write: readers never see a half-written file, and once this returns the
    file is on disk (the file is fsynced before the rename, its directory after), mode 0644
    whatever the umask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fh.fileno(), RESULTS_FILE_MODE)
            fh.write(results.model_dump_json(indent=2))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    """Make a rename in ``directory`` durable. Skipped where directories cannot be opened
    (Windows) or the file system does not support syncing them; any other failure raises."""
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _NO_DIRECTORY_FSYNC:
            raise
    finally:
        os.close(fd)


def read_results(path: Path) -> EvalResults:
    return EvalResults.model_validate_json(path.read_text(encoding="utf-8"))
