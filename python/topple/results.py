"""``results.json`` schema ``topple.results/1`` (docs/contract.md §15)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FORMAT = "topple.results/1"
MAX_DETAIL_CHARS = 300

ErrorCode = Literal[
    "exception",
    "invalid_action",
    "timeout",
    "crash",
    "protocol",
    "output_limit",
    "load_error",
    "sim_unstable",
]
Isolation = Literal["inprocess", "subprocess", "container"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ErrorInfo(_Strict):
    code: ErrorCode
    step: int = Field(ge=0)
    detail: str = ""  # untrusted: produced by submitted code; escape before rendering

    @field_validator("detail")
    @classmethod
    def _truncate(cls, value: str) -> str:
        return value[:MAX_DETAIL_CHARS]


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
    topple: str
    mujoco: str
    numpy: str
    obs_schema: str
    action_spec: str


class ControllerInfo(_Strict):
    id: str
    sha256: str


class EvalResults(_Strict):
    format: Literal["topple.results/1"] = FORMAT
    challenge: str
    protocol: str
    protocol_hash: str
    seed_set: Literal["public", "private"]
    versions: Versions
    platform: str
    robot_hash: str
    controller: ControllerInfo
    isolation: Isolation
    complete: bool
    episodes: list[EpisodeResult] | None
    aggregate: Aggregate

    @property
    def rankable(self) -> bool:
        return self.complete and self.isolation == "container"


def write_results(results: EvalResults, path: Path) -> None:
    """Atomic write: readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(results.model_dump_json(indent=2))
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_results(path: Path) -> EvalResults:
    return EvalResults.model_validate_json(path.read_text(encoding="utf-8"))
