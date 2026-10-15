"""G9: CI succeeded on HEAD (docs/design/phase-2.md §4).

    uv run python scripts/phase2_ci_check.py --head "$(git rev-parse HEAD)" --workflow ci.yml

Lists the workflow's runs on the current branch with ``gh run list --branch <branch> --workflow
ci.yml --json databaseId,status,conclusion,headSha`` and judges the newest run whose
``headSha`` is HEAD. It passes only on a completed ``success`` run whose three jobs
(``python``, ``python-slow``, ``python-train``) each succeeded (``gh run view <id> --json
jobs``); otherwise it fails (exit 1) with one of four distinct messages: no run for HEAD, in
progress, cancelled (a concurrency ``cancel-in-progress`` replaces a superseded run), failed;
or names the job that did not succeed. ``gh`` runs with a timeout and a cap on its output; an
answer it cannot give (gh missing, failing, or printing something else than the expected JSON)
exits 2. Nothing it prints is echoed unfiltered.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from typing import Annotated, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

from phase2_common import CommandError, printable, run_command

REQUIRED_JOBS: Final = ("python", "python-slow", "python-train")
NO_RUN: Final = "no CI run for HEAD"
IN_PROGRESS: Final = "the CI run for HEAD is still in progress"
CANCELLED: Final = "the CI run for HEAD was cancelled"
FAILED: Final = "the CI run for HEAD did not succeed"
GH_TIMEOUT_S: Final = 60.0
MAX_GH_BYTES: Final = 1 << 20
RUN_LIMIT: Final = 100
"""Runs listed per query: far more than a branch gets between two pushes."""
SHA: Final = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")

Word = Annotated[str, StringConstraints(max_length=64)]


class _Gh(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class GhRun(_Gh):
    database_id: int = Field(alias="databaseId", ge=0)
    status: Word
    conclusion: Word
    head_sha: Word = Field(alias="headSha")


class GhJob(_Gh):
    name: Annotated[str, StringConstraints(max_length=256)]
    status: Word
    conclusion: Word


class GhJobs(_Gh):
    jobs: Annotated[list[GhJob], Field(max_length=1024)]


RUN_LIST: Final = TypeAdapter(Annotated[list[GhRun], Field(max_length=1000)])
"""``gh run list --json ...``: a JSON array of runs."""


class GhError(Exception):
    """gh could not answer: not installed, failing, too slow or not the expected JSON."""


def gh_json(args: Sequence[str]) -> bytes:
    try:
        result = run_command(["gh", *args], timeout_s=GH_TIMEOUT_S, max_bytes=MAX_GH_BYTES)
    except CommandError as exc:
        raise GhError(str(exc)) from None
    if result.returncode != 0:
        raise GhError(
            f"gh {' '.join(args[:2])} failed (exit {result.returncode}): "
            f"{result.stderr.strip()[:300]}"
        )
    return result.stdout


def list_runs(branch: str, workflow: str) -> list[GhRun]:
    data = gh_json(
        [
            "run",
            "list",
            "--branch",
            branch,
            "--workflow",
            workflow,
            "--json",
            "databaseId,status,conclusion,headSha",
            "--limit",
            str(RUN_LIMIT),
        ]
    )
    try:
        return RUN_LIST.validate_json(data, strict=True)
    except ValidationError:
        raise GhError("gh run list did not print a list of runs") from None


def judge(runs: Sequence[GhRun], head: str) -> tuple[int, str, GhRun | None]:
    """(exit code, message, the run judged): the newest run for ``head`` (largest
    ``databaseId``) must be completed with ``success``."""
    mine = [r for r in runs if r.head_sha == head]
    if not mine:
        return (
            1,
            f"{NO_RUN} {head[:12]} on this branch (push it, or wait for the run to start)",
            None,
        )
    run = max(mine, key=lambda r: r.database_id)
    rid = f"run {run.database_id}"
    if run.status != "completed":
        return 1, f"{IN_PROGRESS} ({rid}, status {printable(run.status, 32)!r}); wait for it", run
    if run.conclusion == "cancelled":
        return 1, f"{CANCELLED} ({rid}); re-run it", run
    if run.conclusion != "success":
        return 1, f"{FAILED} ({rid}, conclusion {printable(run.conclusion, 32)!r})", run
    return 0, f"CI {rid} for HEAD {head[:12]} succeeded", run


def jobs_problems(run_id: int) -> list[str]:
    """Each required job that is missing from the run or did not succeed."""
    try:
        jobs = GhJobs.model_validate_json(gh_json(["run", "view", str(run_id), "--json", "jobs"]))
    except ValidationError:
        raise GhError("gh run view did not print the run's jobs") from None
    by_name = {job.name: job for job in jobs.jobs}
    problems = []
    for name in REQUIRED_JOBS:
        job = by_name.get(name)
        if job is None:
            problems.append(f"run {run_id} has no {name} job")
        elif (job.status, job.conclusion) != ("completed", "success"):
            problems.append(
                f"run {run_id}: job {name} is {printable(job.status, 32)}/"
                f"{printable(job.conclusion, 32) or 'none'}, not completed/success"
            )
    return problems


def current_branch() -> str:
    try:
        result = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout_s=30.0)
    except CommandError as exc:
        raise GhError(str(exc)) from None
    branch = result.stdout.decode("utf-8", "replace").strip()
    if result.returncode != 0 or not branch or branch == "HEAD":
        raise GhError("cannot tell the current branch (detached HEAD?); pass --branch")
    return branch


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase2_ci_check.py", description="G9: CI succeeded on HEAD (phase-2.md section 4)."
    )
    parser.add_argument("--head", required=True, help="the commit CI must have passed (HEAD)")
    parser.add_argument("--workflow", required=True, help="the workflow file (ci.yml)")
    parser.add_argument("--branch", default=None, help="default: the current branch")
    args = parser.parse_args(argv)
    if not SHA.fullmatch(args.head):
        print("phase2_ci_check: --head must be a full commit sha", file=sys.stderr)
        return 2
    try:
        branch = args.branch or current_branch()
        code, message, run = judge(list_runs(branch, args.workflow), args.head)
        problems = [] if code or run is None else jobs_problems(run.database_id)
    except GhError as exc:
        print(f"phase2_ci_check: {printable(str(exc))}", file=sys.stderr)
        return 2
    if code:
        print(f"phase2_ci_check: {message}", file=sys.stderr)
        return code
    if problems:
        for problem in problems:
            print(f"phase2_ci_check: {FAILED}: {printable(problem)}", file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
