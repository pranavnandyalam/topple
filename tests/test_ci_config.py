"""The CI workflow and the coverage scopes (docs/design/phase-2.md §7, §4 "Ubuntu evidence").

Three jobs: ``python`` (``-m "not slow and not train"``, coverage, ``TOPPLE_REQUIRE_CC=1``),
``python-slow`` (``-m "slow and not train"``, then the ubuntu compare of the shipped policy, a
check that both evidence files exist and their upload) and ``python-train`` (the train extra,
its own coverage scope once ``topple/train/learn.py`` exists, the legacy torch exporter test
advisory). Every action is pinned to a commit sha; the token is read-only and no checkout keeps
it. The default coverage scope omits exactly the five torch modules ``.coveragerc-train``
measures, so the two 80% scopes are disjoint, and both exclude the same lines.
"""

from __future__ import annotations

import configparser
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
TRAIN_MODULES = ("learn", "snapshot", "callbacks", "surgery", "resources")
TRAIN_PATTERNS = [f"*/topple/train/{m}.py" for m in TRAIN_MODULES]
SHIPPED_COMPARE = (
    "uv run topple compare submissions/pid_example.py submissions/lqr_example.py "
    "submissions/ppo_balance -p balance-v1/standard --seeds 0..199 --workers 4 "
    "--json out/cmp_cal_ci.json"
)
# Only push runs carry evidence (§4): a pull request's run executes its own code, which need not
# be the code the policy is shipped with.
SHIPPED = "github.event_name == 'push' && hashFiles('submissions/ppo_balance/policy.onnx') != ''"
LEARN = "hashFiles('python/topple/train/learn.py')"
TRAIN_COV = "--cov=topple.train --cov-config=.coveragerc-train --cov-report=term-missing"


def workflow() -> dict[str, Any]:
    doc = yaml.safe_load(WORKFLOW.read_text())
    assert isinstance(doc, dict)
    return doc


def steps(job: str) -> list[dict[str, Any]]:
    found = workflow()["jobs"][job]["steps"]
    assert isinstance(found, list)
    return found


def fold(run: object) -> str:
    """A ``run:`` folded onto one line."""
    return " ".join(str(run).split())


def runs(job: str) -> list[str]:
    """Each ``run:`` of ``job``, folded onto one line."""
    return [fold(s["run"]) for s in steps(job) if "run" in s]


def pytest_runs(job: str) -> list[str]:
    return [r for r in runs(job) if " pytest " in f" {r} "]


def test_triggers_are_pull_requests_and_pushes_to_main_and_phase_branches() -> None:
    triggers = workflow()[True]  # YAML 1.1 reads the key `on` as true
    assert set(triggers) == {"push", "pull_request"}
    assert triggers["push"] == {"branches": ["main", "phase-*"]}


def test_the_three_jobs() -> None:
    jobs = workflow()["jobs"]
    assert set(jobs) == {"python", "python-slow", "python-train"}
    assert all(job["runs-on"] == "ubuntu-24.04" for job in jobs.values())
    assert jobs["python-train"]["timeout-minutes"] == 20


def test_the_default_job_requires_cc_and_skips_slow_and_train() -> None:
    job = workflow()["jobs"]["python"]
    assert job["env"]["TOPPLE_REQUIRE_CC"] == "1"
    (test,) = pytest_runs("python")
    assert '-m "not slow and not train"' in test
    assert "--cov" in test.split()


def test_the_slow_job_runs_the_slow_tests_then_the_ubuntu_evidence() -> None:
    (test,) = pytest_runs("python-slow")
    assert '-m "slow and not train"' in test
    assert "--cov" not in test
    names = [s.get("name", "") for s in steps("python-slow")]
    compare = next(s for s in steps("python-slow") if "topple compare" in str(s.get("run", "")))
    assert " ".join(str(compare["run"]).split()) == SHIPPED_COMPARE
    assert compare["if"] == SHIPPED
    upload = next(s for s in steps("python-slow") if "upload-artifact" in str(s.get("uses", "")))
    assert upload["if"] == SHIPPED
    evidence = ["out/ci/results.json", "out/cmp_cal_ci.json"]
    assert upload["with"]["path"].split() == evidence
    assert upload["with"]["if-no-files-found"] == "error"
    # if-no-files-found fires only when no path matches at all, so each file is checked first,
    # one simple command per line (bash -e stops on the first that fails).
    check = next(s for s in steps("python-slow") if "test -f" in str(s.get("run", "")))
    assert check["if"] == SHIPPED
    assert str(check["run"]).splitlines() == [f"test -f {path}" for path in evidence]
    order = [names.index(s["name"]) for s in (compare, check, upload)]
    assert names.index("Test (slow)") < order[0] < order[1] < order[2]
    assert workflow()["jobs"]["python-slow"]["env"]["TOPPLE_ARTIFACT_DIR"] == "out/ci"


def test_the_train_job() -> None:
    commands = runs("python-train")
    assert "uv sync --locked --extra train" in commands
    assert any("ruff check ." in c and "ruff format --check ." in c for c in commands)
    assert any(c.endswith("mypy") for c in commands)
    tests = [s for s in steps("python-train") if " pytest " in f" {fold(s.get('run', ''))} "]
    scoped, interim, advisory = tests
    # The 80% train scope measures the five torch modules: until learn.py exists none of them
    # does, coverage reports "No data to report" as 0% and the job would fail on every push
    # (decision log B37), so the same tests run without the scope until then.
    assert scoped["if"] == f"{LEARN} != ''"
    assert interim["if"] == f"{LEARN} == ''"
    for needed in ("-m train", "-n 2", "--ignore=tests/test_pack_torch_legacy.py"):
        assert needed in fold(scoped["run"])
    assert fold(scoped["run"]) == f"{fold(interim['run'])} {TRAIN_COV}"
    assert "--cov" not in fold(interim["run"])
    assert "tests/test_pack_torch_legacy.py" in fold(advisory["run"])
    assert [s for s in steps("python-train") if s.get("continue-on-error")] == [advisory]
    assert advisory["continue-on-error"] is True
    assert advisory["if"] == "hashFiles('tests/test_pack_torch_legacy.py') != ''"


def test_the_token_is_read_only_and_never_persisted() -> None:
    """The workflow runs pull-request code (HANDOFF: CI workflows are trust-boundary code)."""
    doc = workflow()
    assert doc["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in doc["jobs"].values())
    checkouts = [
        s
        for job in doc["jobs"]
        for s in steps(job)
        if str(s.get("uses", "")).startswith("actions/checkout@")
    ]
    assert len(checkouts) == len(doc["jobs"])
    assert all(s["with"] == {"persist-credentials": False} for s in checkouts)
    # phase2_ci_check.py's "cancelled" message names this setting.
    assert doc["concurrency"] == {"group": "ci-${{ github.ref }}", "cancel-in-progress": True}


def test_every_action_is_pinned_to_a_commit() -> None:
    uses = re.findall(r"uses:\s*(\S+)(.*)", WORKFLOW.read_text())
    assert uses
    for action, comment in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), action
        assert re.fullmatch(r"\s*# v\d+\.\d+\.\d+", comment), (action, comment)


def test_the_coverage_scopes_are_disjoint_and_cover_the_torch_modules() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    run = pyproject["tool"]["coverage"]["run"]
    assert run["source"] == ["topple"]
    assert run["omit"] == TRAIN_PATTERNS
    assert pyproject["tool"]["coverage"]["report"]["fail_under"] == 80
    train = configparser.ConfigParser()
    train.read(REPO / ".coveragerc-train")
    assert train["report"]["include"].split() == TRAIN_PATTERNS
    assert "include" not in train["run"]  # ignored when --cov sets the source
    assert train["report"].getint("fail_under") == 80
    assert train["run"].getboolean("branch")
    # Both 80% scopes count the same lines: --cov-config replaces pyproject's settings.
    default_report = pyproject["tool"]["coverage"]["report"]
    assert train["report"]["exclude_also"].strip().splitlines() == default_report["exclude_also"]
    assert train["run"].getboolean("branch") == run["branch"]
