"""The CI workflow and the coverage scopes (docs/design/phase-2.md §7, §4 "Ubuntu evidence").

Three jobs: ``python`` (``-m "not slow and not train"``, coverage, ``TOPPLE_REQUIRE_CC=1``),
``python-slow`` (``-m "slow and not train"``, then the ubuntu compare of the shipped policy and
its upload) and ``python-train`` (the train extra, its own coverage scope, the legacy torch
exporter test advisory). Every action is pinned to a commit sha. The default coverage scope
omits exactly the five torch modules ``.coveragerc-train`` measures, so the two 80% scopes are
disjoint. Pushes trigger only on ``main`` until the B12 merge blocker lands.
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
SHIPPED = "hashFiles('submissions/ppo_balance/policy.onnx') != ''"


def workflow() -> dict[str, Any]:
    doc = yaml.safe_load(WORKFLOW.read_text())
    assert isinstance(doc, dict)
    return doc


def steps(job: str) -> list[dict[str, Any]]:
    found = workflow()["jobs"][job]["steps"]
    assert isinstance(found, list)
    return found


def runs(job: str) -> list[str]:
    """Each ``run:`` of ``job``, folded onto one line."""
    return [" ".join(str(s["run"]).split()) for s in steps(job) if "run" in s]


def pytest_runs(job: str) -> list[str]:
    return [r for r in runs(job) if " pytest " in f" {r} "]


def test_triggers_are_pull_requests_and_pushes_to_main() -> None:
    triggers = workflow()[True]  # YAML 1.1 reads the key `on` as true
    assert set(triggers) == {"push", "pull_request"}
    assert triggers["push"] == {"branches": ["main"]}  # phase-* joins once B12 lands


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
    assert upload["with"]["path"].split() == ["out/ci/results.json", "out/cmp_cal_ci.json"]
    assert upload["with"]["if-no-files-found"] == "error"
    assert names.index("Test (slow)") < names.index(compare["name"]) < names.index(upload["name"])
    assert workflow()["jobs"]["python-slow"]["env"]["TOPPLE_ARTIFACT_DIR"] == "out/ci"


def test_the_train_job() -> None:
    commands = runs("python-train")
    assert "uv sync --locked --extra train" in commands
    assert any("ruff check ." in c and "ruff format --check ." in c for c in commands)
    assert any(c.endswith("mypy") for c in commands)
    main, advisory = pytest_runs("python-train")
    for needed in ("-m train", "-n 2", "--cov=topple.train", "--cov-config=.coveragerc-train"):
        assert needed in main
    assert "--ignore=tests/test_pack_torch_legacy.py" in main
    assert "tests/test_pack_torch_legacy.py" in advisory
    advisory_steps = [s for s in steps("python-train") if s.get("continue-on-error")]
    assert len(advisory_steps) == 1
    assert advisory_steps[0]["continue-on-error"] is True
    assert " ".join(str(advisory_steps[0]["run"]).split()) == advisory


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
