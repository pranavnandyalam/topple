"""waypoint-v1 known answers, frozen at the end of Phase 2b (docs/design/phase-2.md §8.3, §8.9
item 1): the golden scenarios of seeds 0, 3000 and 2**53 - 1 on each preset (course included)
must come out the same on every platform, like balance-v1's (``test_golden.py``); the bundle and
protocol hashes and the waypoint env algorithm version are pinned. A change to any of them is a
``waypoint-v2`` (contract §13), not an edit here."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from topple.challenge import Challenge, env_algo_version, load_challenge
from topple.disturbances import make_scenario
from topple.evaluate import EvalProtocol, iter_episodes, platform_tag
from topple.loader import load_submission
from topple.rng import MAX_SEED

GOLDEN = Path(__file__).parent / "golden" / "scenarios_waypoint_v1.json"
SCENARIO_SEEDS = (0, 3000, MAX_SEED - 1)
PRESET_NAMES = ("clean", "standard")
BUNDLE_HASH = "7ed34a22ba18f68bc713ba629226f0d858762525fadd026dbe45548d3870812d"
PROTOCOL_HASHES = {
    "clean": "a6d443cfd1709935f2e1dc66dc691f189725b5057e9f6b54bbb4752818cdd05c",
    "standard": "8d2229df811a6e6965de43b75c931cab44bc53dc005c177351859846854e8cd3",
}
WAYPOINT_ENV_ALGO_VERSION = 1


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@pytest.fixture(scope="module")
def challenge() -> Challenge:
    return load_challenge("waypoint-v1")


@pytest.fixture(scope="module")
def golden() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return data


def test_the_golden_file_covers_every_preset_and_seed(
    challenge: Challenge, golden: dict[str, dict[str, Any]]
) -> None:
    assert set(golden) == set(PRESET_NAMES) == set(challenge.config.presets)
    for preset in PRESET_NAMES:
        assert set(golden[preset]) == {str(s) for s in SCENARIO_SEEDS}
        for seed in SCENARIO_SEEDS:
            assert golden[preset][str(seed)]["course"]  # pins the TARGETS stream too
    assert all(golden["standard"][str(s)]["pushes"] for s in SCENARIO_SEEDS)
    assert not any(golden["clean"][str(s)]["pushes"] for s in SCENARIO_SEEDS)


@pytest.mark.parametrize("seed", SCENARIO_SEEDS)
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_make_scenario_known_answers(
    challenge: Challenge, golden: dict[str, dict[str, Any]], preset: str, seed: int
) -> None:
    actual = make_scenario(
        seed,
        challenge.preset(preset),
        timestep=0.002,
        control_dt=challenge.config.control_dt,
        episode_seconds=challenge.config.episode_seconds,
        course=challenge.config.course,
    ).to_dict()
    assert canonical(actual) == canonical(golden[preset][str(seed)])


def test_bundle_hash_known_answer(challenge: Challenge) -> None:
    assert challenge.bundle_hash() == BUNDLE_HASH


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_protocol_hash_known_answers(challenge: Challenge, preset: str) -> None:
    assert challenge.protocol_hash(preset) == PROTOCOL_HASHES[preset]
    assert EvalProtocol.from_id(f"waypoint-v1/{preset}").protocol_hash() == PROTOCOL_HASHES[preset]


def test_wp_eval_v1_is_the_public_standard_protocol() -> None:
    assert EvalProtocol.from_id("wp_eval_v1").protocol_hash() == PROTOCOL_HASHES["standard"]


def test_the_waypoint_env_algo_version_is_pinned() -> None:
    assert env_algo_version("waypoint") == WAYPOINT_ENV_ALGO_VERSION


# --- reference driver episodes, per platform (gate finding: drift after the freeze) ---------------

DRIVER_SEEDS = (0, 1, 2, 63)
"""Three ordinary standard seeds and one fall (63), from the anchors comparisons."""
DRIVER_EPISODES: dict[str, dict[int, tuple[float, bool, str]]] = {
    # docs/results/phase2b/anchors/wp_anchors_darwin.json (Apple Silicon)
    "darwin-arm64": {
        0: (79.5, True, "timeout"),
        1: (83.28871800773045, True, "timeout"),
        2: (83.13900621442112, True, "timeout"),
        63: (31.656947311015543, False, "fell"),
    },
    # docs/results/phase2b/anchors/wp_anchors_linux.json (GitHub Actions ubuntu-24.04)
    "linux-x86_64": {
        0: (79.5, True, "timeout"),
        1: (83.33408901363678, True, "timeout"),
        2: (83.14087568514255, True, "timeout"),
        63: (31.642570489726804, False, "fell"),
    },
}


def test_the_reference_driver_scores_its_known_answers() -> None:
    """The waypoint scorer, env and driver are frozen with the challenge: these exact scores
    must hold on each platform (contract §13). In CI (linux) a missing pin fails, as in
    ``test_golden.py``."""
    tag = platform_tag()
    if tag not in DRIVER_EPISODES:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(f"no waypoint driver pins for {tag}")
        pytest.skip(f"no waypoint driver pins for {tag}")
    proto = replace(EvalProtocol.from_id("waypoint-v1/standard"), seeds=DRIVER_SEEDS)
    driver = Path(__file__).resolve().parents[1] / "submissions" / "waypoint_pid_example.py"
    factory = load_submission(driver, challenge=proto.challenge).factory
    for k, run in enumerate(iter_episodes(proto, factory)):
        m = run.result.metrics
        assert m is not None
        got = (run.result.score, run.result.success, m["termination"])
        assert got == DRIVER_EPISODES[tag][DRIVER_SEEDS[k]], (tag, DRIVER_SEEDS[k], got)
