"""waypoint-v1 known answers, frozen at the end of Phase 2b (docs/design/phase-2.md §8.3, §8.9
item 1): the golden scenarios of seeds 0, 3000 and 2**53 - 1 on each preset (course included)
must come out the same on every platform, like balance-v1's (``test_golden.py``); the bundle and
protocol hashes and the waypoint env algorithm version are pinned. A change to any of them is a
``waypoint-v2`` (contract §13), not an edit here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from topple.challenge import Challenge, env_algo_version, load_challenge
from topple.disturbances import make_scenario
from topple.evaluate import EvalProtocol
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
