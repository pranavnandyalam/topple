"""Golden known-answer tests for balance-v1 (fix spec J).

These pin values that must never move once balance-v1 is published (docs/contract.md §13): the
controller seed derivation, the scenarios materialized from episode seeds, the bundle and protocol
hashes, and the baseline controllers' episode scores. A failure here means a change can move a
fixed controller's score on a fixed seed. Do not "update the numbers": either the change is a bug,
or it needs a new challenge major version (or an ``ENV_ALGO_VERSION`` bump before publication) and
a deliberate, reviewed re-pin.

Seeds, scenario draws and hashes are platform-independent (numpy's ``SeedSequence``/PCG64 raw
bits and sha256 are portable, and scenario draws apply their affine maps in Python; see
``topple.disturbances._uniforms``), so the scenario file must pass unchanged on every platform.
Episode scores come from MuJoCo physics, which is only deterministic per platform (PLAN.md, trust
model), so they are pinned per platform tag; a platform without pins skips and prints what it
measured.

Who enforces which pins: CI (GitHub Actions ubuntu-24.04, ``linux-x86_64``, the platform ranked
scores come from) checks the linux-x86_64 scores on every push. The ``darwin-arm64`` scores, the
development platform's, are checked only when the suite runs on an Apple Silicon Mac; no CI job
runs there. So a deliberate re-pin (after a reviewed physics change, or a dependency bump that
moves scores) updates every platform in ``EPISODES`` in the same change: the darwin-arm64 values
from a local run of this file, the linux-x86_64 values from the CI failure message. Both print
every measured value. Re-pinning only the platform that failed would leave the other stale.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from topple.challenge import ENV_ALGO_VERSION, Challenge, load_challenge
from topple.controllers.lqr import LQRController
from topple.controllers.pid import PIDController
from topple.disturbances import PRESETS, make_scenario
from topple.evaluate import ControllerFactory, EvalProtocol, evaluate, platform_tag
from topple.results import ControllerInfo
from topple.rng import MAX_SEED, controller_seed

GOLDEN_DIR = Path(__file__).parent / "golden"
SCENARIO_SEEDS = (0, 1000, MAX_SEED - 1)
PRESET_NAMES = ("ideal", "clean", "standard")
SCORE_TOL = 1e-9

# (episode_seed, salt) -> controller_seed (contract §9).
CONTROLLER_SEEDS: dict[tuple[int, int], int] = {
    (0, 0): 4901256600916046,
    (1, 0): 6613977791233682,
    (1000, 0): 4276227377776896,
    (1019, 0): 6315642410118544,
    (MAX_SEED - 1, 0): 1432483101498164,
    (1000, 1): 2203642777153128,
    (1000, 2**128 - 1): 4108420700505304,
    (0, 2**127 + 12345): 8639439415161446,
}

BUNDLE_HASH = "a3b4846310a1a174f288da82d9532fe986a4b0428ab5069d1620e1237f5017a9"
# Public protocols: the public seed list, salt 0 (contract §13).
PROTOCOL_HASHES = {
    "ideal": "6ad4cf45adfef1b813229aec1e0b899c99f8e4d7953031e70ac8bf27093cb368",
    "clean": "f30f5119c86f1ef9196d4f4de92cf1b4fc5d7d4204458c8ffe2e5efb6f4194d6",
    "standard": "a32517413cace0c73c300cddbaf6e480ab78a01ee68b717b67adf0fe7f59c5f4",
}
PRIVATE_SEEDS = (1, 2, 3)
PRIVATE_SALT = 2**127 + 5
PRIVATE_PROTOCOL_HASH = "aa7220e396ffefda76e044adbeb8f496d7bb12a45fa36c81e55e069b180c749e"

# platform tag -> controller -> seed -> (score, success, time_upright_s), balance-v1/standard.
EpisodeKey = tuple[float, bool, float]
EPISODE_SEEDS = (1000, 1001, 1002)
EPISODES: dict[str, dict[str, dict[int, EpisodeKey]]] = {
    "darwin-arm64": {
        "pid": {
            1000: (92.05698981231265, True, 10.0),
            1001: (77.65148730115902, True, 10.0),
            1002: (46.296017498043454, False, 5.97),
        },
        "lqr": {
            1000: (91.5273532508896, True, 10.0),
            1001: (84.07772455809453, True, 10.0),
            1002: (83.74288560556353, True, 10.0),
        },
    },
    # The CI platform (GitHub Actions ubuntu-24.04), the source of truth for ranked scores.
    "linux-x86_64": {
        "pid": {
            1000: (92.06970433366325, True, 10.0),
            1001: (77.66275112229928, True, 10.0),
            1002: (46.37032752230015, False, 5.98),
        },
        "lqr": {
            1000: (91.52687435124143, True, 10.0),
            1001: (85.40370568144897, True, 10.0),
            1002: (83.79327852535813, True, 10.0),
        },
    },
}

FACTORIES: dict[str, ControllerFactory] = {
    "pid": lambda _spec: PIDController(),
    "lqr": lambda _spec: LQRController(),
}


def canonical(value: object) -> str:
    """Canonical JSON (sorted keys): ``1``, ``1.0`` and ``true`` all differ, unlike ``==``."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@pytest.fixture(scope="module")
def challenge() -> Challenge:
    return load_challenge("balance-v1")


@pytest.fixture(scope="module")
def golden_scenarios() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = json.loads(
        (GOLDEN_DIR / "scenarios_balance_v1.json").read_text(encoding="utf-8")
    )
    return data


# ----------------------------------------------------------------------------- controller seed


@pytest.mark.parametrize(("key", "expected"), sorted(CONTROLLER_SEEDS.items()))
def test_controller_seed_known_answers(key: tuple[int, int], expected: int) -> None:
    seed, salt = key
    assert controller_seed(seed, salt) == expected


def test_controller_seed_known_answers_fit_a_json_number() -> None:
    assert all(0 <= value < MAX_SEED for value in CONTROLLER_SEEDS.values())


# --------------------------------------------------------------------------------- scenarios


def test_golden_scenarios_cover_every_preset_and_seed(
    golden_scenarios: dict[str, dict[str, Any]],
) -> None:
    assert set(golden_scenarios) == set(PRESET_NAMES) == set(PRESETS)
    for preset in PRESET_NAMES:
        assert set(golden_scenarios[preset]) == {str(s) for s in SCENARIO_SEEDS}


@pytest.mark.parametrize("seed", SCENARIO_SEEDS)
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_make_scenario_known_answers(
    challenge: Challenge, golden_scenarios: dict[str, dict[str, Any]], preset: str, seed: int
) -> None:
    actual = make_scenario(
        seed,
        PRESETS[preset],
        timestep=0.002,
        control_dt=challenge.config.control_dt,
        episode_seconds=challenge.config.episode_seconds,
    ).to_dict()
    assert canonical(actual) == canonical(golden_scenarios[preset][str(seed)])


def test_standard_golden_scenarios_have_pushes(
    golden_scenarios: dict[str, dict[str, Any]],
) -> None:
    """Guards the fixture itself: a regenerated file without pushes would pin nothing useful."""
    for seed in SCENARIO_SEEDS:
        assert golden_scenarios["standard"][str(seed)]["pushes"]
        assert not golden_scenarios["clean"][str(seed)]["pushes"]


# ------------------------------------------------------------------------------------ hashes


def test_env_algo_version_is_pinned() -> None:
    assert ENV_ALGO_VERSION == 1


def test_bundle_hash_known_answer(challenge: Challenge) -> None:
    assert challenge.bundle_hash() == BUNDLE_HASH


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_protocol_hash_known_answers(challenge: Challenge, preset: str) -> None:
    assert challenge.protocol_hash(preset) == PROTOCOL_HASHES[preset]
    protocol = EvalProtocol.from_id(f"balance-v1/{preset}")
    assert protocol.protocol_hash() == PROTOCOL_HASHES[preset]


def test_private_protocol_hash_known_answer(challenge: Challenge) -> None:
    assert (
        challenge.protocol_hash("standard", seeds=PRIVATE_SEEDS, salt=PRIVATE_SALT)
        == PRIVATE_PROTOCOL_HASH
    )


# ---------------------------------------------------------------------------- episode scores


def _measure(name: str) -> dict[int, EpisodeKey]:
    protocol = replace(EvalProtocol.from_id("balance-v1/standard"), seeds=EPISODE_SEEDS)
    results = evaluate(protocol, FACTORIES[name], ControllerInfo(id=name, sha256="0" * 64))
    measured: dict[int, EpisodeKey] = {}
    for episode in results.episodes or []:
        assert episode.seed is not None
        assert episode.metrics is not None
        measured[episode.seed] = (
            episode.score,
            episode.success,
            float(episode.metrics["time_upright_s"]),
        )
    return measured


def test_every_platform_pins_the_same_episodes() -> None:
    """A re-pin cannot drop a controller or a seed on one platform."""
    assert set(EPISODES) == {"darwin-arm64", "linux-x86_64"}
    for table in EPISODES.values():
        assert set(table) == set(FACTORIES)
        assert all(set(scores) == set(EPISODE_SEEDS) for scores in table.values())


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_baseline_episode_scores_known_answers(name: str) -> None:
    tag = platform_tag()
    measured = _measure(name)
    if tag not in EPISODES:
        pytest.skip(f"no golden episode scores for {tag}; measured {name}: {measured!r}")
    expected = EPISODES[tag][name]
    assert set(measured) == set(expected) == set(EPISODE_SEEDS)
    for seed, (score, success, upright) in expected.items():
        got = measured[seed]
        context = (
            f"{tag} {name} seed {seed}: measured {measured!r}. A deliberate re-pin updates "
            "every platform in EPISODES (module docstring)"
        )
        assert got[0] == pytest.approx(score, abs=SCORE_TOL, rel=0), context
        assert got[1] is success, context
        assert got[2] == upright, context
