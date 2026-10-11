"""Golden known-answer tests for balance-v1 (fix spec J).

These pin values that must never move once balance-v1 is published (docs/contract.md §13): the
controller seed derivation, the scenarios materialized from episode seeds, the bundle and protocol
hashes, and the baseline controllers' episode scores. A failure here means a change can move a
fixed controller's score on a fixed seed. Do not "update the numbers": either the change is a bug,
or it needs a new challenge major version (or an ``ENV_ALGO_VERSION`` bump before publication) and
a deliberate, reviewed re-pin.

Seeds, scenario draws and hashes are platform-independent (numpy's ``SeedSequence``/PCG64 and
sha256 are portable). Episode scores come from MuJoCo physics, which is only deterministic per
platform (PLAN.md, trust model), so they are pinned per platform tag.
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

BUNDLE_HASH = "4354497568c748625a73b4ddb5187fdafef1761f7f9b30636445d25493d52a70"
# Public protocols: the public seed list, salt 0 (contract §13).
PROTOCOL_HASHES = {
    "ideal": "47b2ca3748c223986f4c797c415dc8932b5b442de9a3857b8d2994f126a5baeb",
    "clean": "2fbc269d89f28f106d8d528e21d9f4db85d11def73ae7068fc83338055cf9ddc",
    "standard": "0cd7059ce234a0c1006036bb25f634f02824793969560ea7db0876eb1c7fe180",
}
PRIVATE_SEEDS = (1, 2, 3)
PRIVATE_SALT = 2**127 + 5
PRIVATE_PROTOCOL_HASH = "ac2ad94f5d48205c3dc8adf51cda625bf2ede0b0dfd910cb69f1a3f96c8cb13f"

# platform tag -> controller -> seed -> (score, success, time_upright_s), balance-v1/standard.
EpisodeKey = tuple[float, bool, float]
EPISODE_SEEDS = (1000, 1001, 1002)
EPISODES: dict[str, dict[str, dict[int, EpisodeKey]]] = {
    "darwin-arm64": {
        "pid": {
            1000: (92.10566273263854, True, 10.0),
            1001: (77.502526600719, True, 10.0),
            1002: (46.38905576742924, False, 5.97),
        },
        "lqr": {
            1000: (91.5961524979523, True, 10.0),
            1001: (84.50243299777665, True, 10.0),
            1002: (83.80593189848054, True, 10.0),
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
        context = f"{tag} {name} seed {seed}: measured {measured!r}"
        assert got[0] == pytest.approx(score, abs=SCORE_TOL, rel=0), context
        assert got[1] is success, context
        assert got[2] == upright, context
