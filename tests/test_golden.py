"""Golden known-answer tests for balance-v1 (fix spec J).

These pin values that must never move once balance-v1 is published (docs/contract.md §13): the
controller seed derivation, the scenarios materialized from episode seeds, the bundle and protocol
hashes, and the baseline controllers' episode scores. A failure here means a change can move a
fixed controller's score on a fixed seed. Do not "update the numbers": either the change is a bug,
or it is deliberate and needs a reviewed re-pin: before publication the re-pin itself (with an
``ENV_ALGO_VERSION`` bump if env code changed, as balance-v1's E_ref recalibration was done),
after publication a new challenge major version (contract §13).

Seeds, scenario draws and hashes are platform-independent (numpy's ``SeedSequence``/PCG64 raw
bits and sha256 are portable, and scenario draws apply their affine maps in Python; see
``topple.disturbances._uniforms``), so the scenario file must pass unchanged on every platform.
Episode scores come from MuJoCo physics, which is only deterministic per platform (PLAN.md, trust
model), so they are pinned per platform tag; a platform without pins skips and prints what it
measured, except in CI.

Who enforces which pins: CI (GitHub Actions ubuntu-24.04, ``linux-x86_64``, the platform ranked
scores come from) checks the linux-x86_64 scores on every push. In CI (``GITHUB_ACTIONS=true``)
a platform without pins fails instead of skipping, and so does any platform but linux-x86_64: a
runner moved to another platform would otherwise pass while enforcing no pin, or not the ranked
platform's. The ``darwin-arm64`` scores, the development platform's, are checked only when the
suite runs on an Apple Silicon Mac; no CI job runs there. So a deliberate re-pin (after a
reviewed physics change, or a dependency bump that moves scores) updates every platform in
``EPISODES`` in the same change: the darwin-arm64 values from a local run of this file, the
linux-x86_64 values from the CI failure message. Both print every measured value. Re-pinning
only the platform that failed would leave the other stale.
"""

from __future__ import annotations

import json
import os
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

BUNDLE_HASH = "89a62007e2f56e61b3a87f3e30034dafc5e6bde4d6c675c07cd79c8d0bbaedd0"
# Public protocols: the public seed list, salt 0 (contract §13).
PROTOCOL_HASHES = {
    "ideal": "79450098ce9ad069f7d812e451ff1403e38d3e597fd7166bcb643b66d309c30c",
    "clean": "3f27d1c91d9e2eaaacddf4609b47c6d582342a6bc23b31524751c679f406e3c2",
    "standard": "b243606351363f966776bd618f15df12dfdaf7103fcc67e87864139436bd1ea3",
}
PRIVATE_SEEDS = (1, 2, 3)
PRIVATE_SALT = 2**127 + 5
PRIVATE_PROTOCOL_HASH = "8db2d33e96f576cc2e2f62bd4815c8f156097450a17098b09de28ad0592ea193"

RANKED_PLATFORM = "linux-x86_64"
"""Where CI runs and ranked scores come from (contract §13)."""

# platform tag -> controller -> seed -> (score, success, time_upright_s), balance-v1/standard.
EpisodeKey = tuple[float, bool, float]
EPISODE_SEEDS = (1000, 1001, 1002)
EPISODES: dict[str, dict[str, dict[int, EpisodeKey]]] = {
    # The development platform (Apple Silicon), re-pinned under E_ref 13.1 J on 2026-10-15.
    "darwin-arm64": {
        "pid": {
            1000: (92.07486036832013, True, 10.0),
            1001: (77.65148730115902, True, 10.0),
            1002: (46.327503731785804, False, 5.97),
        },
        "lqr": {
            1000: (91.54282414455297, True, 10.0),
            1001: (84.11681322780457, True, 10.0),
            1002: (83.76543245729982, True, 10.0),
        },
    },
    # The CI platform (GitHub Actions ubuntu-24.04), the source of truth for ranked scores.
    "linux-x86_64": {
        "pid": {
            1000: (92.08753235441824, True, 10.0),
            1001: (77.66275112229928, True, 10.0),
            1002: (46.4017119240978, False, 5.98),
        },
        "lqr": {
            1000: (91.54233637656888, True, 10.0),
            1001: (85.43305302210213, True, 10.0),
            1002: (83.81544021121424, True, 10.0),
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


def in_ci() -> bool:
    """GitHub Actions sets ``GITHUB_ACTIONS=true`` in every step."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


def pins_for(tag: str, name: str, measured: dict[int, EpisodeKey]) -> dict[int, EpisodeKey]:
    """``name``'s pinned scores on platform ``tag``. Without pins: skip and print what was
    measured, except in CI, where a skip would pass the job while enforcing no pin: fail."""
    if tag in EPISODES:
        return EPISODES[tag][name]
    message = f"no golden episode scores for {tag}; measured {name}: {measured!r}"
    if in_ci():
        pytest.fail(f"{message}. CI must run on {RANKED_PLATFORM}, whose scores are pinned")
    pytest.skip(message)


def check_ci_platform(tag: str) -> None:
    """In CI, ``tag`` must be the platform ranked scores come from, so that CI enforces its
    pins; outside CI there is nothing to check."""
    if in_ci():
        assert tag == RANKED_PLATFORM, (
            f"CI runs on {tag}, not {RANKED_PLATFORM}: its golden scores would go unchecked "
            "(contract §13); move the job back or change the ranked platform deliberately"
        )


def test_ci_runs_on_the_ranked_platform() -> None:
    check_ci_platform(platform_tag())


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_baseline_episode_scores_known_answers(name: str) -> None:
    tag = platform_tag()
    measured = _measure(name)
    expected = pins_for(tag, name, measured)
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


UNPINNED = "linux-aarch64"  # a GitHub-hosted arm runner, which has no pins


@pytest.mark.parametrize(
    ("ci", "outcome"), [(True, pytest.fail.Exception), (False, pytest.skip.Exception)]
)
def test_an_unpinned_platform_fails_in_ci_and_skips_elsewhere(
    monkeypatch: pytest.MonkeyPatch, ci: bool, outcome: type[BaseException]
) -> None:
    """In CI a skip would pass the job while enforcing no score pin at all."""
    if ci:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
    else:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(outcome, match=f"no golden episode scores for {UNPINNED}; measured pid"):
        pins_for(UNPINNED, "pid", {})


def test_a_pinned_platform_returns_its_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert pins_for("linux-x86_64", "lqr", {}) is EPISODES["linux-x86_64"]["lqr"]


@pytest.mark.parametrize(
    ("ci", "tag", "ok"),
    [
        (True, "linux-x86_64", True),
        (True, "darwin-arm64", False),  # pinned, but not the platform ranked scores come from
        (True, UNPINNED, False),
        (False, UNPINNED, True),  # outside CI there is nothing to check
    ],
)
def test_ci_must_run_on_the_ranked_platform(
    monkeypatch: pytest.MonkeyPatch, ci: bool, tag: str, ok: bool
) -> None:
    if ci:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
    else:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    if ok:
        check_ci_platform(tag)
    else:
        with pytest.raises(AssertionError, match=f"CI runs on {tag}, not linux-x86_64"):
            check_ci_platform(tag)
