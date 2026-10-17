"""Challenge kinds and bundle presets (docs/design/phase-2.md §8.1) and the 2b slice 1 review.

balance-v1 keeps every hash and scenario; presets resolve only through ``Challenge.preset``;
bundle presets are named by their key and stay inside the scenario parser's bounds; waypoint
configs are consistent when they load; ``BalanceEnv`` refuses another kind.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from topple.challenge import ChallengeConfig, load_challenge
from topple.disturbances import PRESETS, check_preset_bounds, make_scenario
from topple.envs.balance import BalanceEnv
from topple.evaluate import EvalProtocol

REPO = Path(__file__).resolve().parents[1]
WP_YAML = REPO / "python" / "topple" / "challenges" / "waypoint-v1" / "challenge.yaml"
# sha256 of the JSON of balance-v1/standard scenarios for seeds 1000..1019 (the merged Phase 1
# draws; the 2b slice 1 review confirmed them identical to 386df1e)
STANDARD_SCENARIOS_SHA = "29e33f5aa58a951d25d541f98e04cac794ba4852cb44915bfa7a932bebeb269b"


def _scenarios_digest() -> str:
    challenge = load_challenge("balance-v1")
    preset = challenge.preset("standard")
    docs = [
        make_scenario(seed, preset, timestep=0.002, control_dt=0.01, episode_seconds=10.0).to_dict()
        for seed in range(1000, 1020)
    ]
    return hashlib.sha256(json.dumps(docs, sort_keys=True).encode()).hexdigest()


def test_balance_standard_scenarios_are_unchanged() -> None:
    assert _scenarios_digest() == STANDARD_SCENARIOS_SHA


def test_protocols_stay_hashable() -> None:
    assert isinstance(hash(EvalProtocol.from_id("balance-v1/standard")), int)
    assert isinstance(hash(EvalProtocol.from_id("wp_eval_v1")), int)


def test_presets_resolve_only_through_the_challenge() -> None:
    """No ``PRESETS[...]`` lookup outside the modules that own preset resolution (§8.1)."""
    allowed = {"disturbances.py", "challenge.py"}
    offenders = []
    for path in (REPO / "python" / "topple").rglob("*.py"):
        if path.name in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "PRESETS"
            ):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert offenders == []


def _wp_doc() -> dict[str, object]:
    doc = yaml.safe_load(WP_YAML.read_text())
    assert isinstance(doc, dict)
    return doc


def test_a_bundle_preset_must_be_named_by_its_key() -> None:
    doc = _wp_doc()
    doc["preset_defs"]["standard"]["name"] = "standard-x"  # type: ignore[index]
    with pytest.raises(ValueError, match="is named"):
        ChallengeConfig.model_validate(doc)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("randomization", "actuator_delay_ms"), [30]),
        (("randomization", "mass_scale"), [0.1, 1.0]),
        (("pushes", "impulse_ns"), [0.15, 6.0]),
    ],
)
def test_bundle_presets_stay_inside_the_parser_bounds(path: tuple[str, str], value: object) -> None:
    doc = _wp_doc()
    std = doc["preset_defs"]["standard"]  # type: ignore[index]
    std.setdefault(path[0], {})[path[1]] = value
    with pytest.raises(ValueError, match=r"outside|exceeds"):
        ChallengeConfig.model_validate(doc)


def test_every_shipped_preset_is_inside_the_bounds() -> None:
    for preset in PRESETS.values():
        check_preset_bounds(preset)
    wp = load_challenge("waypoint-v1")
    for name in wp.config.presets:
        check_preset_bounds(wp.preset(name))


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("course", "gate_radius_m", 0.8, "gate_radius_m < leg_m"),
        ("course", "leg_m", 2.0, "leg_m <= max_radius_m"),
        ("course", "max_attempts", 10**7, "max_attempts"),
        ("score", "t_par_s", 25.0, "t_par_s"),
    ],
)
def test_waypoint_configs_are_consistent_when_they_load(
    section: str, key: str, value: object, match: str
) -> None:
    doc = _wp_doc()
    doc[section][key] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=match):
        ChallengeConfig.model_validate(doc)


def test_balance_env_refuses_another_kind() -> None:
    with pytest.raises(ValueError, match="make_env"):
        BalanceEnv("standard", challenge=load_challenge("waypoint-v1"))


def test_the_protocol_challenge_is_a_copy_safe_value() -> None:
    proto = EvalProtocol.from_id("wp_eval_v1")
    assert replace(proto, seeds=(3000,)).challenge is proto.challenge


def test_the_frozen_waypoint_scorer_and_push_ceiling() -> None:
    """§8.7 calibration on linux-x86_64: impulse_hi 0.375 N.s, E_ref 21.4 J. Every scorer
    constant is in the score config and so in protocol_hash (§8.1)."""
    wp = load_challenge("waypoint-v1")
    assert wp.config.score.model_dump(mode="json") == {
        "weights": [55.0, 25.0, 15.0, 5.0],
        "t_par_s": 5.0,
        "dock_progress_weights": [0.8, 0.2],
        "success_energy_factor": 5.0,
        "e_ref_j": 21.4,
    }
    pushes = wp.preset("standard").pushes
    assert pushes is not None
    assert pushes.impulse_ns == (0.15, 0.375)
