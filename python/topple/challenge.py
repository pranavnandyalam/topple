"""Challenge bundles: everything that defines a scored task, frozen in ``challenge.yaml``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, PositiveFloat, field_validator

from topple.contract import (
    BALANCE_V1_ACTION,
    FEATURES,
    OBS_SCHEMA_ID,
    RobotGeometry,
    TaskSpec,
)
from topple.disturbances import PRESETS
from topple.ids import validate_challenge_id
from topple.rng import validate_seed
from topple.robot import RobotParams, challenge_dir, load_params


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoreConfig(_Frozen):
    e_ref_j: PositiveFloat
    tilt_ref_rad: PositiveFloat
    drift_ref_m: PositiveFloat
    success_max_drift_m: PositiveFloat
    success_energy_factor: PositiveFloat


class SeedConfig(_Frozen):
    public: tuple[int, ...]

    @field_validator("public")
    @classmethod
    def _seeds(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(seeds)) != len(seeds) or not seeds:
            raise ValueError("public seeds must be non-empty and unique")
        return tuple(validate_seed(s) for s in seeds)


class ChallengeConfig(_Frozen):
    id: str
    obs_schema: str
    action_spec: str
    robot: str
    params: str
    scene: str
    control_dt: PositiveFloat
    episode_seconds: PositiveFloat
    presets: tuple[str, ...]
    score: ScoreConfig
    seeds: SeedConfig

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_challenge_id(value)

    @field_validator("presets")
    @classmethod
    def _presets(cls, presets: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(presets) - set(PRESETS)
        if unknown:
            raise ValueError(f"unknown presets {sorted(unknown)}")
        return presets


@dataclass(frozen=True)
class Challenge:
    config: ChallengeConfig
    root: Path
    params: RobotParams

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def robot_xml(self) -> Path:
        return self.root / self.config.robot

    @property
    def scene_xml(self) -> Path:
        return self.root / self.config.scene

    def task_spec(self, wheel_radius_m: float, track_width_m: float) -> TaskSpec:
        if (
            self.config.obs_schema != OBS_SCHEMA_ID
            or self.config.action_spec != BALANCE_V1_ACTION.id
        ):
            raise ValueError(f"unsupported schema/action for {self.id}")
        return TaskSpec(
            challenge=self.id,
            obs_schema=OBS_SCHEMA_ID,
            obs_features=FEATURES,
            action=BALANCE_V1_ACTION,
            control_dt=self.config.control_dt,
            episode_seconds=self.config.episode_seconds,
            robot=RobotGeometry(wheel_radius_m=wheel_radius_m, track_width_m=track_width_m),
        )

    def bundle_hash(self) -> str:
        """sha256 over every file that defines the challenge (sorted, path-qualified)."""
        digest = hashlib.sha256()
        for path in sorted(
            p for p in self.root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        ):
            digest.update(str(path.relative_to(self.root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def protocol_hash(self, preset: str) -> str:
        payload = {
            "challenge": self.bundle_hash(),
            "preset": PRESETS[preset].model_dump(mode="json"),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_challenge_dir(root: Path) -> Challenge:
    config = ChallengeConfig.model_validate(yaml.safe_load((root / "challenge.yaml").read_text()))
    return Challenge(config=config, root=root, params=load_params(root / config.params))


@lru_cache(maxsize=8)
def load_challenge(challenge_id: str) -> Challenge:
    """Load a built-in challenge bundle by id (e.g. ``balance-v1``)."""
    return load_challenge_dir(challenge_dir(validate_challenge_id(challenge_id)))
