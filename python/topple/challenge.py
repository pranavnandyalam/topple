"""Challenge bundles: everything that defines a scored task, frozen in ``challenge.yaml``."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, PositiveFloat, field_validator, model_validator

from topple.contract import (
    BALANCE_V1_ACTION,
    OBS_SCHEMA_ID,
    TARGET_OBS_SCHEMA_ID,
    RobotGeometry,
    TaskSpec,
    obs_features,
    validate_schema_id,
)
from topple.disturbances import PRESETS, Preset
from topple.ids import validate_challenge_id
from topple.rng import validate_seed
from topple.robot import RobotParams, challenge_dir, hashed_bytes, load_params, referenced_files

# Version of the env algorithm (sensing, estimation, termination, pushes, model overrides). Bump it
# whenever a code change can change a fixed controller's results on a fixed seed (contract §13):
# it is part of every protocol_hash, so results from different env algorithms never compare.
ENV_ALGO_VERSION = 1
BUNDLE_HASH_TAG = b"topple.bundle/1\n"
PROTOCOL_HASH_FORMAT = "topple.protocol/1"

ChallengeKind = Literal["balance", "waypoint"]
KIND_OBS_SCHEMA: dict[str, str] = {"balance": OBS_SCHEMA_ID, "waypoint": TARGET_OBS_SCHEMA_ID}
"""The observation schema each challenge kind's env emits (docs/design/phase-2.md §8.1)."""

# waypoint-v1's env algorithm version (§8.1): bumped by every course-generator, event or
# observation change before the 2b freeze; recorded in phase2.json at the freeze.
WAYPOINT_ENV_ALGO_VERSION = 1


def env_algo_version(kind: str) -> int:
    """The env algorithm version of a challenge kind, read at call time (§8.1): balance's is the
    module attribute ``ENV_ALGO_VERSION`` (monkeypatching it must change ``protocol_hash``)."""
    if kind == "balance":
        return ENV_ALGO_VERSION
    if kind == "waypoint":
        return WAYPOINT_ENV_ALGO_VERSION
    raise ValueError(f"unknown challenge kind {kind!r}")


def score_model(kind: str) -> type[BaseModel]:
    """The score config model of a challenge kind: every scorer constant lives in it, so all of
    them are in ``protocol_hash`` (§8.1)."""
    if kind == "balance":
        return ScoreConfig
    if kind == "waypoint":
        from topple.waypoint import WaypointScore

        return WaypointScore
    raise ValueError(f"unknown challenge kind {kind!r}")


def _framed(data: bytes) -> bytes:
    return len(data).to_bytes(8, "big") + data


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
    kind: ChallengeKind = "balance"
    obs_schema: str
    action_spec: str
    robot: str
    params: str
    scene: str
    control_dt: PositiveFloat
    episode_seconds: PositiveFloat
    presets: tuple[str, ...]
    score: Any  # the kind's score model (ScoreConfig for balance), validated below
    seeds: SeedConfig
    preset_defs: dict[str, Preset] = {}
    """Bundle-defined presets (§8.1): a name here shadows ``PRESETS``."""
    course: Any = None  # waypoint: its CourseConfig (topple.course), validated below

    @model_validator(mode="before")
    @classmethod
    def _kind_models(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        kind = data.get("kind", "balance")
        out = dict(data)
        if kind in ("balance", "waypoint") and "score" in out:
            out["score"] = score_model(kind).model_validate(out["score"])
        if kind == "waypoint":
            from topple.course import CourseConfig

            if out.get("course") is None:
                raise ValueError("a waypoint challenge needs a course")
            out["course"] = CourseConfig.model_validate(out["course"])
        elif out.get("course") is not None:
            raise ValueError(f"a {kind} challenge takes no course")
        return out

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_challenge_id(value)

    @field_validator("obs_schema", "action_spec")
    @classmethod
    def _schema(cls, value: str) -> str:
        return validate_schema_id(value)

    @model_validator(mode="after")
    def _presets(self) -> ChallengeConfig:
        unknown = set(self.presets) - set(PRESETS) - set(self.preset_defs)
        if unknown:
            raise ValueError(f"unknown presets {sorted(unknown)}")
        if self.obs_schema != KIND_OBS_SCHEMA[self.kind]:
            raise ValueError(f"a {self.kind} challenge emits {KIND_OBS_SCHEMA[self.kind]}")
        return self


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

    @property
    def kind(self) -> str:
        return self.config.kind

    def preset(self, name: str) -> Preset:
        """The preset ``name`` of this challenge (§8.1): the bundle's own definition if it has
        one, else ``PRESETS[name]``; only names in ``config.presets``."""
        if name not in self.config.presets:
            raise ValueError(f"{self.id} has no preset {name!r}")
        defined = self.config.preset_defs.get(name)
        return defined if defined is not None else PRESETS[name]

    def task_spec(self, wheel_radius_m: float, track_width_m: float) -> TaskSpec:
        if self.config.action_spec != BALANCE_V1_ACTION.id:
            raise ValueError(f"unsupported action for {self.id}")
        schema = self.config.obs_schema
        extras = self.config.course.task_extras() if self.config.course is not None else {}
        return TaskSpec(
            challenge=self.id,
            obs_schema=schema,
            obs_features=obs_features(schema),
            action=BALANCE_V1_ACTION,
            control_dt=self.config.control_dt,
            episode_seconds=self.config.episode_seconds,
            robot=RobotGeometry(wheel_radius_m=wheel_radius_m, track_width_m=track_width_m),
            extras=extras,
        )

    def bundle_files(self) -> tuple[Path, ...]:
        """The files that define the challenge, sorted: ``challenge.yaml``, the scene, the robot
        params, every ``*.xml`` next to the robot MJCF, and every file the scene or robot MJCF
        references (meshes, textures, includes). All must lie inside the bundle directory."""
        robot_dir = self.robot_xml.parent
        files = {
            self.root / "challenge.yaml",
            self.scene_xml,
            self.root / self.config.params,
            *robot_dir.glob("*.xml"),
            *referenced_files(self.scene_xml),
            *referenced_files(self.robot_xml),
        }
        root = self.root.resolve()
        resolved = {path.resolve() for path in files}
        outside = sorted(str(p) for p in resolved if not p.is_relative_to(root))
        if outside:
            raise ValueError(f"bundle files outside {root}: {outside}")
        return tuple(sorted(resolved, key=lambda p: p.relative_to(root).as_posix()))

    def bundle_hash(self) -> str:
        """sha256 over ``bundle_files`` (contract §13).

        A tag, then per file in POSIX-path order: the length-framed relative path and the
        length-framed bytes, with CRLF normalized to LF in text files (``.xml``, ``.yaml``,
        ``.yml``: ``topple.robot.hashed_bytes``, as in the robot hash). Stray files (caches,
        notes, editor droppings) do not count, and where the bundle lives does not matter.
        """
        root = self.root.resolve()
        digest = hashlib.sha256(BUNDLE_HASH_TAG)
        for path in self.bundle_files():
            digest.update(_framed(path.relative_to(root).as_posix().encode("utf-8")))
            digest.update(_framed(hashed_bytes(path)))
        return digest.hexdigest()

    def protocol_hash(self, preset: str, seeds: Sequence[int] | None = None, salt: int = 0) -> str:
        """sha256 of everything that fixes a protocol's scores (contract §13).

        Covers the bundle hash, the preset definition (pushes, randomization ranges), the score
        config, the seed list in order (default: the public seeds), the controller-seed salt and
        ``ENV_ALGO_VERSION``, as canonical JSON.
        """
        resolved = self.preset(preset)
        seed_list = list(self.config.seeds.public if seeds is None else seeds)
        if not seed_list:
            raise ValueError("a protocol needs at least one seed")
        for seed in seed_list:
            validate_seed(seed)
        if isinstance(salt, bool) or not isinstance(salt, int) or salt < 0:
            raise ValueError(f"salt must be a non-negative int, got {salt!r}")
        payload = {
            "format": PROTOCOL_HASH_FORMAT,
            "env_algo_version": env_algo_version(self.kind),
            "bundle": self.bundle_hash(),
            "preset": resolved.model_dump(mode="json"),
            "score": self.config.score.model_dump(mode="json"),
            "seeds": seed_list,
            "salt": salt,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_challenge_dir(root: Path) -> Challenge:
    config = ChallengeConfig.model_validate(yaml.safe_load((root / "challenge.yaml").read_text()))
    return Challenge(config=config, root=root, params=load_params(root / config.params))


@lru_cache(maxsize=8)
def load_challenge(challenge_id: str) -> Challenge:
    """Load a built-in challenge bundle by id (e.g. ``balance-v1``)."""
    return load_challenge_dir(challenge_dir(validate_challenge_id(challenge_id)))
