"""Challenge bundles: ``bundle_hash`` and ``protocol_hash`` (docs/contract.md §13)."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import topple.challenge as challenge_module
from topple.challenge import ENV_ALGO_VERSION, Challenge, load_challenge, load_challenge_dir
from topple.disturbances import PRESETS
from topple.robot import challenge_dir

BUNDLE_FILES = ("challenge.yaml", "robot/balancer.xml", "robot/params.yaml", "scene.xml")


def reference_bundle_hash(root: Path, rels: tuple[str, ...]) -> str:
    """The normative framing, written out independently of the implementation."""
    digest = hashlib.sha256(b"topple.bundle/1\n")
    for rel in sorted(rels):
        data = (root / rel).read_bytes().replace(b"\r\n", b"\n")
        name = rel.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "balance-v1"
    shutil.copytree(challenge_dir("balance-v1"), root)
    return root


def test_bundle_hash_follows_the_framing(bundle: Path) -> None:
    assert load_challenge_dir(bundle).bundle_hash() == reference_bundle_hash(bundle, BUNDLE_FILES)


def test_bundle_hash_does_not_depend_on_where_the_bundle_lives(bundle: Path) -> None:
    assert load_challenge_dir(bundle).bundle_hash() == load_challenge("balance-v1").bundle_hash()


@pytest.mark.parametrize("stray", ["notes.txt", "__pycache__/x.pyc", ".DS_Store", "robot/old.bak"])
def test_bundle_hash_ignores_files_outside_the_list(bundle: Path, stray: str) -> None:
    before = load_challenge_dir(bundle).bundle_hash()
    (bundle / stray).parent.mkdir(parents=True, exist_ok=True)
    (bundle / stray).write_bytes(b"anything")
    assert load_challenge_dir(bundle).bundle_hash() == before


@pytest.mark.parametrize("rel", BUNDLE_FILES)
def test_bundle_hash_covers_every_listed_file(bundle: Path, rel: str) -> None:
    before = load_challenge_dir(bundle).bundle_hash()
    path = bundle / rel
    comment = b"# edited\n" if rel.endswith(".yaml") else b"<!-- edited -->\n"
    path.write_bytes(path.read_bytes() + comment)
    assert load_challenge_dir(bundle).bundle_hash() != before


def test_bundle_hash_covers_extra_robot_xml_files(bundle: Path) -> None:
    before = load_challenge_dir(bundle).bundle_hash()
    (bundle / "robot" / "parts.xml").write_text("<mujoco/>\n")
    assert load_challenge_dir(bundle).bundle_hash() != before


def test_bundle_hash_normalizes_line_endings(bundle: Path) -> None:
    before = load_challenge_dir(bundle).bundle_hash()
    for rel in BUNDLE_FILES:
        path = bundle / rel
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert load_challenge_dir(bundle).bundle_hash() == before


def test_bundle_hash_frames_lengths(bundle: Path) -> None:
    """Moving bytes from the end of one file to the start of the next changes the hash."""
    before = load_challenge_dir(bundle).bundle_hash()
    params = bundle / "robot" / "params.yaml"
    robot = bundle / "robot" / "balancer.xml"
    text = params.read_bytes()
    assert text.endswith(b"\n")
    params.write_bytes(text[:-1])
    robot.write_bytes(b"\n" + robot.read_bytes())
    assert load_challenge_dir(bundle).bundle_hash() != before


def test_bundle_files_must_stay_inside_the_bundle(bundle: Path, tmp_path: Path) -> None:
    shutil.copy(bundle / "scene.xml", tmp_path / "outside.xml")
    yaml_path = bundle / "challenge.yaml"
    yaml_path.write_text(yaml_path.read_text().replace("scene: scene.xml", "scene: ../outside.xml"))
    with pytest.raises(ValueError, match="outside"):
        load_challenge_dir(bundle).bundle_hash()


# ------------------------------------------------------------------------------- protocol_hash


def payload_hash(challenge: Challenge, preset: str, seeds: list[int], salt: int) -> str:
    payload = {
        "format": "topple.protocol/1",
        "env_algo_version": ENV_ALGO_VERSION,
        "bundle": challenge.bundle_hash(),
        "preset": PRESETS[preset].model_dump(mode="json"),
        "score": challenge.config.score.model_dump(mode="json"),
        "seeds": seeds,
        "salt": salt,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_protocol_hash_follows_the_payload() -> None:
    c = load_challenge("balance-v1")
    public = list(c.config.seeds.public)
    assert c.protocol_hash("standard") == payload_hash(c, "standard", public, 0)
    assert c.protocol_hash("clean", seeds=(5, 6), salt=7) == payload_hash(c, "clean", [5, 6], 7)


def test_protocol_hash_covers_seeds_salt_and_preset() -> None:
    c = load_challenge("balance-v1")
    base = c.protocol_hash("standard")
    assert c.protocol_hash("standard", seeds=c.config.seeds.public) == base
    assert c.protocol_hash("standard", seeds=c.config.seeds.public[:-1]) != base
    assert c.protocol_hash("standard", seeds=tuple(reversed(c.config.seeds.public))) != base
    assert c.protocol_hash("standard", salt=1) != base
    assert c.protocol_hash("clean") != base


def test_protocol_hash_covers_the_env_algorithm_version(monkeypatch: pytest.MonkeyPatch) -> None:
    c = load_challenge("balance-v1")
    base = c.protocol_hash("standard")
    monkeypatch.setattr(challenge_module, "ENV_ALGO_VERSION", ENV_ALGO_VERSION + 1)
    assert c.protocol_hash("standard") != base


def test_protocol_hash_covers_the_score_config(bundle: Path) -> None:
    base = load_challenge_dir(bundle).protocol_hash("standard")
    yaml_path = bundle / "challenge.yaml"
    yaml_path.write_text(yaml_path.read_text().replace("tilt_ref_rad: 0.2", "tilt_ref_rad: 0.3"))
    assert load_challenge_dir(bundle).protocol_hash("standard") != base


def test_protocol_hash_rejects_unknown_presets_and_bad_seeds() -> None:
    c = load_challenge("balance-v1")
    with pytest.raises(ValueError, match="preset"):
        c.protocol_hash("windy")
    with pytest.raises(ValueError, match="seed"):
        c.protocol_hash("standard", seeds=(-1,))
    with pytest.raises(ValueError, match="seed"):
        c.protocol_hash("standard", seeds=())
    with pytest.raises(ValueError, match="salt"):
        c.protocol_hash("standard", salt=-1)


def test_env_algo_version_starts_at_one() -> None:
    assert ENV_ALGO_VERSION == 1
