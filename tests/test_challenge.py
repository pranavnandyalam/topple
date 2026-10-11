"""Challenge bundles: ``bundle_hash`` and ``protocol_hash`` (docs/contract.md §13)."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import topple.challenge as challenge_module
from topple.challenge import ENV_ALGO_VERSION, Challenge, load_challenge, load_challenge_dir
from topple.disturbances import PRESETS
from topple.robot import challenge_dir, robot_hash

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


def test_a_crlf_checkout_has_the_same_hashes_everywhere(bundle: Path) -> None:
    """A Windows checkout (``core.autocrlf``) gets CRLF XML and YAML. It is the same challenge
    and the same robot, so the bundle, protocol and robot hashes must all be unchanged: a
    trajectory recorded there must not carry a foreign ``robot_hash``."""
    stock = load_challenge("balance-v1")
    for rel in BUNDLE_FILES:
        path = bundle / rel
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    crlf = load_challenge_dir(bundle)
    assert crlf.bundle_hash() == stock.bundle_hash()
    assert crlf.protocol_hash("standard") == stock.protocol_hash("standard")
    assert robot_hash(crlf.robot_xml, crlf.params) == robot_hash(stock.robot_xml, stock.params)


def test_yml_files_are_text_too(bundle: Path) -> None:
    """Contract §13: CRLF -> LF in ``.xml``, ``.yaml`` and ``.yml`` files."""
    (bundle / "robot" / "params.yaml").rename(bundle / "robot" / "params.yml")
    yaml_path = bundle / "challenge.yaml"
    yaml_path.write_text(yaml_path.read_text().replace("params.yaml", "params.yml"))
    before = load_challenge_dir(bundle).bundle_hash()
    params = bundle / "robot" / "params.yml"
    params.write_bytes(params.read_bytes().replace(b"\n", b"\r\n"))
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


def test_bundle_hash_covers_files_the_scene_references(bundle: Path) -> None:
    """A scene asset (here a floor texture) is part of the bundle: editing it changes the hash."""
    scene = bundle / "scene.xml"
    scene.write_text(
        scene.read_text().replace(
            "<asset>", '<asset>\n    <texture name="tiles" type="2d" file="tiles.png"/>', 1
        )
    )
    tiles = bundle / "tiles.png"
    tiles.write_bytes(b"first")
    challenge = load_challenge_dir(bundle)
    assert tiles.resolve() in challenge.bundle_files()
    before = challenge.bundle_hash()
    assert before == reference_bundle_hash(bundle, (*BUNDLE_FILES, "tiles.png"))
    tiles.write_bytes(b"second")
    assert load_challenge_dir(bundle).bundle_hash() != before


def test_bundle_hash_covers_the_files_of_an_attached_sub_model(bundle: Path) -> None:
    """A mesh used only by a robot sub-model (``<asset><model>``) is part of the bundle."""
    robot = bundle / "robot" / "balancer.xml"
    compiler = '<compiler angle="radian" autolimits="true"/>'
    robot.write_text(
        robot.read_text()
        .replace(compiler, compiler + '<asset><model name="part" file="parts/part.xml"/></asset>')
        .replace('<site name="imu"', '<attach model="part" body="pb" prefix="p_"/><site name="imu"')
    )
    parts = bundle / "robot" / "parts"
    parts.mkdir()
    (parts / "part.xml").write_text(
        '<mujoco><asset><mesh name="m" file="mesh.obj"/></asset><worldbody><body name="pb">'
        '<geom type="mesh" mesh="m" contype="0" conaffinity="0" mass="0.001"/></body>'
        "</worldbody></mujoco>\n"
    )
    mesh = parts / "mesh.obj"
    mesh.write_text(
        "v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nv 0 0 0.01\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
    )
    challenge = load_challenge_dir(bundle)
    assert mesh.resolve() in challenge.bundle_files()
    before = challenge.bundle_hash()
    rels = (*BUNDLE_FILES, "robot/parts/part.xml", "robot/parts/mesh.obj")
    assert before == reference_bundle_hash(bundle, rels)
    mesh.write_text(mesh.read_text().replace("0.01", "0.02"))
    assert load_challenge_dir(bundle).bundle_hash() != before
