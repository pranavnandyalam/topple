import hashlib
import json
import math
import re
import struct
import zlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from topple.robot import (
    RobotParams,
    bind_robot,
    build_model,
    challenge_dir,
    load_params,
    referenced_files,
    robot_hash,
    validate_robot,
)

BUNDLE = challenge_dir("balance-v1")
ROBOT_XML = BUNDLE / "robot" / "balancer.xml"
SCENE_XML = BUNDLE / "scene.xml"


@pytest.fixture(scope="module")
def params() -> RobotParams:
    return load_params(BUNDLE / "robot" / "params.yaml")


@pytest.fixture(scope="module")
def model(params: RobotParams) -> mujoco.MjModel:
    return build_model(ROBOT_XML, SCENE_XML, params)


def pitch_of(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    r = d.xmat[m.body("chassis").id].reshape(3, 3)
    return math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))


def set_tilt(m: mujoco.MjModel, d: mujoco.MjData, pitch: float) -> None:
    mujoco.mj_resetData(m, d)
    adr = m.joint("root").qposadr[0]
    d.qpos[adr + 3 : adr + 7] = [math.cos(pitch / 2), 0, math.sin(pitch / 2), 0]
    mujoco.mj_forward(m, d)


def test_params_load(params: RobotParams) -> None:
    assert params.motor.tau_stall_nm == pytest.approx(0.3)
    assert params.safety.fall_angle_rad == pytest.approx(0.70)


def test_params_reject_unknown_keys(tmp_path: Path) -> None:
    bad = tmp_path / "p.yaml"
    bad.write_text((BUNDLE / "robot" / "params.yaml").read_text() + "\nbogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        load_params(bad)


def test_params_reject_non_positive(params: RobotParams) -> None:
    data = params.model_dump()
    data["motor"]["tau_stall_nm"] = 0.0
    with pytest.raises(ValueError, match="tau_stall_nm"):
        RobotParams.model_validate(data)


def test_scene_options(model: mujoco.MjModel) -> None:
    assert model.opt.timestep == pytest.approx(0.002)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET


def test_injected_motors(model: mujoco.MjModel, params: RobotParams) -> None:
    assert model.nu == 2
    for name in ("left_motor", "right_motor"):
        act = model.actuator(name)
        assert act.gainprm[0] == pytest.approx(params.motor.tau_stall_nm)
        assert act.biasprm[2] == pytest.approx(
            -params.motor.tau_stall_nm / params.motor.omega_noload_rads
        )
        np.testing.assert_allclose(act.ctrlrange, [-1, 1])
        np.testing.assert_allclose(
            act.forcerange, [-params.motor.torque_limit_nm, params.motor.torque_limit_nm]
        )
        assert model.actuator_history[act.id][0] >= 6  # room for up to 10 ms native delay


def test_wheel_joints(model: mujoco.MjModel, params: RobotParams) -> None:
    for name in ("left_wheel", "right_wheel"):
        dof = model.joint(name).dofadr[0]
        assert model.dof_damping[dof] == 0.0
        assert model.dof_frictionloss[dof] == pytest.approx(params.motor.frictionloss_nm)
        assert model.dof_armature[dof] == pytest.approx(params.motor.armature)


def test_imu_sensors(model: mujoco.MjModel) -> None:
    assert model.sensor("imu_gyro").dim[0] == 3
    assert model.sensor("imu_accel").dim[0] == 3


def test_motor_torque_law(model: mujoco.MjModel, params: RobotParams) -> None:
    """torque = tau_stall*u - (tau_stall/omega_nl)*omega, capped at the current limit."""
    d = mujoco.MjData(model)
    dof = model.joint("left_wheel").dofadr[0]
    for u, omega in [(1.0, 0.0), (1.0, 15.0), (0.5, 10.0), (-1.0, -30.0), (0.0, 20.0)]:
        mujoco.mj_resetData(model, d)
        d.qvel[dof] = omega
        d.ctrl[:] = u
        mujoco.mj_forward(model, d)
        expected = (
            params.motor.tau_stall_nm * u
            - params.motor.tau_stall_nm / params.motor.omega_noload_rads * omega
        )
        expected = max(-params.motor.torque_limit_nm, min(params.motor.torque_limit_nm, expected))
        assert d.actuator_force[model.actuator("left_motor").id] == pytest.approx(
            expected, abs=1e-9
        )


def test_falls_without_control(model: mujoco.MjModel) -> None:
    d = mujoco.MjData(model)
    set_tilt(model, d, math.radians(2))
    for _ in range(400):  # 0.8 s
        mujoco.mj_step(model, d)
        if abs(pitch_of(model, d)) > 0.70:
            break
    assert abs(pitch_of(model, d)) > 0.70
    assert pitch_of(model, d) > 0  # falls the way it leans


def test_forward_command_rolls_forward_and_tips_back(model: mujoco.MjModel) -> None:
    d = mujoco.MjData(model)
    set_tilt(model, d, 0.0)
    d.ctrl[:] = 0.5
    for _ in range(50):  # 0.1 s
        mujoco.mj_step(model, d)
    assert d.qpos[model.joint("root").qposadr[0]] > 0  # moved toward +x
    assert pitch_of(model, d) < 0  # reaction torque tips the top backwards
    assert d.qvel[model.joint("left_wheel").dofadr[0]] > 0


def test_turn_left_when_right_wheel_faster(model: mujoco.MjModel) -> None:
    d = mujoco.MjData(model)
    set_tilt(model, d, 0.0)
    d.ctrl[:] = [-0.3, 0.3]
    for _ in range(100):
        mujoco.mj_step(model, d)
    w = d.qvel[model.joint("root").dofadr[0] + 3 : model.joint("root").dofadr[0] + 6]
    assert w[2] > 0  # counter-clockwise from above


def test_unstable_pole_magnitude(params: RobotParams) -> None:
    """Linearized about upright, the fastest unstable mode is a few rad/s (balanceable)."""
    model = build_model(ROBOT_XML, SCENE_XML, params, delay_buffer=False)
    assert model.actuator_history[0][0] == 0
    d = mujoco.MjData(model)
    set_tilt(model, d, 0.0)
    for _ in range(100):  # settle contacts
        mujoco.mj_step(model, d)
    nv = model.nv
    a = np.zeros((2 * nv, 2 * nv))
    mujoco.mjd_transitionFD(model, d, 1e-6, True, a, None, None, None)
    lam = np.log(np.abs(np.linalg.eigvals(a)).max()) / model.opt.timestep
    assert 5.0 < lam < 12.0


def test_binding(model: mujoco.MjModel) -> None:
    b = bind_robot(model)
    assert b.wheel_radius_m == pytest.approx(0.034)
    assert b.track_width_m == pytest.approx(0.14)
    assert b.wheel_sign == (1.0, 1.0)
    assert b.chassis_body == model.body("chassis").id
    assert set(b.wheel_geoms) == {model.geom("left_tire").id, model.geom("right_tire").id}


def test_validate_good_model(model: mujoco.MjModel) -> None:
    assert validate_robot(model) == []


def _variant(params: RobotParams, tmp_path: Path, old: str, new: str) -> mujoco.MjModel:
    xml = ROBOT_XML.read_text()
    assert old in xml
    path = tmp_path / "variant.xml"
    path.write_text(xml.replace(old, new))
    return build_model(path, SCENE_XML, params)


def test_mirrored_wheel_axis_detected(params: RobotParams, tmp_path: Path) -> None:
    m = _variant(
        params,
        tmp_path,
        '<joint name="right_wheel" type="hinge" axis="0 1 0"/>',
        '<joint name="right_wheel" type="hinge" axis="0 -1 0"/>',
    )
    assert bind_robot(m).wheel_sign == (1.0, -1.0)
    assert validate_robot(m) == []


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('<site name="imu"', '<site name="imu_board"', "site 'imu'"),
        ('<site name="push_point"', '<site name="pushpt"', "site 'push_point'"),
        ('<joint name="left_wheel"', '<joint name="lw"', "joint 'left_wheel'"),
        (
            '<joint name="right_wheel" type="hinge" axis="0 1 0"/>',
            '<joint name="right_wheel" type="hinge" axis="1 0 0"/>',
            "axis",
        ),
    ],
)
def test_validate_reports_problems(
    params: RobotParams, tmp_path: Path, old: str, new: str, message: str
) -> None:
    xml = ROBOT_XML.read_text().replace(old, new)
    path = tmp_path / "broken.xml"
    path.write_text(xml)
    try:
        m = build_model(path, SCENE_XML, params)
    except ValueError as exc:  # build_model refuses robots it cannot wire up
        problems = [str(exc)]
    else:
        problems = validate_robot(m)
    assert any(message in p for p in problems), problems


def test_robot_hash_changes_with_params(params: RobotParams) -> None:
    h1 = robot_hash(ROBOT_XML, params)
    data = params.model_dump()
    data["motor"]["frictionloss_nm"] = 0.03
    h2 = robot_hash(ROBOT_XML, RobotParams.model_validate(data))
    assert h1 != h2
    assert h1 == robot_hash(ROBOT_XML, params)
    assert len(h1) == 64


BALANCE_V1_ROBOT_HASH = "03f5654f03eff5be5cb7689b3b06356f42c8c4a1b3ed60795d7e8501a767ca59"
LEFT_BODY = '<body name="left_wheel_body" pos="0 0.07 0">'
RIGHT_BODY = '<body name="right_wheel_body" pos="0 -0.07 0">'
LEFT_JOINT = '<joint name="left_wheel" type="hinge" axis="0 1 0"/>'
LEFT_TIRE = '<geom name="left_tire" type="cylinder"'
RIGHT_TIRE = '<geom name="right_tire" type="cylinder"'
WORLD_END = "  </worldbody>"


def model_from(params: RobotParams, tmp_path: Path, *edits: tuple[str, str]) -> mujoco.MjModel:
    xml = ROBOT_XML.read_text()
    for old, new in edits:
        assert old in xml, old
        xml = xml.replace(old, new, 1)
    path = tmp_path / "variant.xml"
    path.write_text(xml)
    return build_model(path, SCENE_XML, params)


VIOLATIONS = {
    "wheels on the wrong sides": (
        [
            (LEFT_BODY, '<body name="left_wheel_body" pos="0 -0.07 0">'),
            (RIGHT_BODY, '<body name="right_wheel_body" pos="0 0.07 0">'),
        ],
        "y > 0",
    ),
    "limited wheel joint": (
        [(LEFT_JOINT, LEFT_JOINT.replace("/>", ' range="-1 1"/>'))],
        "unlimited",
    ),
    "second collision geom on a wheel": (
        [(LEFT_JOINT, LEFT_JOINT + '<geom type="sphere" size="0.01" pos="0 0 0.02"/>')],
        "exactly one collision geom",
    ),
    "box tire": (
        [
            (
                LEFT_TIRE + ' size="0.034 0.013"',
                '<geom name="left_tire" type="box" size="0.03 0.01 0.03"',
            )
        ],
        "cylinder, sphere or capsule",
    ),
    "equality constraint": (
        [("</worldbody>", '</worldbody><equality><weld body1="chassis"/></equality>')],
        "equality",
    ),
    "extra world body": (
        [
            (
                WORLD_END,
                '<body name="ramp" pos="1 0 0"><geom type="box" size="0.1 0.1 0.01"/></body>\n'
                + WORLD_END,
            )
        ],
        "world",
    ),
    "extra world geom": (
        [
            (
                WORLD_END,
                '<geom name="wall" type="box" size="0.1 0.1 0.1" pos="2 0 0"/>\n' + WORLD_END,
            )
        ],
        "world",
    ),
    "root cannot fall": (
        [('<freejoint name="root"/>', '<joint name="root" type="slide" axis="0 0 1"/>')],
        "free joint",
    ),
    "gravity compensation": (
        [
            (
                '<body name="chassis" pos="0 0 0.034">',
                '<body name="chassis" pos="0 0 0.034" gravcomp="1">',
            )
        ],
        "gravcomp",
    ),
    "right wheel on the left": (
        [(RIGHT_BODY, '<body name="right_wheel_body" pos="0 0.07 0">')],
        "y < 0 (right)",
    ),
    "left wheel on the right": (
        [(LEFT_BODY, '<body name="left_wheel_body" pos="0 -0.07 0">')],
        "y > 0 (left)",
    ),
    "tires of different radii": (
        [(RIGHT_TIRE + ' size="0.034 0.013"', RIGHT_TIRE + ' size="0.030 0.013"')],
        "same radius",
    ),
}


@pytest.mark.parametrize(("edits", "message"), VIOLATIONS.values(), ids=VIOLATIONS.keys())
def test_validate_reports_each_violation(
    params: RobotParams, tmp_path: Path, edits: list[tuple[str, str]], message: str
) -> None:
    problems = validate_robot(model_from(params, tmp_path, *edits))
    assert any(message in p for p in problems), problems


def test_visual_geoms_on_a_wheel_are_allowed_and_do_not_set_the_radius(
    params: RobotParams, tmp_path: Path
) -> None:
    hubcap = '<geom type="sphere" size="0.05" contype="0" conaffinity="0" mass="0.001"/>'
    m = model_from(params, tmp_path, (LEFT_JOINT, LEFT_JOINT + hubcap))
    assert validate_robot(m) == []
    b = bind_robot(m)
    assert b.wheel_radius_m == pytest.approx(0.034)
    assert set(b.wheel_geoms) == {m.geom("left_tire").id, m.geom("right_tire").id}


def test_bind_robot_refuses_ambiguous_tires(params: RobotParams, tmp_path: Path) -> None:
    m = model_from(params, tmp_path, (LEFT_JOINT, LEFT_JOINT + '<geom type="sphere" size="0.02"/>'))
    with pytest.raises(ValueError, match="exactly one collision geom"):
        bind_robot(m)


# ------------------------------------------------------------------------------- robot_hash


def test_balance_v1_robot_hash_is_unchanged(params: RobotParams) -> None:
    """balance-v1 references no asset files, so covering them leaves its hash as it was."""
    assert referenced_files(ROBOT_XML) == ()
    assert robot_hash(ROBOT_XML, params) == BALANCE_V1_ROBOT_HASH


MESH = "v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nv 0 0 0.01\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"


def mesh_robot(tmp_path: Path, meshdir: str | None = "assets") -> Path:
    compiler = '<compiler angle="radian" autolimits="true"/>'
    new_compiler = (
        compiler if meshdir is None else compiler.replace("/>", f' meshdir="{meshdir}"/>')
    )
    xml = (
        ROBOT_XML.read_text()
        .replace(compiler, new_compiler + '<asset><mesh name="logo" file="logo.obj"/></asset>')
        .replace(
            '<site name="imu"',
            '<geom type="mesh" mesh="logo" contype="0" conaffinity="0" mass="0.001"/>'
            '<site name="imu"',
        )
    )
    path = tmp_path / "robot" / "bot.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    mesh_dir = path.parent / meshdir if meshdir else path.parent
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "logo.obj").write_text(MESH)
    return path


@pytest.mark.parametrize("meshdir", ["assets", None])
def test_robot_hash_covers_referenced_mesh_files(
    params: RobotParams, tmp_path: Path, meshdir: str | None
) -> None:
    path = mesh_robot(tmp_path, meshdir)
    mesh = path.parent / (meshdir or ".") / "logo.obj"
    assert referenced_files(path) == (mesh.resolve(),)
    build_model(path, SCENE_XML, params)  # MuJoCo resolves the file the same way
    before = robot_hash(path, params)
    mesh.write_text(MESH.replace("0.01", "0.02"))
    assert robot_hash(path, params) != before


def test_robot_hash_covers_included_files(params: RobotParams, tmp_path: Path) -> None:
    part = tmp_path / "part.xml"
    part.write_text('<mujoco><default><geom rgba="1 0 0 1"/></default></mujoco>\n')
    xml = ROBOT_XML.read_text().replace(
        '<compiler angle="radian" autolimits="true"/>',
        '<compiler angle="radian" autolimits="true"/><include file="part.xml"/>',
    )
    path = tmp_path / "bot.xml"
    path.write_text(xml)
    assert referenced_files(path) == (part.resolve(),)
    before = robot_hash(path, params)
    part.write_text(part.read_text().replace("1 0 0 1", "0 1 0 1"))
    assert robot_hash(path, params) != before


def reference_robot_hash(xml: Path, params: RobotParams, rels: list[str]) -> str:
    """Contract §3 written out independently: records in POSIX-path string order."""
    digest = hashlib.sha256(xml.read_bytes())
    digest.update(json.dumps(params.model_dump(), sort_keys=True).encode())
    for rel in sorted(rels):
        name, data = rel.encode("utf-8"), (xml.parent / rel).read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def test_robot_hash_orders_records_by_posix_path_string(
    params: RobotParams, tmp_path: Path
) -> None:
    """``a-c.obj`` sorts before ``a/b.obj`` as a string ('-' < '/'), after it as a Path."""
    path = mesh_robot(tmp_path, "meshes")
    xml = path.read_text().replace(
        '<mesh name="logo" file="logo.obj"/>',
        '<mesh name="m1" file="a/b.obj"/><mesh name="m2" file="a-c.obj"/>',
    )
    path.write_text(xml.replace('mesh="logo"', 'mesh="m1"'))
    meshes = path.parent / "meshes"
    (meshes / "logo.obj").unlink()
    (meshes / "a").mkdir()
    (meshes / "a" / "b.obj").write_text(MESH)
    (meshes / "a-c.obj").write_text(MESH.replace("0.01", "0.02"))
    build_model(path, SCENE_XML, params)
    expected = reference_robot_hash(path, params, ["meshes/a/b.obj", "meshes/a-c.obj"])
    assert robot_hash(path, params) == expected


def tiny_png() -> bytes:
    """A valid 1x1 RGB PNG, built from its chunks."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data).to_bytes(4, "big")
        return len(data).to_bytes(4, "big") + kind + data + crc

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"\x00\xff\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
    )


def asset_robot(tmp_path: Path, compiler_attrs: str) -> Path:
    """A robot with one mesh and one 2D texture, loaded through ``<compiler {compiler_attrs}>``."""
    compiler = '<compiler angle="radian" autolimits="true"/>'
    xml = (
        ROBOT_XML.read_text()
        .replace(
            compiler,
            compiler.replace("/>", f" {compiler_attrs}/>")
            + '<asset><mesh name="logo" file="logo.obj"/>'
            '<texture name="skin" type="2d" file="skin.png"/>'
            '<material name="skin" texture="skin"/></asset>',
        )
        .replace(
            '<site name="imu"',
            '<geom type="mesh" mesh="logo" material="skin" contype="0" conaffinity="0"'
            ' mass="0.001"/><site name="imu"',
        )
    )
    path = tmp_path / "robot" / "bot.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    return path


ASSET_LAYOUTS = {
    "texturedir": ('meshdir="m" texturedir="t"', "m/logo.obj", "t/skin.png"),
    "assetdir": ('assetdir="a"', "a/logo.obj", "a/skin.png"),
    "assetdir with meshdir": ('assetdir="a" meshdir="m"', "m/logo.obj", "a/skin.png"),
    "assetdir with texturedir": ('assetdir="a" texturedir="t"', "a/logo.obj", "t/skin.png"),
}


@pytest.mark.parametrize(
    ("attrs", "mesh_rel", "texture_rel"), ASSET_LAYOUTS.values(), ids=ASSET_LAYOUTS.keys()
)
def test_referenced_files_resolve_like_mujoco(
    params: RobotParams, tmp_path: Path, attrs: str, mesh_rel: str, texture_rel: str
) -> None:
    path = asset_robot(tmp_path, attrs)
    base = path.parent
    for rel, data in ((mesh_rel, MESH.encode()), (texture_rel, tiny_png())):
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_bytes(data)
    for decoy in ("logo.obj", "skin.png"):  # next to the MJCF, where a wrong resolver would look
        (base / decoy).write_bytes(
            (base / (mesh_rel if decoy.endswith("obj") else texture_rel)).read_bytes()
        )
    build_model(path, SCENE_XML, params)
    assert referenced_files(path) == tuple(
        sorted([(base / mesh_rel).resolve(), (base / texture_rel).resolve()])
    )
    for rel in (mesh_rel, texture_rel):
        before = robot_hash(path, params)
        target = base / rel
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        assert robot_hash(path, params) != before, rel
        target.write_bytes(original)
        target.unlink()  # MuJoCo needs exactly this file: the decoy does not stand in for it
        with pytest.raises(ValueError, match=re.escape(target.name)):
            build_model(path, SCENE_XML, params)
        target.write_bytes(original)


def test_nested_includes_resolve_against_the_model_directory(
    params: RobotParams, tmp_path: Path
) -> None:
    """``sub/a.xml`` includes ``b.xml``; MuJoCo looks for it next to the top-level MJCF."""
    robot_dir = tmp_path / "robot"
    (robot_dir / "sub").mkdir(parents=True)
    (robot_dir / "sub" / "a.xml").write_text('<mujoco><include file="b.xml"/></mujoco>\n')
    b = robot_dir / "b.xml"
    b.write_text('<mujoco><default><geom rgba="1 0 0 1"/></default></mujoco>\n')
    xml = ROBOT_XML.read_text().replace(
        '<compiler angle="radian" autolimits="true"/>',
        '<compiler angle="radian" autolimits="true"/><include file="sub/a.xml"/>',
    )
    path = robot_dir / "bot.xml"
    path.write_text(xml)
    build_model(path, SCENE_XML, params)
    assert referenced_files(path) == tuple(
        sorted([b.resolve(), (robot_dir / "sub/a.xml").resolve()])
    )
    before = robot_hash(path, params)
    b.write_text(b.read_text().replace("1 0 0 1", "0 1 0 1"))
    assert robot_hash(path, params) != before


def test_robot_hash_refuses_missing_referenced_files(params: RobotParams, tmp_path: Path) -> None:
    path = mesh_robot(tmp_path)
    (path.parent / "assets" / "logo.obj").unlink()
    with pytest.raises(FileNotFoundError, match=r"logo\.obj"):
        robot_hash(path, params)


def test_validate_reports_a_bare_robot_without_injected_parts() -> None:
    """The raw MJCF, compiled alone, lacks everything build_model injects."""
    problems = validate_robot(mujoco.MjModel.from_xml_path(str(ROBOT_XML)))
    for message in (
        "actuator 'left_motor' is missing",
        "expected 2 actuators, got 0",
        "sensor 'imu_gyro' is missing",
        "sensor 'imu_accel' is missing",
    ):
        assert message in problems


def test_validate_reports_a_missing_chassis() -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="box"><freejoint/>'
        '<geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>'
    )
    assert validate_robot(model) == ["body 'chassis' is missing"]


@pytest.mark.parametrize(
    ("edits", "message"),
    [
        ([(LEFT_JOINT, '<joint name="left_wheel" type="slide" axis="0 1 0"/>')], "hinge"),
        (
            [
                (LEFT_BODY, '<body name="hub" pos="0 0.07 0"><body name="left_wheel_body">'),
                (
                    '</body>\n      <body name="right_wheel_body"',
                    '</body></body>\n      <body name="right_wheel_body"',
                ),
            ],
            "direct child body",
        ),
        ([('<site name="imu"', '<site name="imu_board"')], "site 'imu' is missing"),
    ],
)
def test_validate_reports_wheel_and_site_problems(
    params: RobotParams, tmp_path: Path, edits: list[tuple[str, str]], message: str
) -> None:
    xml = ROBOT_XML.read_text()
    for old, new in edits:
        assert old in xml, old
        xml = xml.replace(old, new, 1)
    path = tmp_path / "variant.xml"
    path.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(path))  # raw: build_model would refuse some
    assert any(message in p for p in validate_robot(model)), validate_robot(model)
