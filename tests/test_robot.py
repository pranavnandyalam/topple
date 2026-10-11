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

from topple.envs.balance import BalanceEnv
from topple.robot import (
    FRICTIONLOSS_SOLIMP,
    FRICTIONLOSS_SOLREF,
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
    assert b.track_width_m == 0.14  # bit for bit: the tire centres are the wheel bodies' here
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
# The params exactly as robot_hash serializes them (contract §3): Python's json.dumps defaults with
# sorted keys, so ", " and ": " separators and floats in repr form ("30.0", never "30").
BALANCE_V1_PARAMS_JSON = (
    '{"contact": {"wheel_friction": 1.0}, "estimation": {"cf_tau_s": 0.5}, '
    '"motor": {"armature": 0.0001, "frictionloss_nm": 0.02, "omega_noload_rads": 30.0, '
    '"stall_current_a": 1.5, "tau_stall_nm": 0.3, "torque_limit_nm": 0.3, "v_nom": 7.4}, '
    '"safety": {"fall_angle_rad": 0.7}, '
    '"sensing": {"accel_noise_ms2": 0.05, "accel_range_ms2": 39.2, "encoder_cpr": 1320, '
    '"gyro_noise_rads": 0.01, "gyro_range_rads": 8.73}}'
)
LEFT_BODY = '<body name="left_wheel_body" pos="0 0.07 0">'
RIGHT_BODY = '<body name="right_wheel_body" pos="0 -0.07 0">'
LEFT_JOINT = '<joint name="left_wheel" type="hinge" axis="0 1 0"/>'
LEFT_TIRE = '<geom name="left_tire" type="cylinder"'
RIGHT_TIRE = '<geom name="right_tire" type="cylinder"'
WORLD_END = "  </worldbody>"
FREE_ROOT = '<freejoint name="root"/>'
CHASSIS_BODY = '<body name="chassis" pos="0 0 0.034">'
TETHER = (
    '<tendon><spatial limited="true" range="0 0.37">'
    '<site site="anchor"/><site site="imu"/></spatial></tendon>'
)


FLEX_TETHER = (
    '<deformable><flex name="tether" dim="1" body="world chassis" vertex="0 0 0.6 0 0 0.10"'
    ' element="0 1"><edge stiffness="2000" damping="50"/><contact contype="0" conaffinity="0"/>'
    "</flex></deformable>"
)
# A jointed robot body that is not a wheel: a mast on a hinge, which a flex may spring.
MAST = (
    '<body name="mast" pos="0 0 0.16"><joint name="mast_hinge" type="hinge" axis="1 0 0"/>'
    '<geom name="mast_tip" type="sphere" size="0.01" mass="0.01" contype="0" conaffinity="0"/>'
    "</body>"
)
ROBOT_FLEX = FLEX_TETHER.replace('name="tether"', 'name="bumper"').replace(
    'body="world chassis" vertex="0 0 0.6 0 0 0.10"',
    'body="chassis mast" vertex="0 0 0.10 0 0 0.05"',
)
# Robot-file elements that act on a wheel or a tire's floor contact, or hide the robot's floor
# contacts (§3, §11): each once passed validate_robot.
WHEEL_DRAG = (
    '<tendon><fixed name="drag" damping="0.05"><joint joint="left_wheel" coef="1"/></fixed>'
    "</tendon>"
)
WHEEL_COUPLING = (
    '<tendon><fixed name="diff"><joint joint="left_wheel" coef="1"/>'
    '<joint joint="right_wheel" coef="-1"/></fixed></tendon>'
)
RIM_SITE = '<site name="rim" pos="0.02 0 0"/>'
STRAP = (
    '<tendon><spatial name="strap" stiffness="1"><site site="imu"/><site site="rim"/></spatial>'
    "</tendon>"
)
BELT = (
    '<tendon><spatial name="belt"><site site="imu"/><geom geom="left_tire"/>'
    '<site site="push_point"/></spatial></tendon>'
)
WHEEL_DAMPER = (
    '<deformable><flex name="damper" dim="1" body="chassis left_wheel_body"'
    ' vertex="0 0.07 0.10 0.02 0 0" element="0 1"><edge damping="5"/>'
    '<contact contype="0" conaffinity="0"/></flex></deformable>'
)
TIRE_FLOOR_PAIR = (
    '<contact><pair name="grip" geom1="left_tire" geom2="floor"'
    ' friction="5 5 0.005 0.0001 0.0001"/></contact>'
)
BRAKE_PAIR = '<contact><pair name="brake" geom1="chassis_box" geom2="right_tire"/></contact>'
WORLD_EXCLUDE = '<contact><exclude name="ghost" body1="world" body2="chassis"/></contact>'
ROBOT_EXCLUDE = '<contact><exclude body1="chassis" body2="left_wheel_body"/></contact>'


def after_body_tree(xml: str) -> tuple[str, str]:
    """An edit that adds ``xml`` (tendons, flexes, contacts) after the robot's body tree."""
    return ("</worldbody>", "</worldbody>" + xml)


# A soft block with no passive forces of its own: MuJoCo warns that it is not rigid.
SOFT_BLOCK_WARNING = "ignore:flex 'soft' is not rigid:UserWarning"
SOFT_BLOCK = (
    '<flexcomp name="soft" type="grid" count="2 2 2" spacing="0.02 0.02 0.02" pos="0 0 0.3"'
    ' dim="3" dof="trilinear" mass="0.01"><contact contype="0" conaffinity="0"'
    ' selfcollide="none"/><edge equality="false"/><pin id="0"/>'
    "</flexcomp>"
)
ROTATED_LEFT_WHEEL = [
    (LEFT_BODY, LEFT_BODY.replace('">', '" zaxis="0 1 0">')),
    (LEFT_JOINT, LEFT_JOINT.replace('axis="0 1 0"', 'axis="0 0 1"')),
]
LEFT_TIRE_SIZED = LEFT_TIRE + ' size="0.034 0.013"'
RIGHT_TIRE_SIZED = RIGHT_TIRE + ' size="0.034 0.013"'
LEFT_TIRE_ALONG_Y = LEFT_TIRE_SIZED + ' zaxis="0 1 0"'
RIGHT_TIRE_ALONG_Y = RIGHT_TIRE_SIZED + ' zaxis="0 1 0"'
IMU_SITE = '<site name="imu" pos="0 0 0.10"'
IMU_SITE_ELEMENT = '<site name="imu" pos="0 0 0.10" size="0.006" rgba="1 0.8 0 1"/>'
PUSH_SITE = '<site name="push_point" pos="0 0 0.08"'
PUSH_SITE_ELEMENT = '<site name="push_point" pos="0 0 0.08" size="0.006" rgba="1 0.2 0.2 1"/>'
# A CAD-style wheel body at the motor flange (y = ±0.05 m) with its tire 2 cm further out along
# the axle: the tires still roll 0.14 m apart, as on the stock robot.
FLANGE_WHEELS = [
    (LEFT_BODY, '<body name="left_wheel_body" pos="0 0.05 0">'),
    (LEFT_TIRE_ALONG_Y, LEFT_TIRE_ALONG_Y + ' pos="0 0.02 0"'),
    (RIGHT_BODY, '<body name="right_wheel_body" pos="0 -0.05 0">'),
    (RIGHT_TIRE_ALONG_Y, RIGHT_TIRE_ALONG_Y + ' pos="0 -0.02 0"'),
]
# The same, but the left wheel body is turned so its own z axis is the axle: the tire's offset
# is along the body's z, which only the body's rotation turns into chassis +y.
ROTATED_FLANGE_WHEELS = [
    (LEFT_BODY, '<body name="left_wheel_body" pos="0 0.05 0" zaxis="0 1 0">'),
    (LEFT_JOINT, LEFT_JOINT.replace('axis="0 1 0"', 'axis="0 0 1"')),
    (LEFT_TIRE_ALONG_Y, LEFT_TIRE_SIZED + ' pos="0 0 0.02"'),
    *FLANGE_WHEELS[2:],
]


def free_root(attrs: str) -> str:
    """The root as an explicit free joint carrying ``attrs`` (``<freejoint>`` takes none)."""
    return f'<joint name="root" type="free" {attrs}/>'


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
    "tires a tenth of a millimetre apart": (
        [(RIGHT_TIRE + ' size="0.034 0.013"', RIGHT_TIRE + ' size="0.0341 0.013"')],
        "same radius",
    ),
    "damped root": ([(FREE_ROOT, free_root('damping="1000"'))], "'root' has damping 1000"),
    "root with armature": ([(FREE_ROOT, free_root('armature="100"'))], "'root' has armature 100"),
    "root with friction loss": (
        [(FREE_ROOT, free_root('frictionloss="5"'))],
        "'root' has frictionloss 5",
    ),
    "sprung root": ([(FREE_ROOT, free_root('stiffness="10"'))], "'root' has stiffness 10"),
    "root with nonlinear damping": (
        [(FREE_ROOT, free_root('damping="0 0 1000"'))],
        "'root' has damping 1000",
    ),
    "root with a nonlinear spring": (
        [(FREE_ROOT, free_root('stiffness="0 10"'))],
        "'root' has stiffness 10",
    ),
    "world site": (
        [(WORLD_END, '<site name="anchor" pos="0 0 0.5"/>\n' + WORLD_END)],
        "world site 'anchor'",
    ),
    "tethered by a tendon": (
        [
            (WORLD_END, '<site name="anchor" pos="0 0 0.5"/>\n' + WORLD_END),
            ("</worldbody>", "</worldbody>" + TETHER),
        ],
        "tendon '#0' is tied to the world",
    ),
    "vertical cylinder tire": (
        [(LEFT_TIRE + ' size="0.034 0.013" zaxis="0 1 0"', LEFT_TIRE + ' size="0.034 0.013"')],
        "tire of 'left_wheel' must lie along its hinge axis",
    ),
    "tethered by a flex": (
        [("</worldbody>", "</worldbody>" + FLEX_TETHER)],
        "flex 'tether' is tied to the world",
    ),
    "interpolated flex pinned to the world": (
        [(WORLD_END, SOFT_BLOCK + "\n" + WORLD_END)],
        "flex 'soft' is tied to the world",
    ),
    "eccentric cylinder tire": (
        [(LEFT_TIRE_SIZED, LEFT_TIRE_SIZED + ' pos="0.02 0 0"')],
        "tire of 'left_wheel' must be centred on its hinge axis",
    ),
    "eccentric sphere tire": (
        [
            (
                LEFT_TIRE_SIZED + ' zaxis="0 1 0"',
                '<geom name="left_tire" type="sphere" size="0.034"',
            ),
            (
                RIGHT_TIRE_SIZED + ' zaxis="0 1 0"',
                '<geom name="right_tire" type="sphere" size="0.034" pos="0 0 0.01"',
            ),
        ],
        "tire of 'right_wheel' must be centred on its hinge axis",
    ),
    "hinge off the tire centre": (
        [(LEFT_JOINT, LEFT_JOINT.replace("/>", ' pos="0.02 0 0"/>'))],
        "tire of 'left_wheel' must be centred on its hinge axis",
    ),
    "tire left along y on a rotated wheel body": (
        ROTATED_LEFT_WHEEL,
        "tire of 'left_wheel' must lie along its hinge axis",
    ),
    "tilted capsule tire": (
        [
            (
                RIGHT_TIRE + ' size="0.034 0.013" zaxis="0 1 0"',
                '<geom name="right_tire" type="capsule" size="0.034 0.013" zaxis="0 1 0.01"',
            )
        ],
        "tire of 'right_wheel' must lie along its hinge axis",
    ),
    # MuJoCo's IMU sensors read in the site's frame, which the estimator takes as the chassis's.
    "imu board turned a quarter turn": (
        [(IMU_SITE, IMU_SITE + ' euler="0 0 1.5707963267948966"')],
        "site 'imu' must not be rotated relative to body 'chassis'",
    ),
    "imu board pitched by 2 mrad": (
        [(IMU_SITE, IMU_SITE + ' euler="0 0.002 0"')],
        "site 'imu' must not be rotated relative to body 'chassis'",
    ),
    "imu board on a wheel": (
        [(IMU_SITE_ELEMENT, ""), (LEFT_JOINT, LEFT_JOINT + '<site name="imu" pos="0 0 0.02"/>')],
        "site 'imu' must be on body 'chassis'",
    ),
    "push point on a wheel": (
        [
            (PUSH_SITE_ELEMENT, ""),
            (LEFT_JOINT, LEFT_JOINT + '<site name="push_point" pos="0 0 0.02"/>'),
        ],
        "site 'push_point' must be on body 'chassis'",
    ),
    # The env tilts the robot about the chassis origin at reset: only an axle through it keeps
    # the tires on the floor and rolling without slip (contract §2, §3).
    "axle below the chassis origin": (
        [
            (LEFT_BODY, '<body name="left_wheel_body" pos="0 0.07 -0.01">'),
            (RIGHT_BODY, '<body name="right_wheel_body" pos="0 -0.07 -0.01">'),
        ],
        "tire of 'left_wheel' must be centred on the chassis y axis",
    ),
    "axle ahead of the chassis origin": (
        [
            (LEFT_BODY, '<body name="left_wheel_body" pos="0.01 0.07 0">'),
            (RIGHT_BODY, '<body name="right_wheel_body" pos="0.01 -0.07 0">'),
        ],
        "tire of 'right_wheel' must be centred on the chassis y axis",
    ),
    "hinge and tire below a wheel body on the axle": (
        [
            (LEFT_JOINT, LEFT_JOINT.replace("/>", ' pos="0 0 -0.01"/>')),
            (LEFT_TIRE_ALONG_Y, LEFT_TIRE_ALONG_Y + ' pos="0 0 -0.01"'),
        ],
        "tire of 'left_wheel' must be centred on the chassis y axis",
    ),
    "chassis origin off the axle midpoint": (
        [(LEFT_BODY, '<body name="left_wheel_body" pos="0 0.08 0">')],
        "the chassis origin must be the axle midpoint",
    ),
    "tires crossed under their wheel bodies": (
        [
            (LEFT_BODY, '<body name="left_wheel_body" pos="0 0.01 0">'),
            (LEFT_TIRE_ALONG_Y, LEFT_TIRE_ALONG_Y + ' pos="0 -0.08 0"'),
            (RIGHT_BODY, '<body name="right_wheel_body" pos="0 -0.01 0">'),
            (RIGHT_TIRE_ALONG_Y, RIGHT_TIRE_ALONG_Y + ' pos="0 0.08 0"'),
        ],
        "tire of 'left_wheel' must be on the left (y > 0)",
    ),
    # A wheel's dynamics come from params.yaml alone (§3): no tendon or flex may act on a wheel.
    "damped fixed tendon on a wheel": (
        [after_body_tree(WHEEL_DRAG)],
        "tendon 'drag' acts on a wheel",
    ),
    "fixed tendon coupling the wheels": (
        [after_body_tree(WHEEL_COUPLING)],
        "tendon 'diff' acts on a wheel",
    ),
    "spring to a site on a wheel": (
        [(LEFT_JOINT, LEFT_JOINT + RIM_SITE), after_body_tree(STRAP)],
        "tendon 'strap' acts on a wheel",
    ),
    "spring to a site on a body below a wheel": (
        [(LEFT_JOINT, LEFT_JOINT + f'<body name="hub">{RIM_SITE}</body>'), after_body_tree(STRAP)],
        "tendon 'strap' acts on a wheel",
    ),
    "tendon wrapped around a tire": ([after_body_tree(BELT)], "tendon 'belt' acts on a wheel"),
    "flex damper on a wheel": (
        [after_body_tree(WHEEL_DAMPER)],
        "flex 'damper' is tied to a wheel",
    ),
    # The env sets every tire-floor contact's parameters (§3) and ends an episode when anything
    # but a tire touches the floor (§11).
    "contact pair between a tire and the floor": (
        [after_body_tree(TIRE_FLOOR_PAIR)],
        "contact pair 'grip' is not allowed",
    ),
    "contact pair between the chassis and a tire": (
        [after_body_tree(BRAKE_PAIR)],
        "contact pair 'brake' is not allowed",
    ),
    "contact exclude between the world and the chassis": (
        [after_body_tree(WORLD_EXCLUDE)],
        "contact exclude 'ghost' between 'world' and 'chassis' is not allowed",
    ),
    # A reset keeps the root's position from the MJCF and tilts the chassis about the axle (§3,
    # §9): only a chassis whose origin starts one tire radius above the floor spawns with its
    # tires touching it (§5).
    "chassis spawned at the floor": (
        [(CHASSIS_BODY, '<body name="chassis" pos="0 0 0">')],
        "the chassis must start with its origin (the axle) at z = 0.034 m, the tire radius, "
        "so the tires spawn touching the floor; got z = 0 m",
    ),
    "chassis spawned in the air": (
        [(CHASSIS_BODY, '<body name="chassis" pos="0 0 0.2">')],
        "got z = 0.2 m",
    ),
    "chassis a tenth of a millimetre into the floor": (
        [(CHASSIS_BODY, '<body name="chassis" pos="0 0 0.0339">')],
        "got z = 0.0339 m",
    ),
    # The root's height is then a mount's, not the chassis's: no spawn height is checked.
    "root on a mount above the chassis": (
        [
            (FREE_ROOT, ""),
            (CHASSIS_BODY, f'<body name="mount" pos="0 0 0.034">{FREE_ROOT}<body name="chassis">'),
            ("    </body>\n  </worldbody>", "    </body></body>\n  </worldbody>"),
        ],
        "free joint 'root' on body 'chassis' is missing",
    ),
}


@pytest.mark.filterwarnings(SOFT_BLOCK_WARNING)
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


@pytest.mark.parametrize(
    "edits", [FLANGE_WHEELS, ROTATED_FLANGE_WHEELS], ids=["flange", "rotated flange"]
)
def test_track_width_is_the_distance_between_the_tire_centres(
    params: RobotParams, tmp_path: Path, edits: list[tuple[str, str]]
) -> None:
    """§3: the track width is measured between the tires' centres in the chassis frame, where
    they roll, not between the wheel bodies' origins (0.10 m here): odometry (§5 step 6) and
    ``TaskSpec.robot`` need the lever arm the tires have."""
    m = model_from(params, tmp_path, *edits)
    assert validate_robot(m) == []
    b = bind_robot(m)
    assert b.track_width_m == pytest.approx(0.14, abs=1e-12)
    assert b.wheel_radius_m == pytest.approx(0.034)


def test_bind_robot_refuses_ambiguous_tires(params: RobotParams, tmp_path: Path) -> None:
    m = model_from(params, tmp_path, (LEFT_JOINT, LEFT_JOINT + '<geom type="sphere" size="0.02"/>'))
    with pytest.raises(ValueError, match="exactly one collision geom"):
        bind_robot(m)


# ----------------------------------------------------------------------- what build_model injects


def test_tires_take_the_params_sliding_friction_and_govern_their_contacts(
    params: RobotParams, tmp_path: Path
) -> None:
    """§3: every wheel-body geom gets the params' sliding friction and priority 1, so a
    tire-floor contact takes its parameters from the tire alone: the params' sliding friction
    (whatever the MJCF says), and the tire's own condim and torsional and rolling friction, even
    below the floor's. With equal priorities MuJoCo would mix them, the frictions by maximum."""
    tire = LEFT_TIRE_ALONG_Y
    m = model_from(params, tmp_path, (tire, tire + ' friction="0.4 0.001 0.0002" condim="4"'))
    left, floor = m.geom("left_tire").id, m.geom("floor").id
    assert m.geom_friction[left].tolist() == [params.contact.wheel_friction, 0.001, 0.0002]
    assert (m.geom_priority[left], m.geom_priority[floor]) == (1, 0)
    assert m.geom_friction[floor][1] > 0.001  # the floor's own torsional friction is larger
    d = mujoco.MjData(m)
    for _ in range(3):  # the tires spawn exactly touching the floor: let them settle into it
        mujoco.mj_step(m, d)
    contacts = [
        d.contact[i] for i in range(d.ncon) if left in (d.contact[i].geom1, d.contact[i].geom2)
    ]
    assert contacts
    for contact in contacts:
        assert contact.dim == 4
        assert (
            contact.friction.tolist()
            == [params.contact.wheel_friction] * 2 + [0.001] + [0.0002] * 2
        )


def test_the_replay_camera_name_is_reserved(params: RobotParams, tmp_path: Path) -> None:
    """§3: build_model adds the replay camera ``track`` to the chassis, so a robot may not bring
    a camera of that name. It once failed with MuJoCo's bare "repeated name 'track' in camera"."""
    own = '<camera name="track" pos="0 -1 0.2"/>'
    with pytest.raises(ValueError, match=r"balancer\.robot\.v1: camera 'track' is reserved"):
        model_from(params, tmp_path, (IMU_SITE, own + IMU_SITE))
    with pytest.raises(ValueError, match=r"balancer\.robot\.v1: camera 'track' is reserved"):
        BalanceEnv("clean", robot_xml=tmp_path / "variant.xml")  # what model_from just wrote
    m = model_from(params, tmp_path, (IMU_SITE, own.replace("track", "side") + IMU_SITE))
    assert validate_robot(m) == []
    assert sorted(m.camera(i).name for i in range(m.ncam)) == ["side", "track"]
    assert m.cam_bodyid[m.camera("track").id] == m.body("chassis").id


ASSETS_BEFORE = "<worldbody>"
SCENE_NAMES = {
    "material": ((ASSETS_BEFORE, '<asset><material name="grid"/></asset><worldbody>'), "material"),
    "texture": (
        (
            ASSETS_BEFORE,
            '<asset><texture name="grid" type="2d" builtin="flat" width="8" height="8"/>'
            "</asset><worldbody>",
        ),
        "texture",
    ),
    "geom": (
        (
            IMU_SITE,
            '<geom name="floor" type="sphere" size="0.01" contype="0" conaffinity="0"/>' + IMU_SITE,
        ),
        "geom",
    ),
    "light": ((IMU_SITE, '<light name="key"/>' + IMU_SITE), "light"),
}


@pytest.mark.parametrize(("edit", "kind"), SCENE_NAMES.values(), ids=SCENE_NAMES.keys())
def test_a_robot_may_not_reuse_a_name_the_scene_gives_an_element_of_its_kind(
    params: RobotParams, tmp_path: Path, edit: tuple[str, str], kind: str
) -> None:
    """§3: the robot is attached with prefix "", so a name the scene already gives an element of
    the same kind (balance-v1: texture and material ``grid``, lights ``key`` and ``fill``, geom
    ``floor``) would name two elements. Such robots once failed with MuJoCo's bare "Error:
    repeated name 'grid' in material"; a name the scene uses for another kind is fine."""
    name = re.search(r'name="(\w+)"', edit[1])
    assert name is not None
    expected = rf"balancer\.robot\.v1: {kind} '{name.group(1)}' is the scene's; rename the robot's"
    with pytest.raises(ValueError, match=expected):
        model_from(params, tmp_path, edit)
    with pytest.raises(ValueError, match=expected):
        BalanceEnv("clean", robot_xml=tmp_path / "variant.xml")  # what model_from just wrote
    elsewhere = '<site name="grid" pos="0 0 0.05"/><site name="floor" pos="0 0 0.06"/>'
    assert validate_robot(model_from(params, tmp_path, (IMU_SITE, elsewhere + IMU_SITE))) == []


def array_fields(m: mujoco.MjModel) -> list[str]:
    """Every array a compiled model holds: its numbers, names and structure."""
    return [name for name in dir(m) if isinstance(getattr(m, name, None), np.ndarray)]


RIGHT_JOINT = '<joint name="right_wheel" type="hinge" axis="0 1 0"/>'
WHEEL_JOINT_EXTRAS = {
    "spring": 'stiffness="0.05"',
    "polynomial spring": 'stiffness="0 0.5"',
    "spring-damper": 'springdamper="0.1 1"',
    "damping": 'damping="0.02"',
    "polynomial damping": 'damping="0 0.5"',
    "joint torque cap": 'actuatorfrcrange="-0.01 0.01"',
    "friction-loss solver": 'solreffriction="0.5 1" solimpfriction="0.5 0.6 0.01"',
}


@pytest.mark.parametrize("attrs", WHEEL_JOINT_EXTRAS.values(), ids=WHEEL_JOINT_EXTRAS.keys())
def test_a_wheel_joints_own_dynamics_never_reach_the_simulated_wheel(
    params: RobotParams, model: mujoco.MjModel, tmp_path: Path, attrs: str
) -> None:
    """§3: a wheel's dynamics come from params.yaml alone. A robot file's wheel spring, damping,
    joint torque cap or friction-loss solver settings are replaced, so the composed robot is the
    stock one, bit for bit. A 0.05 N·m/rad spring or a 0.01 N·m cap once passed validate_robot,
    and the shipped PID then survived none of the clean seeds it was tried on."""
    edits = [(joint, joint.replace("/>", f" {attrs}/>")) for joint in (LEFT_JOINT, RIGHT_JOINT)]
    m = model_from(params, tmp_path, *edits)
    assert validate_robot(m) == []
    for field in array_fields(model):
        got, want = getattr(m, field), getattr(model, field)
        assert (got.dtype, got.shape, got.tobytes()) == (want.dtype, want.shape, want.tobytes()), (
            field
        )


def test_the_friction_loss_solver_settings_are_mujocos_defaults() -> None:
    joint = mujoco.MjSpec().worldbody.add_body().add_joint()
    assert tuple(joint.solref_friction) == FRICTIONLOSS_SOLREF
    assert tuple(joint.solimp_friction) == FRICTIONLOSS_SOLIMP


# ------------------------------------------------------------------------------- robot_hash


def test_balance_v1_robot_hash_is_unchanged(params: RobotParams) -> None:
    """balance-v1 references no asset files, so covering them leaves its hash as it was."""
    assert referenced_files(ROBOT_XML) == ()
    assert robot_hash(ROBOT_XML, params) == BALANCE_V1_ROBOT_HASH


def test_robot_hash_recipe_is_reproducible_from_the_contract(params: RobotParams) -> None:
    """§3 written out as a port would: the MJCF bytes, then the params' JSON text exactly as
    pinned above (a compact or JavaScript serializer would write ``30`` and change the hash)."""
    assert json.dumps(params.model_dump(), sort_keys=True) == BALANCE_V1_PARAMS_JSON
    digest = hashlib.sha256(ROBOT_XML.read_bytes() + BALANCE_V1_PARAMS_JSON.encode("ascii"))
    assert digest.hexdigest() == BALANCE_V1_ROBOT_HASH


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


def to_crlf(path: Path) -> None:
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))


def test_robot_hash_ignores_line_endings_like_the_bundle_hash(
    params: RobotParams, tmp_path: Path
) -> None:
    """A CRLF checkout of the stock robot is the stock robot (contract §3, §13)."""
    crlf = tmp_path / "balancer.xml"
    crlf.write_bytes(ROBOT_XML.read_bytes())
    to_crlf(crlf)
    assert crlf.read_bytes() != ROBOT_XML.read_bytes()
    assert robot_hash(crlf, params) == BALANCE_V1_ROBOT_HASH


def test_robot_hash_normalizes_text_files_only(params: RobotParams, tmp_path: Path) -> None:
    """Included MJCF files are text (CRLF -> LF, as in the bundle hash); a mesh is data and
    counts byte for byte."""
    lf, crlf = mesh_robot(tmp_path / "lf"), mesh_robot(tmp_path / "crlf")
    include = '<compiler angle="radian" autolimits="true"/><include file="part.xml"/>'
    for path in (lf, crlf):
        path.write_text(
            path.read_text().replace('<compiler angle="radian" autolimits="true"/>', include)
        )
        (path.parent / "part.xml").write_text(
            '<mujoco>\n<default><geom rgba="1 0 0 1"/></default>\n</mujoco>\n'
        )
    to_crlf(crlf)
    to_crlf(crlf.parent / "part.xml")
    assert robot_hash(crlf, params) == robot_hash(lf, params)
    to_crlf(crlf.parent / "assets" / "logo.obj")
    assert robot_hash(crlf, params) != robot_hash(lf, params)


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


def submodel_robot(robot_dir: Path, part_compiler: str = "") -> Path:
    """A robot that attaches ``parts/part.xml`` (an ``<asset><model>``), which has its own mesh.

    MuJoCo loads a sub-model like a model of its own: its files resolve against its directory
    and its own ``<compiler>``, never against the top-level MJCF's.
    """
    parts = robot_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    (parts / "part.xml").write_text(
        f'<mujoco><compiler angle="radian" {part_compiler}/>'
        '<asset><mesh name="m" file="mesh.obj"/></asset>'
        '<worldbody><body name="pb"><geom type="mesh" mesh="m" contype="0" conaffinity="0"'
        ' mass="0.001"/></body></worldbody></mujoco>\n'
    )
    compiler = '<compiler angle="radian" autolimits="true"/>'
    xml = (
        ROBOT_XML.read_text()
        .replace(compiler, compiler + '<asset><model name="part" file="parts/part.xml"/></asset>')
        .replace('<site name="imu"', '<attach model="part" body="pb" prefix="p_"/><site name="imu"')
    )
    path = robot_dir / "balancer.xml"
    path.write_text(xml)
    return path


SUBMODEL_LAYOUTS = {
    "next to the sub-model": ("", "parts/mesh.obj"),
    "sub-model meshdir": ('meshdir="pm"', "parts/pm/mesh.obj"),
}


@pytest.mark.parametrize(
    ("part_compiler", "mesh_rel"), SUBMODEL_LAYOUTS.values(), ids=SUBMODEL_LAYOUTS.keys()
)
def test_robot_hash_covers_the_files_of_an_attached_sub_model(
    params: RobotParams, tmp_path: Path, part_compiler: str, mesh_rel: str
) -> None:
    path = submodel_robot(tmp_path / "robot", part_compiler)
    base = path.parent
    mesh = base / mesh_rel
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_text(MESH)
    for decoy in ("mesh.obj", "pm/mesh.obj"):  # where a resolver using the top level would look
        (base / decoy).parent.mkdir(parents=True, exist_ok=True)
        (base / decoy).write_text(MESH)
    model = build_model(path, SCENE_XML, params)
    assert validate_robot(model) == []
    assert referenced_files(path) == tuple(
        sorted([(base / "parts" / "part.xml").resolve(), mesh.resolve()])
    )
    before = robot_hash(path, params)
    mesh.write_text(MESH.replace("0.01", "0.05"))
    assert robot_hash(path, params) != before
    mesh.unlink()  # MuJoCo needs exactly this file: the decoys do not stand in for it
    with pytest.raises(ValueError, match=r"mesh\.obj"):
        build_model(path, SCENE_XML, params)
    with pytest.raises(FileNotFoundError, match=r"mesh\.obj"):
        robot_hash(path, params)


def test_sub_models_that_attach_each_other_are_hashed_once(tmp_path: Path) -> None:
    """Two sub-models naming each other cannot send ``referenced_files`` round in circles."""
    a = tmp_path / "a.xml"
    b = tmp_path / "b.xml"
    a.write_text('<mujoco><asset><model name="b" file="b.xml"/></asset></mujoco>\n')
    b.write_text('<mujoco><asset><model name="a" file="a.xml"/></asset></mujoco>\n')
    assert referenced_files(a) == tuple(sorted([a.resolve(), b.resolve()]))


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
        (
            [(LEFT_JOINT, LEFT_JOINT.replace('name="left_wheel"', 'name="lw"'))],
            "joint 'left_wheel' is missing",
        ),
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


@pytest.mark.filterwarnings(SOFT_BLOCK_WARNING)
@pytest.mark.parametrize(
    "edits",
    [
        [
            (
                LEFT_TIRE + ' size="0.034 0.013" zaxis="0 1 0"',
                LEFT_TIRE + ' size="0.034 0.013" zaxis="0 -1 0"',
            ),
        ],
        [
            (
                LEFT_TIRE + ' size="0.034 0.013" zaxis="0 1 0"',
                '<geom name="left_tire" type="sphere" size="0.034"',
            ),
            (
                RIGHT_TIRE + ' size="0.034 0.013" zaxis="0 1 0"',
                '<geom name="right_tire" type="sphere" size="0.034"',
            ),
        ],
        [
            (
                LEFT_TIRE + ' size="0.034 0.013"',
                '<geom name="left_tire" type="capsule" size="0.034 0.013"',
            ),
            (
                RIGHT_TIRE + ' size="0.034 0.013"',
                '<geom name="right_tire" type="capsule" size="0.034 0.013"',
            ),
        ],
        [(RIGHT_TIRE + ' size="0.034 0.013"', RIGHT_TIRE + ' size="0.0340000000001 0.013"')],
        [(FREE_ROOT, free_root('damping="0" armature="0" frictionloss="0" stiffness="0"'))],
        [
            (
                '<site name="imu"',
                '<site name="anchor" pos="0 0 0.02"/><site name="imu"',
            ),
            ("</worldbody>", "</worldbody>" + TETHER),
        ],
        [*ROTATED_LEFT_WHEEL, (LEFT_TIRE_SIZED + ' zaxis="0 1 0"', LEFT_TIRE_SIZED)],
        [(LEFT_TIRE_SIZED + ' zaxis="0 1 0"', LEFT_TIRE_SIZED + ' zaxis="0 1 0.0001"')],
        [(LEFT_JOINT, LEFT_JOINT.replace("/>", ' pos="0 0.01 0"/>'))],
        [
            (
                "</worldbody>",
                '</worldbody><tendon><spatial name="block"><site site="imu"/>'
                '<site site="push_point"/><pulley divisor="2"/><site site="push_point"/>'
                '<site site="imu"/></spatial></tendon>',
            )
        ],
        [(IMU_SITE, MAST + IMU_SITE), after_body_tree(ROBOT_FLEX)],
        [('<site name="imu"', SOFT_BLOCK + '<site name="imu"')],
        [(IMU_SITE, IMU_SITE + ' euler="0 0.001 0"')],
        [(PUSH_SITE, PUSH_SITE + ' euler="0 0 1"')],
        FLANGE_WHEELS,
        ROTATED_FLANGE_WHEELS,
        [after_body_tree(ROBOT_EXCLUDE)],
        [(CHASSIS_BODY, '<body name="chassis" pos="0.5 -0.3 0.034" euler="0 0 1">')],
    ],
    ids=[
        "flipped cylinder axis",
        "sphere tires",
        "capsule tires",
        "radii within tolerance",
        "explicit zero root properties",
        "tendon between robot sites",
        "rotated wheel body with its tire along the hinge",
        "cylinder axis within tolerance",
        "hinge placed further along the axle",
        "pulley between robot sites",
        "flex between the chassis and a jointed body",
        "interpolated flex on the robot",
        "imu board within the axis tolerance",
        "rotated push point (only its position counts)",
        "tires offset along the axle from their wheel bodies",
        "rotated wheel body with its tire offset along the axle",
        "contact exclude inside the robot (it only removes contacts the robot has with itself)",
        "chassis placed elsewhere on the floor and turned (a reset sets its orientation)",
    ],
)
def test_validate_accepts_variants_that_can_still_fall(
    params: RobotParams, tmp_path: Path, edits: list[tuple[str, str]]
) -> None:
    assert validate_robot(model_from(params, tmp_path, *edits)) == []


def test_validate_reports_a_tendon_wrapped_around_a_scene_geom(
    params: RobotParams, tmp_path: Path
) -> None:
    """A tendon between robot sites that wraps a world geom still ties the robot to the world."""
    scene = tmp_path / "scene.xml"
    scene.write_text(
        SCENE_XML.read_text().replace(
            '<geom name="floor"',
            '<geom name="post" type="sphere" size="0.02" pos="0 0 0.5" contype="0"'
            ' conaffinity="0"/>\n    <geom name="floor"',
        )
    )
    xml = ROBOT_XML.read_text().replace(
        "</worldbody>",
        '</worldbody><tendon><spatial limited="true" range="0 0.5"><site site="imu"/>'
        '<geom geom="post"/><site site="push_point"/></spatial></tendon>',
    )
    robot = tmp_path / "robot.xml"
    robot.write_text(xml)
    model = build_model(robot, scene, params)
    problems = validate_robot(model, scene_geoms=frozenset({"floor", "post"}))
    assert len(problems) == 1, problems
    assert problems[0].startswith("tendon '#0' is tied to the world"), problems


@pytest.mark.parametrize(
    "case",
    [
        "damped root",
        "root with armature",
        "tethered by a tendon",
        "tethered by a flex",
        "vertical cylinder tire",
        "eccentric cylinder tire",
    ],
)
def test_the_env_refuses_a_robot_that_cannot_fall_or_misstates_its_radius(
    tmp_path: Path, case: str
) -> None:
    """Each of these once ran: a damped or tethered robot survived 1000 idle steps from 0.3 rad,
    and a vertical tire reported a radius it does not roll on."""
    xml = ROBOT_XML.read_text()
    for old, new in VIOLATIONS[case][0]:
        xml = xml.replace(old, new, 1)
    path = tmp_path / "variant.xml"
    path.write_text(xml)
    with pytest.raises(ValueError, match=r"balancer\.robot\.v1"):
        BalanceEnv("ideal", robot_xml=path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("imu board turned a quarter turn", "site 'imu' must not be rotated"),
        ("axle below the chassis origin", "must be centred on the chassis y axis"),
        ("chassis origin off the axle midpoint", "axle midpoint"),
        ("chassis spawned at the floor", "the chassis must start with its origin"),
        ("chassis spawned in the air", "the chassis must start with its origin"),
    ],
)
def test_the_env_refuses_a_robot_it_would_sense_or_spawn_wrongly(
    tmp_path: Path, case: str, message: str
) -> None:
    """Each of these once ran. A board turned a quarter turn on its site fed the estimator the
    chassis's x axis as its y: from 0.1 rad and 20 idle steps the observed pitch read 0.07 while
    the robot lay at 0.21 and fell at 1.3 rad/s, and the shipped PID survived 0 of 10 clean
    seeds. With the chassis frame 8 cm above the axle, a reset at 0.5 rad spawned the tires
    9.8 mm above the floor and 38 mm off in x. With the chassis at z = 0, clean seeds 1000-1002
    all ended at step 1 with floor_contact; at z = 0.2 m the robot dropped 166 mm first and the
    shipped PID scored 88.9-91.5 there, against 96.0-96.4 for the stock robot."""
    xml = ROBOT_XML.read_text()
    for old, new in VIOLATIONS[case][0]:
        xml = xml.replace(old, new, 1)
    path = tmp_path / "variant.xml"
    path.write_text(xml)
    with pytest.raises(ValueError, match=rf"balancer\.robot\.v1.*{message}"):
        BalanceEnv("clean", robot_xml=path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("damped fixed tendon on a wheel", "tendon 'drag' acts on a wheel"),
        ("flex damper on a wheel", "flex 'damper' is tied to a wheel"),
        ("contact pair between a tire and the floor", "contact pair 'grip' is not allowed"),
        ("contact exclude between the world and the chassis", "contact exclude 'ghost'"),
    ],
)
def test_the_env_refuses_a_robot_that_changes_its_wheels_or_floor_contacts(
    tmp_path: Path, case: str, message: str
) -> None:
    """§3: a wheel's dynamics come from params.yaml alone and a tire-floor contact's parameters
    from the tire alone; §11: only the tires may touch the floor. Each of these once ran. Fixed
    tendons with damping 0.05 on both wheels made the shipped PID fall on clean seeds 1000-1002
    at steps 212, 138 and 72 (the stock robot survives all three, scoring about 96); a tire-floor
    pair held the contact's friction at its own value on every seed, whatever params.yaml and
    the scenario's friction scale (§10) said; and an exclude between the world and the chassis
    let a tail reach through the floor for 1000 steps (score 96.21) where the same robot without
    it ends at step 1 with floor_contact."""
    xml = ROBOT_XML.read_text()
    for old, new in VIOLATIONS[case][0]:
        xml = xml.replace(old, new, 1)
    path = tmp_path / "variant.xml"
    path.write_text(xml)
    with pytest.raises(ValueError, match=rf"balancer\.robot\.v1.*{message}"):
        BalanceEnv("clean", robot_xml=path)
