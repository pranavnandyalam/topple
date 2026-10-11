"""Robot interface ``balancer.robot.v1``: parameters, model building, binding, validation.

The robot MJCF holds geometry and inertia only. ``build_model`` composes it into the challenge
scene with ``mujoco.MjSpec`` and injects motors, IMU sensors and joint friction from
``params.yaml``, so a placeholder, a CAD export or an evolved body all become the same simulated
robot (docs/contract.md §3).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import mujoco
import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

CHASSIS = "chassis"
ROOT_JOINT = "root"
WHEEL_JOINTS = ("left_wheel", "right_wheel")
MOTORS = ("left_motor", "right_motor")
IMU_SITE = "imu"
PUSH_SITE = "push_point"
GYRO_SENSOR = "imu_gyro"
ACCEL_SENSOR = "imu_accel"
TRACK_CAMERA = "track"
DELAY_HISTORY = 8  # native actuator delay buffer: enough for 10 ms at 2 ms per sample
SCENE_GEOMS = frozenset({"floor"})  # the only world-body geoms a balance scene may have
TIRE_TYPES = frozenset(
    int(t)
    for t in (
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
    )
)
_FILE_ATTRS = ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")
_MESH_DIR_TAGS = frozenset({"mesh", "skin", "hfield"})  # MuJoCo resolves these via meshdir


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MotorParams(_Frozen):
    tau_stall_nm: PositiveFloat
    omega_noload_rads: PositiveFloat
    torque_limit_nm: PositiveFloat
    stall_current_a: PositiveFloat
    v_nom: PositiveFloat
    armature: PositiveFloat
    frictionloss_nm: PositiveFloat


class ContactParams(_Frozen):
    wheel_friction: PositiveFloat


class SensingParams(_Frozen):
    encoder_cpr: PositiveInt
    gyro_noise_rads: float
    accel_noise_ms2: float
    gyro_range_rads: PositiveFloat
    accel_range_ms2: PositiveFloat


class EstimationParams(_Frozen):
    cf_tau_s: PositiveFloat


class SafetyParams(_Frozen):
    fall_angle_rad: PositiveFloat


class RobotParams(_Frozen):
    motor: MotorParams
    contact: ContactParams
    sensing: SensingParams
    estimation: EstimationParams
    safety: SafetyParams


def challenge_dir(challenge_id: str) -> Path:
    """Directory of a built-in challenge bundle shipped inside the package."""
    path = Path(str(resources.files("topple").joinpath("challenges", challenge_id)))
    if not path.is_dir():
        raise FileNotFoundError(f"unknown challenge bundle {challenge_id!r} (looked in {path})")
    return path


def load_params(path: Path) -> RobotParams:
    return RobotParams.model_validate(yaml.safe_load(path.read_text()))


def _mjcf_trees(path: Path, model_dir: Path, seen: set[Path]) -> list[ET.Element]:
    """The root element of ``path`` and of every file it ``<include>``s, recursively."""
    root = ET.parse(path).getroot()
    trees = [root]
    for include in root.iter("include"):
        target = _resolve(model_dir, "", include.get("file", ""), path)
        if target not in seen:
            seen.add(target)
            trees += _mjcf_trees(target, model_dir, seen)
    return trees


def _resolve(model_dir: Path, subdir: str, name: str, owner: Path) -> Path:
    path = (model_dir / subdir / name).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{owner.name} references {name}, which does not exist ({path})")
    return path


def _compiler_dirs(trees: list[ET.Element]) -> tuple[str, str]:
    """(meshdir, texturedir) as MuJoCo applies them: ``assetdir`` sets both, specific ones win."""
    meshdir = texturedir = ""
    for tree in trees:
        for compiler in tree.iter("compiler"):
            assetdir = compiler.get("assetdir")
            meshdir = compiler.get("meshdir", assetdir if assetdir is not None else meshdir)
            texturedir = compiler.get(
                "texturedir", assetdir if assetdir is not None else texturedir
            )
    return meshdir, texturedir


def referenced_files(xml_path: Path) -> tuple[Path, ...]:
    """Every file an MJCF references, resolved the way MuJoCo resolves it, sorted.

    Covers ``<include>`` (recursively), meshes, skins and height fields (``meshdir``), textures
    including cube faces (``texturedir``), and any other element with a ``file`` attribute
    (relative to the model's directory). A missing file raises ``FileNotFoundError``.
    """
    model_dir = xml_path.parent
    includes: set[Path] = set()
    trees = _mjcf_trees(xml_path, model_dir, includes)
    meshdir, texturedir = _compiler_dirs(trees)
    found = set(includes)
    for tree in trees:
        for element in tree.iter():
            if element.tag in ("include", "compiler"):
                continue
            subdir = (
                meshdir
                if element.tag in _MESH_DIR_TAGS
                else texturedir
                if element.tag == "texture"
                else ""
            )
            for attr in _FILE_ATTRS:
                name = element.get(attr)
                if name:
                    found.add(_resolve(model_dir, subdir, name, xml_path))
    return tuple(sorted(found))


def _framed(data: bytes) -> bytes:
    return len(data).to_bytes(8, "big") + data


def robot_hash(robot_xml: Path, params: RobotParams) -> str:
    """sha256 of the robot MJCF, its params and every file the MJCF references.

    Referenced files are appended as length-framed (POSIX path relative to the MJCF's directory,
    bytes) records in POSIX-path string order (``a-c.obj`` before ``a/b.obj``), so a robot without
    asset files hashes exactly as before they were covered.
    """
    digest = hashlib.sha256()
    digest.update(robot_xml.read_bytes())
    digest.update(json.dumps(params.model_dump(), sort_keys=True).encode())
    base = robot_xml.parent.resolve()
    records = sorted(
        (Path(os.path.relpath(path, base)).as_posix(), path) for path in referenced_files(robot_xml)
    )
    for rel, path in records:  # POSIX-path string order, as in the bundle hash (contract §3)
        digest.update(_framed(rel.encode("utf-8")))
        digest.update(_framed(path.read_bytes()))
    return digest.hexdigest()


def _require(spec: mujoco.MjSpec) -> None:
    names = {
        "body": {b.name for b in spec.bodies},
        "joint": {j.name for j in spec.joints},
        "site": {s.name for s in spec.sites},
    }
    required = [("body", CHASSIS), ("joint", ROOT_JOINT), ("site", IMU_SITE), ("site", PUSH_SITE)]
    required += [("joint", j) for j in WHEEL_JOINTS]
    missing = [f"{kind} '{name}'" for kind, name in required if name not in names[kind]]
    if missing:
        raise ValueError(f"robot is missing required elements: {', '.join(missing)}")


def build_model(
    robot_xml: Path, scene_xml: Path, params: RobotParams, *, delay_buffer: bool = True
) -> mujoco.MjModel:
    """Compose robot + scene and inject motors, sensors and joint properties from ``params``.

    ``delay_buffer=False`` omits the native actuator history buffer; use it for linearization
    (``mjd_transitionFD`` does not support delays), never for evaluation.
    """
    scene = mujoco.MjSpec.from_file(str(scene_xml))
    robot = mujoco.MjSpec.from_file(str(robot_xml))
    _require(robot)
    for actuator in list(robot.actuators):
        robot.delete(actuator)
    for sensor in list(robot.sensors):
        robot.delete(sensor)
    scene.attach(robot, prefix="", frame=scene.worldbody.add_frame())

    m = params.motor
    for joint_name, motor_name in zip(WHEEL_JOINTS, MOTORS, strict=True):
        joint = scene.joint(joint_name)
        joint.damping = [0.0, 0.0, 0.0]  # back-EMF lives in the actuator bias
        joint.frictionloss = m.frictionloss_nm
        joint.armature = m.armature
        for geom in scene.body(joint.parent.name).geoms:
            geom.friction[0] = params.contact.wheel_friction
            # MuJoCo combines equal-priority contacts with max(); priority makes the tire's own
            # friction (and its per-episode randomization) govern tire-floor contacts.
            geom.priority = 1

        act = scene.add_actuator(name=motor_name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = joint_name
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = m.tau_stall_nm
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[0:3] = [0.0, 0.0, -m.tau_stall_nm / m.omega_noload_rads]
        act.ctrllimited = True
        act.ctrlrange = [-1.0, 1.0]
        act.forcelimited = True
        act.forcerange = [-m.torque_limit_nm, m.torque_limit_nm]
        act.nsample = DELAY_HISTORY if delay_buffer else 0
        act.delay = 0.0

    for name, kind in (
        (GYRO_SENSOR, mujoco.mjtSensor.mjSENS_GYRO),
        (ACCEL_SENSOR, mujoco.mjtSensor.mjSENS_ACCELEROMETER),
    ):
        scene.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=IMU_SITE)

    scene.body(CHASSIS).add_camera(
        name=TRACK_CAMERA,
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
        pos=[0.0, -0.9, 0.25],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 0.164, 0.986],
    )
    return scene.compile()


@dataclass(frozen=True)
class RobotBinding:
    """Ids and geometry resolved by name from a compiled model."""

    chassis_body: int
    root_qposadr: int
    root_dofadr: int
    wheel_qposadr: tuple[int, int]
    wheel_dofadr: tuple[int, int]
    motor_ids: tuple[int, int]
    imu_site: int
    push_site: int
    gyro_adr: int
    accel_adr: int
    wheel_geoms: frozenset[int]
    wheel_radius_m: float
    track_width_m: float
    wheel_sign: tuple[float, float]


def _axis_in_chassis(model: mujoco.MjModel, joint_id: int) -> npt.NDArray[np.float64]:
    body = model.jnt_bodyid[joint_id]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_quat[body])
    axis: npt.NDArray[np.float64] = rot.reshape(3, 3) @ np.asarray(model.jnt_axis[joint_id])
    return axis


def _wheel_geoms(model: mujoco.MjModel, body: int) -> list[int]:
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]


def _collision_geoms(model: mujoco.MjModel, body: int) -> list[int]:
    return [
        g
        for g in _wheel_geoms(model, body)
        if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
    ]


def _tire_geom(model: mujoco.MjModel, joint_name: str) -> int:
    """The single collision geom of a wheel body: the tire that touches the floor."""
    body = int(model.jnt_bodyid[model.joint(joint_name).id])
    tires = _collision_geoms(model, body)
    if len(tires) != 1:
        raise ValueError(
            f"wheel body of '{joint_name}' must have exactly one collision geom, got {len(tires)}"
        )
    return tires[0]


def bind_robot(model: mujoco.MjModel) -> RobotBinding:
    joints = [model.joint(name) for name in WHEEL_JOINTS]
    bodies = [int(model.jnt_bodyid[j.id]) for j in joints]
    signs = tuple(1.0 if _axis_in_chassis(model, j.id)[1] > 0 else -1.0 for j in joints)
    tires = [_tire_geom(model, name) for name in WHEEL_JOINTS]
    radius = float(model.geom_size[tires[0]][0])  # validate_robot checks both tires agree
    track = abs(float(model.body_pos[bodies[0]][1] - model.body_pos[bodies[1]][1]))
    root = model.joint(ROOT_JOINT)
    return RobotBinding(
        chassis_body=model.body(CHASSIS).id,
        root_qposadr=int(root.qposadr[0]),
        root_dofadr=int(root.dofadr[0]),
        wheel_qposadr=(int(joints[0].qposadr[0]), int(joints[1].qposadr[0])),
        wheel_dofadr=(int(joints[0].dofadr[0]), int(joints[1].dofadr[0])),
        motor_ids=(model.actuator(MOTORS[0]).id, model.actuator(MOTORS[1]).id),
        imu_site=model.site(IMU_SITE).id,
        push_site=model.site(PUSH_SITE).id,
        gyro_adr=int(model.sensor(GYRO_SENSOR).adr[0]),
        accel_adr=int(model.sensor(ACCEL_SENSOR).adr[0]),
        wheel_geoms=frozenset(tires),
        wheel_radius_m=radius,
        track_width_m=track,
        wheel_sign=(signs[0], signs[1]),
    )


def _lookup(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    return int(mujoco.mj_name2id(model, kind, name))


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"#{index}"


def _root_problems(model: mujoco.MjModel, chassis: int) -> list[str]:
    """The robot must be free to fall: a free root joint, no welds, no gravity compensation."""
    problems: list[str] = []
    root = _lookup(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT)
    if (
        root < 0
        or model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE
        or model.jnt_bodyid[root] != chassis
    ):
        problems.append(f"free joint '{ROOT_JOINT}' on body '{CHASSIS}' is missing")
    if model.neq > 0:
        problems.append(f"equality constraints are not allowed, found {model.neq}")
    for body in range(1, model.nbody):
        if model.body_gravcomp[body] != 0:
            name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            problems.append(f"body '{name}' uses gravcomp; the robot must feel full gravity")
    return problems


def _tire_problems(model: mujoco.MjModel, name: str, body: int) -> list[str]:
    tires = _collision_geoms(model, body)
    if len(tires) != 1:
        return [f"wheel body of '{name}' must have exactly one collision geom, got {len(tires)}"]
    if int(model.geom_type[tires[0]]) not in TIRE_TYPES:
        return [f"tire of '{name}' must be a cylinder, sphere or capsule"]
    return []


TIRE_RADIUS_TOLERANCE_M = 1e-9


def _tire_radius_problems(model: mujoco.MjModel) -> list[str]:
    """Both tires must have one radius: ``wheel_radius_m`` (and odometry) assume a single one."""
    radii: list[float] = []
    for name in WHEEL_JOINTS:
        jid = _lookup(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        tires = _collision_geoms(model, int(model.jnt_bodyid[jid])) if jid >= 0 else []
        if len(tires) != 1:
            return []  # already reported by _wheel_problems
        radii.append(float(model.geom_size[tires[0]][0]))
    if abs(radii[0] - radii[1]) > TIRE_RADIUS_TOLERANCE_M:
        left, right = WHEEL_JOINTS
        return [
            f"tires of '{left}' and '{right}' must have the same radius, "
            f"got {radii[0]:g} and {radii[1]:g} m"
        ]
    return []


def _wheel_problems(model: mujoco.MjModel, chassis: int, name: str, side: float) -> list[str]:
    """One wheel joint: an unlimited ±y hinge on a direct child of the chassis, on its side."""
    jid = _lookup(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        return [f"joint '{name}' is missing"]
    problems: list[str] = []
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
        problems.append(f"joint '{name}' must be a hinge")
    if model.jnt_limited[jid]:
        problems.append(f"joint '{name}' must be unlimited (wheels turn freely)")
    body = int(model.jnt_bodyid[jid])
    if model.body_parentid[body] != chassis:
        problems.append(f"joint '{name}' must be on a direct child body of '{CHASSIS}'")
    if float(model.body_pos[body][1]) * side <= 0:
        where = "y > 0 (left)" if side > 0 else "y < 0 (right)"
        problems.append(f"joint '{name}' body must sit at {where} in the chassis frame")
    axis = _axis_in_chassis(model, jid)
    if abs(abs(float(axis[1])) - 1.0) > 1e-6:
        got = axis.round(3).tolist()
        problems.append(f"joint '{name}' axis must be +-y in the chassis frame, got {got}")
    return problems + _tire_problems(model, name, body)


def _interface_problems(model: mujoco.MjModel, chassis: int) -> list[str]:
    """Sites on the chassis, the injected motors and IMU sensors."""
    problems: list[str] = []
    for name in (IMU_SITE, PUSH_SITE):
        sid = _lookup(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            problems.append(f"site '{name}' is missing")
        elif model.site_bodyid[sid] != chassis:
            problems.append(f"site '{name}' must be on body '{CHASSIS}'")
    for name in MOTORS:
        if _lookup(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0:
            problems.append(f"actuator '{name}' is missing")
    if model.nu != len(MOTORS):
        problems.append(f"expected {len(MOTORS)} actuators, got {model.nu}")
    for name in (GYRO_SENSOR, ACCEL_SENSOR):
        if _lookup(model, mujoco.mjtObj.mjOBJ_SENSOR, name) < 0:
            problems.append(f"sensor '{name}' is missing")
    return problems


def _world_problems(model: mujoco.MjModel, chassis: int, scene_geoms: frozenset[str]) -> list[str]:
    """Nothing in the world but the robot and the scene's own geoms (the floor)."""
    problems: list[str] = []
    for body in range(1, model.nbody):
        if model.body_rootid[body] != chassis:
            name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            problems.append(f"world body '{name}' is not part of the robot")
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if model.geom_bodyid[geom] == 0 and name not in scene_geoms:
            problems.append(f"world geom '{name or f'#{geom}'}' is not allowed; the scene owns it")
    return problems


def validate_robot(
    model: mujoco.MjModel, *, scene_geoms: frozenset[str] = SCENE_GEOMS
) -> list[str]:
    """Every way ``model`` violates ``balancer.robot.v1`` (empty list = valid, contract §3)."""
    chassis = _lookup(model, mujoco.mjtObj.mjOBJ_BODY, CHASSIS)
    if chassis < 0:
        return [f"body '{CHASSIS}' is missing"]
    problems = _root_problems(model, chassis)
    for name, side in zip(WHEEL_JOINTS, (1.0, -1.0), strict=True):
        problems += _wheel_problems(model, chassis, name, side)
    problems += _tire_radius_problems(model)
    problems += _interface_problems(model, chassis)
    problems += _world_problems(model, chassis, scene_geoms)
    return problems


def quat_tilt(quat: npt.ArrayLike) -> float:
    """Angle between the chassis z axis and world z, from the root quaternion (wxyz).

    Covers pitch and roll falls. Computed from ``qpos`` rather than MuJoCo's ``xmat``, which lags
    one substep behind after ``mj_step`` (contract §11). Tolerates a non-unit quaternion.
    """
    w, x, y, z = (float(v) for v in np.asarray(quat, dtype=np.float64))
    norm2 = w * w + x * x + y * y + z * z
    cos_tilt = 1.0 - 2.0 * (x * x + y * y) / norm2
    return math.acos(max(-1.0, min(1.0, cos_tilt)))
