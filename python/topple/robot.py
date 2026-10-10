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


def robot_hash(robot_xml: Path, params: RobotParams) -> str:
    digest = hashlib.sha256()
    digest.update(robot_xml.read_bytes())
    digest.update(json.dumps(params.model_dump(), sort_keys=True).encode())
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


def bind_robot(model: mujoco.MjModel) -> RobotBinding:
    joints = [model.joint(name) for name in WHEEL_JOINTS]
    bodies = [int(model.jnt_bodyid[j.id]) for j in joints]
    signs = tuple(1.0 if _axis_in_chassis(model, j.id)[1] > 0 else -1.0 for j in joints)
    geoms = [g for body in bodies for g in _wheel_geoms(model, body)]
    radius = max(float(model.geom_size[g][0]) for g in geoms)
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
        wheel_geoms=frozenset(geoms),
        wheel_radius_m=radius,
        track_width_m=track,
        wheel_sign=(signs[0], signs[1]),
    )


def _lookup(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    return int(mujoco.mj_name2id(model, kind, name))


def validate_robot(model: mujoco.MjModel) -> list[str]:
    """Every way ``model`` violates ``balancer.robot.v1`` (empty list = valid)."""
    problems: list[str] = []
    chassis = _lookup(model, mujoco.mjtObj.mjOBJ_BODY, CHASSIS)
    if chassis < 0:
        return [f"body '{CHASSIS}' is missing"]
    root = _lookup(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT)
    if (
        root < 0
        or model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE
        or model.jnt_bodyid[root] != chassis
    ):
        problems.append(f"free joint '{ROOT_JOINT}' on body '{CHASSIS}' is missing")
    for name in WHEEL_JOINTS:
        jid = _lookup(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            problems.append(f"joint '{name}' is missing")
            continue
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            problems.append(f"joint '{name}' must be a hinge")
        body = model.jnt_bodyid[jid]
        if model.body_parentid[body] != chassis:
            problems.append(f"joint '{name}' must be on a direct child body of '{CHASSIS}'")
        axis = _axis_in_chassis(model, jid)
        if abs(abs(float(axis[1])) - 1.0) > 1e-6:
            got = axis.round(3).tolist()
            problems.append(f"joint '{name}' axis must be +-y in the chassis frame, got {got}")
        if not _wheel_geoms(model, body):
            problems.append(f"joint '{name}' body has no wheel geom")
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


def tilt_angle(xmat: np.ndarray) -> float:
    """Angle between the chassis z axis and world z (covers pitch and roll falls)."""
    return math.acos(max(-1.0, min(1.0, float(xmat[8]))))
