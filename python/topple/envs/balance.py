"""``topple/Balance-v1``: keep a two-wheeled balancer upright (docs/contract.md §4-11)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, model_validator

from topple import __version__
from topple.challenge import Challenge, load_challenge
from topple.contract import FEATURES, BalancerObs, clip_action, validate_action
from topple.disturbances import (
    PRESETS,
    Preset,
    Push,
    Scenario,
    check_push_schedule,
    make_scenario,
    push_force_world,
)
from topple.estimation import (
    cf_step,
    encoder_counts,
    imu_reading,
    odometry_yaw,
    wheel_angle_from_counts,
    wrap_angle,
)
from topple.rendering import make_renderer
from topple.rng import MAX_SEED, Stream, stream_rng
from topple.robot import (
    TRACK_CAMERA,
    RobotParams,
    bind_robot,
    build_model,
    quat_tilt,
    robot_hash,
    validate_robot,
)
from topple.traj import Trajectory

_RESET_OPTIONS = frozenset({"scenario", "init_pitch_rad", "init_pitch_rate", "pushes"})
_UNSTABLE_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQPOS,
)


class RewardConfig(BaseModel):
    """Bounded training reward ``r = 1 - sum(w_k * c_k)`` with every cost ``c_k`` in [0, 1]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tilt: NonNegativeFloat = 0.30
    rate: NonNegativeFloat = 0.10
    drift: NonNegativeFloat = 0.25
    yaw_rate: NonNegativeFloat = 0.05
    effort: NonNegativeFloat = 0.10
    smooth: NonNegativeFloat = 0.20

    @model_validator(mode="after")
    def _bounded(self) -> RewardConfig:
        if (
            self.tilt + self.rate + self.drift + self.yaw_rate + self.effort + self.smooth
            > 1.0 + 1e-9
        ):
            raise ValueError("reward weights must sum to at most 1 so the reward stays in [0, 1]")
        return self


@dataclass(frozen=True)
class _Nominal:
    body_mass: npt.NDArray[np.float64]
    body_inertia: npt.NDArray[np.float64]
    body_ipos: npt.NDArray[np.float64]
    geom_friction: npt.NDArray[np.float64]
    actuator_gainprm: npt.NDArray[np.float64]
    dof_frictionloss: npt.NDArray[np.float64]
    actuator_delay: npt.NDArray[np.float64]


def _quat_pitch_yaw(quat: npt.NDArray[np.float64]) -> tuple[float, float]:
    """ZYX pitch and yaw of the chassis from its root quaternion (wxyz).

    Uses ``qpos`` rather than MuJoCo's ``xmat``: after ``mj_step`` the derived data still describe
    the state at the start of the substep (contract §11).
    """
    q = np.asarray(quat, dtype=np.float64)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, q / np.linalg.norm(q))
    r = rot.reshape(3, 3)
    pitch = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))
    yaw = math.atan2(r[1, 0], r[0, 0])
    return pitch, yaw


@dataclass(frozen=True)
class _TrueState:
    """Exact chassis orientation and body-frame rates at the current ``qpos``/``qvel``."""

    pitch: float
    yaw: float
    pitch_rate: float
    yaw_rate: float
    tilt: float


class BalanceEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """Gymnasium environment for ``balance-v1``. The episode seed fully determines the scenario."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}  # noqa: RUF012

    def __init__(
        self,
        preset: str = "standard",
        *,
        challenge: str | Challenge = "balance-v1",
        robot_xml: str | Path | None = None,
        params: RobotParams | None = None,
        render_mode: str | None = None,
        render_size: tuple[int, int] = (320, 240),
        reward: RewardConfig | None = None,
        max_drift_m: float | None = None,
        record: bool = False,
    ) -> None:
        # A Challenge object is used as is (never re-loaded by id), so a protocol's hashing and
        # scoring stay on the one bundle it was built from.
        self.challenge: Challenge = (
            challenge if isinstance(challenge, Challenge) else load_challenge(challenge)
        )
        if preset not in self.challenge.config.presets:
            raise ValueError(f"preset {preset!r} is not part of {self.challenge.id}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.params = params or self.challenge.params
        self.robot_xml = Path(robot_xml) if robot_xml is not None else self.challenge.robot_xml
        self.model = build_model(self.robot_xml, self.challenge.scene_xml, self.params)
        problems = validate_robot(self.model)
        if problems:
            raise ValueError("robot does not satisfy balancer.robot.v1: " + "; ".join(problems))
        self.data = mujoco.MjData(self.model)
        self.binding = bind_robot(self.model)
        self.task_spec = self.challenge.task_spec(
            self.binding.wheel_radius_m, self.binding.track_width_m
        )
        self.control_dt = self.task_spec.control_dt
        timestep = float(self.model.opt.timestep)
        self.frame_skip = round(self.control_dt / timestep)
        if abs(self.frame_skip * timestep - self.control_dt) > 1e-12:
            raise ValueError("control_dt must be an integer multiple of the physics timestep")
        self.max_steps = self.task_spec.max_steps
        self._preset: Preset = PRESETS[preset]
        self._pending_preset: Preset | None = None
        self.reward_config = reward or RewardConfig()
        self.max_drift_m = max_drift_m
        self.record = record
        self.render_mode = render_mode
        self._render_size = render_size
        self._renderer: mujoco.Renderer | None = None
        self._floor_geom = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
        m = self.model
        self._nominal = _Nominal(
            body_mass=m.body_mass.copy(),
            body_inertia=m.body_inertia.copy(),
            body_ipos=m.body_ipos.copy(),
            geom_friction=m.geom_friction.copy(),
            actuator_gainprm=m.actuator_gainprm.copy(),
            dof_frictionloss=m.dof_frictionloss.copy(),
            actuator_delay=m.actuator_delay.copy(),
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(len(FEATURES),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._scenario: Scenario | None = None
        self._done = True
        # Per-episode state, (re)initialized by reset().
        self._sensors_ideal = False
        self._noise_rng = stream_rng(0, Stream.SENSOR)
        self._step_count = 0
        self._termination: str | None = None
        self._energy_j = 0.0
        self._saturated_steps = 0
        self._prev_u: tuple[float, ...] = (0.0, 0.0)
        self._push_force = (0.0, 0.0, 0.0)
        self._push_remaining = 0
        self._pushes_by_step: dict[int, Push] = {}
        self._xy0: npt.NDArray[np.float64] = np.zeros(2)
        self._yaw0 = 0.0
        self._wheel_q0: tuple[float, ...] = (0.0, 0.0)
        self._counts0: tuple[int, ...] = (0, 0)
        self._counts: tuple[int, ...] = (0, 0)
        self._prev_wheel_pos: tuple[float, float] = (0.0, 0.0)
        self._prev_wheel_q: tuple[float, ...] = (0.0, 0.0)
        self._pitch_est = 0.0
        self._gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._epilogue: list[npt.NDArray[np.float64]] = []
        self._rec: dict[str, list[Any]] = {}
        self._last_obs_for_record: npt.NDArray[np.float32] = np.zeros(len(FEATURES), np.float32)
        self._stable_qpos = np.zeros(self.model.nq)  # state before the current substep
        self._stable_qvel = np.zeros(self.model.nv)

    # ------------------------------------------------------------------ public helpers

    @property
    def preset(self) -> Preset:
        return self._preset

    @property
    def scenario(self) -> Scenario:
        if self._scenario is None:
            raise RuntimeError("call reset() first")
        return self._scenario

    @property
    def termination_reason(self) -> str | None:
        """Why the current episode terminated (``fell``, ``floor_contact``, ...), else None."""
        return self._termination

    def set_preset(self, preset: str) -> None:
        """Curriculum hook: switch disturbance preset at the next reset (works via env_method)."""
        if preset not in self.challenge.config.presets:
            raise ValueError(f"preset {preset!r} is not part of {self.challenge.id}")
        self._pending_preset = PRESETS[preset]

    # ------------------------------------------------------------------ gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if self._pending_preset is not None:
            self._preset, self._pending_preset = self._pending_preset, None
        opts = dict(options or {})
        unknown = set(opts) - _RESET_OPTIONS
        if unknown:
            raise ValueError(f"unknown reset options {sorted(unknown)}")

        if "scenario" in opts:
            given = opts["scenario"]
            scenario = (
                given
                if isinstance(given, Scenario)
                else Scenario.from_dict(given, max_steps=self.max_steps)
            )
        else:
            episode_seed = seed if seed is not None else int(self.np_random.integers(MAX_SEED))
            scenario = make_scenario(
                episode_seed,
                self._preset,
                timestep=float(self.model.opt.timestep),
                control_dt=self.control_dt,
                episode_seconds=self.task_spec.episode_seconds,
            )
        if "init_pitch_rad" in opts:
            scenario = replace(scenario, init_pitch_rad=float(opts["init_pitch_rad"]))
        if "pushes" in opts:
            pushes = tuple(opts["pushes"])
            if not all(isinstance(p, Push) for p in pushes):
                raise ValueError("pushes must be a sequence of topple.disturbances.Push")
            scenario = replace(scenario, pushes=pushes)
        if "init_pitch_rate" in opts:
            scenario = replace(scenario, init_pitch_rate_rads=float(opts["init_pitch_rate"]))
        if not (
            math.isfinite(scenario.init_pitch_rad) and math.isfinite(scenario.init_pitch_rate_rads)
        ):
            raise ValueError("initial pitch and pitch rate must be finite")
        check_push_schedule(scenario.pushes)

        self._scenario = scenario
        self._sensors_ideal = PRESETS[scenario.preset].sensors == "ideal"
        self._apply_model_overrides(scenario)
        self._set_initial_state(scenario.init_pitch_rad, scenario.init_pitch_rate_rads)
        self._init_episode_state(scenario)
        obs = self._observe(last_action=(0.0, 0.0))
        if self.record:
            self._start_recording(obs)
        info = self._info(reward_terms={}, push_force=(0.0, 0.0, 0.0))
        return obs.to_array(), info

    def step(
        self, action: Any
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("episode is over: call reset()")
        raw = validate_action(action, dim=2)
        u, saturated = clip_action(raw)
        b, d, m = self.binding, self.data, self.model
        for i, motor in enumerate(b.motor_ids):
            d.ctrl[motor] = b.wheel_sign[i] * u[i]

        noise = self._draw_noise_block()
        self._start_push_if_due()
        push_force = self._push_force if self._push_remaining > 0 else (0.0, 0.0, 0.0)
        zero3 = np.zeros(3)
        for j in range(self.frame_skip):
            d.qfrc_applied[:] = 0.0
            if self._push_remaining > 0:
                mujoco.mj_applyFT(
                    m,
                    d,
                    np.asarray(self._push_force),
                    zero3,
                    d.site_xpos[b.push_site],
                    b.chassis_body,
                    d.qfrc_applied,
                )
                self._push_remaining -= 1
            np.copyto(self._stable_qpos, d.qpos)
            np.copyto(self._stable_qvel, d.qvel)
            mujoco.mj_step(m, d)
            if self._unstable():
                self._roll_back()
                self._end("sim_unstable")
                break
            if self._sensors_ideal:
                self._gyro, self._accel = self._true_imu()
            else:
                self._gyro, self._accel = self._imu(noise[j])
                self._pitch_est = cf_step(
                    self._pitch_est,
                    self._gyro[1],
                    self._accel[0],
                    self._accel[2],
                    self.params.estimation.cf_tau_s,
                    float(m.opt.timestep),
                )
            reason = self._fall_reason()
            if reason is not None:
                self._end(reason)
                break
        d.qfrc_applied[:] = 0.0
        self._step_count += 1

        self._accumulate_energy(u)
        obs = self._observe(last_action=u)
        reward, terms = self._reward(u)
        self._prev_u = u
        self._saturated_steps += int(saturated)
        if (
            self.max_drift_m is not None
            and self._termination is None
            and self._drift() > self.max_drift_m
        ):
            self._end("drift")
        terminated = self._termination is not None
        truncated = not terminated and self._step_count >= self.max_steps
        self._done = terminated or truncated
        if self.record:
            self._record_step(obs, raw, u, push_force)
        info = self._info(reward_terms=terms, push_force=push_force)
        return obs.to_array(), reward, terminated, truncated, info

    def render(self) -> npt.NDArray[np.uint8] | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            width, height = self._render_size
            self._renderer = make_renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=TRACK_CAMERA)
        frame: npt.NDArray[np.uint8] = self._renderer.render()
        return frame

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ recording

    def simulate_epilogue(self, seconds: float = 1.0) -> None:
        """After the episode ends, keep simulating with zero command (replay only, unscored)."""
        b, d, m = self.binding, self.data, self.model
        d.ctrl[list(b.motor_ids)] = 0.0
        rows = []
        for _ in range(round(seconds / self.control_dt)):
            for _ in range(self.frame_skip):
                mujoco.mj_step(m, d)
                if not np.isfinite(d.qpos).all():
                    break
            if not np.isfinite(d.qpos).all():
                break
            rows.append(d.qpos.copy())
        self._epilogue = rows

    def trajectory(self, extra_meta: dict[str, Any] | None = None) -> Trajectory:
        if not self.record or self._scenario is None:
            raise RuntimeError("construct the env with record=True and run an episode first")
        nq = self.model.nq
        rec = self._rec
        meta = {
            "challenge": self.challenge.id,
            "preset": self._scenario.preset,
            "scenario": self._scenario.to_dict(),
            "control_dt": self.control_dt,
            "timestep": float(self.model.opt.timestep),
            "frame_skip": self.frame_skip,
            "max_steps": self.max_steps,
            "obs_schema": self.task_spec.obs_schema,
            "obs_features": list(FEATURES),
            "joint_qpos": {
                "root": self.binding.root_qposadr,
                "wheels": list(self.binding.wheel_qposadr),
            },
            "wheel_sign": list(self.binding.wheel_sign),
            "robot_hash": robot_hash(self.robot_xml, self.params),
            "termination": {"step": self._step_count, "reason": self._termination or "timeout"},
            "versions": {
                "topple": __version__,
                "mujoco": mujoco.__version__,
                "numpy": np.__version__,
            },
            **(extra_meta or {}),
        }
        return Trajectory(
            qpos=np.array(rec["qpos"]),
            qvel=np.array(rec["qvel"]),
            obs=np.array(rec["obs"], dtype=np.float32).reshape(-1, len(FEATURES)),
            action=np.array(rec["action"], dtype=np.float64).reshape(-1, 2),
            ctrl=np.array(rec["ctrl"], dtype=np.float64).reshape(-1, 2),
            enc_counts=np.array(rec["enc"], dtype=np.int32).reshape(-1, 2),
            gyro=np.array(rec["gyro"], dtype=np.float32).reshape(-1, 3),
            accel=np.array(rec["accel"], dtype=np.float32).reshape(-1, 3),
            push_force=np.array(rec["push"], dtype=np.float64).reshape(-1, 3),
            epilogue_qpos=np.array(self._epilogue, dtype=np.float64).reshape(-1, nq),
            meta=meta,
        )

    # ------------------------------------------------------------------ internals

    def _apply_model_overrides(self, scenario: Scenario) -> None:
        m, b, nom, mo = self.model, self.binding, self._nominal, scenario.model
        m.body_mass[:] = nom.body_mass
        m.body_inertia[:] = nom.body_inertia
        m.body_ipos[:] = nom.body_ipos
        m.geom_friction[:] = nom.geom_friction
        m.actuator_gainprm[:] = nom.actuator_gainprm
        m.dof_frictionloss[:] = nom.dof_frictionloss
        m.actuator_delay[:] = nom.actuator_delay
        c = b.chassis_body
        m.body_mass[c] = nom.body_mass[c] * mo.chassis_mass_scale
        m.body_inertia[c] = nom.body_inertia[c] * mo.chassis_mass_scale
        m.body_ipos[c, 0] = nom.body_ipos[c, 0] + mo.com_x_offset_m
        for g in b.wheel_geoms:
            m.geom_friction[g, 0] = nom.geom_friction[g, 0] * mo.wheel_friction_scale
        for i, motor in enumerate(b.motor_ids):
            m.actuator_gainprm[motor, 0] = nom.actuator_gainprm[motor, 0] * mo.motor_gain[i]
            m.actuator_delay[motor] = mo.actuator_delay_s
        for dof in b.wheel_dofadr:
            m.dof_frictionloss[dof] = nom.dof_frictionloss[dof] * mo.frictionloss_scale
        mujoco.mj_setConst(m, self.data)

    def _set_initial_state(self, pitch: float, pitch_rate: float) -> None:
        m, d, b = self.model, self.data, self.binding
        mujoco.mj_resetData(m, d)
        qa, da = b.root_qposadr, b.root_dofadr
        d.qpos[qa + 3 : qa + 7] = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
        d.qvel[da + 4] = pitch_rate
        for i, dof in enumerate(b.wheel_dofadr):
            d.qvel[dof] = -pitch_rate * b.wheel_sign[i]  # wheels do not slip at t = 0
        mujoco.mj_forward(m, d)

    def _init_episode_state(self, scenario: Scenario) -> None:
        b, d = self.binding, self.data
        cpr = self.params.sensing.encoder_cpr
        self._noise_rng = stream_rng(scenario.seed, Stream.SENSOR)
        noise = self._draw_noise_block()
        self._step_count = 0
        self._termination = None
        self._done = False
        self._energy_j = 0.0
        self._saturated_steps = 0
        self._prev_u = (0.0, 0.0)
        self._push_force = (0.0, 0.0, 0.0)
        self._push_remaining = 0
        self._pushes_by_step = {p.start_step: p for p in scenario.pushes}
        self._xy0 = d.qpos[b.root_qposadr : b.root_qposadr + 2].copy()
        self._yaw0 = self._true_state().yaw
        self._wheel_q0 = tuple(float(d.qpos[a]) for a in b.wheel_qposadr)
        self._counts0 = tuple(
            encoder_counts(d.qpos[a], b.wheel_sign[i], cpr) for i, a in enumerate(b.wheel_qposadr)
        )
        self._counts = self._counts0
        self._prev_wheel_pos = (0.0, 0.0)
        self._prev_wheel_q = (0.0, 0.0)  # reset-relative, like _wheel_angles()
        # The filter starts where a standstill accelerometer reading puts it: the true pitch plus
        # the mount offset (contract §5). The reset reading below uses the extra noise block.
        self._pitch_est = scenario.init_pitch_rad + scenario.sensors.imu_pitch_offset_rad
        if self._sensors_ideal:
            self._gyro, self._accel = self._true_imu()
        else:
            self._gyro, self._accel = self._imu(noise[-1])
        self._epilogue = []

    def _draw_noise_block(self) -> npt.NDArray[np.float64]:
        """Fixed consumption: one (frame_skip, 6) standard-normal block per control step."""
        s = self.params.sensing
        block = self._noise_rng.standard_normal((self.frame_skip, 6))
        scaled: npt.NDArray[np.float64] = block * np.array(
            [s.gyro_noise_rads] * 3 + [s.accel_noise_ms2] * 3
        )
        return scaled

    def _imu(
        self, noise_row: npt.NDArray[np.float64]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        d, b, s = self.data, self.binding, self.params.sensing
        sc = self.scenario.sensors
        g = d.sensordata[b.gyro_adr : b.gyro_adr + 3]
        a = d.sensordata[b.accel_adr : b.accel_adr + 3]
        return imu_reading(
            gyro_true=(float(g[0]), float(g[1]), float(g[2])),
            accel_true=(float(a[0]), float(a[1]), float(a[2])),
            mount_pitch_rad=sc.imu_pitch_offset_rad,
            gyro_bias=sc.gyro_bias_rads,
            gyro_noise=(float(noise_row[0]), float(noise_row[1]), float(noise_row[2])),
            accel_noise=(float(noise_row[3]), float(noise_row[4]), float(noise_row[5])),
            gyro_range=s.gyro_range_rads,
            accel_range=s.accel_range_ms2,
        )

    def _true_imu(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Noiseless gyro and accelerometer readings as MuJoCo computed them (preset ``ideal``)."""
        d, b = self.data, self.binding
        g = d.sensordata[b.gyro_adr : b.gyro_adr + 3]
        a = d.sensordata[b.accel_adr : b.accel_adr + 3]
        return (float(g[0]), float(g[1]), float(g[2])), (float(a[0]), float(a[1]), float(a[2]))

    def _true_state(self) -> _TrueState:
        d, b = self.data, self.binding
        quat = d.qpos[b.root_qposadr + 3 : b.root_qposadr + 7]
        pitch, yaw = _quat_pitch_yaw(quat)
        return _TrueState(
            pitch=pitch,
            yaw=yaw,
            pitch_rate=float(d.qvel[b.root_dofadr + 4]),  # free-joint angular velocity: body frame
            yaw_rate=float(d.qvel[b.root_dofadr + 5]),
            tilt=quat_tilt(quat),
        )

    def _start_push_if_due(self) -> None:
        push = self._pushes_by_step.get(self._step_count)
        if push is None:
            return
        self._push_force = push_force_world(push, self._true_state().yaw)
        self._push_remaining = push.n_substeps

    def _unstable(self) -> bool:
        d = self.data
        if any(d.warning[w].number > 0 for w in _UNSTABLE_WARNINGS):
            return True
        return not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())

    def _roll_back(self) -> None:
        """Undo the substep that blew up, so observations, rewards and records stay finite."""
        d = self.data
        d.qpos[:] = self._stable_qpos
        d.qvel[:] = self._stable_qvel
        d.qacc_warmstart[:] = 0.0  # the failed solve can leave NaN here
        d.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, d)

    def _fall_reason(self) -> str | None:
        """Tilt from the current qpos; contacts from MuJoCo's last evaluation (one substep old)."""
        b, d = self.binding, self.data
        quat = d.qpos[b.root_qposadr + 3 : b.root_qposadr + 7]
        if quat_tilt(quat) > self.params.safety.fall_angle_rad:
            return "fell"
        for i in range(d.ncon):
            c = d.contact[i]
            if self._floor_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self._floor_geom else c.geom1
                if other not in b.wheel_geoms:
                    return "floor_contact"
        return None

    def _end(self, reason: str) -> None:
        if self._termination is None:
            self._termination = reason

    def _wheel_angles(self) -> tuple[float, float]:
        d, b = self.data, self.binding
        return (
            b.wheel_sign[0] * (float(d.qpos[b.wheel_qposadr[0]]) - self._wheel_q0[0]),
            b.wheel_sign[1] * (float(d.qpos[b.wheel_qposadr[1]]) - self._wheel_q0[1]),
        )

    def _accumulate_energy(self, u: tuple[float, ...]) -> None:
        """Battery energy of an ideal DC motor, from command and mean wheel speed (contract §12)."""
        motor = self.params.motor
        q = self._wheel_angles()
        for i in range(2):
            omega = (q[i] - self._prev_wheel_q[i]) / self.control_dt
            self._energy_j += (
                max(
                    0.0,
                    motor.v_nom
                    * motor.stall_current_a
                    * u[i]
                    * (u[i] - omega / motor.omega_noload_rads),
                )
                * self.control_dt
            )
        self._prev_wheel_q = q

    def _observe(self, last_action: tuple[float, ...]) -> BalancerObs:
        d, b = self.data, self.binding
        t = self._step_count * self.control_dt
        la = (float(last_action[0]), float(last_action[1]))
        cpr = self.params.sensing.encoder_cpr
        counts = tuple(
            encoder_counts(d.qpos[a], b.wheel_sign[i], cpr) for i, a in enumerate(b.wheel_qposadr)
        )
        self._counts = counts
        if self._sensors_ideal:
            true = self._true_state()
            pos = self._wheel_angles()
            vel = (
                b.wheel_sign[0] * float(d.qvel[b.wheel_dofadr[0]]),
                b.wheel_sign[1] * float(d.qvel[b.wheel_dofadr[1]]),
            )
            yaw = wrap_angle(true.yaw - self._yaw0)
            return BalancerObs(t, true.pitch, true.pitch_rate, yaw, true.yaw_rate, pos, vel, la)

        pos2 = (
            wheel_angle_from_counts(counts[0], self._counts0[0], cpr),
            wheel_angle_from_counts(counts[1], self._counts0[1], cpr),
        )
        if self._step_count == 0:
            vel2 = (0.0, 0.0)
        else:
            vel2 = (
                (pos2[0] - self._prev_wheel_pos[0]) / self.control_dt,
                (pos2[1] - self._prev_wheel_pos[1]) / self.control_dt,
            )
        self._prev_wheel_pos = pos2
        yaw = odometry_yaw(pos2[0], pos2[1], b.wheel_radius_m, b.track_width_m)
        return BalancerObs(t, self._pitch_est, self._gyro[1], yaw, self._gyro[2], pos2, vel2, la)

    def _drift(self) -> float:
        b, d = self.binding, self.data
        xy = d.qpos[b.root_qposadr : b.root_qposadr + 2]
        return float(math.hypot(xy[0] - self._xy0[0], xy[1] - self._xy0[1]))

    def _reward(self, u: tuple[float, ...]) -> tuple[float, dict[str, float]]:
        w, true = self.reward_config, self._true_state()
        costs = {
            "tilt": min((true.pitch / 0.2) ** 2, 1.0),
            "rate": min((true.pitch_rate / 3.0) ** 2, 1.0),
            "drift": min(self._drift() ** 2, 1.0),
            "yaw_rate": min((true.yaw_rate / 2.0) ** 2, 1.0),
            "effort": (u[0] ** 2 + u[1] ** 2) / 2.0,
            "smooth": min(
                ((u[0] - self._prev_u[0]) ** 2 + (u[1] - self._prev_u[1]) ** 2) / 2.0 / 0.25, 1.0
            ),
        }
        penalty = sum(getattr(w, k) * c for k, c in costs.items())
        return 1.0 - penalty, costs

    def _info(
        self, reward_terms: dict[str, float], push_force: tuple[float, float, float]
    ) -> dict[str, Any]:
        true = self._true_state()
        info: dict[str, Any] = {
            "episode_seed": self.scenario.seed,
            "t": self._step_count * self.control_dt,
            "step": self._step_count,
            "true_pitch": true.pitch,
            "tilt": true.tilt,
            "drift_m": self._drift(),
            "energy_j": self._energy_j,
            "push_force": push_force,
            "reward_terms": dict(reward_terms),
            "termination_reason": self._termination,
        }
        return info

    def _start_recording(self, first_obs: BalancerObs) -> None:
        d = self.data
        self._rec = {
            "qpos": [d.qpos.copy()],
            "qvel": [d.qvel.copy()],
            "obs": [],
            "action": [],
            "ctrl": [],
            "enc": [],
            "gyro": [],
            "accel": [],
            "push": [],
        }
        self._last_obs_for_record = first_obs.to_array()

    def _record_step(
        self,
        obs: BalancerObs,
        raw: tuple[float, ...],
        u: tuple[float, ...],
        push_force: tuple[float, float, float],
    ) -> None:
        d, rec = self.data, self._rec
        rec["qpos"].append(d.qpos.copy())
        rec["qvel"].append(d.qvel.copy())
        rec["obs"].append(self._last_obs_for_record)
        rec["action"].append(raw)
        rec["ctrl"].append(u)
        rec["enc"].append(self._counts)
        rec["gyro"].append(self._gyro)
        rec["accel"].append(self._accel)
        rec["push"].append(push_force)
        self._last_obs_for_record = obs.to_array()
