# Topple contract v1 (normative)

This document defines what a Topple controller sees, what it returns, how an episode runs and how it
is scored. Python (`topple/`), the browser arena, the GitHub Action and robot firmware all implement
exactly this. Anything that could change a fixed controller's score on a fixed seed is versioned
(see §12). Values marked **[calibrated]** are frozen at the end of Phase 1 in
`challenges/balance-v1/challenge.yaml` and never edited afterwards.

## 1. Identifiers

| Kind | Grammar | Example |
|---|---|---|
| Challenge | `^[a-z][a-z0-9]*(-[a-z0-9]+)*-v[1-9][0-9]*$` | `balance-v1`, `waypoint-v1`, `sumo-v1` |
| Protocol | `<challenge>/<preset>` | `balance-v1/standard`, `balance-v1/clean` |
| Gymnasium id | `topple/<Pascal>-v<N>` (1:1 with challenge) | `topple/Balance-v1` |
| Schema | `<family>.<obs\|action\|robot>.v<N>` | `balancer.obs.v1`, `diffdrive.effort.v1`, `balancer.robot.v1` |
| Artifact format | `topple.<kind>/<N>` | `topple.traj/1`, `topple.results/1` |

Permanent aliases: `eval_v1` → `balance-v1/standard`, `eval_v1_clean` → `balance-v1/clean`,
`obs_v1` → `balancer.obs.v1`.

## 2. Frames, signs, units

SI units throughout (m, s, rad, N, N·m, V, A, J).

- **Body frame** (chassis): x forward, y left (along the axle, toward the left wheel), z up.
  Origin at the axle midpoint.
- **pitch**: rotation about body +y. 0 = chassis z axis vertical. Positive = top leans toward +x.
- **yaw**: rotation about world +z, positive counter-clockwise seen from above (turning left).
- **Wheels**: joint angle is chassis-relative (what an encoder measures). Positive = the wheel rolls
  the robot forward (+x). Both hinge axes are body +y; a model whose axis is mirrored declares
  `wheel_sign = -1` and the env applies it to observations and actions.
- **Action order** is always `(left, right)`.

## 3. Robot interface `balancer.robot.v1`

A robot MJCF (body tree only; no `<option>`, floor or lights) must contain, by exact name:

| Element | Name | Notes |
|---|---|---|
| body + free joint | `chassis`, free joint `root` | root of the robot |
| hinge joints | `left_wheel`, `right_wheel` | axes ±y in the chassis frame |
| site | `imu` | child of `chassis`; where the IMU board is mounted |
| site | `push_point` | child of `chassis`; where pushes are applied |
| geoms | wheel geoms in `wheel_left`/`wheel_right` bodies or any geom on the wheel bodies | only wheel geoms may touch the floor |

The env **builds** the simulated robot from the MJCF plus a `RobotParams` file (`params.yaml`) with
`mujoco.MjSpec`: it removes any actuators/sensors in the file and injects its own:

- Motors `left_motor`, `right_motor`: `general` actuator, `gainprm=[tau_stall]`,
  `biastype=affine`, `biasprm=[0, 0, -tau_stall/omega_noload]`, `ctrlrange=[-1,1]`,
  `forcerange=[-torque_limit, torque_limit]`, history buffer `nsample=6` (native delay).
  This is an ideal DC motor without inductance: torque = tau_stall·u − (tau_stall/omega_noload)·ω.
- Wheel joints: `frictionloss = frictionloss_nm` (gearbox Coulomb friction), `armature`,
  `damping = 0` (back-EMF lives in the actuator).
- Sensors `imu_gyro`, `imu_accel` on site `imu`.

`validate_robot(model) -> list[str]` reports every violation; robot-ci, CAD export (Phase 8),
evolved bodies (Phase 10) and generated robots (Phase 12) all go through it.

The scene (`scene.xml`) owns physics options: `timestep=0.002`, `integrator=implicitfast`,
`<flag autoreset="disable"/>`, a floor plane and lights. The robot is attached with prefix `""`.

## 4. Timing

- Physics step: 0.002 s (500 Hz). Control period: `control_dt = 0.01` s (100 Hz), so 5 substeps.
- `act()` is called at exactly 100 Hz on every platform, firmware included (hardware timer,
  zero-order hold). Sensing and estimation run at the physics rate (500 Hz).
- Time is an integer step counter `k`; `obs.t = k * control_dt`. Never derived from `data.time`.
- Episode length: `episode_seconds = 10.0` → `max_steps = 1000`.

## 5. Sensing and estimation (how the Observation is produced)

The observation is produced from simulated sensors by the same estimator firmware runs, so sim and
real logs are comparable. Implemented once in `topple/estimation.py` with plain float math
(portable to C, JS and MicroPython).

Per physics substep:
1. `gyro = imu_gyro + gyro_bias + N(0, gyro_noise)` (rad/s, body frame), clamped to ±`gyro_range`.
2. `accel = imu_accel + N(0, accel_noise)` (m/s², specific force, body frame), clamped to
   ±`accel_range`. A mounting offset rotates the reading by `imu_pitch_offset` about y.
3. Complementary filter: `a = tau/(tau + dt)`;
   `pitch_est ← a·(pitch_est + gyro_y·dt) + (1 − a)·atan2(−accel_x, accel_z)`.
   Initialized at reset to the true pitch (firmware calibrates at standstill).

Per control step (after the 5 substeps):
4. Encoders: `counts_i = round(wheel_sign_i · q_i · cpr / 2π)` (integer, chassis-relative);
   `wheel_pos_i = (counts_i − counts_i(reset)) · 2π / cpr`.
5. `wheel_vel_i = (wheel_pos_i[k] − wheel_pos_i[k−1]) / control_dt` (0 at k = 0).
6. `yaw = wheel_radius · (wheel_pos_r − wheel_pos_l) / track_width` (odometry: no gyro drift,
   immune to pitch), wrapped with `atan2(sin, cos)`.
7. `pitch_rate = gyro_y`, `yaw_rate = gyro_z` (last substep's filtered reading).

Preset `ideal` replaces steps 1–7 with ground truth (unit tests only).

## 6. Observation `balancer.obs.v1`

```python
@dataclass(frozen=True, slots=True)
class BalancerObs:            # alias: topple.contract.Observation
    t: float                  # s since reset (metadata, not a feature)
    pitch: float              # rad
    pitch_rate: float         # rad/s
    yaw: float                # rad, (-pi, pi]
    yaw_rate: float           # rad/s
    wheel_pos: tuple[float, float]    # rad since reset (left, right)
    wheel_vel: tuple[float, float]    # rad/s (left, right)
    last_action: tuple[float, float]  # command applied during the previous step (zeros at reset)
```

- Feature vector (`to_array`, float32, fixed order, 10 values):
  `pitch, pitch_rate, yaw, yaw_rate, wheel_pos_l, wheel_pos_r, wheel_vel_l, wheel_vel_r,
  last_action_l, last_action_r`. `t` is excluded.
- Wire format (`to_wire`/`from_wire`): 11 little-endian float64 in the order
  `t` + the feature order. Bitwise lossless.
- Every field is coerced to Python `float` on construction.
- A published schema id never changes meaning or order; changes get a new id. Later tasks extend by
  appending fields under a new id (e.g. `balancer-target.obs.v1` appends `target_x, target_y`).

## 7. Action `diffdrive.effort.v1`

- `act()` returns any length-2 sequence (list, tuple, ndarray) of finite real numbers:
  normalized motor voltage `u = V_cmd / V_nom` per wheel.
- Not length 2, not real, or non-finite → `invalid_action` error (episode fails).
- Values are clipped to [-1, 1] (not an error; reported as `saturation_fraction`).
- Firmware: `duty = clip(u · V_nom / V_batt, −1, 1)`, slow-decay PWM ≥ 20 kHz, no deadband
  compensation (or identical compensation in sim and firmware).

## 8. Controller

```python
class Controller(Protocol[ObsT]):
    def reset(self, rng_seed: int) -> None: ...
    def act(self, obs: ObsT) -> Sequence[float]: ...

def make_controller(spec: TaskSpec) -> Controller: ...   # in the submission file
```

- `TaskSpec` (frozen) gives the challenge id, obs schema id + field names, action spec,
  `control_dt`, `episode_seconds` and robot geometry (`wheel_radius_m`, `track_width_m`).
  A zero-argument `make_controller()` is also accepted (lesson 1).
- `make_controller` is called once per episode; instances must not share mutable module state.
- `rng_seed` is a controller-private seed derived from the episode seed (§9). It carries no
  information about the episode; the episode seed never crosses the controller boundary.
- Submission kinds: a `.py` file, or a directory with `controller.py` plus assets located via
  `Path(__file__).parent` (ONNX policies arrive in Phase 2).

## 9. Randomness, seeds and scenarios

- Seeds are integers in `[0, 2**53)`.
- Streams: `np.random.default_rng(np.random.SeedSequence(episode_seed, spawn_key=(agent, stream)))`
  with append-only stream ids `INIT=0, PUSHES=1, SENSOR=2, MODEL=3, DELAY=4, CONTROLLER=5`.
  `agent` is 0 for single-robot tasks.
- Controller seed: `SeedSequence(entropy=(episode_seed, salt), spawn_key=(0, CONTROLLER))
  .generate_state(1, uint64)[0] >> 11`; `salt = 0` for public protocols, a 128-bit secret for
  private ones.
- The SENSOR stream draws a fixed-size block every control step (all channels, all substeps) even
  when a noise std is 0, so presets never change stream consumption.
- `make_scenario(episode_seed, preset) -> Scenario` materializes everything that is not per-step
  noise as JSON-serializable data: initial pitch/pitch_rate, push list, realized model parameters
  (absolute values), actuator delay, IMU biases and mounting offset. Trajectories embed it.
- `reset(seed=None)` (training): `episode_seed = int(self.np_random.integers(2**53))`, returned in
  `info["episode_seed"]` so any training failure can be replayed.

## 10. Disturbance presets

| Preset | Sensors/estimator | Model randomization | Pushes |
|---|---|---|---|
| `ideal` | ground truth | none | none |
| `clean` | full (§5) | full | none |
| `standard` | full | full | 1–3 per episode |

Randomization (`clean`, `standard`), per episode:
chassis mass ×U(0.9, 1.1) (inertia scaled with it), COM x offset U(−3, 3) mm, wheel–floor friction
×U(0.9, 1.1), motor gain common ×U(0.9, 1.1) × per-wheel ×U(0.95, 1.05) (gain only; back-EMF
fixed), frictionloss ×U(0.5, 1.5), actuator delay ∈ {0, 2, 4, 6, 8, 10} ms, IMU mounting pitch
offset U(−1°, 1°), gyro bias U(−0.01, 0.01) rad/s per axis. Initial pitch U(−3°, 3°), pitch rate 0.

Pushes (`standard`): count ∈ {1, 2, 3}; first at t ≥ 1.0 s, spacing ≥ 1.5 s, all before
t = 8.5 s; rectangular pulse of 0.1 s applied at site `push_point` (`mj_applyFT`); direction fixed
in the world frame at onset: ± along the heading, lateral component ≤ 0.3 of the magnitude;
impulse magnitude **[calibrated]** (target: reference LQR survives ≥ 80% and PID 40–90% of
`standard` episodes; initial band U(0.25, 0.45) N·s). Force is capped at 50 N.

## 11. Termination, stability

- `terminated` with reason `fell` when the angle between body z and world z exceeds
  `fall_angle_rad` (0.70 rad ≈ 40°, a robot parameter also used by firmware cutoff), or reason
  `floor_contact` when any non-wheel geom touches the floor. Checked after every substep.
- `sim_unstable`: MuJoCo warnings BADQACC/BADQVEL/BADQPOS or non-finite state → episode error,
  score 0. Auto-reset is disabled.
- `truncated` when `k == max_steps` and not terminated.
- After termination, 1.0 s of epilogue (ctrl = 0, no controller calls) is simulated for replays
  only; it never affects metrics.

## 12. Metrics and score (`balance-v1`)

Computed from the trajectory by a pure function (re-runnable on stored data):

- `upright_frac = steps_alive / max_steps`
- `rms_tilt` over alive steps (true tilt angle, rad)
- `drift_m` = final xy distance of the axle midpoint from its start; also `max_drift_m`
- `energy_j` = Σ over substeps and wheels of `max(0, V_nom·i_stall·u·(u − ω/ω_noload))·dt`
  (battery energy of an ideal DC motor in slow decay; computable from real logs)
- `saturation_fraction` = fraction of steps where any |u| ≥ 0.99 after clipping

Episode score (0–100):
`100 · upright_frac · (0.6 + 0.2·(1 − min(rms_tilt/0.2, 1)) + 0.1·(1 − min(drift_m/1, 1))
+ 0.1·(1 − min(energy_j/E_ref, 1)))`. An episode with an error scores 0.

`success = survived ∧ drift_m < 1 ∧ energy_j ≤ 5·E_ref`. `E_ref` **[calibrated]** = 2× the PID
baseline's median energy on `balance-v1/standard`, rounded, frozen.

Protocol score = `fsum` of episode scores in seed order / N. Ranking: round(score, 2), ties broken
by success rate, then mean time upright.

## 13. Bump rule

Anything that can change a fixed controller's score on a fixed seed requires a new challenge
major version (`balance-v2`): MJCF, robot params, physics options, presets, scorer, E_ref, seeds,
episode length, control rate, observation semantics. Reward shaping and training configs never bump.

## 14. Trajectory `topple.traj/1`

One `.npz` per episode, written with only `<f8`, `<f4`, `<i4`, `|u1` C-contiguous arrays (no
pickles, no strings). Row `k` of state arrays is the state at `t = k·control_dt`, before action `k`.

| Array | Shape | dtype | Meaning |
|---|---|---|---|
| `qpos` | T+1 × nq | f8 | state; row T is the final/terminal state |
| `qvel` | T+1 × nv | f8 | |
| `obs` | T × 10 | f4 | exactly what the controller received |
| `action` | T × 2 | f8 | raw controller output (NaN row on error) |
| `ctrl` | T × 2 | f8 | clipped command issued at step k (before native delay) |
| `enc_counts` | T × 2 | i4 | encoder counts after the step |
| `gyro`, `accel` | T × 3 | f4 | last-substep IMU readings (after bias/noise) |
| `push_force` | T × 3 | f8 | world-frame push force active during step k (0 if none) |
| `epilogue_qpos` | E × nq | f8 | 1 s after termination, replay only |
| `meta_json` | n | u1 | UTF-8 JSON: format, challenge, protocol, episode_index, scenario, versions, model sha256, termination {step, reason}, control_dt, timestep, field names |

The env is a pure function of (protocol, scenario, ctrl sequence): `verify_episode` re-simulates
open loop from `scenario` + `ctrl` and must reproduce `qpos` exactly on the same platform.
Loading uses `allow_pickle=False`, an allow-list of keys, dtype checks, `T ≤ max_steps`, and a
per-member size cap.

## 15. Results `topple.results/1`

`results.json` (pydantic, `extra="forbid"`, no NaN/Inf): format, challenge, protocol,
protocol_hash, seed_set (`public`/`private`), versions (topple, mujoco, numpy, obs schema, action
spec), platform, robot_hash, controller {id, sha256}, isolation (`inprocess`/`subprocess`/
`container`; only `container` is rankable), complete, episodes [{index, success, score, metrics,
error {code, step, detail}}], aggregate {score, success_rate, metrics}. Error codes: `exception`,
`invalid_action`, `timeout`, `crash`, `protocol`, `output_limit`, `load_error`, `sim_unstable`.
Private-seed results are redacted: no per-episode data, aggregate only.
