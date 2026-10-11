# Topple contract v1 (normative)

This document defines what a Topple controller sees, what it returns, how an episode runs and how it
is scored. Python (`topple/`), the browser arena, the GitHub Action and robot firmware all implement
exactly this. Anything that could change a fixed controller's score on a fixed seed is versioned
(see §12). Values marked **[calibrated]** are frozen at the end of Phase 1 in
`challenges/balance-v1/challenge.yaml` and never edited afterwards.

## 1. Identifiers

| Kind | Grammar | Example |
|---|---|---|
| Challenge | `[a-z][a-z0-9]*(-[a-z][a-z0-9]*)*-v[1-9][0-9]*` | `balance-v1`, `waypoint-v1`, `sumo-v1` |
| Protocol | `<challenge>/<preset>` | `balance-v1/standard`, `balance-v1/clean` |
| Gymnasium id | `topple/<Pascal>-v<N>` (1:1 with challenge) | `topple/Balance-v1` |
| Schema | `<family>.<obs\|action\|robot>.v<N>` | `balancer.obs.v1`, `diffdrive.effort.v1`, `balancer.robot.v1` |
| Artifact format | `topple.<kind>/<N>` | `topple.traj/1`, `topple.results/1` |

Permanent aliases: `eval_v1` → `balance-v1/standard`, `eval_v1_clean` → `balance-v1/clean`,
`obs_v1` → `balancer.obs.v1`.

Every grammar is matched against the whole string (a trailing newline is not a match). Each
segment of a challenge name starts with a letter, so the Gymnasium id (segments capitalized and
joined) maps back to exactly one challenge id: `foo-1bar` would collide with `foo1bar`, so it is
not a valid name. Presets match `[a-z][a-z0-9_]*`.

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
| body + free joint | `chassis`, free joint `root` | root of the robot, a child of the world |
| hinge joints | `left_wheel`, `right_wheel` | unlimited, axes ±y in the chassis frame |
| wheel bodies | the bodies of the two hinges | direct children of `chassis`; left at y > 0, right at y < 0 |
| tire geoms | the one collision geom of each wheel body | cylinder, sphere or capsule; size[0] is the wheel radius |
| site | `imu` | child of `chassis`; where the IMU board is mounted |
| site | `push_point` | child of `chassis`; where pushes are applied |

A collision geom is one with `contype` or `conaffinity` nonzero. Wheel bodies may also carry
visual geoms (`contype = conaffinity = 0`). Only the tires may touch the floor (§11).

The env **builds** the simulated robot from the MJCF plus a `RobotParams` file (`params.yaml`) with
`mujoco.MjSpec`: it removes any actuators/sensors in the file and injects its own:

- Motors `left_motor`, `right_motor`: `general` actuator, `gainprm=[tau_stall]`,
  `biastype=affine`, `biasprm=[0, 0, -tau_stall/omega_noload]`, `ctrlrange=[-1,1]`,
  `forcerange=[-torque_limit, torque_limit]`, history buffer `nsample=8` (native delay, up to
  10 ms).
  This is an ideal DC motor without inductance: torque = tau_stall·u − (tau_stall/omega_noload)·ω.
- Wheel joints: `frictionloss = frictionloss_nm` (gearbox Coulomb friction), `armature`,
  `damping = 0` (back-EMF lives in the actuator).
- Sensors `imu_gyro`, `imu_accel` on site `imu`.

`validate_robot(model) -> list[str]` reports every violation of the composed model; robot-ci, CAD
export (Phase 8), evolved bodies (Phase 10) and generated robots (Phase 12) all go through it.
Besides the table above it checks that the robot can fall freely and that nothing else shares the
world:

- `root` is a free joint on `chassis`; no equality constraints; no body uses `gravcomp`;
- both motors and both IMU sensors exist and there are exactly 2 actuators;
- every body other than the world belongs to the robot (its root body is `chassis`), and the world
  body has no geoms except the scene's own (`floor`).

`wheel_radius_m` is the tire geom's radius and `track_width_m` the distance between the wheel bodies
along y; both reach the controller through `TaskSpec.robot`.

`robot_hash(robot_xml, params)` = sha256 of the MJCF bytes, then the params as sorted-key JSON,
then, for every file the MJCF references (`<include>` recursively, meshes, skins and height fields
via `meshdir`, textures via `texturedir`, other `file` attributes relative to the MJCF; `assetdir`
sets both directories), a record `len(path)‖path‖len(bytes)‖bytes` (8-byte big-endian lengths, POSIX
path relative to the MJCF's directory) in path order. A referenced file that does not exist is an
error. A robot without asset files (balance-v1) hashes the MJCF and params only.

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
1. Mounting: the IMU board sits rotated by `imu_pitch_offset` about y. The true readings of
   `imu_gyro` (rad/s) and `imu_accel` (m/s², specific force) are both rotated into the board
   frame: `(x, y, z) → (c·x − s·z, y, s·x + c·z)` with `c, s = cos, sin(imu_pitch_offset)`.
2. `gyro = clamp(gyro_board + gyro_bias + N(0, gyro_noise), ±gyro_range)`;
   `accel = clamp(accel_board + N(0, accel_noise), ±accel_range)`.
3. Complementary filter: `a = tau/(tau + dt)`;
   `pitch_est ← a·(pitch_est + gyro_y·dt) + (1 − a)·atan2(−accel_x, accel_z)`.
   At reset the filter starts at the pitch a standstill accelerometer implies, as firmware does
   when it calibrates at rest: the true initial pitch **plus** `imu_pitch_offset` (the mount
   offset is a real, uncorrected error that the controller has to live with).

The IMU reads MuJoCo's sensor data after each `mj_step`, and MuJoCo computes that data during the
step from the state at the *start* of the substep: the readings lag the simulated state by one
substep (2 ms), like a real sample-and-hold. The reading at reset (k = 0, after `mj_forward`) is
current; it uses one extra SENSOR noise block (§9).

Per control step (after the 5 substeps):
4. Encoders: `counts_i = round(wheel_sign_i · q_i · cpr / 2π)` (integer, chassis-relative);
   `wheel_pos_i = (counts_i − counts_i(reset)) · 2π / cpr`.
5. `wheel_vel_i = (wheel_pos_i[k] − wheel_pos_i[k−1]) / control_dt` (0 at k = 0).
6. `yaw = wheel_radius · (wheel_pos_r − wheel_pos_l) / track_width` (odometry: no gyro drift,
   immune to pitch), wrapped with `atan2(sin, cos)`.
7. `pitch_rate = gyro_y`, `yaw_rate = gyro_z` (last substep's board-frame reading).

Preset `ideal` replaces steps 1–7 with the exact current state (unit tests only): pitch and yaw
from the root quaternion in `qpos` (yaw relative to reset), `pitch_rate`/`yaw_rate` = the root's
body-frame angular velocity from `qvel`, wheel angles and rates from the joint state. The recorded
`gyro`/`accel` rows (§14) are then MuJoCo's noiseless sensor data, one substep old as above.

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
- **Quantization.** The env emits the features as float32 (`obs`, also what trajectories store).
  A harness builds the controller's observation only with `contract.controller_obs(obs_arr, t)`,
  which rounds the 10 features to float32 and widens them exactly to Python floats (float64);
  `t = k · control_dt` is not quantized. The wire carries exactly these values, so an in-process
  controller and one behind a pipe receive bit-identical observations, and `obs` row k of a
  trajectory reproduces observation k bit for bit.
- A published schema id never changes meaning or order; changes get a new id. Later tasks extend by
  appending fields under a new id (e.g. `balancer-target.obs.v1` appends `target_x, target_y`).

## 7. Action `diffdrive.effort.v1`

- `act()` returns an ordered, re-iterable container of 2 finite real numbers: a list, tuple,
  1-D numpy array, `range`, `array.array` or `memoryview`. Each value is the normalized motor
  voltage `u = V_cmd / V_nom` for that wheel.
- `invalid_action` (the episode fails, score 0) when the returned object is: not length 2; not a
  container (`None`, a number); a string, bytes, mapping or set; a one-shot iterator or generator
  (rejected without being advanced, so its code never runs); not 1-dimensional; or holds a value
  that is a bool, not real (complex, str), does not fit a float (e.g. `10**400`) or is NaN/±inf.
  The harness reads at most 3 items, so an endless or huge container fails fast. If the object's
  own code raises an `Exception` while it is inspected, iterated or converted, that is also
  `invalid_action`; a `BaseException` that is not an `Exception` (e.g. `SystemExit`) is
  `exception` (§8).
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

- `TaskSpec` (frozen, hashable, picklable) gives the challenge id, obs schema id + field names,
  action spec, `control_dt`, `episode_seconds`, robot geometry (`wheel_radius_m`,
  `track_width_m`) and `extras`, a read-only copy of `str → float` task parameters.
  A zero-argument `make_controller()` is also accepted (lesson 1).
- `make_controller` is called once per episode; instances must not share mutable module state.
- `rng_seed` is a controller-private seed derived from the episode seed (§9); the episode seed
  itself never crosses the controller boundary. With a private protocol's secret salt it carries
  no usable information about the episode.
- **Public seeds are identifiable.** Public protocols use salt 0, so `rng_seed` (and equally the
  initial observation) identifies which public episode is running: a controller can recognize a
  public seed and replay a memorized answer. Scores on public seeds are therefore provisional.
  Only salted private-seed runs produced by trusted CI are rankable (Phase 3).
- **Misbehaviour never crashes the harness.** Anything the controller raises in `make_controller`,
  `reset` or `act` (any `BaseException`, including `SystemExit` and `GeneratorExit`) ends the
  episode with error `exception` at that step; an unusable return value is `invalid_action` (§7).
  Errored episodes score 0. Only `KeyboardInterrupt` (the operator stopping the run) propagates.
  In-process runs are for trusted code; isolation from hostile code is the runner's job (Phase 3).
- Submission kinds: a `.py` file, or a directory with `controller.py` plus assets located via
  `Path(__file__).parent` (ONNX policies arrive in Phase 2). A directory may hold at most 50
  regular files and 20 MiB in total (enforced while reading); symlinks, special files,
  `__pycache__` and `.pyc` files are refused, and loading never writes bytecode next to the
  submission. The submission sha256 covers every file as length-framed
  `(POSIX relpath, size, bytes)` records in path order.

## 9. Randomness, seeds and scenarios

- Seeds are integers in `[0, 2**53)`.
- Streams: `np.random.default_rng(np.random.SeedSequence(episode_seed, spawn_key=(agent, stream)))`
  with append-only stream ids `INIT=0, PUSHES=1, SENSOR=2, MODEL=3, DELAY=4, CONTROLLER=5`.
  `agent` is 0 for single-robot tasks.
- Controller seed: `SeedSequence(entropy=(episode_seed, salt), spawn_key=(0, CONTROLLER))
  .generate_state(1, uint64)[0] >> 11`; `salt = 0` for public protocols, a 128-bit secret for
  private ones.
- The SENSOR stream draws a fixed-size block of `frame_skip × 6` standard normals (gyro xyz, accel
  xyz, every substep) every control step, plus one extra block at reset whose last row drives the
  reset reading (§5). Blocks are drawn even when a noise std is 0 or the preset is `ideal`, so
  presets never change stream consumption.
- `make_scenario(episode_seed, preset) -> Scenario` materializes everything that is not per-step
  noise as JSON-serializable data: initial pitch/pitch_rate, push list, model overrides, actuator
  delay, IMU biases and mounting offset. Trajectories embed it. Model overrides are
  **multiplicative scales** relative to the robot params (`chassis_mass_scale`,
  `wheel_friction_scale`, `motor_gain` per wheel, `frictionloss_scale`) plus `com_x_offset_m`
  (added to the chassis COM x) and `actuator_delay_s`, not absolute values.
- `Scenario.from_dict` validates what it reads, and `ValueError` rejects anything else: exact key
  sets; finite numbers; integer `start_step`/`n_substeps`; pushes sorted by `start_step` with no
  two at the same step, `0 ≤ start_step < max_steps` (when the episode length is known),
  `n_substeps ≥ 1`, `0 ≤ magnitude_n ≤ 50` N, `forward_sign = ±1`, `|lateral_ratio| ≤ 1`; scales in
  [0.5, 2] (frictionloss [0, 3]), `|com_x_offset_m| ≤ 0.02`, `0 ≤ actuator_delay_s ≤ 0.010`,
  `|imu_pitch_offset_rad| ≤ 0.1`, `|gyro_bias| ≤ 0.1` rad/s per axis. Every preset draws well
  inside these bounds; `reset(options={"scenario": dict})` goes through this check.
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
  `floor_contact` when any non-tire geom touches the floor. Checked after every substep.
- The tilt is computed from the current root quaternion in `qpos`, so a fall ends the episode on
  the substep that crosses the angle. Floor contacts come from MuJoCo's contact list, which
  `mj_step` builds from the state at the start of the substep: contact detection lags by one
  substep (2 ms).
- `sim_unstable`: MuJoCo warnings BADQACC/BADQVEL/BADQPOS or non-finite state → episode error,
  score 0. The failed substep is rolled back so the record stays finite. Auto-reset is disabled.
- `truncated` when `k == max_steps` and not terminated.
- **Epilogue.** After a fall (`fell`, `floor_contact`) or a controller error, 1.0 s (100 control
  steps) is simulated with `ctrl = 0` and no controller calls, for replays only; it never affects
  metrics. There is no epilogue after truncation (a surviving robot's replay ends upright) or after
  `sim_unstable`; `epilogue_qpos` then has 0 rows. `evaluate.wants_epilogue(termination_reason,
  has_error)` is the single rule, used by evaluation and verification alike.

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

Protocol score = `fsum` of episode scores in seed order / N, and success rate = successes / N,
where N is the number of episodes the protocol expects: a missing or errored episode counts as 0.
Mean time upright is averaged over the episodes present. Ranking: round(score, 2), ties broken by
success rate, then mean time upright.

## 13. Bump rule and hashes

Anything that can change a fixed controller's score on a fixed seed requires a new challenge
major version (`balance-v2`): MJCF, robot params, physics options, presets, scorer, E_ref, seeds,
episode length, control rate, observation semantics. Reward shaping and training configs never bump.

Env code is shared by all challenges. A code change that can change a fixed controller's results
on a fixed seed (sensing, estimation, termination, pushes, model overrides) must bump
`topple.challenge.ENV_ALGO_VERSION` (currently 1). It is part of every `protocol_hash`, so results
from different env algorithms are never compared; once a challenge is published such a change also
needs its new major version.

- `bundle_hash()` = sha256 of the tag `topple.bundle/1\n`, then, for each bundle file in POSIX-path
  order, `len(path)‖path‖len(bytes)‖bytes` (8-byte big-endian lengths, path relative to the bundle
  root, CRLF → LF in `.xml`/`.yaml` files). The bundle files are exactly: `challenge.yaml`, the
  scene, the robot params, every `*.xml` in the robot MJCF's directory, and every file the scene
  or robot MJCF references (§3). Other files in the directory do not count; a listed file outside
  the bundle directory is an error.
- `protocol_hash(preset, seeds, salt)` = sha256 of the canonical JSON (sorted keys, no
  whitespace) of `{format: "topple.protocol/1", env_algo_version, bundle: bundle_hash,
  preset: <the full preset definition: name, sensor model, randomize flag, randomization ranges,
  push config>, score: <ScoreConfig>, seeds: <seed list in order>, salt}`. Seeds default to the
  public list, salt to 0. A private protocol's hash includes its 128-bit secret salt, which keeps
  the published hash from revealing the private seeds.

## 14. Trajectory `topple.traj/1`

One `.npz` per episode, written with only `<f8`, `<f4`, `<i4`, `|u1` C-contiguous arrays (no
pickles, no strings). Row `k` of state arrays is the state at `t = k·control_dt`, before action `k`.

| Array | Shape | dtype | Meaning |
|---|---|---|---|
| `qpos` | T+1 × nq | f8 | state; row T is the final/terminal state |
| `qvel` | T+1 × nv | f8 | |
| `obs` | T × 10 | f4 | exactly what the controller received |
| `action` | T × 2 | f8 | raw controller output, before clipping |
| `ctrl` | T × 2 | f8 | clipped command issued at step k (before native delay) |
| `enc_counts` | T × 2 | i4 | encoder counts after the step |
| `gyro`, `accel` | T × 3 | f4 | last-substep IMU readings (after bias/noise) |
| `push_force` | T × 3 | f8 | world-frame push force active during step k (0 if none) |
| `epilogue_qpos` | E × nq | f8 | 1 s after termination, replay only |
| `meta_json` | n | u1 | UTF-8 JSON: format, challenge, preset, episode_index, scenario, versions, robot_hash, wheel_sign, joint addresses, termination {step, reason}, error (if any), control_dt, timestep, frame_skip, max_steps, obs features |

If the controller errors at step `k`, the episode stops there (T = k) and the error is stored in
`meta.error`. The env is a pure function of (protocol, scenario, ctrl sequence): `verify_episode` re-simulates
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
