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
| Schema | `<family>.<obs\|robot\|effort>.v<N>` | `balancer.obs.v1`, `diffdrive.effort.v1`, `balancer.robot.v1` |
| Artifact format | `topple.<kind>/<N>` | `topple.traj/1`, `topple.results/1` |

Permanent aliases: `eval_v1` → `balance-v1/standard`, `eval_v1_clean` → `balance-v1/clean`,
`obs_v1` → `balancer.obs.v1`.

Every grammar is matched against the whole string (a trailing newline is not a match). Each
segment of a challenge name starts with a letter, so the Gymnasium id (segments capitalized and
joined) maps back to exactly one challenge id: `foo-1bar` would collide with `foo1bar`, so it is
not a valid name. Presets match `[a-z][a-z0-9_]*`.

A schema's family is built like a challenge name without its version (`[a-z][a-z0-9]*` segments
joined by `-`, e.g. `balancer-target`). Its kind says what it describes: `obs` an observation
(§6), `robot` a robot interface (§3), and an action schema is named by its control mode, `effort`
(normalized motor voltages, §7); a new kind or control mode is a contract change.
`topple.contract.validate_schema_id` checks this grammar, and a challenge's `obs_schema` and
`action_spec` must match it.

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
| tire geoms | the one collision geom of each wheel body | cylinder, sphere or capsule, centred on the hinge axis; size[0] is the wheel radius, equal on both tires; a cylinder's or capsule's axis (its local z) lies along the hinge axis |
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

- `root` is a free joint on `chassis` with zero `damping`, `armature` and `frictionloss` on all
  six of its DOFs and zero `stiffness`, polynomial terms included (`damping`/`stiffness` given as
  several coefficients); no equality constraints; no body uses `gravcomp`;
- both motors and both IMU sensors exist and there are exactly 2 actuators;
- every body other than the world belongs to the robot (its root body is `chassis`), the world
  body has no geoms except the scene's own (`floor`) and no sites, no tendon passes through
  a site or wraps a geom that is not on the robot, and no deformable flex has a vertex or node
  on a body that is not on the robot (a tendon or flex tied to the world is a tether). Tendons
  and flexes that join parts of the robot are allowed.
- each tire's centre lies on its wheel's hinge axis line (through the joint's `pos`, along its
  `axis`, both in the wheel body's frame) within 1e-6 m, and each cylinder or capsule tire's axis
  is parallel (or antiparallel) to that hinge axis, within 1e-6 in |cos| of the angle between
  them. Together they make `size[0]` the radius the tire rolls on: an off-centre tire would roll
  as a cam.

`wheel_radius_m` is the tire geom's radius (both tires must agree to within 1e-9 m) and
`track_width_m` the distance between the wheel bodies along y; both reach the controller through
`TaskSpec.robot`.

`robot_hash(robot_xml, params)` = sha256 of the MJCF bytes, then the params as JSON text exactly as
Python's `json.dumps(params.model_dump(), sort_keys=True)` writes it (UTF-8; keys sorted at every
level; separators `", "` and `": "`; floats in Python's shortest round-trip form, so `30.0` is
written `30.0`, never `30`; balance-v1's text is pinned in `tests/test_robot.py`), then, for every
file the MJCF references (`<include>` recursively, meshes, skins and height fields
via `meshdir`, textures via `texturedir`, other `file` attributes relative to the MJCF; `assetdir`
sets both directories; a sub-model `<asset><model file>` together with every file *it*
references, recursively, resolved as MuJoCo does against the sub-model's own directory and
`<compiler>`), a record `len(path)‖path‖len(bytes)‖bytes` (8-byte big-endian lengths, POSIX path
relative to the MJCF's directory) in POSIX-path string order (`a-c.obj` before `a/b.obj`). A
referenced file that does not exist is an error. A robot without asset files (balance-v1) hashes
the MJCF and params only. Text files (`.xml`, `.yaml`, `.yml`: the MJCF and any it includes) are
hashed with CRLF → LF, exactly as in `bundle_hash` (§13), so a checkout's line endings (Git's
`core.autocrlf` on Windows) never change a robot's hash; other files count byte for byte.

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

The robot spawns with its tires exactly touching the floor (distance 0), and MuJoCo makes a
contact a constraint only once it penetrates. So the reset state and the first substep are
computed as free fall: the reset reading and the first substep's accelerometer sample measure a
specific force of about 0 (noise only) where a robot at rest reads +g. The first filter update
(step 3) then pulls the estimate toward `atan2` of noise, an error of at most
`(1 − a)·(π + |pitch|)` ≈ 12.8 mrad that decays with `tau`. Measured on darwin-arm64: std 7.1 mrad
after control step 1 over 300 `clean` seeds (0.55 mrad with the tires 0.1 mm into the floor),
and no effect on the calibration (§10, §12: on `standard` seeds 0..199, LQR 86.5% and PID 64.5%
survival and E_ref 13.0 either way). Whether a contact at exactly zero distance counts rests on
rounding, so this first sample can differ by platform, as physics may (§14). It is part of
balance-v1; changing the spawn changes every score, so it needs a new major version (§13).

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
  (rejected without being advanced, so its code never runs); not 1-dimensional (an object that
  declares `ndim` must declare exactly the `int` 1; any other type of `ndim` is rejected by its
  class name, never compared or printed); or holds a value
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
  no usable information about the episode. `EvalProtocol` enforces the salts: a public
  protocol's salt is 0 and a private protocol's is an `int` in `[2**64, 2**128)`; anything else
  (a forgotten salt of 0, a hand-typed `99`, a bool) raises `ValueError`.
- **Public seeds are identifiable.** Public protocols use salt 0, so `rng_seed` (and equally the
  initial observation) identifies which public episode is running: a controller can recognize a
  public seed and replay a memorized answer. Scores on public seeds are therefore provisional.
  Only salted private-seed runs produced by trusted CI are rankable (Phase 3).
- **Misbehaviour never crashes the harness.** Anything the controller raises in `make_controller`,
  `reset` or `act` (any `BaseException`, including `SystemExit` and `GeneratorExit`) ends the
  episode with error `exception` at that step; an unusable return value is `invalid_action` (§7).
  Errored episodes score 0. Only `KeyboardInterrupt` (the operator stopping the run) propagates.
  This holds even for exceptions whose own machinery is hostile (a failing `__str__`, a metaclass
  `__name__` that raises, a `str` subclass as message or name, a raising `__class__`): the error
  detail is `Type: message` built from the real class name and exact `str` copies, and the
  harness never asks the controller object or its exceptions what class they are. That
  includes the loader: a loaded `make_controller`'s result must have `reset` and `act`, found by
  the same static lookup a runtime protocol check uses (instance dict, then class) but without
  reading `__class__`; otherwise the episode ends with `exception` at step 0. A returned action
  *is* inspected with `isinstance` (§7), so an action whose `__class__` raises is
  `invalid_action`. Every message, `invalid_action` details included, names a class only by its
  real name, so no metaclass or `str`-subclass code runs while it is built.
  In-process runs are for trusted code; isolation from hostile code is the runner's job (Phase 3).
- **Loading never crashes either.** Whatever a submission raises while it is imported, or while
  its `make_controller` is looked up or its signature read (any `BaseException` except
  `KeyboardInterrupt`, including `SystemExit`), `load_submission` raises `LoadError` with the
  same `Type: message` detail, and the CLI exits 2 with that message, never a traceback. A load
  that fails leaves no submission module behind in `sys.modules`.
- Submission kinds: a `.py` file, or a directory with `controller.py` plus assets located via
  `Path(__file__).parent` (ONNX policies arrive in Phase 2). A directory may hold at most 50
  regular files in at most 50 subdirectories (at any depth) and 20 MiB in total, enforced while
  it is walked and read, so a huge tree is refused at once; symlinks, special files,
  `__pycache__`, `.pyc` files and names (the submission's own or any file's) that are not valid
  UTF-8 are refused, and so is a file or directory that cannot be read (a `LoadError` naming it,
  never an `OSError`). Untrusted names in load errors are quoted with `repr`, so they cannot
  carry terminal escapes or line breaks. Loading a submission turns bytecode writing off for the
  rest of the process (`sys.dont_write_bytecode`), so neither its import nor a later import of a
  sibling module (in `make_controller`, `reset` or `act`) writes bytecode next to it. The
  submission sha256 is sha256 of the tag `topple.submission/1\n`, then, for every file in
  POSIX-path string order, `len(path)‖path‖len(bytes)‖bytes` (8-byte big-endian lengths; `path`
  is the UTF-8 POSIX path relative to the submission directory, or the file's name for a
  single-file submission; `bytes` exactly as on disk, never normalized).
- **The sha256 covers the code that runs.** The entry file (the `.py` file, or `controller.py`)
  is read once: those bytes are hashed, and those bytes are compiled (as Python compiles a
  module, inheriting no `__future__` flags from the loader) and run as the submission's module.
  No bytecode cache is consulted, so a `.pyc` planted in a `__pycache__` next to a single-file
  submission (an unchecked-hash `.pyc` would run without being compared to its source) never
  runs, and the file cannot change between being hashed and being run. Modules the submission
  imports itself are ordinary imports: in a directory submission they are hashed files.

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
  noise as JSON-serializable data, identically on every platform: initial pitch/pitch_rate, push
  list, model overrides, actuator delay, IMU biases and mounting offset. Trajectories embed it.
  Model overrides are **multiplicative scales** relative to the robot params
  (`chassis_mass_scale`, `wheel_friction_scale`, `motor_gain` per wheel, `frictionloss_scale`)
  plus `com_x_offset_m` (added to the chassis COM x) and `actuator_delay_s`, not absolute values.
- Scenario draws are portable bit for bit. A draw from U[low, high) takes one raw double `u` from
  the stream (`Generator.random()`: the top 53 bits of a 64-bit output times 2⁻⁵³, the same raw
  draw `Generator.uniform` consumes) and computes `low + (high − low)·u` in IEEE double arithmetic
  with each operation rounded (never a fused multiply-add: numpy's C `uniform` computes the same
  expression, but its macOS arm64 and Linux x86_64 builds disagree in the last bit). Choices (push
  count, push sign, actuator delay) use `Generator.integers`, which only uses raw bits.
- `Scenario.from_dict(data, *, clock=None)` validates what it reads, and `ValueError` rejects
  anything else: exact key sets of string keys (every key `to_dict` writes, none optional:
  `init_pitch_rate_rads` included); numbers that fit a float (a huge JSON integer does not) and
  are finite; integer `start_step`/`n_substeps`, exact in a float (|n| ≤ 2**53);
  pushes sorted by `start_step` with no two at the same step, `start_step ≥ 0`,
  `n_substeps ≥ 1`, `0 ≤ magnitude_n ≤ 50` N, `forward_sign = ±1`, `|lateral_ratio| ≤ 1`; scales
  in [0.5, 2] (frictionloss [0, 3]), `|com_x_offset_m| ≤ 0.02`, `0 ≤ actuator_delay_s ≤ 0.010`,
  `|imu_pitch_offset_rad| ≤ 0.1`, `|gyro_bias| ≤ 0.1` rad/s per axis. With the env's
  `EpisodeClock` (`timestep`, `frame_skip`, `max_steps`: `BalanceEnv.clock`) every push must also
  fit the episode: `start_step < max_steps`; `n_substeps ≤ max_steps · frame_skip`; impulse
  `magnitude_n · n_substeps · timestep ≤ 5` N·s (the force cap held for a standard 0.1 s pulse);
  and each push ends before the next begins
  (`start_step · frame_skip + n_substeps ≤ next.start_step · frame_skip`). Every preset draws
  well inside these bounds; `reset(options={"scenario": dict})` goes through this check with the
  env's clock. In-process `Scenario` objects and the `pushes`/`init_pitch_rad`/`init_pitch_rate`
  reset options are trusted test hooks and skip the bounds (robustness tests use absurd pushes),
  but every reset path still requires finite initial pitch and rate and pushes sorted by
  `start_step` that never overlap: the env runs one push at a time, so two at one step, or one
  starting while another is active, would silently cut one short.
- `reset(seed=None)` (training): `episode_seed = int(self.np_random.integers(2**53))`, returned in
  `info["episode_seed"]` so any training failure can be replayed.

## 10. Disturbance presets

| Preset | Sensors/estimator | Model randomization | Pushes |
|---|---|---|---|
| `ideal` | ground truth | none | none |
| `clean` | full (§5) | full | none |
| `standard` | full | full | 1–3 per episode |

Every preset, `ideal` included, draws the initial pitch U(−3°, 3°) from the INIT stream (pitch
rate 0), so one seed starts every preset at the same pitch.

Randomization (`clean`, `standard`), per episode:
chassis mass ×U(0.9, 1.1) (inertia scaled with it), COM x offset U(−3, 3) mm, wheel–floor friction
×U(0.9, 1.1), motor gain common ×U(0.9, 1.1) × per-wheel ×U(0.95, 1.05) (gain only; back-EMF
fixed), frictionloss ×U(0.5, 1.5), actuator delay ∈ {0, 2, 4, 6, 8, 10} ms, IMU mounting pitch
offset U(−1°, 1°), gyro bias U(−0.01, 0.01) rad/s per axis.

Pushes (`standard`): count ∈ {1, 2, 3}; start times drawn with the first at t ≥ 1.0 s, spacing
≥ 1.5 s and the last at t ≤ 8.5 s, then rounded to the control grid
(`start_step = round(t / control_dt)`), so pushes start on steps 100 to 850 inclusive, at least
150 steps apart (the last lands exactly on 8.50 s for about 0.15% of seeds, public seed 1001
among them); rectangular pulse of 0.1 s applied at site `push_point` (`mj_applyFT`); direction fixed
in the world frame at onset: ± along the heading, lateral component ≤ 0.3 of the magnitude;
the application point (`push_point`'s position) and the Jacobian `mj_applyFT` uses come from
MuJoCo's data of the previous substep, so like contacts and the IMU (§5, §11) they lag the state
by one substep (2 ms);
impulse magnitude **[calibrated]** U(0.25, 0.45) N·s (target: the reference LQR survives ≥ 80%
and the shipped PID 40–90% of `standard` episodes; measured on seeds 0..199 (darwin-arm64):
LQR 86.5%, PID 64.5%, both 100% on `clean`). Force is capped at 50 N.

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

Computed from the trajectory by a pure function (`scoring.episode_metrics`), using only stored data,
so every score is re-runnable on a `topple.traj/1` file. On the trusted path the trajectory is the
verifier's own replay (§16), never an upload. With T steps, `qpos` rows 0..T−1 are the states the
controller acted on and row T is the final (possibly terminal) state:

- `upright_frac = T / max_steps`; `time_upright_s = T · control_dt`;
  `survived` = the episode was truncated (`termination.reason == "timeout"` and `T == max_steps`)
- `rms_tilt_rad`: root mean square of the true tilt angle (between body z and world z, from the
  root quaternion) over the alive rows 0..T−1 only; the terminal row T never counts (row 0 alone
  when T = 0). `max_tilt_rad` is the maximum over all rows 0..T
- `drift_m` = planar (xy) distance of the root from row 0 at the final row T; `max_drift_m` = its
  maximum over all rows
- `energy_j`, over control steps k and wheels i:
  `fsum_{k,i}(max(0, V_nom·i_stall·u_i[k]·(u_i[k] − ω_i[k]/ω_noload))) · control_dt`, where
  `u_i[k]` is the clipped command `ctrl` of step k (as issued, before the native actuator delay)
  and `ω_i[k] = (θ_i[k+1] − θ_i[k]) / control_dt` is the mean wheel speed over the step, from the
  recorded chassis-relative wheel angles `θ_i[k] = (qpos[k, wheel_i] − qpos[0, wheel_i]) ·
  wheel_sign_i` (forward positive). This is the battery energy of an ideal DC motor in slow
  decay; the per-step powers are summed exactly (`math.fsum`) and the sum is multiplied by
  `control_dt` once. The same formula applies to real logs (commands plus encoder angles).
  0 when T = 0
- `saturation_fraction` = fraction of the T steps where any |u| ≥ 0.99 after clipping (0 when T = 0)

Episode score (0–100):
`100 · upright_frac · (0.6 + 0.2·(1 − min(rms_tilt/0.2, 1)) + 0.1·(1 − min(drift_m/1, 1))
+ 0.1·(1 − min(energy_j/E_ref, 1)))`. An episode with an error scores 0.

`success = survived ∧ drift_m < 1 ∧ energy_j ≤ 5·E_ref`. `E_ref` **[calibrated]** = 2× the PID
baseline's median energy on `balance-v1/standard` (seeds 0..199, all episodes), rounded to 0.1 J,
frozen: 2 × 6.49 J → `E_ref = 13.0` J (measured on darwin-arm64; physics, unlike scenarios, is
only reproducible per platform, §14).

Protocol score = `fsum` of episode scores in seed order / N, and success rate = successes / N,
where N is the number of episodes the protocol expects: a missing or errored episode counts as 0.
Mean time upright is averaged over the episodes present (those with metrics). Ranking: round(score,
2), ties broken by success rate, then mean time upright.

## 13. Bump rule and hashes

Anything that can change a fixed controller's score on a fixed seed requires a new challenge
major version (`balance-v2`): MJCF, robot params, physics options, presets, scorer, E_ref, seeds,
episode length, control rate, observation semantics. Reward shaping and training configs never bump.

Env code is shared by all challenges. A code change that can change a fixed controller's results
on a fixed seed (sensing, estimation, termination, pushes, model overrides) must bump
`topple.challenge.ENV_ALGO_VERSION` (currently 1). It is part of every `protocol_hash`, so results
from different env algorithms are never compared; once a challenge is published such a change also
needs its new major version. Known-answer tests (`tests/test_golden.py`) pin controller seeds,
scenarios (seeds 0, 1000 and 2**53−1 on every preset), the bundle and protocol hashes and the
PID/LQR baseline scores on public seeds 1000–1002 of `standard` (per platform, to 1e-9), so such
a change cannot land unnoticed.

- `bundle_hash()` = sha256 of the tag `topple.bundle/1\n`, then, for each bundle file in POSIX-path
  order, `len(path)‖path‖len(bytes)‖bytes` (8-byte big-endian lengths, path relative to the bundle
  root, CRLF → LF in text files: `.xml`, `.yaml` and `.yml`, as for `robot_hash`, §3). The bundle
  files are exactly: `challenge.yaml`, the scene, the robot params, every `*.xml` in the robot
  MJCF's directory, and every file the scene or robot MJCF references (§3). Other files in the
  directory do not count; a listed file outside the bundle directory is an error.
- `protocol_hash(preset, seeds, salt)` = sha256 of the canonical JSON (UTF-8; keys sorted at
  every level; separators `,` and `:` with no whitespace; floats in Python's shortest round-trip
  form, so `13.0` is written `13.0`, never `13`: Python's
  `json.dumps(…, sort_keys=True, separators=(",", ":"), allow_nan=False)`; the standard
  protocol's text is pinned in `tests/test_challenge.py`) of
  `{format: "topple.protocol/1", env_algo_version, bundle: bundle_hash,
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
| `obs` | T × 10 | f4 | the features of observation k, exactly as the controller received them (§6) |
| `action` | T × 2 | f8 | raw controller output, before clipping (validated, so finite) |
| `ctrl` | T × 2 | f8 | clipped command issued at step k (before native delay) |
| `enc_counts` | T × 2 | i4 | encoder counts after the step |
| `gyro`, `accel` | T × 3 | f4 | last-substep IMU readings (after bias/noise) |
| `push_force` | T × 3 | f8 | world-frame push force active during step k (0 if none) |
| `epilogue_qpos` | E × nq | f8 | the 1 s epilogue (E = 100) after a fall or a controller error (§11); E = 0 otherwise. Replay only |
| `meta_json` | n | u1 | UTF-8 JSON: format, challenge, preset, episode_index, scenario, versions, robot_hash, wheel_sign, joint addresses, termination {step, reason}, error (null or {code, step, detail}), control_dt, timestep, frame_skip, max_steps, obs schema and features |

If the controller errors at step `k`, the episode stops there (T = k) and the error is stored in
`meta.error`. The env is a pure function of (protocol, scenario, ctrl sequence), so a trajectory
can be re-simulated open loop and must reproduce exactly on the same platform and MuJoCo version.
A trajectory file is untrusted data: nothing in it is ever scored directly; the trusted path
verifies and re-scores it by replay (§16).

**Writing** (`Trajectory.save_npz(path)`) is atomic: the file is written to a new temporary file
in the target directory (created exclusively, under a random name), mode 0644 whatever the umask,
then renamed over `path`. An output directory can come from someone else's checkout, so a symlink
at `path` is replaced, never written through; a failed write leaves `path` as it was. Replay
videos are written the same way.

**Loading** (`load_npz(path, *, max_steps)`; `max_steps` is required, so every caller bounds T)
reads the archive and each member by hand and raises `ValueError`, naming the file, for anything
else, including unreadable, truncated or missing files and corrupted members:

- a regular file (symlinks are followed) of at most `MAX_FILE_BYTES = 33554432` bytes (32 MiB;
  an honest balance-v1 file is about 0.2 MB); a FIFO, device or directory is refused without
  blocking on it or reading from it;
- a zip archive whose end-of-central-directory record ends the file (no archive comment, no
  zip64 records, one disk) and whose central directory is at most `MAX_DIRECTORY_BYTES = 16384`
  bytes (an honest one is under 1 KiB), checked before the directory is parsed: a file listing
  thousands of entries is refused at the cost of reading its last 42 bytes;
- exactly the 11 members above, no duplicates, each stored or deflated (no other compression
  method, no encryption) and at most 16 MiB uncompressed as its directory entry declares.
  Members are read at most 64 KiB at a time, so neither a size that lies nor a large read can
  make decompression produce more than the read asked for;
- each an `.npy` of format 1.0 or 2.0 whose header is at most `MAX_NPY_HEADER_BYTES = 4096`
  bytes, checked before the header is read. The header text is parsed by the loader, never
  evaluated: exactly `{'descr': <str>, 'fortran_order': <bool>, 'shape': <tuple of ints>}` in
  that key order, as numpy writes it (whitespace and a trailing comma are free; anything else
  is malformed). The array is C-ordered with the exact dtype of the table (no object arrays),
  and its header shape needs exactly the bytes the member holds (a header can never make the
  loader allocate more than the file contains);
- the size limits, checked on each member's header before any of its data is read (a few KB of
  compressed zeros can never make the loader allocate more than an honest file needs):
  `T ≤ max_steps` (every per-step array), `E ≤ max_steps`, at most `MAX_STATE_DIM = 256`
  columns in `qpos`, `qvel` and `epilogue_qpos` (so nq, nv ≤ 256), and `meta_json` at most
  `MAX_META_BYTES = 262144` bytes (256 KiB; an honest one is a few KiB);
- the fixed trailing shapes of the table (`obs` 10; `action`, `ctrl`, `enc_counts` 2; `gyro`,
  `accel`, `push_force` 3), `qpos`/`qvel` with T+1 rows and at least one column, `epilogue_qpos`
  as wide as `qpos`;
- `meta_json` a UTF-8 JSON object with `"format": "topple.traj/1"` and only finite numbers: no
  `NaN`/`Infinity` tokens and no literal that overflows to infinity, such as `1e400`.

`read_meta(path)` applies the same member checks and returns only the metadata.

## 15. Results `topple.results/1`

`results.json` (pydantic, `extra="forbid"`, no NaN/Inf): format, challenge, protocol,
protocol_hash, seed_set (`public`/`private`), versions (topple, mujoco, numpy, obs schema, action
spec), platform, robot_hash, controller {id, sha256}, isolation, complete,
episodes [{index, seed, success, score, metrics, error {code, step, detail}}],
aggregate {score, success_rate, mean_time_upright_s, episodes, errors} (§12).

- **Error codes.** A run records `exception`, `invalid_action`, `timeout`, `crash`, `protocol`,
  `output_limit`, `load_error` or `sim_unstable`. Only trusted verification (§16) assigns
  `incomplete`, `not_reproducible` and `missing`. Every error scores 0. `timeout`, `crash`,
  `protocol`, `output_limit` and `load_error` are the runner's (Phase 3): its controller proxy
  raises `topple.evaluate.RunnerError(code, detail)` from `make_controller`, `reset` or `act`,
  and the episode ends with that code at that step. Only that exact class, holding the code and
  detail it was built with, names a code; anything else raised there is `exception` (§8).
- **`complete`** is true only when every one of the protocol's N episodes is present and none is
  `missing`. A single-seed `topple run` and a single-file `topple verify --out` write false.
- **`isolation`** (`inprocess`/`subprocess`/`container`) records how the controller ran. It does
  not make a run rankable: Phase 3 ranks only results produced by its trusted CI jobs.
- **Bounds.** `controller.id`, `challenge` and `protocol`: 1–200 characters; `platform` and each
  version string ≤ 64; `controller.sha256`, `protocol_hash` and `robot_hash`: 64 lowercase hex
  digits; at most 10,000 episodes; at most 32 metrics per episode, each name 1–64 characters and
  each string value ≤ 64. An error `detail` is untrusted text: it keeps at most 300 characters,
  and lone surrogates and control characters other than `\n` and `\t` become U+FFFD (one for
  one, so re-validating a stored detail changes nothing). `controller.id` is untrusted text too
  (a submission's name): control, format (bidi overrides, zero-width characters) and
  line-separator characters and lone surrogates become U+FFFD, one for one. Escape both before
  rendering.
- **Cross-field checks.** An episode with an error has score 0 and is not a success. Public
  results list their episodes with strictly increasing `index` and a `seed` each,
  `aggregate.episodes == len(episodes)` and `aggregate.errors` equal to the number of episodes
  with an error. A `complete` result has no `missing` episode, and its aggregate is exactly §12's
  aggregate of the listed episodes (N = their number; every episode with metrics reports a
  finite `time_upright_s`). Private-seed results are redacted: `episodes` is null, aggregate
  only.
- **Reading** (`read_results`) reads at most 16 MiB and validates all of the above. Validation
  proves a file well formed, never its numbers true: whoever publishes a score (Phase 3's
  publish step, the leaderboard) takes every number from trusted verification (§16), never from
  an uploaded `results.json`.
- **Writing** is atomic and durable: a temporary file in the target directory, mode 0644
  whatever the umask, flushed and `fsync`ed, then renamed over the target, then the directory
  `fsync`ed so the rename survives a power loss. The directory sync is skipped on Windows, when
  the directory cannot be opened for reading (`EACCES`/`EPERM`, as in a write-only drop
  directory), and on file systems that do not support syncing a directory; any other failure to
  sync it is an error, raised after the rename (the complete file is already in place).

## 16. Trusted verification and re-scoring

**Principle.** The trusted path never scores data that an untrusted party wrote. It covers
`topple verify`, `topple score`, Phase 3 publishing, nightly re-evaluation and the leaderboard.

- **Trusted inputs:** the protocol (challenge bundle, preset, seed list, salt), the episode
  index `i` (so the seed is `protocol.seeds[i]`) and the verifier's own Topple and MuJoCo build.
  The controller identity and isolation that `rescore` records come from the trusted caller.
- **Untrusted input:** the uploaded `traj_*.npz`, loaded with
  `load_npz(path, max_steps=<the protocol's max_steps>)` (§14). It only supplies the commands to
  replay and the records the replay must reproduce bit for bit.

**API** (`topple.verify`):

```python
VerifyCode = Literal["not_reproducible", "incomplete"]

@dataclass(frozen=True)
class EpisodeVerdict:
    index: int
    ok: bool                       # bound to (protocol, index) and reproduced bit-exactly
    code: VerifyCode | None
    reason: str                    # human-readable; the first failing check
    replay: Trajectory | None      # trusted replay as far as it ran (None if not replayed)
    result: EpisodeResult          # scored only from `replay` and the env's own metadata

@dataclass(frozen=True)
class RunVerdict:
    results: EvalResults           # rescore's results (redacted for a private protocol)
    failed: int                    # episodes not_reproducible, incomplete or missing
    ok: bool                       # property: failed == 0

def verify_episode(upload, protocol, index) -> EpisodeVerdict
def verify_with_seed(upload, protocol, seed, *, index=0) -> EpisodeVerdict
def rescore(protocol, uploads: Mapping[int, Trajectory], controller, isolation) -> EvalResults
def verify_run(protocol, uploads: Mapping[int, Trajectory], controller, isolation) -> RunVerdict
def protocol_max_steps(protocol) -> int
```

Verification never raises for anything in the upload. `verify_episode` raises `ValueError` only
for an index outside the protocol, `verify_with_seed` for a seed outside `[0, 2**53)` or a
negative index (caller bugs).
`replay` is the trusted env's own record of the replay, with one exception: `action`, the raw
controller output (§14), cannot be re-derived open loop, so `replay.action` is the upload's,
checked in step 3 (finite, and `clip(action, −1, 1) == ctrl`). Nothing is scored from it.
`verify_with_seed` is the local/debug binding for seeds outside the protocol list (`topple run
--seed`): the scenario is still re-derived from the seed, `meta.episode_index` is not checked,
and `index` only labels the verdict.

**Checks, in order.** The first failing check is the verdict's `reason`; its code is
`not_reproducible` unless stated otherwise.

1. **Meta binding.** `meta.challenge == protocol.challenge.id`, `meta.preset == protocol.preset`,
   `meta.robot_hash == robot_hash(<the challenge's robot>)` (all strings) and, for
   `verify_episode`, `meta.episode_index == index` (an integer, not a boolean). `meta.error`, if
   present and not null, must be exactly `{code, step, detail}` with `code` a run error code
   (§15; never a verification code), `step == T` and `detail` a string. A malformed report is a
   forgery; a well-formed one is sanitized (§15).
2. **Scenario.** The trusted env (the protocol's own challenge object and preset, recording) is
   reset on the trusted seed, which materializes the trusted scenario exactly as evaluation does
   (`make_scenario(seed, preset, timestep, control_dt, episode_seconds)`, §9). `meta.scenario`
   must equal the trusted scenario's `to_dict()` as canonical JSON (sorted keys, so `1`, `1.0`
   and `true` all differ): "scenario differs from the scenario of the protocol seed". The replay
   always runs the trusted scenario, never the uploaded one.
3. **Commands.** `T ≤ max_steps`; every `ctrl` value finite and in [−1, 1]; every `action` value
   finite; `clip(action, −1, 1) == ctrl` row by row.
4. **Replay.** The uploaded `ctrl` rows are stepped in order. If the env terminates or truncates
   before the rows run out, the upload does not reproduce; the verdict's replay is then the
   replay up to that step (no error, no epilogue). Then the episode's trusted error is
   decided: a `sim_unstable` termination in the replay gives error `sim_unstable` at step T
   (whatever the upload claims); otherwise the upload's own well-formed error report is kept;
   otherwise, if the replay neither terminated nor truncated at step T (so T < max_steps), the
   verdict is **`incomplete`** (score 0): the commands stop early with no explanation, which
   covers controller errors stripped from `meta`. The epilogue (§11) is simulated when
   `evaluate.wants_epilogue(<replay termination reason or "timeout">, <trusted error present>)`.
5. **Integrity.** These arrays must equal the replay's byte for byte (so `−0.0 ≠ 0.0`), compared
   in this order, the reason naming the first differing array and row: `qpos, qvel, obs, ctrl,
   enc_counts, gyro, accel, push_force, epilogue_qpos`. Then the env-produced metadata
   `termination, max_steps, control_dt, timestep, frame_skip, joint_qpos, wheel_sign, obs_schema,
   obs_features` must equal the replay's as canonical JSON. `meta.versions` is informational.
6. **Score.** Metrics, score and success come only from the replay and its env-produced metadata
   (`episode_metrics(replay, motor)`, §12), exactly as `evaluate` scores a live episode. An
   episode with a trusted error (step 4) scores 0 and is not a success: an honest error report
   can only lower a score. A failed verdict (`ok == False`) scores 0, is not a success and
   carries the error `{code: <verify code>, step: T, detail: reason}`; its metrics are the
   replay's if the replay ran, else null.

An honest run verifies OK and its re-score equals `evaluate()`'s result exactly (score, success,
metrics, error), on the same platform and MuJoCo version.

**`rescore`.** Exactly `N = len(protocol.seeds)` episodes are expected. An upload keyed by
anything but an index in `[0, N)` raises `ValueError`. A missing index gives an episode with
error `missing`, score 0 and no metrics. The aggregate is §12's (score = `fsum`/N, success rate
= successes/N, mean time upright over the episodes with metrics), and `complete` is true only
when no episode is missing (§15). Private-protocol results are redacted as usual. `rescore`
looks each index up once, in index order, and keeps no upload once it is verified, so a mapping
that loads on lookup holds one upload in memory at a time.

**`verify_run`** returns `rescore`'s results together with `failed`, the number of episodes that
did not verify (`not_reproducible`, `incomplete` or `missing`), counted from the verdicts before
a private protocol's results are redacted; `ok` means none did. Whether a run verified always
comes from `failed`, never from the results: a private protocol's results list no episodes. An
episode with a run error (§15) that reproduces did verify; it only scores 0.

**CLI.** Exit codes: 0 reproduced, 1 verification failed, 2 usage or load error. A crash inside
the verifier exits 2 with `verification crashed: <Type: message>`, never 1, and never prints a
traceback. An error is one line: every character of it that is not printable (control
characters, line breaks, lone surrogates) is shown as its backslash escape, so untrusted text
quoted in it (paths, file names, upload contents) can neither drive a terminal nor start a
GitHub Actions workflow command.

- `topple verify PATH [--protocol P] [--index I | --seed S] [--out results.json]
  [--submission SUB] [--isolation inprocess|subprocess|container]`
  - PATH a **run directory**: every `traj_<index>.npz` in it (1–6 digits) is re-scored with
    `rescore`; the protocol is `--protocol`, else the challenge/preset named by the
    lowest-numbered file. The files are loaded one at a time, as `rescore` reaches them. An
    index outside the protocol, two files for one index, a file that does not load, or
    `--index`/`--seed` exit 2. Exit 1 unless every episode reproduced and none is missing,
    decided by `verify_run` (`ok`), so a private protocol's redacted results cannot hide a
    failure.
  - PATH a **file**: by default the protocol comes from `meta.challenge`/`meta.preset` and the
    index from `meta.episode_index`. If that index is outside the protocol or its seed differs
    from `meta.scenario.seed`, the file is verified with `verify_with_seed` on the scenario's
    seed and the output says `UNBOUND (seed taken from file)`; the scenario is still re-derived
    from that seed. `--index I` binds explicitly; `--seed S` verifies unbound on S.
  - The first output line is `OK: …` for a pass bound to the protocol and `OK (UNBOUND): …` for
    an unbound one. Both exit 0, so the exit code of a default file binding never proves that
    the file is a given episode of the protocol: a script that needs that passes `--index` (or
    re-scores a run directory), as the trusted path does.
  - Without `--protocol`, the protocol (challenge and preset) is read from the upload's own
    labels, so it says only which protocol the uploader chose: a run on the easier `clean`
    preset, labelled `clean`, verifies OK as `balance-v1/clean`. The printed protocol id is the
    only sign of it. Trusted callers (Phase 3 publishing, nightly re-evaluation) always pass
    `--protocol`.
  - `--out` writes the trusted results (§15): for a directory the `rescore` results, for a file
    a one-episode result with `complete: false`. The controller is recorded as
    `{id: <SUB's name>, sha256: <SUB's submission sha256>}`, computed without importing SUB (the
    name is that of the resolved path, a directory's name or a file's stem, exactly the id
    `topple eval SUB` records, so `.` names the current directory), or
    `{id: "unknown", sha256: <64 zeros>}` without `--submission`. `--submission` is resolved
    before anything is verified: one that cannot be hashed, or whose name is not valid UTF-8,
    exits 2 before any verdict is printed. `--isolation` is recorded as given: a label that
    nothing checks (§15).
- `topple score FILE [--protocol P] [--index I | --seed S]` verifies first, then prints JSON:
  `ok, code, reason, bound, protocol, index, seed, score, success, error, metrics`, all from the
  verdict (so `seed` is null for a private protocol). Exit 1 if not ok (the score is then 0); an
  errored episode prints score 0.
- `--seed` (here and in `topple run`) must be an integer in `[0, 2**53)`; `--index` ≥ 0.
