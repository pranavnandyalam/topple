# Deferred risks after Phase 1

Residual risks that the merged Phase 1 (main `f997394`, PR #3) leaves open and that no Phase 2
deliverable closes. They were found while syncing `docs/design/phase-2.md` with the merged code
(decision log, "Sync with merged Phase 1"). Each entry names the phases that must close it, each
before its own done-when passes, and a suggested rule; the first owning phase's design decides the
final rule and logs it.
"Contract §n" points into `docs/contract.md`.

## (a) A mass below the floor turns any robot into a passive pendulum

- **Risk.** `validate_robot` checks the robot's interface, wheels, tires, axle, spawn height, IMU
  frame, tendons, flexes and contact rules (contract §3), but not where the robot's mass is. A body
  with no geom, or only visual geoms (`contype = conaffinity = 0`), can hang below the floor plane.
  It never touches the floor, so `floor_contact` (contract §11) never fires, and it puts the
  centre of mass below the axle: the chassis rights itself with no control at all.
- **Evidence.** Reproduced for this sync on darwin-arm64 with the merged code: balance-v1's robot
  plus `<body name="keel" pos="0 0 -0.30">` (inertial mass 2.0 kg, one visual sphere) under the
  chassis passes `validate_robot` with no problem. A controller that always returns `(0, 0)` then
  survives the full 10 s and scores 96.93 and 95.25 on `balance-v1/standard` seeds 1000 and 1001.
  The stock robot with the same controller falls at 0.50 s and 0.55 s and scores 3.99 and 4.39.
- **Owning phases.** Every phase that scores a robot it did not ship, before that robot is scored.
  First Phase 3, the first public release: robot-ci's `robot:` input scores a user's own MJCF
  (PLAN.md pillar 2 and Phase 3; contract §3 names robot-ci first among `validate_robot`'s users),
  so the rule lands before that input is accepted, or robot-ci refuses every robot but a shipped
  one until it does. Then Phase 8 (CAD export and real robots go through `validate_robot`),
  Phase 10 (evolved bodies), Phase 12 (generated robots). Ranked balance-v1 and waypoint-v1 scores
  are not exposed: their robot is fixed, and verification binds `robot_hash` (contract §16).
- **Suggested rule.** End the episode as a fall (a new termination reason such as `below_floor`,
  counted like `fell`) when any robot body's centre of mass (`data.xipos[b, 2]`) goes below the
  floor plane, checked after every substep like the tilt. At validation, also refuse a body that
  has mass but no collision geom. The termination rule changes env code, so it needs its own
  `ENV_ALGO_VERSION` bump, or a new challenge version for a published challenge (contract §13).

## (b) Mechanisms inside the robot can still change the wheels' dynamics

- **Risk.** Contract §3 promises that a wheel's dynamics come from `params.yaml` alone.
  `build_model` overwrites the wheel joints' damping, stiffness, friction loss and armature, and
  `validate_robot` refuses tendons and flexes that act on a wheel, explicit contact pairs and
  contact excludes with the world. Contacts between robot bodies remain allowed. A body on its
  own joint (a brake shoe, a clutch arm, a skid) whose collision geom touches a tire or another
  wheel-body geom adds friction, damping or a stop to that wheel through ordinary self-collision,
  and a spring on that body's joint can press it on.
- **Evidence.** By construction; not measured yet. MuJoCo makes contacts between geoms of any two
  bodies that are not parent and child whenever their `contype`/`conaffinity` bits match, robot
  excludes are allowed (contract §3), and no check looks at which robot geoms can reach a tire.
  The Phase 1 re-gate measured the analogous tendon case: a damped tendon on the wheels made the
  shipped PID fall at steps 72–212 on clean seeds 1000–1002, where the stock robot survives at
  about 96 (commit `20e0ed4`, which then refused tendons on wheels).
- **Owning phases.** Phases 3, 8, 10 and 12, as in (a) (Phase 3 first).
- **Suggested rule.** A tire may touch only the floor. At validation, refuse any robot collision
  geom outside the wheel bodies whose collision bits can pair with a tire's, unless MuJoCo already
  filters the pair: geoms of one weld body, the parent-child filter (`filterparent`, on by default
  and in every shipped scene, which owns the physics options, contract §3; it compares weld bodies,
  so it covers the chassis, the wheel bodies' parent, and every body welded to it) or an exclude
  between the two bodies. Without that exemption the rule refuses the shipped robot: its
  `chassis_box` (contype = conaffinity = 1, on `chassis`) pairs with both tires by bits, and only
  the parent-child filter keeps them apart (probe on `build_model` of balance-v1). At run time,
  end the episode with an error on any contact between a wheel-body geom and a geom other than the
  floor. Add a red-team robot with a jointed brake arm to `tests/test_robot.py`.

## (c) Calibration and E_ref were measured on darwin-arm64 only

**Status (2026-10-15): closed for balance-v1.** Re-measured on linux-x86_64 before any Phase 2
training: both bands hold (LQR 87.5%, PID 65.0% survival; both 100% on `clean`), and 2× PID's
median energy (6.566 J) rounds to 13.1, so E_ref moved 13.0 → 13.1 J and linux-x86_64 is now the
calibration platform (`challenge.yaml`, contract §10 and §12; decision log rows R1–R3). Left open:
the darwin-arm64 golden scores in `tests/test_golden.py` must be re-pinned on a Mac. The text
below is the original risk, kept for its rule: every future challenge is calibrated on the ranked
platform.

- **Risk.** balance-v1's calibrated values, the push band U(0.25, 0.45) N·s and E_ref = 13.0 J,
  and the bands they were checked against (LQR ≥ 80% and PID 40–90% survival on `standard` seeds
  0..199) were measured on darwin-arm64. Ranked scores come from linux-x86_64 CI (contract §13),
  and physics differs per platform: on public seeds 1000–1002 the pinned scores differ by up to
  1.33 points (LQR seed 1001: 84.08 on darwin-arm64, 85.40 on linux-x86_64). PID's median energy
  on the ranked platform may round to another E_ref, and the bands may not hold there.
- **Evidence.** The `challenge.yaml` comment ("Re-measured 2026-10-11 on darwin-arm64"); contract
  §10 and §12 name darwin-arm64 as the measuring platform; per-platform pins in
  `tests/test_golden.py`. The Phase 1 merge itself moved E_ref from 13.3 to 13.0 when twice the
  re-measured median (6.49 J) rounded differently (commit `f78d84b`).
- **Owning phase.** Phase 3 (first public release), before balance-v1 is published.
- **Suggested rule.** In the linux-x86_64 CI image, re-measure PID and LQR survival on `standard`
  0..199 and PID's median energy, and record both platforms in the `challenge.yaml` comment. If a
  band fails or twice the median rounds to another E_ref, re-calibrate before publication (new
  bundle and protocol hashes, re-pinned goldens) and re-run or exclude every Phase 2 run bound to
  the old protocol hash (phase-2.md G10). Name the ranked platform as the calibration platform in
  contract §10 and §12. Running this check before 2a training starts costs about a minute of CI
  and avoids re-training later.

## (d) Per-step sensor noise is not bit-portable across platforms

- **Risk.** Scenario draws are portable bit for bit (contract §9), but the SENSOR stream is drawn
  per control step with `Generator.standard_normal` (`BalanceEnv._draw_noise_block`). numpy's
  ziggurat sampler calls `exp`/`log` from the platform's C library on its rare slow path and runs C
  expressions a compiler may contract into fused multiply-adds: the same cause that made
  `Generator.uniform` differ in the last bit between macOS arm64 and Linux x86_64 in Phase 1
  (commit `f78d84b`). Physics also reproduces only per platform and MuJoCo version (contract §14).
  So an observation sequence, and every trajectory, reproduces only on the platform that made it.
- **Evidence.** `python/topple/envs/balance.py` (`_draw_noise_block`); contract §9 limits its
  portability promise to scenario draws; golden episode scores are pinned per platform
  (`tests/test_golden.py`).
- **Owning phases.** Phase 3 (verification and ranking), Phase 5 (Arena), Phase 8 (firmware and
  real logs).
- **Suggested rule.** Verify and rank only on the platform that produced the run (linux-x86_64
  CI). A `results.json` or trajectory from another platform is display-only and never re-scored
  as ranked. Before any feature needs the same observations on two platforms (an in-browser
  re-simulation, hardware in the loop), replace the noise draw with a portable one (integer
  arithmetic on raw bits, as contract §9 does for uniforms) under a new `ENV_ALGO_VERSION` or
  challenge version, with a cross-platform golden of the SENSOR stream.
