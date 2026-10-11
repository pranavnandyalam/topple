# Phase 1 fix spec: trusted verification and gate findings

Sources:
- Phase 1 review gate `wf_53606a22-6d9`.
- Phase 3 critique `wf_ba18111f-e35`. All three critics rated "verify trusts meta" a BLOCKER.

Normative text goes into `docs/contract.md`. Code changes go in `python/topple/`. Every behaviour change is test-first.

## A. Trusted verification and re-scoring (new contract §16; §14 updated)

**Principle.** The trusted path never scores from data that an untrusted party wrote. The trusted path covers `topple verify`, `topple score`, Phase 3 publish, nightly runs and the leaderboard.

**Trusted inputs:**
- The protocol: challenge bundle, preset, seed list and salt.
- The episode index `i`, so the seed is `protocol.seeds[i]`.
- The verifier's own topple and MuJoCo build.

**Untrusted input:** the uploaded `traj_*.npz`. It is loaded with `load_npz(path, max_steps=<protocol max_steps>)`.

**API** (`topple/verify.py`):

```python
VerifyCode = Literal["not_reproducible", "incomplete"]

@dataclass(frozen=True)
class EpisodeVerdict:
    index: int
    ok: bool                       # bound to (protocol, index) and reproduced bit-exactly
    code: VerifyCode | None
    reason: str                    # human-readable; first failing check
    replay: Trajectory | None      # trusted replay (None if replay impossible)
    result: EpisodeResult          # scored ONLY from `replay` + trusted meta

def verify_episode(upload: Trajectory, protocol: EvalProtocol, index: int) -> EpisodeVerdict
def verify_with_seed(upload: Trajectory, protocol: EvalProtocol, seed: int) -> EpisodeVerdict
    # local/debug binding for seeds outside the protocol list (e.g. `topple run --seed`)
def rescore(protocol: EvalProtocol, uploads: Mapping[int, Trajectory], controller: ControllerInfo,
            isolation: Isolation) -> EvalResults
```

**Checks, in order.** The first failing check becomes `reason`.
1. **Meta binding.** Each of these must match:
   - `meta.challenge == protocol.challenge.id`
   - `meta.preset == protocol.preset`
   - `meta.robot_hash == robot_hash(challenge robot)`
   - `meta.episode_index == index` (`verify_episode` only)

   A mismatch gives `not_reproducible`.
2. **Scenario.** The trusted scenario is `make_scenario(seed, PRESETS[preset], timestep, control_dt, episode_seconds)`, where seed is `protocol.seeds[index]` (or the explicit seed). The upload's `meta.scenario` must be JSON-equal to the trusted scenario `.to_dict()`. A mismatch gives `not_reproducible` ("scenario differs from the protocol seed"). The replay always uses the trusted scenario and never the uploaded one.
3. **Replay.** Use a trusted env: protocol challenge and preset, `record=True`, reset with the trusted scenario. Step the uploaded `ctrl` rows until the env terminates or truncates, or the rows run out.
   - Replay ends earlier than T rows: `not_reproducible`.
   - T < max_steps and the replay neither terminated nor truncated at step T: `incomplete`, score 0. This covers controller errors, genuine or stripped from meta.
   - A `sim_unstable` reason in the replay: the replayed result carries the `sim_unstable` error, score 0.
4. **Integrity.** These arrays must be bit-equal to the replay's: `qpos, qvel, obs, ctrl, enc_counts, gyro, accel, push_force, epilogue_qpos`. In addition, `action` must be finite and `clip(action, -1, 1) == ctrl` row-wise. Any difference gives `not_reproducible`, naming the first differing array.
5. **Score.** Metrics, score and success come only from the replay and its env-produced meta (`episode_metrics(replay, motor)`). The epilogue comes from `env.simulate_epilogue`, following the §11 rule (C).
   - If `meta.error` is present in the upload, the episode keeps that error and scores 0. It is sanitized: the code must be in the enum and the detail is truncated and sanitized. An honest error report can only lower a score.
   - `ok == False` gives score 0, success False and the error `{code: <verify code>, step: T, detail: reason}`.

**`rescore`:**
- Exactly `N = len(protocol.seeds)` indices are expected.
  - A missing index gives an `EpisodeResult` with error code `missing` and score 0.
  - Extra or out-of-range indices raise `ValueError` (the entry is rejected).
- Aggregate: `score = fsum(scores)/N` and `success_rate = successes/N`. `complete` is True only when all N are present and none is `missing`.
- `isolation` and `controller` are arguments supplied by the trusted caller. Nothing is copied from upload data.

**Error codes.** Add `incomplete`, `not_reproducible` and `missing` to `results.ErrorCode` and to contract §15.

**CLI:**
- `topple verify PATH [--protocol P] [--index I | --seed S] [--out results.json]`:
  - If PATH is a run directory with `traj_000..`, run `rescore` and print the aggregate.
  - If PATH is a file:
    - Binding defaults: protocol from `meta.challenge`/`meta.preset` and index from `meta.episode_index`.
    - If that index's protocol seed differs from `meta.scenario.seed`, fall back to `verify_with_seed` with the scenario's seed and print `UNBOUND (seed taken from file)`. The scenario is still re-derived from that seed.
  - Exit codes: 0 OK, 1 verification FAIL, 2 usage/load error. A crash inside verification must not reuse exit code 1; it maps to 2 with a clean message.
- `topple score PATH [same binding flags]`: verify first, then print the replay's metrics and score. Exit 1 if not ok. An errored episode prints score 0.

**Red-team tests** (`tests/test_verify_forgery.py`; every forgery must fail or score 0):
- A: `clean` run relabelled `standard`.
- B: `meta.max_steps` set to the fallen step with termination edited to `timeout`.
- C: `meta.error` deleted from an errored run.
- D: seed swap (index i holds the trajectory of index j).
- E: `push_force` or `epilogue_qpos` zeroed.
- F: pushes removed from `meta.scenario`, seed unchanged.
- G: rescore with dropped episodes gives score/N, `complete=false`.
- H: one good episode copied to every index.
- I: truncated without error gives `incomplete`.
- J: `robot_hash` mismatch.
- K: `ctrl` outside [-1, 1], or `clip(action) != ctrl`.
- L: T > max_steps is rejected at load.
- M: `obs` edited by one ulp.

Honest PID, LQR and random runs must still verify OK, and their rescore must reproduce the `evaluate()` results exactly (score, success, metrics).

## B. Controller misbehaviour and error mapping (§7, §8)
- `run_episode` must never raise for controller misbehaviour.
  - `KeyboardInterrupt` still propagates.
  - `SystemExit`, `GeneratorExit` and other `BaseException`s map to `exception`.
  - Returned objects that are generators, too long, non-real, or raise while being iterated or converted (TypeError, ValueError, OverflowError for huge ints) map to `invalid_action`.
  - `validate_action` itself must catch OverflowError.
- Tests:
  - the `sim_unstable` mapping gives score 0 (force it with an extreme scenario or monkeypatched physics);
  - private-seed protections: the salt is used, no trajectories are saved, and episodes are redacted.

## C. Epilogue (§11)
- The 1.0 s zero-command epilogue runs only after termination (`fell`, `floor_contact`) or a controller error. It never runs after truncation (survival), so success videos end upright.
- Update contract §11, `evaluate.run_episode`, video and verify accordingly.
- Test: a surviving episode has an empty `epilogue_qpos`.

## D. Energy (§12)
- Contract §12 is changed to what the code computes from stored data: per control step, ω = Δθ/control_dt (mean wheel speed over the step) and `energy_j = Σ_k Σ_wheels max(0, V_nom·i_stall·u·(u − ω/ω_nl))·control_dt`. This keeps the score recomputable from `topple.traj/1`.
- E_ref is unchanged.

## E. Aggregates (§12, §15)
- In `summarize` and `rescore`, score = fsum/N with N the expected episodes, and missing or errored episodes count as 0.
- The success rate also uses N.
- The mean upright time is taken over present episodes and documented that way.
- `complete` is defined in §15.

## F. Observation quantization (§6)
- Add `contract.controller_obs(obs_arr: float32[10], t) -> BalancerObs`, the only way a harness builds the controller's observation. It is used by `run_episode`; the Phase 3 runner sends `controller_obs(...).to_wire()`.
- Contract §6: features are float32-quantized, then widened to Python float or f64. The wire carries those exact values, so in-process and out-of-process runs are bit-identical.

## G. Contract §8 honesty
- Public protocols use salt 0, so `rng_seed` (and the initial state) identify a public episode. Public-seed scores are provisional.
- Only salted private-seed runs from trusted CI are rankable (Phase 3).

## H. Cheap hardening (LOW)
- `results.json` mode 0644 (chmod after mkstemp), and fsync before the rename.
- `ids` regexes use `fullmatch`; a trailing newline is rejected.
- CLI: `--seed` validated to [0, 2**53); bad files and protocols give clean messages and exit 2, never a traceback.
- `load_npz`:
  - enforce fixed trailing shapes (obs 10, action/ctrl 2, gyro/accel/push 3, nq/nv match);
  - wrap any non-ValueError parse exception as ValueError;
  - test the per-member cap.
- `ErrorInfo.detail`: replace lone surrogates and control characters (keep `\n` and `\t`) so `write_results` never crashes.

## I. Env and core findings (gate run `wf_753ce331-409`, plus the v2 reviewers whose verifiers died)

### Env (`envs/balance.py`)
- **Stale kinematics.** After `mj_step`, MuJoCo's derived data (`xmat`, contacts, sensordata) describe the state at the start of the substep.
  - Compute the tilt used by the fall check, and the `ideal` observation pitch and pitch_rate, from the current `qpos`/`qvel` (root quaternion), so they are exact.
  - Keep floor contact and IMU readings on MuJoCo's post-step data. Document the one-substep (2 ms) lag in contract §5 and §11.
  - Add a test that compares the ideal obs pitch with the pitch derived from `qpos`.
- **Energy seed.** `_prev_wheel_q` starts at the reset-relative zero (0, 0), not raw `qpos0`. Test with a robot whose wheel hinge has a `ref`.
- **`ideal` sensors.** Record noiseless per-step gyro/accel from sensordata, not constant rows.
- **Complementary filter init.** Keep the code: the filter starts at the accelerometer-implied pitch, i.e. true pitch plus the mount offset, as on real hardware. Rewrite contract §5 to say exactly that, and add that the gyro is rotated by the mount offset. State in §9 the one extra SENSOR block drawn at reset.

### Evaluate (`evaluate.py`)
- `iter_episodes` builds the env from `protocol.challenge` (the object), not by re-loading by id. This keeps hashing and scoring on one bundle.
- Add a helper `wants_epilogue(termination_reason: str, has_error: bool) -> bool`, used by both `evaluate` and `verify` (C).

### Contract types (`contract.py`)
- `TaskSpec.extras` becomes immutable (`MappingProxyType` over a copy). TaskSpec stays hashable or documents why it is not.
- `validate_action` covers OverflowError (B).

### Identifiers (`ids.py`)
- Use `fullmatch` everywhere.
- `gym_id` must be injective: reject challenge ids with a digit-leading segment in the grammar, or map them without collision. Add a collision test.

### Scenarios (`disturbances.py`)
- `Scenario.from_dict` validates: finite numbers, pushes sorted with unique `start_step`, steps inside the episode, magnitudes ≤ the force cap, model scales inside sane bounds. A bad scenario raises ValueError.
- Contract §9: the scenario stores multiplicative scales relative to the robot params, not absolute values. Fix the text.

### Robot (`robot.py`)
`validate_robot` must report each of these violations, with a negative test for each:
- left wheel at y > 0 and right wheel at y < 0;
- wheel joints unlimited;
- wheel bodies have exactly one collision geom, of type cylinder, sphere or capsule;
- no equality constraints;
- no extra world-body geoms or bodies other than the robot;
- the robot can fall (free root joint).

`bind_robot` takes the wheel radius from the single tire geom.

`robot_hash` also covers the bytes of every file the robot XML references (meshes, textures, hfields). Balance-v1 has none, so its hash is unchanged.

### Challenge (`challenge.py`)
- `bundle_hash` hashes an explicit file list (challenge.yaml, scene.xml, robot/*.xml, robot/params.yaml) with POSIX relative paths, CRLF→LF normalization and length framing.
- `protocol_hash(preset)` also covers:
  - the canonical JSON of the preset definition (push config and randomization ranges);
  - the ScoreConfig;
  - the seed list;
  - a module constant `ENV_ALGO_VERSION = 1`, which must change whenever env code changes a fixed controller's results (bump rule, §13).

### Loader (`loader.py`)
- `submission_sha256` uses length-framed `(posix relpath, size, bytes)` records.
- A submission directory containing `__pycache__` or any `.pyc` is a LoadError.
- The size cap is enforced while reading, and symlinks are refused.

### Results (`results.py`)
- Bound string lengths: id ≤ 200, platform ≤ 64, detail ≤ 300.
- Cross-field checks: `aggregate.episodes == len(episodes)` when public; private means `episodes is None`.
- `rankable` is removed or marked advisory. Phase 3 decides rankability by which trusted job produced the run.

### CLI
`topple run` (a single seed) writes `complete: false`. Only full-protocol runs are complete.

### Video
Refuse to render a trajectory whose `meta.robot_hash` differs from the bundle's, with a clear error. Never draw the wrong robot.

### Tests (MEDIUM: test quality)
- Scoring tests that distinguish final drift from max drift, and alive-only RMS tilt from all rows.
- `validate_robot` violation tests.
- A test for the per-member size cap in `load_npz`.

## J. Golden regression (integration stage, after all behaviour changes)
- Known-answer tests pin:
  - `controller_seed` for (seed, salt) samples;
  - `make_scenario` draws for seeds 0, 1000 and 2**53-1 on every preset;
  - `protocol_hash` per preset;
  - PID and LQR episode scores on public seeds 1000..1002 (standard) to 1e-9.
- Re-measure calibration: PID and LQR survival on standard seeds 0..199, and the median PID energy.
  - Update the challenge.yaml comment and contract numbers.
  - Keep `E_ref = 13.3` unless 2× the new median rounds differently. If the bands (LQR ≥ 80%, PID 40–90%) still hold, the push band stays.
- `PLAN.md` and `docs/contract.md` must agree with the code everywhere touched.

## Order of work
1. **Runtime builder:** env, evaluate, contract.py, ids, disturbances, robot, challenge, loader, plus contract §3, §5–§9, §11 and §13. Commit.
2. **Trust builder:** verify, scoring, results, traj, cli, video, plus contract §12, §14–§16 and the forgery red-team tests. Commit.
3. **Integrator:** golden tests, recalibration, full gates (ruff, format, mypy strict, pytest with ≥80% coverage, done-when commands). Commit, push, CI green.
