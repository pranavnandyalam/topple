# Phase 2 design: Learn (draft for critique)

PLAN.md Phase 2: PPO (16 envs, curriculum none → small pushes → full randomization), TensorBoard,
checkpoints; `topple export` → ONNX with VecNormalize + clipping baked in; `onnx_policy.py`
controller; WaypointTask; SAC comparison. Done when: PPO beats PID on `balance-v1/standard`
(eval_v1) with disturbances; ONNX actions match torch within 1e-5; a side-by-side PID vs PPO video
exists.

Phase 1 facts this builds on (see docs/contract.md):
- `topple.envs.BalanceEnv(preset, record=False, reward=RewardConfig, max_drift_m=None)`, Gymnasium,
  obs float32 (10,) = obs_v1 features, action Box(-1,1,(2,)), bounded reward r ∈ [0,1],
  `set_preset()` curriculum hook (next reset), `reset(seed=None)` draws an episode seed.
- Env speed ≈ 14k control steps/s per process (M5 Pro). 18 cores.
- Baselines on balance-v1/standard (200 held-out seeds): LQR ~90% survival, PID ~66%; E_ref ~12.7 J
  (being re-measured after a friction-randomization fix).
- Scoring/eval: `topple.evaluate.evaluate(protocol, factory, ControllerInfo)`; results.json;
  trajectories verify bit-exactly by open-loop replay.

## 1. Training (`topple/train/`)

- `TrainConfig` (pydantic, frozen, extra=forbid) loaded from YAML (`configs/train/*.yaml`):
  algo (`ppo` | `sac`), seed, total_timesteps, n_envs (16), curriculum
  `[{preset, until_step}]`, env (max_drift_m=2.0, reward weights), policy (net_arch [64, 64],
  activation tanh, log_std_init), algo hyperparameters, normalize (VecNormalize obs only,
  clip 10; reward already bounded), eval (every N steps on held-out seeds 2000..2049 with the real
  harness → best checkpoint by protocol score), checkpoint_every.
- `topple train CONFIG --out runs/<name>`: SubprocVecEnv (seeds seed+i), TensorBoard logs,
  checkpoints, `best/` (model.zip + vecnormalize.pkl), `train_meta.json` (config hash, git
  commit, versions, wall-clock, steps/s).
- Curriculum: a callback calls `vec_env.env_method("set_preset", p)` when crossing a stage boundary.
- Reproducibility: seeded SB3 + env seeds; `torch.set_num_threads(1)` in workers.

## 2. Export (`topple/export.py`)

- `export_onnx(run_dir, out_dir)`: torch wrapper = VecNormalize obs normalization (mean/var, clip,
  epsilon exactly as SB3) → actor MLP → Gaussian mean → clip to [-1, 1]. opset 17, float32,
  input `obs` [batch, 10], output `action` [batch, 2]. `metadata_props`: topple.challenge,
  topple.obs_schema (balancer.obs.v1), topple.obs_features, topple.action_spec, topple.control_dt,
  topple.train_config_sha256, topple.git_commit.
- Writes a ready-to-submit directory: `policy.onnx`, `controller.py` (make_controller → ONNXPolicy),
  `policy.json` (provenance).
- Parity test: torch deterministic action vs onnxruntime on 1k random + 1k on-policy observations,
  max abs diff ≤ 1e-5.

## 3. `controllers/onnx_policy.py`

- `ONNXPolicy(path, spec)`: onnxruntime CPU EP, intra/inter op threads 1, validates metadata against
  the TaskSpec (schema id, features, control_dt) and refuses mismatches; `act(obs)` →
  `obs.to_array()[None]` → run → (left, right). Deterministic.

## 4. Results + report

- `topple compare A B ... --protocol eval_v1` → markdown table (score, success, mean upright,
  drift, energy, saturation) + `topple video --side-by-side` composite of the same seed.
- `docs/results/phase2.md`: PID vs LQR vs PPO vs SAC on eval_v1 (public) and 200 held-out seeds,
  training curves (steps, wall-clock), reward-design notes.
- Shipped `submissions/ppo_balance/` (ONNX + controller.py).

## 5. WaypointTask (`waypoint-v1`)

- New challenge bundle sharing the robot; obs schema `balancer-target.obs.v1` = obs_v1 features +
  (target_x, target_y) in the robot frame; new RNG stream TARGETS=6 (append-only).
- Episode 20 s; 4 waypoints within 1.5 m, reached when within 0.10 m (robot frame distance), next
  appears; pushes as `standard`.
- Score: 100·(0.5·waypoints_reached/4 + 0.3·upright_frac + 0.2·quality terms) — to be critiqued.
- Train PPO from scratch vs fine-tune the balance policy (input layer extended with zeros for the
  2 new features); compare sample efficiency.

## 6. SAC comparison

- Same env and seeds; record wall-clock, env steps to reach PID-level score, final score.
