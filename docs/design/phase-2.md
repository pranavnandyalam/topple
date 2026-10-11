# Phase 2 design: Learn (v4, normative)

v4 resolves re-critique round 2. Every issue from the six critiques (`wf_3f0c2a9a-a51`,
`wf_2fb93652-cd0`, `wf_bc1d1edf-2dc`) and both re-critiques is decided in
`docs/design/phase-2-decisions.md`.

The design targets the **post-fix** Phase 1 APIs (`phase-1-sim-core/docs/design/phase-1-fixes.md`):
`controller_obs(obs_arr, t)`, `wants_epilogue`, trusted `verify_episode` / `rescore`, error codes
`incomplete`, `not_reproducible`, `missing`, aggregate = fsum/N, no self-attested `rankable`,
`ENV_ALGO_VERSION` in `protocol_hash`, float32-quantized observations. **2a training starts only
after the Phase 1 fix branch has merged** (§2.3 binds every run to the env version).

"MUST" marks a check the gate enforces. Phase 2 has two parts, each with its own done-when and
review gate:
- **2a, balance:** PPO and SAC training, a pickle-free ONNX export, the data-only
  `topple.policy/1` format, `ONNXPolicy`, the results package and the video.
- **2b, waypoint-v1:** the challenge, a reference driver, PPO from scratch vs fine-tuned. 2b
  starts after 2a passes.

**No literals for E_ref.** E_ref means `challenge.config.score.e_ref_j` of the protocol in use
(balance-v1: 13.3 J today). Code reads it from the `ScoreConfig`; `train_meta.json` and
`phase2.json` record the value used. Every number this spec quotes from a pilot is replaced in
`phase2.md` by a template filled from `phase2.json` (§5.1).

## 0. Facts used

Balance-v1/standard. Gates compare against baselines **measured in the same run**.

| Controller | Seeds | Env | Score | Success | Median energy |
|---|---|---|---|---|---|
| PID (`submissions/pid_example.py`) | 0..199 | pre-fix | 68.88 | 0.66 | 6.65 J |
| PID | 2000..2199 | post-fix | 70.29 | 0.64 | — |
| LQR (`submissions/lqr_example.py`) | 0..199 | post-fix | 80.14 | 0.865 | 3.31 J (pre-fix) |
| PID | eval_v1 (1000..1019) | pre-fix | 71.88 | 0.70 | — |

- **PPO** (8 envs × 2048, 5M steps): pre-fix pilots 84.4–87.3 / success 0.88–0.965 / 2.5–3.4 J /
  2–8% saturated on 0..199. Post-fix seed 0: selection 85.69; on 0..199 PPO − LQR = +5.55 [3.97,
  7.31], success 0.94 vs 0.865. At 1–1.5M steps: 14–23 J, 12–24% saturated, success 0.49–0.60.
- **SAC** (the §2.2 config, post-fix, seed 0, 1.0M steps): selection success by 100k step .12 .125
  .145 .225 .355 .39 .315 .38 .29 .355 (PID 0.64); best 0.39 / 65.2 at 600k; final 0.355 / 67.72,
  survival 0.54, never eligible; `ent_coef` 0.031 → 0.066. Tested and worse: `target_entropy −4`
  (≤ 0.205 / 52.5); UTD 1 (2 envs × gs 2: 400k steps in 729 s, 0.27 / 54.3, 549 steps/s). SAC does
  not reach PID level in about 500 s on this task; a fair "can SAC match PPO" study needs about
  30 min per run and is deferred (§10).
- **Time** (shared machine, load 12–13 on 18 cores): PPO 5M steps train-only 365–406 s (12,763
  steps/s at load 12.9); SAC 1.0M steps 497–513 s (1,949–2,010 steps/s). One 200-seed selection
  eval with 4 spawn workers: PID 3.9 s, ONNX 5.2–5.3 s, waypoint about 13 s. ONNX act() 4–6 µs.
  Estimated PPO train CPU ≈ 9 cores × 400 s ≈ 3,600 CPU-s (§1 caps derive from it; the first run
  records the real `cpu_s`).
- **Memory:** learner + 8 env workers + 4 eval workers = 16 processes, summed RSS 3,224 MB; each
  spawn eval worker about 315 MB. `getrusage` self+children reported 570 MB (5.7× undercount).
- **Parity:** ONNX vs SB3 2.4–7.2e-7 on float32 rows. The closed loop amplifies 1e-7 to above
  1e-3 within about 30 steps, so per-episode results are runtime-specific (§3.5).

Seed sets (all public):

| Set | Seeds | Use |
|---|---|---|
| balance eval_v1 | 1000..1019 | public evaluation |
| waypoint `wp_eval_v1` | 3000..3049 | public evaluation (provisional) |
| calibration | 0..199 | done-when statistics (both challenges) |
| selection | 2000..2199 | checkpoint choice and on-policy parity rows |
| training | [2^52, 2^53) | training episodes |

Phase 3 private seeds MUST come from [0, 2^52) and be disjoint from every set above.

## 1. Machine budget (normative; defaults in code)

The dev box is a shared 48 GB / 18-core laptop; a past run exhausted its memory.
- **Processes.** Shipped configs use `n_envs` ≤ 8 (TrainConfig allows 1..16 for Phase 13). The
  learner calls `torch.set_num_threads(1)`; workers get `OMP_NUM_THREADS=1`.
- **Eval pools** (selection, compare, shove, report): `topple.pool.eval_pool(workers)`, spawn
  `ProcessPoolExecutor`, `max_workers = min(workers, 4, os.cpu_count())`, created once, shut down
  in `finally`. Hand-off is by `PolicyRef` (§3.5), never a pickled session or factory.
- **One training run at a time**, under `~/.claude/resource-guard/heavy.sh` when present.
- **Resource sampler** (`topple/train/resources.py`, needs `psutil`): a daemon thread in the
  learner samples every 1.0 s the summed `memory_info().rss` (bytes) and `cpu_times()` of the
  learner and `children(recursive=True)`. It records `peak_tree_rss_mib`,
  `peak_tree_rss_at_step`, `peak_procs`, and `cpu_s` = tree user+system of live processes plus
  `RUSAGE_CHILDREN` utime+stime of reaped ones.
- **Hard limits** (`limits` in TrainConfig), checked at every callback:
  - tree RSS > `max_tree_rss_mib` (8,192) → `rss_abort`;
  - `cpu_s` > `max_cpu_s` → `cpu_abort` (load-independent budget, 3× the estimated cost:
    balance 12,000, waypoint 15,000);
  - elapsed > `max_wall_clock_s` → `wall_clock_abort` (runaway guard only: balance 2,400,
    waypoint 3,600; about 4.5× the expected duration).

  Each stops training, runs the final snapshot + selection eval, sets the flag and exits 4.
- **Aborted runs.** A run with any abort flag is never shipped or reported as a result. A
  `wall_clock_abort` run is re-run with the same seed (up to 2 re-runs, after which work waits for
  lower load); every attempt is logged in `phase2.json` `attempts[]`. `rss_abort` and `cpu_abort`
  are bugs to fix, not to re-run.
- **Soft target:** 600 s per run, reported only.
- **Tests:** at most 2 envs (DummyVecEnv) and pools of at most 2 workers.

## 2. Training (`topple/train/`)

### 2.1 Modules

```python
# --- torch-free (default CI) ---
# topple/yamlsafe.py
def safe_load_capped(data: bytes, cap: int = 64 * 1024, max_depth: int = 32) -> object
    # size cap; yaml.parse event scan rejects AliasEvent/anchors and depth > max_depth; then SafeLoader.
    # RecursionError / ValueError / UnicodeDecodeError -> ConfigError
# topple/train/config.py  pydantic frozen extra="forbid"; action_excess and algo blocks are discriminated unions
def load_train_config(path: Path, *, seed: int | None = None) -> TrainConfig   # applies --seed; result is the effective config
def config_sha256(cfg: TrainConfig) -> str   # of the EFFECTIVE config (seed included): sha256(json.dumps(model_dump(mode="json"), sort_keys=True, separators=(",", ":")))
# topple/train/wrappers.py  gymnasium only
TRAIN_SEED_LO = 2**52
class TrainingSeeds(gym.Wrapper):        # reset seed = TRAIN_SEED_LO + rng.integers(2**52); incoming seed ignored
    def __init__(self, env, base_seed: int, env_index: int)  # rng = default_rng(SeedSequence([base_seed, env_index, 0x7709]))
class ActionExcessPenalty(gym.Wrapper):  # PPO only; action_space Box(-box, box, (2,)); env still clips to [-1, 1]
    def __init__(self, env, box: float, weight: float)      # r -= weight * min(mean_i(max(|a_i| - 1, 0)**2), 1); info["excess_penalty"]
def make_train_env(cfg: TrainConfig, env_index: int) -> gym.Env  # env -> Monitor -> penalty -> TrainingSeeds
# topple/train/selection.py  ONNX bytes through the real harness
@dataclass(frozen=True)
class SelectionResult:
    step: int; success_rate: float; score: float; median_energy_j: float; mean_saturation: float
    scores: tuple[float, ...]          # per seed, seed order (paired comparisons)
    extras: Mapping[str, float]        # waypoint: mean_gates, docked, fell, median_t_finish_s
def evaluate_policy_bytes(ref: PolicyRef, protocol: EvalProtocol, pool: Executor
                          ) -> tuple[SelectionResult, NDArray[np.float32]]  # + 5 on-policy rows per seed (§2.4)
def is_eligible(r: SelectionResult, base: GateBaseline, sel: SelectionConfig, e_ref_j: float) -> bool  # §2.4
def rank_key(r: SelectionResult, kind: Literal["balance", "waypoint"]) -> tuple[float, ...]       # §2.4
# --- torch (train extra; CLI imports lazily) ---
def train(cfg: TrainConfig, out: Path) -> Path                                   # learn.py
def snapshot(model, vecnorm, step: int, ckpt_root: Path, onnx_bytes: bytes, onpolicy: NDArray[np.float32]) -> Path
class SelectionCallback(BaseCallback); class HealthCallback(BaseCallback)        # callbacks.py
def extend_obs(ckpt_dir: Path, init: InitConfig, balance_scale: float) -> ExtendedInit   # surgery.py (2b), pickle-free
def sanity_check(ckpt_dir: Path, seeds: range) -> float                          # surgery.py (2b item 7): survival rate
```

### 2.2 Configs (complete)

`configs/train/ppo_balance.yaml` (the measured layout):

```yaml
algo: ppo
seed: 0                     # runs use --seed 0, 1, 2
total_timesteps: 5_000_000  # SB3 stops at 5,013,504 (306 rollouts of 16,384)
n_envs: 8
vec_env: subproc            # subproc | dummy (tests)
env:
  challenge: balance-v1
  preset: standard          # from step 0; no curriculum
  max_drift_m: 2.0          # training-only termination (reason "drift")
  reward: {tilt: 0.30, rate: 0.10, drift: 0.25, yaw_rate: 0.05, effort: 0.10, smooth: 0.20}
  action_excess: {enabled: true, box: 2.0, weight: 0.10}
normalize: {obs: true, reward: false, clip_obs: 10.0, epsilon: 1.0e-8}
policy: {net_arch: {pi: [64, 64], vf: [64, 64]}, activation: tanh, log_std_init: 0.0}
ppo: {n_steps: 2048, batch_size: 2048, n_epochs: 10, learning_rate: 3.0e-4, gamma: 0.99, gae_lambda: 0.95,
      clip_range: 0.2, ent_coef: 0.0, vf_coef: 0.5, max_grad_norm: 0.5, target_kl: null}
selection: {protocol: balance-v1/standard, seeds: {first: 2000, last: 2199}, every_steps: 250_000,
            workers: 4, gate: pid, max_saturation: 0.08}   # 19 periodic + final = 20 evals
limits: {max_wall_clock_s: 2400, max_cpu_s: 12000, max_tree_rss_mib: 8192}
health: {min_std: 0.03, min_std_before_step: 2_000_000}
```

`configs/train/sac_balance.yaml` (comparison only; written in full):

```yaml
algo: sac
seed: 0
total_timesteps: 1_000_000  # ~500 s train-only at the measured 1,949-2,010 steps/s
n_envs: 8
vec_env: subproc
env:
  challenge: balance-v1
  preset: standard
  max_drift_m: 2.0
  reward: {tilt: 0.30, rate: 0.10, drift: 0.25, yaw_rate: 0.05, effort: 0.10, smooth: 0.20}
  action_excess: {enabled: false}   # no other keys allowed; tanh actions never exceed +-1
normalize: {obs: true, reward: false, clip_obs: 10.0, epsilon: 1.0e-8}
policy: {net_arch: {pi: [64, 64], qf: [256, 256]}, activation: relu, log_std_init: -3.0}  # SB3 SAC default
sac: {buffer_size: 1_000_000, learning_starts: 10_000, batch_size: 256, learning_rate: 3.0e-4, gamma: 0.99,
      tau: 0.005, train_freq: 1, gradient_steps: 2, ent_coef: auto}   # UTD 0.25 (2 grad steps per 8 env steps)
selection: {protocol: balance-v1/standard, seeds: {first: 2000, last: 2199}, every_steps: 50_000,
            workers: 4, gate: pid, max_saturation: 0.08}   # 19 periodic + final = 20 evals
limits: {max_wall_clock_s: 2400, max_cpu_s: 12000, max_tree_rss_mib: 8192}
# no health block: health is PPO-only (forbidden for SAC)
```

`configs/train/smoke_{ppo,sac}.yaml` (train CI job): the matching file with `total_timesteps:
2048`, `n_envs: 2`, `vec_env: dummy`, `selection: {seeds: {first: 2000, last: 2001},
every_steps: 1024, workers: 2, gate: none}`, `ppo.n_steps: 512`, `ppo.batch_size: 256`,
`sac.learning_starts: 256`, `limits: {max_wall_clock_s: 300, max_cpu_s: 600, max_tree_rss_mib:
4096}`.

`configs/train/ablate_nopenalty.yaml`: `ppo_balance.yaml` with `seed: 1`, `total_timesteps:
1_000_000`, `action_excess: {enabled: false}` (about 80 s; source of the §5.2 item 6 "x").

Rules:
- `algo ∈ {ppo, sac}`; the other algorithm's block is forbidden, `health` is forbidden for SAC.
  There is no `checkpoint_every`: snapshots happen exactly at selection evals.
- **Reward.** `RewardConfig` unchanged. The u² effort term MUST NOT be replaced by a battery-energy
  cost (measured 64.9 / 0.505, 82% saturated); energy terms may only be added on top.
- **log_std.** PPO `log_std_init` 0.0 (−0.5 and −1.0 measured worse).
- **SAC.** UTD 0.25 is the best of the three measured settings for this budget (§0); it is a
  shared-machine choice, not SAC's default (UTD 1).
- **Curriculum.** None. A future study gets a training-only `push_scale`, never `ideal`.

### 2.3 Seeding and metadata

- `PPO/SAC(seed=cfg.seed)`; env i wrapped in `TrainingSeeds(cfg.seed, i)`.
- `train_meta.json` stores only run-relative paths:
  - **identity:** `config_sha256` (effective config), `config_file_sha256` (raw YAML bytes),
    `config` (effective, `seed` included), `git_commit`, `git_dirty`, `versions` {python, topple,
    torch, stable_baselines3, gymnasium, mujoco, numpy, onnx, onnxruntime, protobuf, psutil},
    `torch_threads`, `platform`, `cpu_isa`, `e_ref_j`, `selection_protocol_hash` (protocol + the
    selection seeds), `env_algo_version`, `bundle_hash`;
  - **resources:** `load_avg_start`, `load_avg_end`, `wall_clock_s`, `train_s`, `selection_s`,
    `cpu_s`, `env_steps_per_s`, `peak_tree_rss_mib`, `peak_tree_rss_at_step`, `peak_procs`;
  - **selection:** `gate_baseline`, `selection_history` (rows: step, all SelectionResult fields
    except `scores`, eligible, policy_std_mean, frac_mean_outside_1, excess_penalty_mean, cpu_s,
    wall_s, onnx_sha256), `best`, `flags`, `num_timesteps`.
- `phase2_gate.py` fails if the shipped run's `selection_protocol_hash` or `env_algo_version`
  differs from the current code's.
- Determinism per (config, seed, platform) is best-effort: recorded, not gated.

### 2.4 Snapshots and selection

**Trigger** (exactly one rule per algorithm; labels are unique and strictly increasing):
- **PPO:** in `_on_rollout_start`, when `num_timesteps // every_steps` exceeds its last value.
  The policy has just been updated, and the label `N = num_timesteps` equals the env steps it
  trained on. For 5M: evals at 262,144, 507,904, …, 4,751,360 (19), then the final.
- **SAC:** in `_on_step`, when `num_timesteps // every_steps` increases **and** `num_timesteps <
  total_timesteps`. For 1.0M: 50,000 … 950,000 (19), then the final at 1,000,000.
- **Final:** `_on_training_end` always evaluates at `model.num_timesteps`; by construction no
  periodic label equals it. `snapshot()` raises `FileExistsError` if `ckpt/step_<N>/` exists, and
  `selection_history` steps are asserted unique.

`snapshot()` writes the **immutable** `ckpt/step_<N>/` via `ckpt/.step_<N>.tmp/` + `os.rename`:

| File | Content | Read by |
|---|---|---|
| `actor.npz` | float32 `[out,in]`, keys `W0,b0,W1,b1,W2,b2` for both algorithms (table in §8.8) | export, surgery |
| `critic.npz` | PPO only: `mlp_extractor.value_net.{0,2}.{weight,bias}`, `value_net.{weight,bias}` | surgery |
| `dist.npz` | PPO only: `log_std` (float32 [2]) | surgery |
| `normalize.json` | `mean`, `var` (float64 via `repr`), `count`, `epsilon`, `clip_obs` | export, surgery |
| `parity_ref.npz` | `obs` f32 [N,D], `action_ref` f64 [N,2], `set` u1 [N], `onnx_sha256` (U64 scalar) | export, CI |
| `meta.json` | `step, algo, action_rule, net_arch, activation, config_sha256, onnx_sha256, n_onpolicy, n_onpolicy_seeds` | export |
| `model.zip`, `vecnormalize.pkl` | SB3 pickles | **resume only**, in `learn.py` |

All files come from the live model and `VecNormalize` in the same callback call; `obs_rms` is
read, never updated. `onnx_sha256` is the sha256 of the in-memory ONNX bytes the selection eval
scored.

**Parity rows** (float32 `[N,D]`):
- (a) **on-policy**, from the selection eval itself: each worker returns
  `traj.obs[np.linspace(0, T−1, 5).round().astype(int)]` per episode, in seed order: 5 rows per
  selection seed (1,000 for the shipped configs, 10 for smoke), covering 200 randomizations and
  push schedules at no extra simulation;
- (b) 1,000 random rows `f32(mean + sqrt(var + eps)·U(−12, 12))`, `default_rng(SeedSequence(0))`;
- (c) 64 edge rows `f32` of: `mean`, `mean ± 10·std`, `mean ± 10.001·std`, zeros, ±1e6.

**Reference actions** use SB3's own pipeline on the widened rows:
`a_ref = clip(model.predict(vecnorm.normalize_obs(obs.astype(f64)), deterministic=True)[0], -1, 1)`.

**Selection eval:**
1. Build the ONNX in memory (§3.2) and run `evaluate_policy_bytes` on the selection seeds.
2. **Gate baseline**, measured once at train start with the same harness and seeds: `pid` →
   `submissions/pid_example.py`; `waypoint` → the creeper anchor and the reference driver (2b,
   §8.8); `none` → every checkpoint eligible.
3. **Eligible** iff (`pid`): `success_rate ≥ PID's` ∧ `median_energy_j ≤ E_ref` ∧
   `mean_saturation ≤ selection.max_saturation` (0.08: margin under G4's 0.10); (`waypoint`): §8.8.
4. **Rank key** (per kind, matching the leaderboard): balance `(round(score, 2),
   success_rate)`; waypoint `(round(score, 2), success_rate, −median_t_finish_s)`. Eligibility
   already enforces success ≥ PID, and score-first was never worse on held-out data in 6 runs.
5. `best.json = {"step": N, "eligible": bool, "rule": "<gate>"}` is rewritten atomically (temp +
   `os.replace`) whenever the best changes: the top eligible checkpoint by `rank_key`, or, while
   none is eligible, the top checkpoint with `eligible: false`.
6. Train exits 0 in both cases (4 on a hard limit). Only shipping requires `eligible: true`.

Selection numbers only choose checkpoints. Reported numbers come only from fresh evaluations of
exported artifacts (§4–§5).

### 2.5 Logging, health, runs

- **TensorBoard:** SB3 defaults plus `topple/selection/{success,score,median_energy,saturation}`,
  `topple/policy/std_mean` (PPO: `get_distribution`; SAC: `actor.action_dist`),
  `topple/policy/frac_mean_outside_1` (PPO), `topple/train/excess_penalty_mean` per rollout. The
  Monitor sits inside the penalty wrapper, so `rollout/ep_rew_mean` excludes the penalty.
- **Flags:** `std_collapse` (PPO std < 0.03 before 2M; warning), `high_energy` (any eval above
  E_ref), `no_eligible`, `rss_abort`, `cpu_abort`, `wall_clock_abort`.
- **Runs:** `topple train configs/train/{ppo,sac}_balance.yaml --seed S --out runs/{algo}_s{S}`,
  seeds 0, 1, 2, one at a time, plus `ablate_nopenalty.yaml --out runs/ablate_nopenalty`.
  `runs/` is gitignored.
- **Shipping.** Candidates: PPO runs with `best.eligible`, ordered by their best's `rank_key`
  on the selection seeds (never on 0..199). `phase2_results.py` exports the first, evaluates it
  fresh and moves to the next only if it fails G1–G4 (its own fresh evals; same lock and platform
  as `.venv-eval`, so the numbers are identical). Every attempt is reported. No candidate passing
  → 2a fails.

## 3. Export and the `topple.policy/1` format

### 3.1 Submission layout (new contract §17; §8 "Submission kinds")

A policy submission is **data only**: a real directory (lstat: not a symlink) whose entries are
exactly `policy.onnx`, `policy.json` and optionally `README.md` (≤ 64 KiB, untrusted, never printed
by the CLI), all regular files with `st_nlink == 1`.
- Anything else is `LoadError("unexpected entry: <safe(name)>")`, including subdirectories, hidden
  files (`.DS_Store`), FIFOs, sockets and `.py/.pyc/.so/.pkl/.pickle/.zip/.pt/.pth/.joblib/.ckpt/
  .onnx.data`. `policy.json` plus any `.py` is `LoadError("ambiguous submission kind")`.
- The loader checks for `policy.json` before `controller.py`. Python submissions are unchanged.

`policy.json` is the normative manifest (onnxruntime-web cannot read `metadata_props`).
**Parsing:** `read_regular_file(cap=16 KiB)`; reject a BOM; a rejection-only pre-pass
`json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicates, parse_constant=_reject)`
that also rejects any key or string that fails `.encode("utf-8")` (lone surrogates); then
`PolicyManifest.model_validate_json(data)` (JSON mode: arrays become tuples, strict scalars, int
accepted for float). `RecursionError`, `ValueError` and `UnicodeDecodeError` become `LoadError`.
Every model uses `ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)`:

```python
HEX64 = r"^[0-9a-f]{64}$"; PRINTABLE = r"^[\x20-\x7e]{0,64}$"
class TensorSig(_Strict):  name: Literal["obs", "action"]; dtype: Literal["float32"]; shape: tuple[Literal["batch"], int]
class OnnxInfo(_Strict):   file: Literal["policy.onnx"]; sha256: str  # HEX64 = sha256(policy.onnx) = policy_sha256
                           ir_version: int; opset: int; input: TensorSig; output: TensorSig
class ExportParity(_Strict): reference: Literal["sb3.predict"]; n: int; n_onpolicy: int; n_onpolicy_seeds: int
                             n_unsaturated: int; max_abs_diff: float; tolerance: float; vacuous: bool
class Claims(_Strict):     # SELF-REPORTED: displayed as such; never used for ranking, divisions or "verified"
    name: str | None                      # ^[A-Za-z0-9 _.-]{1,64}$
    algo: Literal["ppo", "sac", "other"] | None; activation: Literal["tanh", "relu", "other"] | None
    net_arch: list[int] | None            # <= 8 entries, each 1..4096
    total_timesteps: int | None; selected_step: int | None; train_seed: int | None   # ints in [0, 2**53)
    train_config_sha256: str | None       # HEX64
    git_commit: str | None                # ^[0-9a-f]{40}$
    git_dirty: bool | None; selection_eligible: bool | None
    versions: dict[str, str] | None       # <= 16 entries; keys ^[a-z0-9_.-]{1,32}$; values PRINTABLE
    export_parity: ExportParity | None    # exporter self-check; never shown as verified
class PolicyManifest(_Strict):
    format: Literal["topple.policy/1"]; kind: Literal["onnx"]; challenge: str; obs_schema: str  # canonical ids
    obs_features: tuple[str, ...]; action_spec: str; action_dim: int; control_dt: float
    action_rule: Literal["clip(mean)", "tanh(mean)", "other"]; onnx: OnnxInfo; claims: Claims
def read_manifest(data: bytes) -> PolicyManifest
def check_against_challenge(m: PolicyManifest, ch: Challenge) -> list[str]
def check_against_model(m: PolicyManifest, proto: onnx.ModelProto) -> list[str]
    # ir_version, the single opset version, input/output (name, float32, ("batch", dim)) equal the proto's
```

Every other claims string matches `PRINTABLE`.

**Identity.** `policy_sha256 = sha256(policy.onnx)`. The loader requires canonical bytes (§3.4),
so this is also `policy.json` `onnx.sha256`, `LoadedPolicy.policy_sha256`, the R2 key and the
Phase 8 header hash. It identifies **bytes**: canonical form removes all non-semantic content, but
two functionally equal graphs (e.g. an extra `Mul` by 1) can still differ; function-level near-copy
detection is the Phase 3 fingerprint (§10). `submission_sha256` (fix I) remains the record of the
whole directory.

### 3.2 Graph construction (`topple/policy/build.py`, torch-free)

`torch.onnx` is not used: its default dynamo exporter needs onnxscript (absent from the lock). The
graph is built with `onnx.helper` from `actor.npz` + `normalize.json`.

```python
@dataclass(frozen=True) class ObsNorm: mean: NDArray[f64]; var: NDArray[f64]; epsilon: float; clip: float
@dataclass(frozen=True) class MlpWeights: layers: tuple[tuple[NDArray[f32], NDArray[f32]], ...]; activation: Literal["tanh", "relu"]
def build_policy_onnx(w: MlpWeights, norm: ObsNorm, *, action_rule: str, metadata: Mapping[str, str]) -> bytes
def reference_actions(w: MlpWeights, norm: ObsNorm, action_rule: str, obs: NDArray[f32]) -> NDArray[f64]
```

- **Normalization** = SB3 `normalize_obs`: `z = clip((x − μ)/sqrt(σ² + ε), −c, c)`, ε =
  `VecNormalize.epsilon` = 1e-8, c = `clip_obs`. Constants are computed in f64 and rounded once:
  `f32(μ)`, `f32(1/sqrt(σ² + ε))`. The builder warns (stderr) for any feature with
  `sqrt(var + ε) < 1e-6·max(1, |μ|)`.
- **`reference_actions`** mirrors the graph: f32 rows, the same f32 constants widened to f64, f64
  arithmetic after that.
- **PPO (9 nodes):** `Sub(obs, mean) → Mul(inv_std) → Clip(lo, hi) → Gemm(W0,b0,transB=1) → Tanh →
  Gemm(W1,b1,transB=1) → Tanh → Gemm(W2,b2,transB=1) → Clip(act_lo, act_hi)`.
- **SAC (10 nodes):** the same with `Relu` hidden layers and head `Gemm(W2,b2) → Tanh → Clip`.
- **Names (canonical, §3.4):** input `obs` `["batch", D]`, output `action` `["batch", 2]`, nodes
  `n{i}`, the output of node i `v{i}` (the last node's is `action`), initializers `c{j}` in first-use
  order (node index, then input position). The Phase 8 C path reads initializers by position in
  the §3.2 sequence.
- **Model:** IR 8, one opset import `("", 17)`, weights inside, `SerializeToString(
  deterministic=True)`; the build output is already canonical. Two builds from the same inputs are
  byte-identical (measured).
- **`metadata_props`** (required): `topple.format`, `topple.challenge`, `topple.obs_schema`,
  `topple.obs_features`, `topple.action_spec`, `topple.control_dt`, `topple.action_rule`; each
  value is `json.dumps(field)`. Load compares `json.loads(value)` with
  `json.loads(json.dumps(manifest.model_dump(mode="json")[field]))` (both sides normalized).

### 3.3 Export and pack (torch-free)

`topple export RUN_DIR --out DIR [--step N] [--report-only] [--test-fixture NAME] [--replace]`
(`export_policy(run_dir, out_dir, *, step=None, report_only=False, replace=False) -> PolicyManifest`):
1. **Resolve.** `step` = `--step` or the integer `best.json.step` (0 ≤ step < 2^53; other keys
   ignored). Path `run_dir/"ckpt"/f"step_{step}"` MUST NOT be a symlink and MUST resolve under
   `run_dir.resolve()`. Without `--report-only`, the step's `selection_history` row MUST have
   `eligible: true` and the output MUST NOT be under `runs/`; with it, the output MUST be
   `run_dir/report_policy/`. `claims.selection_eligible` records the flag.
2. **Read** only `best.json`, `train_meta.json` (json ≤ 64 KiB), `meta.json`, `normalize.json`,
   `actor.npz`, `parity_ref.npz` (npz ≤ 8 MiB), each with `read_regular_file`. npz files are opened
   via `zipfile`: ≤ 8 members, expected names only, total uncompressed ≤ 8 MiB, then
   `np.load(BytesIO, allow_pickle=False)`, dtypes float32/float64/uint8/str only. JSON rejects
   NaN/Infinity.
3. **Build and bind.** Build (§3.2); `sha256(bytes)` MUST equal `meta.json.onnx_sha256` and
   `parity_ref.npz` `onnx_sha256`. `validate_policy_bytes` MUST accept it; `smoke_test` MUST pass.
4. **Parity (normative).** Every `parity_ref` row at batch 1 through `make_session`. `max |a_onnx
   − a_ref| ≤ 1e-5` always MUST hold. Coverage: ≥ 300 rows with both `|a_ref| < 0.99` and
   `n_onpolicy == 5 · n_onpolicy_seeds`. Without `--report-only` coverage MUST hold; with it, a
   coverage miss sets `export_parity.vacuous = true` instead of failing, so every G7 row exists.
5. **Write.** `--out` MUST NOT exist, unless `--replace`: the new directory is written as a temp
   sibling, the old one renamed aside, the new one renamed in, the old one removed. Claims come from
   `meta.json` and `train_meta.json` through a whitelist. `--test-fixture NAME`
   (`^[a-z0-9_]{1,32}$`) writes `tests/data/NAME_parity.npz` (copy), `tests/data/NAME_parity.bin`
   (little-endian `u32 N, u32 D, f32 obs[N·D], f32 a_ref[N·2]`) and `tests/data/NAME_parity.json`
   (`{"onnx_sha256", "n", "d"}`) for C and JS consumers.

Export never imports torch or calls `PPO/SAC/VecNormalize.load`, `pickle`, `cloudpickle`,
`torch.load`. Shared run directories are untrusted.

`topple policy pack MODEL.onnx --challenge ID --out DIR` (third-party models):
1. check 1 (size, parse);
2. `canonicalize` (§3.4), which in pack mode also: drops every field outside the allow-list
   (doc strings, `value_info`, foreign metadata, producer, …) instead of rejecting it, removes
   `Identity` nodes by rewiring, renames the single input/output to `obs`/`action`, and turns a
   static batch dim of 1 into `dim_param "batch"`; it still rejects external data first;
3. add the `topple.*` metadata from the challenge; checks 1–9 on the result; `smoke_test`;
4. write `policy.onnx` + `policy.json` with `action_rule` from the §3.2 sequence match (else
   `"other"`) and `claims.export_parity = None`.

Documented third-party recipe: `torch.onnx.export(net, x, f, dynamo=False, input_names=["obs"],
output_names=["action"], dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}})` (legacy
exporter; measured: one file, IR 9, opset 20, Gemm/Tanh/Constant/Clip), then `topple policy pack`.

### 3.4 Acceptance policy (`topple/policy/validate.py`, torch-free)

```python
MAX_ONNX_BYTES = 1 << 20; MAX_ELEMENTS = 131_072; MAX_COST = 1 << 18; MAX_NODE_INPUTS = 192
def read_regular_file(path: Path, cap: int) -> bytes
@dataclass(frozen=True)
class ValidatedPolicy: canonical: bytes; policy_sha256: str; cost: int; obs_dim: int; action_dim: int
    # constructible only inside validate.py (module-private sentinel checked in __post_init__)
def validate_policy_bytes(data: bytes, obs_dim: int, action_dim: int) -> ValidatedPolicy | list[str]
def canonicalize(proto: onnx.ModelProto, *, pack: bool = False) -> bytes
def static_cost(proto: onnx.ModelProto) -> int
def smoke_test(v: ValidatedPolicy) -> list[str]
def safe(s: str) -> str            # re.sub(r"[^A-Za-z0-9_.:/-]", "?", s)[:64]
```

**`read_regular_file`:** `st = lstat(path)` (not a symlink, `S_ISREG`, `st_nlink == 1`, size ≤
cap); `fd = os.open(path, O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY",
0))`; `fstat(fd)` (st_dev, st_ino) MUST equal the lstat; read at most cap+1 bytes.

**Checks, in order, short-circuiting:** the first check that produces any message returns. No onnx
C++ checker, shape inference, `numpy_helper`, `onnx.external_data_helper` or ORT call ever runs on
bytes that failed checks 1–7.
1. **Size and parse.** ≤ 1 MiB (a [64,64] export is about 21 KB); `onnx.ModelProto.FromString`.
   Never `onnx.load(path)`, `check_model(path)` or `version_converter`.
2. **Field allow-list.** Walk the proto with `ListFields()`; reject any
   set field outside, and any `UnknownFields()`: ModelProto {ir_version, opset_import,
   producer_name, graph, metadata_props}; OperatorSetId {domain ∈ {"", "ai.onnx"}, version};
   GraphProto {node, name, initializer, input, output}; NodeProto {input, output, name, op_type,
   attribute}; AttributeProto {name, type, f, i, t, floats}; TensorProto {dims, data_type,
   raw_data, float_data, name}; ValueInfoProto {name, type}; TypeProto {tensor_type}; Tensor
   {elem_type, shape}; Dimension {dim_value | dim_param}; StringStringEntry {key, value}. This
   rejects `configuration`, `quantization_annotation`, `device_configurations`, `overload`, every
   `denotation`, `ref_attr_name`, `metadata_props` outside the model, node `domain`,
   `external_data` and `data_location`.
3. **Header.** One opset import; (IR, opset) ∈ {(8, 17–18), (9, 19–20), (10, 21)} (ORT-web 1.20
   caps IR at 10); no `functions`, `training_info`, `sparse_initializer`.
4. **Strings.** Node, value, initializer and graph names match `^[A-Za-z0-9_.:/-]{1,128}$`;
   `op_type` and attribute names match `^[A-Za-z]{1,32}$`; ≤ 16 `metadata_props`, keys
   `^topple\.[a-z_]{1,32}$`, values ≤ 1,024 chars; `producer_name` ≤ 128 chars.
5. **I/O.** One input `obs` float32 `[sym, obs_dim]`, one output `action` float32
   `[sym, action_dim]`, same symbolic dim; no initializer in `graph.input`.
6. **Ops, arity, attributes.** Ops and input counts: `Gemm` 2..3; `MatMul Add Sub Mul Div` 2;
   `Neg Sqrt Reciprocal Tanh Relu LeakyRelu Elu Sigmoid Softsign Softplus` 1; `Clip` 1..3;
   `Min Max` 2..4; `Constant` 0; every op exactly 1 output; no empty input names;
   `Σ len(node.input) ≤ 192`. Attributes (each `type` matching the schema): Constant only `value`
   (float32 tensor) | `value_float` | `value_floats`; Gemm `alpha`, `beta` (FLOAT), `transA = 0`,
   `transB ∈ {0, 1}` (INT); LeakyRelu/Elu `alpha`; others none. Every attribute float is finite.
   Gemm's B and MatMul's second input MUST be an initializer or Constant output.
7. **Tensors** (initializers and Constant values): float32, exactly one of `raw_data` /
   `float_data`, rank ≤ 2, dims 1..4096, exactly prod(dims) finite values. Model-wide: ≤ 32
   initializers, ≤ 131,072 elements incl. Constants, ≤ 4,096 elements per Constant, ≤ 64 nodes.
   Values are read with `struct`/`np.frombuffer` on the raw fields only.
8. **Structure and cost.** `check_model(proto, full_check=True)`; `infer_shapes(proto,
   strict_mode=True, check_type=True)`. Every node output except Constant outputs MUST be
   float32, rank 2, dim0 the `obs` batch symbol, dim1 static ≤ 4096. Every initializer is used and
   every node reaches `action`. `static_cost = Σ_{Gemm,MatMul} K·N + Σ_{other} max(1, n_inputs −
   1)·dim1(output) ≤ 2^18` (first-party ≈ 5,000; the variadic term closes the 245,000-input `Max`
   bomb measured at 89 ms per act).
9. **Canonical.** `canonicalize(proto) == data`, else `"policy.onnx is not canonical; run topple
   policy pack"`.

**Error text** is fixed vocabulary plus `safe(s)`. Pydantic errors render only `'.'.join(safe(str(p))
for p in loc)` and `type` (never `msg`, `input` or `ctx`). onnx/ORT exceptions become fixed codes
(`onnx_checker_failed`, `shape_inference_failed`, `ort_load_failed`); raw text only with
`--debug`, locally.

**Canonical form** (`canonicalize`) is constructive: a fresh `ModelProto` built only from
allow-listed parts. It reads tensors from raw fields only (`struct.pack` of `float_data`, copy of
`raw_data`), asserts `data_location == DEFAULT` and empty `external_data` itself, and never calls
`numpy_helper` or `onnx.external_data_helper`. It sets IR and the single opset (domain `""`),
`producer_name = "topple"`, `graph.name = "topple_policy"`, the §3.2 names, batch `dim_param
"batch"`, `topple.*` model metadata in key order, every tensor as little-endian `raw_data`, then
`SerializeToString(deterministic=True)`. Node order is kept (ONNX requires topological order).
It is a fixed point; ORT only ever sees canonical bytes.

**smoke_test:** 32 batch-1 calls (zeros, then `default_rng(1)` normals) through `make_session`;
each output float32 `(1, action_dim)` and finite. Per call: a fresh `ort.RunOptions()`, a
`threading.Timer(1.0)` that sets `terminate = True`, `run`, `Timer.cancel()` in `finally`. A
terminate error is `"smoke call exceeded 1 s"`, anything else `"smoke call failed: ort_run_failed"`.
ORT checks `terminate` only between kernels, so it is not a bound; the static cost is the
acceptance bound and the Phase 3 process kill the runtime bound. The p99 is recorded as
`latency_p99_us` (never an acceptance criterion). `ONNXPolicy.act` uses no RunOptions.

**Structure match.** If the graph equals a §3.2 sequence, `action_rule` MUST match its head;
otherwise it MUST be `"other"`. The Phase 8 C path accepts only the two §3.2 sequences.

Loosening is additive; tightening is a format bump to `topple.policy/2`.

### 3.5 Runtime (`topple/policy/runtime.py`, `topple/controllers/onnx_policy.py`, `topple/pool.py`)

```python
def make_session(v: ValidatedPolicy) -> ort.InferenceSession  # the only InferenceSession call site; re-asserts no external data
class ONNXPolicy:                                             # implements Controller
    def __init__(self, session: ort.InferenceSession, obs_dim: int, action_dim: int) -> None
    def reset(self, rng_seed: int) -> None                     # no-op: stateless, deterministic
    def act(self, obs: Observation) -> tuple[float, ...]
@dataclass(frozen=True)
class LoadedPolicy: manifest: PolicyManifest | None; validated: ValidatedPolicy; session: ort.InferenceSession
def load_policy_dir(path: Path, *, challenge: Challenge) -> LoadedPolicy        # full validation + smoke
def policy_from_bytes(data: bytes, *, challenge: Challenge, expected_sha256: str | None = None) -> LoadedPolicy
# topple/pool.py
@dataclass(frozen=True)
class PolicyRef: kind: Literal["onnx", "python"]; payload: bytes | str; sha256: str; challenge_id: str
def policy_ref(sub: Path, *, challenge: Challenge) -> PolicyRef   # parent: reads + validates ONCE
def worker_factory(ref: PolicyRef) -> ControllerFactory           # worker: re-checks sha256 before the first episode
def eval_pool(workers: int) -> ProcessPoolExecutor
```

- **SessionOptions:** `intra_op_num_threads=1`, `inter_op_num_threads=1`, `ORT_SEQUENTIAL`,
  `ORT_DISABLE_ALL`, `log_severity_level=3`, `use_deterministic_compute=True`; config entries
  exactly `session.load_model_format=ONNX`, `session.allow_released_opsets_only=1`,
  `session.strict_shape_type_inference=1`; `providers=["CPUExecutionProvider"]`, asserted via
  `get_providers()`. Bytes only; scrub `ORT_*` from `os.environ` before the first session.
- **`act()`** feeds `obs.to_array()[None]`, checks float32 `(1, A)`, returns Python floats.
  Non-finite values become `invalid_action` via `validate_action`.
- **`load_policy_dir`:** directory rules (§3.1), `read_regular_file` both files, manifest parse,
  `sha256 == manifest.onnx.sha256`, `validate_policy_bytes`, `check_against_model`, structure
  match, metadata vs manifest, `check_against_challenge`, `smoke_test`. Dims come only from
  `challenge.task_spec`. Any failure is one `LoadError`.
- **Pools.** The parent builds a `PolicyRef` once: ONNX → (canonical bytes ≤ 1 MiB,
  `policy_sha256`); Python → (path, `submission_sha256`). A worker calls `policy_from_bytes(...,
  expected_sha256=...)` or re-hashes the Python submission and raises on mismatch before its
  first episode. `LoadedPolicy`, sessions and factories are never pickled. Results return by
  pickle; this is acceptable locally only (local eval is not a security boundary; §10 (iii)).
- **Determinism** (contract §17): bit-exact per (os, arch, `cpu_isa`, onnxruntime version);
  across runtimes within 1e-5, never bit-exact, so replays play stored trajectories. No test pins
  an ONNX episode score; tests compare orderings only (PPO > PID).

### 3.6 Loader, results, contract

- **Loader.** `load_submission(path, *, challenge)`: `challenge` is required for `kind == "onnx"`
  (`TypeError` if missing). CLI `eval`, `run`, `compare` pass the protocol's challenge.
- **Results** (before `topple.results/1` is published): `Versions` gains `onnxruntime`, `onnx`,
  `cpu_isa: str | None` (`avx512f` / `avx2` / `sse4` from `/proc/cpuinfo` on Linux; `arm64-sme` /
  `arm64-neon` from `sysctl hw.optional` on macOS); `ControllerInfo` gains `policy_sha256`.
- **Contract.** §8 (submission kinds), new §17 (`topple.policy/1` = §3.1–§3.5), §15.

## 4. Done-when for 2a (every gate MUST pass)

`scripts/phase2_done.sh` (`set -euo pipefail`; evidence from the **shipped ONNX directory** in a
separate torch-free environment, so the developer's `.venv` is never modified):

```bash
export UV_PROJECT_ENVIRONMENT=.venv-eval            # gitignored
uv sync --locked --no-dev
uv run --no-sync python -c 'import importlib.util,sys; sys.exit(importlib.util.find_spec("torch") is not None)'
uv run --no-sync topple eval submissions/ppo_balance -p eval_v1 --out out/ppo_eval_v1
S="submissions/pid_example.py submissions/lqr_example.py submissions/ppo_balance"
uv run --no-sync topple compare $S -p balance-v1/standard --seeds 0..199 --workers 4 --json out/cmp_cal.json
uv run --no-sync topple compare $S -p eval_v1 --workers 4 --json out/cmp_eval_v1.json
unset UV_PROJECT_ENVIRONMENT
K=$(uv run python -c 'import json;print(json.load(open("docs/results/phase2.json"))["shipped"]["run"])')
D=$(mktemp -d)/p2x
uv run topple export "runs/$K" --out "$D" && cmp "$D/policy.onnx" submissions/ppo_balance/policy.onnx
uv run python scripts/phase2_gate.py out/cmp_cal.json out/cmp_eval_v1.json out/ppo_eval_v1/results.json \
    docs/results/phase2.json --submission submissions/ppo_balance --results-dir docs/results/phase2
gh run list --branch "$(git branch --show-current)" --workflow ci.yml --limit 1 --json conclusion,headSha \
  | uv run python scripts/phase2_ci_check.py "$(git rev-parse HEAD)"   # G9
```

`phase2_gate.py` exits 0 iff G1–G8, G10 and G11 pass and names each failure. Mean saturation is
the unweighted mean of per-episode `saturation_fraction`. Paired rows are read from the compare
JSON and recomputed with `topple.stats.paired_bootstrap` from the per-seed arrays; the two MUST be
equal.

| Gate | Check |
|---|---|
| G1 | eval_v1 protocol score: PPO > PID |
| G2 | 0..199: `paired[ppo, pid]` lower bound > 0 |
| G2b | 0..199: `paired[ppo, lqr]` lower bound > 0 (measured +5.55 [3.97, 7.31] post-fix), and success(PPO) ≥ success(LQR) − 0.03 |
| G3 | success(PPO) ≥ success(PID) on 0..199 and on eval_v1 |
| G4 | 0..199: PPO median energy ≤ E_ref and mean saturation ≤ 0.10 |
| G5 | batch-1 parity of `submissions/ppo_balance` against `tests/data/ppo_balance_parity.npz` (≤ 1e-5, ≥ 300 unsaturated rows, fixture `onnx_sha256` == `onnx.sha256`) |
| G6 | `results.json` `controller.policy_sha256` == `onnx.sha256` and `controller.sha256` == `submission_sha256(submissions/ppo_balance)`; `versions.onnxruntime` set; `load_policy_dir` accepts it |
| G7 | `phase2.json` has 3 PPO + 3 SAC runs, each with a fresh-eval row (ineligible ones via `--report-only`, marked "below PID gate") |
| G8 | `docs/results/phase2/` files exist within the §5.2 caps |
| G9 | latest `ci.yml` run on this branch has `headSha == HEAD` and `conclusion == success` (both jobs, coverage ≥ 80% in each scope, §7, `test_shipped_policy` on ubuntu) |
| G10 | every reported run: no `rss_abort` / `cpu_abort` / `wall_clock_abort`, `num_timesteps ≥ total_timesteps`, `peak_tree_rss_mib < 6144`; the shipped run's protocol hash and env version are current (§2.3) |
| G11 | ≥ 2 of 3 PPO runs have `best.eligible` and their exported policies pass G2, G3 and G4 on 0..199 in fresh evals; per-run table in `phase2.json` |

## 5. Results package

### 5.1 Regeneration

`heavy.sh uv run python scripts/phase2_results.py --ppo runs/ppo_s{0,1,2} --sac runs/sac_s{0,1,2}
--ablation runs/ablate_nopenalty --out docs/results`:
- exports every run (candidates per §2.5; the shipped one with `topple export runs/ppo_sK --out
  submissions/ppo_balance --test-fixture ppo_balance --replace`; the rest `--report-only`);
- evaluates each fresh on eval_v1 and 0..199 (with PID and LQR);
- writes `docs/results/phase2.json`: every number, every eval row with `platform`, `cpu_isa` and
  `onnxruntime`; each run's committed summary (config_sha256, wall_clock_s, train_s,
  selection_s, cpu_s, load_avg, peak_tree_rss_mib, peak_procs, selection_history, best, flags,
  e_ref_j, attempts); `shipped`; the G11 table; and `pilot` constants (energy-cost variant 64.9 /
  0.505; log_std −0.5/−1 results), each with `source: "<workflow run id>"`, rendered as "pilot
  measurement";
- renders `phase2.md` and `docs/results/phase2/*` from it.

Nothing is hand-edited; prose with numbers is a template filled from `phase2.json`, and the script
fails on a missing template field or an exceeded size cap. Charts are matplotlib PNGs (`dev`).

### 5.2 Contents of `phase2.md`

Under the scoreboard: "Measured on {platform} {cpu_isa}, onnxruntime {v}. CI's ubuntu numbers differ
per episode (§3.5); the CI reference is the `test_shipped_policy` `results.json` artifact of run
{ci_run}."

1. **Hero.** PID vs shipped PPO via `topple video --side-by-side`: MP4 960×480 at 50 fps (≤ 8 MB;
   measured 0.24 MB) and GIF 480×240 (≤ 6 MB; measured 4.96 MB at 20 fps). GIF ladder: fps 20 →
   15 → 12, then pane −10%; the settings used go into `phase2.json`; the build fails only if the
   last rung exceeds the cap. Seed rule: among eval_v1 seeds where PID fell and PPO survived, the
   largest score(PPO) − score(PID), ties to the lowest index; if none, the largest difference
   overall. Caption template: "Seed chosen as the largest PPO−PID gap among eval_v1 seeds where PID
   fell; a median seed is below." Also MP4s of the median-difference seed (lower median) for PID vs
   PPO and LQR vs PPO. Both panes play stored trajectories on one clock; a terminated pane plays its
   epilogue, then holds its last frame stamped `FELL at 4.31 s`.
2. **Scoreboard.** Rows PID, LQR, shipped PPO, each PPO seed, each SAC seed (label "SAC (1.0M
   steps, UTD 0.25)"); "k of 3 PPO seeds reached the PID gate"; blocks eval_v1 and 0..199; columns
   score [95% CI], falls, survival, success, median energy, mean saturation, median drift, parameter
   count, env steps, train CPU-minutes. Paired PPO − PID and PPO − LQR. Rows below the gate are
   marked "below PID gate".
3. **Shove test.** Preset `clean`, seeds 0..99, one push at 2.0 s (0.1 s pulse, lateral 0),
   impulses 0.30:0.65:0.025 N·s (15 levels), `forward_sign` +1 and −1. Survival vs impulse for both
   directions; i50 per direction (first downward 50% crossing, linear interpolation) and their
   minimum. Pilot: PID 0.42/0.42, LQR 0.45/0.45, PPO 0.48–0.50.
4. **Learning curves.** Selection success and score vs env steps and vs CPU-minutes, per seed and
   algorithm, PID and LQR lines. "Env steps to PID level" = [previous eval, first eligible eval].
   Wall-clock only as a footnote with the recorded load.
5. **Failure gallery.** PPO's 3 lowest-scoring seeds in 0..199 as GIF 320×320 at 10 fps (≤ 2 MB,
   same ladder), captions: termination reason, time, `fall_cause` (§8.5), last push impulse/time.
6. **Reward-design notes** (templated): u² kept (pilot: energy-cost variant); log_std 0 beat −0.5
   and −1 (pilot); "the action-excess penalty cut the share of steps with the pre-clip action mean
   outside ±1 from {x}% to {y}% at 1M steps (seed 1)", x from `runs/ablate_nopenalty`, y from
   `runs/ppo_s1` at its eval nearest 1M; the 1M-step failure mode; LQR chatter, if observed.
7. **SAC vs PPO**, three labelled comparisons from logged data: (a) equal env steps (PPO's eval
   nearest SAC's best step vs SAC's best), (b) equal training CPU-seconds, (c) full budget. Fixed
   sentences: SAC is "learner-throughput-limited (UTD 0.25, serial updates)"; UTD 0.25 was chosen
   for the shared-machine budget (SAC's default UTD 1 measured 2.6–4.2× slower); target_entropy −4
   and UTD 1 were tested and worse. Peak step and final-vs-best come from `phase2.json`, never fixed
   text.
8. **README snippet.** Hero GIF and a 4-row table PID, LQR, PPO, SAC.

Total size of `docs/results/phase2/` ≤ 20 MB.

### 5.3 Library code (torch-free) and CLI

```python
# topple/stats.py  (normative algorithm; G2/G2b and compare use only this)
def paired_bootstrap(a, b, *, n_resamples=10_000, seed=0, level=0.95) -> tuple[float, float, float]:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)        # seed order
    idx = np.random.default_rng(seed).integers(0, d.size, size=(n_resamples, d.size))   # one call
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2], method="linear")
    return float(d.mean()), float(lo), float(hi)
def bootstrap_ci(x, **kw) -> tuple[float, float, float]                 # paired_bootstrap(x, zeros)
# topple/compare.py: paired rows for every ordered pair i > j (a = entry_i, b = entry_j)
def compare(subs: Sequence[Path], protocol: EvalProtocol, *, workers: int = 4) -> Comparison
def to_markdown(c: Comparison) -> str; def to_json(c: Comparison) -> str
# topple/bench.py
def shove_curve(sub: Path, impulses, seeds, *, forward_sign: int = 1, workers: int = 4) -> ShoveCurve
def impulse_at_50(curve: ShoveCurve) -> float | None
# topple/video.py (extend; scene and obs width from the challenge registry)
def render_side_by_side(trajs, labels, out: Path, *, pane: int = 480, fps: int = 50) -> None
def render_overlay(trajs, labels, out: Path, *, alphas, frame_m: tuple[float, float]) -> None  # waypoint ghost
```

`topple.compare/1` JSON (pydantic, frozen): `{format, protocol, protocol_hash, complete, seeds[],
entries[{id, policy_sha256|null, sha256, scores[], success[], energy_j[], saturation_fraction[],
termination[], extras{}}], paired[{a, b, mean, lo, hi, n_resamples, seed}]}`; arrays in seed order.

CLI: `topple train CONFIG --out DIR [--seed S]` (without the train extra: message + exit 2);
`topple export ...`; `topple policy pack ...`; `topple compare SUB... -p P [--seeds A..B]
[--workers N] [--json PATH]`; `topple shove SUB... [--impulses 0.30:0.65:0.025] [--seeds 0..99]
[--direction both]`; `topple video --side-by-side A.npz B.npz --labels A B --out X.mp4 [--fps N]
[--size W]`. `--seeds` builds a non-protocol `EvalProtocol` whose results have `complete: false`.

## 6. Tests

**Default CI job** (`-m "not train"`, torch-free):

| Test file | What it checks |
|---|---|
| `test_train_config` | Shipped YAMLs validate. Rejected: extra keys, `n_envs` ∉ 1..16, the other algo's block, `health` on SAC, `action_excess: {enabled: false, box: 2}`, `checkpoint_every`, anchors/aliases, depth 8,000 (ConfigError, no RecursionError), > 64 KiB. `config_sha256` independent of key order and different per `--seed`. |
| `test_train_wrappers` | `TrainingSeeds` in [2^52, 2^53), deterministic per (base, index), never hits a §0 set over 10k draws. `ActionExcessPenalty` exact for (0.5, 1.5) and (2, −2), Box(−2, 2), `info["excess_penalty"]`. |
| `test_selection` | `is_eligible` truth table (pid, waypoint, none); `rank_key` per kind; best.json fallback; 5 on-policy rows per seed incl. a 3-step episode; trigger arithmetic: labels unique, increasing, last == total for smoke and the shipped PPO/SAC configs (pure function `eval_labels(cfg)`). |
| `test_policy_build` | PPO 9-node and SAC 10-node sequences, IR 8 / opset 17, canonical names, fixed point, byte-identical rebuild, no `/Users/` `/home/` `site-packages`. Golden: the committed tiny fixture builds to a pinned sha256. `reference_actions` vs ORT ≤ 1e-5 for var ∈ {0, 1e-12, 1e-8, 1, 1e6} × \|μ\| ∈ {0, 1e3}. Negative controls fail at > 1e-3. |
| `test_policy_validate` | Accepts first-party PPO and SAC. Rejects, each at its named check: one case per check-2 field (`configuration`, `quantization_annotation`, `device_configurations`, `overload`, `type.denotation`, `dim.denotation`, input `metadata_props`, initializer `metadata_props`, `ref_attr_name`, node domain, `external_data`); `Sin`; `Flatten`; `Identity`; `variadic_max_245k` (check 6 arity); `max4_width4096_x40` (check 8 cost); width 11; initializer as input; NaN weight; 2 KB metadata; (IR 8, opset 21); IR 14; truncated bytes; `If`; function; sparse initializer; int64 tensor; Constant `value_ints`; Gemm `alpha=NaN`, `transB` as FLOAT; both payloads set; 65 nodes; 131,073 elements; > 1 MiB; `gemm_transA_outer_product`; `bcast_matmul`; `max_bcast` (Max([n,1],[1,n]), check 8); unused initializer; dead node; spoofed `value_info` (check 2); one 256×256 weight reused by 5 Gemms (check 8 cost only; 3 reuses pass); `static_cost == 2^18` accepted, `2^18 + 1` rejected. External-data red team: `chdir` to a tmp dir holding a correctly sized `.env` and a FIFO, references to both rejected at check 2 with `InferenceSession`, `check_model`, `numpy_helper.to_array` monkeypatched to raise. `<b>` in a node name, `op_type`, domain, attribute name: no `<` in any message. |
| `test_policy_canonical` | Fixed point; `float_data` and `raw_data` give equal bytes; renamed values/initializers canonicalize identically; golden: a committed non-canonical fixture canonicalizes (pack mode) to a pinned sha256; non-canonical file rejected at load. |
| `test_policy_pack` | Third-party models (doc strings, foreign metadata, a `;` value name, `x`/`y` I/O names, static batch 1, empty node names, `Identity`) pack to a loadable dir with `action_rule: other`; packing the first-party build is the identity; external data rejected before any tensor read. |
| `test_policy_manifest` | Round-trip from bytes on disk of a real first-party `policy.json`; 16 KiB cap; BOM, NaN, Infinity, 1e400, duplicate keys, lone surrogate, 5,000-digit int, `3.0` for an int rejected; `1` for a float accepted; extra key `<script>` and `claims.name` `<img ...>` give no `<` in the error; claims charset; `check_against_model` per field; challenge mismatches named. |
| `test_onnx_policy` | CPU-only providers; tuple of Python floats; LoadError on sha/metadata/challenge mismatch; one `ideal` episode via `evaluate`; smoke timer: fresh RunOptions per call, cancelled timer, terminate mapped to its message. |
| `test_pool` | `PolicyRef` for ONNX and Python; mutating the files after the parent load leaves the compare on the original sha; a worker given a wrong sha raises. |
| `test_loader_policy` | Detection; ambiguous kind; extra/pickle-family/hidden/subdir entries named; missing `challenge` → TypeError; hard-linked `policy.json`. Red-team (Linux, macOS): symlinked `policy.onnx`, symlink to `/dev/zero`, symlinked dir, lstat/open swap; oversize rejected before read. |
| `test_export_inputs` | Ignores `best.json.dir`; rejects step ≥ 2^53, symlinked or escaping step dirs, npz with 9 members / bad names / oversize / object dtype, NaN in `normalize.json`, bad fixture name, sha mismatch; existing `--out` without `--replace` → exit 2; `--report-only` vacuous coverage recorded, not failed; `.bin`/`.json` fixture layout. |
| `test_shipped_policy` (slow) | `submissions/ppo_balance` loads; fixture sha equals `onnx.sha256`; parity ≤ 1e-5 with ≥ 300 unsaturated rows; eval_v1 PPO > PID; writes `results.json` as a CI artifact. |
| `test_import_hygiene` | Importing `topple.cli`, `topple.export`, `topple.policy.*`, `topple.pool`, `topple.controllers.onnx_policy` leaves no `torch`, `stable_baselines3`, `tensorboard`, `psutil` in `sys.modules`. |
| `test_no_unsafe_loads` | AST scan of `python/topple`, `scripts/`, `tests/`. Outside `topple/train/`: no `pickle, cloudpickle, dill, joblib, shelve, marshal, torch, stable_baselines3` imports (tests: only `train`-marked tests in `test_snapshot_reference`). Torch-free tree: onnx entry point only `onnx.ModelProto.FromString`; banned `onnx.load(`, `load_external_data_for_model`, `onnx.external_data_helper`, `numpy_helper.to_array`, `register_custom_ops_library`; `InferenceSession(` and `add_session_config_entry` (the three §3.5 literal keys) only in `runtime.make_session`. `PPO.load`, `SAC.load`, `VecNormalize.load`, `load_from_zip_file` only in `learn.resume_from`. Everywhere: no `yaml.load/full_load/unsafe_load`, `allow_pickle=True`, `torch.load` without `weights_only=True`. Name-based; `getattr` evasion out of scope. |
| `test_repo_hygiene` | No pickle-family, `.onnx.data` or hidden files under `submissions/`; `.gitignore` covers `runs/`, `*.pkl`, `.venv-eval/`, `.DS_Store`. |
| `test_stats`, `test_compare`, `test_bench` | `paired_bootstrap` known answer pinning (mean, lo, hi) for a committed 200-element vector; n = 1; all-equal; compare round-trip, all ordered pairs on 2 `ideal` seeds; shove both directions and interpolation. |
| `test_video_sbs` | Timeline (hold-last-frame, labels, trims), GIF ladder, frame rule; one `render`-marked 64 px smoke. |
| `test_resource_caps` | Pools clamp to 4; shipped configs `n_envs` ≤ 8; `limits` complete. |
| `test_phase2_gate` | `phase2_gate.py` and `phase2_ci_check.py` on synthetic JSON: pass a good set, name each failing gate. |

**Train CI job** (`-m train`, `uv sync --locked --extra train`):

| Test file | What it checks |
|---|---|
| `test_train_smoke` | `smoke_ppo.yaml`, `smoke_sac.yaml` finish (total = 2 × every_steps): exactly 2 distinct step dirs, labels increasing, last == `model.num_timesteps`; every snapshot file, `best.json`, `train_meta.json` with no absolute paths; export and parity pass; `onnx_sha256` binding holds. < 60 s. |
| `test_snapshot_reference` | `parity_ref` matches SB3 reloaded from `model.zip` + `vecnormalize.pkl` (`training=False`) within 1e-6. |
| `test_pack_torch_legacy` | A Linear-Tanh MLP exported with `torch.onnx.export(dynamo=False)` packs and loads. |
| `test_resources` | 2 spawned children each holding `np.ones(200 * 2**20, np.uint8)`: tree RSS ≥ 380 MiB, `peak_procs ≥ 3`; `rss_abort` (1 MiB limit) and `cpu_abort` (1 s limit) exit 4 after a final snapshot. |
| `test_surgery` (2b) | With the pickles deleted: the 15-input policy's actions equal the 10-input policy's within 1e-6 with new dims at any value, values satisfy `V_15 == 0.1·V_10` within 1e-6; wheel_pos +100 rad gives the same actions as wheel_pos at its mean; `log_std` = −1.0. |

## 7. CI and packaging

- **pyproject.** Base deps: `onnx>=1.23.2,<1.24`, `onnxruntime>=1.31,<1.32`, `protobuf>=7.36,<8`
  (canonical bytes depend on it; the golden sha tests catch drift). An upgrade of any of them is a
  deliberate re-score. `train` extra: `stable-baselines3>=2.9`, `torch>=2.14`, `tensorboard>=2.21`,
  `psutil>=6`; no onnxscript. `dev` gains `matplotlib`. New marker `train`.
- **mypy.** `ignore_missing_imports` for `torch.*`, `stable_baselines3.*`, `tensorboard.*`,
  `psutil.*`; the train job runs mypy with them installed.
- **Coverage.** Default job omits `topple/train/{learn,snapshot,callbacks,surgery,resources}.py`;
  the train job covers them with `--cov=topple.train --cov-config=.coveragerc-train`,
  `fail_under = 80`.
- **ci.yml.** Job `python` adds `-m "not train"` and uploads `test_shipped_policy`'s
  `results.json`. Job `python-train` (ubuntu-24.04, 20 min): `uv sync --locked --extra train`,
  ruff, mypy, `pytest -m train -n 2 --cov...`, torch from the pytorch-cpu index.
- **Repository.** `.gitignore` adds `runs/`, `*.pkl`, `.venv-eval/`, `.DS_Store`. `cli.py` imports
  `topple.train.learn` only inside `train`.

## 8. Phase 2b: `waypoint-v1` (frozen at the end of 2b)

### 8.1 Generalization (balance-v1 results and hashes MUST NOT change)

- **Kinds.** `challenge.yaml` gains `kind: balance | waypoint` (default `balance`; balance-v1 bytes
  unchanged). Registry `CHALLENGE_KINDS[kind] = KindSpec(env_cls, obs_cls, obs_schema,
  obs_features, metrics_fn, score_fn, success_fn, score_model, rank_key, env_algo_version)`. YAML
  never names code.
- **Score config.** `ChallengeConfig.score` is validated against the kind's `score_model`
  (`BalanceScore` = today's `ScoreConfig`, dump unchanged; `WaypointScore` new).
- **Presets.** A bundle may define `preset_defs`. `Challenge.preset(name) -> Preset` returns
  `config.preset_defs[name]` if defined, else `PRESETS[name]`, and accepts only names in
  `config.presets`. Every lookup goes through it: env init / `set_preset` / reset (sensors),
  `make_scenario` callers, trusted `verify_episode`, `protocol_hash` (dumps the resolved preset)
  and `Scenario.from_dict(challenge=...)`. `_MAX_PUSHES` stays 3.
- **ENV_ALGO_VERSION** becomes per kind; balance stays 1. Course-generator changes bump waypoint's.
- **Dispatch** through the registry: `iter_episodes`, `verify_episode`, `rescore`,
  `Challenge.task_spec`, `build_results` (`Versions.obs_schema`), `controller_obs(obs_arr, t, *,
  schema: str = "balancer.obs.v1")` (returns `BalancerObs` or `BalancerTargetObs`; the 2-arg call
  is unchanged), `make_scenario(seed, preset, ..., course: CourseConfig | None)`,
  `load_npz(path, *, max_steps, obs_dim)` (both always from the **trusted** protocol's challenge,
  never from `meta`), `Trajectory._check_shapes`, and `video.py`.
- **Golden tests.** Fix-J goldens pass unchanged; new goldens pin balance-v1 `protocol_hash` per
  preset, the `ScoreConfig` dump, and `make_scenario` for seeds 1000..1019. An AST test forbids
  `PRESETS[` outside `disturbances.py` and `challenge.py`.

### 8.2 Bundle `challenges/waypoint-v1/`

- **Identity.** gym id `topple/Waypoint-v1`; action `diffdrive.effort.v1`; `robot/` byte-identical
  to balance-v1's (test on `robot_hash`); `scene.xml` same physics, visual floor ≥ 5 × 5 m.
- **Episode.** `episode_seconds: 20.0` (`max_steps` 2000), `control_dt: 0.01`.
- **Course.** `course: {n_waypoints: 5, leg_m: 0.70, turn_deg: [30, 150], max_radius_m: 1.5,
  max_attempts: 1000, gate_radius_m: 0.10, dock_hold_steps: 100}`, also in `TaskSpec.extras`.
- **Presets** (`preset_defs`). `clean`: full sensors, balance-v1 randomization, no pushes.
  `standard`: plus 1–2 pushes U(0.15, `impulse_hi`) N·s, 0.1 s pulse, in [2.0, 18.0] s, ≥ 3.0 s
  apart, lateral share ≤ 0.3, 50 N cap; `impulse_hi` set by calibration (§8.7).
- **Seeds.** Public `wp_eval_v1` = 3000..3049, permanent alias `wp_eval_v1 → waypoint-v1/standard`
  in `ids.py`. Per-episode sd on `standard` is large (pilot: driver 5.7 at 0.30 N·s, 8.5 at 0.35;
  learned 19.7–23.3), so 50-seed public scores are provisional (templated ± from `phase2.json`).
  Every 2b comparison uses calibration seeds 0..199 with a paired bootstrap; Phase 3 ranks
  waypoint-v1 on ≥ 200 private seeds.
- **RNG.** New stream `TARGETS = 6`, append-only (contract §9).

### 8.3 Course generation (`topple/course.py`)

`make_course(seed, cfg) -> tuple[tuple[float, float], ...]`, `rng = stream_rng(seed, Stream.TARGETS)`:
1. Each attempt draws, in order, `h0 = rng.uniform(-π, π)`, `mags = rng.uniform(rad(30),
   rad(150), size=4)`, `signs = rng.integers(0, 2, size=4)`.
2. `θ_0 = h0`, `θ_k = θ_{k−1} ± mags[k−1]` (`+` iff `signs[k−1]`).
3. `p_k = p_{k−1} + 0.70·(cos θ_k, sin θ_k)` from `p_{−1} = (0, 0)`.
4. Accept iff every `|p_k| ≤ 1.5` (about 44% accepted); after 1000 failures, `RuntimeError`.

Path length 3.5 m, mean turn 98°. Points are in the start frame (origin = axle midpoint at reset,
x = initial heading). `Scenario.course` is serialized only when non-empty. Waypoints 0–3 are
gates, 4 is the dock.

### 8.4 Observation `balancer-target.obs.v1` (contract §6)

The 10 `obs_v1` features, then `gate_x, gate_y, next_x, next_y, final`: 15 float32 features; on
the wire 16 float64, `t` first (128 B).
- `gate` = current waypoint, `next` = the one after (equal to `gate` at the dock); `final = 1.0`
  iff the current waypoint is the dock. After docking, `gate = next = dock`, `final = 1`.
- Points are in the yaw-only robot frame (x forward, y left), from the **true** axle-midpoint pose
  at the same instant as the other features (an idealized beacon).
- The observation returned by the step that reaches gate k already shows gate k+1.
- `wheel_pos` (≈ 100 rad over a course) and `yaw` (wraps at ±π) are non-stationary on courses.

### 8.5 Rules and events

One pure function, used by the env and the scorer:
`waypoint_events(qpos_xy, course, cfg, *, terminated: bool) -> WaypointEvents` (`reached_step[0..4]`,
`docked`, `t_finish_s`, `d_min`, `max_hold_steps`, `waypoints`), distances via `math.hypot`.
- **Rows.** Reach checks, `d_min` and hold counting use rows k = 1..T if truncated, 1..T−1 if
  terminated (the env skips the check on the terminating step).
- **Gates.** Waypoint i < 4 is reached at the first k with distance ≤ 0.10 m; at most one
  waypoint advances per step.
- **Dock.** `hold_k = hold_{k−1} + 1` if `d_k ≤ 0.10` else 0, counting from the step the dock
  becomes current; docked at the first k with `hold_k = 100`; `t_finish = k·control_dt`;
  `max_hold_steps` = the longest streak (< 100 unless docked).
- **Termination.** The full 20 s unless the §11 rules end it (`fell`, `floor_contact`,
  `sim_unstable`); docking does not end it. Eval has no drift termination.
- **fall_cause** (metric, both kinds): `"push"` if a push started in the 1.0 s before termination,
  else `"self"`; `None` if not terminated.
- **Scorer.** Pure function of the trajectory; waypoints from the trusted scenario's course; a
  test asserts env events == scorer events on 50 seeds.

### 8.6 Score and success

With T = 20 s, L = 0.70 m, r = `gate_radius_m` = 0.10, t_par = 5.0 s:

`score = 55·(g + p)/5 + 25·docked·clip((T − t_finish)/(T − t_par), 0, 1) + 15·(t_alive/T)
+ 5·(t_alive/T)·(1 − min(E/E_ref, 1))`

- **g**: waypoints completed (0..5; the dock counts once held).
- **p**: 0 when g = 5. Current waypoint a gate: `1 − min(d_min/L, 1)`. Current waypoint the dock:
  `0.8·clip((L − d_min)/(L − r), 0, 1) + 0.2·max_hold_steps/100`. `d_min` = closest true distance
  to the current waypoint since the previous one was completed (or the start). Dock progress is
  monotone in closeness and hold and tends to 1 as the hold completes (g + p < 5 until docked).
- **E**: battery energy, contract §12. **E_ref [calibrated]** = 2 × the driver's median energy on
  `standard` 0..199 at the frozen `impulse_hi`, rounded to 0.1 J (pilot 22.4 J).
- Errors score 0. **Success:** `docked ∧ not terminated ∧ E ≤ 5·E_ref`. **Ranking:**
  round(score, 2), then success rate, then median `t_finish` with non-finishers counted as T.
- No drift term. **User wording:** "55 for the course, 25 for speed (full marks at a 5 s finish),
  15 for staying up, 5 for battery". t_par 5 keeps the driver's speed optimum at 0.30 m/s (at 0.30
  and 0.35 N·s) and leaves the first PPO baseline about 5.7 speed points of headroom (19.3 of 25
  per docked episode; at t_par 7 it was 22.3). No hand driver finished under 11.4 s, so t_par is
  not lowered below 5 without a feasibility probe.
- **Metrics:** `waypoints, progress, docked, t_finish_s, max_hold_steps, time_alive_s, energy_j,
  saturation_fraction, rms_tilt, fall_cause`.

### 8.7 Reference driver, calibration, anchors

**Driver** `Driver(v_max, a_max)` (`controllers/waypoint_pid.py`; shipped as
`submissions/waypoint_pid_example.py` with `v_max 0.30`, `a_max 0.8`): balances with the shipped
`LQRGains`, tracking a slew-limited speed reference toward `gate`; along-track error
`d·max(cos herr, 0)²`; `turn = clip(0.8·herr, ±0.30) − 0.06·yaw_rate`, `herr = 0` when `d < 0.03`.

**Calibration** (recorded in a `challenge.yaml` comment and `phase2.json`), in this order:
1. **impulse_hi.** Measure the driver on `standard` 0..199 at impulse_hi ∈ {0.30, 0.35, 0.40,
   0.45, 0.50}; choose the value minimising |falls − 6%| subject to docked ≥ 88%, ties to the lower
   value. Record the whole curve. (Pilot: falls 2.0 / 3.5 / 9.0 / 15.5 / 19.5%, docked 98.0 /
   96.5 / 91.0 / 85.5 / 84.5% → 0.35.) Record the clean − standard driver gap (target ≥ 1.5;
   pilot 0.89 at 0.30, 1.77 at 0.35).
2. **E_ref** from the driver at the chosen impulse_hi.
3. **Anchors** re-measured with the frozen scorer (table below).
4. **Pilot PPO.** One from-scratch seed (`ppo_waypoint.yaml`, seed 0) at the frozen values; check
   §8.9 item 6 and the budget rule (§8.8) before any other 2b training.
5. Every number in §8.2, §8.6, §8.7 and §8.9 is re-measured and templated from these values.

**Anchors** (`topple/controllers/anchors.py`, shipped as `submissions/anchors/*.py` for
`compare`; measured on `standard` 0..199):

| Strategy | Definition | Requirement | Pilot (t_par 5, 0.30 N·s) |
|---|---|---|---|
| Balance in place | LQR, zero speed reference | ≤ 25 | 19.9 (20 seeds) |
| Spin in place | LQR, constant turn 0.30 | ≤ 25 | 19.3 (20 seeds) |
| Creeper | `Driver(v_max=0.12, a_max=0.8)` | ≤ driver − 10 | 60.78 vs 82.87 |
| Gates-only | Driver whose last target is the dock moved 0.30 m further along the last leg (passes through, never holds) | ≤ driver − 5 | 70.29 (earlier variant) |
| Reckless | `Driver(v_max=0.60, a_max=0.8)` | < driver | 73.33 |

### 8.8 Training for 2b

`configs/train/ppo_waypoint.yaml` = `ppo_balance.yaml` with:

```yaml
total_timesteps: 6_000_000     # adopted reward plateaued by 3M (mean waypoint idx 4.52 @3M, 4.55 @4M, 4.28 @5M)
env:
  challenge: waypoint-v1
  preset: standard
  max_drift_m: 2.5             # training-only: course radius 1.5 m + 1 m
  reward: {tilt: 0.30, rate: 0.10, drift: 0.0, yaw_rate: 0.05, effort: 0.10, smooth: 0.20}
  waypoint_reward: {balance_scale: 0.1, progress: 20.0, gate: 10.0, dock: 20.0, terminated: -50.0, dock_hold: 0.0}
  action_excess: {enabled: true, box: 2.0, weight: 0.10}
selection: {protocol: waypoint-v1/standard, seeds: {first: 2000, last: 2199}, every_steps: 500_000,
            workers: 4, gate: waypoint}   # 11 periodic + final = 12 evals
limits: {max_wall_clock_s: 3600, max_cpu_s: 15000, max_tree_rss_mib: 8192}
```

**Budget rule.** If the §8.7 pilot's final selection score exceeds its best at ≤ 4M by more than
2 points, `total_timesteps` becomes 8,000,000 (every_steps unchanged) for both arms; the choice
and its evidence go into `phase2.json`.

**Reward** (not frozen; never bumps, contract §13):
`r = 0.1·r_bal + 20·(d(s, w) − d(s′, w))/L + 10·[gate reached] + 20·[dock completed]
+ dock_hold·[final ∧ d ≤ 0.10] − 50·[terminated]` minus the action-excess penalty.
- `w` is the waypoint current at the **start** of the step; `d` is the true axle distance.
  Progress is **undiscounted** (deliberately not γ-matched: with γ = 0.99 the potential form paid
  `0.2·d′/L` per step for staying away, and 9/30 pilot courses drove away from gate 0). The
  terminal step uses the real `d(s_T)`.
- `[terminated]` covers every training termination (`fell`, `floor_contact`, `sim_unstable`,
  `drift`), not time-limit truncation. Pilot, same seed, 5M: 32.7 vs 77.1 score, 0% vs 86% docked.
- Scale: `0.1·r_bal ≤ 0.1` per step (≤ 200 per episode); course ≈ 100 progress + 40 gates + 20
  dock; a fall −50 plus the lost alive stream.
- **Pre-planned reward iterations** if §8.9 item 6 fails: (1) `dock_hold: 0.2`; (2) additionally
  `terminated: -100.0`. Each is one pilot seed, reported. If both fail, 2b stops and the owner
  decides.

**Selection (gate `waypoint`).** At train start the harness measures the creeper and the
reference driver on the selection seeds. Eligible iff score > creeper's ∧ docked ≥ 0.60 ∧
`paired_bootstrap(scores, driver.scores).lo > −5.0`. Rank by the waypoint `rank_key` (§2.4). Each
eval records mean gates, docked, fell, median `t_finish`.

**Two arms, 3 seeds each:**
- **(a) From scratch:** `ppo_waypoint.yaml`.
- **(b) Fine-tune:** `ppo_waypoint_ft.yaml` = (a) plus `init: {from_run: runs/ppo_sK, step: best,
  zero_inputs: [2, 4, 5], log_std: -1.0}`. `extend_obs` (pickle-free; reads only `best.json`
  (integer step), `actor.npz`, `critic.npz`, `dist.npz`, `normalize.json`, `meta.json`, with the
  §3.3 file rules):
  1. Build `ActorCriticPolicy(obs_space(15), act_space, lr_schedule, **cfg.policy)` and fill it
     via `load_state_dict` using this mapping:

     | npz | SB3 PPO state dict | SB3 SAC (export only) |
     |---|---|---|
     | `W0, b0` | `mlp_extractor.policy_net.0.{weight,bias}` | `actor.latent_pi.0.*` |
     | `W1, b1` | `mlp_extractor.policy_net.2.{weight,bias}` | `actor.latent_pi.2.*` |
     | `W2, b2` | `action_net.{weight,bias}` | `actor.mu.*` |
     | `critic.npz` keys | same names | — |
     | `dist.npz` `log_std` | `log_std` | — |

     Zero-pad input columns 10..14 of `policy_net.0.weight` and `value_net.0.weight` (64×10 →
     64×15).
  2. Zero columns 2, 4, 5 (yaw, wheel_pos_l, wheel_pos_r) of both first layers: with frozen
     balance stats this equals holding them at their training mean (pilot: 0.99 / 87.2 vs 1.00 /
     88.4; unmodified at +100 rad: 0% survival).
  3. Multiply `value_net.weight` and `value_net.bias` by `balance_scale` (0.1), so V matches the
     balance-alive part of the new return (an unscaled critic predicts ≈ 95 vs ≈ 9.5 and makes
     early falls look about 2.7× costlier than −50).
  4. `log_std = −1.0` (std 0.37; the copied 0.04–0.06 would starve the new columns); fresh
     optimizer.
  5. `VecNormalize(15)`, `training=False`: dims 0..9 from the balance run; dims 10..14 analytic:
     gate/next x,y mean 0, var L² = 0.49; `final` mean 0.5, var 0.25.
- Report env steps with and without the 5M pretraining steps, and PPO median energy vs E_ref
  ("learned policies spend ~{k}× the driver's energy and forgo most of the 5 battery points").

### 8.9 Done-when for 2b

`scripts/phase2b_done.sh` (`set -euo pipefail`, evaluation in `.venv-eval`):

```bash
A="submissions/waypoint_pid_example.py submissions/anchors/creeper.py submissions/anchors/gates_only.py \
   submissions/anchors/reckless.py submissions/anchors/balance_in_place.py submissions/anchors/spin.py"
uv run --no-sync topple compare $A submissions/ppo_waypoint -p waypoint-v1/standard --seeds 0..199 --json out/wp_cal.json
uv run --no-sync topple compare submissions/waypoint_pid_example.py submissions/anchors/creeper.py \
   submissions/ppo_waypoint -p wp_eval_v1 --json out/wp_eval.json
uv run python scripts/phase2b_gate.py out/wp_cal.json out/wp_eval.json docs/results/phase2.json
```

`phase2b_gate.py` exits 0 iff all items pass:
1. **Challenge frozen.** Bundle, scorer and calibration comment frozen; the §8.7 band (falls in
   [3%, 10%], docked ≥ 88%) and all five anchor requirements hold on `standard` 0..199.
2. **No regression.** balance-v1 golden tests, per-preset `protocol_hash` and `ScoreConfig` dump
   unchanged.
3. **Verification.** `verify_episode` / `rescore` reproduce 5 waypoint trajectories; an edited
   `meta.scenario.course` gives `not_reproducible`; a 10-wide obs in a waypoint upload and a
   15-wide obs in a balance upload are rejected by `load_npz`; trusted and env scenarios are equal
   on 5 seeds; `waypoint-v1/standard` pushes lie in [2.0, 18.0] s.
4. **Contract** updated: §1 (gym id, `wp_eval_v1`), §4 (`max_steps` per challenge), §6
   (`balancer-target.obs.v1`, replacing the "target_x, target_y" example), §9 (`TARGETS = 6`),
   §10 (per-bundle presets), §12a (waypoint score), §13 (per-kind `ENV_ALGO_VERSION`; mujoco and
   onnxruntime upgrades mean a re-score, not a challenge bump), §14 (obs width), §16 (trusted
   scenario includes `course`). Schema id, feature order, gym id and alias frozen.
5. **Results.** Both arms × 3 seeds in `phase2.md` vs the driver on 0..199 (paired bootstrap) and
   on `wp_eval_v1`, with gates, docked, falls, `t_finish` per selection eval.
6. **Learnability** (relative to the driver, at the frozen `impulse_hi`). `submissions/ppo_waypoint`
   (the best checkpoint of the better arm by `rank_key`) on 0..199: (a) docked ≥ 60%;
   (b) `paired[ppo, driver]` lower bound > −5.0; (c) score > creeper's; (d) fell ≤ driver's fell
   + 0.20. Falls are reported split by `fall_cause`. (Pilot PPO − driver: −0.5 at 0.35 N·s, +0.9 at
   0.40; fell 19.5% vs 3.5%, 24.5% vs 9.0%.)
7. **Fine-tune sanity.** `surgery.sanity_check` on `balance-v1/clean` seeds 0..19 with a test
   wrapper that appends 5 zeros and adds +100.0 rad to features 4 and 5: survival of the full
   10 s ≥ 95%.
8. **Video** `docs/results/phase2/waypoint.{mp4,gif}`: driver and `ppo_waypoint` on one course in
   one scene (driver a 50%-alpha ghost, one clock); seed = the `wp_eval_v1` seed at index 24 of
   the 50 sorted by (PPO score, seed); trimmed at max(t_finish, fall) + 1.5 s per robot, full 20 s
   if neither docks; frame = bounding box of the course ∪ both trajectories' xy + 0.5 m, clamped
   to [3.4, 5.0] m; numbered gates, checkered dock, fading trails, per-robot gate counter, timer
   and finish time; MP4 ≤ 8 MB; GIF 480×480 ≤ 5 MB with ladder fps 10 → 8, then pane −10%.
9. **Export.** `submissions/ppo_waypoint/` is a 15-input `topple.policy/1`; it validates and
   passes parity.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Phase 1 fixes move PID/LQR/E_ref | Train after the fix merges; baselines measured in the same run; E_ref from ScoreConfig; env version and protocol hash bound in train_meta (G10). |
| A seed lands low | 3 seeds, 200 selection seeds, eligibility mirrors G3/G4, fallback to the next run; G11 exposes a broken recipe. |
| Shared-machine load | Wall-clock is a runaway guard only; CPU-seconds cap and axis; aborted runs re-run, never reported. |
| SAC never eligible (expected, §0) | Always a best checkpoint; `--report-only` with vacuous-parity flag gives every G7 row. |
| Linux ORT diverges per episode | Tolerance-based open-loop parity; CI gates only orderings; platform and `cpu_isa` on every row. |
| onnx parser CVEs / model DoS | Bytes-only parsing, field allow-list, arity and static-cost caps, canonical form, pinned onnx/protobuf; sandbox and process kill in Phase 3. |
| Waypoint reward fails the learnability gate | Calibrate first, pilot one seed, two pre-planned reward iterations, then an owner decision. |
| Registry refactor changes balance-v1 | Golden scenario, score, ScoreConfig and per-preset protocol-hash tests; `Challenge.preset` with an AST guard. |

## 10. Deferred (explicitly not Phase 2)

- **Phase 3 runner (MUST).**
  - **Re-run, never replay, for ONNX entries.** The trusted stage fetches `policy.onnx` +
    `policy.json` from the PR head SHA through the API, validates and runs them in a fresh sandbox,
    and produces results itself. Uploaded trajectories are display-only.
  - **(i) No PR code.** The trusted stage runs only base-branch code from a pinned image digest;
    it never checks out, installs (`uv sync`) or imports anything from the PR head. PR content
    arrives only as files fetched through the API, size-capped before parsing (1 MiB, 16 KiB,
    64 KiB).
  - **(ii) Budget.** Total policy CPU ≤ 60 s per protocol run (first-party needs about 2.4 s for
    200 × 2000 × 6 µs), plus the per-act kill (provisional 5 ms) from `scripts/bench_policy_caps.py`
    run on the ubuntu x86 image over variadic, broadcast and denormal-weight shapes (measuring
    `session.set_denormal_as_zero`). Exceeding it is `timeout`, score 0 for the remaining episodes.
  - **(iii) Boundary.** Data crosses the sandbox boundary only as length-capped JSON or
    `topple.traj/1` npz (`allow_pickle=False`, validated), never multiprocessing/pickle; Phase 2's
    pools are not reused there.
  - **(iv) Ownership.** Intake rejects a PR touching `submissions/` outside
    `submissions/<PR author handle>/`; leaderboard rows key on (author, `policy_sha256`).
  - **Identity, R2.** Scores bind to `policy_sha256` computed by the trusted stage. Before upload, a
    second sandbox checks validation and the canonical fixed point; the Arena verifies the fetched
    bytes' sha256 against the leaderboard record before creating a session.
  - **Sandbox:** no network, read-only, cap-drop ALL, no-new-privileges, cgroup `--memory 512m`,
    64 pids, 1 CPU, non-root; one sandbox per policy; 10 s CPU at load; no RLIMIT_AS.
  - **Intake:** scrub `ORT_*`; pin onnx/onnxruntime/protobuf by image digest; check `cpu_isa`.
  - **Provenance:** `OFFICIAL.json` or attestations; near-copy flag via a 256-probe fingerprint
    at 1e-3 (function-level identity, §3.1).
  - **Private seeds:** from [0, 2^52), disjoint from §0; waypoint-v1 ranks on ≥ 200.
  - **`docs/design/phase-3.md` draft MUST be revised:** drop the `controller.py` shim, RLIMIT_AS
    for ONNX children, the 0.25 s act limit for ONNX and the fixed 88 B frame (frame = 8·(1 +
    len(obs_features)) bytes; waypoint 128 B).
- **Phases 3 and 5 (display).** `claims`, README, metadata and errors are untrusted: escaped or not
  rendered; `claims` labelled self-reported.
- **Phase 5 (Arena).** Canonical bytes from R2 by sha256 in a Web Worker on the WASM EP,
  `numThreads=1`; IR/opset checked from `policy.json` (bound to the model, §3.1); a Node CI test of
  onnxruntime-web 1.20 against `tests/data/*_parity.bin` (≤ 1e-5).
- **Phase 6 (autotune).** Linear drift cost with `max_drift_m` 1.5, energy terms on top of u²,
  `push_scale` curriculum. Model-generated YAML goes through `safe_load_capped`.
- **Phase 8 (firmware).** `topple export --target c` reads only `policy.onnx`, accepts only the two
  §3.2 sequences, dispatches on `action_rule`, emits `policy_weights.h` (`float.hex()` literals,
  `policy_sha256`). Done-when: host parity within 1e-5 compiled with and without
  `-ffp-contract=off`; on the ESP32-S3, the `.bin` rows (EMBED_FILES) within 1e-5 and `act()` ≤
  1 ms at 240 MHz (≈ 5k MACs for [64,64]).
- **Phase 4.** Escalating-shove clip 0.35 → 0.60 N·s in 0.05 steps, 2 s apart.
- **Phase 11.** "Verified training" badge only via repo-owned workflows with attestations.
- **Phase 13.** SAC at ~30 min wall-clock per run (UTD 1) for a fair SAC vs PPO claim.
