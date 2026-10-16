# Phase 2 design: Learn (v5.3, normative)

v5 resolves re-critique round 3. Every issue from the six critiques (`wf_3f0c2a9a-a51`,
`wf_2fb93652-cd0`, `wf_bc1d1edf-2dc`) and the three re-critiques is decided in
`docs/design/phase-2-decisions.md`. v5.1 syncs the spec with the merged Phase 1 (main `f997394`,
PR #3); its changes are rows S1–S22 of the log's "Sync with merged Phase 1". v5.2 fixes the 18
findings of the sync's adversarial verification (rows V1–V18 of the log's "Sync verification").
v5.3 records slices 1a, 1b and 1c's resolved deviations (rows B1–B12 of the log's "Slice 1a build",
B13–B16 of "Slice 1b build", B17–B26 of "Slice 1c build"), slice 2's torch-free half (rows B27–B36
of "Slice 2 torch-free build"), slice 3a's gate scripts, C reference and CI jobs as built and
their review fixes (rows B37–B45 of "Slice 3a build") and the linux-x86_64 recalibration
(E_ref 13.0 → 13.1; rows R1–R6 of the log's "Linux recalibration") and Pranav's answers to the
open questions (rows A1–A8 of "Owner answers"), which keep the waypoint scene identical to
balance-v1's (§8.2).

The design builds on the merged Phase 1 code and `docs/contract.md` ("contract §n", also for a
list such as "contract §9, §14"; a bare §n is this spec): `controller_obs(obs_arr, t)`,
`wants_epilogue(termination_reason, has_error)`, trusted `verify_episode` / `verify_with_seed` /
`rescore` / `verify_run` (→ `RunVerdict`), verification codes `incomplete`, `not_reproducible`,
`missing` and the runner's `RunnerError` codes, aggregate = fsum/N, no self-attested `rankable`,
`ENV_ALGO_VERSION` (1) in `protocol_hash`, float32-quantized observations, portable scenario draws,
the canonical-layout trajectory reader and strict `results.json` (contract §9, §14, §15). Phase 2
reuses these; it never re-implements them. Phase 1 has merged, so 2a training may start (§2.3
binds every run to the env version).

"MUST" marks a check the gate enforces. Phase 2 has two parts, each with its own done-when and
review gate:
- **2a, balance:** PPO and SAC training, a pickle-free ONNX export, the data-only
  `topple.policy/1` format, `ONNXPolicy`, the results package and the video.
- **2b, waypoint-v1:** the challenge, a reference driver, PPO from scratch vs fine-tuned. 2b
  starts after 2a passes.

**No literals for E_ref.** E_ref means `challenge.config.score.e_ref_j` of the protocol in use
(balance-v1: 13.1 J = 2 × PID's median 6.5655 J on linux-x86_64, the calibration platform since
2026-10-15; 13.0 when it was measured on darwin-arm64, 13.3 before the Phase 1 recalibration). Code reads
it from the `ScoreConfig`; `train_meta.json` and `phase2.json` record the value used. Every number
this spec quotes from a pilot is replaced in `phase2.md` by a template filled from `phase2.json`
(§5.1).

## 0. Facts used

Balance-v1/standard. Gates compare against baselines **measured in the same run**. Baselines on the
merged Phase 1 env after the linux-x86_64 recalibration (E_ref 13.1, measured 2026-10-15 on
linux-x86_64, Xeon with AVX-512; survival equals success in every row; mean saturation on 0..199:
PID 0.074, LQR 0.046):

| Controller | Seeds | Score | Success | Median energy |
|---|---|---|---|---|
| PID (`submissions/pid_example.py`) | 0..199 | 68.67 | 0.650 | 6.57 J |
| PID | 2000..2199 | 70.90 | 0.660 | 5.99 J |
| PID | eval_v1 (1000..1019) | 72.82 | 0.75 | 7.36 J |
| LQR (`submissions/lqr_example.py`) | 0..199 | 80.36 | 0.875 | 3.28 J |
| LQR | 2000..2199 | 82.77 | 0.905 | 2.77 J |
| LQR | eval_v1 | 85.48 | 0.95 | 3.03 J |

The same rows on darwin-arm64 under E_ref 13.0 (before the recalibration): PID 68.70 / 0.645 /
6.49 J, 70.63 / 0.655 / 5.89 J, 73.05 / 0.75 / 7.38 J; LQR 80.06 / 0.865 / 3.26 J, 82.51 / 0.90 /
2.76 J, 85.42 / 0.95 / 2.84 J.

Every PPO and SAC number below is a pilot on the Phase 1 fix branch before the merge (E_ref 13.3;
there LQR scored 80.14 / 0.865 on 0..199 and PID 70.29 / 0.64 on 2000..2199). It motivates the
recipe only: every gate bar is relative to PID, LQR or E_ref measured in the same run (§2.4, §4),
so the bars follow the recalibration without edits; the energy bar is now 13.1 J.

- **PPO** (pilot env, 8 envs × 2048, batch 2048, 10 epochs, penalty box 2 / weight 0.1, 5M
  steps; pick by selection on 2000..2199, fresh score / success on 0..199, PPO − LQR paired 95% CI):

  | LR | seed 0 | seed 1 | seed 2 | mean (sd) |
  |---|---|---|---|---|
  | constant 3e-4 | 85.69 / 0.94, [3.97, 7.31] | 81.57 / 0.83, [−0.37, 3.18] (G2b fails) | 87.94 / 0.975, [5.94, 9.85] | 85.07 (3.2) |
  | **linear 3e-4 → 0** (shipped) | 85.63 / 0.94, [3.93, 7.23] | 85.82 / 0.955, [3.95, 7.59] | 86.97 / 0.955, [5.12, 8.73] | 86.14 (0.72) |

  Constant LR, seed 1: selection peak 0.85 / 82.83 at 2.26M, then 0.58–0.75 with saturation
  0.108–0.149 from 3.5M (ineligible); policy std 0.057–0.073 by 5M. Linear LR: every checkpoint
  from 1.5–1.75M on is eligible, mean saturation ≤ 0.029, no regression after the peak. The
  pick's selection → calibration shrinkage was −0.75 to −1.33 points in all 6 runs; selection
  seeds are easier (merged, linux-x86_64: LQR 82.77 vs 80.36, PID 70.90 vs 68.67). With the original, unfixed
  Phase 1 env at 1–1.5M steps: 14–23 J, 12–24% saturated, success 0.49–0.60.
- **Determinism:** two seed-1 runs (311,296 steps, 8 forkserver workers, torch 1 thread) gave the
  same params + `obs_rms` sha at loads 6.9 and 5.9; a seed-1 5M run matched bit-for-bit where
  compared (std 0.8825 at step 262,144).
- **SAC** (the §2.2 config, pilot env, seed 0, 1.0M steps): selection success by 100k step .12 .125
  .145 .225 .355 .39 .315 .38 .29 .355 (PID 0.64); best 0.39 / 65.2 at 600k; final 0.355 / 67.72,
  survival 0.54, never eligible; `ent_coef` 0.031 → 0.066. Tested and worse: `target_entropy −4`
  (≤ 0.205 / 52.5); UTD 1 (2 envs × gs 2: 400k steps in 729 s, 0.27 / 54.3, 549 steps/s). SAC does
  not reach PID level in about 500 s on this task; a fair "can SAC match PPO" study needs about
  30 min per run and is deferred (§10).
- **Time** (shared machine, load 12–13 on 18 cores): PPO 5M steps train-only 365–406 s (12,763
  steps/s at load 12.9); SAC 1.0M steps 497–513 s (1,949–2,010 steps/s). One 200-seed selection
  eval with 4 spawn workers: PID 3.9 s, ONNX 5.2–5.3 s, waypoint about 13 s. ONNX act() 4–6 µs.
- **CPU** (4 PPO runs × 5,013,504 steps, train only, load 5.6–6.7): process-tree CPU before
  `venv.close()` 995–1,036 s; after close only the learner's 199–211 s remains, because
  forkserver-reaped workers never reach the learner's `RUSAGE_CHILDREN` (0.3 s). 20 selection
  evals add about 400 CPU-s, so a balance PPO run costs about 1,450 CPU-s.
- **Memory:** learner + 8 env workers + 4 eval workers = 16 processes, summed RSS 3,224 MB; each
  spawn eval worker about 315 MB. `getrusage` self+children reported 570 MB (5.7× undercount).
- **Parity:** ONNX vs SB3 2.4–7.2e-7 on float32 rows. The closed loop amplifies 1e-7 to above
  1e-3 within about 30 steps, so per-episode results are runtime-specific (§3.5).

Seed sets (all public):

| Set | Seeds | Use |
|---|---|---|
| balance eval_v1 | 1000..1019 | public evaluation |
| waypoint `wp_eval_v1` | 3000..3049 | public evaluation (provisional) |
| calibration | 0..199 (balance), 0..399 (waypoint) | done-when statistics |
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
  learner samples every 1.0 s the learner and `children(recursive=True)` (forkserver workers
  included: they are descendants of the learner's forkserver child). It records summed
  `memory_info().rss` → `peak_tree_rss_mib`, `peak_tree_rss_at_step`, `peak_procs`, and keeps a
  **ledger** `ledger[(pid, create_time)] = user + system` (last seen). `cpu_s = sum(ledger.values())`
  over every process ever seen, dead ones included; it is monotone and uses no rusage. The final
  reading is taken in `_on_training_end` **before** `venv.close()` and stored as `cpu_s`.
- **Hard limits** (`limits` in TrainConfig), checked at every callback:
  - tree RSS > `max_tree_rss_mib` (8,192) → `rss_abort`;
  - `cpu_s` > `max_cpu_s` → `cpu_abort` (load-independent budget, about 3× the measured cost:
    balance PPO 4,500 (3 × 1,450); SAC 6,000 (bound: 9 processes × 513 s measured wall + 400 s
    selection ≈ 5,000, unmeasured as CPU); waypoint 7,500 until the calibrated pilot's `cpu_s`
    is measured, then 3 × that value, recorded in `phase2.json`);
  - elapsed > `max_wall_clock_s` → `wall_clock_abort` (runaway guard only: balance 2,400,
    waypoint 3,600; about 4.5× the expected duration).

  Each stops training, runs the final snapshot + selection eval and sets the flag; `topple train`
  exits 4 when the finished run's `train_meta.flags` holds an abort flag.
- **Aborted runs.** A run with any abort flag is never shipped or reported as a result; §2.5
  candidates exclude it and `topple export` without `--report-only` refuses it (exit 2). A
  `wall_clock_abort` run is re-run with `topple train ... --retry`: it renames the existing `--out`
  to `<out>.attempt<k>` (k = 1, 2; only if its `train_meta.json` carries `wall_clock_abort` and the
  re-run's effective `config_sha256`, i.e. the same config and seed; only after
  `topple.train.learn` imports, so a missing train extra never moves a run aside) and starts
  afresh with the same seed (up to 2 re-runs, after which work waits for lower load). The
  re-run MUST reproduce the aborted attempt's `onnx_sha256` at every label both reached;
  otherwise it gets flag `nondeterministic` and is not reportable (G10). `phase2_results.py`
  globs `runs/<name>.attempt*` for each passed run and writes `attempts[] = [{dir, flags,
  wall_clock_s, load_avg_start, cpu_s}]`. `rss_abort` and `cpu_abort` are bugs to fix, not to
  re-run. `topple train` exits 2 if `--out` exists and is not an empty directory (without
  `--retry`).
- **Soft target:** 600 s per run, reported only.
- **Tests:** at most 2 envs (DummyVecEnv) and pools of at most 2 workers.

## 2. Training (`topple/train/`)

### 2.1 Modules

```python
# --- torch-free (default CI) ---
# topple/yamlsafe.py
class ConfigError(ValueError)
def safe_load_capped(data: bytes, cap: int = 64 * 1024, max_depth: int = 32) -> object
    # bytes only; size cap; one yaml.parse event scan rejects AliasEvent/anchors, depth > max_depth (32 levels read, 33
    # refused) and, per mapping, a repeated key, `<<`, a key with an explicit tag (`!!merge`, `!!str`, `!`) or a non-scalar
    # key (PyYAML keeps the last of two equal keys and merges any `!!merge` key); then SafeLoader. RecursionError /
    # UnicodeDecodeError / yaml.YAMLError / any other exception -> ConfigError, one line
def clean_text(text: str, limit: int = 160) -> str   # non-printables escaped, cut with "..."; every input-derived
    # ConfigError text (the parser's problem, pydantic messages; locations cut at 96) goes through it
# topple/train/config.py  pydantic frozen extra="forbid" strict allow_inf_nan=False; discriminated on `algo`: (ppo block,
#   PpoPolicy, health?) | (sac block, SacPolicy, no health); action_excess is a discriminated union on `enabled` (exactly
#   true/false: pydantic's own tag lookup takes 1 for true)
#   PpoPolicy: net_arch {pi: [h1, h2], vf: [h1, h2]} (exactly 2 ints each, 1..256), activation Literal["tanh"], log_std_init (−10..2)
#   SacPolicy: net_arch {pi: [h1, h2], qf: [h1, h2]}, activation Literal["relu"]; no log_std_init (SB3 SAC ignores it without SDE)
#   LrSchedule: {schedule: Literal["constant", "linear"], initial: (0, 1], final: [0, initial]}; constant: final == initial
#   MAX_SELECTION_SEEDS = 250; MAX_SELECTION_EVALS = 40; MAX_TRAIN_META_BYTES = 64 KiB (§2.2, §2.3)
def lr_fn(s: LrSchedule) -> Callable[[float], float]   # linear: p -> final + (initial - final) * max(p, 0.0); a picklable object
def load_train_config(path: Path, *, seed: int | None = None) -> TrainConfig   # = read_train_config(path, seed=seed).config
def read_train_config(path: Path, *, seed: int | None = None) -> LoadedTrainConfig   # (config, file_sha256) from one read
    # reads the file with topple._inputs.open_regular(path, 64 KiB): never blocks on a FIFO, refuses non-regular files
def parse_train_config(data: bytes, *, seed: int | None = None) -> TrainConfig   # safe_load_capped, validate, then --seed
    # replaces the file's seed and the result is validated again: the effective config
def validate_train_config(raw: object) -> TrainConfig   # json.dumps(allow_nan=False), then strict JSON-mode validation
    # (dates, sets, binary, NaN refused; arrays become tuples; int accepted for float); export re-validates
    # train_meta.config with it. Every failure is a ConfigError of at most 8 "<loc>: <message>" parts
def config_sha256(cfg: TrainConfig) -> str   # of the EFFECTIVE config (seed included): sha256(json.dumps(model_dump(mode="json"), sort_keys=True, separators=(",", ":")))
# topple/train/wrappers.py  gymnasium only
TRAIN_SEED_LO = 2**52
class TrainingSeeds(gym.Wrapper):        # reset seed = TRAIN_SEED_LO + int(rng.integers(2**52)) (a Python int); incoming seed ignored
    def __init__(self, env, base_seed: int, env_index: int)  # rng = default_rng(SeedSequence([base_seed, env_index, 0x7709]))
class ActionExcessPenalty(gym.Wrapper):  # PPO only; action_space Box(-box, box, (2,)); env still clips to [-1, 1]
    def __init__(self, env, box: float, weight: float)      # r -= weight * min(mean_i(max(|a_i| - 1, 0)**2), 1); info["excess_penalty"]
def make_train_env(cfg: TrainConfig, env_index: int) -> gym.Env  # env -> Monitor -> penalty -> TrainingSeeds
    # env: the kind's env class (§8.1; balance: BalanceEnv(preset, challenge=load_challenge(id), reward=RewardConfig(...),
    # max_drift_m=...), record=False). No wrapper changes max_steps or truncation: the env ends every episode itself (contract §8, §11).
    # SB3's Monitor is imported inside the function, never at module import
# topple/train/selection.py  ONNX bytes through the real harness; the pure logic the SB3 callbacks glue together
@dataclass(frozen=True)
class SelectionResult:
    step: int; success_rate: float; score: float; median_energy_j: float; mean_saturation: float   # all finite
    scores: tuple[float, ...]          # per seed, seed order (paired comparisons)
    extras: Mapping[str, float]        # balance: fell (fall rate); waypoint: mean_gates, docked, fell, median_t_finish_s
def summarize_episodes(episodes: Sequence[EpisodeResult], *, step: int) -> SelectionResult
    # score and success rate from results.summarize (contract §12); fell = termination in evaluate.FALL_REASONS;
    # a missing episode counts 0 energy, 0 saturation and no fall
def selection_protocol(sel: SelectionConfig) -> EvalProtocol   # dataclasses.replace(EvalProtocol.from_id(protocol), seeds=...)
def evaluate_policy_bytes(ref: PolicyRef, protocol: EvalProtocol, pool: Executor, *, step: int = 0
                          ) -> tuple[SelectionResult, NDArray[np.float32]]  # + 5 on-policy rows per seed with T ≥ 1 (§2.4)
def onpolicy_seed_count(onpolicy: NDArray[np.float32]) -> int   # n_onpolicy_seeds = rows / ONPOLICY_ROWS
@dataclass(frozen=True)
class GateBaseline: rule: Literal["pid", "waypoint", "none"]; reference: SelectionResult | None; anchor: SelectionResult | None
def measure_gate_baseline(rule, protocol: EvalProtocol, pool: Executor, *, pid_submission: Path) -> GateBaseline  # waypoint: 2b
def is_eligible(r: SelectionResult, base: GateBaseline, sel: SelectionConfig, e_ref_j: float) -> bool  # §2.4
def rank_key(r: SelectionResult, kind: Literal["balance", "waypoint"]) -> tuple[float, ...]       # §2.4
@dataclass(frozen=True)
class HistoryRow: result: SelectionResult; eligible: bool; onnx_sha256: str   # required; to_json() is a §2.3 row
    std_mean, frac_mean_outside_1, excess_penalty_mean, cpu_s, wall_s: float | None = None
def choose_best(history, kind) -> HistoryRow | None; def best_json(row, rule) -> dict; def write_best_json(path, row, rule)
def run_flags(history, *, health: HealthConfig | None, e_ref_j: float) -> tuple[str, ...]   # the warning flags, FLAGS order
FLAGS; ABORT_FLAGS = ("rss_abort", "cpu_abort", "wall_clock_abort"); NOT_SHIPPABLE = {*ABORT_FLAGS, "nondeterministic"}
class EvalTrigger: every_steps; total_timesteps: int | None; last_label   # due(n), final_due(n), resumed(every, n, total)
def final_timesteps(cfg: TrainConfig) -> int; def eval_labels(cfg: TrainConfig) -> tuple[int, ...]   # §2.4
def random_rows(norm: ObsNorm) -> NDArray[np.float32]; def edge_rows(norm: ObsNorm) -> NDArray[np.float32]   # §2.4 (b), (c)
def parity_obs(onpolicy, norm) -> tuple[NDArray[np.float32], NDArray[np.uint8]]   # (a), (b), (c) and `set` labels 0, 1, 2
# topple/export.py  (§3.3) the snapshot format's one writer and one reader
RULE_OF = {"tanh": "clip(mean)", "relu": "tanh(mean)"}
def build_onnx(layers: tuple[tuple[f32[out, in], f32[out]], ...], norm: ObsNorm, *, activation, challenge: Challenge) -> bytes
    # §3.2 with the challenge's metadata and head bounds: the bytes the selection eval scores and export rebuilds
def snapshot_files(cfg, step: int, layers, norm: ObsNorm, *, count: float, onpolicy, action_ref, onnx_sha256: str) -> dict[str, bytes]
    # actor.npz, normalize.json, parity_ref.npz (rows from parity_obs), meta.json; byte-deterministic (npz members dated
    # 1980-01-01); ValueError for an array of the wrong dtype or shape (export's own schemas)
def write_snapshot(ckpt_dir: Path, step: int, files: Mapping[str, bytes], *, more: Callable[[Path], None] | None = None) -> Path
    # §2.4's immutable step_<N>/ via .step_<N>.tmp/ (a stale one removed first); FileExistsError if it exists
def read_run_json(path: Path, cap: int = 64 * 1024) -> object   # §3.3 step 2's strict reader (also topple train's)
def coverage_counts(action_ref: NDArray[np.float64]) -> tuple[int, int]   # (n_unsaturated, n_saturated), §3.3 step 4
class ExportError(Exception); class ExportWarning(UserWarning)
# --- torch (train extra; CLI imports lazily) ---
def train(cfg: TrainConfig, out: Path, *, config_file_sha256: str) -> Path      # learn.py; the CLI reads the file once
def snapshot(model, vecnorm, step: int, ckpt_root: Path, onnx_bytes: bytes, onpolicy: NDArray[np.float32]) -> Path
    # snapshot_files (+ PPO's critic.npz, dist.npz), written by write_snapshot; more(tmp) writes model.zip, vecnormalize.pkl
class SelectionCallback(BaseCallback); class HealthCallback(BaseCallback)        # callbacks.py
def policy_stats(model, vecnorm, onpolicy: NDArray[np.float32]) -> PolicyStats     # callbacks.py: std_mean, frac_mean_outside_1 (§2.5)
def resume_from(run_dir: Path) -> tuple[Model, VecNormalize]                      # learn.py; refuses files not owned by os.getuid()
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
start_method: forkserver    # SubprocVecEnv start method, pinned and recorded (determinism key)
env:
  challenge: balance-v1
  preset: standard          # from step 0; no curriculum
  max_drift_m: 2.0          # training-only termination (reason "drift")
  reward: {tilt: 0.30, rate: 0.10, drift: 0.25, yaw_rate: 0.05, effort: 0.10, smooth: 0.20}
  action_excess: {enabled: true, box: 2.0, weight: 0.10}
normalize: {obs: true, reward: false, clip_obs: 10.0, epsilon: 1.0e-8}
policy: {net_arch: {pi: [64, 64], vf: [64, 64]}, activation: tanh, log_std_init: 0.0}
ppo: {n_steps: 2048, batch_size: 2048, n_epochs: 10, gamma: 0.99, gae_lambda: 0.95,
      learning_rate: {schedule: linear, initial: 3.0e-4, final: 0.0},   # clamped: progress reaches -0.0027
      clip_range: 0.2, ent_coef: 0.0, vf_coef: 0.5, max_grad_norm: 0.5, target_kl: null}
selection: {protocol: balance-v1/standard, seeds: {first: 2000, last: 2199}, every_steps: 250_000,
            workers: 4, gate: pid, max_saturation: 0.08}   # 19 periodic + final = 20 evals
limits: {max_wall_clock_s: 2400, max_cpu_s: 4500, max_tree_rss_mib: 8192}
health: {min_std: 0.03, min_std_before_step: 2_000_000, regression_after_step: 2_000_000, regression_drop: 0.10}
```

`configs/train/sac_balance.yaml` (comparison only; written in full):

```yaml
algo: sac
seed: 0
total_timesteps: 1_000_000  # ~500 s train-only at the measured 1,949-2,010 steps/s
n_envs: 8
vec_env: subproc
start_method: forkserver
env:
  challenge: balance-v1
  preset: standard
  max_drift_m: 2.0
  reward: {tilt: 0.30, rate: 0.10, drift: 0.25, yaw_rate: 0.05, effort: 0.10, smooth: 0.20}
  action_excess: {enabled: false}   # no other keys allowed; tanh actions never exceed +-1
normalize: {obs: true, reward: false, clip_obs: 10.0, epsilon: 1.0e-8}
policy: {net_arch: {pi: [64, 64], qf: [256, 256]}, activation: relu}   # state-dependent log_std head (no SDE)
sac: {buffer_size: 1_000_000, learning_starts: 10_000, batch_size: 256, gamma: 0.99,
      learning_rate: {schedule: constant, initial: 3.0e-4, final: 3.0e-4},
      tau: 0.005, train_freq: 1, gradient_steps: 2, ent_coef: auto}   # UTD 0.25 (2 grad steps per 8 env steps)
selection: {protocol: balance-v1/standard, seeds: {first: 2000, last: 2199}, every_steps: 50_000,
            workers: 4, gate: pid, max_saturation: 0.08}   # 19 periodic + final = 20 evals
limits: {max_wall_clock_s: 2400, max_cpu_s: 6000, max_tree_rss_mib: 8192}
# no health block: health is PPO-only (forbidden for SAC)
```

**Every shipped YAML is a complete file; there is no inheritance or merge.** The derived files
below are written out in full; `test_train_config` asserts that `model_dump(load(derived))`
differs from `model_dump(load(base))` in exactly the listed key paths:
- `smoke_ppo.yaml` / `smoke_sac.yaml` (train CI job) vs `ppo_balance` / `sac_balance`:
  `total_timesteps: 2048`, `n_envs: 2`, `vec_env: dummy`, `selection.seeds: {first: 2000, last:
  2001}`, `selection.every_steps: 1024`, `selection.workers: 2`, `selection.gate: none`,
  `limits: {max_wall_clock_s: 300, max_cpu_s: 600, max_tree_rss_mib: 4096}`; PPO also
  `ppo.n_steps: 512`, `ppo.batch_size: 256`; SAC also `sac.learning_starts: 256`.
- `ablate_nopenalty.yaml` vs `ppo_balance`: `seed: 1`, `total_timesteps: 1_000_000`,
  `env.action_excess: {enabled: false}` (about 80 s; source of the §5.2 item 6 "x"). Its LR
  schedule runs over 1M steps; the comparison point is the 1M eval of both runs.
- `ppo_waypoint.yaml`, `ppo_waypoint_ft.yaml`: §8.8.

Rules:
- `algo ∈ {ppo, sac}`; the other algorithm's block is forbidden, `health` is forbidden for SAC.
  There is no `checkpoint_every`: snapshots happen exactly at selection evals.
- **Validation** (`validate_train_config`; every failure a `ConfigError`, exit 2). Types are strict
  (no coercion: `8` is not `"8"`, `true` is not `1`). `seed` in [0, 2^32) (SB3 seeds numpy's
  legacy RNG); `total_timesteps` 1..10^10; `n_envs` 1..16; `vec_env` ∈ {subproc, dummy};
  `start_method` ∈ {forkserver, spawn, fork}; widths 1..256; reward weights 0..1 (sum ≤ 1, as
  `RewardConfig`), all six written out; `action_excess.box` 1..10, `weight` 0..1, and SAC takes
  only `{enabled: false}`; `max_drift_m` (0, 100]; `normalize.obs` MUST be true (the exported graph
  always normalizes), `clip_obs` (0, 10^6], `epsilon` (0, 1]; `ppo.batch_size ≤ n_steps · n_envs`;
  `sac.buffer_size` ≤ 10^7; `env.challenge` MUST exist and offer `env.preset`;
  `selection.protocol` MUST be the canonical `<challenge>/<preset>` id (no alias) of
  `env.challenge`; `selection.seeds` within [0, 2^52), at most `MAX_SELECTION_SEEDS` = 250;
  `total_timesteps // selection.every_steps ≤ MAX_SELECTION_EVALS` = 40; `selection.workers`
  1..64 (pools clamp to 4, §1); `selection.max_saturation` is optional but required by gate
  `pid` (2b's `ppo_waypoint` omits it); `health` is optional for PPO. The selection-seed and eval
  bounds keep `train_meta.json` under 64 KiB (§2.3).
- **Reward.** `RewardConfig` unchanged. The u² effort term MUST NOT be replaced by a battery-energy
  cost (measured 64.9 / 0.505, 82% saturated); energy terms may only be added on top.
- **log_std.** PPO `log_std_init` 0.0 (−0.5 and −1.0 measured worse).
- **Learning rate.** `final ≤ initial`; a constant schedule has `final == initial`. PPO uses the
  linear schedule: `lr(p) = 3e-4·max(p, 0)` with SB3's `progress_remaining` p. The clamp is
  required: p reaches 1 − 5,013,504/5,000,000 = −0.0027 on the last update (lr −8e-7 would be
  gradient ascent). Constant LR collapsed 1 of 3 pilot seeds
  after 2.3M steps (§0). A fine-tune restarts the schedule at 3e-4.
- **SAC.** UTD 0.25 is the best of the three measured settings for this budget (§0); it is a
  shared-machine choice, not SAC's default (UTD 1).
- **Curriculum.** None. A future study gets a training-only `push_scale`, never `ideal`.

### 2.3 Seeding and metadata

- `PPO/SAC(seed=cfg.seed)`; env i wrapped in `TrainingSeeds(cfg.seed, i)`.
- `train_meta.json` stores only run-relative paths:
  - **identity:** `config_sha256` (effective config), `config_file_sha256` (raw YAML bytes, from
    the CLI's single read, `read_train_config`, passed as `train(..., config_file_sha256=)`),
    `config` (effective, `seed` included), `git_commit`, `git_dirty`, `versions` {python, topple,
    torch, stable_baselines3, gymnasium, mujoco, numpy, onnx, onnxruntime, protobuf, psutil},
    `torch_threads` (`torch.get_num_threads()`), `mp_start_method`, `platform`, `cpu_isa`,
    `e_ref_j`, `selection_protocol_hash` (protocol + the selection seeds), `env_algo_version`,
    `bundle_hash`, and for a fine-tune `init_source {run, step, onnx_sha256}`;
  - **resources:** `load_avg_start`, `load_avg_end`, `wall_clock_s`, `train_s`, `selection_s`,
    `cpu_s`, `env_steps_per_s`, `peak_tree_rss_mib`, `peak_tree_rss_at_step`, `peak_procs`;
  - **selection:** `gate_baseline` (`GateBaseline.to_json()`: `rule`, `reference`, `anchor`, each
    a result with its per-seed `scores` and fall rate `extras.fell`), `selection_history` (rows
    `HistoryRow.to_json()`: step, all SelectionResult fields except `scores`, eligible, std_mean,
    frac_mean_outside_1, excess_penalty_mean, cpu_s, wall_s, and `onnx_sha256`, which every row
    MUST carry: export binds it, §3.3 step 3), `best`, `flags` (a list of strings in `FLAGS`
    order, §2.5), `num_timesteps`. Per-seed `scores` of every eval go to `selection_scores.npz`
    (float64 `[n_evals, n_seeds]`, `steps` int64), read by the 2b budget rule and the learning
    curves.
- `train_meta.json` MUST be at most 64 KiB (`MAX_TRAIN_META_BYTES`, its readers' cap: §3.3 step 2,
  and `topple train`'s exit-4 and `--retry` checks). The §2.2 bounds (≤ 250 selection seeds, ≤ 40
  periodic evals) keep it under when written with `indent=2` (a test builds the worst case, the
  waypoint gate's two baselines included: ≤ 56 KiB).
- `phase2_gate.py` recomputes the selection protocol's hash (`balance-v1/standard` with seeds
  2000..2199, built with `dataclasses.replace` from `EvalProtocol.from_id`, as `topple run` builds
  its one-seed protocol) and the current `ENV_ALGO_VERSION` and fails if **any** reported run (PPO,
  SAC, ablation) differs, or if the shipped run has `git_dirty: true`. Known answers at v5.3 (they
  move only with a reviewed re-pin, contract §13; these are after the E_ref 13.1 recalibration):
  selection `f30ef19048989c9b6edf34e5711b5c116ecd386568a5a1e6673360fc4fcf6287`,
  calibration 0..199 `9d3630d321a70821c754a9c64ae121edb8e6847f28a9862f6b4139c5bde59d5e`, bundle
  `89a62007e2f56e61b3a87f3e30034dafc5e6bde4d6c675c07cd79c8d0bbaedd0`, `ENV_ALGO_VERSION` 1.
- **Determinism per (config, seed, platform, start_method) is required and tested** (CLAUDE.md:
  seeded runs are deterministic per platform). `test_train_smoke` runs each smoke config twice
  with the same seed and asserts identical sha256 of every `actor.npz` / `critic.npz`, identical
  `onnx_sha256` per label and identical `selection_history`. Retries must reproduce (§1).

### 2.4 Snapshots and selection

**Trigger** (exactly one rule per algorithm; labels are unique and strictly increasing):
- **PPO:** in `_on_rollout_start`, when `num_timesteps // every_steps` exceeds its last value.
  The policy has just been updated, and the label `N = num_timesteps` equals the env steps it
  trained on. For 5M: evals at 262,144, 507,904, …, 4,751,360 (19), then the final.
- **SAC:** in `_on_step`, when `num_timesteps // every_steps` increases **and** `num_timesteps <
  total_timesteps`. For 1.0M: 50,000 … 950,000 (19), then the final at 1,000,000.
- **Final:** `_on_training_end` evaluates at `model.num_timesteps` unless that label was already
  evaluated (`EvalTrigger.final_due`: a hard limit seen in the callback call that ran a periodic
  eval, or a resume point; the eval there is then the final one); otherwise no periodic label
  equals it. `snapshot()` raises `FileExistsError` if `ckpt/step_<N>/` exists, and
  `selection_history` steps are asserted unique.
- **Code.** `EvalTrigger(every_steps, None)` (PPO, `_on_rollout_start`) or `EvalTrigger(every_steps,
  total_timesteps)` (SAC, `_on_step`); a resumed run uses `EvalTrigger.resumed(every_steps,
  num_timesteps, total_timesteps)`, due at the next label, never at the resume point.
  `eval_labels(cfg)` computes the labels arithmetically; the last is `final_timesteps(cfg)`, SB3's
  final `num_timesteps` (whole rollouts of `n_steps · n_envs` for PPO, whole collections of
  `train_freq · n_envs` for SAC): 5,013,504 for `ppo_balance`, 1,015,808 for the ablation,
  `total_timesteps` for SAC and the smoke configs.

`snapshot()` writes the **immutable** `ckpt/step_<N>/` via `ckpt/.step_<N>.tmp/` + `os.rename`. Its
torch-free files come from `topple.export.snapshot_files(cfg, step, layers, norm, *, count,
onpolicy, action_ref, onnx_sha256)` and the directory from `topple.export.write_snapshot(ckpt_dir,
step, files, *, more=None)` (a stale `.step_<N>.tmp/` is removed first), where `files` adds PPO's
`critic.npz` and `dist.npz` and `more(tmp)` writes `model.zip` and `vecnormalize.pkl`:

| File | Content | Read by |
|---|---|---|
| `actor.npz` | float32 `[out,in]`, keys `W0,b0,W1,b1,W2,b2` for both algorithms (table in §8.8) | export, surgery |
| `critic.npz` | PPO only: `mlp_extractor.value_net.{0,2}.{weight,bias}`, `value_net.{weight,bias}` | surgery |
| `dist.npz` | PPO only: `log_std` (float32 [2]) | surgery |
| `normalize.json` | `mean`, `var` (float64 via `repr`), `count`, `epsilon`, `clip_obs` | export, surgery |
| `parity_ref.npz` | `obs` f32 [N,D], `action_ref` f64 [N,A], `set` u1 [N], `onnx_sha256` (U64 scalar) | export, CI |
| `meta.json` | `step, algo, action_rule` (`RULE_OF[activation]`), `net_arch` (the actor's hidden widths `[h1, h2]`, `policy.net_arch.pi`), `activation, config_sha256, onnx_sha256, n_onpolicy, n_onpolicy_seeds` | export |
| `model.zip`, `vecnormalize.pkl` | SB3 pickles | **resume only**, in `learn.py` |

All files come from the live model and `VecNormalize` in the same callback call; `obs_rms` is
read, never updated. `onnx_sha256` is the sha256 of the in-memory ONNX bytes the selection eval
scored.

**Parity rows** (float32 `[N,D]`):
- (a) **on-policy**, from the selection eval itself: each worker returns
  `traj.obs[np.linspace(0, T−1, 5).round().astype(int)]` per episode with T ≥ 1 (repeats
  allowed for T < 5), in seed order; an episode with T = 0 (e.g. `invalid_action` at step 0) contributes no rows.
  5 rows per selection seed (1,000 for the shipped configs, 10 for smoke), covering
  200 randomizations and push schedules at no extra simulation; `n_onpolicy_seeds` counts the
  episodes with T ≥ 1 (`onpolicy_seed_count`); the pool task is `submit_episodes_with_rows`
  (§3.5);
- (b) 1,000 random rows `f32(mean + sqrt(var + eps)·U)`, U = `−12 + 24·rng.random((1000, D))`
  with `rng = default_rng(SeedSequence(0))` (contract §9's portable rule, never
  `Generator.uniform`);
- (c) **6·D + 4 edge rows** `f32`: for each feature j, `mean` with `x_j` = μ_j + 10σ_j, μ_j −
  10σ_j, μ_j + 10.001σ_j, μ_j − 10.001σ_j, +1e6, −1e6 in that order (σ = sqrt(var + ε); float64,
  rounded once), then `mean`, zeros, all +1e6, all −1e6 (64 rows for D = 10, 94 for D = 15).

`parity_obs(onpolicy, norm)` gives the rows (a), (b), (c) in that order and their `set` labels
0, 1, 2; export regenerates (b) and (c) from `normalize.json` and requires them bitwise (§3.3
step 4).

**Reference actions** use SB3's own pipeline on the widened rows:
`a_ref = clip(model.predict(vecnorm.normalize_obs(obs.astype(f64)), deterministic=True)[0], -1, 1)`.

**Selection eval:**
1. Build the ONNX in memory with `topple.export.build_onnx` (§3.2) and run
   `evaluate_policy_bytes(PolicyRef("onnx", ...), selection_protocol(cfg.selection), pool,
   step=N)` on the selection seeds.
2. **Gate baseline**, measured once at train start with the same harness and seeds
   (`measure_gate_baseline(gate, protocol, pool, pid_submission=...)`): `pid` →
   `submissions/pid_example.py`; `waypoint` → the creeper anchor and the reference driver (2b,
   §8.8); `none` → every checkpoint eligible.
3. **Eligible** iff (`pid`): `success_rate ≥ PID's` ∧ `median_energy_j ≤ E_ref` ∧
   `mean_saturation ≤ selection.max_saturation` (0.08: margin under G4's 0.10); (`waypoint`): §8.8.
4. **Rank key** (per kind, matching the leaderboard): balance `(round(score, 2),
   success_rate)`; waypoint `(round(score, 2), success_rate, −median_t_finish_s)`. Eligibility
   already enforces success ≥ PID, and score-first was never worse on held-out data in 6 runs.
5. `best.json = {"step": N, "eligible": bool, "rule": "<gate>"}` is rewritten atomically (temp +
   `os.replace`) whenever the best changes: the top eligible checkpoint by `rank_key`, or, while
   none is eligible, the top checkpoint with `eligible: false`; a tie goes to the earlier step
   (`choose_best`).
6. Train exits 0 in both cases (4 on a hard limit). Only shipping requires `eligible: true`.

Selection numbers only choose checkpoints. Reported numbers come only from fresh evaluations of
exported artifacts (§4–§5).

### 2.5 Logging, health, runs

- **Policy stats** (`policy_stats`, computed in the selection callback for every eval and stored
  in `selection_history`; inputs are the eval's on-policy parity rows (a), deterministic
  selection episodes, normalized with `vecnorm.normalize_obs`):
  - `std_mean`: PPO `exp(policy.log_std).mean()` (state-independent); SAC the mean over rows of
    `exp(clamp(actor.log_std(actor.latent_pi(z)), −20, 2))`;
  - `frac_mean_outside_1`: the fraction of the 2·N entries of the pre-clip policy mean (PPO
    `action_net(latent_pi)`; SAC the pre-tanh `actor.mu`) with |mean| > 1. This is the §5.2
    item 6 quantity for both the ablation and `ppo_s1`.
- **TensorBoard:** SB3 defaults plus `topple/selection/{success,score,median_energy,saturation}`,
  `topple/policy/{std_mean,frac_mean_outside_1}` (per selection eval, as above),
  `topple/policy/frac_rollout_mean_outside_1` (PPO, any-dim, from the rollout buffer in
  `_on_rollout_end`; never in `_on_rollout_start`, where the buffer is already reset) and
  `topple/train/excess_penalty_mean` per rollout. The Monitor sits inside the penalty wrapper, so
  `rollout/ep_rew_mean` excludes the penalty.
- **Flags:** `std_collapse` (PPO std < 0.03 before 2M; warning), `late_regression` (PPO: a
  selection success after `regression_after_step` that is ≥ `regression_drop` (0.10) below the
  running best; warning, reported), `high_energy` (any eval above E_ref), `no_eligible`,
  `rss_abort`, `cpu_abort`, `wall_clock_abort`, `nondeterministic` (§1). `train_meta.flags` lists
  them in `FLAGS` order, exactly this order (a tuple, never a set's hash-seed order); `run_flags`
  computes the first four (the health ones only with a `health` block), training sets the rest.
- **Runs:** `topple train configs/train/{ppo,sac}_balance.yaml --seed S --out runs/{algo}_s{S}`,
  seeds 0, 1, 2, one at a time, plus `ablate_nopenalty.yaml --out runs/ablate_nopenalty`.
  `runs/` is gitignored.
- **Shipping.** Candidates: PPO runs with `best.eligible` and no abort flag, ordered by their
  best's `rank_key` on the selection seeds (never on 0..199). `phase2_results.py` exports the
  first, evaluates it fresh and moves to the next if it fails **any** of
  `SHIP_GATES = ("G1", "G2", "G2b", "G3", "G4", "G5", "G6")` (a constant in
  `scripts/phase2_common.py`, used by `phase2_results.py`, `phase2_gate.py` and
  `test_phase2_gate`), evaluated on the candidate's own fresh Mac evals (same lock and platform as
  `.venv-eval`, so the numbers are identical). The ubuntu half of G2–G4 (§4) is checked after
  push; if it fails there, the next candidate is tried and the attempt is reported. Every attempt
  is reported with where its failures were measured (`failed_on`: `"local"`, `"ubuntu"` (G2–G4
  only) or `null` when nothing failed); G7 recomputes the local ones (§4). No candidate passing →
  2a fails.

## 3. Export and the `topple.policy/1` format

### 3.1 Submission layout (new contract §17; §8 "Submission kinds")

A policy submission is **data only**: a real directory (lstat: not a symlink) whose entries are
exactly `policy.onnx`, `policy.json` and optionally `README.md` (≤ 64 KiB, untrusted, never printed
by the CLI), all regular files with `st_nlink == 1`.
- Anything else is `LoadError(f"unexpected entry: {safe(name)!r}")` (contract §8 quotes every
  name a load error shows with `repr`; `safe`, §3.4, runs first), including subdirectories, hidden
  files (`.DS_Store`), FIFOs, sockets and `.py/.pyc/.so/.pkl/.pickle/.zip/.pt/.pth/.joblib/.ckpt/
  .onnx.data`. `policy.json` plus any `.py` is `LoadError("ambiguous submission kind")`.
- As contract §8 rules for `controller.py`: names match exactly as `os.scandir` returns them
  (a directory whose only manifest is `POLICY.JSON` is no policy on any OS: it gets the no-entry
  error), non-UTF-8 names are refused, the listing stops at the
  4th entry (a bounded walk), and a path that cannot be examined is a `LoadError`, never `OSError`.
- **Read once.** Each file is read once (`read_regular_file`); `submission_sha256` (contract §8's
  recipe) is computed from those bytes, and those `policy.onnx` bytes are validated and run.
- The loader checks for `policy.json` before `controller.py`, by exact name: an `lstat`, then a
  listing capped at 101 top-level entries that must contain the name `policy.json` (reaching the
  cap stays a policy); a `policy.json` that cannot be examined does not make a policy (B13).
  Python submissions are unchanged but
  for one case: a Python directory with a top-level `policy.json` asset, which contract §8 allows
  today, becomes ambiguous and is refused; the §8 update (§3.6) states this.

`policy.json` is the normative manifest (onnxruntime-web cannot read `metadata_props`).
**Parsing:** `read_regular_file(cap=16 KiB)`; reject a BOM; a byte scan refusing nesting deeper
than 32 levels (`manifest_too_deep`, before any JSON parser, B13); a rejection-only pre-pass
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
                             n_unsaturated: int; n_saturated: int; max_abs_diff: float; tolerance: float; vacuous: bool
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
def read_manifest(data: bytes) -> PolicyManifest   # every exception class of §3.4 "Error text" -> LoadError(fixed code)
def check_against_challenge(m: PolicyManifest, ch: Challenge) -> list[str]
def check_against_model(m: PolicyManifest, proto: onnx.ModelProto) -> list[str]
    # ir_version, the single opset version, input/output (name, float32, ("batch", dim)) equal the proto's
```

Every other claims string matches `PRINTABLE`.

**Identity.** `policy_sha256 = sha256(policy.onnx)`. The loader requires canonical bytes (§3.4),
so this is also `policy.json` `onnx.sha256`, `LoadedPolicy.policy_sha256`, the R2 key and the
Phase 8 header hash. It identifies **bytes**: canonical form removes all non-semantic content, but
two functionally equal graphs (e.g. an extra `Mul` by 1) can still differ; function-level near-copy
detection is the Phase 3 fingerprint (§10). `submission_sha256` (contract §8) remains the record
of the whole directory.

### 3.2 Graph construction (`topple/policy/build.py`, torch-free)

`torch.onnx` is not used: its default dynamo exporter needs onnxscript (absent from the lock). The
graph is built with `onnx.helper` from `actor.npz` + `normalize.json`.

```python
@dataclass(frozen=True) class ObsNorm: mean: NDArray[f64]; var: NDArray[f64]; epsilon: float; clip: float
@dataclass(frozen=True) class MlpWeights: layers: tuple[tuple[NDArray[f32], NDArray[f32]], ...]; activation: Literal["tanh", "relu"]
def build_policy_onnx(w: MlpWeights, norm: ObsNorm, *, action_rule: str, metadata: Mapping[str, str], action: ActionSpec) -> bytes
def reference_actions(w: MlpWeights, norm: ObsNorm, action_rule: str, obs: NDArray[f32], *, action: ActionSpec) -> NDArray[f64]
def policy_metadata(spec: TaskSpec, action_rule: str) -> dict[str, str]   # the 7 topple.* values (json.dumps, ASCII)
```

`action` is the challenge's `TaskSpec.action` (head bounds, action dim; decision log B1). Only the
pairings (tanh, `clip(mean)`) and (relu, `tanh(mean)`) with exactly two hidden layers are built;
the builder checks `metadata` against what it builds and refuses float32 constants that overflow.

- **Normalization** = SB3 `normalize_obs`: `z = clip((x − μ)/sqrt(σ² + ε), −c, c)`, ε =
  `VecNormalize.epsilon` = 1e-8, c = `clip_obs`. Constants are computed in f64 and rounded once:
  `f32(μ)`, `f32(1/sqrt(σ² + ε))`. The builder warns (stderr) for any feature with
  `sqrt(var + ε) < 1e-6·max(1, |μ|)`.
- **`reference_actions`** mirrors the graph: f32 rows, the same f32 constants widened to f64, f64
  arithmetic after that.
- **PPO (9 nodes):** `Sub(obs, mean) → Mul(inv_std) → Clip(lo, hi) → Gemm(W0,b0,transB=1) → Tanh →
  Gemm(W1,b1,transB=1) → Tanh → Gemm(W2,b2,transB=1) → Clip(act_lo, act_hi)`.
- **SAC (10 nodes):** the same with `Relu` hidden layers and head `Gemm(W2,b2) → Tanh → Clip`.
- **Initializer shapes** (PPO; SAC identical): `mean [D]`, `inv_std [D]`, `lo []`, `hi []`,
  `W0 [H1,D]`, `b0 [H1]`, `W1 [H2,H1]`, `b1 [H2]`, `W2 [A,H2]`, `b2 [A]`, `act_lo []`, `act_hi []`.
  Clip bounds are rank 0 (ONNX Clip requires scalars). `lo = −clip_obs`, `hi = +clip_obs`;
  `act_lo = −1.0`, `act_hi = +1.0` come from the challenge's `TaskSpec.action` bounds, **never** from
  `model.action_space` (Box(−2, 2) under the penalty wrapper). Gemm sets only `transB = 1`;
  attributes equal to their schema default are never written.
- **Names (canonical, §3.4):** input `obs` `["batch", D]`, output `action` `["batch", 2]`, nodes
  `n{i}`, the output of node i `v{i}` (the last node's is `action`), initializers `c{j}` in first-use
  order (node index, then input position). The Phase 8 C path reads initializers by position in
  the §3.2 sequence.
- **Model:** opset 17 with IR `CANONICAL_IR[17] = 8` (§3.4), one opset import, weights inside,
  `SerializeToString(deterministic=True)`; `producer_version` and `doc_string` are never set
  (proto2 presence would make them "set", even when empty). The build output is already
  canonical. Two builds from the same inputs are byte-identical (measured).
- **`metadata_props`** (exactly these 7, no others): `topple.format`, `topple.challenge`,
  `topple.obs_schema`, `topple.obs_features`, `topple.action_spec`, `topple.control_dt`,
  `topple.action_rule`; each value is `json.dumps(field)` (ASCII, `ensure_ascii=True`), so it is
  fully derived from the manifest. Load compares `json.loads(value)` with
  `json.loads(json.dumps(manifest.model_dump(mode="json")[field]))` (both sides normalized).

### 3.3 Export and pack (torch-free)

`topple export RUN_DIR --out DIR [--step N] [--report-only] [--test-fixture NAME] [--replace]`
(`export_policy(run_dir, out_dir, *, step=None, report_only=False, replace=False,
test_fixture=None, fixture_dir=Path("tests/data")) -> PolicyManifest`; every refusal is one
`ExportError` line, exit 2):
1. **Resolve.** `--test-fixture` is checked first (step 5). `step` = `--step` or the integer
   `best.json.step` (0 ≤ step < 2^53; other keys ignored). Path `run_dir/"ckpt"/f"step_{step}"`
   MUST be a directory by `lstat` (not a symlink, so a looping one is refused as a symlink) and
   MUST resolve under `run_dir.resolve()`; a resolve that fails (a looping `ckpt`: Python 3.12's
   `Path.resolve` raises `RuntimeError`, not `OSError`, on a loop) is one line, never a traceback.
   `train_meta.flags` MUST be a list of strings and `selection_history` MUST hold exactly one row
   for the step, with a boolean `eligible` and a string `onnx_sha256`. Without `--report-only`,
   that row MUST have `eligible: true`, `train_meta.flags` MUST hold no abort flag and no
   `nondeterministic`, and the output MUST NOT resolve under `<cwd>/runs/`, or under the `runs/`
   the run directory lives in (when its parent is named `runs`), nor inside the run directory; a
   directory named `runs` elsewhere (e.g. an ancestor of the checkout) does not count. With it,
   the output MUST be `run_dir/report_policy/` (its resolved parent the run's) and `./runs` is
   never resolved; `report_policy/` is replaced if present (report-only implies `--replace` for
   that path only, so `phase2_results.py` is idempotent). `claims.selection_eligible` records the
   flag.
2. **Read** only `best.json`, `train_meta.json` (json ≤ 64 KiB), `meta.json`, `normalize.json`,
   `actor.npz`, `parity_ref.npz` (npz ≤ 8 MiB), each with `read_regular_file`. npz bytes go through
   the Phase 1 trajectory reader's parts (contract §14), moved from `topple/traj.py` into
   `topple/_npz.py` with no change to `load_npz`: `topple._zip.Archive` (numpy's canonical layout
   only, exactly the expected member names, directory ≤ 16 KiB, each member ≤ 8 MiB through
   `max_member_bytes`; `Archive` has no total cap, so `_npz.py` also requires
   `sum(m.size for m in archive.members.values()) ≤ 8 MiB` before any member is read) and the
   hand-parsed npy header (never evaluated; header ≤ 4096 bytes; the exact expected dtype, C order;
   the shape checked against the expected one and needing exactly the member's bytes before any
   data is read, so a header declaring 2^42 elements allocates nothing). dtypes `<f4`, `<f8`,
   `|u1`, `<U64` (`onnx_sha256`) only; every member's exact shape, `parity_ref` with N =
   `n_onpolicy + 1000 + 6·D + 4`. JSON (`read_run_json`) rejects a BOM, NaN/Infinity, duplicate
   keys, integers over 20 digits, floats beyond float range and nesting deeper than 32 (a byte
   scan before the parser). **Bindings:** `train_meta.config` MUST re-validate
   (`validate_train_config`) and its `config_sha256` MUST equal `train_meta.config_sha256` and
   `meta.json.config_sha256`; `meta.json` MUST have the resolved `step`, (`algo`, `activation`,
   `net_arch`) equal to the config's (`algo`, `policy.activation`, `policy.net_arch.pi`) and
   `action_rule == RULE_OF[activation]`; `normalize.json` `epsilon` and `clip_obs` MUST equal
   the config's `normalize` block, `mean` and `var` have D entries.
3. **Build and bind.** Build (`build_onnx`, §3.2); `sha256(bytes)` MUST equal
   `meta.json.onnx_sha256`, `parity_ref.npz` `onnx_sha256` and the step's `selection_history` row
   `onnx_sha256` (so "eligible" names these exact bytes). **Constant binding**, bitwise and
   independent of parity (a wrong ε of 1e-6 differs from SB3 by only 2.9e-6 for SAC; measured on
   synthetic runs, about 1.1e-6 with small heads): initializers by §3.2 position MUST
   equal `f32(mean)`, `f32(1/sqrt(var + ε))`, `∓f32(clip_obs)`, the `actor.npz` arrays, and
   `∓1.0` from the challenge's action spec (`TaskSpec.action.low/high`, §3.5).
   `validate_policy_bytes` MUST accept it; `smoke_test` MUST pass.
4. **Parity (normative).** The rows MUST be the §2.4 sets: `set` labels 0, 1, 2 in that order,
   (b) and (c) bitwise equal to `random_rows` and `edge_rows` of `normalize.json`, `obs` and
   `action_ref` finite. Every `parity_ref` row at batch 1 through `make_session`. `max |a_onnx
   − a_ref| ≤ 1e-5` always MUST hold (a NaN difference on any row fails; a later finite row never
   replaces it). Coverage (`coverage_counts`): ≥ 300 rows with `max_i |a_ref_i| < 0.99`
   (`n_unsaturated`), ≥ 32 rows with `max_i |a_ref_i| ≥ 0.999` (`n_saturated`; exercises the head
   Clip / tanh), `n_onpolicy == 5 · n_onpolicy_seeds`, and `1 ≤ n_onpolicy_seeds ≤` the number of
   selection seeds in `train_meta.config`. Without `--report-only` coverage MUST
   hold; with it, a coverage miss sets `export_parity.vacuous = true` instead of failing, so every
   G7 row exists.
5. **Write.** The two files MUST load through `load_policy_files` (what loading the directory
   runs) before anything is written. `--out` MUST NOT exist, unless `--replace`: the new directory
   is written as a temp sibling (`tempfile.mkdtemp` beside it), the old one renamed aside, the new
   one renamed in, the old one removed; a failure up to the swap puts the old one back, and
   failing to remove the old one after it is an `ExportWarning` (printed `warning: ...`), not an
   error. `--replace` replaces only a symlink (the link, never its target) or a directory holding
   nothing but `policy.onnx`, `policy.json`, `README.md` (no subdirectory); anything else is
   refused. A relative `--out` through a symlinked directory exits 2 first (contract §14; §5.3).
   Claims come from `meta.json` and `train_meta.json` through a whitelist: `name` null; `algo`, `activation`,
   `net_arch` from `meta.json`; `total_timesteps`, `train_seed` from the config; `git_commit` 40
   lowercase hex or null (anything else refused), `git_dirty` a bool or null, a missing one null;
   `versions` keeps only string values matching the claims patterns, at most 16 sorted by key.
   `--test-fixture NAME` (`^[a-z0-9_]{1,32}$`; refused with `--report-only`) writes
   `tests/data/NAME_parity.npz` (copy), `tests/data/NAME_parity.bin` and
   `tests/data/NAME_parity.json` for C and JS consumers (`fixture_dir`, checked like `--out`).
   Existing fixture files need `--replace`; a fixture target that is a directory is refused before
   anything is written; the files are written to temp names beside their targets before the
   `--out` swap and renamed over them after it (errors name the target; a rename failure after the
   swap says `--out` was written). **`.bin` layout v1**, little-endian:
   `b"TPAR"`, `u32 version = 1`, `u32 N`, `u32 D`, `u32 A`, `f32 obs[N·D]`, `f32 a_ref[N·A]`,
   `u8 set[N]` (0 on-policy, 1 random, 2 edge), zero-padded to a multiple of 4 bytes (f32 a_ref
   adds about 3e-8). The JSON sidecar: `{"onnx_sha256", "n", "d", "a", "tolerance": 1e-5,
   "n_unsaturated", "n_saturated", "policy_sha256"}`. C and JS readers check magic and version
   and assert D and A against `policy.json`.

Export never imports torch or calls `PPO/SAC/VecNormalize.load`, `pickle`, `cloudpickle`,
`torch.load`. Shared run directories are untrusted.

`topple policy pack MODEL.onnx --challenge ID --out DIR` (third-party models):
1. check 1 (size, parse);
2. `canonicalize` (§3.4), which in pack mode also: drops every field outside the allow-list
   (doc strings, `value_info`, all metadata, producer, …) instead of rejecting it, removes
   `Identity` nodes by rewiring, renames the single input/output to `obs`/`action`, turns a
   static batch dim of 1 into `dim_param "batch"`, and sets IR to `CANONICAL_IR[opset]`; it still
   rejects external data first;
3. set exactly the 7 `topple.*` metadata keys from the challenge, replacing any the input
   carried; checks 1–9 on the result;
   `smoke_test`;
4. write `policy.onnx` + `policy.json` with `action_rule` from the structure match (§3.4) and
   `claims.export_parity = None`.

**Third-party recipes.** Primary (dynamo; needs `pip install onnxscript`, which is not in
Topple's lock): `torch.onnx.export(net, (x,), f, dynamo=True, external_data=False, input_names=["obs"],
output_names=["action"], dynamic_shapes=({0: torch.export.Dim("batch")},))` (measured with
onnxscript 0.7.2: IR 10, opset 20, `Sub Mul Clip Gemm Tanh Gemm Tanh Gemm Clip`, 20 `value_info`,
rank-0 initializers). Legacy (`dynamo=False`, deprecated by torch 2.14.1 with "will be removed"):
IR 9, opset 20, 5 inline rank-0 Constants. Both then `topple policy pack`, which folds Constants
and sets IR 9 (`CANONICAL_IR[20]`). Two torch-free fixtures generated once by
`scripts/make_thirdparty_fixtures.py` (run with `uv run --extra train --with onnxscript`) are
committed: `tests/data/thirdparty_legacy.onnx` (IR 9 / opset 20, interleaved Constants) and
`tests/data/thirdparty_dynamo.onnx` (IR 10 / opset 20, `value_info`, rank-0 initializers); the
default job packs both.

### 3.4 Acceptance policy (`topple/policy/validate.py`, torch-free)

```python
MAX_ONNX_BYTES = 1 << 20; MAX_ELEMENTS = 131_072; MAX_COST = 1 << 17; MAX_NODE_INPUTS = 192
CANONICAL_IR = {17: 8, 18: 8, 19: 9, 20: 9, 21: 10}   # IR <= 10 keeps ORT-web >= 1.20 compatible
OP_WEIGHT = {**dict.fromkeys(("Add", "Sub", "Mul", "Div", "Neg", "Relu", "LeakyRelu", "Clip", "Min", "Max",
             "Sqrt", "Reciprocal", "Softsign"), 1), "Tanh": 2, "Sigmoid": 2, "Elu": 4, "Softplus": 48}
def read_regular_file(path: Path, cap: int) -> bytes
@dataclass(frozen=True)
class ValidatedPolicy: canonical: bytes; policy_sha256: str; cost: int; obs_dim: int; action_dim: int
    # constructible only inside validate.py (module-private sentinel checked in __post_init__);
    # keyword-only and final (no subclass); make_session requires the exact type
def validate_policy_bytes(data: bytes, obs_dim: int, action_dim: int) -> ValidatedPolicy | list[str]   # data: exact bytes
def canonicalize(proto: onnx.ModelProto, *, pack: bool = False) -> bytes
def static_cost(proto: onnx.ModelProto) -> int
def smoke_test(v: ValidatedPolicy) -> list[str]       # = smoke_run(v).errors
def smoke_run(v: ValidatedPolicy) -> SmokeReport       # SmokeReport(errors: tuple[str, ...], latency_p99_us: float | None)
def safe(s: str | bytes) -> str    # bytes -> decode("utf-8", "replace"); re.sub(r"[^A-Za-z0-9_.:/-]", "?", s)[:64]
def structure_match(proto: onnx.ModelProto) -> Literal["clip(mean)", "tanh(mean)", "other"]
```

Every Python-side pattern check uses `re.fullmatch` with an unanchored pattern (`re.match` with
`$` accepts a trailing `"\n"`); patterns quoted below with `^…$` mean full match.

**`read_regular_file`** (in `topple/_inputs.py`, beside Phase 1's `open_regular`, which follows
symlinks and serves `results.json` and trajectories, contract §14–§15; this stricter sibling is for
files inside submission and run directories): `st = lstat(path)` (not a symlink, `S_ISREG`,
`st_nlink == 1`, size ≤ cap); `fd = os.open(path, os.O_RDONLY | flag("O_NOFOLLOW") |
flag("O_NONBLOCK") | flag("O_NOCTTY") | flag("O_CLOEXEC") | flag("O_BINARY"))` with `flag(n) =
getattr(os, n, 0)`, as the merged `open_regular` and `loader._read_capped` do (win32, one of the
lock's environments, has only `O_BINARY` of these; round 1 deploy #8). `O_NONBLOCK`: a FIFO swapped
in after the lstat cannot block the open. `fstat(fd)` MUST be `S_ISREG` with (st_dev, st_ino)
equal to the lstat, which carries the no-symlink guarantee where `O_NOFOLLOW` is missing; read at
most cap+1 bytes. `OSError` becomes `LoadError` naming the file (contract §8).

**Checks, in order, short-circuiting:** the first check that produces any message returns. No onnx
C++ checker, shape inference, `numpy_helper`, `onnx.external_data_helper` or ORT call ever runs on
bytes that failed checks 1–7.
1. **Size and parse.** ≤ 1 MiB (a [64,64] export is about 21 KB); `onnx.ModelProto.FromString`;
   `google.protobuf.message.DecodeError` (incl. nesting deeper than about 35 levels) →
   `onnx_parse_failed`. (protobuf 7.36's limit is 100 message levels; about 34 nested subgraphs
   reach it, B4.) Never `onnx.load(path)`, `check_model(path)` or `version_converter`.
2. **Field allow-list.** Walk the proto recursively with `ListFields()` (repeated fields via
   `fd.is_repeated`; protobuf 7 has no `fd.label`). On every visited message reject
   `len(google.protobuf.unknown_fields.UnknownFieldSet(msg)) > 0` (`msg.UnknownFields()` raises
   `NotImplementedError` on the pinned upb protobuf 7.36). Every string and repeated-string value
   MUST be a `str` that survives `.encode("utf-8")` (upb returns `bytes` for invalid UTF-8 in
   proto2 strings) → else `invalid_utf8`, before any regex. Allowed set fields: ModelProto
   {ir_version, opset_import, producer_name, graph, metadata_props}; OperatorSetId {domain ∈
   {"", "ai.onnx"}, version}; GraphProto {node, name, initializer, input, output}; NodeProto
   {input, output, name, op_type, attribute}; AttributeProto {name, type, f, i, t, floats};
   TensorProto {dims, data_type, raw_data, float_data, name}, with `name` absent inside
   `AttributeProto.t`; ValueInfoProto {name, type}; TypeProto {tensor_type}; Tensor {elem_type,
   shape}; TensorShapeProto {dim}; Dimension {dim_value | dim_param}; StringStringEntry {key,
   value}. This rejects `producer_version`, `doc_string`, `configuration`,
   `quantization_annotation`, `device_configurations`, `overload`, every `denotation`,
   `ref_attr_name`, `metadata_props` outside the model, node `domain`, `external_data` and
   `data_location`. (Observed set fields of a §3.2 build are exactly within this list.)
3. **Header.** One opset import with version 17..21 and `ir_version == CANONICAL_IR[opset]` (IR
   9/10 features are excluded by check 2, so lowering IR in pack is safe; checker and ORT accept
   IR 8 with opsets 18 and 21); no `functions`, `training_info`, `sparse_initializer`.
4. **Strings.** Node, value, initializer, graph and every TensorProto name match
   `^[A-Za-z0-9_.:/-]{1,128}$`; `op_type` matches `^[A-Za-z]{1,32}$` and attribute names
   `^[A-Za-z_]{1,32}$` (so `value_float` and `value_floats` reach check 6, B2); exactly
   the 7 §3.2 `metadata_props` keys (`^topple\.[a-z_]{1,32}$`), each value ASCII and ≤ 1,024
   bytes; `producer_name` ≤ 128 chars.
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
   `float_data`, rank 0..2, each dim 1..4096, exactly prod(dims) finite values (1 for rank 0;
   every torch Constant and ONNX Clip bound is rank 0). Model-wide: ≤ 32
   initializers, ≤ 131,072 elements incl. Constants, ≤ 4,096 elements per Constant, ≤ 64 nodes.
   Values are read with `struct`/`np.frombuffer` on the raw fields only.
8. **Structure and cost.** `check_model(proto, full_check=True)`; `infer_shapes(proto,
   strict_mode=True, check_type=True)`. Every node output except Constant outputs MUST be
   float32, rank 2, dim0 the `obs` batch symbol, dim1 static ≤ 4096. Every initializer is used,
   every node reaches `action`, and a node other than a Constant computes `action`
   (`output_not_computed`, B9). `static_cost = Σ_{Gemm,MatMul} K·N + Σ_{other} OP_WEIGHT[op]·max(1,
   n_inputs − 1)·dim1(output) ≤ 2^17`. First-party PPO ≈ 5,200 weighted (25× headroom). The
   variadic term closes the 245,000-input `Max` bomb (89 ms per act); the weights close compute-
   heavy unary chains: at cost 2^18 unweighted, 52 Softplus nodes at width 4096 took 1,060 µs per
   act (Elu 113, Tanh 60.9, Relu 27.8; first-party 4.6). Under the weighted 2^17 rule the worst
   case measured 16.3–21.1 µs per act on M5, ≤ 8.4 s per 400k acts. ORT's
   `session.set_denormal_as_zero` is never enabled in-process (it would set FTZ/DAZ on the
   calling thread on x86 and change MuJoCo arithmetic).
9. **Canonical.** `canonicalize(proto) == data`, else `"not_canonical: policy.onnx is not
   canonical; run topple policy pack"`.

**Error text** is fixed vocabulary plus `safe(s)`. Every validator message has the form
`code: detail` (a fixed code, then fixed text and `safe` names); `CHECK_OF` maps each code to its
check. An exception inside a check becomes `<code>: raised during check N`; every `Exception` is
caught there, not only the classes below; at most 8 messages are returned per rejection (B3). Pydantic errors render only `'.'.join(safe(str(p))
for p in loc)` and `type` (never `msg`, `input` or `ctx`). onnx/ORT exceptions become fixed codes
(`onnx_parse_failed`, `invalid_utf8`, `onnx_checker_failed`, `shape_inference_failed`,
`ort_load_failed`). `validate_policy_bytes`, `read_manifest` and `load_policy_dir` wrap their
bodies and map `DecodeError`, `ValueError`, `TypeError`, `UnicodeError`, `RecursionError` and
`MemoryError` to fixed codes; they never re-raise raw text (an `UnicodeDecodeError` message
contains the offending bytes); raw text only with `--debug`, locally.

**Canonical form** (`canonicalize`) is constructive: a fresh `ModelProto` built only from
allow-listed parts. It reads tensors from raw fields only (`struct.pack` of `float_data`, copy of
`raw_data`), asserts `data_location == DEFAULT` and empty `external_data` itself, and never calls
`numpy_helper` or `onnx.external_data_helper`. It sets `ir_version = CANONICAL_IR[opset]` and the
single opset (domain `""`), `producer_name = "topple"` (never `producer_version`),
`graph.name = "topple_policy"`, the §3.2 names, batch `dim_param "batch"`, the 7 `topple.*`
metadata keys in key order (pack drops all others; load rejects them at check 4), every tensor
as little-endian `raw_data` with no `name` inside attributes. It **folds every `Constant`** node
(value tensor, `value_float`, `value_floats`) into an initializer `c{j}` in first-use order, and
**drops every attribute equal to its schema default** (Gemm `alpha = 1.0`, `beta = 1.0`,
`transA = 0`, `transB = 0`; also LeakyRelu `alpha = float32(0.01)` and Elu `alpha = 1.0`, B10),
deciding default-ness on the attribute it emits, ordering the rest by name. Outside pack mode it
raises on any field off the allow-list instead of dropping it, and keeps unused initializers and
folded Constants so check 8 still sees them. Pack mode keeps the input's 7 `topple.*` keys and
drops every other metadata key (so `topple policy pack` **replaces** those keys with the
challenge's, never appends); it refuses nodes in a non-default domain, Identity cycles, nodes with
two outputs, undefined or duplicate values, tensors with both payloads, Constants it cannot fold,
invalid UTF-8 it would copy and (slice 1b) a node that repeats an attribute name (B8, B11). Then
`SerializeToString(deterministic=True)`. Node order is kept (ONNX requires topological order).
It is a fixed point; canonical bytes contain no `Constant` node, so a model with one fails check
9 at load ("run topple policy pack"). ORT only ever sees canonical bytes.

**smoke_test:** 32 batch-1 calls (zeros, then `default_rng(1)` normals) through `make_session`;
each output float32 `(1, action_dim)` and finite. Per call: a fresh `ort.RunOptions()`, a
`threading.Timer(1.0)` that sets `terminate = True`, `run`, `Timer.cancel()` in `finally`. A
terminate error is `"smoke call exceeded 1 s"`, anything else `"smoke call failed: ort_run_failed"`
(also `smoke call failed: bad_output` for a wrong dtype or shape, `smoke call failed:
non_finite_output`, and `ort_load_failed` when no session can be made; B6).
ORT checks `terminate` only between kernels, so it is not a bound; the static cost is the
acceptance bound and the Phase 3 process kill the runtime bound. The p99 is recorded as
`latency_p99_us` in `smoke_run`'s `SmokeReport` (never an acceptance criterion). `ONNXPolicy.act` uses no RunOptions.

**Structure match** (`structure_match`, on canonical bytes). A graph matches a §3.2 sequence iff
op types **and** attributes (after default dropping: only Gemm `transB = 1`) **and** the exact
initializer shapes listed in §3.2 (with D, H1, H2, A from the shapes, A = `action_dim`) **and**
each node's exact inputs and outputs (so `Sub(c0, obs)`, mean − obs, does not match) are equal,
every initializer is float32 `raw_data`, the head Clip bounds are bitwise `−1.0` / `+1.0`, and the
normalization Clip bounds are symmetric (`lo == −hi`), finite and `hi > 0` (B7). Then `action_rule` MUST be the sequence's head (`clip(mean)`
or `tanh(mean)`); otherwise it MUST be `"other"`. A dynamo export with a rank-0 normalization
scale is `other`; broadcasting that scale to `[D]` makes it `clip(mean)`. The Phase 8 C path
accepts only the two matching sequences and reads initializers by position.

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
    # property policy_sha256 = validated.policy_sha256; never pickled or copied
def load_policy_dir(path: Path, *, challenge: Challenge) -> LoadedPolicy        # full validation + smoke
def policy_from_bytes(data: bytes, *, challenge: Challenge, expected_sha256: str | None = None) -> LoadedPolicy
# topple/pool.py
@dataclass(frozen=True)
class PolicyRef: kind: Literal["onnx", "python"]; payload: bytes | str; sha256: str; challenge_id: str
def policy_ref(sub: Path, *, challenge: Challenge) -> PolicyRef   # parent: reads + validates ONCE
def submission_ref(sub: Path, *, challenge: Challenge) -> tuple[Submission, PolicyRef]
    # policy_ref with its load: a report's id, sha256 and policy_sha256 come from the read the workers check
def worker_factory(ref: PolicyRef) -> ControllerFactory           # worker: re-checks sha256 before the first episode
def eval_pool(workers: int) -> ProcessPoolExecutor
def submit_episodes(pool: Executor, ref: PolicyRef, protocol: EvalProtocol, indices: Iterable[int], *,
                    pushes: Iterable[Push] | None = None) -> Future[list[EpisodeResult]]
    # refuses a private protocol, another challenge, bad indices or bad pushes in the PARENT, before anything
    # is pickled; indices (≤ 10,000) and pushes (≤ 16) are read once, bounded, into tuples (B16)
def run_episodes(ref: PolicyRef, protocol: EvalProtocol, indices: Iterable[int],
                 pushes: Iterable[Push] | None = None) -> list[EpisodeResult]   # the task; checks again in the worker
ONPOLICY_ROWS = 5
@dataclass(frozen=True)
class EpisodeRows: result: EpisodeResult; obs_rows: NDArray[np.float32]   # the trajectory never leaves the worker
def onpolicy_rows(obs: NDArray[np.float32]) -> NDArray[np.float32]   # §2.4 (a): obs[linspace(0, T-1, 5).round()]; (0, D) for T = 0
def submit_episodes_with_rows(pool: Executor, ref: PolicyRef, protocol: EvalProtocol, indices: Iterable[int]
                              ) -> Future[list[EpisodeRows]]   # a selection eval's task; checked in the parent as above
def run_episodes_with_rows(ref: PolicyRef, protocol: EvalProtocol, indices: Iterable[int]) -> list[EpisodeRows]
```

- **SessionOptions:** `intra_op_num_threads=1`, `inter_op_num_threads=1`, `ORT_SEQUENTIAL`,
  `ORT_DISABLE_ALL`, `log_severity_level=3`, `use_deterministic_compute=True`; config entries
  exactly `session.load_model_format=ONNX`, `session.allow_released_opsets_only=1`,
  `session.strict_shape_type_inference=1`; `providers=["CPUExecutionProvider"]`, asserted via
  `get_providers()`. Bytes only; scrub `ORT_*` from `os.environ` before the first session.
- **`act()`** feeds `obs.to_array()[None]`, checks float32 `(1, A)`, returns Python floats.
  Non-finite **observations are outside the policy domain** (contract §17): `evaluate`'s control
  loop checks `np.isfinite(obs_arr).all()` before every `act()` and raises `RuntimeError`, a
  harness bug like its `max_steps` bound (contract §8; not an `assert`, which `python -O` strips).
  It holds by construction: `sim_unstable` rolls the bad substep back and ends the episode, and a
  non-finite reset raises (contract §9, §11). Runtimes disagree on NaN inputs (ORT 1.31 Python on
  arm64 maps an all-NaN observation to a finite [0.943, 0.892]; ORT-web 1.20/1.30 return NaN), so
  the mapping of non-finite **actions** to `invalid_action` via `validate_action` applies only to
  runtimes that propagate NaN.
- **`load_policy_dir`:** directory rules (§3.1), `read_regular_file` both files (read once),
  manifest parse, `sha256 == manifest.onnx.sha256`, `validate_policy_bytes`,
  `check_against_model`, structure match, metadata vs manifest, `check_against_challenge`,
  `smoke_test`. Dims and action bounds come only from the challenge's `TaskSpec` (`obs_features`,
  `action`; `Challenge.task_spec` also wants the robot geometry, which does not change them, so a
  caller without an env reads them from the kind entry, §8.1). Any failure is one `LoadError`.
- **Pools.** The parent builds a `PolicyRef` once: ONNX → (canonical bytes ≤ 1 MiB,
  `policy_sha256`); Python → (path, `submission_sha256`). A worker calls `policy_from_bytes(...,
  expected_sha256=...)` or `load_submission(path, challenge=..., expected_sha256=...)`, which
  compares the hash of its single read before compiling anything, and raises on mismatch before
  its first episode (files the submission reads later are not covered, B13). `LoadedPolicy`, sessions and factories are never pickled. Results return by
  pickle; this is acceptable locally only (local eval is not a security boundary; §10 (iii)).
  Workers receive public protocols only (Phase 2 never runs a private one; a private
  `EvalProtocol`'s salt would reach every worker, and only its `repr` is redacted, contract §8):
  `submit_episodes` refuses a private protocol in the parent. `pushes` replaces each episode's
  scenario pushes through the trusted `reset(options={"pushes": …})` hook (contract §9; `None`
  keeps them, and `run_episode` then calls `env.reset(seed=seed)` as before); each injected push
  MUST be exactly a `Push` with an int `start_step ≥ 0`, an int `n_substeps ≥ 1` and finite
  `magnitude_n`, `forward_sign` and `lateral_ratio`, sorted by `start_step` without repeats, else
  `ValueError` (B21).
- **Determinism** (contract §17): bit-exact per (os, arch, `cpu_isa`, onnxruntime version);
  across runtimes within 1e-5, never bit-exact, so replays play stored trajectories. No test pins
  an ONNX episode score; tests compare orderings only (PPO > PID).

### 3.6 Loader, results, contract

- **Loader.** `load_submission(path, *, challenge: Challenge | None = None, expected_sha256: str
  | None = None)` (Phase 1 takes `path` only): `challenge` is required for `kind == "onnx"`
  (`TypeError` if missing, raised after the directory is listed and read). CLI `eval`, `run`,
  `compare` pass the protocol's challenge. `Submission.sha256` is §3.1's read-once hash.
  `topple verify --submission` records `policy_sha256` through `loader.submission_hashes`, which
  reads a policy directory under the policy entry rules (B13).
- **Results** (before `topple.results/1` is published; contract §15's strict reader stays as is):
  `Versions` gains `onnxruntime`, `onnx`, `cpu_isa` (`avx512f` / `avx2` / `sse4` from
  `/proc/cpuinfo` on Linux; `arm64-sme` / `arm64-neon` from `sysctl hw.optional` on macOS), each
  `VersionStr | None = None`; `ControllerInfo` gains `policy_sha256: Sha256 | None = None` (JSON
  `null` for a Python submission; `read_results` reads both).
- **Contract.** §8 (submission kinds, with §3.1's ambiguity rule), new §17 (`topple.policy/1` =
  §3.1–§3.5), §15.

## 4. Done-when for 2a (every gate MUST pass)

`scripts/phase2_done.sh` (`set -euo pipefail`; evidence from the **shipped ONNX directory** in a
separate torch-free environment, so the developer's `.venv` is never modified):

```bash
dirty=$(git status --porcelain -- submissions docs/results tests/data configs scripts python)
test -z "$dirty"
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
uv run topple export "runs/$K" --out "$D"
cmp "$D/policy.onnx" submissions/ppo_balance/policy.onnx
head=$(git rev-parse HEAD)
uv run python scripts/phase2_ci_check.py --head "$head" --workflow ci.yml   # G9
uv run python scripts/phase2_gate.py out/cmp_cal.json out/cmp_eval_v1.json out/ppo_eval_v1/results.json \
    docs/results/phase2.json --ci-dir docs/results/phase2/ci_ubuntu \
    --submission submissions/ppo_balance --results-dir docs/results/phase2
```

- **The script.** Line 2 is a file-wide `# shellcheck disable=SC2086` (the block splits `$S`
  into three submissions on purpose; the directive must follow the shebang directly), and a
  preamble `top=$(git rev-parse --show-toplevel)`, `cd "$top"` lets it run from anywhere in the
  checkout. Every line is a simple command, so errexit stops on any failure: no `&&` or `||`
  (bash ignores a failure left of them, so the re-export and its `cmp` are two lines), and
  command output is taken only by an assignment (`x=$(cmd)` fails when `cmd` does; `test -z
  "$(cmd)"` passes when `cmd` fails and prints nothing). `test_phase2_gate` checks the script's
  lines against this block.
- **CI trigger.** `ci.yml` runs on `pull_request` and on `push` to `main` and `phase-*`, so a
  branch without a PR is still checked.
- **`phase2_ci_check.py`** (`--head` a full sha; `--branch` overrides the current branch)
  queries `gh run list --branch <branch> --workflow ci.yml --event push --limit 100 --json
  databaseId,status,conclusion,headSha` and judges the newest **push** run for HEAD (largest
  `databaseId`; a `pull_request` run of the same commit skips the push-only evidence steps, B54). It fails (exit 1) with distinct messages: no run for HEAD, in progress,
  cancelled (concurrency cancel-in-progress), failure. After a completed `success` run with
  `headSha == HEAD` it calls `gh run view <id> --json jobs` and passes only if `python`,
  `python-slow` and `python-train` each completed with `success` (otherwise the failure message
  names the job). It exits 2 when `gh` is missing, fails, runs past its timeout, prints more than
  1 MiB or prints unexpected JSON.
- **Ubuntu evidence.** In CI's `python-slow` job (where `slow` tests run, §7), on a `push` run
  once a policy is shipped (each step's `if:` is `github.event_name == 'push' &&
  hashFiles('submissions/ppo_balance/policy.onnx') != ''`: a pull request runs its own code, so
  its numbers are never evidence), after `test_shipped_policy` (its `results.json` goes under
  `TOPPLE_ARTIFACT_DIR=out/ci`), `topple compare submissions/pid_example.py
  submissions/lqr_example.py submissions/ppo_balance -p balance-v1/standard --seeds 0..199
  --workers 4 --json out/cmp_cal_ci.json` (about 1–2 min on the private repo's 2-vCPU runner,
  where the pool clamps `--workers 4` to 2, §1; 15 s locally) runs. A check step (`test -f
  out/ci/results.json`, `test -f out/cmp_cal_ci.json`, one per line: `upload-artifact`'s
  `if-no-files-found` fires only when no path matches at all) precedes the upload of both as the
  `ubuntu-evidence` artifact. Because a commit's own CI run exists only after push, the binding is
  to the **policy** and the **code**, not the commit: `phase2_results.py --fetch-ci` considers only
  `push` runs of `ci.yml` on the current branch whose `headSha` is an ancestor of HEAD (`git
  merge-base --is-ancestor`) and between which and HEAD no `EVIDENCE_PATHS` changed (`python`,
  `submissions`, `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`,
  `tests/test_shipped_policy.py`), downloads (`gh run download`) the newest successful one whose
  `results.json` has `controller.policy_sha256 ==` the shipped `onnx.sha256` and whose compare
  `protocol_hash` equals the current one, and copies both files to
  `docs/results/phase2/ci_ubuntu/{results.json,cmp_cal.json}`, with `ci_ubuntu/source.json`
  (`topple.ci_source/1`: `run_id`, `head_sha`, `event: "push"`, `branch`; ≤ 4096 bytes), in a
  follow-up commit that changes no evidence path (so the binding holds; G8 rechecks it with
  local git, so the done-when needs the history, not a shallow clone); `phase2.md` cites the
  committed files, never an expiring artifact URL.

`phase2_gate.py` (`--ci-dir`, `--submission`, `--results-dir`; `--fixture`, default
`tests/data/<submission name>_parity.npz`) exits 0 iff every gate in `PASS_GATES = ("G1", "G2",
"G2b", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11")` passes (G9 is `phase2_ci_check.py`),
naming each failure, and exits 2 when it cannot hash the PID and LQR baselines; the constants
live in `scripts/phase2_common.py` with `SHIP_GATES` (§2.5). Every input file is read as contract
§15 reads `results.json`: `open_regular` (≤ 16 MiB, never blocking), strict JSON types, `format`
required (`read_results` for results, the §5.3 model for compare JSON); `phase2.json` is also
refused for NaN or Infinity anywhere (the free-form summary, `pilot` and `render` objects
included) and for an integer with |n| ≥ 2^53, and the parity fixture is read with an 8 MiB cap
(`MAX_NPZ_BYTES`). In each compare entry 0 MUST be `submissions/pid_example.py` and entry 1
`submissions/lqr_example.py` (by submission sha256), and the shipped policy is the one entry
whose `policy_sha256` is the submission's `onnx.sha256` and whose `sha256` is its submission
sha256. Mean saturation is the unweighted mean of per-episode `saturation_fraction` (fsum/N).
A paired row a gate uses MUST be `topple.stats.paired_bootstrap`'s (`n_resamples` 10,000, seed
0) and equal its recomputation from the per-seed arrays within `PAIRED_ABS_TOL = 1e-12` (B19: a
platform's SIMD summation order moves the last bits); the gate decides on the recomputed bound.
G2b's success margin is decided on exact fractions (k/n). **G2, G2b, G3 (0..199 part) and G4 are
evaluated on both** the Mac compare (`out/cmp_cal.json`) and the ubuntu one
(`ci_ubuntu/cmp_cal.json`, which MUST record a `linux-*` platform, `cpu_isa` and `onnxruntime`);
both MUST pass (measured: three parity-equivalent runtimes changed all 200 episode scores, one by
22.8 points, and moved the PPO − LQR lower bound over 3.07–3.17).

| Gate | Check |
|---|---|
| G1 | eval_v1 protocol score (fsum/N): PPO > PID, strictly |
| G2 | 0..199: `paired[ppo, pid]` lower bound > 0 (Mac and ubuntu) |
| G2b | 0..199: `paired[ppo, lqr]` lower bound > 0 and success(PPO) ≥ success(LQR) − 0.03 (Mac and ubuntu) |
| G3 | success(PPO) ≥ success(PID) on 0..199 (Mac and ubuntu) and on eval_v1 |
| G4 | 0..199: PPO median energy ≤ E_ref (the code's `ScoreConfig` value) and mean saturation ≤ 0.10, both inclusive (Mac and ubuntu) |
| G5 | batch-1 parity of `submissions/ppo_balance` against `tests/data/ppo_balance_parity.npz` (≤ 1e-5, ≥ 300 unsaturated and ≥ 32 saturated rows, fixture `onnx_sha256` == `onnx.sha256`) |
| G6 | `results.json` `controller.policy_sha256` == `onnx.sha256` and `controller.sha256` == `submission_sha256(submissions/ppo_balance)`; `versions.onnxruntime` set; its `protocol_hash` is eval_v1's and `complete: true`; `load_policy_dir` accepts the submission and the loaded policy's sha256 is the hashed one |
| G7 | `phase2.json` has 3 PPO + 3 SAC runs and exactly 1 ablation run (so G10 judges it). Each PPO and SAC run has `best.rule` `pid` and one fresh-eval row: its policy is its best's `onnx_sha256` and an entry (index ≥ 2) of both fresh compares, which have the current protocol hashes, are complete and have PID as entry 0 and LQR as entry 1; ineligible ones via `--report-only`, marked "below PID gate"; no fresh rows for runs that are not reported. The shipped run is a PPO run not exported report-only whose `policy_sha256` is the submission's; it and `shipped.attempts` equal a §2.5 replay over the recorded failures, and every attempt's local G1–G4 result, recomputed from the fresh compares, is what it records (`failed_on: "local"`: exactly those gates; `"ubuntu"` and the shipped attempt: none). Only ubuntu failures and G5/G6 stay recorded claims |
| G8 | every required `docs/results/phase2/` file (`RESULT_FILES`: `hero.mp4`, `hero.gif`, `median_pid_ppo.mp4`, `median_lqr_ppo.mp4`, `gallery_1..3.gif`, `learning_curves.png`, `shove.png`, `ci_ubuntu/{results.json,cmp_cal.json,source.json}`) is a regular file (symlinks are not followed) within its §5.2 cap (MB = 10^6), the package within 20 MB in at most 256 entries; the required files, `phase2.json` and `phase2.md` are tracked (`git ls-files`) and regular files (a symlink counts as missing); nothing under `docs/results` is untracked (not ignored) or modified (`git diff --name-only HEAD`, so staged changes count); `ci_ubuntu/results.json` `policy_sha256` == `onnx.sha256`; `ci_ubuntu/source.json`'s run commit is an ancestor of HEAD with no `EVIDENCE_PATHS` changed since |
| G9 | `phase2_ci_check.py`: CI on HEAD succeeded (all three jobs `python`, `python-slow`, `python-train`; coverage ≥ 80% in each scope, §7, the train scope from when `topple/train/learn.py` exists; `test_shipped_policy` and the ubuntu compare) |
| G10 | every reported run's summary has exactly the §5.1 fields (a missing or extra key fails): no abort flag and no `nondeterministic`, `num_timesteps ≥ total_timesteps`, `peak_tree_rss_mib < 6144`, `selection_protocol_hash`, `env_algo_version` and `bundle_hash` equal the recomputed current values; shipped run `git_dirty: false` (§2.3) |
| G11 | ≥ 2 of 3 PPO runs have `best.eligible` and their exported policies pass G2, G3 and G4 on 0..199 in fresh evals; the per-run `g11` table in `phase2.json` lists exactly the 3 PPO runs and equals the recomputation on `fresh.calibration` (`eligible`, `g2`, `g2b`, `g3`, `g4`), its G2b column reported, not gating |

Every input G10 reads comes from the committed per-run summary in `phase2.json` (§5.1), never
from `runs/`.

## 5. Results package

### 5.1 Regeneration

`heavy.sh uv run python scripts/phase2_results.py --ppo runs/ppo_s{0,1,2} --sac runs/sac_s{0,1,2}
--ablation runs/ablate_nopenalty --out docs/results`:
- exports every run (candidates per §2.5; the shipped one with `topple export runs/ppo_sK --out
  submissions/ppo_balance --test-fixture ppo_balance --replace`; the rest `--report-only`);
- evaluates each fresh on eval_v1 and 0..199 (with PID and LQR);
- writes `docs/results/phase2.json`: every number, every eval row with `platform`, `cpu_isa` and
  `onnxruntime`; each run's committed summary, **exactly** these fields (normative; G10 and the
  report read only these): `run`, `algo`, `seed`, `config_sha256`, `config_file_sha256`,
  `total_timesteps`, `num_timesteps`, `selection_protocol_hash`, `env_algo_version`,
  `bundle_hash`, `git_commit`, `git_dirty`, `versions`, `mp_start_method`, `torch_threads`,
  `wall_clock_s`, `train_s`, `selection_s`, `cpu_s`, `load_avg_start`, `load_avg_end`,
  `peak_tree_rss_mib`, `peak_procs`, `selection_history`, `best` (with its `onnx_sha256`),
  `flags`, `e_ref_j`, `attempts`, `init_source` (fine-tune only); `shipped` (with every attempt:
  its run, the ship gates it failed and `failed_on`, §2.5);
  the G11 table; and `pilot` constants (energy-cost variant 64.9 / 0.505; log_std −0.5/−1
  results), each with `source` naming where it was measured (the spec section holding the pilot
  table; the pilot workflow runs are not kept, B53), rendered as "pilot measurement";
- renders `phase2.md` and `docs/results/phase2/*` from it.

Nothing is hand-edited; prose with numbers is a template filled from `phase2.json`, and the script
fails on a missing template field or an exceeded size cap. Charts are matplotlib PNGs (`dev`).

### 5.2 Contents of `phase2.md`

Under the scoreboard: "Measured on {platform} {cpu_isa}, onnxruntime {v}. CI's ubuntu numbers differ
per episode (§3.5) and are shown in the second block, from `docs/results/phase2/ci_ubuntu/` (CI
run {ci_run}, policy {policy_sha256[:12]})."

**GIF ladders** (`video.render_gif_ladder`: every rung is checked before any renders, then each in
order; the rung used goes into `phase2.json`; the build fails only if the last rung exceeds the
cap, `GifTooLargeError`; `video.py` renders only sizes that are multiples of 16 px (`MACRO_BLOCK`)
and fit the scene's offscreen buffer; every fps divides the 100 Hz control rate, which
`video._stride` requires of every render, so each fps rung is rendered at its own stride). A
rung's width is the whole GIF's: its square panes render at the smallest multiple of 16 px
covering width/panes, and the composed frame is resampled (LANCZOS) to width × round(width/panes)
(rung 389: two 208 px panes rendered, 389×194 written):
- hero: GIF 480×240 (two 240 px panes) at 20 → 10 → 5 fps, then width × 0.9 three times at 5 fps
  (432, 389, 350 px wide);
- failure galleries: 320 px at 10 → 5 → 4 fps, then × 0.9 three times at 4 fps;
- waypoint: 480 px at 10 → 5 → 4 fps, then × 0.9 three times at 4 fps.

1. **Hero.** PID vs shipped PPO via `topple video --side-by-side`: MP4 960×480 at 50 fps (≤ 8 MB;
   measured 0.24 MB) and GIF 480×240 (≤ 6 MB; measured 4.96 MB at 20 fps; hero ladder). Seed rule: among eval_v1 seeds where PID fell and PPO survived, the
   largest score(PPO) − score(PID), ties to the lowest index; if none, the largest difference
   overall. Caption template: "Seed chosen as the largest PPO−PID gap among eval_v1 seeds where PID
   fell; a median seed is below." Also MP4s of the median-difference seed (lower median) for PID vs
   PPO and LQR vs PPO. Both panes play stored trajectories on one clock; a pane whose episode ended
   in a fall or an error plays its epilogue (`wants_epilogue`, contract §11), then holds its last
   frame stamped `FELL at 4.31 s` (or `ERROR at`, `sim_unstable` included; t = T · `control_dt`).
   The video ends when the longest pane does, or 1.0 s (`STAMP_HOLD_S`) after the last stamped
   pane, whichever is later. Labels are printable ASCII (others shown as `?`, ≤ 200 characters,
   Pillow's default font), trimmed to the pane with `...`. Trajectory meta is untrusted: a
   termination reason that is not a string names no fall (the pane is unstamped, exit 0).
2. **Scoreboard.** Rows PID, LQR, "PPO (shipped: best of 3 by selection seeds)", "PPO recipe (3
   seeds)" (mean ± sd and min of fresh score and success), each PPO seed, each SAC seed (label
   "SAC (1.0M steps, UTD 0.25)"); "k of 3 PPO seeds reached the PID gate" and "j of 3 at LQR level
   (G2b)"; blocks eval_v1, 0..199 (Mac) and 0..199 (ubuntu CI); columns score [95% CI], falls,
   survival, success, median energy, mean saturation, median drift, parameter count, env steps,
   train CPU-minutes (median drift and parameter count are not in `phase2.json`; B53). Paired PPO − PID and PPO − LQR. Rows below the gate are marked "below PID
   gate".
3. **Shove test.** Preset `clean`, seeds 0..99, one push at 2.0 s (0.1 s pulse, lateral 0;
   injected with the trusted `reset(options={"pushes": …})` hook, contract §9),
   impulses 0.30:0.65:0.025 N·s (15 levels), `forward_sign` +1 and −1. Survival vs impulse for both
   directions; i50 per direction (the first adjacent pair of levels whose survival goes from ≥ 50%
   to < 50%, linearly interpolated on impulse between them; a plateau at exactly 50% crosses at its
   last level; none if there is no such pair) and their minimum (none unless both directions have
   one). Pilot: PID 0.42/0.42, LQR 0.45/0.45, PPO 0.48–0.50.
4. **Learning curves.** Selection success and score vs env steps and vs CPU-minutes, per seed and
   algorithm, PID and LQR lines. "Env steps to PID level" = [previous eval, first eligible eval].
   Wall-clock only as a footnote with the recorded load.
5. **Failure gallery.** PPO's 3 lowest-scoring seeds in 0..199 as GIF 320×320 (≤ 2 MB, gallery
   ladder), captions: termination reason, time, `fall_cause` (§8.5), last push impulse/time.
6. **Reward-design notes** (templated): u² kept (pilot: energy-cost variant); log_std 0 beat −0.5
   and −1 (pilot); "the action-excess penalty cut the share of steps with the pre-clip action mean
   outside ±1 from {x}% to {y}% at 1M steps (seed 1)", x and y = `frac_mean_outside_1` (§2.5) of
   `runs/ablate_nopenalty` and `runs/ppo_s1` at their evals nearest 1M; the 1M-step failure mode;
   "with a constant learning rate, 1 of 3 seeds regressed after 2.3M steps" (pilot, §0); LQR
   chatter, if observed.
7. **SAC vs PPO**, three labelled comparisons from logged data: (a) equal env steps (PPO's eval
   nearest SAC's best step vs SAC's best), (b) equal training CPU-seconds, (c) full budget. Fixed
   sentences: SAC is "learner-throughput-limited (UTD 0.25, serial updates)"; UTD 0.25 was chosen
   for the shared-machine budget (SAC's default UTD 1 measured 2.6–4.2× slower); target_entropy −4
   and UTD 1 were tested and worse. Peak step and final-vs-best come from `phase2.json`, never fixed
   text.
8. **README snippet.** Hero GIF and a 4-row table PID, LQR, PPO (shipped), SAC, with a footnote
   giving the PPO recipe's mean and min over 3 seeds.

Total size of `docs/results/phase2/` ≤ 20 MB.

### 5.3 Library code (torch-free) and CLI

```python
# topple/stats.py  (normative algorithm; G2/G2b and compare use only this)
def paired_bootstrap(a, b, *, n_resamples=10_000, seed=0, level=0.95) -> tuple[float, float, float]:
    # checks first (ValueError): a, b finite, 1-D, one length ≥ 1; int n_resamples ≥ 1; int seed ≥ 0;
    # 0 < level < 1; n_resamples · len(a) ≤ MAX_BOOTSTRAP_ELEMENTS = 10**8 (the protocol maximum)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)        # seed order
    idx = np.random.default_rng(seed).integers(0, d.size, size=(n_resamples, d.size))   # one call
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2], method="linear")
    return float(d.mean()), float(lo), float(hi)
def bootstrap_ci(x, **kw) -> tuple[float, float, float]                 # paired_bootstrap(x, zeros)
# topple/compare.py: paired rows for every ordered pair i > j (a = entry_i, b = entry_j)
MAX_ENTRIES = 32; CHUNK_EPISODES = 25                                   # episodes per pool task; results never depend on it
def compare(subs: Sequence[Path], protocol: EvalProtocol, *, workers: int = 4) -> Comparison
def to_markdown(c: Comparison) -> str; def to_json(c: Comparison) -> str   # Markdown escapes every ASCII punctuation char of an id
def read_comparison(path: Path) -> Comparison                           # as read_results: open_regular, ≤ 16 MiB
# topple/bench.py: protocol fixed to balance-v1/clean; ≤ MAX_IMPULSES = 200 levels, strictly increasing, in
# [0, 5] N·s (MAX_PUSH_IMPULSE_NS); 1..10,000 int seeds in [0, 2^53); CHUNK_EPISODES = 25
def shove_curve(sub: Path, impulses, seeds, *, forward_sign: int = 1, workers: int = 4) -> ShoveCurve
def shove_curves(sub: Path, impulses, seeds, *, forward_signs=(1, -1), workers: int = 4) -> tuple[ShoveCurve, ...]
def run_shoves(subs, impulses, seeds, signs, workers) -> list[tuple[ShoveCurve, ...]]   # one pool for all (the CLI)
def shove_push(impulse: float, forward_sign: int, protocol: EvalProtocol) -> Push  # force = impulse / (n_substeps · timestep)
def parse_impulses(text: str) -> tuple[float, ...]                     # START:STOP:STEP, decimal-exact, STOP included
def impulse_at_50(curve: ShoveCurve) -> float | None                   # §5.2 item 3's rule
def to_markdown(curves: Sequence[ShoveCurve]) -> str
# topple/video.py (extend; scene and obs width from the challenge registry)
def render_side_by_side(trajs, labels, out: Path, *, pane: int = 480, fps: int = 50) -> int   # frames; 1–4 panes
def render_gif_ladder(trajs, labels, out: Path, rungs: Sequence[GifRung], *, max_bytes: int) -> GifRung   # §5.2
def render_overlay(trajs, labels, out: Path, *, alphas, frame_m: tuple[float, float]) -> None  # waypoint ghost
```

`topple.compare/1` JSON (pydantic, frozen, `extra="forbid"`, `allow_inf_nan=False`; read like
contract §15's `read_results`): `{format, protocol, protocol_hash, complete, seeds[], platform,
versions{topple, mujoco, numpy, obs_schema, action_spec, onnxruntime, onnx, cpu_isa}, entries[{id,
policy_sha256|null, sha256, scores[], success[], energy_j[], saturation_fraction[], termination[],
extras{}}], paired[{a, b, mean, lo, hi, n_resamples, seed}]}`; `platform` and `versions` as in
`results.json` (ONNX episodes are bit-exact only per platform, `cpu_isa` and onnxruntime, §3.5);
1–10,000 seeds, 1–32 entries; every array in seed order with len(seeds) values, bounded as in
contract §15. `termination` is one of `fell`, `floor_contact`, `sim_unstable`, `timeout`, `error`
(contract §11; `error` for an episode the harness stopped) or `missing`; an `error`,
`sim_unstable` or `missing` episode has score 0 and success false (`compare` writes a `missing`
one's energy and saturation as 0.0), and only a `timeout` episode can be a success (contract §12).
`complete` iff no episode is `missing`. Entry ids are unique (`compare` refuses two submissions
with one name). `paired` rows come in the order (i, j) = (1, 0), (2, 0), (2, 1), …, `a` and `b`
the entry ids, `lo ≤ hi`, differences in [−100, 100], `n_resamples` in 1..10^6. `extras` maps a
metric name to its per-seed array: for waypoint `docked`, `fell`, `fall_cause`, `t_finish_s`,
`course_pts`, `speed_pts`, `alive_pts`, `battery_pts` (§8.6).

CLI: `topple train CONFIG --out DIR [--seed S] [--retry]` (`--seed` in [0, 2^32); exit 2 with one
line for a bad config, an `--out` that exists and is not an empty directory (§1) or a missing
train extra; exit 4 when the run's `train_meta.flags` holds an abort flag);
`topple export ...`; `topple policy pack ...`; `topple compare SUB... -p P [--seeds A..B]
[--workers N] [--json PATH]`; `topple shove SUB... [--impulses 0.30:0.65:0.025] [--seeds 0..99]
[--direction both|forward|backward] [--workers N]`; `topple video --side-by-side A.npz B.npz
--labels A B --out X.mp4 [--fps N] [--size W]` (pane size: each pane W×W px, default 480).
`--seeds A..B` replaces the named protocol's seeds, as `topple run --seed` does
(`dataclasses.replace(EvalProtocol.from_id(P), seeds=...)`, ≤ 10,000, contract §8), so
`protocol_hash` covers them; there is no `is_named` field. Every new command checks a relative
output path with `cli._check_output` before anything runs (never through a symlinked directory,
contract §14), writes through `_atomic.replacement` (`results.json` via `write_results`; `policy
pack` through `_atomic.new_directory`, contract §14) and fails through `cli._fail` (one escaped
line, exit 2). `topple video` loads inputs with `load_npz`, bounded by their challenge's
`max_steps` and layout (§8.1a).

## 6. Tests

**Default CI jobs** (`python`: `-m "not slow and not train"`; `python-slow`: `-m "slow and not
train"`; torch-free):

| Test file | What it checks |
|---|---|
| `test_train_config` | Shipped YAMLs validate. Rejected: extra keys, `n_envs` ∉ 1..16, the other algo's block, `health` on SAC, `action_excess: {enabled: false, box: 2}`, `checkpoint_every`, PPO + `relu`, `net_arch` depth 3, `qf` on PPO, `vf` on SAC, `log_std_init` on SAC, LR `final > initial`, anchors/aliases, depth 8,000 (ConfigError, no RecursionError), > 64 KiB; repeated, `<<`, tagged and non-scalar keys; `action_excess.enabled: 1`; 251 selection seeds and 41 periodic evals (250 and 40 accepted); 33 nesting levels (32 read). A worst-case `train_meta.json` at those bounds (two baselines, `indent=2`) is ≤ 56 KiB. `lr_fn` linear: 3e-4 at p = 1, 0 at p = 0 and at p = −0.0027 (clamp), never negative. Derived YAMLs differ from their base in exactly the §2.2-listed key paths. `config_sha256` independent of key order and different per `--seed`. |
| `test_train_wrappers` | `TrainingSeeds` seeds are `int` (`isinstance(seed, int)`, not `np.int64`), in [2^52, 2^53), deterministic per (base, index), never hit a §0 set over 10k draws. `ActionExcessPenalty` exact for (0.5, 1.5) and (2, −2), Box(−2, 2), `info["excess_penalty"]`. `make_train_env` layers env → Monitor → penalty → seeds (a stand-in Monitor; SB3's in a `train`-marked case). |
| `test_selection` | `is_eligible` truth table (pid; waypoint incl. the fall term and docked 0.65; none); `rank_key` per kind; best.json fallback; on-policy rows: 5 per seed incl. a 3-step episode, 0 rows and no crash for a T = 0 episode, `n_onpolicy_seeds` counts T ≥ 1; edge set has 6·D + 4 rows for D = 10 and 15; `late_regression` flag on a synthetic history; trigger arithmetic: labels unique, increasing, last == `final_timesteps(cfg)` (SB3's final `num_timesteps`; == total for SAC and smoke) for smoke and the shipped configs (pure function `eval_labels(cfg)`, equal to a step-by-step `EvalTrigger` simulation); `final_due` after an abort at a periodic label; `EvalTrigger.resumed` continues the schedule; `FLAGS` order identical under 3 `PYTHONHASHSEED` values; the selection protocol hash known answer. |
| `test_policy_build` | PPO 9-node and SAC 10-node sequences with the §3.2 initializer shapes (rank-0 Clip bounds), IR 8 / opset 17, canonical names, fixed point, byte-identical rebuild, `not HasField("producer_version")` and every `ListFields()` name within the check-2 allow-list, no `/Users/` `/home/` `site-packages`. Golden: the committed tiny fixture builds to a pinned sha256. `reference_actions` vs ORT ≤ 1e-5 for var ∈ {0, 1e-12, 1e-8, 1, 1e6} × \|μ\| ∈ {0, 1e3}. Head bounds come from the challenge even when the model's action space is Box(−2, 2). Named negative controls, each > 1e-3 on the fixture rows: clip_obs 5, normalization omitted, mean and var swapped, W1 transposed, head clip ±2, tanh/relu swapped, SAC head tanh omitted; and ε = 1e-6 caught by the export constant binding (not by parity). |
| `test_policy_validate` | Accepts first-party PPO and SAC. Rejects, each at its named check: unknown field 99 on ModelProto, NodeProto and TensorProto (check 2, via `UnknownFieldSet`); one case per check-2 field (`producer_version=""`, `doc_string`, `configuration`, `quantization_annotation`, `device_configurations`, `overload`, `type.denotation`, `dim.denotation`, input `metadata_props`, initializer `metadata_props`, `ref_attr_name`, node domain, `external_data`, a named Constant value tensor); invalid UTF-8 in a node name, a metadata key, a metadata value and an initializer name (`invalid_utf8`); truncated varint and depth-40 nesting (`onnx_parse_failed`); a trailing `"\n"` in a node name, a metadata key and an op_type; a 1 KiB Constant tensor name; an extra `topple.note` key, a missing required key, a non-ASCII value; `Sin`; `Flatten`; `Identity`; `variadic_max_245k` (check 6 arity); `max4_width4096_x40`, `softplus_chain_w4096_x52`, `elu_chain_w4096_x52` (check 8 cost); width 11; initializer as input; NaN weight; 2 KB metadata; (IR 8, opset 21), (IR 10, opset 20); IR 14; `If`; function; sparse initializer; int64 tensor; Constant `value_ints`; Gemm `alpha=NaN`, `transB` as FLOAT; both payloads set; 65 nodes; 131,073 elements; > 1 MiB; `gemm_transA_outer_product`; `bcast_matmul`; `max_bcast` (Max([n,1],[1,n]), check 8); unused initializer; dead node; spoofed `value_info` (check 2); one 256×256 weight reused by 2 Gemms (check 8 cost only; 1 use passes); `static_cost == 2^17` accepted, `2^17 + 1` rejected; a canonical-looking model with a Constant node (check 9). Every failure is a fixed-code `LoadError`, never a traceback. External-data red team: `chdir` to a tmp dir holding a correctly sized `.env` and a FIFO, references to both rejected at check 2 with `InferenceSession`, `check_model`, `numpy_helper.to_array` monkeypatched to raise. `<b>` in a node name, `op_type`, domain, attribute name: no `<` in any message. |
| `test_policy_canonical` | Fixed point; `float_data` and `raw_data` give equal bytes; renamed values/initializers canonicalize identically; Constants fold to initializers in first-use order; default-valued Gemm attributes are dropped (explicit `alpha=1.0, beta=1.0, transA=0` equals omitted); IR set from `CANONICAL_IR`; golden: a committed non-canonical fixture canonicalizes (pack mode) to a pinned sha256; non-canonical file rejected at load. |
| `test_policy_pack` | Third-party models (doc strings, foreign metadata, a `;` value name, `x`/`y` I/O names, static batch 1, empty node names, `Identity`) pack to a loadable dir with `action_rule: other`; the committed `thirdparty_legacy.onnx` and `thirdparty_dynamo.onnx` pack to IR 9 / opset 20 and load; the dynamo fixture with its rank-0 scale gets `other`, the same fixture with the scale broadcast to `[D]` gets `clip(mean)`; packing the first-party build is the identity; extra metadata dropped; external data rejected before any tensor read. |
| `test_policy_manifest` | Round-trip from bytes on disk of a real first-party `policy.json`; 16 KiB cap; BOM, NaN, Infinity, 1e400, duplicate keys, lone surrogate, 5,000-digit int, `3.0` for an int rejected; `1` for a float accepted; extra key `<script>` and `claims.name` `<img ...>` give no `<` in the error; claims charset; `check_against_model` per field; challenge mismatches named. |
| `test_onnx_policy` | CPU-only providers; tuple of Python floats; LoadError on sha/metadata/challenge mismatch; one `ideal` episode via `evaluate`; smoke timer: fresh RunOptions per call, cancelled timer, terminate mapped to its message. |
| `test_pool` | `PolicyRef` for ONNX and Python; mutating the files after the parent load leaves the compare on the original sha; a worker given a wrong sha raises. |
| `test_loader_policy` | Detection; ambiguous kind; extra/pickle-family/hidden/subdir entries named; missing `challenge` → TypeError; hard-linked `policy.json`; `POLICY.JSON` and a non-UTF-8 name refused on every OS; a directory of 10,000 entries refused after at most 4; `Submission.sha256` == `submission_sha256(dir)` from the same single read. Red-team (Linux, macOS): symlinked `policy.onnx`, symlink to `/dev/zero`, symlinked dir, lstat/open swap to a regular file and to a FIFO (rejected, never blocks; 5 s test timeout); oversize rejected before read. |
| `test_export_inputs` | Ignores `best.json.dir`; rejects step ≥ 2^53, symlinked or escaping step dirs, npz with 9 members / bad names / oversize / object dtype / a header declaring 2^42 elements (clean error, no allocation) / bytes prepended to the archive (contract §14 layout), NaN in `normalize.json`, bad fixture name, sha mismatch, a constant-binding mismatch, a run with an abort or `nondeterministic` flag (without `--report-only`); existing `--out` without `--replace` → exit 2; a relative `--out` under a symlinked `out` → exit 2 before any write (also `compare --json`, `policy pack`, `video`, `train`); `--report-only` twice succeeds (idempotent); vacuous coverage recorded, not failed; `.bin` v1 layout (magic, version, N, D, A, set) and `.json` sidecar fields. Also the §3.3 step 1–5 rules as built: symlink loops (step dir, `ckpt`, `./runs`) give one `ExportError`, never a traceback; the `runs/` rule (a checkout under `.../runs/topple` exports); `--replace` limits; every binding; `1 ≤ n_onpolicy_seeds ≤` selection seeds; a NaN ONNX output on a middle row fails; fixture staging and the `ExportWarning`; `snapshot_files` / `write_snapshot` round-trip through export. |
| `test_shipped_policy` (slow; parametrized over `ppo_balance` and, once it exists, `ppo_waypoint`) | The submission loads; fixture sha equals `onnx.sha256`; parity ≤ 1e-5 with ≥ 300 unsaturated and ≥ 32 saturated rows; balance: eval_v1 PPO > PID; writes `results.json` as a CI artifact. |
| `test_c_reference` | Generates a C header from a first-party `policy.onnx`'s initializers by §3.2 position (`float.hex` literals and the policy's sha256; `tests/c/genc.py`), compiles `tests/c/mlp.c` (about 40 lines) with the system `cc -O2 -Wall -Wextra` (no warning allowed), with and without `-ffp-contract=off`, and checks every `.bin` row within 1e-5 (measured ≤ 6.6e-7). It runs on tiny PPO and SAC exports from a synthetic run, so it runs before `ppo_balance` exists; the `ppo_balance` case skips until that is committed (even with `TOPPLE_REQUIRE_CC=1`). Negative controls: another policy's rows are off by more than 1e-3; a `.bin` with a bad magic, version or dims, or a truncated one, is refused with exit 2; `genc` refuses a graph that is not a §3.2 sequence and bytes that are not ONNX. Skips locally without `cc`; CI sets `TOPPLE_REQUIRE_CC=1`, which makes a missing `cc` fail. |
| `test_harness_finite_obs` | A fake env emitting NaN makes `evaluate` raise `RuntimeError` before `act()` is called, also under `python -O`. |
| `test_driver_golden` (2b) | Per platform tag, as `tests/test_golden.py` pins balance-v1's baselines (contract §13; physics reproduces only per platform, e.g. LQR seed 1001 scored 84.08 on darwin-arm64 and 85.40 on linux-x86_64 under E_ref 13.0): `EPISODES[platform_tag()]` holds `submissions/waypoint_pid_example.py`'s scores on `waypoint-v1/clean` seeds 0..4 (to 1e-9) and docked/fell on `standard` 0..19, for `darwin-arm64` and `linux-x86_64`; a platform without pins skips locally and fails in CI, and CI must run on `linux-x86_64` (the same `pins_for` / `check_ci_platform` rule); a re-pin updates both platforms in one change. The platform-independent `WaypointScore` dump is pinned once. |
| `test_traj_v2` (2b) | The §8.1a cases. |
| `test_import_hygiene` | A fresh interpreter with an import hook imports `topple.cli`, `topple.pool`, `topple.controllers.onnx_policy`, `topple.compare`, `topple.bench`, `topple.stats`, every `topple.policy.*` module it finds, `topple.export`, `topple._npz`, `topple.yamlsafe`, `topple.train` and its torch-free `config`, `wrappers`, `selection`, and (with `PYTHONPATH=scripts`) the scripts `phase2_common`, `phase2_gate`, `phase2_ci_check`: no attempt to import `torch`, `stable_baselines3`, `tensorboard` or `psutil` (a guarded `try: import` counts, installed or not; positive control) and none in `sys.modules`. |
| `test_no_unsafe_loads` | AST scan of `python/topple`, `scripts/`, `tests/`. Outside `topple/train/`'s torch half (its `__init__`, `config`, `wrappers` and `selection` are scanned): no `pickle, cloudpickle, dill, joblib, shelve, marshal, torch, stable_baselines3` imports (`__import__` / `importlib.import_module` of a literal included), except a fixed (file → names) allow-list in the test, each entry still used by its file (a stale one fails): the `train`-marked test files (`test_train_smoke`, `test_snapshot_reference`, `test_pack_torch_legacy`, `test_resources`, `test_surgery`) → `torch`, `stable_baselines3`; the merged `tests/test_contract.py` → `pickle` (its `TaskSpec` pickle round trip, contract §8); `tests/test_onnx_policy.py`, `tests/test_pool.py` → `pickle` (pickling refused, `PolicyRef` pickled); `topple/train/wrappers.py`, `tests/test_train_wrappers.py` → `stable_baselines3` (`make_train_env`'s lazy `Monitor` import, its `train`-marked case); `tests/test_train_config.py`, `tests/test_selection.py` → `pickle` (`lr_fn` and `EpisodeRows` round trips); `scripts/make_thirdparty_fixtures.py` → `torch` (§3.3, run once with the train extra); `tests/test_policy_validate.py`, `tests/test_policy_canonical.py` → `numpy_helper` and `tests/test_policy_pack.py` → `numpy_helper`, `external_data_helper` (they monkeypatch the readers to raise). Torch-free tree: onnx entry point only `onnx.ModelProto.FromString`; banned any `onnx.load*` and any reference to `numpy_helper`, `external_data_helper`, `load_external_data_for_model`, `register_custom_ops_library`, `onnx.shape_inference.infer_shapes_path`, `onnx.hub`, `onnx.utils.extract_model`, and `check_model` or `onnx.reference.ReferenceEvaluator` given a path (a string, f-string, `/` join, `Path(…)`/`str(…)` call, or a name containing "path" or "file"). `InferenceSession(` and `add_session_config_entry` (the three §3.5 literal keys) only in `runtime.make_session`. `PPO.load`, `SAC.load`, `VecNormalize.load` (qualified or not), `load_from_zip_file` only in `learn.resume_from` (`test_snapshot_reference` reloads through it). Everywhere: no `yaml.load/full_load/unsafe_load` (and their `_all` forms), `allow_pickle=True`, `torch.load` without `weights_only=True`. Every rule has a positive control. Name-based, reading import aliases as what they import; `getattr` evasion out of scope. |
| `test_repo_hygiene` | No pickle-family (`.zip` included: SB3's `model.zip`), native-code (`.so/.dylib/.pyd/.dll`), external-data (`.onnx.data`, `.onnx_data`, `.data`) or hidden files under `submissions/`, tracked or untracked and not ignored; `.gitignore` covers `runs/`, `*.pkl`, `.venv-eval/`, `.DS_Store`; `git check-ignore docs/results/phase2/hero.mp4` and `.../waypoint.mp4` return non-zero. |
| `test_stats`, `test_compare`, `test_bench` | `paired_bootstrap` known answer pinning (mean, lo, hi) for a committed 200-element vector; n = 1; all-equal; compare round-trip, all ordered pairs on 2 `ideal` seeds; shove both directions and interpolation. |
| `test_video_sbs` | Timeline (hold-last-frame, labels, trims), GIF ladder, frame rule; one `render`-marked 64 px smoke. |
| `test_resource_caps` | Pools clamp to 4; shipped configs `n_envs` ≤ 8; `limits` complete; `topple train` on a non-empty `--out` exits 2; `--retry` renames only a `wall_clock_abort` attempt of the same effective config (`config_sha256`), at most twice; an abort flag exits 4; a missing train extra exits 2 with one line and moves nothing. |
| `test_phase2_gate` | `phase2_gate.py`, `phase2b_gate.py` and `phase2_ci_check.py` on synthetic JSON: pass a good set; name each failing gate, incl. a G2b-only failure (and the §2.5 fallback picking the next candidate on it), an ubuntu-only G4 failure, each missing G10 field, a stale `selection_protocol_hash` on a SAC row, `git_dirty` on the shipped run, an untracked MP4; the exact bars (G2b's margin at 169/200 and 85/200, G4 at exactly E_ref and 0.10, G1 at PPO equal to PID); a recorded §2.5 attempt that the fresh compares contradict, in either direction, and `failed_on`'s rules; the ubuntu evidence's `source.json` (strict reading; not an ancestor of HEAD, an evidence path changed since, a commit git does not have, from a real git checkout); the CI check's four distinct messages and its job check. Also `run_command`'s timeout and output caps, strict `phase2.json` reading (`format`, exact types, a FIFO, NaN and integers ≥ 2^53), the symlinked-report G8 case, and `phase2_done.sh`: its lines equal §4's block, no `&&`/`||` and `$(` only as an assignment's value, the shellcheck directive (and a shellcheck run when installed), and runs with stand-in `uv` and `git` where a failing or different re-export, or a failing `git rev-parse` or `git status`, stops it before G9. Uses the `SHIP_GATES` / `PASS_GATES` constants. |
| `test_ci_config` | `ci.yml` and the coverage scopes: the triggers; the three jobs, their runner and the train job's 20 min; `TOPPLE_REQUIRE_CC=1` on `python`; the slow job's order (slow tests, ubuntu compare, evidence check, upload) and their push-only `if:`; the train job's commands (interim step without the train scope before `learn.py`, scoped step after, advisory legacy-exporter step); every action pinned by sha with a version comment; `permissions: contents: read` with no job-level permissions, `persist-credentials: false` on every checkout, concurrency cancel-in-progress; disjoint coverage scopes with equal `exclude_also` and `branch` settings. |

**Train CI job** (`-m train`, `uv sync --locked --extra train`):

| Test file | What it checks |
|---|---|
| `test_train_smoke` | `smoke_ppo.yaml`, `smoke_sac.yaml` finish (total = 2 × every_steps): exactly 2 distinct step dirs, labels increasing, last == `model.num_timesteps`; every snapshot file, `best.json`, `train_meta.json` (incl. `mp_start_method`, `torch_threads`) with no absolute paths; export and parity pass; `onnx_sha256` binding holds. **Determinism:** each config runs twice with the same seed; sha256 of every `actor.npz` / `critic.npz`, `onnx_sha256` per label and `selection_history` are identical. < 120 s. |
| `test_snapshot_reference` | `parity_ref` matches SB3 reloaded from `model.zip` + `vecnormalize.pkl` (`training=False`) within 1e-6. |
| `test_pack_torch_legacy` | Advisory (allowed to fail without failing the job; a torch bump can remove the legacy exporter): a Linear-Tanh MLP exported with `torch.onnx.export(dynamo=False)` packs and loads. |
| `test_resources` | 2 spawned children each holding `np.ones(200 * 2**20, np.uint8)`: tree RSS ≥ 380 MiB, `peak_procs ≥ 3`; a `SubprocVecEnv(start_method="forkserver")` of 2 envs burning ≥ 2 CPU-s each: `cpu_s` ≥ 2 after `close()`, and `cpu_s` never decreases between samples; `rss_abort` (1 MiB limit) and `cpu_abort` (1 s limit) exit 4 after a final snapshot. |
| `test_surgery` (2b) | With the pickles deleted: the 15-input policy's actions equal the 10-input policy's within 1e-6 with new dims at any value and features 2, 4, 5 at their normalized mean (z = 0); `V_15 == 0.1·V_10` within 1e-6 under the same condition; wheel_pos +100 rad gives the same actions as wheel_pos at its mean; `log_std` = −1.0; `init_source` recorded. |

## 7. CI and packaging

- **pyproject.** Base deps: `onnx>=1.23.2,<1.24` (moved from the merged `train` extra),
  `onnxruntime>=1.31,<1.32` (merged: `>=1.31`, unbounded), `protobuf>=7.36,<8` (canonical bytes
  depend on it; the golden sha tests catch drift). An upgrade of any of them is a deliberate
  re-score. `train` extra: `stable-baselines3>=2.9`, `torch>=2.14`, `tensorboard>=2.21`,
  `psutil>=6`; no onnxscript. `dev` gains `matplotlib`. New marker `train` (beside `render`,
  `slow`).
- **mypy.** The merged `ignore_missing_imports` list (`onnxruntime.*`, `stable_baselines3.*`, …)
  gains `torch.*`, `tensorboard.*`, `psutil.*` and `google.protobuf.*`: §3.4's `validate.py`
  imports `google.protobuf.message` and `google.protobuf.unknown_fields`, protobuf 7.36 ships no
  `py.typed` and `types-protobuf` is not in the lock, so strict mypy would stop at
  `import-untyped` (measured). onnx's messages then derive from `Any`: bind
  `onnx.ModelProto.FromString(data)` to a variable annotated `onnx.ModelProto` before returning
  it (`warn_return_any`). Strict stays on everywhere except one override:
  `[[tool.mypy.overrides]] module = ["topple.train.learn", "topple.train.callbacks",
  "topple.train.snapshot", "topple.train.surgery", "topple.train.resources"]`,
  `disallow_subclassing_any = false` (the default job has no SB3, so `class
  SelectionCallback(BaseCallback)` subclasses `Any`). The train job runs the same mypy config with
  the extra installed. The override is added with the torch modules: until they exist the train
  job's mypy passes without it.
- **Coverage.** pyproject's `[tool.coverage.run]` (merged: `source = ["topple"]`, with
  `[tool.coverage.report] fail_under = 80`; the `python` job measures it) gains `omit` for
  `topple/train/{learn,snapshot,callbacks,surgery,resources}.py`; `.coveragerc-train` sets
  `[report] include` to exactly those five files (a `[run] include` is ignored once `--cov` sets
  a source), `branch = True`, `show_missing`, `skip_covered` and a copy of pyproject's
  `exclude_also` (`--cov-config` replaces pyproject's settings; a test keeps them equal), and the
  train job runs `--cov=topple.train --cov-config=.coveragerc-train`, `fail_under = 80`, so the
  two 80% scopes are disjoint. Until `topple/train/learn.py` exists none of the five files does
  and coverage reports "No data to report" as 0%, so the train job runs the same tests without
  the scope until then (B37).
- **ci.yml.** Triggers: `pull_request`, and `push` to `main` and `phase-*` (merged: `main` only).
  Merged jobs keep their roles: `python` (`-m "not slow and not train"`, coverage) also sets
  `TOPPLE_REQUIRE_CC=1`; `python-slow` (`-m "slow and not train"`) runs `test_shipped_policy`, then
  §4's ubuntu `topple compare`, checks both evidence files exist and uploads that `results.json`
  (`TOPPLE_ARTIFACT_DIR: out/ci`) and `out/cmp_cal_ci.json`; those three steps run only on a
  `push` once `submissions/ppo_balance/policy.onnx` exists (§4). New job `python-train`
  (ubuntu-24.04, 20 min): `uv sync --locked --extra train`, ruff and mypy with `--extra train`,
  `pytest -m train -n 2 --ignore=tests/test_pack_torch_legacy.py --cov...` (without the `--cov`
  flags while `topple/train/learn.py` is absent), then `test_pack_torch_legacy` alone in a
  `continue-on-error` step (`if: hashFiles(...)` of the test file), torch from the pytorch-cpu
  index. Security posture (the workflow runs pull-request code): every action pinned by commit
  sha, top-level `permissions: contents: read` and none per job, `persist-credentials: false` on
  every checkout, and `concurrency` with `cancel-in-progress` (whose cancelled runs
  `phase2_ci_check.py` names).
- **Tooling for `scripts/`.** `[tool.mypy] files` adds `scripts`, `[tool.pytest.ini_options]
  pythonpath = ["scripts"]` (the tests import `phase2_common`, `phase2_gate`,
  `phase2_ci_check`), and ruff's `src` adds `scripts`.
- **Repository.** `.gitignore` (merged: `runs/`, `.DS_Store`, `*.mp4`, `/out/`) adds `*.pkl`,
  `.venv-eval/` and the negation `!docs/results/**/*.mp4` (the `*.mp4` rule would otherwise drop
  every results video). `cli.py` imports `topple.train.learn` only inside `train`.

## 8. Phase 2b: `waypoint-v1` (frozen at the end of 2b)

waypoint-v1 freezes at the end of 2b, earlier than contract §13 requires (there a challenge
freezes when it is published, Phase 3): after the 2b freeze it is treated as published, so any
change to a fixed controller's score on a fixed seed is `waypoint-v2`, not a re-pin.

### 8.1 Generalization (balance-v1 must not change)

balance-v1 `protocol_hash`, scores, success and every existing metric value MUST NOT change; new
metric keys may be added (`fall_cause`, §8.5). Golden tests pin the existing keys only.

- **Kinds.** `challenge.yaml` gains `kind: balance | waypoint` (default `balance`; balance-v1 bytes
  unchanged). Registry `CHALLENGE_KINDS[kind] = KindSpec(env_cls, obs_cls, obs_schema,
  obs_features, metrics_fn, score_fn, success_fn, score_model, rank_key, env_algo_version)`. YAML
  never names code.
- **Score config.** `ChallengeConfig.score` is validated against the kind's `score_model`
  (`BalanceScore` = today's `ScoreConfig`, dump unchanged; `WaypointScore` new). `WaypointScore`
  holds **every** waypoint scorer constant, so all of them are in `protocol_hash`: `weights` (55,
  25, 15, 5), `t_par_s` 5.0, `dock_progress_weights` (0.8, 0.2), `success_energy_factor` 5,
  `e_ref_j`; `gate_radius_m`, `leg_m` and `dock_hold_steps` come from `course`. No scorer
  constant lives in code; its dump is pinned by a golden test.
- **Presets.** A bundle may define `preset_defs`. `Challenge.preset(name) -> Preset` returns
  `config.preset_defs[name]` if defined, else `PRESETS[name]`, and accepts only names in
  `config.presets`. Every lookup goes through it: `ChallengeConfig`'s preset-name validator, env
  init / `set_preset` / reset (sensors), `make_scenario` callers, trusted verification (its env),
  `protocol_hash` (dumps the resolved preset; for balance-v1 exactly today's
  `PRESETS[name].model_dump(mode="json")`) and `Scenario.from_dict(data, *, clock=None,
  challenge=None)` (Phase 1 checks the name against `PRESETS`). `topple.disturbances.MAX_PUSHES`
  (public since Phase 1) stays 3: the PUSHES stream always draws 3 candidates, whatever a preset's
  `count`.
- **ENV_ALGO_VERSION** becomes per kind. `KindSpec.env_algo_version` is a zero-argument getter
  read at call time, never an int copied at import: balance's returns the module attribute
  `topple.challenge.ENV_ALGO_VERSION` (name and value 1 unchanged), because Phase 1's
  `test_protocol_hash_covers_the_env_algorithm_version` monkeypatches that attribute and requires
  `protocol_hash` to change (a stored copy fails it; measured); waypoint's returns its own module
  constant. Waypoint: **before the 2b freeze**, every course-generator, event or observation code
  change bumps it (nothing is published, and stale 2b runs become detectable, §8.9 item 11); **at
  the freeze** its value is recorded in `phase2.json`; **after the freeze** any change to a fixed
  controller's score on a fixed seed is `waypoint-v2` (the early freeze above, then contract §13), and the version only guards
  against accidental drift. Stream `TARGETS = 6` and `balancer-target.obs.v1` freeze at the same
  point.
- **Dispatch** through the registry (Phase 1 hard-codes balance in each): `iter_episodes` and
  `verify._replay_env` (both construct `BalanceEnv`), `verify_episode`, `verify_with_seed`,
  `rescore`, `verify_run`, `Challenge.task_spec` (raises for any other schema), `build_results`
  (`Versions.obs_schema`, `action_spec`), `scoring.episode_metrics`, `controller_obs(obs_arr, t,
  *, schema: str = "balancer.obs.v1")` (returns `BalancerObs` or `BalancerTargetObs`; the 2-arg
  call is unchanged), `make_scenario(seed, preset, *, timestep, control_dt, episode_seconds,
  course: CourseConfig | None = None)`, `load_npz` / `Trajectory._check_shapes` / `save_npz`
  (§8.1a), the CLI's upload loaders and `video.py`.
- **Golden tests.** Phase 1's pass unchanged: `tests/test_golden.py` (bundle `89a62007…0bbaedd0`;
  `protocol_hash` ideal `79450098…d309c30c`, clean `3f27d1c9…f406e3c2`, standard
  `b2436063…36bd1ea3`, a salted private one;
  `ENV_ALGO_VERSION` 1; scenarios of seeds 0, 1000, 2**53−1 per preset; PID/LQR scores per platform)
  and `tests/test_challenge.py` (the standard payload text, `ScoreConfig` dump included). New:
  `make_scenario` for standard seeds 1000..1019; an AST test forbidding `PRESETS[` in
  `python/topple` outside `disturbances.py` and `challenge.py`.

### 8.1a Trajectory format `topple.traj/2` (F41; new contract §14a)

`topple.traj/1` fixes `obs` at 10 columns (contract §14), so a 15-feature waypoint record cannot be
written in it. `/1` stays exactly as it is (bytes, reader, tests) for challenges whose `obs_schema`
is `balancer.obs.v1`; every other challenge writes `topple.traj/2`, which is `/1` (members,
dtypes, other widths, canonical zip layout, every contract §14 cap, `T, E ≤ max_steps`,
finite-only `meta_json`) except: `meta.format = "topple.traj/2"`, and `obs` is T × D float32 with
D = `len(TaskSpec.obs_features)` of its challenge, 1 ≤ D ≤ `MAX_OBS_WIDTH = 64`,
`meta.obs_features` naming exactly D features (waypoint: D = 15).
- **Reader.** `TrajLayout(format, obs_width)` in `topple/traj.py` (frozen; 1 ≤ `obs_width` ≤
  `MAX_OBS_WIDTH`), `TRAJ_V1 = TrajLayout("topple.traj/1", 10)`; `load_npz(path, *, max_steps,
  layout=TRAJ_V1)`. The `obs` header's width must equal `layout.obs_width` before its data is read
  (like the T limit), and `meta.format` must equal `layout.format`. Under `TRAJ_V1` that is all:
  the `/1` path is the merged reader unchanged (a Phase 1 test file's meta may hold neither
  `obs_schema` nor `obs_features`). Only a `/2` layout also requires `meta.obs_features` to be a
  list of exactly `obs_width` strings, checked inside the reader's guard (a bad value is a
  `ValueError` naming the file). Trusted callers pass `verify.protocol_traj_layout(protocol)`,
  derived from the protocol's challenge kind and never from the file, as they pass
  `protocol_max_steps`; the default fails closed on a waypoint file. `read_meta` accepts both
  formats (it only names the challenge and preset; the load that follows enforces the protocol's
  layout).
- **Format rule.** A trajectory is `/2` iff its `meta` has an `obs_schema` that is not exactly the
  string `balancer.obs.v1`; a meta without one (Phase 1's `test_traj.py` writes such files) is
  `/1`, and `save_npz` writes the rule's format. `Trajectory._check_shapes` requires `obs` of
  shape (T, 10) for `/1` and (T, D) with 1 ≤ D ≤ `MAX_OBS_WIDTH` for `/2`. It touches `meta` only
  with `in`, `isinstance` and `==`, never `len()` or another call on an untrusted value, so a
  forged-meta copy (`dataclasses.replace`, as `tests/verify_support.with_meta` makes) still
  constructs and then fails verification as `not_reproducible`. A 15-wide `obs` whose meta claims
  `balancer.obs.v1` cannot be constructed, so such a file fails to load (closed).
- **Verify / rescore** (contract §16) are unchanged: the replay records its challenge's layout and
  step 5 compares `obs` (shape, then bytes) and `obs_schema` / `obs_features` with it, so a forged
  width or feature list fails even past the reader. Every upload loader (`cli._load_upload`,
  `_UploadFiles`, `_render`, `topple video`, Phase 3 `publish`) passes the protocol's layout.
- **Tests** (`test_traj_v2`): waypoint round trip, written as `/2`; balance-v1 records identical
  to the merged writer's member by member (array bytes, dtypes, shapes and `meta_json`; numpy
  stamps zip entries with the save time, so whole files are not compared); refused before any
  data is read: 10-wide `obs` under the waypoint layout, 15-wide under `TRAJ_V1`, a 65-wide
  header; refused: D = 65 in `TrajLayout` and `Trajectory`, `/2` offered to balance-v1, `/1` to
  waypoint, `obs_features` not a list of exactly D names, a 15-wide `obs` claiming
  `balancer.obs.v1`; contract §14's zip/npy red-team cases over both formats. Phase 1's suite
  passes unchanged, in particular `test_traj.py::test_round_trip`,
  `::test_a_symlink_to_a_regular_file_loads`, `::test_meta_is_utf8_json_bytes`,
  `::test_fixed_trailing_shapes` and
  `test_verify.py::test_every_env_meta_field_must_equal_the_replay[obs_features-retyped]`,
  `::test_arbitrary_meta_values_never_crash_verification` (measured on a prototype of these rules:
  all 1,806 Phase 1 tests that run on macOS pass; v5.1's literal rules failed five of the six
  named).

### 8.2 Bundle `challenges/waypoint-v1/`

- **Identity.** gym id `topple/Waypoint-v1`; action `diffdrive.effort.v1`; `robot/` byte-identical
  to balance-v1's (test on `robot_hash`), so it passes `validate_robot` (contract §3) as is.
  `scene.xml` is byte-identical to balance-v1's (same options, lights, `grid` material, 40 × 40 m
  plane `floor` at z = 0, `<global offwidth="1280" offheight="720"/>`; decision log A3):
  `video.py` refuses frames larger than the offscreen buffer, so the §8.9 MP4 is 720×720. The
  scene holds no other world geom or site
  (`validate_robot` refuses them) and no name the robot uses (`build_model`), so gates and the dock
  are drawn by the renderer, never as scene geoms.
- **Episode.** `episode_seconds: 20.0` (`max_steps` 2000), `control_dt: 0.01`.
- **Course.** `course: {n_waypoints: 5, leg_m: 0.70, turn_deg: [30, 150], max_radius_m: 1.5,
  max_attempts: 1000, gate_radius_m: 0.10, dock_hold_steps: 100}`, also in `TaskSpec.extras`
  (str → float only, contract §8: `turn_deg` becomes `turn_deg_min`, `turn_deg_max`).
- **Presets** (`preset_defs`). `clean`: full sensors, balance-v1 randomization, no pushes.
  `standard`: plus `PushConfig(count=(1, 2), impulse_ns=(0.15, impulse_hi), duration_s=0.1,
  earliest_s=2.0, latest_s=18.0, min_spacing_s=3.0, lateral_fraction_max=0.3, max_force_n=50)`,
  existing fields only (so `Preset`'s dump, and balance-v1's hashes, keep their keys); the PUSHES
  stream still draws `MAX_PUSHES` = 3 candidates; pushes start on steps 200..1800 and satisfy
  contract §9's clock checks (≤ 5 N·s each); `impulse_hi` set by calibration (§8.7).
- **Seeds.** Public `wp_eval_v1` = 3000..3049, permanent alias `wp_eval_v1 → waypoint-v1/standard`
  in `ids.py`. Per-episode sd on `standard` is large (pilot: driver 5.7 at 0.30 N·s, 8.5 at 0.35;
  learned 19.7–23.3), so 50-seed public scores are provisional (templated ± from `phase2.json`).
  Every 2b statistic uses waypoint calibration seeds **0..399** with a paired bootstrap (at n =
  200 the PPO − driver CI half-width is about ±3.2; n = 400 raises P(lo > −5) for a true gap of
  −1.3 from about 0.61 to 0.89); Phase 3 ranks waypoint-v1 on ≥ 200 private seeds disjoint from
  0..399.
- **RNG.** New stream `TARGETS = 6`, append-only (contract §9).

### 8.3 Course generation (`topple/course.py`)

`make_course(seed, cfg) -> Course`, `rng = stream_rng(seed, Stream.TARGETS)`. Draws follow contract
§9's portable rule (`disturbances._uniforms`: one `Generator.random()` double `u`, then `low +
(high − low)·u` in Python), never `Generator.uniform`, whose last bit differs between macOS arm64
and Linux x86_64:
1. Each attempt draws, in order, `h0` = U(−π, π) (one double), `mags` = U(rad(30), rad(150)) (four
   doubles), `signs = rng.integers(0, 2, size=4)` (raw bits).
2. `θ_0 = h0`, `θ_k = θ_{k−1} ± mags[k−1]` (`+` iff `signs[k−1]`).
3. `p_k = p_{k−1} + 0.70·(pcos θ_k, psin θ_k)` from `p_{−1} = (0, 0)` (portable trigonometry,
   below).
4. Accept iff every `p_x² + p_y² ≤ 1.5²` (about 44% accepted); after 1000 failures, `RuntimeError`.

**Portable trigonometry.** Acceptance decides how many attempts the TARGETS stream consumes, so it
must come out the same on every platform (contract §9: scenarios materialize identically
everywhere). libm's `math.cos`/`math.sin` may differ in the last bit between platforms (numpy's
even within one, cf. contract §12's arccos rule), which could accept a course near the 1.5 m
radius on one platform and refuse it on another. `course.py` therefore computes `pcos`/`psin`
with Python float `+`, `−` and `×` only (IEEE-754, correctly rounded on every supported platform;
CPython never fuses them): `k = round(θ / (2π))`, `r = (θ − k·6.28125) − k·(2π − 6.28125)` with
2π = `2.0 * math.pi`, then the Taylor polynomials of cos and sin on [−π, π] through r^30 and
r^31 (16 terms each) by Horner in a fixed order, with coefficients `(−1)^n / (2n)!` and
`(−1)^n / (2n + 1)!` from exact integer division. Measured: within 1.3e-15 of libm on |θ| ≤ 14,
the same accepted course as libm on 20,000 seeds, 44.0% of attempts accepted, mean accepted turn
98.1°. A known-answer test pins `pcos`/`psin` at fixed inputs and runs on every platform.

Path length 3.5 m, mean turn 98°. Points are in the start frame (origin = axle midpoint at reset,
x = initial heading). `Scenario.course` stores only the accepted draws `{h0, turns}` (signed
radians), so the scenario stays portable bit for bit (contract §9); `Scenario.from_dict` checks
`|h0| ≤ π`, 4 turns, each `|turn|` in [rad(30), rad(150)]. `course_points(course)` derives the 5
points wherever they are used (env, scorer, video) with the same `pcos`/`psin`, never libm's or
numpy's, so the points too are the same doubles on every platform. The waypoint golden scenarios
(seeds 0, 3000, 2**53−1 per preset) must pass unchanged on every platform, like Phase 1's.
`course` is serialized only when non-empty (balance-v1's scenarios keep their key set). Waypoints
0–3 are gates, 4 is the dock.

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
- **Rows.** Reach checks, `d_min` and hold counting use rows k = 1..T−1 if the env terminated the
  episode (recorded reason `fell`, `floor_contact` or `sim_unstable`; the env skips the check on
  the terminating step), else 1..T (`timeout`, or `error` for an episode the harness stopped,
  contract §11).
- **Gates.** Waypoint i < 4 is reached at the first k with distance ≤ 0.10 m; at most one
  waypoint advances per step.
- **Dock.** `hold_k = hold_{k−1} + 1` if `d_k ≤ 0.10` else 0, counting from the step the dock
  becomes current; docked at the first k with `hold_k = 100`; `t_finish = k·control_dt`;
  `max_hold_steps` = the longest streak (< 100 unless docked).
- **Termination.** The full 20 s unless the §11 rules end it (`fell`, `floor_contact`,
  `sim_unstable`); docking does not end it. Eval has no drift termination.
- **fall_cause** (metric, both kinds; added to the contract's metrics vocabulary): for a fall
  (`termination` in `evaluate.FALL_REASONS` = {`fell`, `floor_contact`}) `"push"` if a push
  started in the 1.5 s before it, else `"self"`; `"none"` for every other recorded reason
  (`timeout`, `error`, `sim_unstable`; metrics are `float | int | bool | str`, so never `None`).
  `fell` (the metric and every "fell" rate below) means the same fall reasons. 1.5 s, not 1.0:
  measured push-to-fall latency reaches 0.92 s (driver) and 1.26 s (PPO), while every no-push PPO
  fall happened before 2.0 s, the earliest push time.
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
  `standard` 0..399 at the frozen `impulse_hi`, rounded to 0.1 J (pilot 22.4–22.6 J on 0..199).
- t_alive = `time_upright_s` = T·control_dt (contract §12). Errors score 0. **Success:** `docked ∧
  survived ∧ E ≤ 5·E_ref` (`survived` as in contract §12: reason `timeout` and T = `max_steps`).
  **Ranking:** round(score, 2), then success rate, then median `t_finish` with non-finishers
  counted as T.
- No drift term. **User wording:** "55 for the course, 25 for speed (full marks at a 5 s finish),
  15 for staying up, 5 for battery". t_par 5 keeps the driver's speed optimum at 0.30 m/s (at 0.30
  and 0.35 N·s) and leaves the first PPO baseline about 5.7 speed points of headroom (19.3 of 25
  per docked episode; at t_par 7 it was 22.3). No hand driver finished under 11.4 s, so t_par is
  not lowered below 5 without a feasibility probe.
- **Metrics** (18, within contract §15's 32): balance-v1's `steps, termination, survived,
  time_upright_s, energy_j, saturation_fraction, rms_tilt_rad` (`summarize` and `read_results`'
  complete-run check need `time_upright_s` in every present episode), then `waypoints, progress,
  docked, fell, t_finish_s` (= `episode_seconds` for non-finishers), `max_hold_steps, fall_cause`,
  and the four score parts `course_pts, speed_pts, alive_pts, battery_pts` (they sum to the score).
- **Exchange rate** (templated in `phase2.md` from `phase2.json`): "a fall costs about {Δ}
  points (mean PPO score of docked survivors minus fallers on 0..399), so each extra 1% of falls
  costs about {Δ/100}; finishing 1 s faster on a docked run earns 25/(T − t_par) = 1.67 points".

### 8.7 Reference driver, calibration, anchors

**Driver** `Driver(v_max, a_max, k_v=1.2)` (`controllers/waypoint_pid.py`; shipped as
`submissions/waypoint_pid_example.py` with `v_max 0.30`, `a_max 0.8`). This control law is
**normative**; anchors reuse exactly it, and `test_driver_golden` pins it. Per control step
(dt = 0.01 s), with `d`, `herr` the distance and bearing of `gate` in the robot frame:
1. **Direction.** `dir = −1` iff |herr| > 90°, else +1; `dir` may switch only while |v_ref| <
   0.05 m/s. When `dir = −1`, `herr` is wrapped by π.
2. **Speed.** `along = d·max(cos herr, 0)²`; `v_des = dir·clip(k_v·along, 0, v_max)`; `v_ref`
   moves toward `v_des` by at most `a_max·dt`.
3. **Position reference.** `x_ref += (v_ref/r)·dt` with r = 0.034 m, clamped to `travel ± 3`
   rad (travel = mean wheel angle).
4. **Balance.** `drive = (v_ref/r)/30 + k_pitch·pitch + k_pitch_rate·pitch_rate +
   k_travel·(travel − x_ref) + k_speed·(speed − v_ref/r)`, gains from the shipped LQR
   (`topple.controllers.lqr.DEFAULT_GAINS`: k_pitch 5.803, k_pitch_rate 0.7635, k_travel 0.2058,
   k_speed 0.1671).
5. **Turn.** `turn = clip(0.8·herr, ±0.30) − 0.06·yaw_rate`, with `herr = 0` when d < 0.03 m.
6. **Output** `(drive − turn, drive + turn)`.

The driver is a baseline, not an optimum (k_v = 2.0 scores about 1.2 points higher; without the
reverse rule it falls 8.0% instead of 3.5%).

**Calibration** (recorded in a `challenge.yaml` comment and `phase2.json`; all on `standard`
0..399, measured on linux-x86_64, the calibration platform (decision log R1), with each other
platform's numbers recorded beside them; the §8.9 item-1 band check runs there too), in this order:
1. **impulse_hi.** Measure the driver at impulse_hi ∈ 0.300:0.500:0.025 (9 values). Among values
   with falls ∈ [3%, 10%] and docked ≥ 88%, choose the one minimising |falls − 6%|, ties to the
   lower value; if none qualifies, 2b stops for an owner decision. Record the curve with Wilson
   95% CIs. (Pilot on 0..199: falls 2.0 / 2.5 / 3.5 / 5.0 / 9.0 / 10.5 / 15.5% at 0.300..0.450,
   docked 98.0 / 97.5 / 96.5 / 95.0 / 91.0 / 89.5 / 85.5% → 0.375.) Record the clean − standard
   driver gap (target ≥ 1.5; pilot 1.71 at 0.35, 2.80 at 0.375).
2. **E_ref** from the driver at the chosen impulse_hi.
3. **Anchors** re-measured with the frozen scorer (table below).
4. **Pilot PPO.** One from-scratch seed (`ppo_waypoint.yaml`, seed 0) at the frozen values; check
   §8.9 item 6, measure its `cpu_s` (sets waypoint `max_cpu_s` = 3×, §1) and apply the budget rule
   (§8.8) before any other 2b training.
5. Every number in §8.2, §8.6, §8.7 and §8.9 is re-measured and templated from these values.
   Waypoint `ENV_ALGO_VERSION` is recorded (§8.1).

**Anchors** (`topple/controllers/anchors.py`, shipped as `submissions/anchors/*.py` for
`compare`; measured on `standard` 0..399; the table records score, docked %, fell % and median
`max_hold_steps`):

| Strategy | Definition | Requirement | Pilot (t_par 5) |
|---|---|---|---|
| Balance in place | LQR, zero speed reference | ≤ 25 | 19.9 (20 seeds, 0.30 N·s) |
| Spin in place | LQR, constant turn 0.30 | ≤ 25 | 19.3 (20 seeds, 0.30 N·s) |
| Creeper | `Driver(v_max=0.12, a_max=0.8)` | ≤ driver − 10 | 60.78 vs 82.87 (0.30 N·s) |
| Gates-only | the driver law aiming 0.30 m past the dock along the last leg; docks only incidentally (a push can stop it inside the dock) | ≤ driver − 5 | 70.40 vs 82.03, docked 1.0%, undocked `max_hold` median 65 (0.35 N·s) |
| Reckless | `Driver(v_max=0.60, a_max=0.8)` | < driver | 73.33 (0.30 N·s) |

### 8.8 Training for 2b

`configs/train/ppo_waypoint.yaml` is a complete file (§2.2) that differs from `ppo_balance.yaml`
in exactly these key paths (it inherits the linear LR schedule, `start_method` and health):

```yaml
total_timesteps: 6_000_000     # adopted reward plateaued by 3M (mean waypoint idx 4.52 @3M, 4.55 @4M, 4.28 @5M)
env:
  challenge: waypoint-v1
  preset: standard
  max_drift_m: 2.5             # training-only: course radius 1.5 m + 1 m
  reward: {tilt: 0.30, rate: 0.10, drift: 0.0, yaw_rate: 0.05, effort: 0.10, smooth: 0.20}
  waypoint_reward: {balance_scale: 0.1, progress: 20.0, gate: 10.0, dock: 20.0, terminated: -50.0, dock_hold: 0.0}
selection: {protocol: waypoint-v1/standard, seeds: {first: 2000, last: 2199}, every_steps: 500_000,
            workers: 4, gate: waypoint}   # 11 periodic + final = 12 evals
limits: {max_wall_clock_s: 3600, max_cpu_s: 7500, max_tree_rss_mib: 8192}   # max_cpu_s reset to 3x the pilot's cpu_s
```

**Budget rule.** From the §8.7 pilot's stored per-seed selection scores, compute
`paired_bootstrap(mean of the two latest evals' per-seed scores (labels nearest 5.5M and the
final), mean of the evals nearest 3.5M and 4.0M)`. If its lower bound is > 0, `total_timesteps`
becomes 8,000,000 (every_steps unchanged) for both arms; the CI and the choice go into
`phase2.json`. (Comparing like with like: a max over 8 noisy evals sits about 1.4 SE ≈ 2 points
above the true value, so a max-based rule is biased toward stopping.)

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
- **Pre-planned reward iterations** if §8.9 item 6 fails for every candidate: (1) `dock_hold:
  0.2`; (2) additionally `terminated: -100.0`. Each is one pilot seed, reported. If both fail, 2b
  stops and the owner decides.

**Selection (gate `waypoint`).** At train start the harness measures the creeper and the
reference driver on the selection seeds (scores and fall rate). Eligible iff score > creeper's ∧
docked ≥ 0.65 ∧ `fell ≤ driver_fell + 0.20` ∧ `paired_bootstrap(scores, driver.scores).lo >
−5.0`, so eligibility mirrors all four item-6 sub-gates (docked with a 0.05 margin like balance's
saturation 0.08 vs 0.10; the fall term without margin, because the pilot's 16 pp gap would make a
+0.15 term reject a score-equal policy). Rank by the waypoint `rank_key` (§2.4). Each eval
records mean gates, docked, fell, median `t_finish`.

**Two arms, 3 seeds each:**
- **(a) From scratch:** `ppo_waypoint.yaml`.
- **(b) Fine-tune:** `ppo_waypoint_ft.yaml` = (a) plus `init: {from_runs: [runs/ppo_s0,
  runs/ppo_s1, runs/ppo_s2], step: best, zero_inputs: [2, 4, 5], log_std: -1.0}`. Fine-tune
  seed k starts from `from_runs[k]` at its `best.json.step`, which MUST be `eligible: true` with
  no abort flag; otherwise the next index (k+1, k+2 mod 3) is used. Source run, step and
  `onnx_sha256` go into `train_meta.init_source` and `phase2.json` (not all from the shipped run:
  that would add best-of-3 selection bias and hide pretraining variance). The LR schedule
  restarts at 3e-4. `extend_obs` (pickle-free; reads only `best.json` (integer step),
  `actor.npz`, `critic.npz`, `dist.npz`, `normalize.json`, `meta.json`, with the §3.3 file
  rules):
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
     gate/next x,y mean 0, var L² = 0.49; `final` mean 0.5, var 0.25 (measured in range: wheel_vel
     at 0.30 m/s is z ≈ 1.1, wheel_pos after 3.5 m z ≈ 8–9, below clip 10).
- **Headline comparison:** env steps to the first eligible selection eval, median over 3 seeds,
  with and without the 5M pretraining steps (pilot: fine-tune docked 56% at 1.5M vs 0% from
  scratch, but finished slightly lower at 3M, 80.45 vs 82.4). Also per arm, mean ± sd over 3
  seeds of the fresh 0..399 score and fell rate, and PPO median energy vs E_ref ("learned
  policies spend ~{k}× the driver's energy and forgo most of the 5 battery points").
- **Shipping** (mirrors §2.5). Candidates: the eligible `best` checkpoints of all 6 runs (both
  arms; no abort flag), ordered by the waypoint `rank_key` on the selection seeds; the arm is
  recorded. Export the first to `submissions/ppo_waypoint` (`--test-fixture ppo_waypoint
  --replace`), evaluate it fresh on 0..399 and move to the next if it fails item 6 or item 9.
  Every attempt and a per-run item-6 table ("k of 6 runs pass") are reported. "Better arm" is a
  reporting term only: the arm with the higher mean fresh 0..399 score.

### 8.9 Done-when for 2b

`scripts/phase2b_done.sh` (`set -euo pipefail`):

```bash
test -z "$(git status --porcelain -- submissions challenges docs/results tests/data configs scripts python)"
export UV_PROJECT_ENVIRONMENT=.venv-eval
uv sync --locked          # with the dev group (pytest, below); no extras, so still torch-free
uv run --no-sync python -c 'import importlib.util,sys; sys.exit(importlib.util.find_spec("torch") is not None)'
A="submissions/waypoint_pid_example.py submissions/anchors/creeper.py submissions/anchors/gates_only.py \
   submissions/anchors/reckless.py submissions/anchors/balance_in_place.py submissions/anchors/spin.py"
uv run --no-sync topple compare $A submissions/ppo_waypoint -p waypoint-v1/standard --seeds 0..399 --workers 4 --json out/wp_cal.json
uv run --no-sync topple compare submissions/waypoint_pid_example.py submissions/ppo_waypoint \
   -p waypoint-v1/clean --seeds 0..399 --workers 4 --json out/wp_clean.json
uv run --no-sync topple compare submissions/waypoint_pid_example.py submissions/anchors/creeper.py \
   submissions/ppo_waypoint -p wp_eval_v1 --workers 4 --json out/wp_eval.json
uv run --no-sync pytest -m "not train" tests/test_golden.py tests/test_challenge.py tests/test_traj_v2.py \
   tests/test_waypoint_verify.py tests/test_driver_golden.py \
   "tests/test_shipped_policy.py::test_shipped[ppo_waypoint]"                              # items 2, 3, 9
unset UV_PROJECT_ENVIRONMENT
K=$(uv run python -c 'import json;print(json.load(open("docs/results/phase2.json"))["waypoint"]["shipped"]["run"])')
D=$(mktemp -d)/p2bx
uv run topple export "runs/$K" --out "$D" && cmp "$D/policy.onnx" submissions/ppo_waypoint/policy.onnx
uv run python scripts/phase2_ci_check.py --head "$(git rev-parse HEAD)" --workflow ci.yml          # item 10
uv run python scripts/phase2b_gate.py out/wp_cal.json out/wp_clean.json out/wp_eval.json docs/results/phase2.json \
   --submission submissions/ppo_waypoint --fixture tests/data/ppo_waypoint_parity.npz --results-dir docs/results/phase2
```

`phase2b_gate.py` exits 0 iff all items pass (item 7 is read from `phase2.json`, written by
`phase2_results.py` from `surgery.sanity_check` in the train environment):
1. **Challenge frozen.** Bundle, scorer and calibration comment frozen; the §8.7 band (falls in
   [3%, 10%], docked ≥ 88%) and all five anchor requirements hold on `standard` 0..399.
2. **No regression.** balance-v1's Phase 1 goldens (`tests/test_golden.py`: bundle and
   per-preset `protocol_hash`, scenarios, scores) and the pinned standard payload with its
   `ScoreConfig` dump (`tests/test_challenge.py`) unchanged (pytest above).
3. **Verification** (pytest above). `verify_episode` reproduces 5 waypoint trajectories recorded
   on `p5 = dataclasses.replace(EvalProtocol.from_id("wp_eval_v1"), seeds=(3000, …, 3004))`, and
   `verify_run(p5, …).failed == 0` (`verify_run` expects all `len(protocol.seeds)` episodes and
   counts each absent one as `missing`). `topple verify` exits 0 on each of the 5 files and, in a
   slow test, on the directory of a full 50-episode `topple eval -p wp_eval_v1` run: the CLI checks
   a directory against the protocol's whole public seed list and has no `--seeds`, so a 5-file
   directory would count 45 `missing` and exit 1. An edited `meta.scenario.course` gives
   `not_reproducible`; a 10-wide obs in a waypoint upload and a 15-wide obs in a balance upload
   are rejected by `load_npz` (§8.1a); trusted and env scenarios are equal on 5 seeds;
   `waypoint-v1/standard` pushes lie in [2.0, 18.0] s.
4. **Contract** updated: §1 (gym id, `wp_eval_v1`, `topple.traj/2`), §4 (`max_steps` per
   challenge), §6 (`balancer-target.obs.v1`, replacing the "target_x, target_y" example), §9
   (`TARGETS = 6`; course draws and their portable acceptance, §8.3), §10 (per-bundle presets),
   §12 (metrics vocabulary: `fall_cause`), §12a (waypoint score and score parts), §13 (per-kind
   `ENV_ALGO_VERSION` with its call-time getter, the §8.1 versioning rule; mujoco and onnxruntime
   upgrades mean a re-score, not a challenge bump), §14a (`topple.traj/2`, §8.1a), §16 (trusted
   scenario includes `course`; uploads load with the protocol's layout), §17 (non-finite
   observations outside the policy domain). Schema id, feature order, gym id and alias frozen.
   Checked by a grep of `docs/contract.md` for each named term.
5. **Results.** Both arms × 3 seeds in `phase2.md` vs the driver on 0..399 (paired bootstrap) and
   on `wp_eval_v1`, with gates, docked, falls, `t_finish` per selection eval, the headline
   comparison (§8.8) and an anchors-plus-PPO table of score parts, docked, fell and median
   `t_finish`, plus the exchange-rate sentence (§8.6).
6. **Learnability** (relative to the driver, at the frozen `impulse_hi`, on `standard` 0..399).
   `submissions/ppo_waypoint` (§8.8 shipping rule): (a) docked ≥ 60%; (b) `paired[ppo, driver]`
   lower bound > −5.0; (c) score > creeper's; (d) fell ≤ driver's fell + 0.20. Falls are reported
   split by `fall_cause`. (e) **Report only:** PPO and driver falls on `waypoint-v1/clean` 0..399
   (pilot on 0..199: PPO 5.0%, all launch falls at t = 0.60–1.89 s at gate index 0 or 1; driver
   0%). Pilot numbers are templated from `phase2.json`.
7. **Fine-tune sanity.** `surgery.sanity_check` on `balance-v1/clean` seeds 0..19 with a test
   wrapper that appends 5 zeros and adds +100.0 rad to features 4 and 5: survival of the full
   10 s ≥ 95%.
8. **Video** `docs/results/phase2/waypoint.{mp4,gif}` (tracked, §4 G8 rule): driver and
   `ppo_waypoint` on one course in one scene, one clock; seed = the `wp_eval_v1` seed at index 24
   of the 50 sorted by (PPO score, seed).
   - Camera fixed, elevation −55° to −60°, azimuth along the course's principal axis; frame =
     bounding box of the course ∪ both trajectories' xy + 0.5 m, clamped to [3.4, 5.0] m.
   - Each robot gets a coloured disc of radius 0.08 m and a heading arrow under it (the robot
     itself is 16–23 × 7–10 px at GIF scale); the driver is drawn at 50% alpha.
   - Gates and dock drawn as discs of radius `gate_radius_m`: numbered gates, a checkered dock
     disc; fading trails; push arrows for both robots (video.py's arrows); per-robot gate
     counter, timer and finish time.
   - Trim: `e_i` = t_finish if docked, t_fall if fell, else 20 s; end = min(20 s, max_i e_i +
     1.5 s). A docked robot keeps playing; a fallen robot plays its 1.0 s epilogue, then holds
     with a `FELL at t` stamp.
   - MP4 720×720 at 50 fps, ≤ 8 MB; GIF 480×480 ≤ 5 MB (waypoint ladder, §5.2).
   - **Waypoint failure gallery:** PPO's 3 lowest-scoring seeds on 0..399 as overview GIFs at
     480×480 (gallery ladder from 10 fps, ≤ 3 MB each), captioned with `fall_cause`, t_fall and
     the current waypoint index.
9. **Export.** `submissions/ppo_waypoint/` is a 15-input `topple.policy/1`, re-exported
   byte-identically (`cmp` above); fixture `tests/data/ppo_waypoint_parity.npz` `onnx_sha256` ==
   `onnx.sha256`; parity ≤ 1e-5 with ≥ 300 unsaturated and ≥ 32 saturated rows (G5 binding).
10. **CI.** `phase2_ci_check.py` passes on HEAD, including `test_shipped_policy[ppo_waypoint]`
    on ubuntu.
11. **Binding.** Every reported 2b run's `selection_protocol_hash` and waypoint
    `env_algo_version` equal the current values; a run trained before a re-calibration (e.g. the
    §8.7 pilot) is re-trained or excluded.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Phase 1 changes move PID/LQR/E_ref (the merge moved E_ref 13.3 → 13.0, the linux-x86_64 recalibration 13.0 → 13.1, §0) | Baselines measured in the same run; E_ref from ScoreConfig; env version and protocol hash bound in train_meta (G10); the linux-x86_64 recalibration (decision log R1) preceded all training, and any later re-pin means re-running or excluding runs bound to the old protocol hash. |
| A seed lands low | Linear LR decay (constant LR collapsed 1 of 3 seeds), 3 seeds, 200 selection seeds, eligibility mirrors G3/G4, `late_regression` flag, fallback to the next run on any `SHIP_GATES` failure; G11 and the recipe row expose a broken recipe. |
| Training silently loses determinism | Twice-run smoke test with sha equality; retries must reproduce the aborted attempt's `onnx_sha256` (`nondeterministic` flag). |
| Mac-only evidence flips on the ranking platform | G2–G4 evaluated on both the Mac and the ubuntu CI compare. |
| Shared-machine load | Wall-clock is a runaway guard only; CPU-seconds cap and axis; aborted runs re-run, never reported. |
| SAC never eligible (expected, §0) | Always a best checkpoint; `--report-only` with vacuous-parity flag gives every G7 row. |
| Linux ORT diverges per episode | Tolerance-based open-loop parity; CI gates only orderings; platform and `cpu_isa` on every row. |
| onnx parser CVEs / model DoS | Bytes-only parsing, field allow-list, arity and static-cost caps, canonical form, pinned onnx/protobuf; sandbox and process kill in Phase 3. |
| Waypoint reward fails the learnability gate | Calibrate first on 0..399, pilot one seed, eligibility mirrors all four item-6 sub-gates, fallback across 6 runs, two pre-planned reward iterations, then an owner decision. |
| Registry refactor changes balance-v1 | Golden scenario, score, ScoreConfig and per-preset protocol-hash tests; `Challenge.preset` with an AST guard. |

## 10. Deferred (explicitly not Phase 2)

- **Phase 3 runner (MUST).**
  - **Re-run, never replay, for ONNX entries.** Uploaded trajectories are display-only; results
    come only from the `rescore` job below (it re-runs the policy; `topple.verify.rescore`
    replays uploaded commands).
  - **(i) No PR code, and untrusted bytes never share a VM with a secret or a write token.**
    `publish.yml` is split into two jobs on separate VMs:
    - `rescore`: `permissions: {}`, no secrets, `actions/checkout` with `persist-credentials:
      false`, runs only base-branch code from a pinned image digest (never checks out, installs
      or imports anything from the PR head). It fetches the PR files through the API by the
      immutable `github.event.workflow_run.head_sha` (re-checked against the PR API),
      size-capped before parsing (1 MiB, 16 KiB, 64 KiB), validates and runs them in the
      sandbox (workspace not mounted), and emits only length-capped JSON (≤ 64 KiB) plus
      trajectory npz (`topple.traj/1`, or `/2` for waypoint, §8.1a).
    - `publish` (`needs: rescore`): holds the write token, consumes only that output and
      re-validates it; it never parses ONNX.
    - Nightly private-seed re-evaluation (a private `EvalProtocol`: salt in [2**64, 2**128),
      repr redacted, contract §8): the salt lives only in the host process of a job with
      `permissions: {}`; the sandbox gets an empty environment (`docker run --env-file
      /dev/null`, no secret mounts); the salt is rotated each season. Red-team item: the sandbox
      can read neither `GITHUB_TOKEN` nor the salt (env, `/proc/1/environ`, `.git/config`).
  - **(ii) Budget.** Total policy CPU ≤ 60 s per protocol run (first-party needs about 2.4 s for
    200 × 2000 × 6 µs; the weighted cost cap bounds the worst accepted policy at ≤ 8.4 s per 400k
    acts on M5). The per-act bound is a CPU-time limit, or a wall limit ≥ 50 ms (a 5 ms wall kill
    on a shared 1-CPU container kills on jitter); the 60 s aggregate CPU budget does the real
    bounding. A timed-out episode is retried once before it scores 0; exceeding the aggregate is
    `timeout` (the proxy raises `RunnerError`, contract §15), score 0 for the remaining episodes.
    `scripts/bench_policy_caps.py` MUST show worst accepted × 400k ≤ 20 s on the ubuntu x86
    image, including subnormal activations, and checks in the sandboxed child that ORT does not
    set FTZ/DAZ.
  - **(iii) Boundary.** Data crosses the sandbox boundary only as length-capped JSON or
    trajectory npz read by `load_npz` with the protocol's `max_steps` and layout (contract §14,
    §8.1a), never multiprocessing/pickle; Phase 2's pools are not reused there.
  - **(iv) Ownership.** Community entries live under `submissions/community/<lowercased login>/`;
    first-party entries stay outside that tree, so no login can shadow them. A submission PR may
    change only files under its own directory, else it is not processed as a submission.
    Leaderboard rows key on (numeric `user.id`, `policy_sha256`); the login is display-only and
    refreshed from the API. An exact `policy_sha256` already on the board from another `user.id`
    gets error `duplicate` (earliest merge wins) and is not ranked. `claims.name` is shown only
    as quoted self-reported text; names matching `/official|topple/i` are reserved.
  - **Identity, R2.** Scores bind to `policy_sha256` computed by the trusted stage. Before upload, a
    second sandbox checks validation and the canonical fixed point; the Arena verifies the fetched
    bytes' sha256 against the leaderboard record before creating a session.
  - **Sandbox:** no network, read-only, cap-drop ALL, no-new-privileges, cgroup `--memory 512m`,
    64 pids, 1 CPU, non-root, empty environment, no workspace or credential mounts; one sandbox
    per policy; 10 s CPU at load; no RLIMIT_AS.
  - **Intake:** scrub `ORT_*`; pin onnx/onnxruntime/protobuf by image digest; check `cpu_isa`.
  - **Provenance:** `OFFICIAL.json` or attestations; near-copy flag via a 256-probe fingerprint
    at 1e-3 (function-level identity, §3.1).
  - **Private seeds:** from [0, 2^52), disjoint from §0; waypoint-v1 ranks on ≥ 200.
  - **`docs/design/phase-3.md` draft MUST be revised:** drop the `controller.py` shim, RLIMIT_AS
    for ONNX children, the 0.25 s act limit for ONNX and the fixed 88 B frame (frame = 8·(1 +
    len(obs_features)) bytes; waypoint 128 B).
- **Phases 3 and 5 (display).** `claims`, README, metadata and errors are untrusted: escaped or not
  rendered; `claims` labelled self-reported.
- **Residual Phase 1 risks** (robot validation gaps, the pending darwin-arm64 golden re-pin
  (decision log R2), sensor-noise portability): `docs/design/deferred-risks.md`, each with its owning phases (Phase 3 first for
  the robot validation gaps, since robot-ci scores users' own robots).
- **Phase 5 (Arena).** Pin `onnxruntime-web@1.30.0` exactly (lockfile; measured: PPO 7.7e-7, SAC
  4.9e-7 vs SB3 for IR 8–11; 1.20.1 also passes IR 8–10). Canonical bytes from R2 by sha256 in a
  Web Worker with `executionProviders: ['wasm']`, `graphOptimizationLevel: 'disabled'`,
  `ort.env.wasm.numThreads = 1` (WebGPU/WebGL would add numeric variance); IR/opset checked from
  `policy.json` (bound to the model, §3.1). The **first** Phase 5 done-when line is a Node CI test
  (`tests/web/parity.mjs`) of the frozen `submissions/ppo_balance` and `ppo_waypoint` against
  their `.bin` v1 fixtures (≤ 1e-5), recording version and max diff. The Arena worker enforces a
  per-act time budget, and R2 never receives a policy whose Phase 3 run hit `timeout`.
- **Phase 6 (autotune).** Linear drift cost with `max_drift_m` 1.5, energy terms on top of u²,
  `push_scale` curriculum. Model-generated YAML goes through `safe_load_capped`.
- **Phase 8 (firmware).** `topple export --target c` reads only `policy.onnx`, accepts only the two
  §3.2 structure matches (§3.4; Phase 2's `test_c_reference` already enforces the format on the
  host), dispatches on `action_rule`, emits `policy_weights.h` (`float.hex()` literals,
  `policy_sha256`). Firmware checks `isfinite` on the observation and on the action and fails
  safe to zero command with an error flag (runtimes disagree on NaN, §3.5). Done-when: host
  parity within 1e-5 compiled with and without
  `-ffp-contract=off`; on the ESP32-S3, the `.bin` rows (EMBED_FILES) within 1e-5 and `act()` ≤
  1 ms at 240 MHz (≈ 5k MACs for [64,64]).
- **Phase 4.** Escalating-shove clip 0.35 → 0.60 N·s in 0.05 steps, 2 s apart.
- **Phase 11.** "Verified training" badge only via repo-owned workflows with attestations.
- **Phase 13.** SAC at ~30 min wall-clock per run (UTD 1) for a fair SAC vs PPO claim.
