# Phase 2 decisions: one row per critique issue

Source: `docs/design/phase-2-critiques.json`. It holds 6 critiques with 68 issues in total. Every
issue appears below exactly once. The re-critique of v2 (round 1, 54 issues) is decided in
"Re-critique round 1", and the re-critique of v3 (round 2, 54 issues) in "Re-critique round 2" at
the end. Later rounds supersede earlier rows where they conflict; each superseded row is named in
the later round's preamble. The round-1 and round-2 critique texts were supplied inline to the
designer and are not yet committed as JSON (see round 2, consistency #15).

Decision values:
- **accepted:** adopted as recommended.
- **modified:** the problem is accepted, but the fix differs.
- **rejected:** not adopted.
- **deferred:** assigned to a later phase.

Section references (§) point into `docs/design/phase-2.md`.

## wf_3f0c2a9a-a51 / security

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Exported controller.py shim makes every ONNX submission executable code | HIGH | accepted | The data-only `topple.policy/1` directory holds only policy.onnx and policy.json. Adding any .py is a LoadError (§3.1). |
| 2 | No acceptance policy for untrusted ONNX; path loading lets the model choose files | HIGH | accepted | The validator runs on bytes only and ORT loads canonical bytes (§3.4, §3.5). Caps were reconciled with run wf_2fb93652-cd0. |
| 3 | Export reads pickles (`topple export RUN_DIR` executes code) | HIGH | accepted | Strengthened: export reads only actor.npz, normalize.json and parity_ref.npz, and never imports torch. An AST test bans unsafe loads (§3.3, §6). |
| 4 | Export recipe fails on torch 2.14; naive fixes emit external data and paths | HIGH | modified | The graph is built with onnx.helper (IR 8, opset 17) and torch.onnx is not used at all. This is the route deploy runs recommended and measured (§3.2). |
| 5 | Exported models leak local paths; model strings reach PR comments unescaped | MEDIUM | accepted | Canonicalization, run-relative metadata and a byte-scan test are in Phase 2. Escaping when rendering is deferred to Phase 3/5 (§10). |
| 6 | Provenance in policy.json and metadata is self-asserted and forgeable | MEDIUM | accepted | Provenance moves into a `claims` block that is labelled self-reported and never used for ranking. OFFICIAL.json and the fingerprint are deferred to Phase 3. |
| 7 | ORT session settings incomplete; EP not asserted | MEDIUM | accepted | `make_session`: CPU EP asserted, 1 thread, DISABLE_ALL, load from bytes, ONNX format pinned (§3.5). |
| 8 | Phase 3 runner duties for ONNX unstated | MEDIUM | deferred | The duties are written into §10 for Phase 3 (sandbox, intake, rlimits, Arena WASM EP). Moving onnx to the base deps is done now. |
| 9 | ONNX submission format missing from the contract | LOW | accepted | New contract §17 `topple.policy/1`. Tightening it is a format bump (§3.4, §3.6). |

## wf_3f0c2a9a-a51 / deploy

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Parity reference undefined; testing against the wrapper misses normalization bugs | HIGH | modified | The reference is SB3's own predict pipeline, computed from the live model at checkpoint time instead of by reloading pickles. The observation sets were adopted. Negative controls run torch-free against a numpy reference (§2.4, §3.3, §6). |
| 2 | Exporter, IR version and single-file output unpinned | HIGH | accepted | onnx.helper build with an explicit IR 8, opset 17 and a single file. The onnxruntime-web Node CI job is deferred to Phase 5, when `web/` exists (§10). |
| 3 | Action parity does not equal episodes; "beats PID" must use the shipped ONNX | MEDIUM | accepted | Done-when evidence comes from `topple eval submissions/ppo_balance`. Selection uses in-memory ONNX. `Versions.onnxruntime` is recorded (§4, §3.6). |
| 4 | SAC needs its own export (tanh, ReLU, actor.mu) | MEDIUM | accepted | Export dispatches by algorithm, with `action_rule` `tanh(mean)` for SAC (§3.2). |
| 5 | Waypoint fine-tune needs explicit VecNormalize and policy surgery | MEDIUM | modified | `extend_obs` was adopted with 15 features. The new-dim stats come from a 50k rollout, then normalization is frozen instead of reset to count 1e4, to keep the pretrained inputs' scale (§8.8). |
| 6 | Browsers cannot read metadata_props; policy.json must be the manifest | MEDIUM | accepted | `PolicyManifest` is normative and the metadata mirrors it with JSON encoding (§3.1, §3.2). |
| 7 | ONNX submissions as data: controller.py optional or a fixed template; no pickles | MEDIUM | modified | No controller.py at all, not even a template, so there is no hash allow-list. Pickle-family files are rejected (§3.1). |
| 8 | Portable graph profile and C path; cross-runtime agreement is tolerance-based | MEDIUM | accepted | Canonical node sequence and initializer names; agreement within 1e-5, never bit-exact (§3.2, §3.5). `--target c` is deferred to Phase 8. |
| 9 | best/ must pair model and stats atomically; export should not unpickle | LOW | accepted | Immutable `ckpt/step_N/` with an atomic rename, plus a `best.json` pointer (§2.4). |
| 10 | ONNXPolicy runtime details (Python floats, batch-1 parity, threads) | LOW | accepted | Returns a tuple of floats; batch-1 parity; deterministic compute; one immutable session per loaded policy (§3.5). |
| 11 | Curriculum stages do not map to presets; torch thread pinning hits the wrong process | LOW | accepted | Curriculum dropped. Threads are pinned in the learner and OMP=1 in workers (§1, §2.2). |

## wf_3f0c2a9a-a51 / product

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Waypoint score pays about 45/100 for ignoring the course and ranks slower drivers higher | HIGH | accepted | The 55/25/15/5 formula with progress credit and no drift term was adopted (§8.6). E_ref is calibrated. |
| 2 | "PPO beats PID" undefined on 20 noisy seeds; a chattering policy can win | HIGH | accepted | Gates G1–G4: eval_v1 score, paired-bootstrap CI on 0..199, success ≥ PID, energy and saturation caps. LQR is reported as a stretch goal (§4). |
| 3 | "Within 1.5 m" waypoint distribution makes seed luck dominate | HIGH | accepted | Fixed-leg courses of 5 × 0.70 m, with turns of 30–150° and a 1.5 m radius by rejection (§8.3). |
| 4 | "Pushes as standard" breaks 20 s episodes; raising the push count rewrites balance-v1 | HIGH | modified | Per-bundle `preset_defs` for new challenges only; balance-v1 keeps its built-in table, so its hash is stable. `_MAX_PUSHES` is untouched because waypoint needs at most 2. Adds a golden test on seeds 1000..1019 (§8.1, §8.2). |
| 5 | Target observation pose source, frame, timing and fields unspecified | HIGH | accepted | `balancer-target.obs.v1` has 15 features from the true pose in the yaw-only robot frame (§8.4). |
| 6 | WaypointTask has no done-when and needs a hidden refactor | MEDIUM | accepted | Split into Phase 2b with its own done-when and a `kind` registry. No import strings in YAML (§8.1, §8.9). |
| 7 | Results package: specify shareable artifacts | MEDIUM | accepted | Hero clip, CI scoreboard, shove curve, learning curves, failure gallery, notes and README; one command regenerates them (§5). |
| 8 | Side-by-side video seed, sync, labels; waypoint camera | MEDIUM | accepted | Written seed rule, same-clock playback, hold-last-frame and overlays; overview camera for waypoint (§5.2, §8.9). The escalation clip is deferred to Phase 4. |
| 9 | LQR in the table but not the repo; chatter confuses the table | MEDIUM | modified | LQR now ships (`controllers/lqr.py`, `submissions/lqr_example.py`, a discrete LQR design), so PID-form gains are not shipped. Survival and success are separate columns, and chatter is reported if observed (§5.2). |
| 10 | No waypoint training reward; balance reward fights the task; fine-tune accounting | LOW | accepted | Potential shaping is evaluated against the waypoint current at the start of each step, plus bonuses with drift = 0. Steps are reported with and without pretraining (§8.8). |
| 11 | SAC and wall-clock comparisons need measurement hygiene | LOW | accepted | Env steps are the primary axis. Load average is recorded, and wall-clock is shown only when the start load is below 4. "PID level" is defined. 3 seeds per algorithm (§2.3, §5.2). |
| 12 | Naming: "held-out" seeds are public; compare lacks seed range and CIs | LOW | accepted | Renamed to "calibration seeds". `compare --seeds` with paired CIs and JSON output (§0, §5.3). |

## wf_2fb93652-cd0 / security

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | controller.py shim breaks "ONNX is data, not code" | HIGH | accepted | Same fix as wf_3f0c2a9a-a51 security #1. A directory with both files is a LoadError rather than being silently treated as Python (§3.1). |
| 2 | No ONNX validation policy (only compatibility checks) | HIGH | accepted | Merged policy: IR 8–10, opset 17–21, ai.onnx allow-list, no external data, functions or subgraphs, and caps of 1 MiB, 64 nodes and 262,144 params (§3.4). |
| 3 | Default torch exporter fails or emits external data | HIGH | modified | torch.onnx is not used; the onnx.helper direct build replaces it, so no onnxscript pin is needed (§3.2). |
| 4 | SB3 checkpoints are pickles (code execution on load) | HIGH | accepted | Pickles serve only to resume training. Export and eval are pickle-free. `.gitignore` gains `runs/` and `*.pkl`. Pickle-family files under submissions/ are rejected (§2.4, §6). |
| 5 | Symlinks and non-regular files in submission directories are followed | MEDIUM | accepted | `read_regular_file` uses lstat, O_NOFOLLOW and size-before-read. Red-team tests cover a symlink and /dev/zero (§3.4, §6). The general loader fix is Phase 1 fix I. |
| 6 | ORT session options not hardened | MEDIUM | modified | Adopted, except ORT_DISABLE_ALL is used instead of ENABLE_BASIC: it is measured bitwise equal and is the smaller optimizer surface (§3.5). |
| 7 | Phase 3 runner requirements specific to ONNX unstated | MEDIUM | deferred | Listed in §10 for Phase 3: sandbox per policy, load limits, the per-act limit, the same path for private seeds, and pinning. |
| 8 | Provenance forgeable; becomes untrusted display text | MEDIUM | accepted | `claims` are self-reported, never used for ranking, and have a bounded `name` regex. Render escaping and a verified-training badge are deferred to Phase 3/5/11. |
| 9 | ORT version not recorded in results.json | MEDIUM | accepted | `Versions.onnxruntime` and `Versions.onnx` are added before topple.results/1 is published, and an ORT upgrade means a re-score (§3.6). The AVX2/AVX-512 check is deferred to Phase 3. |
| 10 | Provenance and train_meta can leak machine details | LOW | accepted | Whitelisted claims, run-relative paths, and a byte-scan test for `/Users/`, `/home/` and `site-packages` (§2.3, §6). |
| 11 | Name the 200-seed set; keep it disjoint from future private seeds | LOW | accepted | "Calibration seeds". Training seeds are in [2^52, 2^53), and Phase 3 private seeds come from [0, 2^52) disjoint from the public sets (§0). |

## wf_2fb93652-cd0 / deploy

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | "PPO beats PID" undefined; score alone on 20 seeds is noise | HIGH | accepted | Merged into gates G1–G4, measured on the shipped ONNX (§4). |
| 2 | Parity test can pass vacuously; reference and sampling unspecified | HIGH | modified | SB3 reference, open-loop rows, batch 1, and a torch-free CI re-check with the fixture were adopted. The pre-clip-mean assertion is replaced by requiring at least 300 unsaturated rows (§3.3). |
| 3 | Specified exporter fails in the locked env; two-file artifact leaks paths | HIGH | accepted | Direct onnx.helper build with the exact node sequence. Tests cover single file, no paths, the op set and reproducible bytes (§3.2, §6). |
| 4 | No IR pin; onnx 1.23 default IR 14 rejected by ORT 1.31 and ORT-web | HIGH | accepted | Explicit IR 8, and the validator allows at most IR 10 (§3.2, §3.4). The ORT-web Node smoke test is deferred to Phase 5. |
| 5 | Phase 2 code breaks CI (strict mypy, coverage gate, torch-free imports) | MEDIUM | accepted | A `python-train` CI job, mypy overrides, coverage omit plus the train job's own gate, lazy imports, and an import-hygiene test (§7). |
| 6 | Deterministic clip(mean) gives much more saturation than training | MEDIUM | modified | clip(mean) is kept and the cause is fixed: the 5M budget plus the action-excess penalty (rl run measured 2–4% saturation). log_std −1 is rejected (refuted by rl). G4 caps saturation at 0.10. |
| 7 | VecNormalize details underspecified (two epsilons, training=True, waypoint stats) | MEDIUM | accepted | The formula is written out verbatim with ε = 1e-8. Stats are read from live objects without updating them, and waypoint stats are frozen (§3.2, §2.4, §8.8). |
| 8 | ORT reads files through external-data references | MEDIUM | accepted | External data is banned, models load from bytes only, and a red-team test uses a `.env` reference (§3.4, §6). |
| 9 | ORT-web cannot read metadata_props; make policy.json the manifest | MEDIUM | modified | The manifest and JSON encodings were adopted. The byte-identical template controller.py is rejected: a policy directory has no .py at all (§3.1). |
| 10 | Export path PPO-only; SAC architecture too large for firmware | MEDIUM | accepted | SAC uses net_arch [64, 64] ReLU with a tanh head and the same harness path (§2.2, §3.2). |
| 11 | Fix the firmware C export format now (from ONNX weights) | LOW | deferred | Phase 8. The format is fixed now: read only policy.onnx, require the §3.2 sequence, use `float.hex()` literals and keep parity within 1e-5 (§10). |
| 12 | Best-checkpoint eval: score the ONNX artifact and budget the eval cost | LOW | accepted | Selection uses in-memory ONNX every 250k steps with a 4-worker pool (about 2 min per run). Final numbers come only from fresh evaluations (§2.4). |
| 13 | Gaps in reproducibility and provenance records | LOW | accepted | train_meta records versions, threads, load and peak RSS. The learner pins threads (§2.3). |
| 14 | Curriculum "small pushes" has no preset; set_preset rejects unknown presets | LOW | rejected | The curriculum is dropped: rl measured that standard-only training reached 0.88–0.965, and an ideal→clean→standard curriculum scored worse (0.67). A training-only `push_scale` is deferred to Phase 6 if a study needs it. |

## wf_bc1d1edf-2dc / rl

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | No PPO hyperparameters or sample budget; 1–1.5M steps shows the energy/saturation failure | BLOCKER | modified | The tested hyperparameters were adopted with the exact layout measured (8 envs × 2048) and a 5M budget, instead of 16 × 1024 with 6M: the shared-machine cap is 8 envs and runs must stay under 10 min. An energy check gates shipping (§2.2, §2.5). |
| 2 | Do not replace u² effort with a battery-energy cost | HIGH | accepted | RewardConfig is unchanged. Energy terms may only be added on top of u² (§2.2). |
| 3 | PPO mean unbounded; SB3 clips before env.step and hides noise | HIGH | accepted | `ActionExcessPenalty` (Box ±2, weight 0.1) is on by default, and `frac_mean_outside_1` is logged (§2.1, §2.5). |
| 4 | Lowering log_std_init makes things worse | MEDIUM | accepted | log_std_init is 0.0 and ent_coef is 0.0. A std below 0.03 before 2M steps is flagged (§2.2, §2.5). |
| 5 | Curriculum as written cannot be built and would hurt | MEDIUM | accepted | No curriculum; training uses standard from step 0, and PLAN Phase 2 is edited to match (§2.2). |
| 6 | 50-seed score-only checkpoint selection is noisy | MEDIUM | accepted | 200 selection seeds, every 250k steps, lexicographic (success, score), gated on PID's success measured at train start (§2.4). |
| 7 | "seeds seed+i" leaks held-out seeds 0..15 into training | MEDIUM | modified | `TrainingSeeds` draws every training episode seed from [2^52, 2^53), not 1e9 + 1000·seed + i. This is disjoint by construction for all episodes, not only the first (§2.1, §2.3). |
| 8 | One training seed is not enough evidence | MEDIUM | accepted | 3 seeds per algorithm, reported as mean ± sd. The shipped policy is chosen on the selection seeds (§2.5, G7). |
| 9 | SAC comparison lacks fairness rules | LOW | modified | Same harness, seeds and selection rule, and env steps to PID level. The budget is equal wall-clock at 480 s instead of 30 min, because runs must stay under 10 min. 8 envs with 2 gradient steps keeps the 0.25 updates-per-transition ratio (§2.2). |
| 10 | Drift cost saturates exactly at the 1 m success threshold | LOW | deferred | It was not the binding failure (median drift 0.19–0.31 m) and needs a 3-seed A/B test; Phase 6 autotune explores it (§10). |
| 11 | Spec's Phase 1 facts out of date | LOW | accepted | §0 was updated to LQR 0.87 / 80.25 / 3.31 J, PID 0.66 / 68.88 / 6.65 J and E_ref 13.3. Gates compare against PID measured in the same run. |

## PLAN.md changes (made with this rewrite)

- **Phase 2 section:**
  - PPO uses 8 envs, standard from step 0 and no curriculum, with 3 seeds × 5M steps.
  - The ONNX export is pickle-free.
  - Adds the data-only `topple.policy/1` format.
  - The done-when becomes G1–G6 in short form.
  - waypoint-v1 is split into Phase 2b with its own done-when.

  Driven by rl #1/#5, product #2/#6 and deploy (wf_2fb93652-cd0) #1.
- **§3 RL row:** "16 envs on 18 cores" becomes "8 envs" (shared-machine cap, §1).
- **§4 architecture:** adds `policy/` and `train/`. The trust model adds "ONNX submissions are data
  only".
- **§7 testing:** the ONNX parity bullet now matches §3.3, and the 2k-step smoke train runs in the
  `python-train` job.

## Re-critique round 1 (spec v2 → v3)

Lenses: rl, deploy, security, product, consistency. Several issues were raised by more than one
lens; each row still gets its own decision and points to the shared fix. Superseded v2 decisions:
rl #9 (SAC 480 s wall-clock budget → 1.0M env steps), deploy wf_3f0c2a9a-a51 #5 (50k-rollout stats
for new dims → analytic stats), product wf_3f0c2a9a-a51 #10 (γ-matched potential shaping →
undiscounted progress), wf_2fb93652-cd0 deploy #13 (`peak_rss_mb` → `peak_tree_rss_mib`).

### rl

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Waypoint shaping with γ = 0.99 pays for staying far from the gate | HIGH | accepted | Undiscounted progress `20·(d(s,w) − d(s′,w))/L`, terminal step uses the real distance, stated as deliberately not γ-matched; plus the product lens's measured −50 termination term. Return-scale sentence made explicit (`0.1·r_bal ≤ 0.1`/step, so ≤ 200 was right but unclear). Optional dock-hold term allowed as an untested iteration (§8.8). |
| 2 | extend_obs starts from a policy that falls after ~1.7 m of travel; frozen rollout stats worsen it | HIGH | accepted | Zero input columns 2, 4, 5 in both first layers (equivalent to holding them at their normalized mean 0, so no bias fold is needed), analytic stats for dims 10..14 (`final` mean 0.5 / var 0.25 from the consistency lens instead of 0 / 1), log_std reset to −1.0, fresh optimizer, offset test in test_surgery, non-stationarity stated in §8.4 (§8.8). |
| 3 | getrusage self+children misses ~82% of a run's memory | HIGH | accepted | psutil tree sampler (bytes on every platform, so no unit helper needed), `peak_tree_rss_mib`/`peak_procs`, G10 < 6,144 MiB, hard abort at 8,192 MiB with exit 4 (§1, §4). Sample period 1 s instead of 2 s (cheap at 16 processes). |
| 4 | SAC wall-clock cap irreproducible, includes eval time, different candidate count | MEDIUM | modified | Env-step budget, no wall-clock budget, 20 evals + final like PPO, qf [256,256], UTD 0.25 kept. Budget is 1.0M (not 1.2M): measured qf-256 throughput is 2,113 steps/s (this round), so 1.0M ≈ 475 s of training vs PPO's ~400 s; every_steps 50k gives 20 evals (§2.2). |
| 5 | SAC often has no eligible checkpoint; exit 3 breaks G7 | MEDIUM | accepted | best.json always written with `eligible`; train exits 0; `export --report-only` into `runs/…/report_policy/`; rows marked "below PID gate"; only shipping needs eligible (§2.4, §3.3, G7). |
| 6 | Eligibility ignores the G4 energy/saturation caps; E_ref is a literal | MEDIUM | accepted | Eligible = success ≥ PID ∧ median energy ≤ E_ref ∧ saturation ≤ 0.08; E_ref read from ScoreConfig everywhere and recorded (§0 preamble, §2.4). |
| 7 | G10 makes wall_clock < 600 s a MUST on a shared machine | MEDIUM | accepted | Wall-clock removed from G10; recorded with train_s, selection_s, cpu_s, load; 600 s soft target; hard watchdog (900 s balance, 1,200 s waypoint) that snapshots, evaluates and exits 4 (§1). |
| 8 | Final policy after the last update is never evaluated | LOW | accepted | `_on_training_end` snapshot + selection eval at the true `num_timesteps`; smoke test asserts it (§2.4, §6). |
| 9 | 2b budget still rising at 5M; waypoint ranking unspecified | LOW | accepted | Rank by (success, score) on waypoint selection seeds; 8M steps per arm with a 1,200 s hard limit and the soft target; per-eval gates/docked/fell/t_finish recorded. Eligibility gate (score > creeper) added from product #3 instead of "no gate" (§8.8). |
| 10 | Reward note mislabels frac_mean_outside_1 as saturation | LOW | accepted | Reworded and templated from logged `frac_mean_outside_1` (§5.2 item 6). |
| 11 | test_policy_build var 1e-12, \|μ\| 1e3 case cannot pass against an f64 reference | LOW | accepted | Rows are float32 everywhere; `reference_actions` uses the graph's f32 constants then f64 (consistency measured ≤ 8.7e-7 in every cell); grid kept (§2.4, §3.2, §6). |

### deploy

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Acceptance policy has no compute/memory bound (5 s per act possible) | HIGH | accepted | Merged with security #1: Gemm transA = 0, weights must be initializers/Constants, batch-major rule on every output, static cost ≤ 2^18 per row, smoke test with a per-call terminate timer, red-team cases; the 87 µs claim removed and the Phase 3 limit derived from a re-measurement (§3.4, §10). Cost cap 2^18 (not 2^20): first-party is ~5k, and with the element cap 131,072 it allows each weight about twice. |
| 2 | Canonical form incomplete; which sha256 identifies a policy is ambiguous | MEDIUM | accepted | Full canonical form, `canonicalize(data) == data` required at load, `policy_sha256 = sha256(policy.onnx)` used everywhere, `topple policy pack`, `action_rule: other`, tests (§3.1, §3.3, §3.4). `graph.name = "topple_policy"`. |
| 3 | ≥ 1,000 on-policy parity rows infeasible early; data flow unspecified | MEDIUM | accepted | `collect_onpolicy_rows` in the main process on seeds 2000.., up to 50 seeds; rule n ≥ 1,000 or 50 seeds with ≥ 100; obs stored float32 (§2.4, §3.3). |
| 4 | G10 peak_rss under-reports by the number of workers | MEDIUM | modified | Same fix as rl #3. Per-role peaks are not recorded (tree total, peak step and process count suffice for the gate); gate at 6,144 MiB with the 8,192 abort rather than gating at 8,192. |
| 5 | Torch-free done-when step uninstalls torch from the dev venv | MEDIUM | modified | Separate `UV_PROJECT_ENVIRONMENT=.venv-eval`, `--no-sync`, `find_spec` check, `.venv-eval/` gitignored (§4). The `versions.torch absent` check is dropped: `Versions` has no torch field; the gate checks `versions.onnxruntime` instead. |
| 6 | Nothing binds selected, exported, fixture and shipped bytes | LOW | accepted | `onnx_sha256` in meta.json and parity_ref; export asserts it; fixture checked against policy.json; `cmp` line in the done-when (§2.4, §3.3, §4). Build sets `producer_version = ""` so bytes do not depend on the topple version. |
| 7 | Spec text inconsistencies; forgeable parity record | LOW | accepted | train_meta.json in export's read set; PLAN §3 row fixed; parity moved to `claims.export_parity`; both sequences named (SAC is 10 nodes, not 9); C path dispatches on action_rule (§3.1–§3.4, PLAN). |
| 8 | O_NOFOLLOW missing on Windows | LOW | accepted | `getattr` flags, fstat identity check against lstat, cap+1 read, red-team tests skipped with reason on Windows (§3.4, §6). |
| 9 | Determinism key ignores CPU ISA; onnx unbounded | LOW | accepted | `Versions.cpu_isa`, determinism key (os, arch, cpu_isa, ORT), `onnx<1.24`, `onnxruntime<1.32` (§3.5, §3.6, §7). |

### security

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Acceptance bounds tensor sizes, not graph cost: compute/memory bombs | HIGH | accepted | Batch-major rule, static cost budget, timing never decides acceptance (recorded only; the 1 s terminate is a safety net), all listed red-team tests; Phase 3 per-act limit 5 ms provisional (§3.4, §6, §10). Smoke uses 32 calls and a 1 s per-call terminate instead of 20 ms / 1 s total, so load spikes cannot flip acceptance. |
| 2 | Phase 3 must re-run ONNX policies; uploads do not identify the policy | HIGH | deferred | Written into §10 as Phase 3 MUSTs: re-run from the PR head SHA, trusted identity binding, second-sandbox R2 gate, Arena hash check, no RLIMIT_AS, per-act limit from the batch-major worst case. Nothing in Phase 2 publishes scores. |
| 3 | Policy identity ambiguous across three hashes; README changes identity | MEDIUM | accepted | Canonical-only files, `policy_sha256`, `ControllerInfo.policy_sha256`, real-directory and top-level-regular-files rules, hidden files named in the error, README untrusted (§3.1, §3.6). "Resolved inside the checkout root" is deferred to the Phase 3 intake (local eval may use any path). |
| 4 | policy.json accepts NaN/Infinity and duplicate keys; parity self-reported; unbounded claims | MEDIUM | accepted | strict/allow_inf_nan=False config, duplicate-key and constant hooks, BOM rejected, `claims.export_parity`, bounded claim fields, tests (§3.1, §6). |
| 5 | Validator gaps: Constant INTS, non-finite attribute floats, unreachable element cap | MEDIUM | accepted | Per-op attribute allow-list, finite floats, single payload field, no `segment`, element cap 131,072 incl. Constants, tests that fail only on the intended check (§3.4, §6). |
| 6 | test_no_unsafe_loads does not enforce the stated rules; skips four modules | MEDIUM | accepted | Import bans outside train/, onnx/ORT entry-point allow-list, named call-site allow-list inside train/ (`learn.resume_from`), yaml bans, `safe_load_capped` with alias rejection and a 64 KiB cap (§2.1, §6). `getattr` evasion stated as out of scope. |
| 7 | 2b surgery needs model.zip and vecnormalize.pkl | MEDIUM | accepted | Snapshot writes `critic.npz` and `dist.npz`; `extend_obs` rebuilds `ActorCriticPolicy` via `load_state_dict` from npz; test with the pickles deleted (§2.4, §8.8, §6). |
| 8 | load_policy_dir(challenge=None) silently skips compatibility | LOW | accepted | `challenge` keyword-required for ONNX (TypeError otherwise), dims only from `task_spec`; `policy_from_bytes` also takes the challenge (§3.5, §3.6). |
| 9 | Attacker strings reach error text; canonical form keeps producer/node strings | LOW | modified | String limits run third, and names must match `^[A-Za-z0-9_.:/-]{1,128}$` so later errors cannot carry markup; canonical form fixes producer/domain/graph name and renames nodes `n{i}`; quoted strings via `repr()[:64]` (§3.4). The charset rule replaces escaping as the way the `<img>` test passes. |
| 10 | Export trusts best.json `dir`, uncapped npz, unvalidated fixture name | LOW | accepted | best.json has no `dir`; integer step; symlink and containment checks; zipfile member caps; NaN-rejecting JSON; `NAME` regex (§3.3, §6). |

### product

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Fine-tune arm starts from a policy that falls once wheel travel saturates; wrong tensor name | HIGH | accepted | Columns 4, 5 (and yaw, 2, per rl #2) zeroed; tensor name `mlp_extractor.value_net.0.weight`; behavioural check moved to the 2b gate (item 7) because the train-CI policy cannot balance; both arms on the same 8M budget (§8.8, §8.9). |
| 2 | Starting waypoint reward does not learn in 5M; undiscounted progress + fall penalty does | HIGH | accepted | Adopted the measured variant (γ_shape 1, −50 on termination). The penalty applies to every training termination incl. max_drift (drift is the same failure; truncation is not penalized). 3-seed ablation optional, reported if run (§8.8). |
| 3 | 2b done-when has no learned-policy floor | HIGH | accepted | Item 6 "Learnability": score > creeper, docked ≥ 60%, fell ≤ 20% on 0..199; selection eligibility = score > creeper; PLAN 2b done-when updated (§8.9, §8.8). |
| 4 | 50 public seeds justified with the clean-preset sd | MEDIUM | accepted | Rationale corrected (sd 5.1–23.3 on standard, ±1.4 to ±6.5 on 50 seeds), comparisons on 0..199 with paired bootstrap, Phase 3 ≥ 200 private seeds, 3000..3049 in the §0 disjointness list (§0, §8.2, §10). |
| 5 | Calibration band one-sided | MEDIUM | accepted | Two-sided band (falls 3–10%, docked ≥ 88%), ±0.05 N·s adjustment both ways, clean − standard gap recorded with a 1.5-point target (§8.7). |
| 6 | Hero GIF 640×320 ≤ 5 MB infeasible | MEDIUM | accepted | GIF 480×240 at 20 fps ≤ 6 MB (measured 4.96 MB), MP4 primary, gallery GIF 320×320 at 10 fps ≤ 2 MB, total ≤ 20 MB, script fails on caps (§5.2). |
| 7 | 2a bar is only "beats PID"; add non-inferiority to LQR | MEDIUM | accepted | G2b (paired CI lower bound > −2.0, success ≥ LQR − 0.03); LQR-vs-PPO median clip; stretch kept (§4, §5.2). PLAN done-when updated. |
| 8 | Registry refactor misses trajectory obs width and the protocol alias | MEDIUM | accepted | `load_npz` width from the registry, forgery test, permanent alias `wp_eval_v1` (§8.1, §8.2, §8.9). |
| 9 | Waypoint video unspecified | MEDIUM | accepted | Content, seed, trim, framing, overlays and caps specified (§8.9 item 8). |
| 10 | Speed term "25 for speed" pays 8 to the driver | LOW | accepted | Option (b): par-time normalization with t_par = 7.0 s, measured to keep the driver's speed optimum at 0.30 m/s; anchors re-stated; battery note in phase2.md (§8.6, §8.7). |
| 11 | Event timing and post-dock observation underspecified | LOW | accepted | Hold counting, row ranges, shared `waypoint_events` with `math.hypot`, post-dock obs, non-finishers = T, gates-only and reckless anchors, anchors measured on 0..199 (§8.4–§8.7). |
| 12 | Wall-clock learning-curve panel usually empty | LOW | accepted | `cpu_s` recorded per eval and plotted; wall-clock only as a footnote (§1, §5.2). `cpu_s` comes from the tree sampler because RUSAGE_CHILDREN misses live workers. |

### consistency

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | G10 memory gate measures the wrong thing; wall-clock half load-dependent | HIGH | modified | Same fix as rl #3 / rl #7. The 570 s truncate-and-flag guard is replaced by a hard limit (900 s / 1,200 s) that snapshots, evaluates and exits 4; G10 fails on any abort flag. The sampler test spawns 2 × 200 MB children as proposed. |
| 2 | Smoke test cannot produce best.json/export; on-policy rows unattainable; selection API cannot supply them | HIGH | accepted | Smoke configs with `vec_env: dummy`, `gate: none`, every_steps 1024; rows collected in the main process (deploy #3) rather than returned by pool workers; `checkpoint_every` dropped (snapshots only at selection evals); Monitor wrapper (§2.2, §2.4). |
| 3 | Done-when not fully measurable; compare schema undefined; G10 evidence gitignored | HIGH | accepted | `phase2_gate.py` checks G1–G10; run summaries committed in `phase2.json`; `topple.compare/1` defined; mean saturation = unweighted per-episode mean; `phase2b_gate.py` for every §8.9 item (§4, §5, §8.9). |
| 4 | Parity rows must be float32-representable; degenerate variance | MEDIUM | modified | Float32 rows and the builder warning adopted; the warning is printed, not stored in policy.json (no schema field for it). The 2b variance floor is superseded by analytic stats (§2.4, §3.2, §8.8). |
| 5 | Surgery names a non-existent tensor (value_net.0.weight) | MEDIUM | accepted | Names fixed, copy-by-name for the rest, value prediction asserted in test_surgery (§8.8, §6). |
| 6 | SAC wall-clock budget unreproducible; never-eligible SAC unreportable | MEDIUM | modified | Same fix as rl #4 / rl #5: 1.0M steps as proposed here; eval time excluded and recorded as `selection_s`; a single best.json with `eligible` instead of a separate `best_any.json`. |
| 7 | Validator p99 latency bound makes acceptance non-deterministic | MEDIUM | accepted | Static cost bound decides; p99 recorded as a warning; the 1 s terminate is a safety net only (§3.4). |
| 8 | Selection ignores energy/saturation caps; no shipping fallback | MEDIUM | modified | Same eligibility as rl #6 but with saturation 0.08 (margin) instead of 0.10; shipping fallback to the next run in selection order, every attempt reported (§2.4, §2.5). |
| 9 | 2b registry misses balance-only code paths; global ENV_ALGO_VERSION couples challenges | MEDIUM | accepted | `kind` + score model per kind, `PRESETS ∪ preset_defs`, resolved preset in protocol_hash, `make_scenario(course=)`, obs width, `build_results`, video, per-kind ENV_ALGO_VERSION (balance 1), golden tests (§8.1). |
| 10 | Waypoint edge cases (Φ at termination, undocked t_finish, sd claim) | LOW | modified | Real distance at the terminal step; the fall term is −50 (product's measured value), not −10; non-finishers count as T; sd rewording per product #4 (§8.2, §8.6, §8.8). |
| 11 | Manifest, hash and loader ambiguities | LOW | modified | sha = canonical = file (canonical required, so one hash); metadata JSON-quoted and compared after `json.loads`; metadata stays required (pack adds it); action_rule checked only for §3.2 graphs, else `other`; read set lists best.json and train_meta.json; checks 1–7 → canonicalize → smoke; `.DS_Store` rejected by name and gitignored (§3.1–§3.4). |
| 12 | Residual inconsistencies across PLAN, contract, decision log, constants | LOW | modified | PLAN §3 row fixed; `strict_shape_type_inference=1` added to §3.5 (verified accepted by ORT 1.31.0); 13.3 J literals replaced by ScoreConfig; private-seed rule extended. The contract §6 example is not edited here (outside this change's files): it is a required 2b deliverable (§8.9 item 4). |

### PLAN.md changes (round 1)

- §3 "Policy format" row: pickle-free `onnx.helper` build (IR 8, opset 17) with VecNormalize and
  clip baked in, canonical bytes (deploy #7, consistency #12).
- Phase 2 section: SAC on an env-step budget; eligibility includes energy and saturation; tree-RSS
  limit; done-when (2a) adds LQR non-inferiority (product #7); done-when (2b) adds learnability
  (product #3) and the two-sided calibration band (product #5).

## Re-critique round 2 (spec v3 → v4)

Lenses: rl (9), deploy (11), security (8), product (10), consistency (16): 54 issues. Several were
raised by more than one lens; each row has its own decision and names the shared fix.

Superseded earlier decisions:
- round 1 rl #7 and consistency #1 (900 s / 1,200 s watchdog; G10 fails on any abort flag) → the
  wall-clock limit is a runaway guard (2,400 s / 3,600 s), `max_cpu_s` is the budget, and aborted
  runs are re-run, never reported (§1, G10).
- round 1 rl #4 ("20 evals + final, like PPO") → both algorithms get 19 periodic + final = 20.
- round 1 rl #9 and round 0 rl #6 (8M waypoint steps; lexicographic (success, score)) → 6M and
  score-first rank keys per kind (§2.4, §8.8).
- round 1 product #3 (fell ≤ 20%) → learnability relative to the driver (§8.9 item 6).
- round 1 product #5 (±0.05 N·s adjustment loop) → deterministic impulse sweep (§8.7).
- round 1 product #10 (t_par 7 s) → t_par 5 s (§8.6).
- round 1 deploy #3 (`collect_onpolicy_rows`, 50-seed fallback) → 5 strided rows per selection
  episode (§2.4).
- round 1 deploy #2, consistency #11 and security #9 (blacklist canonical form; checks 1–7 then
  canonicalize; `repr(s)[:64]`) → field allow-list, constructive canonical form, short-circuit
  order, `safe()` (§3.4).
- round 1 product #7 (G2b lower bound > −2.0; stretch "excludes 0") → G2b lower bound > 0.
- round 0 wf_2fb93652-cd0 security #2 ("262,144 params") → 131,072 elements + static cost 2^18
  (superseded since v3; listed here per consistency #15).
- round 0 wf_3f0c2a9a-a51 product #11 ("wall-clock shown only when start load < 4") →
  wall-clock is a footnote only (superseded since v3; listed here per consistency #15).

### rl

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | G10 makes wall-clock a gate after all via the 900/1,200 s abort | HIGH | accepted | `max_wall_clock_s` 2,400 (balance) / 3,600 (waypoint) as a runaway guard; load-independent `max_cpu_s` (3× the estimated cost: 12,000 / 15,000) with `cpu_abort`; G10 gates `rss_abort`, `cpu_abort`, `wall_clock_abort` and `num_timesteps ≥ total_timesteps` on every *reported* run; a wall-clock-aborted run is re-run with the same seed (≤ 2 re-runs), logged in `attempts[]`, never shipped or reported (§1, §2.2, G10). |
| 2 | Spec's SAC config never measured for learning; never reaches PID | MEDIUM | accepted | §0 SAC row replaced with the post-fix measurement of the shipped config; target_entropy −4 and UTD 1 recorded as tested and worse; "learner-throughput-limited (UTD 0.25, serial updates)"; item 7 fully templated (no "peaked early"); CPU-s reported for both; the 30-min fair SAC study deferred to Phase 13 (§0, §2.2, §5.2, §10). |
| 3 | Snapshot step directories collide in smoke configs | MEDIUM | accepted | Shared fix with deploy #1 / consistency #2: PPO evaluates in `_on_rollout_start` (deploy's choice over `_on_rollout_end`: the label then equals the steps the policy trained on), SAC in `_on_step` only while `num_timesteps < total`; final always in `_on_training_end`; `snapshot()` raises on an existing dir; unique-label assertion; smoke test asserts exactly 2 distinct dirs (§2.4, §6). |
| 4 | 2b "fell ≤ 20%" has a 3-point margin and calibration raises the impulse | MEDIUM | accepted | Shared fix with product #1: calibration first, then one pilot PPO seed at the frozen impulse; relative gate (docked ≥ 60%, PPO − driver CI lo > −5, > creeper, fell ≤ driver + 0.20, not +0.15: at 0.35 N·s the measured gap is 16 pp, so +0.15 would fail a policy that is score-equal to the driver); the two pre-planned reward iterations are dock-hold then −100; PPO median energy vs E_ref reported (§8.7–§8.9). |
| 5 | Fine-tune copies a value head 10× off the waypoint reward scale | LOW | accepted | `value_net.weight` and `.bias` multiplied by `balance_scale` (0.1); test asserts `V_15 == 0.1·V_10` within 1e-6 (§8.8 step 3, §6). |
| 6 | Lexicographic (success, score) never better than score-first | LOW | accepted | Balance `rank_key = (round(score, 2), success_rate)` among eligible (eligibility already enforces success ≥ PID); waypoint uses its leaderboard key (product #2) (§2.4). |
| 7 | Waypoint 8M budget rests on old-reward data | LOW | accepted | 6M with every_steps 500k (11 periodic + final); 8M only if the calibrated pilot's final selection score beats its best at ≤ 4M by > 2 points, recorded with evidence (§8.8). |
| 8 | config_sha256 under --seed, Monitor placement, PLAN risk row | LOW | accepted | (a) hash of the effective config with `seed` inside, plus `config_file_sha256`; (b) order kept, `topple/train/excess_penalty_mean` logged and the exclusion stated; (c) PLAN §8 risk row rewritten (§2.1, §2.3, §2.5, PLAN). |
| 9 | G2b non-inferiority bar far below what PPO measures | LOW | accepted | G2b becomes "paired CI lower bound of PPO − LQR > 0" for the shipped policy (measured +5.55 [3.97, 7.31] post-fix), success ≥ LQR − 0.03 kept; the stretch line is removed because it is now the gate (G2b, PLAN). |

### deploy

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Final snapshot reuses the last periodic label; immutable rename fails | HIGH | accepted | Trigger rules (a), (b), (d) adopted as written; (c) "skip the final if parameters unchanged" is unnecessary because the rules make a duplicate label impossible, and the assertion catches a regression. `eval_labels(cfg)` pure-function test covers smoke, PPO 5M (19 + final) and SAC 1M (19 + final) (§2.4, §6). |
| 2 | policy.json parse (json.loads + strict model_validate) rejects every manifest | MEDIUM | accepted | Shared with consistency #1: rejection-only `json.loads` pre-pass, then `model_validate_json`; metadata compared after normalizing both sides through JSON; `check_against_model` binds IR, opset and I/O; manifest test round-trips bytes on disk (§3.1, §3.2, §6). |
| 3 | Blacklist canonicalization keeps unbounded attacker strings | MEDIUM | accepted | Shared with security #2: check 2 field allow-list (reject at load), constructive canonical rebuild (drop in pack mode), values renamed `v{i}` and initializers `c{j}`; one red-team test per field; identity defined as byte identity with function-level near-copy detection left to the Phase 3 fingerprint (§3.1, §3.4, §6). |
| 4 | On-policy parity rows come from one episode | MEDIUM | accepted | 5 strided rows per selection episode from the selection eval itself (1,000 rows over 200 seeds, no extra simulation); `collect_onpolicy_rows` and the 50-seed rule dropped; `n_onpolicy_seeds` stored; `--report-only` records `vacuous: true` instead of failing; also applies to 2b (§2.4, §3.3). |
| 5 | Pool hand-off unspecified; sessions cannot be pickled | MEDIUM | accepted | `topple.pool.PolicyRef`: parent validates once and ships canonical bytes + sha (ONNX) or path + submission sha (Python); the worker re-checks before its first episode; nothing holding a session is pickled; `test_pool` mutates files after load (§1, §3.5, §6). |
| 6 | pack must normalize names/I/O/batch before checks; torch rationale wrong | LOW | accepted | Shared with consistency #12: pack = check 1 → pack-mode canonicalize (drops foreign content, renames I/O, static batch 1 → `batch`, removes Identity, rejects external data first) → checks 1–9 → smoke; §3.2 rationale reworded to "the default dynamo exporter needs onnxscript"; legacy-exporter recipe documented; `test_pack_torch_legacy` in the train job (§3.2, §3.3, §6). |
| 7 | Smoke terminate timer must use fresh RunOptions and cancel | LOW | accepted | Fresh `RunOptions` per call, `Timer.cancel()` in `finally`, fixed messages; `ONNXPolicy.act` uses none (§3.4). |
| 8 | Small export/runtime inconsistencies | LOW | accepted | (1) moot: on-policy rows now come from the selection eval; (2) `W0..b2` keys for both algorithms, mapping table in §8.8; (3) `--out` must not exist unless `--replace` (temp sibling, rename aside, swap); done-when uses `mktemp -d` (consistency #5 named the flag `--replace`, deploy `--force`; `--replace` chosen); (4) IR/opset tied: (8, 17–18), (9, 19–20), (10, 21) (§3.3, §3.4, §4). |
| 9 | Headline numbers are Mac-ORT; ranked numbers will be Linux-ORT | LOW | accepted | `platform`, `cpu_isa`, `onnxruntime` on every eval row; scoreboard caption; `test_shipped_policy` uploads its `results.json` as the CI reference cited in `phase2.md`; no test pins an ONNX episode score (§3.5, §5.1, §5.2, §7). |
| 10 | Firmware and web consumers have no readable fixture or on-target step | LOW | accepted | `--test-fixture` also writes `NAME_parity.bin` (u32 N, u32 D, f32 obs, f32 a_ref) and a JSON header; Phase 8 done-when gains on-device parity ≤ 1e-5, act ≤ 1 ms at 240 MHz, and host parity with and without `-ffp-contract=off`; Phase 5 reads the `.bin` (§3.3, §10). |
| 11 | Pin the paired-bootstrap algorithm | LOW | accepted | Normative `paired_bootstrap` (one `(B, n)` index draw, linear quantiles); the gate recomputes from per-seed arrays and asserts equality with compare's rows; known-answer test (§4, §5.3, §6). |

### security

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Variadic Max/Min bypasses the static cost cap (89 ms per act) | HIGH | accepted | Per-op input-count limits (Min/Max 2..4, not 1..4: a 1-input Max is an identity), model-wide `Σ inputs ≤ 192`, non-Gemm cost `max(1, n_inputs − 1)·dim1`; red-team `variadic_max_245k` (check 6) and a 40-node 4-input Max at width 4096 (check 8 only; the suggested ×64 would trip the input-sum cap first); "terminate is checked only between kernels" stated; bench on ubuntu x86 incl. denormals deferred to Phase 3 (§3.4, §6, §10). |
| 2 | Canonical form is a blacklist; ~1 MiB of attacker data survives | MEDIUM | accepted | Shared with deploy #3. Identity and Flatten removed from the op list (no legitimate use under the rank-2 rule; pack removes Identity); unused initializers and dead nodes rejected; attribute types checked against the schema; `ai.onnx` normalized to `""` (§3.4). |
| 3 | No short-circuit rule; make_session not capability-gated; cwd external data | MEDIUM | accepted | Normative short-circuit order (no C++ checker, inference, numpy_helper or ORT before checks 1–7 pass); canonicalize reads raw fields only; `ValidatedPolicy` constructible only in validate.py and required by `make_session`; `policy_from_bytes` validates fully; external-data red team with `.env` + FIFO and monkeypatched raising entry points; ban list extended (§3.4, §3.5, §6). |
| 4 | Phase 3 runner duties miss PR-head code, budget, pickle boundary, ownership | MEDIUM | deferred | Four MUSTs (i)–(iv) written into §10 for Phase 3 as recommended (no PR code in the trusted stage, 60 s policy-CPU budget, JSON/npz-only boundary, per-author ownership). Phase 2 publishes no scores. |
| 5 | Attacker strings reach error text via pydantic and unbounded op_type/domain | LOW | accepted | `safe()` charset filter; pydantic errors render `loc` + `type` only; onnx/ORT texts become fixed codes (raw only with `--debug`); op_type/attribute names bounded in check 4; `<` tests extended (§3.4, §6). |
| 6 | Manifest fields not bound to the model; claim strings unrestricted | LOW | accepted | `check_against_model` (IR, opset, I/O) at load; every claims string `^[\x20-\x7e]{0,64}$`; the CLI never prints README (§3.1). |
| 7 | safe_load_capped / loader error mapping: RecursionError, ValueError, hard links | LOW | accepted | Event-stream depth limit 32 before composing; RecursionError/ValueError/UnicodeDecodeError mapped to ConfigError/LoadError; `st_nlink == 1`; tests for 8,000-deep YAML, 5,000-digit int, hard link (§2.1, §3.1, §3.4, §6). |
| 8 | test_no_unsafe_loads scope and ban list too narrow | LOW | accepted | Scans `python/topple`, `scripts/`, `tests/`; adds the path/config loaders; `add_session_config_entry` only with the three §3.5 literal keys in `make_session` (the recommendation said four; §3.5 has three entries) (§6). |

### product

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | 2b learnability gate is a coin flip once calibration raises impulse_hi | HIGH | accepted | Shared with rl #4: relative gate (a) docked ≥ 60%, (b) PPO − driver CI lo > −5.0, (c) > creeper, plus (d) fell ≤ driver + 0.20; checked after calibration; `fall_cause` metric (push in the preceding 1.0 s) reported in item 6 and the failure gallery; reward iterations bounded at two, then an owner decision; PLAN 2b done-when updated (§8.5, §8.8, §8.9, PLAN). |
| 2 | Waypoint selection ranks by (success, score), contradicting the leaderboard | MEDIUM | accepted | Waypoint `rank_key = (round(score, 2), success_rate, −median_t_finish)`; eligibility = score > creeper ∧ docked ≥ 0.60 ∧ paired score vs driver on the selection seeds lo > −5; driver measured at train start; `rank_key` per kind in §2.4 (§2.1, §2.4, §8.8). |
| 3 | Dock progress cap is non-monotonic | MEDIUM | accepted | Option (b): `p_dock = 0.8·clip((L − d_min)/(L − r), 0, 1) + 0.2·max_hold_steps/100`; `max_hold_steps` added to the events; scorer test of the ordering (hover 0.101 ≤ entered/hold 0 ≤ hold 50 ≤ hold 99 < docked) to be written with the scorer tests (§8.5, §8.6). |
| 4 | t_par = 7 s leaves no speed headroom above the first PPO | MEDIUM | accepted | t_par 5.0 s, wording "full marks at a 5 s finish"; driver optimum stays 0.30 m/s; anchors re-measured at freeze; no value below 5 s without a feasibility probe (§8.6, §8.7). |
| 5 | Calibration rule lands on a band edge of a steep fall curve | MEDIUM | accepted | Deterministic sweep {0.30..0.50} minimising \|falls − 6%\| subject to docked ≥ 88%, ties lower (0.35 today); full curve recorded; order impulse_hi → E_ref → anchors → pilot PPO → templated numbers; sd sentence templated (§8.2, §8.7). |
| 6 | 2a done-when lets one lucky seed carry the claim | MEDIUM | modified | G11 (≥ 2 of 3 PPO runs eligible and passing G2/G3/G4 fresh on 0..199) is a **gate**, not only a reported line: the recipe is the product claim and is expected to pass 3/3 at no cost. Scoreboard shows each seed and "k of 3"; PLAN done-when updated (§4, §5.2, PLAN). |
| 7 | SAC vs PPO write-up will teach the wrong lesson | LOW | accepted | Three labelled comparisons (equal env steps, equal CPU-s, full budget); UTD 0.25 sentence; row label "SAC (1.0M steps, UTD 0.25)". The premise (SAC ahead at equal steps) came from the pre-fix [64,64]-critic pilot; rl #2's post-fix data may show the opposite, so every claim is templated (§5.2 item 7). |
| 8 | Shove test wastes half its levels; forward only | LOW | accepted | 0.30:0.65:0.025 (15 levels), both directions, i50 per direction and the minimum; LQR quoted as 0.45 (shipped discrete LQR) and all quoted values refreshed from `phase2.json` (§5.2 item 3, §5.3). |
| 9 | GIF caps, ambiguous waypoint seed, frame ignores the robots | LOW | accepted | Deterministic fps/pane ladders with settings recorded; lower median = index 24 of (score, seed); frame = course ∪ both trajectories + 0.5 m, clamped [3.4, 5.0] m; hero caption template (§5.2, §8.9 item 8). |
| 10 | Spec numbers and definitions to refresh before freezing | LOW | accepted | Anchors defined exactly (Creeper = `Driver(0.12, 0.8)`, Reckless = `Driver(0.60, 0.8)`; gates-only, balance-in-place and spin given normative definitions); `d_min` uses the reach-check rows; prototype numbers are labelled pilot and templated from `phase2.json`; energy sentence added (§8.5–§8.9). |

### consistency

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | policy.json parse path rejects every valid manifest | HIGH | accepted | Same fix as deploy #2, plus lone-surrogate rejection in the pre-pass (§3.1). |
| 2 | Final snapshot collides with the last periodic eval | MEDIUM | accepted | Same fix as deploy #1 / rl #3; the SAC comment now reads "19 periodic + final = 20 evals" and PPO is also 20 (§2.2, §2.4). |
| 3 | Wall-clock watchdog plus G10 re-creates a load-dependent gate | MEDIUM | accepted | Same fix as rl #1 (stated in §1, §2.2, G10 and the superseded list above). |
| 4 | preset_defs names collide with the global PRESETS dict | MEDIUM | accepted | `Challenge.preset(name)` with every lookup routed through it; AST test forbidding `PRESETS[` outside disturbances.py/challenge.py; 2b item 3 checks waypoint push windows and trusted == env scenarios (§8.1, §8.9). |
| 5 | 2a done-when not fully checkable; fallback and fixture underspecified | MEDIUM | accepted | `scripts/phase2_done.sh` with `set -euo pipefail`; `phase2_gate.py --submission --results-dir`; G9 via `gh run list` (`headSha == HEAD`, success) in `phase2_ci_check.py`; `phase2_results.py` runs `export --test-fixture ppo_balance --replace`; fallback only on G1–G4 failures from its own fresh evals; `--out`/`--replace` defined (§2.5, §3.3, §4, §5.1). |
| 6 | compare/1 pairs only against the first entry | MEDIUM | accepted | Paired rows for every ordered pair (i > j); gates read `paired[ppo, pid]` / `paired[ppo, lqr]` and recompute with `paired_bootstrap` (§4, §5.3). |
| 7 | 2b done-when has no commands; several items ambiguous | MEDIUM | accepted | `scripts/phase2b_done.sh` command block in `.venv-eval`; `submissions/ppo_waypoint/`; anchors as `submissions/anchors/*.py`; item 7 wrapper and `surgery.sanity_check`; `best.json` in `extend_obs`'s read set; W ↔ state-dict table; `drift` used consistently (§8.7–§8.9). |
| 8 | Cost red-team case trips the element cap first | LOW | accepted | One 256×256 weight reused by 5 Gemms fails only the cost check, 3 reuses pass; the inclusive boundary is pinned on `static_cost` directly (2^18 accepted, 2^18 + 1 rejected), because a 4-reuse chain plus input/output layers already exceeds 2^18 (§3.4, §6). |
| 9 | Templated reward-design numbers have no source | LOW | accepted | Both options: `configs/train/ablate_nopenalty.yaml` (seed 1, 1M, penalty off) supplies x; energy-cost and log_std results are `pilot` constants with a `source` field rendered "pilot measurement"; missing fields fail the script (§2.2, §5.1, §5.2). |
| 10 | SAC config is a diff with undefined merge semantics | LOW | accepted | `sac_balance.yaml` written in full with `log_std_init: -3.0` (SB3 default; the round-2 measurement used SB3 defaults for unspecified keys, re-measured in 2a); discriminated `action_excess`; `health` PPO-only, SAC logs `std_mean` from `actor.action_dist`; `config_sha256` over the effective config (§2.1, §2.2, §2.5). |
| 11 | train_meta does not bind the env/protocol version | LOW | accepted | `selection_protocol_hash`, `env_algo_version`, `bundle_hash` in train_meta and phase2.json; gate fails on a mismatch; 2a training starts after the Phase 1 fix merge (preamble, §2.3, G10). |
| 12 | pack runs checks 1–7 before canonicalize | LOW | accepted | Same fix as deploy #6; fixtures with foreign metadata and a `;` value name (§3.3, §6). |
| 13 | test_resources sensitive to lazy pages and MB/MiB | LOW | accepted | Children hold `np.ones(200 * 2**20, np.uint8)`; assert ≥ 380 MiB and `peak_procs ≥ 3` (§6). |
| 14 | 2b contract/API change list incomplete | LOW | accepted | Item 4 lists §1, §4, §6, §9, §10, §12a, §13, §14, §16; `controller_obs(obs_arr, t, *, schema=...)` keeps the 2-arg call; `load_npz(path, *, max_steps, obs_dim)` from the trusted protocol only; library-upgrade sentence for §13 (§8.1, §8.9). |
| 15 | Cross-document drift: phase-3.md, stale rows, round-1 source not committed | LOW | modified | §10 now requires phase-3.md to drop the controller.py shim, RLIMIT_AS for ONNX, the 0.25 s ONNX act limit and the fixed 88 B frame (this designer may edit only phase-2.md, the decision log and PLAN.md, so phase-3.md itself is not edited); round-0 security #2 and product #11 added to the superseded list. Committing the round-1/round-2 critique JSON is left to the orchestrator: the texts were supplied inline and this change may not create new files. |
| 16 | Canonical bytes not pinned across onnx/protobuf; on-policy rows from one episode | LOW | accepted | `protobuf>=7.36,<8` in base deps; golden sha tests for a built fixture and a canonicalized non-canonical fixture; on-policy rows per deploy #4 (5 strided rows from each of 200 selection episodes rather than the suggested 50 seeds × every 20th step: same coverage goal, no extra simulation) (§2.4, §6, §7). |

### PLAN.md changes (round 2)

- Phase 2 / 2a: SAC described as learner-throughput-limited; CPU-seconds budget, wall-clock as a
  runaway guard, aborted runs re-run (rl #1, rl #2); done-when "beats LQR" (lower bound > 0,
  rl #9) and "≥ 2 of 3 seeds pass the PID gates" (product #6).
- Phase 2 / 2b done-when: calibration by a fixed sweep first; learnability relative to the driver
  (product #1, rl #4, product #5).
- §8 risk table: "curriculum" replaced by "5M budget, 3 seeds, 200-seed selection, action-excess
  penalty" (rl #8c).
