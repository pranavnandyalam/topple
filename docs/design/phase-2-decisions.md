# Phase 2 decisions: one row per critique issue

Source: `docs/design/phase-2-critiques.json`. It holds 6 critiques with 68 issues in total. Every
issue appears below exactly once. The re-critique of v2 (round 1, 54 issues) is decided in
"Re-critique round 1", and the re-critique of v3 (round 2, 54 issues) in "Re-critique round 2", and
the re-critique of v4 (round 3, 52 issues) in "Re-critique round 3". The sync with the merged
Phase 1 (v5.1), its adversarial verification (v5.2, 18 findings) and the Linux recalibration
(v5.3) follow at the end. Later rounds
supersede earlier rows where they conflict; each superseded row is named in the later round's
preamble. The round-1 and round-2 critique texts were supplied inline to the designer and are not
yet committed as JSON (see round 2, consistency #15, and round 3, consistency #16).

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

## Re-critique round 3 (spec v4 → v5)

52 issues (rl 10, deploy 9, security 8, product 8, consistency 17). Every issue has a row below.
Several were raised by more than one lens; each row names the shared fix. All three HIGH issues
(rl #1, security #1, product #1) are adopted. Only edits to `phase-2.md`, this log and
`PLAN.md` were made; the round-1/round-2 critique JSON files still have to be committed by the
orchestrator (consistency #16).

Superseded earlier decisions:
- round 2 rl #1 (`max_cpu_s` 12,000 / 15,000 from a 3,600 CPU-s estimate) → measured: balance
  PPO 4,500, SAC 6,000, waypoint 7,500 until the pilot's `cpu_s` is measured, then 3× it; `cpu_s`
  is a per-process ledger read before `venv.close()` (§1).
- round 2 product #6 ("expected to pass 3/3 at no cost") → with the constant LR only 2 of 3
  post-fix seeds passed G2b; the linear schedule measured 3/3 (§0, §2.2).
- round 2 rl #4 and product #1 (waypoint eligibility without a fall term; "better arm") →
  eligibility mirrors all four item-6 sub-gates; shipping across all 6 runs with fallback;
  "better arm" is a reporting term (§8.8).
- round 2 rl #7 (8M if the final beats the best at ≤ 4M by > 2 points) → paired bootstrap of
  late vs mid evals, lower bound > 0 (§8.8).
- round 2 product #5 (sweep {0.30..0.50} step 0.05) → 0.300:0.500:0.025 with an in-band
  constraint, on seeds 0..399 (§8.7).
- round 1 product #1 / round 2 product #1 (`fall_cause` window 1.0 s, `None`) → 1.5 s, string
  `"none"` (§8.5).
- round 2 deploy #8 (4) (IR/opset pairs (8, 17–18), (9, 19–20), (10, 21)) → `CANONICAL_IR` map;
  load requires `ir == CANONICAL_IR[opset]`, pack sets it (§3.4).
- round 2 deploy #6 (legacy `dynamo=False` as the documented recipe; `test_pack_torch_legacy`
  gating) → dynamo is primary, legacy secondary, committed torch-free fixtures, the torch test is
  advisory (§3.3, §6).
- round 2 deploy #10 (`.bin` = u32 N, u32 D, f32 obs, f32 a_ref) → `.bin` layout v1 with magic,
  version, A and the row-set byte (§3.3).
- round 2 consistency #10 (SAC `log_std_init: -3.0`) → removed; forbidden for SAC (SB3 SAC
  ignores it without SDE) (§2.1, §2.2).
- round 2 consistency #8 (256×256 weight reused by 5 Gemms; boundary 2^18) → 2 reuses fail, 1
  passes; boundary 2^17 under the weighted cost (§3.4, §6).
- round 2 security #1 / round 1 (per-act kill provisional 5 ms) → CPU-time limit or wall ≥ 50 ms,
  one retry, aggregate 60 s CPU budget (§10).
- round 1 deploy #6 ("Build sets producer_version = \"\"") → `producer_version` is never set;
  canonical form sets only `producer_name = "topple"`; test asserts `not HasField` (§3.2, §6).
- round 1 consistency #2 ("rows collected in the main process") → superseded since v4 by round 2
  deploy #4 (5 strided rows per selection episode from the workers); listed per consistency #16.
- round 0 rl #1 ("runs must stay under 10 min") → 600 s soft target, reported only, plus the
  runaway guard (§1); listed per consistency #16.
- round 0 rl #8 ("reported as mean ± sd") → v4 had dropped it; v5 restores it as the "PPO recipe
  (3 seeds)" row (round 3 rl #9).
- Determinism "best-effort: recorded, not gated" (v4 §2.3) → required and tested (round 3 rl #3).

### rl

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Constant LR 3e-4 collapses 1 of 3 post-fix seeds late in training; linear decay fixes it | HIGH | accepted | `learning_rate: {schedule: linear, initial: 3.0e-4, final: 0.0}` with `lr(p) = 3e-4·max(p, 0)` (clamp because p reaches −0.0027); inherited by both waypoint arms, restarted for a fine-tune; §0 PPO row replaced by the 3-seed constant vs linear table (mean 85.07 sd 3.2 vs 86.14 sd 0.72); `late_regression` health flag (drop ≥ 0.10 below running best after 2M); `lr_fn` validated in `test_train_config`; PLAN 2a updated (§0, §2.1, §2.2, §2.5, §6). |
| 2 | Shipping fallback covers only G1–G4, but G2b is the gate a degraded seed actually fails | MEDIUM | accepted | Shared with consistency #10: `SHIP_GATES = (G1, G2, G2b, G3, G4, G5, G6)`; G11 table and scoreboard get a reported, non-gating G2b column ("j of 3 at LQR level") (§2.5, §4, §5.2). |
| 3 | Training is bit-deterministic, but the spec makes determinism best-effort | MEDIUM | accepted | §2.3 now requires determinism per (config, seed, platform, start_method); `start_method: forkserver` pinned in the configs; `mp_start_method` and `torch.get_num_threads()` in train_meta; `test_train_smoke` runs each smoke config twice and compares npz shas, `onnx_sha256` per label and `selection_history`; a `wall_clock_abort` retry must reproduce the aborted attempt's `onnx_sha256` at every shared label, else flag `nondeterministic` and not reportable (G10) (§1, §2.2, §2.3, §4, §6). |
| 4 | cpu_s misses forkserver-reaped env workers; max_cpu_s is ~8× the measured cost | MEDIUM | accepted | Shared with consistency #1: monotone per-process ledger keyed by (pid, create_time), no rusage; final reading in `_on_training_end` before `venv.close()`; forkserver `test_resources` case; caps from measurement (balance 4,500 = 3 × 1,450; waypoint 7,500 until the pilot's `cpu_s` is known, then 3×); SAC 6,000 from an all-busy upper bound because its CPU was not measured; §0 estimate replaced by the measurement (§0, §1, §6). |
| 5 | 2b: waypoint eligibility ignores the fall gate, 'better arm' is undefined, no shipping fallback | MEDIUM | modified | Shared with product #1. Adopted: docked ≥ 0.65 (margin), a fall term, shipping across all 6 runs by `rank_key` with fallback on items 6/9 and every attempt reported, "better arm" a reporting term (higher mean fresh score). Changed: the fall term is `fell ≤ driver_fell + 0.20` (product's value), not + 0.15: the pilot's measured gap is 16 pp, so + 0.15 would make a score-equal pilot policy ineligible and stop 2b; chance failures of (d) are handled by the 6-run fallback and the 0..399 calibration set (§8.8). |
| 6 | Fine-tune arm: source run per seed unspecified; comparison metric hides where fine-tuning helps | MEDIUM | accepted | Fine-tune seed k starts from `runs/ppo_s{k}`'s eligible, abort-free best, else the next index (k+1, k+2 mod 3); `init_source {run, step, onnx_sha256}` in train_meta and phase2.json; headline = env steps to first eligible selection eval (median of 3, with and without the 5M pretraining); final fresh score and fell as mean ± sd. This overrides consistency #17's "shipped K" variant, which adds best-of-3 selection bias (§8.8). |
| 7 | frac_mean_outside_1 is undefined; the obvious implementation reads a zeroed buffer | LOW | accepted | Defined on the selection eval's on-policy rows (deterministic, pre-clip mean via `vecnorm.normalize_obs`, fraction of 2·N entries with |mean| > 1), computed in the selection callback for both the ablation and `ppo_s1`; a separately named `topple/policy/frac_rollout_mean_outside_1` from `_on_rollout_end` (§2.5, §5.2). |
| 8 | SAC log_std_init is a no-op, SAC std_mean undefined, TrainingSeeds yields np.int64 | LOW | accepted | (a) `log_std_init` removed from SAC and forbidden by the per-algo schema; (b) SAC `std_mean` = mean of `exp(clamp(actor.log_std(latent_pi(z)), −20, 2))` over the on-policy rows, PPO `exp(log_std).mean()`; (c) `int(rng.integers(2**52))` and an `isinstance(seed, int)` test (§2.1, §2.2, §2.5, §6). |
| 9 | No recipe-level aggregate row; the shipped best-of-3 is shown as 'PPO' | LOW | accepted | "PPO recipe (3 seeds)" row with mean ± sd and min; shipped row labelled "PPO (shipped: best of 3 by selection seeds)"; README footnote with recipe mean and min (§5.2). |
| 10 | 2b budget-extension rule can almost never trigger | LOW | accepted | `paired_bootstrap(mean of the evals nearest 5.5M and the final, mean of the evals nearest 3.5M and 4.0M)` on the stored per-seed scores (`selection_scores.npz`), extend to 8M if lo > 0, CI recorded (§2.3, §8.8). |

### deploy

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | IR/opset table rejects dynamo's IR 10/opset 20; the documented legacy recipe is being removed | MEDIUM | accepted | `CANONICAL_IR = {17: 8, 18: 8, 19: 9, 20: 9, 21: 10}`; pack sets IR, load requires the map; committed torch-free fixtures `thirdparty_legacy.onnx` and `thirdparty_dynamo.onnx` packed in the default job; `test_pack_torch_legacy` advisory; dynamo recipe primary (§3.3, §3.4, §6). |
| 2 | Structure match compares op types only, not shapes, attributes or Constant nodes | MEDIUM | accepted | Canonical form folds Constants into initializers and drops schema-default attributes; structure match = op types + attributes + exact initializer shapes + head bounds bitwise ∓1 + symmetric normalization clip; check 7 rank 0..2; first-party uses rank-0 Clip bounds; pack tests for the dynamo fixture with scalar vs `[D]` scale (§3.2, §3.4, §6). |
| 3 | Parity cannot catch small constant errors; no saturated-row coverage; negative controls unnamed; head bounds unsourced | MEDIUM | accepted | Bitwise constant binding in export step 3 (head bounds from `challenge.task_spec`, never `model.action_space`); `n_saturated ≥ 32` (rows with max |a_ref| ≥ 0.999); seven named negative controls; edge set defined. The edge set uses consistency #17's 6·D + 4 form (64 rows for D = 10, 94 for D = 15) instead of 4·D + 4, a superset that adds ±1e6 per feature (§2.4, §3.3, §6). |
| 4 | G2/G2b/G3/G4 evaluated only on macOS; Linux CI checks only the eval_v1 ordering | MEDIUM | accepted | CI `python` job runs the 0..199 compare and uploads it; `phase2_results.py --fetch-ci` copies it (bound to the shipped `policy_sha256` and current `protocol_hash`) into `docs/results/phase2/ci_ubuntu/`; `phase2_gate.py` evaluates G2–G4 on both; `phase2.md` shows both blocks; PLAN done-when updated (§4, §5.2, §7, PLAN). |
| 5 | ORT-web version (1.20) stale; browser check deferred although the artifact is frozen | LOW | modified | Accepted: pin `onnxruntime-web@1.30.0`, WASM EP, optimizations disabled, 1 thread; rationale reworded ("IR ≤ 10 keeps ORT-web ≥ 1.20 compatible"). The Node parity test stays in Phase 5 because this branch has no `web/` toolchain, but it becomes Phase 5's first done-when line, run against the frozen `.bin` v1 fixtures (§3.4, §10). |
| 6 | Non-finite observations give different results per runtime | LOW | accepted | Contract §17: non-finite observations are outside the policy domain; harness asserts finite obs (`test_harness_finite_obs`); §3.5 limits the NaN → `invalid_action` statement to NaN-propagating runtimes; Phase 8 firmware fails safe on non-finite obs/actions (§3.5, §6, §8.9 item 4, §10). |
| 7 | The .bin parity fixture has no magic, version, action dim or row-set flags | LOW | accepted | Layout v1 (`TPAR`, version, N, D, A, obs, a_ref, set, padded), sidecar with `a`, `tolerance`, `n_unsaturated`, `n_saturated`, `policy_sha256`; readers check magic/version and D/A (§3.3, §6). |
| 8 | 2b shipped policy has no named parity fixture and no CI re-verification | LOW | accepted | `--test-fixture ppo_waypoint`; `test_shipped_policy` parametrized over both policies; `phase2b_gate.py --fixture` checks the G5 binding; item 9 and item 10 (§8.8, §8.9, §6). |
| 9 | The C path meets the Phase 8 bar; a cheap host-C parity check could be added now | LOW | accepted | Default-job `test_c_reference`: header from initializers by position, ~40-line reference MLP compiled with the system cc with and without `-ffp-contract=off`, `.bin` rows within 1e-5; CI sets `TOPPLE_REQUIRE_CC=1` (§6, §10). |

### security

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | Phase 3 MUST runs untrusted ONNX inside the privileged workflow_run job | HIGH | accepted | §10 (i) rewritten: `rescore` job (`permissions: {}`, no secrets, `persist-credentials: false`, workspace not mounted, fetch by immutable `head_sha`, ≤ 64 KiB JSON + npz out) and a separate `publish` job holding the token that never parses ONNX; nightly salt only in a token-less host process, empty sandbox env, rotated per season; red-team item for `GITHUB_TOKEN`/salt visibility; PLAN CI row reworded. `phase-3.md` itself is not edited here (outside this change's files) (§10, PLAN). |
| 2 | Static cost not weighted by op type: an accepted Softplus chain is 230× first-party | MEDIUM | accepted | `OP_WEIGHT` (1 / Tanh, Sigmoid 2 / Elu 4 / Softplus 48), `MAX_COST = 2^17` (first-party ≈ 5,200, 25× headroom; measured worst case ≤ 8.4 s per 400k acts); red-team `softplus_chain_w4096_x52`, `elu_chain_w4096_x52`; per-act bound as CPU time or ≥ 50 ms wall with one retry; `bench_policy_caps.py` ≤ 20 s on ubuntu x86 incl. subnormals; `set_denormal_as_zero` never in-process; Arena per-act budget and no R2 upload after `timeout` (§3.4, §6, §10). |
| 3 | Check 2 is unimplementable on the pinned protobuf and its allow-list is incomplete | MEDIUM | accepted | Shared with consistency #2: `UnknownFieldSet(msg)` on every visited message; `TensorShapeProto {dim}` added; `TensorProto.name` absent inside `AttributeProto.t` and regex-checked everywhere; round-1 deploy #6 superseded; tests for unknown field 99 on three message types, `producer_version=""`, 1 KiB Constant name, exact build field set (§3.2, §3.4, §6). |
| 4 | Validator crashes on invalid UTF-8 and protobuf decode errors | MEDIUM | accepted | Check 1 maps `DecodeError` → `onnx_parse_failed`; check 2 requires `str` values that encode as UTF-8 → `invalid_utf8`, before any regex; `validate_policy_bytes` / `read_manifest` / `load_policy_dir` map DecodeError, ValueError, TypeError, UnicodeError, RecursionError, MemoryError to fixed codes; npz headers parsed and size-checked before `np.load` in export and Phase 1 `load_npz`; tests for each case (§3.3, §3.4, §6). |
| 5 | Phase 3 ownership keyed on GitHub handle: collisions, renames, copies | MEDIUM | modified | Accepted for Phase 3 (§10 (iv)): `submissions/community/<lowercased login>/`, rows keyed on numeric `user.id`, PRs limited to their own directory, exact-sha `duplicate`, reserved `/official\|topple/i` names shown as quoted self-reported text. Changed: first-party entries stay where they are (outside the community tree) instead of moving to `submissions/official/`, which avoids churning Phase 1/2 paths while still preventing shadowing (§10). |
| 6 | Python regexes anchored with ^...$ accept a trailing newline | LOW | accepted | All Python-side checks use `re.fullmatch` with unanchored patterns; tests for trailing `\n` in a node name, metadata key and op_type (§3.4, §6). |
| 7 | Extra topple.* metadata keys carry ~9 KiB of unrestricted text into identity bytes | LOW | accepted | Exactly the 7 required keys, each `json.dumps` of the manifest field, ASCII; pack drops all others; check 4 enforces it; tests for an extra `topple.note` and non-ASCII values (§3.2, §3.4, §6). |
| 8 | read_regular_file can block on a FIFO swap; resume trusts shared pickles | LOW | accepted | `O_NONBLOCK \| O_NOCTTY \| O_CLOEXEC` added, `S_ISREG(fstat)` required; FIFO swap red-team with a 5 s timeout; `resume_from` refuses files not owned by `os.getuid()` and prints that resume executes pickles (§2.1, §3.4, §6). |

### product

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | 2b learnability gate still ~50% per pilot-quality run; selection ignores (d); no fallback; 'better arm' undefined | HIGH | accepted | Shared with rl #5: eligibility adds `fell ≤ driver_fell + 0.20` with the driver's fall rate measured at train start; 6-run shipping rule with fallback and per-run item-6 table; "better arm" removed as a selection term; all 2b statistics on calibration seeds 0..399 (added to §0 and the Phase 3 disjointness rule); the pilot sentence is templated, the literal dropped; PLAN 2b done-when updated (§0, §8.2, §8.8, §8.9, PLAN). |
| 2 | Reference driver is underspecified | MEDIUM | accepted | Full normative control law in §8.7 (direction/reverse rule, k_v 1.2 speed law, slew limit, x_ref clamp, LQR balance terms, turn law, output); anchors reuse it; `test_driver_golden` pins clean 0..4 scores to 1e-9 and standard 0..19 docked/fell; note that k_v 2.0 is about 1.2 points better (§6, §8.7). |
| 3 | The learned policy's characteristic failure (falling at launch on clean) is invisible | MEDIUM | accepted | Item 6(e) reports clean-preset falls for PPO and driver; standard falls split by `fall_cause`; waypoint failure gallery; score parts as metrics and compare extras with an anchors-plus-PPO table; templated exchange-rate sentence. The optional soft gate on clean falls is rejected: it would add a gate on a metric not in the score (§5.3, §8.6, §8.9). |
| 4 | Waypoint video: robot unreadable, MP4 undefined, trim ambiguous, dock shape wrong | MEDIUM | accepted | Fixed oblique camera; MP4 960×960 at 50 fps; GIF 480×480 at 10 fps (ladder); 0.08 m marker disc + heading arrow per robot; gates and dock as `gate_radius_m` discs; push arrows; trim rule with per-robot `e_i`, docked robots keep playing, fallen robots hold with a stamp after the 1.0 s epilogue (§8.9 item 8). |
| 5 | impulse_hi sweep grid too coarse; lands one seed above the band floor | LOW | accepted | 0.300:0.500:0.025, in-band constraint (falls ∈ [3%, 10%], docked ≥ 88%), ties lower, owner decision if none qualifies, Wilson CIs; selects 0.375 on the pilot data (§8.7). |
| 6 | fall_cause window misses late push falls; None is not a legal metrics value | LOW | accepted | Shared with consistency #11: 1.5 s window; `"push" \| "self" \| "none"`; added to the contract metrics vocabulary (§8.5, §8.9 item 4). |
| 7 | Gates-only anchor 'never holds' is false under pushes | LOW | accepted | Reworded ("docks only incidentally"); anchors table records docked % and median `max_hold`; requirement unchanged (§8.7). |
| 8 | GIF size ladders have no defined final rung | LOW | accepted | Hero, gallery and waypoint ladders spelled out to the last rung (three × 0.9 pane steps), rung recorded, fail only after the last (§5.2). |

### consistency

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| 1 | cpu_s formula drops all SubprocVecEnv worker CPU once envs close | MEDIUM | accepted | Same fix as rl #4: ledger keyed by (pid, create_time), final read before close, monotonicity asserted in `test_resources`, forkserver case; start method pinned to `forkserver` in the configs (§1, §2.2, §6). |
| 2 | Check 2 relies on UnknownFields(), which raises on protobuf 7.36 | MEDIUM | accepted | Same fix as security #3, plus `fd.is_repeated` (no `fd.label` in protobuf 7) and `DecodeError` → `onnx_parse_failed` at check 1 (§3.4, §6). |
| 3 | Default CI job's strict mypy fails on train modules subclassing Any | MEDIUM | accepted | Option (a): mypy override `disallow_subclassing_any = false` for the five torch modules only; `.coveragerc-train` `include` = exactly those five files, so the two 80% scopes are disjoint (§7). |
| 4 | Results MP4s are gitignored, so G8 passes on uncommitted files | MEDIUM | accepted | `!docs/results/**/*.mp4`; G8 requires `git ls-files --error-unmatch` and `git diff --quiet -- docs/results`; `test_repo_hygiene` asserts `git check-ignore` returns non-zero for both results MP4s (§4, §6, §7). |
| 5 | G10 cannot be evaluated from the gate's inputs | MEDIUM | accepted | §5.1 lists the committed per-run summary exhaustively (incl. `num_timesteps`, `total_timesteps`, `selection_protocol_hash`, `env_algo_version`, `bundle_hash`, best `onnx_sha256`, `git_commit`, `git_dirty`); the gate recomputes the protocol hash and env version and checks every reported run; shipped run must have `git_dirty: false`; tests per missing field (§2.3, §4, §5.1, §6). |
| 6 | Abort / re-run flow is undefined | MEDIUM | accepted | `topple train` exits 2 on a non-empty `--out`; `--retry` renames a `wall_clock_abort` attempt to `<out>.attempt<k>`; `phase2_results.py` globs attempts into `attempts[]`; candidates exclude abort flags; export without `--report-only` refuses them; tests (§1, §2.5, §3.3, §6). |
| 7 | TrainConfig permits policy shapes the §3.2 builder cannot express | MEDIUM | accepted | `policy` is part of the per-algo union: PPO `tanh` with `pi`/`vf` of exactly 2 ints in 1..256; SAC `relu` with `pi`/`qf`; rejections tested. The parametric widening of §3.2 is not adopted (not needed for any shipped config) (§2.1, §6). |
| 8 | 2b done-when not measurable from its script; nothing binds 2b runs to the frozen challenge | MEDIUM | accepted | `phase2b_done.sh` written in full: clean tree, `.venv-eval` + torch check, named pytest files for items 2/3/9, re-export `cmp`, `phase2_ci_check.py` for item 10, gate with `--submission --fixture --results-dir`; new item 11 binds every reported 2b run to the current protocol hash and waypoint env version; contract item checked by grep; item 7 read from `phase2.json` (§8.9). |
| 9 | Waypoint versioning: scorer constants may be unhashed; ENV_ALGO_VERSION vs §13 bump unclear | MEDIUM | accepted | `WaypointScore` holds every scorer constant (golden-pinned dump); versioning rule before / at / after the freeze stated in §8.1 and listed for contract §13 (§8.1, §8.9 item 4). |
| 10 | Shipping fallback set ('G1–G4') may exclude G2b | MEDIUM | accepted | Same fix as rl #2: named `SHIP_GATES` and `PASS_GATES` constants in `scripts/phase2_common.py`, used by both scripts and the gate test (§2.5, §4, §6). |
| 11 | fall_cause = None and undocked t_finish_s do not fit the results schema; "MUST NOT change" too strong | LOW | accepted | `"none"` string; `t_finish_s = episode_seconds` for non-finishers; §8.1 reworded to protect `protocol_hash`, scores, success and existing metric values while allowing new keys; golden test pins existing keys (§8.1, §8.5, §8.6). |
| 12 | Derived configs are still diffs with undefined merge semantics | LOW | accepted | "Every shipped YAML is a complete file; no inheritance"; `test_train_config` asserts each derived file differs from its base in exactly the listed key paths (§2.2, §8.8, §6). |
| 13 | phase2_done.sh / G9 assume an open PR and do not require a clean tree | LOW | accepted | Both: `git status --porcelain` check first, and `ci.yml` also triggers on `push` to `phase-*`; `phase2_ci_check.py` gives distinct messages for no run / in progress / cancelled / failure (§4, §7). |
| 14 | phase2_results.py is not idempotent | LOW | accepted | `--report-only` implies replace of `run_dir/report_policy/` only; test runs it twice (§3.3, §6). |
| 15 | The CI-reference citation is circular, and artifacts expire | LOW | accepted | Citation bound to the policy: newest successful run whose `results.json` has the shipped `policy_sha256` (and current protocol hash), copied to `docs/results/phase2/ci_ubuntu/` in a follow-up commit; G8 checks existence and binding (merged with deploy #4) (§4, §5.2). |
| 16 | Decision-log rows contradicting v4 not superseded; round-1/2 critiques not committed | LOW | modified | The four rows (and every round-3 supersession) are in the list above; `test_policy_build` asserts `not HasField("producer_version")` and the allow-listed field set. Committing `phase-2-critiques-r1.json` / `-r2.json` remains open: this change may edit only `phase-2.md`, this log and `PLAN.md`, so it is handed to the orchestrator. |
| 17 | Remaining builder ambiguities in selection, parity rows and 2b init | LOW | modified | T = 0 → 0 rows, `n_onpolicy_seeds` counts T ≥ 1 (repeats allowed for short episodes); edge set 6·D + 4 (per feature {μ ± 10σ, μ ± 10.001σ, ±1e6}, the per-feature 0 dropped so D = 10 gives 64); `std_mean` defined per algorithm; `EvalProtocol.is_named`; `test_surgery` reworded to z = 0. Changed: the fine-tune source is per-seed (rl #6), not the shipped K (§2.4, §2.5, §5.3, §6, §8.8). |

### PLAN.md changes (round 3)

- Phase 2 / 2a: linear LR decay and tested bit-determinism (rl #1, rl #3); done-when gains "the
  0..199 gates hold on both the developer Mac and the ubuntu CI runtime" (deploy #4).
- Phase 2 / 2b done-when: calibration seeds 0..399; the shipped policy is the top eligible
  checkpoint across both arms' 6 runs with fallback; clean-preset falls reported (product #1,
  rl #5, product #3).
- §4 stack table, CI row: untrusted bytes parsed only in a token-less, secret-less `rescore`
  job; `publish` reads only its validated output (security #1).

## Sync with merged Phase 1 (spec v5 → v5.1)

Phase 1 merged as PR #3 (main `f997394`) after v5 was written; its fix rounds changed names,
numbers, hashes, APIs and contract sections that v5 cites. This round syncs `phase-2.md` with the
merged code and `docs/contract.md`; it is not a critique round, so each row is a change, its source
in the merged code or contract, and the spec section it lands in. Baselines were re-measured for
this sync on darwin-arm64 with the merged code (`evaluate`, PID and LQR on `balance-v1/standard`
seeds 0..199, 2000..2199 and 1000..1019). Only `phase-2.md`, this log and the new
`docs/design/deferred-risks.md` were edited.

Superseded earlier decisions:
- round 0 rl #11 (§0 numbers PID 0.66 / 68.88 / 6.65 J, LQR 0.87 / 80.25 / 3.31 J, E_ref 13.3) →
  S1, S2.
- round 0 wf_3f0c2a9a-a51 product #4 ("`_MAX_PUSHES` is untouched") → S3: the constant is the
  public `MAX_PUSHES`, still 3.
- round 1 product #8 (`load_npz` obs width from the registry) and round 2 consistency #14
  (`load_npz(path, *, max_steps, obs_dim)`) → S16: `topple.traj/2` with a trusted `layout`.
- round 3 security #4 (npz headers parsed and size-checked before `np.load` in export and in
  Phase 1's `load_npz`) → S6: Phase 1 already reads every member by hand; export reuses that reader.
- round 3 deploy #6 ("the harness asserts finite observations") → S9: a `RuntimeError`, not an
  `assert`.
- round 3 deploy #4 (ubuntu compare in the `python` job) and consistency #13 (G9 "both jobs") →
  S17: `python-slow` runs it; G9 covers three jobs.
- round 3 consistency #8 (`phase2b_done.sh` syncs `.venv-eval` with `--no-dev`, then runs pytest)
  → S19.
- round 3 consistency #17 (`EvalProtocol.is_named`) → S12.

| id | What changed | Source (code or contract) | Spec section |
|---|---|---|---|
| S1 | E_ref is 13.0 J (2 × PID's median 6.49 J), not 13.3; the energy bars of eligibility and G4 read it from `ScoreConfig`, so they move with it. | `challenges/balance-v1/challenge.yaml` (`e_ref_j: 13.0`); contract §12; `tests/test_challenge.py` (`STANDARD_PAYLOAD`) | preamble, §0, §9 |
| S2 | §0 baselines re-measured on the merged env: PID 68.70 / 0.645 / 6.49 J (0..199), 70.63 / 0.655 (2000..2199), 73.05 / 0.75 (eval_v1); LQR 80.06 / 0.865 / 3.26 J, 82.51 / 0.90, 85.42 / 0.95; survival equals success; mean saturation PID 0.074, LQR 0.046. PPO/SAC numbers relabelled as fix-branch pilots (E_ref 13.3). Every gate bar is relative to PID, LQR or E_ref measured in the same run, so no bar is re-derived by hand; only E_ref's value changes (S1). | measured for this sync; `challenge.yaml` comment (LQR 86.5%, PID 64.5% survival) | §0, §2.2 |
| S3 | `_MAX_PUSHES` is the public `topple.disturbances.MAX_PUSHES` (3, fixed consumption of the PUSHES stream); the waypoint `standard` preset is a `PushConfig` using existing fields only, so `Preset` dumps and balance-v1 hashes keep their keys. | `disturbances.py` (`MAX_PUSHES`, `PushConfig`, `_draw_pushes`) | §8.1, §8.2 |
| S4 | Course draws follow the portable rule (one `Generator.random()` double, affine map in Python), never `Generator.uniform`; `Scenario.course` stores the draws `{h0, turns}`, and points are derived per platform with scalar `math.cos`/`math.sin`; acceptance compares `x² + y² ≤ 1.5²`. | contract §9; `disturbances._uniforms`; commit `f78d84b` | §8.3 |
| S5 | Known-answer hashes cited: bundle `a3b48463…7f5017a9`, protocol hashes per preset, selection (2000..2199) `148b7b6b…dc9e9ea6`, calibration (0..199) `c7b0a50c…0a4156c1`, `ENV_ALGO_VERSION` 1. Phase 1 already pins per-preset protocol hashes and the standard payload text (with the `ScoreConfig` dump); the 2b golden list and file names follow (`tests/test_golden.py`, `tests/test_challenge.py`; there is no `test_golden_balance.py`). | `tests/test_golden.py`, `tests/test_challenge.py`; computed with the merged code | §2.3, §8.1, §8.9 |
| S6 | Export reads checkpoint npz files through Phase 1's trajectory reader parts (`topple._zip.Archive`, the hand-parsed npy header), moved into `topple/_npz.py`; no `zipfile`, `np.lib.format` or `np.load`. The note "the same header rule applies to Phase 1 `load_npz`" is dropped: it already holds. | contract §14; `_zip.py`, `traj.py` | §3.3, §6 |
| S7 | Policy directories follow the loader's rules: each file read once and `submission_sha256` computed from those bytes (the hash covers what runs), exact names as `os.scandir` returns them, a bounded listing, non-UTF-8 names refused, `OSError` → `LoadError`; `load_submission(path, *, challenge=None)` extends Phase 1's `load_submission(path)`. | contract §8; `loader.py` | §3.1, §3.5, §3.6, §6 |
| S8 | Recorded termination reasons: `timeout` only for survivors, `error` for an episode the harness stopped. Waypoint event rows, `fall_cause`, `fell` (fall reasons only) and success (`survived`) use them; compare `termination[]` holds them; the hero pane plays an epilogue after a fall or an error. | contract §11, §12, §14; `BalanceEnv._recorded_end`, `evaluate.FALL_REASONS`, `scoring.SURVIVED` | §5.2, §5.3, §8.5, §8.6 |
| S9 | The control loop is bounded by `max_steps` and raises `RuntimeError` for a harness bug; the finite-observation check does the same instead of an `assert`. Training wrappers never change `max_steps` or truncation. | contract §8; `evaluate._control_loop` | §2.1, §3.5, §6 |
| S10 | `results.json` is strict: the new `Versions` and `ControllerInfo` fields are optional and bounded; every gate input is read like `read_results` (regular file, ≤ 16 MiB, exact JSON types, `format` required); compare JSON gets the same model rules; waypoint metrics include `steps`, `termination`, `survived`, `time_upright_s` (needed by `summarize` and the complete-run check) and `rms_tilt_rad`, 18 in all (≤ 32). | contract §15; `results.py` (`read_results`, `summarize`, `MAX_METRICS`) | §3.6, §4, §5.3, §8.6 |
| S11 | The trusted API includes `verify_with_seed` and `verify_run` → `RunVerdict` (`failed`, `ok`): both are in the 2b dispatch list, and 2b item 3 requires `verify_run(...).failed == 0`. §10 notes that the Phase 3 job named `rescore` re-runs policies, unlike `topple.verify.rescore`. | contract §16; `verify.py`, `cli.py` | preamble, §8.1, §8.9, §10 |
| S12 | `EvalProtocol.is_named` is dropped: `--seeds` replaces a named protocol's seeds with `dataclasses.replace` (as `topple run --seed` does; ≤ 10,000 seeds), compare JSON's `complete` is defined, Phase 2 pools carry public protocols only, and §10 states the private salt range. | contract §8, §15; `evaluate.EvalProtocol` (salt rules, redacted `repr`), `cli.run` | §2.3, §3.5, §5.3, §10 |
| S13 | Waypoint robot and scene against the robot rules: the robot is byte-identical (passes `validate_robot`); the scene is balance-v1's except `offheight="960"` for the 960×960 video, with no other world geom or site, so gates and the dock are drawn by the renderer; course parameters reach `TaskSpec.extras` as floats. | contract §3, §8; `robot.validate_robot`, `build_model`, `video._playback`, `contract.TaskSpec` | §8.2 |
| S14 | New CLI outputs follow the output rules: `cli._check_output` for relative paths (never through a symlinked directory), atomic writes (`_atomic.replacement`, `write_results`), one-line escaped errors (`cli._fail`); export's `--out` swap uses `tempfile.mkdtemp`. | contract §14, §16; `_atomic.py`, `cli.py` | §3.3, §5.3, §6 |
| S15 | Inputs: `read_regular_file` sits in `topple/_inputs.py` beside `open_regular` (stricter: no symlinks or hard links); `load_train_config` reads with `open_regular`. | `_inputs.py`; contract §14, §15 | §2.1, §3.4 |
| S16 | F41: `topple.traj/2` for challenges whose obs schema is not `balancer.obs.v1` (see below). | contract §14; `traj.py` (`_SCHEMA["obs"]` fixed at 10 columns) | §8.1a, §6, §8.9, §10 |
| S17 | CI and packaging as merged: jobs `python` (`not slow`, coverage) and `python-slow` (`slow`); `test_shipped_policy` is slow, so `python-slow` runs the ubuntu compare and uploads; G9 covers `python`, `python-slow` and `python-train`; markers `render`, `slow` plus `train`; coverage config in pyproject; existing mypy overrides, deps and `.gitignore` entries named. | `.github/workflows/ci.yml`, `pyproject.toml`, `.gitignore` | §4, §6, §7 |
| S18 | Video limits: rendered sizes are multiples of 16 px (`MACRO_BLOCK`) within the scene's offscreen buffer; GIF ladder rungs resample the composed frame. | `video.py` | §5.2, §8.2 |
| S19 | `phase2b_done.sh` syncs `.venv-eval` with the dev group (pytest lives there; `--no-dev` dropped it), still without extras, so still torch-free. | `pyproject.toml` (`[dependency-groups] dev`, `default-groups`) | §8.9 |
| S20 | Dims and action bounds come from the challenge's `TaskSpec` (`Challenge.task_spec(wheel_radius_m, track_width_m)` also wants robot geometry, which does not change them); the driver's gains are named (`controllers.lqr.DEFAULT_GAINS`). | `challenge.py`, `contract.ActionSpec`, `controllers/lqr.py` | §3.2, §3.3, §3.5, §8.7 |
| S21 | Phase 1 is merged: the "train after the fix merges" gate, the `phase-1-sim-core/…` path and "fix I/J" references are replaced by the merged code, contract and tests; residual Phase 1 risks go to `docs/design/deferred-risks.md`, linked from §10. | main `f997394` | preamble, §3.1, §9, §10 |
| S22 | Trusted hooks and runner codes named: the shove test's pushes are injected with the trusted `reset(options={"pushes": …})` hook (a test hook that skips the scenario bounds but not the push-schedule check), and a Phase 3 run that exhausts its CPU budget ends its remaining episodes through the proxy's `RunnerError` (`timeout`). Added in the sync without a row (verification finding V18). | contract §9, §15; `envs/balance.py` (`_RESET_OPTIONS`, `check_push_schedule`), `evaluate.RunnerError` | §5.2, §10 |

### F41 decision: `topple.traj/2`

`topple.traj/1` fixes `obs` at 10 columns (contract §14; `traj._SCHEMA`), so waypoint-v1's
15-feature observation cannot be recorded in it. Options: (1) widen `/1` in place (it is
unpublished), (2) a new format for non-balance challenges, (3) store extra features in a separate
member. Decision: (2). `/1` stays byte-for-byte as merged and is the only format for challenges
whose `obs_schema` is `balancer.obs.v1`, so balance-v1 records, readers, goldens and the
canonical-layout red-team tests are untouched. `topple.traj/2` equals `/1` except `meta.format`
and an `obs` width D = `len(TaskSpec.obs_features)` (1 ≤ D ≤ 64), with `meta.obs_features` naming
exactly D features. The reader takes the width from the trusted protocol (`layout` =
`protocol_traj_layout(protocol)`; the default `TRAJ_V1` fails closed), checks the `obs` header
before reading data, and checks `meta.format` and the feature count; verification's step 5 still
compares `obs` and `obs_schema`/`obs_features` with the replay. (1) was rejected: it changes a
layout that the merged contract, reader and red-team tests pin at 10 columns, and a format id is
how the contract versions a layout change (contract §1); a width read from `meta` would also let
an upload choose its own limit, so the trusted width is needed either way. (3) was rejected
because it changes the member set, which every zip-layout check pins. Specified normatively in
`phase-2.md` §8.1a with its tests; the contract text (§1, a new §14a, §16) is a 2b deliverable
(§8.9 item 4).

### Round-3 HIGH issues re-checked after the sync

- **rl #1 (linear LR).** Still resolved: §2.2's config keeps `learning_rate: {schedule: linear,
  initial: 3.0e-4, final: 0.0}` with the clamp, §2.1 `lr_fn`, the `test_train_config` clamp
  test and the `late_regression` flag are unchanged. The sync labels §0's constant-vs-linear table
  as a fix-branch pilot (E_ref 13.3); the decision rests on late collapse under a constant LR, not
  on E_ref, and the LQR it was compared with moved only from 80.14 to 80.06 on 0..199.
- **security #1 (Phase 3 `rescore` job).** Still resolved: §10 (i) keeps `permissions: {}`, no
  secrets, `persist-credentials: false`, base-branch code only, fetch by `head_sha`, and a separate
  `publish` job that never parses ONNX. The sync only names the trajectory formats it may emit,
  makes the boundary (iii) read them with `load_npz` and the protocol's layout, and states the
  private salt range; none of this widens what the token-holding job sees.
- **product #1 (2b learnability).** Still resolved: §8.8 eligibility (score > creeper, docked ≥
  0.65, `fell ≤ driver_fell + 0.20`, paired lower bound > −5), the 6-run shipping fallback and
  item 6 on calibration seeds 0..399 are unchanged. The sync defines `fell` as the fall reasons
  only (`fell`, `floor_contact`), which the fall term depends on, and success through `survived`;
  an errored episode still scores 0 and cannot pass the docked or score gates.

## Sync verification (spec v5.1 → v5.2)

An adversarial verifier checked v5.1 (commit `aa0767f`) against the merged code (main `f997394`)
and reported 18 findings: 1 HIGH, 7 MEDIUM, 10 LOW. Each finding was re-checked against the code
before it was decided, with prototypes and probes on scratch copies of main (darwin-arm64,
nothing written to the repository). All 18 are confirmed; none is rejected. V10 is modified: its
claim that `--workers 4` oversubscribes the runner does not hold, because the pool clamps workers
to `os.cpu_count()` (§1). Only `phase-2.md`, this log and `deferred-risks.md` were edited.

Superseded earlier decisions:
- S16 and the "F41 decision" (a feature-count check on every read, `Trajectory` width from
  `len(meta["obs_features"])`, `/2` iff `obs_schema` is not `balancer.obs.v1`) → V1. The format
  choice itself (a new `topple.traj/2`, `/1` frozen) stands.
- S4 (course points from libm `math.cos`/`math.sin`, exact per platform, with a radius within a
  few ulps of 1.5 m allowed to differ across platforms) → V17.
- Round 2 product #9 and round 3 product #8 (the ladders' 15, 12, 8 and 6 fps rungs) → V3.
- Round 3 product #2 (`test_driver_golden` pins with no platform named) → V6.
- Round 1 consistency #9 (per-kind `ENV_ALGO_VERSION`, form unstated) → V7.
- Round 1 security #6 and round 2 security #8 (`test_no_unsafe_loads` import bans with only
  `test_snapshot_reference` exempt) → V5.

| # | Issue | Impact | Decision | Reason |
|---|---|---|---|---|
| V1 | §8.1a's `/2` rules break merged Phase 1 tests: a meta without `obs_schema` would be written as `/2`, which the default reader refuses, and `Trajectory` calls `len()` on untrusted `meta.obs_features` | HIGH | accepted | Confirmed: a prototype of the literal rules fails 5 Phase 1 tests (`test_traj.py` round trip, symlink load and meta bytes; `test_verify.py` `obs_features-retyped` and the arbitrary-meta property). Fix (§8.1a): `/2` iff `meta` has an `obs_schema` other than exactly `balancer.obs.v1`; `Trajectory` requires width 10 (`/1`) or 1..64 (`/2`) and touches meta only with `in`, `isinstance`, `==`; only a `/2` layout checks the feature count; the `/1` path is the merged reader. "Width 1..64 always" was rejected: `test_fixed_trailing_shapes` requires 9 and 11 columns to fail. A prototype of the fixed rules passes all 1,806 Phase 1 tests that run on macOS and every §8.1a case; the six named tests are listed. Balance records are compared member by member, since numpy stamps zip entries with the save time. |
| V2 | 2b item 3: `verify_run` on 5 trajectories, and `topple verify DIR` on them, cannot pass against the 50-seed public protocol | MEDIUM | accepted | `verify_run` counts every absent index as `missing`, and the CLI builds a directory's protocol with `from_id` and has no `--seeds`. Item 3 verifies on `dataclasses.replace(wp_eval_v1, seeds=3000..3004)` and runs `topple verify` on each file and, in a slow test, on a full 50-episode `wp_eval_v1` run directory (§8.9). |
| V3 | GIF ladder rungs at 15, 12, 8 and 6 fps cannot render | MEDIUM | accepted | `video._stride` refuses any fps that does not divide the 100 Hz control rate, on every render path. Rungs are divisors of 100: hero 20 → 10 → 5, galleries and waypoint 10 → 5 → 4 (§5.2). |
| V4 | Strict mypy fails on `validate.py`'s `google.protobuf` imports | MEDIUM | accepted | protobuf 7.36.2 ships no `py.typed`, and `types-protobuf` is not in the lock. `google.protobuf.*` joins `ignore_missing_imports` instead of a new stub dependency. Measured: the import errors go away; `FromString` then returns `Any`, so §7 says to bind it to an annotated variable, which passes strict mode (§7). |
| V5 | `test_no_unsafe_loads` fails on the merged `tests/test_contract.py` (`import pickle`) and on the spec's own torch and SB3 tests | MEDIUM | accepted | A fixed (file → modules) allow-list: `train`-marked test files → `torch`, `stable_baselines3`; `tests/test_contract.py` → `pickle` (contract §8's `TaskSpec` round trip); `scripts/make_thirdparty_fixtures.py` → `torch` (it needs it too, though the finding missed it). `test_snapshot_reference` reloads SB3 through `learn.resume_from`, the only allowed load site. Banning load calls instead of imports was not adopted: `test_contract.py` calls `pickle.loads` too (§6). |
| V6 | `test_driver_golden` pins one value for every platform | MEDIUM | accepted | Physics reproduces only per platform (LQR seed 1001: 84.08 on darwin-arm64, 85.40 on linux-x86_64), and the test runs in the ubuntu job. Pins are per platform tag under `tests/test_golden.py`'s `pins_for` / `check_ci_platform` rule; the `WaypointScore` dump stays one pin (§6). |
| V7 | A per-kind `ENV_ALGO_VERSION` stored in the registry breaks `test_protocol_hash_covers_the_env_algorithm_version` | MEDIUM | accepted | That test monkeypatches `topple.challenge.ENV_ALGO_VERSION`. `KindSpec.env_algo_version` is a call-time getter; balance's reads that module attribute. Prototype: `test_challenge.py` and the hash goldens pass (§8.1). |
| V8 | `deferred-risks.md` (a) and (b) omit Phase 3, whose robot-ci scores users' own robots | MEDIUM | accepted | Contract §3 names robot-ci first among `validate_robot`'s users, and PLAN.md's Phase 3 ships the `robot:` input. Phase 3 is the first owning phase: the rule lands before that input is accepted, or robot-ci refuses robots it did not ship until it does (`deferred-risks.md`, §10). |
| V9 | Item 4 cites "§11/metrics vocabulary" | LOW | accepted | Contract §11 is termination; metrics are §12 (bounds §15). Item 4 says §12 (§8.9). |
| V10 | The ubuntu compare is timed on a "4-vCPU runner" | LOW | modified | The merged `ci.yml` runs on the private repo's 2-vCPU runner. The time is re-estimated at 1–2 min. The pool clamps `--workers 4` to 2 there, so nothing is oversubscribed (§4). |
| V11 | `read_regular_file` passes POSIX-only `os` flags directly | LOW | accepted | win32 is one of the lock's environments. Every optional flag goes through `getattr(os, name, 0)`, as merged `open_regular` and `loader._read_capped` do, restoring round 1 deploy #8. The lstat/fstat identity check carries the guarantee where `O_NOFOLLOW` is missing (§3.4). |
| V12 | `Archive` has no total size cap | LOW | accepted | Each member is capped by `max_member_bytes`, and `topple/_npz.py` adds an explicit `sum(member sizes) ≤ 8 MiB` check before any member is read (§3.3). |
| V13 | Policy-directory errors are not repr-quoted | LOW | accepted | `f"unexpected entry: {safe(name)!r}"`, as contract §8 requires of every name a load error shows (§3.1). |
| V14 | "Python submissions are unchanged" is false for a Python directory with a `policy.json` asset | LOW | accepted | The ambiguity rule stays (fail closed). The spec states the one changed case, and the 2a contract §8 update records it (§3.1, §3.6). |
| V15 | "one owning phase each" contradicts `deferred-risks.md` | LOW | accepted | §10 and the document's intro now speak of owning phases (§10, `deferred-risks.md`). |
| V16 | `deferred-risks.md` (b)'s suggested rule would refuse the shipped robot | LOW | accepted | Probe on `build_model(balance-v1)`: `chassis_box` and both tires have contype = conaffinity = 1, so they pair by bits, and only `filterparent` (on) keeps them apart. The rule now exempts pairs MuJoCo already filters: one weld body, parent-child, or an exclude (`deferred-risks.md`). |
| V17 | Course acceptance with libm `cos`/`sin` is not portable bit for bit, though contract §9 promises portable scenarios | LOW | accepted | Portable arithmetic was chosen over a contract §9 exception. `pcos`/`psin` use Python float `+`, `−`, `×` only (Cody–Waite reduction, 16-term Taylor polynomials by Horner), for both acceptance and `course_points`. Measured: within 1.3e-15 of libm, the same courses as libm on 20,000 seeds, 44.0% of attempts accepted, mean accepted turn 98.1° (§8.3, §8.9 item 4). |
| V18 | Two sync edits had no S-row | LOW | accepted | Row S22 added: the shove pushes use the trusted reset hook (§5.2), and an exhausted budget raises `RunnerError` (§10). |

## Linux recalibration (2026-10-15, spec v5.3)

Deferred risk (c) asked for balance-v1's calibration to be re-measured on the ranked platform before
2a training. Measured in a cloud session on linux-x86_64 (Intel Xeon, AVX-512; mujoco 3.15.0,
numpy 2.5.3), with the merged Phase 1 code, before any change: the linux-x86_64 golden scores of
`tests/test_golden.py` reproduced exactly, so the machine matches CI. On `balance-v1/standard` seeds
0..199: LQR 87.5% survival (median 3.28 J), shipped PID 65.0% (median 6.5655 J); `clean`: both
100%. Both bands hold (LQR ≥ 80%, PID 40–90%), so the push band stays. 2 × 6.5655 = 13.131 rounds to
13.1, not 13.0 (darwin-arm64: 2 × 6.49 = 12.98 → 13.0). Pranav chose to recalibrate.

Superseded earlier decisions: S1 (E_ref 13.0), S2 (§0 baselines measured on darwin-arm64) and S5
(the known-answer hashes) → R1. The adversarial review of the recalibration (2 reviewers, 2
skeptics) confirmed two findings, fixed in R4 and R5.

| # | Change | Reason |
|---|---|---|
| R1 | E_ref 13.0 → 13.1 J; linux-x86_64 is the calibration platform; `challenge.yaml` records both platforms' numbers; contract §10, §12 and §13 updated. New bundle `89a62007…0bbaedd0`, protocol hashes ideal `79450098…`, clean `3f27d1c9…`, standard `b2436063…`, selection (2000..2199) `f30ef190…4fcf6287`, calibration (0..199) `9d3630d3…bde59d5e`. Linux golden scores re-pinned; the energy term moved PID 1000/1002 and LQR 1000–1002 by 0.015–0.031 points (PID 1001 spends more than E_ref, so its score is unchanged). §0 baselines re-measured on linux-x86_64. | Ranked scores come from linux-x86_64 CI (PLAN.md trust model; contract §13), so the reference energy is measured there. Nothing had been trained against the old protocol hash, so no run is stale. |
| R2 | The darwin-arm64 golden scores in `tests/test_golden.py` stay at their E_ref 13.0 values, marked stale, until a Mac re-pins them. | Physics reproduces only per platform and no Mac was available to the cloud session. CI never checks darwin pins (only `linux-x86_64`), so CI stays green; a local Mac run fails and prints the values to pin. |
| R3 | The §13 serializer example in the contract and `test_challenge.py` now names `5.0`, not `13.0`. | `e_ref_j` is no longer integral, so `13.0` no longer illustrates the "never `13`" rule; `5.0` (`success_energy_factor`) still does. |
| R4 | The PID median is written 6.5655 J, not 6.566 J (a double rounding of 6.565462 J), in `challenge.yaml`, the contract, the spec, this log and `deferred-risks.md`. The yaml comment is hashed, so the bundle and protocol hashes moved again before any run bound to them: bundle `89a62007…0bbaedd0`, ideal `79450098…`, clean `3f27d1c9…`, standard `b2436063…`, selection `f30ef190…4fcf6287`, calibration `9d3630d3…bde59d5e` (the values R1 cites are these). | Review finding C1 (confirmed). A frozen file must be right before publication; no training had started. |
| R5 | Contract preamble and §13: calibrated values freeze when their challenge is **published**; before that, a change is a reviewed re-pin of hashes and goldens (as R1). Contract §12 notes that 2 × 6.5655 J sits 0.019 J below the 13.15 boundary and that E_ref stays 13.1 regardless of later re-measurements. | Review finding C2 (confirmed): the contract said "frozen at the end of Phase 1" and "E_ref needs balance-v2", which contradicted R1. The boundary note answers the reproduction reviewer's robustness finding. |
| R6 | Spec consistency (LOW findings C4–C8): §0's selection-vs-calibration example uses the linux numbers; known answers labelled v5.3; the `test_driver_golden` example names E_ref 13.0; §8.7 waypoint calibration runs on linux-x86_64; §9/§10 no longer list darwin-only calibration as open, only the darwin golden re-pin (R2). | Keeps the spec consistent with R1. |
| R7 | Re-review of R4–R6 (one verifier): contract §12 now says a re-measurement alone never changes E_ref and a change is a deliberate recalibration (re-pin before publication, new version after), matching the preamble and §13; `tests/test_golden.py`'s docstring says the same; §8 states that waypoint-v1 freezes at the end of 2b, earlier than §13 requires. | Remaining contradiction (MEDIUM) and wording (LOW) found by the re-review. |

## Owner answers (2026-10-15, spec v5.3)

Pranav's answers to the open questions in `HANDOFF.md` (defaults unless stated).

| # | Question | Answer | Effect |
|---|---|---|---|
| A1 | Re-measure E_ref on Linux before 2a training? | Yes | Done: E_ref changed, Pranav chose to recalibrate (R1). |
| A2 | Keep `topple.traj/1` frozen for balance-v1, `topple.traj/2` for other challenges? | Yes | §8.1a as written. |
| A3 | Waypoint scene `offheight=960` for a 960×960 video, or render ≤ 1280×720? | Render ≤ 1280×720 | §8.2: the waypoint scene is byte-identical to balance-v1's (offscreen 1280×720); §8.9 item 8's MP4 is 720×720 (the GIF stays 480×480). |
| A4 | One ~7-minute PPO confirmation run on the merged env before building 2a? | Yes | Not run yet: the cloud environment blocks the torch wheel hosts; it runs once torch is installable, before any training of slice 3. The torch-free slices proceed meanwhile. |
| A5 | Mention `topple.traj/2` in PLAN.md's architecture tree? | Yes, when 2b is built | A 2b task. |
| A6 | Drop `EvalProtocol.is_named` (S12)? | Yes | §5.3 as written (no `is_named`). |
| A7 | Commit the round-1/2 critique JSON files? | Skip | They exist only on the old Mac session. |
| A8 | Where does the old Phase 3 draft live? | On `phase-3-design` only | Cherry-pick only its docs later. |
