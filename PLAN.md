# Topple: Plan v2

> Supersedes the v1 plan (in git history). v1's core (balancer, MuJoCo, PPO vs. PID,
> `git push` → score + video) is kept; v2 adds what the research says people want and nobody ships.
> Brand: **Topple** (repo `pranavnandyalam/topple`, PyPI `topple-sim`, npm `topple-arena`, CLI and
> import name `topple`; bare `topple` is taken on PyPI and npm).

**Pitch:** *Can your controller survive being toppled?* The open successor to DeepRacer and robotbenchmark for the MuJoCo era. Design a robot,
push a controller, and get back a deterministic score, a crash-test rating, an AI coach's fix and a
replay anyone can watch and poke in the browser. Free, Mac-native, zero-install to try.

---

## 1. Why this, why now (research summary, Oct 2026)

| Finding | Evidence | What we do with it |
|---|---|---|
| **AWS DeepRacer is dead** (league ended 2024, console retired Dec 2025) after 560k+ builders in 6 years. The community went to self-hosted forks that run poorly on Mac. | AWS blog/FAQ; deepracer-for-cloud | Seasons, monthly challenges, divisions: the DeepRacer loop, open and free |
| **robotbenchmark.net is dead** (domain now spam). It was the only free "Python in browser, get ranked, watch replay" robotics site. | Cyberbotics/EOSC | Browser Arena is a core feature, not a stretch goal |
| **HF Open RL Leaderboard has stalled** (errors, last results Apr 2024). LeRobot has EnvHub but no standing leaderboard. | HF | Publish our env to LeRobot EnvHub; be the leaderboard |
| **"CI for robots" exists only for enterprises** (Applied Intuition, Foretellix). Hobbyists hand-roll Gazebo-in-Actions. | vendor sites, ROS Discourse | Ship a reusable GitHub Action anyone can point at their own MJCF |
| **Isaac Sim/Lab and MJWarp are NVIDIA-only.** Mac users are second-class everywhere. | NVIDIA, MuJoCo docs | Mac-first CPU training; browser for everyone else |
| **MuJoCo now has official WASM bindings** (`@mujoco/mujoco` 3.15). mjswan already runs ONNX policies in-browser. | Measured on M5 Pro: WASM ≈ 0.57× native; ONNX MLP 6 µs; Pyodide PID 3 µs/call | Zero-install arena is feasible now with huge headroom |
| **LLM-in-the-loop control is the 2026 frontier** (Act-Observe-Rewrite, Eureka/DrEureka). | arXiv 2603.04466, 2310.12931 | AI Coach PR bot + reward autotuner |
| **FIRST: ~200k FRC/FTC students.** Programmers get little robot time, rookies have no hardware until kickoff, and sim tools are "complicated to set up". 67% of teams want to sim their own robot. | FIRST AR 2025, Chief Delphi, WPI study | Off-season programmer-training league, team leagues |
| **Teachers are isolated and use Chromebooks.** They need ready lessons and autograding that fits GitHub Classroom. | CSTA 2025, SIGCSE 2025 | Classroom mode + 5-lesson curriculum |
| **"AI learns to X" videos get massive views** (AI Warehouse: ~97M views from 21 videos). | channel trackers | Every submission auto-generates a shareable clip; Sumo and Evolve make good content |

Gap noted: the community pain-point sweep (Reddit/HN) was cut short; the findings above come from
the competitor, growth, frontier-tech and feasibility sweeps.

---

## 2. Product pillars

1. **Sim Core.** Balancer robot, Gymnasium env, the `Controller` contract, PID and PPO baselines, `eval_v1`.
2. **Robot CI.** A reusable GitHub Action: push a controller and the PR gets a score, a GIF and a leaderboard rank. It works on *your own* robot's MJCF too.
3. **Crash-Test Rating.** NCAP-style 1–5 stars per controller, from an adversarial push fuzzer, a region-of-attraction map and formal specs.
4. **Arena (browser).** Watch any replay, shove the robot with your mouse, edit PID gains or Python live (Pyodide), and run ONNX policies. No install, works on a Chromebook.
5. **AI Coach.** Claude reads the metrics and key frames from a failed run and opens a PR that fixes your controller. `topple autotune` evolves reward and randomization configs overnight.
6. **Robot Sumo.** Two balancers in a ring, trained by self-play, with Elo, nightly tournaments and bracket videos.
7. **Seasons and leagues.** Monthly themed challenges (wind, ice, payload), divisions (HS / College / Open), and private class/team leagues.
8. **Sim-to-real.** A ~$50 ESP32 balancer from Onshape CAD, system ID from real logs, the same `Controller` on hardware, and a real-robot leaderboard track.
9. **Open challenges.** Anyone can author a task (MJCF + scorer + seeds) and the community votes it into the next season. Published to LeRobot EnvHub.

---

## 3. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Core | Python 3.12 via `uv`, `mujoco==3.15.*` (pinned) | Determinism is guaranteed only within one version + architecture |
| RL | Gymnasium 1.4, SB3 2.9 (PPO/SAC), TensorBoard | Training runs on CPU with `SubprocVecEnv`: 8 envs, at most 1 run at a time, and eval pools of at most 4 workers. This is because the laptop is shared (Phase 2 spec §1). |
| Policy format | **ONNX everywhere** (pickle-free `onnx.helper` build, IR 8 / opset 17, VecNormalize and clip baked in, canonical bytes) | Eval/CI/browser never need torch: `onnxruntime` ~15 MB vs. torch 555 MB+ |
| CLI | `typer` | `topple run / eval / train / export / fuzz / coach / autotune / sumo` |
| Rendering | `mujoco.Renderer`; `MUJOCO_GL=cgl` (Mac), `osmesa` (Linux CI) | CI renders one short episode at 320×240 → GIF |
| CI | GitHub Actions, `pull_request` → `workflow_run` split | Fork PRs get no secrets; privileged stage only reads validated artifacts |
| Web | Vite + TypeScript + three.js, `@mujoco/mujoco` (single-thread build), `onnxruntime-web`, Pyodide in a Web Worker | Start from mjswan rather than from scratch |
| Hosting | Cloudflare Pages (site), R2 (GIFs, `traj.npz`), D1 (leaderboard) | Est. $0–5/mo at 1k users; Claude API is the real cost, so add quotas |
| AI | Claude: user's own `ANTHROPIC_API_KEY` or their Claude Code login (`claude-sonnet-5-5` default, `claude-opus-5-5` for hard cases) | Bring-your-own key; images, not video: sample 6–10 key frames |
| CAD | Onshape → `onshape-to-robot` 1.8.x → MJCF | Commit generated MJCF; EDU API quota is ~2,500 calls/yr |
| Hardware | ESP32-S3 + MPU-6050/BMI270 IMU + 2 geared DC motors w/ encoders | Policy runs as generated C (tiny MLP) or PID |
| Tests | pytest + pytest-cov (80%+); Vitest for web | Bit-exact tests per platform only; cross-platform uses tolerances |

---

## 4. Architecture

```
topple/
├── python/topple/
│   ├── contract.py      Controller protocol, BalancerObs (obs_v1), TaskSpec, action validation (stdlib only)
│   ├── ids.py, rng.py   identifier grammar; named append-only RNG streams
│   ├── robot.py         balancer.robot.v1: params, MjSpec builder (motors, IMU, friction), validation
│   ├── challenge.py     challenge bundles (challenge.yaml: presets, score constants, public seeds)
│   ├── challenges/      balance-v1/{challenge.yaml, scene.xml, robot/}, later waypoint-v1, sumo-v1, ...
│   ├── disturbances.py  presets (ideal/clean/standard), JSON scenarios, pushes
│   ├── estimation.py    IMU model, complementary filter, encoders, odometry (portable float math)
│   ├── envs/            balance.py (topple/Balance-v1), later waypoint.py, sumo.py
│   ├── controllers/     pid.py, lqr.py, onnx_policy.py (P2), waypoint_pid.py (P2b)
│   ├── policy/          (P2) topple.policy/1: manifest, ONNX build, acceptance validator, hardened ORT session
│   ├── train/           (P2) config + wrappers + selection (torch-free); learn/snapshot/callbacks (train extra)
│   ├── design/          LQR design from the linearized model
│   ├── traj.py, scoring.py, verify.py   topple.traj/1, metrics + score, trusted replay + re-scoring
│   ├── evaluate.py, results.py, loader.py   protocols, topple.results/1, submission loading
│   ├── runner.py        (P3) runs untrusted controllers in a subprocess/container
│   ├── crashtest/       (P4) fuzzer, RoA, STL, rating
│   ├── coach/, sumo/, sysid/   (P6, P7, P8)
│   └── export.py, stats.py, compare.py, bench.py, course.py (P2b), video.py, cli.py
├── web/                 (P5) Arena: replay viewer, live sandbox, leaderboard, share cards
├── action/              (P3) reusable GitHub Action (robot-ci)
├── firmware/            (P8) ESP32 balancer firmware + host log tool
├── curriculum/          (P9) lessons, slides, rubrics, GitHub Classroom template
├── submissions/         example controllers
├── docs/contract.md     the normative contract
└── .github/workflows/   ci.yml; (P3) eval.yml (untrusted), publish.yml (workflow_run), nightly
```

### Core contract

The normative definition is `docs/contract.md`; this is the summary.

```python
@dataclass(frozen=True, slots=True)
class BalancerObs:                     # balancer.obs.v1, alias topple.contract.Observation
    t: float                           # s since reset (metadata, not a feature)
    pitch: float; pitch_rate: float    # from the IMU + complementary filter, not ground truth
    yaw: float; yaw_rate: float        # yaw from wheel odometry, yaw_rate from the gyro
    wheel_pos: tuple[float, float]     # quantized encoders, rad since reset
    wheel_vel: tuple[float, float]     # encoder finite difference per control step
    last_action: tuple[float, float]   # command applied during the previous step

class Controller(Protocol):
    def reset(self, rng_seed: int) -> None: ...      # private seed, never the episode seed
    def act(self, obs: BalancerObs) -> Sequence[float]: ...   # normalized voltage per wheel, [-1, 1]

def make_controller(spec: TaskSpec) -> Controller: ...        # in every submission
```
`act()` runs at exactly 100 Hz everywhere, firmware included; sensing and estimation run at
500 Hz. PID, LQR, ONNX policies, user submissions, coach output and firmware all implement it.
Later tasks append fields under new schema ids (e.g. `balancer-target.obs.v1`) instead of
changing `obs_v1`.

### Trust model (the non-negotiables)
- **Ranked scores come only from CI** (pinned Linux image = source of truth). Browser runs are unranked.
- **The controller never shares a process with the scorer** (it could monkeypatch metrics). Subprocess + pipe; Docker `--network none --read-only` in CI.
- **Hidden seeds:** public PR evals use public seeds. The top-k get re-evaluated nightly on private seeds in the privileged stage to stop overfitting.
- **The replay viewer plays stored trajectories (`traj.npz`), never re-simulates.** Mac/Linux/WASM diverge after ~5k steps.
- **Scores never come from an upload.** The trusted path (`topple verify`/`score`, publish, nightly)
  re-simulates an uploaded trajectory's commands on its own build from the protocol's own seed and
  scores only that replay; any bit that differs fails verification (contract §16).
- **Tournament opponents exchange ONNX weights (data), not code.**
  - An ONNX submission is a directory containing only `policy.onnx` and `policy.json`, with no
    `.py` and no pickles.
  - Models are validated on their bytes against an allow-list before ORT parses them.
  - Pickled training checkpoints never cross a trust boundary.

---

## 5. Phases

Each phase ends in something **shippable and shareable**, with a concrete **done when**. Estimates
assume part-time work alongside school.

### Phase 0: Setup (1 evening)
- `uv init`, Python 3.12, pin `mujoco==3.15.*`, ruff + pytest + coverage config, `.superset/config.json` (`uv sync`).
- Smoke test: step an inline MJCF model 1000× and render `smoke.png` headless (`cgl` / `osmesa`).
  (Inline instead of a Menagerie download so tests need no network; Menagerie models arrive in Phase 13.)
- Python package skeleton (`python/topple`). `web/` and `action/` are created by Phases 5 and 3.
- **Done when:** `uv run pytest` passes and `uv run python -m topple.smoke` writes `smoke.png`.

### Phase 1: Sim Core (1–1.5 weeks)
- Placeholder balancer MJCF (box, 2 wheels, 2 hinge motors, realistic masses and torque limits).
- `Observation`/`Controller` contract, `BalanceTask`, disturbances, PID baseline.
- `eval_v1`: 20 seeded episodes → `results.json` + `traj.npz`; `topple run` writes MP4.
- **Done when:** `env_checker` passes; PID balances >90% of no-push episodes; same seed gives an identical trajectory on the same platform (test); coverage ≥80%.

### Phase 2: Learn (1–2 weeks). Spec: `docs/design/phase-2.md`
- **2a, balance.**
  - Training: PPO with 8 SubprocVecEnv workers on `balance-v1/standard` from step 0. There is no
    curriculum (measured worse). Each seed trains for 5M steps, and 3 seeds are run.
  - Training-only action-excess penalty.
  - Checkpoints are selected on 200 selection seeds through the real harness. Logging uses
    TensorBoard.
  - Checkpoints are eligible only if success ≥ PID's, median energy ≤ E_ref and saturation ≤ 0.08.
  - SAC is the comparison: 1.0M env steps (about PPO's training time), same harness and seeds.
    It is learner-throughput-limited at this budget and is not expected to reach PID level.
  - Each run is capped by a process-tree RSS limit and a CPU-seconds budget. Wall-clock is only a
    runaway guard; an aborted run is re-run, never reported.
- **Export.** `topple export` builds a pickle-free, torch-free ONNX: VecNormalize and clipping are
  baked in, with IR 8 and opset 17. It writes a **data-only** submission (`policy.onnx` +
  `policy.json`, `topple.policy/1`, no code). This needs an ONNX acceptance policy and the
  hardened `ONNXPolicy` runtime.
- **Results.** The package covers the CI scoreboard, the shove test, learning curves and the hero
  video.
- **Done when (2a):** the shipped ONNX directory, run without torch, must pass all of:
  - its eval_v1 score > PID's;
  - on calibration seeds 0..199, the paired-bootstrap 95% CI of (PPO − PID) is above 0;
  - its success rate ≥ PID's on both seed sets;
  - it beats LQR (paired CI lower bound of PPO − LQR > 0, success ≥ LQR's − 0.03);
  - median energy ≤ E_ref and saturation ≤ 0.10;
  - ONNX matches SB3's own predict within 1e-5, open-loop at batch 1, re-checked torch-free in CI;
  - at least 2 of the 3 PPO training seeds independently pass the PID gates (the recipe, not one
    lucky seed);
  - a side-by-side PID vs PPO video exists.
- **2b, waypoint-v1** (after 2a).
  - The course: fixed-leg courses (5 × 0.70 m), the `balancer-target.obs.v1` observation,
    and a frozen scorer (55 course / 25 speed / 15 alive / 5 energy).
  - Baselines and training: a reference driver, and PPO trained from scratch vs fine-tuned from
    the balance policy.
- **Done when (2b):**
  - the challenge is frozen (push strength calibrated first, by a fixed sweep), its two-sided
    calibration band holds, and its degenerate-strategy anchors pass;
  - it is learnable, judged relative to the reference driver at the frozen push level: the best
    learned policy docks on ≥ 60% of calibration seeds, is not clearly worse than the driver
    (paired CI lower bound of PPO − driver > −5), beats the creeper anchor, and falls at most
    20 points more often than the driver;
  - balance-v1 is unchanged;
  - verify and rescore work;
  - both arms are reported (3 seeds each);
  - a course video exists.

### Phase 3: Robot CI (1–1.5 weeks): *first public release*
- `runner.py` subprocess sandbox; rejects bad shapes, NaNs, timeouts, exceptions.
- `eval.yml` (`pull_request`, `permissions: {}`, timeout) → artifacts: metrics, traj, GIF.
- `publish.yml` (`workflow_run`): schema-validate artifacts, upload GIF to R2, PR comment with embedded GIF + metrics table + rank, update `leaderboard.json`.
- Package as a reusable action: `uses: <you>/robot-ci@v1` with `robot:`, `task:`, `controller:` inputs.
- **Done when:** a fork PR with a new controller gets a comment with GIF + score in <3 min; red-team tests (score tampering, infinite loop, network access, huge output) all fail cleanly.

### Phase 4: Crash-Test Rating (1 week)
- `fuzzer.py`: CMA-ES finds the smallest push sequence that topples a controller → "robustness margin".
- `roa.py`: region-of-attraction heatmap over (initial tilt, tilt rate).
- `stl.py`: spec checks, e.g. "|pitch| < 15° within 2 s after any push ≤ X N".
- `rating.py`: 1–5 stars from those three; SVG badge; section in PR comment.
- **Done when:** PID and PPO both have published ratings with heatmaps, and the fuzzer finds a failure in a deliberately fragile controller in <60 s.

### Phase 5: Arena (2–3 weeks): *launch moment ("Season 0")*
- Replay viewer: `traj.npz` → three.js, scrub, slow-mo, side-by-side ghost of the #1 entry.
- Live sandbox: `@mujoco/mujoco` + mouse shove + PID sliders + Pyodide editor for `act()` (Web Worker, kill on hang) + ONNX policies.
- Leaderboard pages, per-submission share page with OG image ("survived 47 shoves, rank #12, ★★★★"), embeddable badge.
- "Submit" button → opens a prefilled PR (git stays the source of truth).
- **Done when:** on a Chromebook, a stranger goes from link to shoving the robot in <10 s and edits a PID gain and sees the change in <60 s; Lighthouse performance ≥80.
- **Launch:** README with GIFs, 60 s video ("I trained an AI to balance a robot… then kicked it"), HN, r/robotics, r/reinforcementlearning, r/FRC, Chief Delphi, DeepRacer community.

### Phase 6: AI Coach + Autotune (1.5–2 weeks)
- `topple coach controller.py`: run eval → send metrics + 6–10 key frames + code to Claude → get patch → re-eval → keep only if score improves → repeat ≤N rounds.
- Bring-your-own key: public users run `topple coach` locally with their own key or Claude Code login. A copy-paste workflow lets them add `/coach` to *their own* repo using their own secret. Fork PRs never get secrets, so the main repo runs no public coach bot.
- On Pranav's repos, `/coach` runs through the Claude GitHub Action with his Max token → opens a follow-up PR with the fix and a before/after GIF.
- `topple autotune`: Eureka-lite. Claude proposes reward + domain-randomization configs, PPO trains candidates in parallel on 18 cores, and stats feed back to Claude.
- **Done when:** on a benchmark of 10 deliberately broken controllers, the coach gets ≥7 above the PID baseline within 3 rounds; autotune finds a reward that beats the hand-written one on `eval_v1`.

### Phase 7: Robot Sumo + Seasons (2 weeks)
- `SumoTask`: two balancers, ring, push the other out or make it fall.
- Self-play PPO with an opponent pool; Elo/TrueSkill; nightly round-robin of all ONNX entries in CI.
- Seasons: monthly theme config, divisions, season archive, auto-generated bracket video.
- **Done when:** the nightly tournament runs unattended, ratings converge (stable top-5 over 3 nights), and a bracket video is generated from one command.

### Phase 8: Sim-to-Real (3–4 weeks + parts lead time)
- Design the balancer in Onshape (mates `dof_left_wheel`/`dof_right_wheel`), export with `onshape-to-robot`, commit MJCF.
- Build: ESP32-S3, IMU, 2 encoder motors, battery (~$50). Firmware runs the IMU + estimator at 500 Hz and calls the controller (PID or a generated-C MLP) at exactly 100 Hz, logging `topple.traj/1`-compatible records over Wi-Fi.
- `sysid/`: fit motor constants, friction and latency from logs; retrain with randomization centered on fitted values.
- **Sim-to-real correlation study** (the credibility test for the whole leaderboard):
  - Run ≥8 controllers (PID variants, PPO, top leaderboard entries) on both sim and real with a scripted push rig (servo or pendulum striker, so pushes are repeatable).
  - Report Spearman rank correlation of sim score vs. real score, plus pitch-trajectory overlays for identical pushes.
  - Test the hypothesis that the **crash-test rating predicts real-world success better than raw score** (robustness > fidelity).
  - Publish the sim-to-real fidelity number on the leaderboard and re-run it whenever the robot model changes.
- Fabrication: UMich makerspaces for printing/cutting; parts ordered early (see timeline).
- **Done when:** the same controller file balances the real robot >30 s and survives a scripted push; sim/real rank correlation ρ ≥ 0.7 across ≥8 controllers (or a documented explanation of why not, with the sim fix); overlay video exists; the real-robot leaderboard track has entries.

### Phase 9: Classrooms, FRC and Open Challenges (2 weeks)
- League mode: private leaderboards, join code, teacher dashboard (submissions, scores, no student PII beyond GitHub handle).
- GitHub Classroom template + `curriculum/`: 5 lessons (P → PID → tuning by data → RL → robustness) with slides and rubrics.
- FRC-flavored challenge: differential-drive auto path following, with a RobotPy-style API.
- Challenge spec format (`challenge.yaml` + MJCF + scorer + seeds) + contributor guide; publish `balance-v1` to LeRobot EnvHub.
- **Done when:** a teacher can create a class league and a student can submit in <10 min from a fresh account; one community-authored challenge passes review and runs in CI.

### Phases 10–15: Extensions (all in scope, built after Phase 9)
- **Phase 10: Evolve (S1).** LLM-guided body + brain co-design (wheel radius, mass placement, extra links), parallel on 18 cores, with a generation time-lapse.
  - **Done when:** an evolved morphology beats the hand-designed balancer on `eval_v1` with an equal training budget; a time-lapse video comes from one command.
- **Phase 11: Human vs. Machine (S2).** Phone-tilt / WebXR teleop in the Arena records demos as LeRobot datasets → train ACT → "human", "imitation" and "RL" leaderboard categories.
  - **Done when:** a phone player can drive a full episode; ACT trained on ≥50 demos completes `eval_v1`; all three categories are live.
- **Phase 12: Text-to-robot (S3).** "A balancer with a tail" → Claude writes MJCF from primitives → compile → render → critique loop → playable in the Arena.
  - **Done when:** ≥8 of 10 benchmark prompts produce MJCF that compiles, stands still under zero torque where expected, and matches the prompt per a rubric.
- **Phase 13: Bigger robots (S4).** SO-101 arm (Menagerie model; real arm ~$150) reach/pick task, and a quadruped trained with MuJoCo Playground on UMich Great Lakes GPUs; both as new challenges.
  - **Done when:** both robots exist as challenges with baselines on the leaderboard; ONNX policies replay in the Arena.
- **Phase 14: Hosted submissions (S5).** Non-git uploads from the Arena run on Modal/e2b (`--network none`) behind a queue with rate limits.
  - **Done when:** a browser upload is scored identically to the CI path for the same controller; red-team suite passes against the hosted runner; spend alert configured.
- **Phase 15: Dream vs. Real (S6).** DreamerV3 world model on the balancer; "dreamed rollout vs. real rollout" video; a model-based entry on the leaderboard.
  - **Done when:** the world model predicts 1 s of balancer state within a stated error bound; the comparison video exists.

### Review gate (every phase and feature)
When a phase's "done when" passes locally, independent agents review it before it counts as done:
1. **Code review** (python-reviewer / code-reviewer; TypeScript reviewer for `web/`).
2. **Security review** (security-reviewer). Mandatory for Phases 3, 5, 6, 9 and 14.
3. **Done-when verifier:** a fresh agent re-runs the tests and the phase's "done when" checks from a clean checkout and reports pass/fail with evidence.

If there are no CRITICAL/HIGH findings, the phase's PR is merged and the next phase starts automatically.
Work pauses for Pranav only on: creating accounts, spending money, hardware, launches/posts, and
making the repo public (planned right after Phase 3's security review passes).

---

## 6. Timeline (part-time, approximate)

| Month | Phases | Public milestone |
|---|---|---|
| 1 | 0, 1, 2 | PID vs. PPO video |
| 2 | 3, 4 | Robot CI action v1 (soft launch) |
| 3 | 5 | **Season 0 launch** (Arena + leaderboard) |
| 4 | 6, 7 | AI Coach + Sumo tournament video |
| 5–6 | 8, 9 | Real robot + sim-to-real study + classroom/FRC off-season league |
| 7–9 | 10–15 | Evolve video, human vs. machine, text-to-robot, arm/quadruped, hosted uploads, dream vs. real |

Order rule: don't start a phase until the previous one passes its review gate, **except** Phase 8
parts ordering, which should happen in month 2 to absorb shipping time.

---

## 7. Testing strategy (TDD, 80%+)
- **Unit:** reward terms, termination, observation construction, clipping, PID math, Elo math, rating formula, config/challenge schema parsing.
- **Determinism:** same seed → bit-identical trajectory per platform; cross-platform within tolerance over 1k steps.
- **Contract:** every controller in `submissions/` and every coach output satisfies `Controller`.
- **Security/red-team:** scorer tampering, `os.system`, network calls, infinite loops, memory bombs, oversized output, malicious artifacts in `workflow_run`.
- **ONNX parity:** ONNX matches SB3's own predict pipeline within 1e-5. The check runs at batch 1
  on fixed observation sets, fed open-loop:
  - on-policy observations;
  - random observations in normalized space;
  - edge cases;
  - with at least 300 unsaturated rows.

  CI re-checks the result without torch.
- **Integration:** `topple eval` → valid `results.json`; action end-to-end on a test repo.
- **Golden regression:** PID score range; CI fails if a sim change moves it.
- **Web:** Vitest for replay decoding and the sandbox worker; Playwright smoke for "load → shove → edit gain".
- **Coach:** recorded-response tests (no live API in CI); live benchmark run manually.
- Training is not unit-tested. A 2k-step smoke train runs in CI, in the separate `python-train` job
  that has the train extra. The default job never installs torch.

---

## 8. Risks and mitigations

| Risk | Sev | Mitigation |
|---|---|---|
| **Scope blowup** (9 pillars, one person) | High | Every phase ships on its own and is publicly useful before the next starts |
| Leaderboard cheating (tampering, forged browser scores, seed overfitting) | High | Subprocess isolation, CI-only ranked scores, private-seed re-eval of top-k |
| Pwn-request / CI abuse | High | `pull_request` + `workflow_run` split, `permissions: {}`, timeouts, first-time-contributor approval, never execute artifacts |
| Cross-platform divergence | Med | Pin MuJoCo, CI image authoritative, replays from stored trajectories |
| MuJoCo WASM is "WIP" | Med | Pin npm version; Arena falls back to pre-rendered GIF/MP4 |
| Reward hacking / RL not converging | Med | PID proves solvability; 5M budget, 3 seeds, 200-seed selection, action-excess penalty; watch videos; autotune |
| Claude cost/abuse in coach | Low | Bring-your-own key for public users; Pranav's Max plan covers only his own use (Claude Code) |
| Onshape API quota / export pain | Med | Placeholder MJCF unblocks everything; export once, commit output |
| Hardware sim-to-real gap | Med | System ID + domain randomization; PID fallback; treat the gap as a write-up finding |
| Nobody shows up | Med | Ship content (videos) with every phase; seed via FRC mentors, teachers, DeepRacer alumni before launch |

---

## 9. Success metrics (Season 0 + 3 months)
- 1,000 GitHub stars; 100 external submissions; 3 classroom or FRC team leagues.
- 1 video >100k views.
- 1 external repo using the `robot-ci` action on its own robot.
- Real robot balancing on the same controller code as sim.

---

## 10. Decisions (2026-10-10)
1. **Scope:** everything, Phases 0–15.
2. **Name:** Topple.
3. **Public:** when safe, i.e. after Phase 3's security review passes.
4. **AI Coach:** bring-your-own key for public users. Pranav's Max plan covers his own runs via Claude Code / the Claude GitHub Action on his repos.
5. **Workflow:** review-agent gate per phase, then auto-continue (see Review gate).
6. **Hardware:** decide by month 2. UMich makerspaces for fabrication; Great Lakes HPC for GPU work.
7. **Accounts** (Cloudflare, Onshape EDU, domain, optional Anthropic API key): created when a phase needs them, with Pranav doing the signup.

---

## 11. Resources
- MuJoCo + WASM: https://mujoco.readthedocs.io, https://github.com/google-deepmind/mujoco/tree/main/wasm
- mjswan (browser MuJoCo + ONNX): https://mjswan.readthedocs.io
- Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- onshape-to-robot (MuJoCo export): https://onshape-to-robot.readthedocs.io/en/latest/exporter_mujoco.html
- SB3 ONNX export: https://stable-baselines3.readthedocs.io/en/master/guide/export.html
- uv + PyTorch CPU index: https://docs.astral.sh/uv/guides/integration/pytorch/
- Secure Actions (pwn requests): https://securitylab.github.com/research/github-actions-preventing-pwn-requests/
- Act-Observe-Rewrite: https://arxiv.org/abs/2603.04466 · Eureka: https://arxiv.org/abs/2310.12931 · DrEureka: https://arxiv.org/abs/2406.01967
- Adversarial attacks on legged controllers (RSS 2024): https://roboticsconference.org/2024/program/papers/57/
- Competitive self-play (sumo): https://openai.com/index/competitive-self-play
- LeRobot: https://github.com/huggingface/lerobot
- DeepRacer retirement: https://aws.amazon.com/deepracer/faqs/
