# Handoff: current state and how to continue

This file moves work between Claude Code sessions (local Mac ⇄ cloud at claude.ai/code).
**Whoever stops working must update it, commit it and push before stopping.** Nothing else
carries over: local memory, scratch files and workflow scripts stay on the machine that made them.

Last updated: 2026-10-17 by the local Mac session (Phase 2b built; review gate next). Branch `phase-2-learn` is pushed; PR #4 (Phase 2a)
is open. The training runs live only on the Mac, in `runs/` (gitignored; the shipped run is
`runs/ppo_s2`, needed by `phase2_done.sh`'s re-export check).

## Where things stand

| Item | State |
|---|---|
| Phase 0 (setup) | Merged to `main` (fcc2b13) |
| Phase 1 (sim core, balance-v1) | Merged to `main` as **f997394** (PR #3) |
| **Phase 2a (learn: balance)** | **Built, done-when passed, review gate passed.** PR #4. Training (torch half `03ae4e3`, review fixes `f9109fe`, load averages `dee02dc`), results package (`a01f552`, review fixes `1031bed`), shipped policy and results (`6e28642`), ubuntu evidence (`3e3ee81`). `scripts/phase2_done.sh` passes all 11 gates + G9 (CI green on HEAD), confirmed from a fresh clone by the gate's done-when verifier. Review gate (5 reviewers + verifier + 2 skeptics): 0 blocking; the confirmed MEDIUM/LOW findings are being fixed before merge (B54) |
| Shipped policy | `submissions/ppo_balance` = ppo_s2 (step 5,013,504). eval_v1 90.93 / 1.000 vs PID 73.06 / 0.750, LQR 85.44 / 0.950. 0..199: Mac 85.58 / 0.895 (PPO−PID +16.9 [+14.2, +19.7], PPO−LQR +5.5 [+4.0, +7.2]); ubuntu 85.55 / 0.890 (+16.9, +5.2 [+3.7, +6.9]). All 3 PPO seeds pass G11. Shove i50: PPO 0.50, LQR 0.45, PID 0.42 N·s |
| Training runs | 3 PPO + 3 SAC + ablation, all on `dee02dc`, `git_dirty: false`, bit-identical to an earlier set at different machine loads (determinism). SAC never reaches the PID gate (as §0 predicts) |
| Linux recalibration | E_ref **13.1 J** on linux-x86_64 (the calibration platform); darwin goldens re-pinned (`07681c8`) |
| Direction update | `PLAN.md` section 0 (2026-10-11): workbench for people's own robot projects, Studio, Arena with FRC drive practice, revised build order |
| Deferred risks | `docs/design/deferred-risks.md` |
| **Phase 2a** | **Merged to `main` as 386df1e (PR #4)** |
| **Phase 2b (waypoint-v1)** | **Built; done-when run and review gate next.** Branch `phase-2b-waypoint` (pushed). Slices 1-3, fine-tune init, pilot (`3654537`); 6 seed runs in `../p2-train` at `3654537` (copied into `runs/`); shipped `submissions/ppo_waypoint` = `ppo_wpft_s1` (fine-tune, step 5,505,024; `1e2bd5d`); compare files carry waypoint extras (`5a65204`); results, gate, done script, course video (`4159c5f`); contract + media (`09367b3`); waypoint goldens (`0b6bc5d`). Fresh standard 0..399: shipped 79.23 vs driver 80.24 (paired −1.02 [−3.35, +1.21]), docked 81%, fell 19%; item 6: 5 of 6 runs pass (`ppo_wp_s1` fails, lo −5.11). Headline: first eligible at a median 3.01M in both arms; fine-tune +5.01M pretraining. Energy: fine-tune 1.03× driver, scratch 2.72×. Item 7 survival 1.0 ×3. Linux anchors match darwin within 0.25 |
| Phase 3 design inputs | Branch `phase-3-design` (draft + critiques, not yet revised). Never merge it wholesale |

## Next steps, in order

1. **Phase 2b next: the gate fix pass, then merge.** Review gate v6 (run wf_a21d6525-01f,
   2026-10-17) on PR #9: the done-when verifier passed `scripts/phase2b_done.sh` from a fresh
   clone with CI green (4,266 tests, 98% coverage). One HIGH, now fixed in `7536b81`:
   `run_versions()` recorded `balancer.obs.v1` for waypoint runs. Still to do, in order:
   - **Regenerate evidence** with the F1 fix: rerun `scripts/phase2b_results.py` (all six runs;
     about 15 min) and `phase2b_media.py`; re-measure `docs/results/phase2b/anchors/
     wp_anchors_darwin.json` (`topple compare` of the driver and 5 anchors on standard 0..399) and
     re-run the `calibrate-waypoint` CI `anchors` job for the linux file. Do this once, after the
     fixes below that change `phase2b.json`.
   - **Gate binding (MEDIUM, `scripts/phase2b_gate.py`):** bind each comparison to its protocol
     (standard 0..399, clean 0..399, wp_eval_v1) and the `ppo_waypoint` entry to the submission's
     `policy_sha256`; check `phase2b.json`'s per-run rows against the committed
     `phase2b/runs_*.json` comparisons; replay the §8.8 shipping order (`candidates(kind=
     "waypoint")` over the summaries) and the attempts list.
   - **Fall rates from the validated termination array** (MEDIUM, `phase2b_common.rates`):
     `fell` = termination in FALL_REASONS, cross-checked against the extras; count `sim_unstable`
     in `no_errors` (LOW).
   - **Item 8 (MEDIUM):** check that the MP4 is really an MP4 (ftyp box) and that the media show
     the shipped policy; accept the ladder's smaller GIF rungs or force 480 (the spec says 480x480).
   - **Video (MEDIUM):** the checkered dock is hard to see at GIF scale (make it larger or higher
     contrast); add tests for a fall (epilogue, then the FELL hold), push arrows and the MP4 path; mark
     rendering tests `render` so they skip when rendering is disabled. LOWs: keep the gate index in the
     caption after a fall, don't let gate numbers hide the robot, anchor trails to a fixed grid,
     remove the unused `_yaw_mat`, and decide whether the camera looks along or across the axis
     (the spec says "along").
   - **Tests (MEDIUM):** known-answer waypoint episode scores per platform (driver on 3 seeds, as
     `test_golden.py` does for PID/LQR); env tests for next/final/after-dock observation semantics
     (contract §6); tests for `phase2b_results.py` shipping, headline and arms. LOW: env ==
     scorer events on 50 seeds (spec §8.5), not 10.
   - **Contract LOWs:** the §12a metric order and the darwin E_ref figure.
   - **Pranav's decision (2026-10-17, "you decide"): add `fall_cause` to balance-v1** as an
     additive metric (spec §8.1 allows new keys; scores, success and hashes don't change). Check
     that the 2a goldens and `phase2_done.sh` still pass; update contract §12, spec §8.5 and the
     "known" note.
   - Update spec §8.9's text: results go to `phase2b.json`/`phase2b.md`.
   - Then: re-run `phase2b_done.sh` with CI green, a short re-review of the fixes (one agent per
     area, high effort), and squash-merge PR #9.
   Notes: `phase2_ci_check.py` cannot resolve the repo in a clone of a local path (set GH_REPO);
   the CI python job went from 7 to 16 min (watch it; the 30-min timeout is the limit).
2. Usage: Pranav asked to stop at 55% weekly (2026-10-17). Pause at 97% of the 5-hour window.
3. Then PLAN.md section 0's build order (Studio, your own robot, test scenarios, Arena, ...), and
   revise the Phase 3 design as a short design note.

**Working mode (Pranav, 2026-10-15):** build slices directly in the main session with TDD; tests and
CI are the main quality tool; one background reviewer per ordinary slice (high effort), a reviewer
plus red-team at max effort for trust-boundary code, and the full gate workflow only per phase;
long training runs as plain background processes (in a separate clean worktree, so edits never
mark them dirty); shell loops under `bash -c` (zsh does not word-split).

## Effort and cost settings (Pranav, 2026-10-15)

The cloud session's usage was ~87% workflows, ~49% at max effort, mostly at >150k context. Use:

| Work | Effort |
|---|---|
| Session default; builder agents; everyday coding | **high** |
| Reviewers/skeptics on trust-boundary code (ONNX validator, loaders, runner sandbox, CI workflows, security reviews of Phases 3, 5, 6, 9, 14) | **max** |
| Reviewers on ordinary code (stats, video, configs, docs) | **high** |
| Design specs and critique rounds (e.g. revising the Phase 3 draft) | **xhigh** |
| Spec syncs, HANDOFF updates, mechanical edits, re-running tests | **medium** |

- Ultracode off by default; turn it on only for review gates and design work.
- Fast mode off (`/fast`); Pranav turned it off on 2026-10-15.
- 2 reviewers per slice (not 3); the fixer syncs the spec instead of a separate spec-sync agent.
- Keep context small: a fresh session per slice, with this file carrying the state.

## Answers from Pranav (2026-10-15)

Q1–Q8 took the defaults in brackets (Q1: re-measured, and E_ref changed, so he chose to recalibrate
to 13.1); decision log rows A1–A8 record each answer's effect (A3: the waypoint scene stays
identical to balance-v1's, MP4 720×720). Q9: all of the above (makers, students/FRC, researchers). Q10: any robot. Q11: hardware
optional. Q12: FRC season still open. PLAN.md section 0 and decisions 9–10 record them.

The original questions, for reference (defaults in brackets):
1. Re-measure E_ref on Linux before 2a training? [yes, step 2 above]
2. Keep `topple.traj/1` frozen for balance-v1 and use `topple.traj/2` for other challenges? [yes]
3. Waypoint scene needs `offheight=960` for the 960×960 video; OK, or render ≤1280×720 so the scene
   stays identical to balance-v1's? [render ≤1280×720]
4. One ~7-minute PPO confirmation run on the merged env before building 2a? [yes]
5. Mention `topple.traj/2` in PLAN.md's architecture tree? [yes, when 2b is built]
6. Drop `EvalProtocol.is_named` (S12 in the decision log)? [yes]
7. Commit the round-1/2 critique JSON files? [they only exist on the old Mac session; skip]
8. Where should the old Phase 3 draft live? [on `phase-3-design` only]

From the direction update (PLAN.md section 0):
9. Main user: hobbyists/makers, students and FRC teams, or researchers?
10. Robot types: mostly wheeled/balancing, or any robot?
11. Is hardware (controllers on a real board) central or optional?
12. Target FRC season for the Arena (e.g. ready by the January 2027 kickoff)?

## Rules (from Pranav; all still in force)

- **Repo stays private.** Ask Pranav before making it public (planned after the Phase 3 security
  review). Never change repo settings.
- **Pause and ask only for:** creating accounts, spending money, hardware, launches or public posts,
  making the repo public.
- **Quality over speed.** Max effort for reviewers. When a subagent type pins a smaller model in its
  definition, override it to Opus explicitly.
- **Workflows stay small:** under ~10 agents each. Builds are: one builder → 2–4 reviewers → 2 batch
  skeptics (correctness and reproduction lenses) that verify every MEDIUM+ finding → one fixer →
  re-review. Split big phases into several workflows.
- **Review gate per phase:** focused reviewers + a fresh-clone done-when verifier + batch skeptics.
  Merge only with no confirmed CRITICAL/HIGH and green CI. Fix confirmed MEDIUMs before merging.
- **Git:** stage explicit paths (never `git add -A` or `git add .`); never force-push; never
  `rm -rf` absolute paths; squash-merge phase PRs. Commits end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: <this session's link>
  ```
  PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Never use Pranav's Max subscription for a public bot serving others.** The AI coach is
  bring-your-own-key for public users.
- **Resources:** bound any test or probe that could allocate without limit (a past test fed an
  endless iterable to `list()` and exhausted memory). Cap pytest workers to the machine. One training
  run at a time.
- Golden pins exist for darwin-arm64 and linux-x86_64 (`tests/test_golden.py`); they must not be
  skipped on either.

## Handing back to the local Mac session

Before credits run out (stop at a clean point; don't leave agents mid-edit):
1. Commit and push every branch you touched (plain push).
2. Update this file: the state table, next steps, anything learned, and open questions.
3. Commit and push it on `phase-2-learn` (or whichever branch is current; say which here).
