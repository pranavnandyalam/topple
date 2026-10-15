# Handoff: current state and how to continue

This file moves work between Claude Code sessions (local Mac ⇄ cloud at claude.ai/code).
**Whoever stops working must update it, commit it and push before stopping.** Nothing else
carries over: local memory, scratch files and workflow scripts stay on the machine that made them.

Last updated: 2026-10-15 by the cloud session (in progress; see "Where things stand").

## Where things stand

| Item | State |
|---|---|
| Phase 0 (setup) | Merged to `main` (fcc2b13) |
| Phase 1 (sim core, balance-v1) | Merged to `main` as **f997394** (PR #3). Passed two review gates (the second all-Opus); every confirmed finding fixed; CI green (1,806 tests, 98.9% coverage) |
| Phase 2 (learn) | **Building 2a.** Spec v5.3 in `docs/design/phase-2.md`; decision log `docs/design/phase-2-decisions.md` (Linux recalibration rows R1–R6). Slice 1a (ONNX builder, validator, canonical form) in progress |
| Linux recalibration | **Done** (2026-10-15, Pranav chose it): E_ref 13.0 → **13.1 J** on linux-x86_64, now the calibration platform. Bands hold (LQR 87.5%, PID 65.0%). New bundle `89a62007…`, protocol hashes and linux golden scores re-pinned. Reviewed (2 reviewers + 2 skeptics), findings fixed |
| Darwin golden pins | **Stale** in `tests/test_golden.py` (`EPISODES["darwin-arm64"]`, measured under E_ref 13.0). The Mac session must re-pin them: run `uv run pytest tests/test_golden.py` on the Mac and copy the measured values (decision log R2). CI only checks linux pins |
| Network (cloud) | The cloud environment's policy blocks `download.pytorch.org` and `download-r2.pytorch.org`, so torch cannot be installed and `uv lock` cannot run (it resolves the train extra). Asked Pranav to allow both hosts. Until then: no training, no PPO probe, no pyproject dependency changes; `onnx==1.23.2` and `protobuf==7.36.2` are installed into `.venv` by hand (`uv pip install`) for the torch-free slices |
| Direction update | `PLAN.md` section 0 (2026-10-11): workbench for people's own robot projects, Studio (desktop + browser), Arena with FRC drive practice, revised build order |
| Deferred risks | `docs/design/deferred-risks.md`: things later phases must fix (e.g. a hidden mass below the floor makes any robot a passive pendulum) |
| Phase 3 design inputs | Branch `phase-3-design`: draft `docs/design/phase-3.md` + `phase-3-critiques.json` (55 issues, 7 BLOCKER, 21 HIGH), not yet revised. Never merge that branch wholesale; cherry-pick only the phase-3 docs |

Work branch: **`phase-2-learn`** (based on `main`). Its PR should be opened when Phase 2 is built.

## Next steps, in order

1. ~~Ask Pranav the open questions~~ (answered 2026-10-15; see below).
2. ~~Linux re-measure of calibration~~ (done: E_ref 13.1, see the table). **Mac session: re-pin the
   darwin-arm64 golden scores** in `tests/test_golden.py` (the STALE comment there points here).
3. **PPO confirmation probe** (Q4: yes): one ~7-minute run of the §0 recipe on the recalibrated env,
   before any slice-3 training. Blocked in the cloud until torch installs (Network row above).
4. **Build Phase 2a** from `docs/design/phase-2.md`, in small slices, each with its own review:
   1. ONNX validator + canonicalization, data-only `topple.policy/1` submissions, hardened
      `ONNXPolicy` runtime (no training needed);
   2. training pipeline (PPO config, wrappers, action-excess penalty, selection on seeds 2000..2199,
      CPU budgets) + torch-free `topple export` + parity tests;
   3. train 3 seeds, export, run done-when gates G1–G10, side-by-side PID vs PPO video.
5. Phase 2a review gate (below), fix, merge. Then Phase 2b (waypoint-v1, `topple.traj/2`).
6. Then follow the revised build order in `PLAN.md` section 0.

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
