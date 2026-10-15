# Handoff: current state and how to continue

This file moves work between Claude Code sessions (local Mac ⇄ cloud at claude.ai/code).
**Whoever stops working must update it, commit it and push before stopping.** Nothing else
carries over: local memory, scratch files and workflow scripts stay on the machine that made them.

Last updated: 2026-10-14 by the local session (handing off to cloud).

## Where things stand

| Item | State |
|---|---|
| Phase 0 (setup) | Merged to `main` (fcc2b13) |
| Phase 1 (sim core, balance-v1) | Merged to `main` as **f997394** (PR #3). Passed two review gates (the second all-Opus); every confirmed finding fixed; CI green (1,806 tests, 98.9% coverage) |
| Phase 2 (learn) | **Designed, not built.** Spec v5.2 in `docs/design/phase-2.md` (synced with merged Phase 1 and verified against the code); decision log `docs/design/phase-2-decisions.md` |
| Direction update | `PLAN.md` section 0 (2026-10-11): workbench for people's own robot projects, Studio (desktop + browser), Arena with FRC drive practice, revised build order |
| Deferred risks | `docs/design/deferred-risks.md`: things later phases must fix (e.g. a hidden mass below the floor makes any robot a passive pendulum) |
| Phase 3 design inputs | Branch `phase-3-design`: draft `docs/design/phase-3.md` + `phase-3-critiques.json` (55 issues, 7 BLOCKER, 21 HIGH), not yet revised. Never merge that branch wholesale; cherry-pick only the phase-3 docs |

Work branch: **`phase-2-learn`** (based on `main`). Its PR should be opened when Phase 2 is built.

## Next steps, in order

1. **Ask Pranav the open questions below** (or use the stated default if he says "use defaults").
2. **Linux re-measure of calibration** (deferred risk (c)). The cloud machine is linux-x86_64, so
   do it there first: re-run the PID/LQR calibration sweeps on seeds 0..199 for `balance-v1/standard`
   and compare with `challenge.yaml` (E_ref 13.0 J; darwin: LQR 86.5%, PID 64.5% survival). If E_ref
   would change, stop and tell Pranav before training anything (it would stale every protocol hash).
3. **Build Phase 2a** from `docs/design/phase-2.md`, in small slices, each with its own review:
   1. ONNX validator + canonicalization, data-only `topple.policy/1` submissions, hardened
      `ONNXPolicy` runtime (no training needed);
   2. training pipeline (PPO config, wrappers, action-excess penalty, selection on seeds 2000..2199,
      CPU budgets) + torch-free `topple export` + parity tests;
   3. train 3 seeds, export, run done-when gates G1–G10, side-by-side PID vs PPO video.
4. Phase 2a review gate (below), fix, merge. Then Phase 2b (waypoint-v1, `topple.traj/2`).
5. Then follow the revised build order in `PLAN.md` section 0.

## Open questions for Pranav

From the Phase 2 spec sync (defaults in brackets):
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
