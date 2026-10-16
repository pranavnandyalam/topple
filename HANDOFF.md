# Handoff: current state and how to continue

This file moves work between Claude Code sessions (local Mac ⇄ cloud at claude.ai/code).
**Whoever stops working must update it, commit it and push before stopping.** Nothing else
carries over: local memory, scratch files and workflow scripts stay on the machine that made them.

Last updated: 2026-10-15 21:40 by the local Mac session. Branch `phase-2-learn` is pushed and clean at
the commit that adds this text. The training runs live only on the Mac, in `runs/` (gitignored).

## Where things stand

| Item | State |
|---|---|
| Phase 0 (setup) | Merged to `main` (fcc2b13) |
| Phase 1 (sim core, balance-v1) | Merged to `main` as **f997394** (PR #3). Passed two review gates (the second all-Opus); every confirmed finding fixed; CI green (1,806 tests, 98.9% coverage) |
| Phase 2 (learn) | **2a torch-free half built and reviewed; torch half not started.** Spec v5.3 (`docs/design/phase-2.md`) matches the code; decision log rows R1–R7 (recalibration), A1–A8 (answers), B1–B36 (slices). Done, each with builder → reviewers → skeptics → fixer → re-review: **1a** ONNX builder/validator/canonical form (`08a47b7`, `c0469d6`); **1b** manifest, policy dirs, ONNXPolicy, pools, loader/results (`0da4ce6`, `9e2153f`); **1c** policy pack, compare + stats, shove bench, side-by-side video, hygiene tests (`dbe4203`, `eaf4d54`); **2 (torch-free)** yamlsafe, train config + 5 YAMLs, wrappers, selection, `_npz`, `topple export`, `topple train` shell (`5d13f3e`, `6249245`, `c735213`); **3a** gate scripts (`scripts/phase2_*`), C reference parity (`tests/c/`), CI jobs, coverage scopes (`ceff1d3`, fix `4fe6e7f`). Fast suite ≈3,400 tests green on linux (`-m "not slow and not train"`) |
| Slice 3a re-review | **Done** (local, 2026-10-15): re-review of `4fe6e7f` found 3 MEDIUM + 4 LOW, all fixed (`143ad0f`, `c877be8`, `c1b6d1e`, `7293486`); spec §4/§6/§7 synced as B37–B45 (`677efeb`). The fixes get their re-review in the Phase 2a gate |
| CI trigger | `ci.yml` runs on PRs and pushes to `main` and `phase-*` (B12 landed). The train job runs without its 80% coverage scope until `topple/train/learn.py` exists (B37) |
| Merge blocker B12 | **Done** (`1537db1`): onnx/protobuf base deps, onnxruntime bounded, psutil/matplotlib added, `uv.lock` regenerated on the Mac |
| Linux recalibration | **Done** (2026-10-15, Pranav chose it): E_ref 13.0 → **13.1 J** on linux-x86_64, now the calibration platform. Bands hold (LQR 87.5%, PID 65.0%). New bundle `89a62007…`, protocol hashes and linux golden scores re-pinned. Reviewed (2 reviewers + 2 skeptics), findings fixed |
| Darwin golden pins | **Re-pinned** under E_ref 13.1 J (`07681c8`) |
| Network (cloud) | The cloud environment blocks `download.pytorch.org` and `download-r2.pytorch.org`, so torch could not be installed and `uv lock` could not run there. On the Mac: `uv lock` works; then `uv sync` (it removes nothing needed: onnx/protobuf become base deps) |
| Direction update | `PLAN.md` section 0 (2026-10-11): workbench for people's own robot projects, Studio (desktop + browser), Arena with FRC drive practice, revised build order |
| Deferred risks | `docs/design/deferred-risks.md`: things later phases must fix (e.g. a hidden mass below the floor makes any robot a passive pendulum) |
| Phase 3 design inputs | Branch `phase-3-design`: draft `docs/design/phase-3.md` + `phase-3-critiques.json` (55 issues, 7 BLOCKER, 21 HIGH), not yet revised. Never merge that branch wholesale; cherry-pick only the phase-3 docs |

Work branch: **`phase-2-learn`** (based on `main`). Its PR should be opened when Phase 2 is built.

## Next steps, in order

**Done since the last handoff (local Mac, 2026-10-15):**
- Torch half of slice 2 built (`03ae4e3`, fixtures `6f298e2`, docs B46–B51 `87524d7`).
- Reviewed by one agent; 1 HIGH, 3 MEDIUM and 2 LOW fixed (`f9109fe`, B52).
- All slice 3b training runs done on `f9109fe` with a clean tree (`git_dirty: false`). They are in `runs/` on the Mac. Selection seeds 2000..2199, PID measured in the same run: 70.65 / 0.655.

  | Run | Best step | Score / success / energy | Eligible | Flags |
  |---|---|---|---|---|
  | ppo_s0 | 3,506,176 | 86.63 / 0.955 / 2.16 J | yes | none |
  | ppo_s1 | 1,753,088 | 80.62 / 0.815 / 4.12 J | yes | high_energy (early evals only) |
  | ppo_s2 | 5,013,504 | **87.66 / 0.955 / 2.51 J** | yes | high_energy (early evals only) |
  | sac_s0..2 | — | 67–74 / 0.43–0.62 | no | no_eligible (as §0 predicts) |
  | ablate_nopenalty | — | 71.47 / 0.525 / 23.3 J | no | high_energy, no_eligible |

  - ppo_s1's later checkpoints reach 86.3 / 0.905 but saturate above the 0.08 selection cap, so selection kept its 1.75M checkpoint.
  - ppo_s0 reproduced bit-for-bit an earlier (discarded, dirty-tree) run of the same seed.
  - Expected shipping order by rank_key: ppo_s2, then ppo_s0, then ppo_s1.

**Next:**
1. **Build `scripts/phase2_results.py`** (spec §5, and §2.5 "Shipping"). It does not exist yet. It must:
   - export candidates in rank order with `topple export`;
   - run fresh evaluations on eval_v1 and 0..199;
   - call `choose_shipped` and the `SHIP_GATES` (G1–G6) from `scripts/phase2_common.py`;
   - write `docs/results/phase2/phase2.json` (`topple.phase2/1`), `phase2.md`, the learning curves, the shove test and the side-by-side PID vs PPO hero video;
   - commit `submissions/ppo_balance`;
   - support `--fetch-ci` (B40: push runs of HEAD's own code, `ci_ubuntu/source.json`).

   Build it in the main session with TDD, then one reviewer at max effort (it feeds the gates).
2. Push. Get CI green, including the ubuntu compare of the shipped policy. Run `phase2_results.py --fetch-ci`, then `scripts/phase2_gate.py` / `phase2_done.sh` (G1–G11).
3. Phase 2a review gate: fresh-clone done-when verifier, 2–3 reviewers, batch skeptics. Then fix, open the PR and merge.
4. Then Phase 2b (waypoint-v1, `topple.traj/2`). In 2b, pass the challenge kind to `choose_best` (B52 note). Then PLAN.md section 0's build order.

Process notes from this session: each slice ran as one builder → 2–3 reviewers (max effort) → 2
skeptics → fixer → re-review, then a docs agent synced the spec. See "Effort and cost settings"
below for the leaner setup to use from now on.

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
