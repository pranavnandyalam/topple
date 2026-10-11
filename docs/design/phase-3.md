# Phase 3 design: Robot CI (draft for critique)

PLAN.md Phase 3: `runner.py` subprocess sandbox; `eval.yml` (`pull_request`, `permissions: {}`)
producing artifacts; `publish.yml` (`workflow_run`) validating artifacts, posting a PR comment with
an embedded GIF + metrics + rank and updating the leaderboard; a reusable action. Done when: a fork
PR with a new controller gets a comment with GIF + score in < 3 min; red-team tests (score
tampering, infinite loop, network access, huge output) all fail cleanly. The repo goes public only
after this phase's security review passes (Pranav confirms).

Facts from Phases 1–2: `topple.contract` imports only the stdlib (fast child start); observations
have a lossless 88-byte wire format; trajectories re-simulate bit-exactly from scenario + ctrl
(open-loop verification), so a trusted stage can re-score without running submission code;
`results.json` (`topple.results/1`) has an `isolation` field and only `container` is rankable;
policies ship as ONNX (data) + a tiny `controller.py`.

## 1. Runner (`topple/runner.py`)

- Host process (trusted) owns the env, scoring and seeds. The controller runs in a child process
  started with `python -I -m topple.child <submission>` (isolated mode, no user site, no cwd on
  `sys.path`), communicating over two dedicated pipes (`pass_fds`), never stdin/stdout.
- Frames: request `<B` op + payload; ops `INIT(spec json)`, `RESET(rng_seed u64)`,
  `ACT(obs 88 B)`, `CLOSE`; reply `<B` status + payload (`ACT` → 2 × f64). The host validates only
  decoded numbers; it never unpickles anything.
- Limits per protocol (hashed into the protocol): load 30 s, reset 2 s, act 0.25 s wall, episode CPU
  30 s, RSS 1 GiB (`RLIMIT_AS` on Linux), stderr capped at 64 KiB, `RLIMIT_NPROC`, no core dumps.
  Timeouts → `timeout` error; crash/exit → `crash`; malformed frame → `protocol`; output cap →
  `output_limit`.
- One child per episode (no state leaks between episodes); `make_controller` once per episode.
- `topple eval --isolation subprocess|container|inprocess` (default subprocess for files that are
  not built-in).

## 2. Container

- `ghcr.io/<owner>/topple-eval:<version>` (Python 3.12 slim + pinned wheel + OSMesa), digest
  pinned in workflows. Run: `docker run --rm --network none --read-only --tmpfs /tmp:size=64m
  --cpus 2 --memory 2g --pids-limit 128 --cap-drop ALL --security-opt no-new-privileges`
  `--user 65534` (child) while the host process runs as another uid inside (seeds unreadable to
  the child).

## 3. Workflows

- `eval.yml` on `pull_request` (fork-safe): `permissions: {}`, no secrets, timeout 10 min,
  `paths: submissions/**`. Detect changed submissions (max N per PR), evaluate each on public
  seeds in the container, upload `results.json` + `traj_*.npz` as artifacts. No rendering, no
  comments, no network after image pull.
- `publish.yml` on `workflow_run` (trusted, `pull-requests: write`, `contents: write` only for the
  leaderboard commit on main): never checks out PR code; downloads artifacts as untrusted data;
  validates schemas with size caps; re-verifies every trajectory by open-loop re-simulation and
  re-scores it (reported scores are ignored); renders GIFs from verified qpos; posts/updates one
  sticky PR comment (metrics table, rank vs leaderboard, GIF via `gh pr comment --attach` or a
  `ci-media` orphan branch); escapes all untrusted strings (error details, names).
- `nightly.yml`: re-evaluates the current top-k on private seeds (secret salt) in the container;
  updates `LEADERBOARD.md` with private scores; never publishes private trajectories.

## 4. Reusable action (`action/action.yml`)

- Composite action `owner/topple/action@v1` with inputs `controller`, `protocol`,
  `robot` (optional MJCF for "bring your own robot"), `isolation`; outputs score/success and the
  results path, so other repos get CI-for-robots in five lines of YAML.

## 5. Red-team suite (`tests/redteam/`)

Malicious submissions, each must end as a clean error (or score 0) without affecting the host:
infinite loop in `act`, infinite loop at import, memory bomb, fork bomb, `os.system`/subprocess,
network access, writing outside `/tmp`, reading `/proc/<host>/` or env vars, huge stderr, segfault
(`ctypes`), returning NaN/strings/huge lists, monkeypatching `topple` (child-only, harmless),
sleeping to stall the host, a forged `results.json`/`traj.npz` in the artifact (caught by
re-verification), a valid trajectory replayed with a different scenario (caught), path traversal
in artifact names.
