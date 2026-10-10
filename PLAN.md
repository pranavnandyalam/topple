# Robot Sim Lab: Plan

Design a robot in CAD, drop it into a physics sim, train a neural-network policy to control it, and
**push controller code to the simulated robot** the way you'd push to a real one: every push runs
the robot in sim and reports back with metrics and a video.

Merges two ideas: *"train a robot in an environment with a neural network"* and *"model a robot in an
environment and push code to it in a simulated environment."*

---

## 1. Goal and scope

**One-sentence pitch:** CI for robots. `git push` a controller, get back a score, a leaderboard
position and an MP4 of your robot doing (or failing) the task.

**What "done" looks like (v1):**
1. A two-wheeled self-balancing robot, designed in Onshape, running in MuJoCo.
2. A trained RL policy (PPO) that balances it and drives to waypoints under disturbances.
3. A hand-written PID baseline, so you can show learned vs. classical control.
4. A `simlab` CLI: `simlab run controller.py --task balance` gives metrics JSON and an MP4.
5. A GitHub Action that runs `simlab eval` on every push/PR and comments results with a video link.
6. A write-up: what worked, reward-shaping lessons, PID vs. RL comparison, sim-to-real gaps.

**Explicit non-goals for v1:** multi-user hosted platform, browser IDE, custom physics engine,
real hardware. All of these are stretch phases.

### Why a self-balancing robot first
- Simple CAD (chassis, 2 wheels, 2 motors) that you could actually build later for about $50. That's a sim-to-real story.
- Trains in **minutes on CPU**, so iteration is fast on the M5 Pro.
- Classical controller (PID/LQR) is well understood, so it makes a meaningful baseline for RL.
- The same pipeline later scales to an arm or quadruped by swapping the MJCF + task.

---

## 2. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language / env | Python 3.12 via `uv` | Fast, reproducible; 3.14 is too new for some wheels |
| Physics | **MuJoCo** (`mujoco` pip pkg) | Fast, accurate contacts, runs great on Apple Silicon CPU |
| CAD to sim | **Onshape** to `onshape-to-robot` to URDF/MJCF | Design in CAD, export directly; no hand-written XML |
| RL API | **Gymnasium** custom env | Standard interface; works with every RL library |
| RL algorithms | **Stable-Baselines3** (PPO, SAC) | Battle-tested, minimal code; swap to CleanRL if you want to own the algorithm |
| Logging | TensorBoard (W&B optional) | Reward curves, episode stats |
| Video | `mujoco.Renderer` + `imageio[ffmpeg]` | Headless MP4 per eval |
| CLI | `typer` | Clean CLI with little code |
| Tests | `pytest` + `pytest-cov` | 80%+ coverage target |
| CI runner | GitHub Actions (ubuntu, CPU) | The "push code to robot" loop with zero infra |

**Compute reality:** small MLP policies don't benefit from the GPU. MuJoCo plus SB3 with 16 parallel
envs (`SubprocVecEnv`) on 18 CPU cores is the right setup. GPU-accelerated sim (MJX / MuJoCo
Playground, Isaac Lab) needs NVIDIA, so that's only for the quadruped stretch phase, via the UMich GPU
offering or cloud.

---

## 3. Architecture

```
robot-sim-lab/
├── assets/
│   ├── onshape/            # onshape-to-robot config.json (document URL, settings)
│   └── robots/balancer/    # exported MJCF + meshes (generated, committed)
├── src/simlab/
│   ├── envs/
│   │   ├── base.py         # MujocoRobotEnv: load model, step, obs, render
│   │   ├── balance.py      # BalanceTask: stay upright, rewards, termination
│   │   └── waypoint.py     # WaypointTask: balance + drive to targets
│   ├── disturbances.py     # random pushes, sensor noise, mass/friction randomization
│   ├── controller_api.py   # Controller protocol: reset(), act(obs) -> action
│   ├── controllers/
│   │   ├── pid.py          # classical baseline
│   │   └── policy.py       # wraps a trained SB3 checkpoint as a Controller
│   ├── train.py            # PPO/SAC training entrypoint (config-driven)
│   ├── evaluate.py         # N seeded episodes -> metrics dict
│   ├── runner.py           # runs untrusted controller.py in a subprocess w/ timeout
│   ├── video.py            # render episode -> MP4
│   └── cli.py              # `simlab run | eval | train | export`
├── configs/                # YAML: env params, reward weights, PPO hyperparams
├── submissions/            # example controllers (pid_example.py, random.py)
├── tests/
└── .github/workflows/eval.yml
```

### The controller contract (the core interface everything else hangs off)
```python
class Controller(Protocol):
    def reset(self, seed: int) -> None: ...
    def act(self, obs: Observation) -> np.ndarray: ...   # wheel torques, clipped to limits
```
- `Observation` is a frozen dataclass: pitch, pitch rate, wheel velocities, (waypoint task) relative target.
- PID controllers, trained policies and user submissions all implement this, so they're all
  evaluated identically. That's what makes "push code to the robot" fair and comparable.

### Evaluation protocol (fixed, versioned)
- `N=20` episodes, fixed seed list, fixed disturbance schedule.
- Metrics: `success_rate`, `mean_time_upright`, `max_tilt`, `energy_used`, `waypoint_error`, `score` (weighted).
- Protocol versioned (`eval_v1`), so leaderboard results stay comparable over time.

---

## 4. Phases

Each phase has a concrete **done-when** check. Don't start the next phase until it passes.

### Phase 0: Setup (1 evening)
- `uv init`, pin Python 3.12, add deps, `ruff` + `pytest` config, `.superset/config.json` setup script (`uv sync`).
- Smoke test: load a MuJoCo Menagerie model, step 1000 times, render one frame headless (`MUJOCO_GL=cgl` on macOS; verify).
- Fix the `onshape` MCP server (its venv at `~/onshape-mcp/venv` is missing) if you want Claude to drive Onshape.
- **Done when:** `uv run pytest` passes and `uv run python -m simlab.smoke` writes `smoke.png`.

### Phase 1: Robot model (about 1 week)
- **1a. Placeholder first:** hand-write a minimal balancer MJCF (box body, 2 cylinder wheels, 2 hinge motors) so Phases 2–3 aren't blocked on CAD.
- **1b. Real CAD:** design the balancer in Onshape (chassis, battery mass, motor mounts, wheels). Name mates the way `onshape-to-robot` expects (`dof_left_wheel`, `dof_right_wheel`).
- Export with `onshape-to-robot` to MJCF; set realistic masses, motor torque limits, wheel friction.
- **Done when:** the CAD robot loads, falls over realistically with zero torque, and wheels spin when torque is applied (tests assert this).

### Phase 2: Gym environment + PID baseline (about 1 week)
- `BalanceTask`: observation, action space (2 wheel torques), reward = upright bonus − tilt² − control effort, terminate at |pitch| > 45°.
- Disturbances module: random impulse pushes, sensor noise, ±10% mass/friction randomization.
- PID controller tuned by hand. This is your first "submission" and a sanity check that the task is solvable.
- **Done when:** `gymnasium.utils.env_checker` passes; PID balances >90% of eval episodes without pushes.

### Phase 3: Train the neural network (1–2 weeks)
- PPO via SB3, 16 parallel envs, TensorBoard logging, checkpoint every N steps, config-driven.
- Curriculum: no disturbances, then small pushes, then full randomization.
- Then `WaypointTask` (balance + drive to targets): reuse the balance policy or train from scratch and compare.
- Try SAC as a comparison; record wall-clock and sample efficiency.
- **Done when:** the RL policy beats PID on `eval_v1` *with* disturbances, and a side-by-side video exists.

### Phase 4: "Push code to the robot" (about 1 week)
- `simlab run controller.py`: load the user file, run it in a **subprocess with a timeout and resource limits**, emit `results.json` + `episode.mp4`.
- GitHub Action `eval.yml`: on push/PR, if `submissions/**` changed, run eval, upload MP4 as artifact, post a PR comment with a metrics table.
- `LEADERBOARD.md` auto-updated on main.
- **Done when:** you open a PR with a new controller and the bot comments a score + video link within ~3 minutes.

### Phase 5: Polish + write-up (about 1 week)
- README with GIFs, architecture diagram, PID vs. PPO comparison chart, reward-design lessons.
- Short demo video (60–90s).
- **Done when:** a stranger can clone, `uv sync`, run `simlab run submissions/pid_example.py`, and see a video.

### Stretch phases (pick by interest)
- **S1 Web viewer:** watch episodes in the browser with MuJoCo WASM (e.g. `zalo/mujoco_wasm`) plus a Monaco editor to write `act()` in-browser. This is the "app" version of the idea.
- **S2 Bigger robot:** arm (reach/pick) or quadruped from MuJoCo Menagerie, trained with MuJoCo Playground on a GPU (UMich / cloud).
- **S3 Sim-to-real:** build the balancer (ESP32 + IMU + 2 motors), deploy the policy, measure the gap. Very strong portfolio piece.
- **S4 Hosted platform:** multi-user submissions in Docker sandboxes (no network, CPU/mem caps), public leaderboard.

---

## 5. Testing strategy (TDD, 80%+ coverage)
- **Unit:** reward function terms, termination logic, observation construction, action clipping, PID math, config parsing.
- **Determinism:** same seed gives identical trajectory (critical for fair eval).
- **Contract:** every controller in `submissions/` satisfies the `Controller` protocol; malformed controllers fail cleanly.
- **Runner safety:** infinite loop is killed by the timeout; exceptions are reported, not crashed; wrong action shape is rejected.
- **Integration:** `simlab eval` on PID produces valid `results.json` with all metrics.
- **Regression:** a tiny "golden" PID score range; CI fails if a sim change shifts it unexpectedly.
- Training itself is not unit-tested; a 2k-step smoke train in CI ensures the pipeline runs.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Onshape export pain (mates, mesh scale, inertia) | Phase 1a placeholder MJCF unblocks everything; CAD is swapped in later |
| Reward hacking (policy vibrates, rolls away "upright") | Penalize control effort + drift; watch videos, not just reward curves |
| RL doesn't converge | Verify with PID that the task is solvable; start without disturbances; tune with known-good PPO defaults |
| Headless rendering on Mac/CI | Verify `cgl` (Mac) / `egl` or `osmesa` (Linux CI) in Phase 0 |
| Running untrusted controller code | Subprocess + timeout for v1 (your own code); Docker sandbox before accepting anyone else's |
| Scope creep into a "platform" | Stretch phases stay stretch until Phase 5 ships |

---

## 7. Open decisions
1. **Robot:** self-balancer (recommended) vs. 4-DOF arm vs. quadruped from Menagerie.
2. **CI host:** GitHub Actions (recommended, free) vs. a local runner on the Mac.
3. **Repo visibility:** public from day one (good for portfolio) or after Phase 3.
4. **Stretch priority:** S1 web viewer (more "product") vs. S3 real hardware (more "engineering").

---

## 8. Resources
- MuJoCo docs: https://mujoco.readthedocs.io
- MuJoCo Menagerie (ready-made robot models): https://github.com/google-deepmind/mujoco_menagerie
- onshape-to-robot: https://github.com/Rhoban/onshape-to-robot
- Gymnasium custom env guide: https://gymnasium.farama.org
- Stable-Baselines3: https://stable-baselines3.readthedocs.io
- MuJoCo Playground (GPU stretch): https://github.com/google-deepmind/mujoco_playground
