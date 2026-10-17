# Example submissions

| File | Controller |
|---|---|
| `zero_example.py` | lesson 1: never drives the wheels, so the robot falls |
| `random_example.py` | random commands drawn from the episode's `rng_seed` |
| `pid_example.py` | cascaded PID baseline (`topple/controllers/pid.py`) |
| `lqr_example.py` | LQR baseline designed from the simulator (`topple/controllers/lqr.py`) |
| `waypoint_pid_example.py` | waypoint-v1 reference driver (`topple/controllers/waypoint_pid.py`) |
| `anchors/*.py` | waypoint-v1 degenerate-strategy anchors (`topple/controllers/anchors.py`) |

## The contract in 10 lines

Full text: [docs/contract.md](../docs/contract.md) §6–8.

1. A submission is a `.py` file (or a directory with `controller.py`) defining `make_controller(spec)`.
2. It is called once per episode and returns a new controller; `make_controller()` with no arguments also works.
3. `spec` (`TaskSpec`) holds the challenge id, observation fields, action spec, `control_dt` (0.01 s), `episode_seconds` (10 s) and `spec.robot.wheel_radius_m` / `track_width_m`.
4. The controller has `reset(rng_seed)`, called at the start of each episode, and `act(obs)`, called at 100 Hz.
5. `obs` (`BalancerObs`) has `t`, `pitch`, `pitch_rate`, `yaw`, `yaw_rate`, `wheel_pos`, `wheel_vel` and `last_action`, in SI units; pairs are `(left, right)`.
6. Positive pitch leans forward, positive wheel motion rolls the robot forward, positive yaw turns left.
7. `act` returns `(left, right)` motor commands; values outside [-1, 1] are clipped.
8. Anything but two finite numbers, or an exception, ends the episode with an error and a score of 0.
9. Take all randomness from `rng_seed`, never the clock or the global `random` module.
10. Controllers must not share mutable module-level state.

## Run one

```bash
uv run topple eval submissions/pid_example.py                       # 20 public seeds of balance-v1/standard
uv run topple eval submissions/lqr_example.py -p balance-v1/clean   # no pushes
uv run topple run submissions/zero_example.py                       # one episode and a video
```

`run` and `eval` import the controller and run it in their own process, with your privileges:
only run code you trust (someone else's controller belongs in the sandboxed runner, Phase 3).
