# Topple

**Can your controller survive being toppled?**

CI for robots: design a robot, push a controller, and get back a deterministic score, a crash-test
rating, an AI coach's fix and a replay anyone can watch and shove in the browser. Built on MuJoCo,
Gymnasium and Stable-Baselines3. Mac-native, zero-install to try.

Status: building. See [PLAN.md](PLAN.md).

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                # add --all-extras for RL training (torch, stable-baselines3)
uv run pytest          # tests (add --cov for the coverage report and 80% gate)
uv run topple smoke    # check MuJoCo simulates and renders headlessly -> smoke.png
```

`topple run` and `topple eval` import the controller and run it in their own process, with your
privileges: only run code you trust. Untrusted submissions go through the sandboxed runner
(Phase 3).

Headless rendering picks a backend automatically: CGL on macOS, OSMesa on headless Linux
(`apt-get install libosmesa6`), GLFW when a display is present. Override with `MUJOCO_GL`.
On Apple Silicon, uv prefers a system (Homebrew) Python over uv-managed ones. If `uv sync` reports
an incompatible platform, a Rosetta (x86_64) interpreter was picked; fix it with
`UV_PYTHON=cpython-3.12-macos-aarch64-none uv sync` or remove it with
`uv python uninstall cpython-3.12-macos-x86_64-none`.
