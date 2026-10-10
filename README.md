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
uv run pytest          # tests + coverage
uv run topple smoke    # check MuJoCo simulates and renders headlessly -> smoke.png
```

Headless rendering picks a backend automatically: CGL on macOS, OSMesa on headless Linux
(`apt-get install libosmesa6`), GLFW when a display is present. Override with `MUJOCO_GL`.
On Apple Silicon, uv is configured to prefer the native arm64 Python.
