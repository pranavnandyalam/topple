"""Headless MuJoCo smoke test: load a model, step it, render one frame to PNG.

Usage: ``python -m topple.smoke [--out smoke.png] [--steps 1000]``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import mujoco

SMOKE_XML = """
<mujoco model="smoke-pendulum">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.75 0.78 0.82 1"/>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass(frozen=True)
class SmokeResult:
    image_path: Path
    steps: int
    timestep: float
    sim_time: float
    initial_angle: float
    final_angle: float


def run_smoke(
    out_path: Path,
    steps: int = 1000,
    width: int = 320,
    height: int = 240,
    initial_angle: float = 0.5,
) -> SmokeResult:
    """Step a tilted pendulum ``steps`` times and render the final frame to ``out_path``."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if width < 1 or height < 1:
        raise ValueError(f"image size must be positive, got {width}x{height}")

    model = mujoco.MjModel.from_xml_string(SMOKE_XML)
    data = mujoco.MjData(model)
    data.qpos[0] = initial_angle
    for _ in range(steps):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data)
        pixels = renderer.render()
    finally:
        renderer.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, pixels)
    return SmokeResult(
        image_path=out_path,
        steps=steps,
        timestep=float(model.opt.timestep),
        sim_time=float(data.time),
        initial_angle=initial_angle,
        final_angle=float(data.qpos[0]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("smoke.png"))
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args(argv)
    result = run_smoke(args.out, steps=args.steps)
    print(f"smoke ok: {result.steps} steps, t={result.sim_time:.3f}s -> {result.image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
