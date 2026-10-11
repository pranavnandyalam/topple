"""Headless MuJoCo smoke test: load a model, step it, render one frame to PNG.

Usage: ``python -m topple.smoke [--out smoke.png] [--steps 1000]``
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import mujoco

from topple._atomic import SymlinkedOutputError, check_output_directory, replacement
from topple.rendering import RenderingUnavailableError, make_renderer

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

    renderer = make_renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data)
        pixels = renderer.render()
    finally:
        renderer.close()

    # The default output is ``smoke.png`` in the current directory, which can be a checkout of
    # someone else's branch: never write through a symlink planted there (contract §14).
    with replacement(out_path, suffix=out_path.suffix) as tmp:  # imageio needs the suffix
        iio.imwrite(tmp, pixels)
    return SmokeResult(
        image_path=out_path,
        steps=steps,
        timestep=float(model.opt.timestep),
        sim_time=float(data.time),
        initial_angle=initial_angle,
        final_angle=float(data.qpos[0]),
    )


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("smoke.png"))
    parser.add_argument("--steps", type=_positive_int, default=1000)
    args = parser.parse_args(argv)
    try:  # a relative --out never passes through a symlink (contract §14)
        check_output_directory(args.out.parent)
    except SymlinkedOutputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        print(f"error: cannot write to {args.out.parent}: {reason}", file=sys.stderr)
        return 2
    try:
        result = run_smoke(args.out, steps=args.steps)
    except RenderingUnavailableError as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        return 1
    print(f"smoke ok: {result.steps} steps, t={result.sim_time:.3f}s -> {result.image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
