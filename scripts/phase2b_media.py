"""The Phase 2b media (docs/design/phase-2.md §8.9 item 8), from the committed comparisons and the
shipped submission; no number is chosen by hand.

``uv run python scripts/phase2b_media.py``

- ``docs/results/phase2/waypoint.{mp4,gif}``: the driver and ``ppo_waypoint`` on one course in one
  scene, on the ``wp_eval_v1`` seed at index 24 of the 50 sorted by (PPO score, seed). MP4
  720x720 at 50 fps (at most 8 MB); GIF 480x480 on the waypoint ladder (at most 5 MB).
- ``docs/results/phase2/wp_gallery_{1,2,3}.gif``: the shipped policy's 3 lowest-scoring seeds on
  ``standard`` 0..399, overview GIFs at 480x480 on the gallery ladder from 10 fps (at most 3 MB
  each), captioned with ``fall_cause`` (the HUD adds t_fall and the gate reached).

Episodes are re-run from the shipped submission (deterministic per platform); each must score
what the committed comparison recorded, so the media show the reported episodes.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Final

from phase2b_common import DRIVER_ID, Phase2bError, parse_report
from topple.compare import Entry, read_comparison
from topple.course_video import render_course, render_course_gif
from topple.evaluate import EvalProtocol, iter_episodes
from topple.loader import load_submission
from topple.traj import Trajectory
from topple.video import WAYPOINT_LADDER

REPO: Final = Path(__file__).resolve().parents[1]
RESULTS: Final = REPO / "docs/results"
OUT: Final = RESULTS / "phase2"
SHIPPED: Final = REPO / "submissions/ppo_waypoint"
DRIVER: Final = REPO / "submissions/waypoint_pid_example.py"
HERO_INDEX: Final = 24
GALLERY: Final = 3
MB: Final = 1 << 20


def episode(sub: Path, protocol_id: str, seed: int) -> tuple[Trajectory, float]:
    proto = replace(EvalProtocol.from_id(protocol_id), seeds=(seed,))
    factory = load_submission(sub, challenge=proto.challenge).factory
    run = next(iter(iter_episodes(proto, factory)))
    return run.trajectory, run.result.score


def entry(cmp_path: Path, entry_id: str) -> tuple[Entry, tuple[int, ...]]:
    cmp = read_comparison(cmp_path)
    found = [e for e in cmp.entries if e.id == entry_id]
    if len(found) != 1:
        raise Phase2bError(f"{cmp_path.name}: no single entry {entry_id!r}")
    return found[0], cmp.seeds


def checked(sub: Path, protocol_id: str, seed: int, score: float) -> Trajectory:
    traj, got = episode(sub, protocol_id, seed)
    if got != score:
        raise Phase2bError(f"{sub.name} seed {seed} scores {got}, the comparison {score}")
    return traj


def main() -> int:
    report = parse_report((RESULTS / "phase2b.json").read_bytes())
    run = report.shipped.run
    if run is None:
        print("phase2b_media: nothing shipped", file=sys.stderr)
        return 2
    try:
        ppo, seeds = entry(RESULTS / "phase2b/runs_wp_eval_v1.json", run)
        drv, _ = entry(RESULTS / "phase2b/runs_wp_eval_v1.json", DRIVER_ID)
        order = sorted(range(len(seeds)), key=lambda k: (ppo.scores[k], seeds[k]))
        k = order[HERO_INDEX]
        hero = [checked(SHIPPED, "wp_eval_v1", seeds[k], ppo.scores[k]),
                checked(DRIVER, "wp_eval_v1", seeds[k], drv.scores[k])]  # fmt: skip
        labels = ["ppo_waypoint", "driver"]
        render_course(hero, labels, OUT / "waypoint.mp4", fps=50, size=720)
        if (OUT / "waypoint.mp4").stat().st_size > 8 * MB:
            raise Phase2bError("waypoint.mp4 is over 8 MB")
        render_course_gif(hero, labels, OUT / "waypoint.gif", WAYPOINT_LADDER, max_bytes=5 * MB)
        std, std_seeds = entry(RESULTS / "phase2b/runs_standard.json", run)
        worst = sorted(range(len(std_seeds)), key=lambda i: (std.scores[i], std_seeds[i]))
        for n, i in enumerate(worst[:GALLERY], 1):
            seed = std_seeds[i]
            traj = checked(SHIPPED, "waypoint-v1/standard", seed, std.scores[i])
            cause = std.extras["fall_cause"][i]
            gif = OUT / f"wp_gallery_{n}.gif"
            label = f"seed {seed}, fall cause {cause}"
            render_course_gif([traj], [label], gif, WAYPOINT_LADDER, max_bytes=3 * MB)
            print(f"gallery {n}: seed {seed} score {std.scores[i]:.2f} cause {cause}")
    except Phase2bError as exc:
        print(f"phase2b_media: {exc}", file=sys.stderr)
        return 2
    print(f"hero: wp_eval_v1 seed {seeds[k]}: ppo {ppo.scores[k]:.2f}, driver {drv.scores[k]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
