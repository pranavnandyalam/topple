"""Dispatch by challenge kind (docs/design/phase-2.md §8.1): which env runs a challenge, how a
trajectory of it is scored and which trajectory layout its files use.

Evaluation, verification, rescoring, the eval pool and the CLI's upload loaders go through these
three functions, never through ``BalanceEnv`` or the balance scorer directly, so a new kind is
one more branch here. balance-v1's path is the Phase 1 code unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from topple.challenge import Challenge
from topple.contract import TARGET_OBS_SCHEMA_ID
from topple.course import Course, course_points
from topple.envs.balance import BalanceEnv, _quat_pitch_yaw
from topple.scoring import episode_metrics, episode_score, episode_success
from topple.traj import FORMAT_V2, TRAJ_V1, Trajectory, TrajLayout
from topple.waypoint import (
    FALL_REASONS,
    WaypointEvents,
    WaypointScore,
    env_terminated,
    fall_cause,
    to_course_frame,
    waypoint_events,
    waypoint_progress,
    waypoint_score,
    waypoint_score_parts,
    waypoint_success,
)

Metrics = dict[str, float | int | bool | str]


@dataclass(frozen=True)
class Scored:
    """One episode's metrics, score (0..100) and success."""

    metrics: Metrics
    score: float
    success: bool


def make_env(challenge: Challenge, preset: str, **kwargs: Any) -> BalanceEnv:
    """The env of ``challenge``'s kind (``record``, ``render_mode`` etc. pass through)."""
    if challenge.kind == "waypoint":
        from topple.envs.waypoint import WaypointEnv

        return WaypointEnv(preset, challenge=challenge, **kwargs)
    return BalanceEnv(preset, challenge=challenge, **kwargs)


def traj_layout(challenge: Challenge) -> TrajLayout:
    """The trajectory layout a protocol of ``challenge`` reads (§8.1a), from the challenge,
    never from a file."""
    if challenge.config.obs_schema == TARGET_OBS_SCHEMA_ID:
        return TrajLayout(FORMAT_V2, len(challenge.task_spec(1.0, 1.0).obs_features))
    return TRAJ_V1


def score_trajectory(traj: Trajectory, challenge: Challenge, *, error: bool) -> Scored:
    """Score a trusted trajectory of ``challenge`` (one the harness recorded or a verified
    replay): ``error`` means the harness stopped the episode for a controller error, which
    scores 0 and never succeeds."""
    if challenge.kind == "waypoint":
        return _score_waypoint(traj, challenge, error=error)
    metrics = episode_metrics(traj, challenge.params.motor)
    cfg = challenge.config.score
    score = 0.0 if error else episode_score(metrics, cfg)
    pushes = [int(p["start_step"]) for p in traj.meta["scenario"]["pushes"]]
    fell = metrics.termination in FALL_REASONS
    cause = fall_cause(  # §8.1/§8.5: added to balance-v1, every existing key unchanged
        metrics.termination,
        metrics.steps if fell else None,
        pushes,
        float(traj.meta["control_dt"]),
    )
    return Scored(
        metrics={**metrics.as_dict(), "fall_cause": cause},
        score=min(max(score, 0.0), 100.0),
        success=not error and episode_success(metrics, cfg),
    )


@dataclass(frozen=True)
class WaypointReplay:
    """A waypoint trajectory on its course: the course points in the start frame, that frame
    (``origin``, ``heading``: the axle midpoint and yaw at reset), the axle positions per row in
    it, and the course events (§8.5)."""

    points: tuple[tuple[float, float], ...]
    origin: tuple[float, float]
    heading: float
    rows: npt.NDArray[np.float64]
    events: WaypointEvents


def waypoint_replay(traj: Trajectory, challenge: Challenge, termination: str) -> WaypointReplay:
    """The scorer's view of ``traj`` (its trusted scenario's course; ``termination`` is the
    recorded reason), shared with the course video so both see the same events."""
    cfg = challenge.config.course
    meta = traj.meta
    root = int(meta["joint_qpos"]["root"])
    course = Course.from_dict(meta["scenario"]["course"], cfg)
    points = course_points(course, cfg)
    qpos = traj.qpos
    origin = (float(qpos[0, root]), float(qpos[0, root + 1]))
    _, heading = _quat_pitch_yaw(qpos[0, root + 3 : root + 7])  # exactly the env's reset yaw
    rows = np.array(
        [to_course_frame((float(r[root]), float(r[root + 1])), origin, heading) for r in qpos],
        dtype=np.float64,
    )
    events = waypoint_events(
        rows,
        points,
        gate_radius_m=cfg.gate_radius_m,
        dock_hold_steps=cfg.dock_hold_steps,
        control_dt=float(meta["control_dt"]),
        terminated=env_terminated(termination),
    )
    return WaypointReplay(points, origin, heading, rows, events)


def _score_waypoint(traj: Trajectory, challenge: Challenge, *, error: bool) -> Scored:
    """§8.5/§8.6: the 18 waypoint metrics, from the trajectory and its trusted scenario."""
    cfg = challenge.config.course
    score_cfg: WaypointScore = challenge.config.score
    meta = traj.meta
    base = episode_metrics(traj, challenge.params.motor)
    control_dt = float(meta["control_dt"])
    episode_s = float(meta["max_steps"]) * control_dt
    scenario = meta["scenario"]
    termination = base.termination
    events = waypoint_replay(traj, challenge, termination).events
    progress = waypoint_progress(
        events,
        leg_m=cfg.leg_m,
        gate_radius_m=cfg.gate_radius_m,
        dock_hold_steps=cfg.dock_hold_steps,
        dock_progress_weights=score_cfg.dock_progress_weights,
    )
    parts = waypoint_score_parts(
        events,
        t_alive_s=base.time_upright_s,
        episode_s=episode_s,
        energy_j=base.energy_j,
        score=score_cfg,
        leg_m=cfg.leg_m,
        gate_radius_m=cfg.gate_radius_m,
        dock_hold_steps=cfg.dock_hold_steps,
        error=error,
    )
    pushes = [int(p["start_step"]) for p in scenario["pushes"]]
    fell = termination in FALL_REASONS
    metrics: Metrics = {
        "steps": base.steps,
        "termination": termination,
        "survived": base.survived,
        "time_upright_s": base.time_upright_s,
        "energy_j": base.energy_j,
        "saturation_fraction": base.saturation_fraction,
        "rms_tilt_rad": base.rms_tilt_rad,
        "waypoints": events.waypoints,
        "progress": progress,
        "docked": events.docked,
        "fell": fell,
        "t_finish_s": events.t_finish_s if events.t_finish_s is not None else episode_s,
        "max_hold_steps": events.max_hold_steps,
        "fall_cause": fall_cause(termination, base.steps if fell else None, pushes, control_dt),
        **parts,
    }
    score = 0.0 if error else waypoint_score(parts)
    success = waypoint_success(
        events, survived=base.survived, energy_j=base.energy_j, score=score_cfg, error=error
    )
    return Scored(metrics=metrics, score=min(max(score, 0.0), 100.0), success=success)
