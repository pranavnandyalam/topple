"""The ``waypoint-v1`` env and its kind dispatch (docs/design/phase-2.md §8.1-§8.5).

The env's course events equal the scorer's (§8.5: one shared ``WaypointTracker``), its
observation is ``balancer-target.obs.v1`` from the true pose, its records are ``topple.traj/2``
and every harness path (evaluate, verify, the CLI's loaders) uses the protocol's layout.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from topple.challenge import load_challenge
from topple.contract import TARGET_FEATURES, TARGET_OBS_SCHEMA_ID
from topple.course import course_points
from topple.evaluate import EvalProtocol, run_episode
from topple.kinds import make_env, score_trajectory, traj_layout
from topple.loader import load_submission
from topple.traj import FORMAT_V2, TRAJ_V1, load_npz
from topple.verify import protocol_traj_layout, verify_episode

REPO = Path(__file__).resolve().parents[1]
WP = load_challenge("waypoint-v1")


@pytest.fixture(scope="module")
def driver() -> object:
    return load_submission(REPO / "submissions" / "waypoint_pid_example.py", challenge=WP).factory


def test_the_bundle() -> None:
    balance = load_challenge("balance-v1")
    assert WP.kind == "waypoint"
    assert WP.config.obs_schema == TARGET_OBS_SCHEMA_ID
    assert WP.robot_xml.read_bytes() == balance.robot_xml.read_bytes()
    assert WP.scene_xml.read_bytes() == balance.scene_xml.read_bytes()
    spec = WP.task_spec(0.034, 0.14)
    assert spec.obs_features == TARGET_FEATURES
    assert spec.max_steps == 2000
    assert spec.extras["gate_radius_m"] == 0.10
    assert EvalProtocol.from_id("wp_eval_v1").seeds == tuple(range(3000, 3050))
    assert traj_layout(WP).format == FORMAT_V2
    assert traj_layout(balance) == TRAJ_V1


def test_the_first_observation_points_at_gate_0_in_the_robot_frame() -> None:
    env = make_env(WP, "clean")
    obs, _ = env.reset(seed=3000)
    assert obs.shape == (15,)
    gate = course_points(env.scenario.course, WP.config.course)[0]  # type: ignore[arg-type]
    # at reset the robot frame is the course frame: x = initial heading, origin = the axle
    assert obs[10] == pytest.approx(gate[0], abs=1e-6)
    assert obs[11] == pytest.approx(gate[1], abs=1e-6)
    assert obs[14] == 0.0  # not final


def test_env_events_equal_scorer_events_and_the_record_is_v2(
    driver: object, tmp_path: Path
) -> None:
    env = make_env(WP, "standard", record=True)
    for i, seed in enumerate(range(50)):  # spec §8.5: env events == scorer events on 50 seeds
        run = run_episode(env, driver, seed, i)  # type: ignore[arg-type]
        m = run.result.metrics
        assert m is not None
        ev = env.tracker.events(env.control_dt)  # type: ignore[attr-defined]
        t_finish = ev.t_finish_s if ev.t_finish_s is not None else WP.config.episode_seconds
        assert (ev.waypoints, ev.docked, ev.max_hold_steps, t_finish) == (
            m["waypoints"],
            m["docked"],
            m["max_hold_steps"],
            m["t_finish_s"],
        )
        parts = sum(float(m[k]) for k in ("course_pts", "speed_pts", "alive_pts", "battery_pts"))
        assert math.isclose(parts, run.result.score, abs_tol=1e-9)
    path = tmp_path / "wp.npz"
    run.trajectory.save_npz(path)
    loaded = load_npz(path, max_steps=2000, layout=traj_layout(WP))
    assert loaded.meta["obs_schema"] == TARGET_OBS_SCHEMA_ID
    with pytest.raises(ValueError, match="obs must have shape"):
        load_npz(path, max_steps=2000)  # the balance default fails closed


def test_the_reference_driver_docks_on_clean(driver: object) -> None:
    env = make_env(WP, "clean", record=True)
    run = run_episode(env, driver, 3000, 0)  # type: ignore[arg-type]
    assert run.result.metrics is not None
    assert run.result.metrics["docked"] is True
    assert run.result.success
    assert 0 < float(run.result.metrics["t_finish_s"]) < 20.0


def test_a_driver_episode_verifies_and_rescores_identically(driver: object) -> None:
    protocol = EvalProtocol.from_id("wp_eval_v1")
    env = make_env(WP, protocol.preset, record=True)
    run = run_episode(env, driver, protocol.seeds[0], 0)  # type: ignore[arg-type]
    assert protocol_traj_layout(protocol) == traj_layout(WP)
    verdict = verify_episode(run.trajectory, protocol, 0)
    assert verdict.ok, verdict.reason
    assert verdict.result.score == run.result.score
    assert verdict.result.metrics == run.result.metrics


def test_a_forged_course_fails_verification(driver: object) -> None:
    from dataclasses import replace

    protocol = EvalProtocol.from_id("wp_eval_v1")
    env = make_env(WP, protocol.preset, record=True)
    run = run_episode(env, driver, protocol.seeds[0], 0)  # type: ignore[arg-type]
    meta = dict(run.trajectory.meta)
    scenario = dict(meta["scenario"])
    course = dict(scenario["course"])
    course["h0"] = float(np.nextafter(course["h0"], 10.0))
    scenario["course"] = course
    meta["scenario"] = scenario
    forged = replace(run.trajectory, meta=meta)
    assert not verify_episode(forged, protocol, 0).ok


def test_scoring_dispatches_by_kind(driver: object) -> None:
    env = make_env(WP, "clean", record=True)
    run = run_episode(env, driver, 3001, 0)  # type: ignore[arg-type]
    scored = score_trajectory(run.trajectory, WP, error=True)
    assert scored.score == 0.0
    assert not scored.success
    assert len(scored.metrics) == 18
    from topple.waypoint import WAYPOINT_METRICS

    assert tuple(scored.metrics) == WAYPOINT_METRICS  # names and order (contract §12a)


def test_target_features_follow_the_course(driver: object) -> None:
    """Contract §6: ``final`` is 1 exactly while the dock is current; then ``gate == next``
    (also after docking); before it ``gate != next``; the observation of the step that reaches
    a gate already shows the next one (its gate equals the previous row's next, up to the robot's
    one-step motion)."""
    from topple.kinds import waypoint_replay

    env = make_env(WP, "clean", record=True)
    run = run_episode(env, driver, 3000, 0)  # type: ignore[arg-type]
    traj = run.trajectory
    replay = waypoint_replay(traj, WP, str(traj.meta["termination"]["reason"]))
    reached = replay.events.reached_step
    assert replay.events.docked
    assert all(s is not None for s in reached)
    obs = traj.obs.astype(np.float64)
    gate, nxt, final = obs[:, 10:12], obs[:, 12:14], obs[:, 14]
    dock_from = int(reached[3])  # type: ignore[arg-type]
    assert set(np.unique(final)) == {0.0, 1.0}
    assert (final[:dock_from] == 0.0).all()
    assert (final[dock_from:] == 1.0).all()
    assert np.array_equal(gate[dock_from:], nxt[dock_from:])
    assert (np.abs(gate[:dock_from] - nxt[:dock_from]).max(axis=1) > 0.1).all()
    for k in range(4):
        r = int(reached[k])  # type: ignore[arg-type]
        assert np.abs(gate[r] - nxt[r - 1]).max() < 0.02, k
        assert np.abs(gate[r] - gate[r - 1]).max() > 0.1, k
