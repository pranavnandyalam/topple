"""Waypoint verification (docs/design/phase-2.md §8.9 item 3).

``verify_episode`` reproduces 5 reference-driver trajectories recorded on ``wp_eval_v1``'s first
5 seeds; ``verify_run`` on that 5-seed protocol fails none, and on the full protocol counts the 45
absent episodes as missing; ``topple verify`` passes each file, and (slow) a full 50-episode
``topple eval`` directory. The trusted scenario (``make_scenario`` from the protocol, course
included) equals the one each env episode records, and ``standard`` pushes start in [2.0, 18.0] s.
The forged-course and wrong-width cases live in ``test_waypoint_env.py`` and
``test_traj_v2.py``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from topple.cli import app
from topple.disturbances import make_scenario
from topple.evaluate import EvalProtocol, iter_episodes
from topple.loader import load_submission
from topple.results import ControllerInfo
from topple.traj import Trajectory
from topple.verify import verify_episode, verify_run

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "submissions" / "waypoint_pid_example.py"
FULL = EvalProtocol.from_id("wp_eval_v1")
P5 = replace(FULL, seeds=FULL.seeds[:5])
INFO = ControllerInfo(id="waypoint_pid_example", sha256="0" * 64)
runner = CliRunner()


@pytest.fixture(scope="module")
def uploads() -> dict[int, Trajectory]:
    factory = load_submission(DRIVER, challenge=P5.challenge).factory
    return {k: run.trajectory for k, run in enumerate(iter_episodes(P5, factory))}


def test_p5_is_the_first_five_wp_eval_v1_seeds() -> None:
    assert P5.seeds == (3000, 3001, 3002, 3003, 3004)


def test_each_episode_verifies_bit_exactly(uploads: dict[int, Trajectory]) -> None:
    for k, traj in uploads.items():
        verdict = verify_episode(traj, P5, k)
        assert verdict.ok, (k, verdict.reason)


def test_verify_run_fails_none_and_counts_absent_episodes_as_missing(
    uploads: dict[int, Trajectory],
) -> None:
    assert verify_run(P5, uploads, INFO, "inprocess").failed == 0
    full = verify_run(FULL, uploads, INFO, "inprocess")
    assert full.failed == len(FULL.seeds) - len(uploads)
    assert not full.results.complete


def test_the_trusted_scenario_is_the_one_the_env_records(uploads: dict[int, Trajectory]) -> None:
    challenge = P5.challenge
    preset = next(p for p in challenge.config.preset_defs if p.name == P5.preset)
    for k, traj in uploads.items():
        trusted = make_scenario(
            P5.seeds[k],
            preset,
            timestep=traj.meta["timestep"],
            control_dt=challenge.config.control_dt,
            episode_seconds=challenge.config.episode_seconds,
            course=challenge.config.course,
        )
        assert traj.meta["scenario"] == trusted.to_dict(), k
        assert "course" in traj.meta["scenario"]


def test_standard_pushes_start_inside_the_push_window(uploads: dict[int, Trajectory]) -> None:
    for traj in uploads.values():
        for push in traj.meta["scenario"]["pushes"]:  # start_step counts control steps
            assert 2.0 <= push["start_step"] * traj.meta["control_dt"] <= 18.0 + 1e-9


def test_topple_verify_passes_each_file(uploads: dict[int, Trajectory], tmp_path: Path) -> None:
    for k, traj in uploads.items():
        path = tmp_path / f"traj_{k:03d}.npz"
        traj.save_npz(path)
        result = runner.invoke(app, ["verify", str(path)])
        assert result.exit_code == 0, result.output
        assert result.output.startswith(f"OK: waypoint-v1/standard episode {k} ")


@pytest.mark.slow
def test_topple_verify_passes_a_full_wp_eval_v1_run(tmp_path: Path) -> None:
    out = tmp_path / "run"
    done = runner.invoke(app, ["eval", str(DRIVER), "-p", "wp_eval_v1", "--out", str(out),
                               "--videos", "none"])  # fmt: skip
    assert done.exit_code == 0, done.output
    verified = runner.invoke(app, ["verify", str(out)])
    assert verified.exit_code == 0, verified.output


def test_waypoint_results_record_the_waypoint_obs_schema(uploads: dict[int, Trajectory]) -> None:
    """Gate F1: results, trusted rescores and compare files name the challenge's own schemas."""
    from topple.compare import compare

    verdict = verify_run(P5, uploads, INFO, "inprocess")
    assert verdict.results.versions.obs_schema == "balancer-target.obs.v1"
    assert verdict.results.versions.action_spec == P5.challenge.config.action_spec
    cmp = compare([DRIVER], replace(P5, seeds=P5.seeds[:1]), workers=1)
    assert cmp.versions.obs_schema == "balancer-target.obs.v1"
