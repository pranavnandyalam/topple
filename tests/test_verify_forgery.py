"""Red team: every forged upload must fail verification or score 0 (docs/contract.md §16, A..M).

Each forgery starts from an honest run saved to disk and loaded back (the real upload path), then
edits it the way a cheater would. Nothing here may raise; every verdict is checked for a zero
score, and almost every one for an explicit failure.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from topple.contract import controller_obs
from topple.disturbances import Scenario
from topple.envs.balance import BalanceEnv
from topple.evaluate import EPILOGUE_S, ControllerFactory, wants_epilogue
from topple.traj import Trajectory, load_npz
from topple.verify import (
    EpisodeVerdict,
    protocol_max_steps,
    rescore,
    verify_episode,
)
from verify_support import (
    INFO,
    Honest,
    explodes,
    lqr,
    pid,
    public,
    run_honest,
    truncated,
    with_meta,
    zero,
)

INTEGRITY = (
    "qpos",
    "qvel",
    "obs",
    "ctrl",
    "enc_counts",
    "gyro",
    "accel",
    "push_force",
    "epilogue_qpos",
)


@pytest.fixture(scope="module")
def standard(tmp_path_factory: pytest.TempPathFactory) -> Honest:
    """PID on 3 public standard seeds (seed 1002 falls: a fall with pushes and an epilogue)."""
    return run_honest(public("standard", 3), pid, tmp_path_factory.mktemp("standard"))


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> Honest:
    return run_honest(public("clean", 2), lqr, tmp_path_factory.mktemp("clean"))


@pytest.fixture(scope="module")
def errored(tmp_path_factory: pytest.TempPathFactory) -> Honest:
    return run_honest(
        public("standard", 1), explodes(400, RuntimeError("bug")), tmp_path_factory.mktemp("err")
    )


@pytest.fixture(scope="module")
def falls(tmp_path_factory: pytest.TempPathFactory) -> Honest:
    return run_honest(public("standard", 1), zero, tmp_path_factory.mktemp("falls"))


def rejected(
    verdict: EpisodeVerdict, code: str = "not_reproducible", *, mentions: str = ""
) -> None:
    assert not verdict.ok
    assert verdict.code == code
    assert mentions in verdict.reason
    result = verdict.result
    assert result.score == 0.0
    assert not result.success
    assert result.error is not None
    assert result.error.code == code
    assert result.error.detail == verdict.reason[: len(result.error.detail)]


def survived(run: Honest) -> int:
    """Index of the first surviving episode of an honest run."""
    return next(i for i, t in run.uploads.items() if t.meta["termination"]["reason"] == "timeout")


def fell(run: Honest) -> int:
    return next(i for i, t in run.uploads.items() if t.meta["termination"]["reason"] == "fell")


# ------------------------------------------------------------------ A: relabelled preset


def test_a_clean_run_relabelled_standard(clean: Honest) -> None:
    standard_protocol = public("standard", 2)
    upload = with_meta(clean.uploads[0], preset="standard")
    rejected(verify_episode(upload, standard_protocol, 0), mentions="scenario")


def test_a_clean_run_relabelled_with_the_standard_scenario(clean: Honest) -> None:
    """Even with the right standard scenario in meta, clean physics cannot replay as standard."""
    standard_protocol = public("standard", 2)
    env = BalanceEnv("standard")
    env.reset(seed=standard_protocol.seeds[0])
    upload = with_meta(clean.uploads[0], preset="standard", scenario=env.scenario.to_dict())
    rejected(verify_episode(upload, standard_protocol, 0))


def test_a_wrong_challenge_is_rejected(clean: Honest) -> None:
    upload = with_meta(clean.uploads[0], challenge="balance-v2")
    rejected(verify_episode(upload, clean.protocol, 0), mentions="challenge")


# ------------------------------------------------------------------ B: shortened episode length


def test_b_max_steps_shrunk_to_the_fall_with_a_timeout(standard: Honest) -> None:
    i = fell(standard)
    upload = standard.uploads[i]
    steps = upload.steps
    forged = with_meta(upload, max_steps=steps, termination={"step": steps, "reason": "timeout"})
    rejected(
        verify_episode(forged, standard.protocol, i),
        mentions="meta.termination differs from the replay",
    )
    alone = with_meta(upload, max_steps=steps)
    rejected(
        verify_episode(alone, standard.protocol, i),
        mentions="meta.max_steps differs from the replay",
    )


def test_b_score_comes_from_the_replay_not_the_meta(standard: Honest) -> None:
    """Whatever meta claims, the verdict's metrics are the replay's: max_steps is the protocol's."""
    i = fell(standard)
    upload = standard.uploads[i]
    forged = with_meta(upload, max_steps=upload.steps)
    verdict = verify_episode(forged, standard.protocol, i)
    assert verdict.replay is not None
    assert verdict.replay.meta["max_steps"] == 1000
    assert verdict.result.metrics is not None
    assert verdict.result.metrics["upright_frac"] == upload.steps / 1000
    assert verdict.result.score == 0.0


# ------------------------------------------------------------------ C: stripped error


def test_c_error_deleted_from_an_errored_run(errored: Honest) -> None:
    upload = errored.uploads[0]
    assert upload.meta["error"]["code"] == "exception"
    for forged in (
        with_meta(upload, error=None),
        replace(upload, meta={k: v for k, v in upload.meta.items() if k != "error"}),
    ):
        rejected(verify_episode(forged, errored.protocol, 0), "incomplete")


def test_c_error_replaced_by_a_survival_claim(errored: Honest) -> None:
    upload = errored.uploads[0]
    forged = with_meta(
        upload,
        error=None,
        max_steps=upload.steps,
        termination={"step": upload.steps, "reason": "timeout"},
    )
    rejected(verify_episode(forged, errored.protocol, 0), "incomplete")


@pytest.mark.parametrize(
    "error",
    [
        {"code": "teleported", "step": 400, "detail": ""},
        {"code": "missing", "step": 400, "detail": ""},
        {"code": "exception", "step": 399, "detail": ""},
        {"code": "exception", "step": True, "detail": ""},
        {"code": "exception", "step": 400},
        {"code": "exception", "step": 400, "detail": 5},
        {"code": "exception", "step": 400, "detail": "", "extra": 1},
        "exception",
        [],
    ],
    ids=repr,
)
def test_c_malformed_error_claims_are_rejected(errored: Honest, error: Any) -> None:
    rejected(verify_episode(with_meta(errored.uploads[0], error=error), errored.protocol, 0))


# ------------------------------------------------------------------ D: seed swap


def test_d_index_holds_another_index_trajectory(standard: Honest) -> None:
    swapped = {0: standard.uploads[1], 1: standard.uploads[0], 2: standard.uploads[2]}
    for i in (0, 1):
        rejected(verify_episode(swapped[i], standard.protocol, i), mentions="episode_index")
    again = rescore(standard.protocol, swapped, INFO, "inprocess")
    assert again.episodes is not None
    assert standard.results.episodes is not None
    assert [e.error.code if e.error else None for e in again.episodes][:2] == [
        "not_reproducible",
        "not_reproducible",
    ]
    assert again.episodes[2] == standard.results.episodes[2]


def test_d_seed_swap_with_a_relabelled_index(standard: Honest) -> None:
    forged = with_meta(standard.uploads[1], episode_index=0)
    rejected(verify_episode(forged, standard.protocol, 0), mentions="scenario")


# ------------------------------------------------------------------ E: zeroed records


def test_e_push_force_zeroed(standard: Honest) -> None:
    upload = standard.uploads[0]
    assert np.abs(upload.push_force).sum() > 0
    forged = replace(upload, push_force=np.zeros_like(upload.push_force))
    rejected(verify_episode(forged, standard.protocol, 0), mentions="push_force")


def test_e_epilogue_zeroed_or_dropped(falls: Honest) -> None:
    upload = falls.uploads[0]
    assert upload.epilogue_qpos.shape[0] == round(EPILOGUE_S / 0.01)
    zeroed = replace(upload, epilogue_qpos=np.zeros_like(upload.epilogue_qpos))
    rejected(verify_episode(zeroed, falls.protocol, 0), mentions="epilogue_qpos")
    dropped = replace(upload, epilogue_qpos=upload.epilogue_qpos[:0])
    rejected(verify_episode(dropped, falls.protocol, 0), mentions="epilogue_qpos")


def test_e_epilogue_added_to_a_survivor(standard: Honest) -> None:
    """A survivor's replay ends upright: an epilogue showing anything else is a forgery."""
    i = survived(standard)
    upload = standard.uploads[i]
    forged = replace(upload, epilogue_qpos=np.repeat(upload.qpos[-1:], 100, axis=0))
    rejected(verify_episode(forged, standard.protocol, i), mentions="epilogue_qpos")


# ------------------------------------------------------------------ F: scenario edited


def test_f_pushes_removed_from_the_scenario(standard: Honest) -> None:
    upload = standard.uploads[0]
    scenario = {**upload.meta["scenario"], "pushes": []}
    rejected(
        verify_episode(with_meta(upload, scenario=scenario), standard.protocol, 0),
        mentions="scenario",
    )


def run_on_scenario(scenario: Scenario, factory: ControllerFactory) -> Trajectory:
    """An honest closed-loop episode on an explicit scenario (what a forger can simulate)."""
    env = BalanceEnv(scenario.preset, record=True)
    obs, info = env.reset(options={"scenario": scenario})
    controller = factory(env.task_spec)
    controller.reset(0)
    while True:
        action = controller.act(controller_obs(obs, float(info["t"])))
        obs, _, terminated, truncated_, info = env.step(action)
        if terminated or truncated_:
            break
    if wants_epilogue(env.termination_reason or "timeout", has_error=False):
        env.simulate_epilogue(EPILOGUE_S)
    return env.trajectory({"episode_index": 0, "error": None})


def test_f_a_genuine_push_free_run_with_the_seed_kept(tmp_path: Path, standard: Honest) -> None:
    """A self-consistent episode simulated without the pushes still fails: wrong scenario."""
    env = BalanceEnv("standard")
    env.reset(seed=standard.protocol.seeds[0])
    forged = run_on_scenario(replace(env.scenario, pushes=()), pid)
    path = tmp_path / "forged.npz"
    forged.save_npz(path)
    loaded = load_npz(path, max_steps=protocol_max_steps(standard.protocol))
    assert loaded.meta["scenario"]["seed"] == standard.uploads[0].meta["scenario"]["seed"]
    rejected(verify_episode(loaded, standard.protocol, 0), mentions="scenario")


# ------------------------------------------------------------------ G: dropped episodes


def test_g_rescore_with_dropped_episodes(standard: Honest) -> None:
    assert standard.results.episodes is not None
    kept = {0: standard.uploads[0], 2: standard.uploads[2]}
    again = rescore(standard.protocol, kept, INFO, "inprocess")
    assert not again.complete
    expected = (standard.results.episodes[0].score + standard.results.episodes[2].score) / 3
    assert again.aggregate.score == pytest.approx(expected, abs=1e-12)
    assert again.episodes is not None
    assert again.episodes[1].error is not None
    assert again.episodes[1].error.code == "missing"
    assert again.aggregate.episodes == 3


def zeroed_pushes(traj: Trajectory) -> Trajectory:
    return replace(traj, push_force=np.zeros_like(traj.push_force))


def cut_short(traj: Trajectory) -> Trajectory:
    return truncated(traj, 500)


@pytest.mark.parametrize(
    ("forge", "code"),
    [(zeroed_pushes, "not_reproducible"), (cut_short, "incomplete")],
    ids=["integrity", "incomplete"],
)
def test_honest_episodes_after_a_rejected_replay_still_reproduce(
    standard: Honest, forge: Any, code: str
) -> None:
    """``rescore`` reuses one env: a replay that ends in rejection must leave nothing behind."""
    assert standard.results.episodes is not None
    uploads = {**standard.uploads, 0: forge(standard.uploads[0])}
    again = rescore(standard.protocol, uploads, INFO, "inprocess")
    assert again.episodes is not None
    assert again.episodes[0].error is not None
    assert again.episodes[0].error.code == code
    assert again.episodes[0].metrics is not None  # rejected after the replay ran
    assert again.episodes[1:] == standard.results.episodes[1:]


# ------------------------------------------------------------------ H: one episode everywhere


def test_h_one_good_episode_copied_to_every_index(standard: Honest) -> None:
    best = survived(standard)
    copies = {i: with_meta(standard.uploads[best], episode_index=i) for i in range(3)}
    again = rescore(standard.protocol, copies, INFO, "inprocess")
    assert again.episodes is not None
    assert standard.results.episodes is not None
    for i, e in enumerate(again.episodes):
        if i == best:
            assert e == standard.results.episodes[i]
        else:
            assert e.error is not None
            assert e.error.code == "not_reproducible"
            assert e.score == 0.0
    assert not any(e.error is None for i, e in enumerate(again.episodes) if i != best)


# ------------------------------------------------------------------ I: truncated without an error


def test_i_truncated_without_error_is_incomplete(clean: Honest) -> None:
    upload = clean.uploads[0]
    short = truncated(upload, 500)
    forged = with_meta(short, termination={"step": 500, "reason": "timeout"})
    rejected(verify_episode(forged, clean.protocol, 0), "incomplete")
    rejected(verify_episode(short, clean.protocol, 0), "incomplete")


def test_i_zero_rows_without_error_is_incomplete(clean: Honest) -> None:
    rejected(verify_episode(truncated(clean.uploads[0], 0), clean.protocol, 0), "incomplete")


def test_i_rows_beyond_the_end_of_the_replay(falls: Honest) -> None:
    """More commands than the episode accepted: the replay ends first."""
    upload = falls.uploads[0]
    longer = replace(
        upload,
        qpos=np.vstack([upload.qpos, upload.qpos[-1:]]),
        qvel=np.vstack([upload.qvel, upload.qvel[-1:]]),
        obs=np.vstack([upload.obs, upload.obs[-1:]]),
        action=np.vstack([upload.action, upload.action[-1:]]),
        ctrl=np.vstack([upload.ctrl, upload.ctrl[-1:]]),
        enc_counts=np.vstack([upload.enc_counts, upload.enc_counts[-1:]]),
        gyro=np.vstack([upload.gyro, upload.gyro[-1:]]),
        accel=np.vstack([upload.accel, upload.accel[-1:]]),
        push_force=np.vstack([upload.push_force, upload.push_force[-1:]]),
    )
    rejected(verify_episode(longer, falls.protocol, 0), mentions="replay ended")


# ------------------------------------------------------------------ J: robot hash


def test_j_robot_hash_mismatch(clean: Honest) -> None:
    forged = with_meta(clean.uploads[0], robot_hash="0" * 64)
    rejected(verify_episode(forged, clean.protocol, 0), mentions="robot_hash")


# ------------------------------------------------------------------ K: commands


def test_k_ctrl_outside_the_unit_range(clean: Honest) -> None:
    upload = clean.uploads[0]
    ctrl, action = upload.ctrl.copy(), upload.action.copy()
    ctrl[10] = action[10] = (1.5, 0.0)
    forged = replace(upload, ctrl=ctrl, action=action)
    rejected(
        verify_episode(forged, clean.protocol, 0),
        mentions="ctrl row 10 is not a finite command in [-1, 1]",
    )


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_k_non_finite_commands(clean: Honest, value: float) -> None:
    upload = clean.uploads[0]
    ctrl = upload.ctrl.copy()
    ctrl[3, 1] = value
    rejected(verify_episode(replace(upload, ctrl=ctrl), clean.protocol, 0), mentions="ctrl row 3")
    action = upload.action.copy()
    action[4, 0] = value
    rejected(
        verify_episode(replace(upload, action=action), clean.protocol, 0), mentions="action row 4"
    )


def test_k_ctrl_is_not_the_clipped_action(clean: Honest) -> None:
    upload = clean.uploads[0]
    action = upload.action.copy()
    action[20, 0] = -0.5 if upload.ctrl[20, 0] > 0 else 0.5
    rejected(verify_episode(replace(upload, action=action), clean.protocol, 0), mentions="row 20")


# ------------------------------------------------------------------ L: too many steps


def test_l_more_than_max_steps_is_rejected_at_load(tmp_path: Path, clean: Honest) -> None:
    upload = clean.uploads[0]
    longer = replace(
        upload,
        qpos=np.vstack([upload.qpos, upload.qpos[-1:]]),
        qvel=np.vstack([upload.qvel, upload.qvel[-1:]]),
        obs=np.vstack([upload.obs, upload.obs[-1:]]),
        action=np.vstack([upload.action, upload.action[-1:]]),
        ctrl=np.vstack([upload.ctrl, upload.ctrl[-1:]]),
        enc_counts=np.vstack([upload.enc_counts, upload.enc_counts[-1:]]),
        gyro=np.vstack([upload.gyro, upload.gyro[-1:]]),
        accel=np.vstack([upload.accel, upload.accel[-1:]]),
        push_force=np.vstack([upload.push_force, upload.push_force[-1:]]),
    )
    assert longer.steps == 1001
    path = tmp_path / "long.npz"
    longer.save_npz(path)
    with pytest.raises(ValueError, match="more than 1000"):
        load_npz(path, max_steps=protocol_max_steps(clean.protocol))
    rejected(
        verify_episode(longer, clean.protocol, 0),
        mentions="the trajectory has 1001 steps, more than the protocol's 1000",
    )


# ------------------------------------------------------------------ M: one ulp


@pytest.mark.parametrize("name", INTEGRITY)
def test_m_one_ulp_anywhere_is_caught(falls: Honest, name: str) -> None:
    upload = falls.uploads[0]
    arr = getattr(upload, name).copy()
    flat = arr.reshape(-1)
    k = flat.size // 2
    if arr.dtype.kind == "f":
        flat[k] = np.nextafter(flat[k], np.inf if flat[k] <= 0 else -np.inf)
    else:
        flat[k] += 1
    forged = replace(upload, **{name: arr})
    if name == "ctrl":  # the clipped-action check sees a ctrl edit first
        rejected(verify_episode(forged, falls.protocol, 0), mentions="ctrl row")
    else:
        rejected(verify_episode(forged, falls.protocol, 0), mentions=name)


def test_m_negative_zero_is_a_different_bit_pattern(falls: Honest) -> None:
    upload = falls.uploads[0]
    push = upload.push_force.copy()
    assert push[0, 0] == 0.0
    push[0, 0] = -0.0
    rejected(
        verify_episode(replace(upload, push_force=push), falls.protocol, 0), mentions="push_force"
    )


@settings(
    max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    name=st.sampled_from((*INTEGRITY, "action")),
    position=st.floats(0.0, 1.0, exclude_max=True),
    bit=st.integers(0, 7),
)
def test_any_single_bit_flip_fails_or_changes_nothing(
    standard: Honest, name: str, position: float, bit: int
) -> None:
    """Flip one bit anywhere: verification fails, except in ``action`` where a flip that keeps
    ``clip(action) == ctrl`` (a saturated raw value) is physically meaningless and allowed."""
    i = fell(standard)
    upload = standard.uploads[i]
    arr = getattr(upload, name).copy()
    raw = arr.reshape(-1).view(np.uint8)
    if raw.size == 0:
        return
    byte = int(position * raw.size)
    raw[byte] ^= np.uint8(1 << bit)
    forged = replace(upload, **{name: arr})
    verdict = verify_episode(forged, standard.protocol, i)
    if (
        name == "action"
        and np.isfinite(arr).all()
        and np.array_equal(np.clip(arr, -1, 1), upload.ctrl)
    ):
        assert verdict.ok, verdict.reason
        assert verdict.result == standard.results.episodes[i]  # type: ignore[index]
    else:
        assert not verdict.ok
        assert verdict.result.score == 0.0
