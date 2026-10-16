"""A snapshot's ``parity_ref.npz`` is SB3's own answer (docs/design/phase-2.md §2.4, §6).

The reference actions training stored must match SB3 reloaded from the snapshot's
``model.zip`` + ``vecnormalize.pkl`` (``training=False``) within 1e-6. The reload goes
through ``learn.resume_from``, the one place allowed to load those pickles.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # the train extra; the default CI jobs collect without it
pytest.importorskip("stable_baselines3")

import topple.train.learn as learn_module
from topple.train.config import read_train_config
from topple.train.learn import resume_from, train

pytestmark = pytest.mark.train

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"


@pytest.mark.parametrize("name", ["smoke_ppo", "smoke_sac"])
def test_parity_ref_matches_sb3_reloaded(name: str, tmp_path: Path) -> None:
    loaded = read_train_config(CONFIGS / f"{name}.yaml")
    run = train(loaded.config, tmp_path / "run", config_file_sha256=loaded.file_sha256)
    for step_dir in sorted((run / "ckpt").glob("step_*")):
        step = int(step_dir.name.removeprefix("step_"))
        model, vecnorm = resume_from(run, step)
        assert vecnorm.training is False
        cfg = loaded.config
        if cfg.algo == "ppo":
            # §2.4: PPO evaluates at rollout start, after the update on the env steps it labels
            rollouts = step // (cfg.ppo.n_steps * cfg.n_envs)
            assert model._n_updates == cfg.ppo.n_epochs * rollouts
        with np.load(step_dir / "parity_ref.npz") as parity:
            obs, stored = parity["obs"], parity["action_ref"]
        z = vecnorm.normalize_obs(obs.astype(np.float64))
        actions, _ = model.predict(z, deterministic=True)
        again = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        np.testing.assert_allclose(again, stored, rtol=0, atol=1e-6)


def test_resume_refuses_a_symlinked_pickle(tmp_path: Path) -> None:
    loaded = read_train_config(CONFIGS / "smoke_ppo.yaml")
    run = train(loaded.config, tmp_path / "run", config_file_sha256=loaded.file_sha256)
    step_dir = sorted((run / "ckpt").glob("step_*"))[-1]
    target = tmp_path / "elsewhere.pkl"
    (step_dir / "vecnormalize.pkl").rename(target)
    (step_dir / "vecnormalize.pkl").symlink_to(target)
    with pytest.raises(PermissionError):
        resume_from(run, int(step_dir.name.removeprefix("step_")))


def test_resume_refuses_a_symlinked_step_directory(tmp_path: Path) -> None:
    loaded = read_train_config(CONFIGS / "smoke_ppo.yaml")
    run = train(loaded.config, tmp_path / "run", config_file_sha256=loaded.file_sha256)
    step_dir = sorted((run / "ckpt").glob("step_*"))[-1]
    moved = tmp_path / "moved"
    step_dir.rename(moved)
    step_dir.symlink_to(moved, target_is_directory=True)
    with pytest.raises(PermissionError):
        resume_from(run, int(step_dir.name.removeprefix("step_")))


# --- ownership (§2.1): pickles are loaded only from the current user's files ---------------------

STEP = 1024


def stand_in_run(root: Path) -> Path:
    """A run directory ``resume_from`` reads up to its ownership checks, without training: the
    smoke config in ``train_meta.json`` and placeholder pickles it must never reach."""
    loaded = read_train_config(CONFIGS / "smoke_ppo.yaml")
    run = root / "run"
    step_dir = run / "ckpt" / f"step_{STEP}"
    step_dir.mkdir(parents=True)
    meta = {"config": loaded.config.model_dump(mode="json")}
    (run / "train_meta.json").write_text(json.dumps(meta))
    for name in ("model.zip", "vecnormalize.pkl"):
        (step_dir / name).write_bytes(b"not a pickle: never loaded")
    return run


def another_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every file and directory now belongs to someone else, as far as ``learn`` can tell."""
    uid = os.getuid()
    monkeypatch.setattr(learn_module.os, "getuid", lambda: uid + 1)


def test_resume_refuses_files_of_another_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step_dir = stand_in_run(tmp_path) / "ckpt" / f"step_{STEP}"
    for name in ("model.zip", "vecnormalize.pkl"):
        assert learn_module._owned_file(step_dir / name) == step_dir / name
    another_user(monkeypatch)
    for name in ("model.zip", "vecnormalize.pkl"):
        with pytest.raises(
            PermissionError, match=re.escape(f"{name} is not owned by this user: refusing")
        ):
            learn_module._owned_file(step_dir / name)


def test_resume_refuses_directories_of_another_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = stand_in_run(tmp_path)
    for directory in (run / "ckpt", run / "ckpt" / f"step_{STEP}"):
        learn_module._owned_directory(directory)
    another_user(monkeypatch)
    for directory in (run / "ckpt", run / "ckpt" / f"step_{STEP}"):
        with pytest.raises(PermissionError, match=f"{directory.name} is not owned by this user"):
            learn_module._owned_directory(directory)
    # resume_from checks the parents first: a swapped one would redirect the loads below it
    with pytest.raises(PermissionError, match="ckpt is not owned by this user"):
        resume_from(run, STEP)


def test_resume_checks_the_files_once_the_directories_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directories pass (they are this user's); the pickles inside are someone else's."""
    run = stand_in_run(tmp_path)
    uid, real_lstat = os.getuid(), os.lstat

    def lstat(path: str | os.PathLike[str]) -> os.stat_result:
        st = real_lstat(path)
        if Path(path).suffix not in (".zip", ".pkl"):
            return st
        fields = list(st[:10])
        fields[4] = uid + 1  # st_uid
        return os.stat_result(fields)

    monkeypatch.setattr(learn_module.os, "lstat", lstat)
    with pytest.raises(
        PermissionError, match=re.escape("model.zip is not owned by this user: refusing")
    ):
        resume_from(run, STEP)
