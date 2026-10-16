"""A snapshot's ``parity_ref.npz`` is SB3's own answer (docs/design/phase-2.md §2.4, §6).

The reference actions training stored must match SB3 reloaded from the snapshot's
``model.zip`` + ``vecnormalize.pkl`` (``training=False``) within 1e-6. The reload goes
through ``learn.resume_from``, the one place allowed to load those pickles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
