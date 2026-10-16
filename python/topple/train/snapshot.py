"""Checkpoint snapshots from a live SB3 model (docs/design/phase-2.md §2.4; train extra).

A snapshot is the immutable ``ckpt/step_<N>/`` directory. Its torch-free files come from
``topple.export.snapshot_files`` (the layout export reads), written by
``topple.export.write_snapshot``. This module only pulls the numbers out of the model:
- the actor's float32 layers;
- ``VecNormalize``'s ``obs_rms``;
- SB3's own reference actions on the parity rows;
- PPO's critic and ``log_std``;
- the SB3 pickles a resume needs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from topple.export import _npz_bytes, snapshot_files, write_snapshot
from topple.policy.build import ObsNorm
from topple.train.config import PpoTrainConfig, SacTrainConfig
from topple.train.selection import parity_obs

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
Layers = tuple[tuple[F32, F32], ...]

CRITIC_KEYS = (
    "mlp_extractor.value_net.0.weight",
    "mlp_extractor.value_net.0.bias",
    "mlp_extractor.value_net.2.weight",
    "mlp_extractor.value_net.2.bias",
    "value_net.weight",
    "value_net.bias",
)


def _f32(t: torch.Tensor) -> F32:
    return np.array(t.detach().cpu().numpy(), dtype=np.float32, copy=True)


def actor_layers(model: Any) -> Layers:
    """The actor's ``(W [out, in], b)`` float32 pairs: the two hidden layers, then the head
    giving the action mean (PPO ``action_net``; SAC ``mu``, before the tanh)."""
    if hasattr(model.policy, "action_net"):  # PPO
        hidden = model.policy.mlp_extractor.policy_net
        head = model.policy.action_net
    else:  # SAC
        hidden = model.actor.latent_pi
        head = model.actor.mu
    linears = [m for m in hidden if isinstance(m, torch.nn.Linear)] + [head]
    if len(linears) != 3:
        raise ValueError(f"the actor must have 2 hidden layers and a head, not {len(linears)}")
    return tuple((_f32(m.weight), _f32(m.bias)) for m in linears)


def obs_norm(vecnorm: Any) -> ObsNorm:
    """``VecNormalize``'s observation statistics as the graph and export use them."""
    rms = vecnorm.obs_rms
    return ObsNorm(
        mean=np.array(rms.mean, dtype=np.float64, copy=True),
        var=np.array(rms.var, dtype=np.float64, copy=True),
        epsilon=float(vecnorm.epsilon),
        clip=float(vecnorm.clip_obs),
    )


def reference_actions(model: Any, vecnorm: Any, obs: F32) -> F64:
    """§2.4: ``clip(model.predict(vecnorm.normalize_obs(obs.astype(f64)), deterministic=True),
    -1, 1)`` in float64, SB3's own pipeline (``obs_rms`` is read, never updated)."""
    if obs.shape[0] == 0:
        return np.empty((0, int(model.action_space.shape[0])), dtype=np.float64)
    z = vecnorm.normalize_obs(obs.astype(np.float64))
    actions, _ = model.predict(z, deterministic=True)
    return np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)


def ppo_extra_files(model: Any) -> dict[str, bytes]:
    """PPO's ``critic.npz`` (value network) and ``dist.npz`` (``log_std``), for 2b's surgery."""
    state = model.policy.state_dict()
    critic = {key: _f32(state[key]) for key in CRITIC_KEYS}
    return {
        "critic.npz": _npz_bytes(critic),
        "dist.npz": _npz_bytes({"log_std": _f32(state["log_std"])}),
    }


def snapshot(
    cfg: PpoTrainConfig | SacTrainConfig,
    model: Any,
    vecnorm: Any,
    step: int,
    ckpt_root: Path,
    *,
    layers: Layers,
    norm: ObsNorm,
    onnx_sha256: str,
    onpolicy: F32,
) -> Path:
    """Write ``ckpt_root/step_<step>/`` from the live model, in the same callback call as the
    selection eval that scored ``onnx_sha256``. Those bytes were built from ``layers`` and
    ``norm``, which the caller passes so the snapshot holds exactly them. ``FileExistsError``
    if the step exists (snapshots are immutable)."""
    obs, _ = parity_obs(onpolicy, norm)
    files: dict[str, bytes] = snapshot_files(
        cfg,
        step,
        layers,
        norm,
        count=float(vecnorm.obs_rms.count),
        onpolicy=onpolicy,
        action_ref=reference_actions(model, vecnorm, obs),
        onnx_sha256=onnx_sha256,
    )
    if cfg.algo == "ppo":
        files.update(ppo_extra_files(model))

    def more(tmp: Path) -> None:
        model.save(tmp / "model.zip")
        vecnorm.save(str(tmp / "vecnormalize.pkl"))

    return write_snapshot(ckpt_root, step, files, more=more)


def snapshot_names(cfg: PpoTrainConfig | SacTrainConfig) -> Mapping[str, None]:
    """The file names a snapshot of ``cfg``'s algorithm holds (for tests)."""
    names = ["actor.npz", "normalize.json", "parity_ref.npz", "meta.json", "model.zip"]
    names.append("vecnormalize.pkl")
    if cfg.algo == "ppo":
        names += ["critic.npz", "dist.npz"]
    return dict.fromkeys(sorted(names))
