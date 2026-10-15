"""Synthetic training run directories for the export and ``topple train`` tests
(docs/design/phase-2.md §2.3, §2.4, §3.3).

A run is written with plain ``np.savez`` and ``json`` from the deterministic tiny weights of
``policy_support``, laid out as the snapshot writes it: ``best.json``, ``train_meta.json`` and
``ckpt/step_<N>/{meta.json, normalize.json, actor.npz, parity_ref.npz}``. The reference actions
are ``topple.policy.build.reference_actions`` (what the graph computes, in float64), standing in
for SB3's ``predict``; the ONNX sha256 is of ``topple.export.build_onnx``'s bytes, the bytes a
selection eval scores.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from policy_support import balance_spec, det64, tiny_norm, tiny_weights
from topple.challenge import load_challenge
from topple.export import RULE_OF, build_onnx
from topple.policy.build import MlpWeights, ObsNorm, reference_actions
from topple.train.config import config_sha256, load_train_config, validate_train_config
from topple.train.selection import parity_obs

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "train"
GIT_COMMIT = "0123456789abcdef0123456789abcdef01234567"
VERSIONS = {
    "python": "3.12.3",
    "topple": "0.1.0",
    "torch": "2.14.1",
    "stable_baselines3": "2.9.0",
    "onnx": "1.23.2",
    "psutil": None,
    "Bad Key": "1.0",
    "numpy": "2.5.3\n",
}


@dataclass
class Run:
    """A written run directory and what it was written from."""

    path: Path
    step: int
    sha256: str
    weights: MlpWeights
    norm: ObsNorm
    obs: npt.NDArray[np.float32]
    action_ref: npt.NDArray[np.float64]
    sets: npt.NDArray[np.uint8]
    config: dict[str, Any]
    train_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def step_dir(self) -> Path:
        return self.path / "ckpt" / f"step_{self.step}"

    def rewrite_json(self, name: str, edit: Any) -> None:
        """Load ``name`` (relative to the run), apply ``edit`` to it in place, write it back."""
        path = self.path / name
        doc = json.loads(path.read_text())
        edit(doc)
        path.write_text(json.dumps(doc))

    def rewrite_npz(self, name: str, **arrays: Any) -> None:
        """Rewrite ``ckpt/step_<N>/<name>`` with ``arrays`` replacing members of the same name
        (``None`` drops one)."""
        path = self.step_dir / name
        with np.load(path) as old:
            members = {k: old[k] for k in old.files}
        for key, value in arrays.items():
            if value is None:
                members.pop(key, None)
            else:
                members[key] = value
        path.unlink()
        with path.open("wb") as fh:
            np.savez(fh, **members)


def make_run(
    root: Path,
    *,
    algo: str = "ppo",
    step: int = 1024,
    eligible: bool = True,
    flags: tuple[str, ...] = (),
    onpolicy_seeds: int = 200,
    head_scale: float = 3.0,
    name: str = "run",
) -> Run:
    """Write a run directory ``root/name`` with one snapshot at ``step``."""
    activation = "tanh" if algo == "ppo" else "relu"
    weights = tiny_weights(activation, head_scale=head_scale)
    norm = tiny_norm()  # epsilon 1e-8, clip 10: the shipped configs' normalize block
    cfg_doc = load_train_config(CONFIGS / f"smoke_{algo}.yaml").model_dump(mode="json")
    cfg_doc["policy"]["net_arch"]["pi"] = [8, 8]
    cfg = validate_train_config(cfg_doc)
    challenge = load_challenge("balance-v1")
    data = build_onnx(weights.layers, norm, activation=activation, challenge=challenge)  # type: ignore[arg-type]
    sha = hashlib.sha256(data).hexdigest()
    std = np.sqrt(norm.var + norm.epsilon)
    onpolicy = (norm.mean + std * det64((5 * onpolicy_seeds, 10), 21) * 1.5).astype(np.float32)
    obs, sets = parity_obs(onpolicy, norm)
    action_ref = reference_actions(
        weights, norm, RULE_OF[activation], obs, action=balance_spec().action
    )

    run_dir = root / name
    step_dir = run_dir / "ckpt" / f"step_{step}"
    step_dir.mkdir(parents=True)
    with (step_dir / "actor.npz").open("wb") as fh:
        np.savez(
            fh,
            **{
                key: arr
                for k, (w, b) in enumerate(weights.layers)
                for key, arr in ((f"W{k}", w), (f"b{k}", b))
            },
        )
    with (step_dir / "parity_ref.npz").open("wb") as fh:
        np.savez(fh, obs=obs, action_ref=action_ref, set=sets, onnx_sha256=np.str_(sha))
    (step_dir / "normalize.json").write_text(
        json.dumps(
            {
                "mean": norm.mean.tolist(),
                "var": norm.var.tolist(),
                "count": 2048.0001,
                "epsilon": norm.epsilon,
                "clip_obs": norm.clip,
            }
        )
    )
    (step_dir / "meta.json").write_text(
        json.dumps(
            {
                "step": step,
                "algo": algo,
                "action_rule": RULE_OF[activation],
                "net_arch": [8, 8],
                "activation": activation,
                "config_sha256": config_sha256(cfg),
                "onnx_sha256": sha,
                "n_onpolicy": 5 * onpolicy_seeds,
                "n_onpolicy_seeds": onpolicy_seeds,
            }
        )
    )
    row = {
        "step": step,
        "success_rate": 0.7,
        "score": 75.0,
        "median_energy_j": 5.0,
        "mean_saturation": 0.03,
        "extras": {"fell": 0.3},
        "eligible": eligible,
        "std_mean": 0.5,
        "frac_mean_outside_1": 0.01,
        "excess_penalty_mean": 0.001,
        "cpu_s": 10.0,
        "wall_s": 5.0,
        "onnx_sha256": sha,
    }
    earlier = {**row, "step": step // 2, "onnx_sha256": "0" * 64, "eligible": False}
    train_meta = {
        "config_sha256": config_sha256(cfg),
        "config_file_sha256": "1" * 64,
        "config": copy.deepcopy(cfg_doc),
        "git_commit": GIT_COMMIT,
        "git_dirty": False,
        "versions": VERSIONS,
        "flags": list(flags),
        "selection_history": [earlier, row],
        "best": {"step": step, "eligible": eligible, "onnx_sha256": sha},
        "num_timesteps": step,
    }
    (run_dir / "train_meta.json").write_text(json.dumps(train_meta))
    (run_dir / "best.json").write_text(
        json.dumps({"step": step, "eligible": eligible, "rule": "pid"})
    )
    return Run(run_dir, step, sha, weights, norm, obs, action_ref, sets, cfg_doc, train_meta)
