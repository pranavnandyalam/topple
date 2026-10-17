"""Fine-tune init: a balance policy extended to the waypoint observation (docs/design/phase-2.md
§8.8 arm b). Pickle-free and torch-free: it reads only a run's ``best.json``, ``train_meta.json``
and the step's ``actor.npz``, ``critic.npz``, ``dist.npz``, ``normalize.json``, ``meta.json`` and
``parity_ref.npz`` through ``topple.export``'s readers (the snapshot format's one reader).

``extend_obs`` builds the state of a 15-input SB3 ``ActorCriticPolicy`` from a 10-input one:
1. the first layers of the actor and the critic get zero columns for the 5 new inputs;
2. the columns of ``init.zero_inputs`` (yaw, wheel_pos_l, wheel_pos_r) are zeroed too, which with
   frozen balance statistics is the policy with those features held at their training mean;
3. the critic's head is scaled by ``balance_scale``, so V matches the balance-alive part of the
   waypoint return;
4. ``log_std = init.log_std``.
The observation statistics are the balance run's for dims 0..9 and analytic for 10..14: the
gate and next-waypoint x, y mean 0 and variance ``leg_m**2``; ``final`` mean 0.5, variance 0.25.
Training keeps them frozen (``VecNormalize.training = False``).

``sanity_check`` runs an extended policy on balance-v1 ``clean`` with 5 zero inputs appended and
+100 rad on the wheel positions: the share of episodes that survive the full episode (§8.9
item 7).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from topple.contract import FEATURES, TARGET_FEATURES
from topple.train.config import InitConfig, PpoTrainConfig
from topple.train.selection import NOT_SHIPPABLE

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

SOURCE_DIM: Final = len(FEATURES)
TARGET_DIM: Final = len(TARGET_FEATURES)
FINAL_MEAN: Final = 0.5
FINAL_VAR: Final = 0.25
WHEEL_POS: Final = (4, 5)
SANITY_WHEEL_OFFSET_RAD: Final = 100.0
FIRST_LAYERS: Final = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")
ACTOR_KEYS: Final = {
    "W0": "mlp_extractor.policy_net.0.weight",
    "b0": "mlp_extractor.policy_net.0.bias",
    "W1": "mlp_extractor.policy_net.2.weight",
    "b1": "mlp_extractor.policy_net.2.bias",
    "W2": "action_net.weight",
    "b2": "action_net.bias",
}
"""``actor.npz`` keys -> SB3 PPO state-dict names (§8.8's table)."""


def critic_schema(vf: tuple[int, int], d: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    """``critic.npz`` (``topple.train.snapshot.CRITIC_KEYS``): float32 ``[out, in]``."""
    h1, h2 = vf
    return {
        "mlp_extractor.value_net.0.weight": ("<f4", (h1, d)),
        "mlp_extractor.value_net.0.bias": ("<f4", (h1,)),
        "mlp_extractor.value_net.2.weight": ("<f4", (h2, h1)),
        "mlp_extractor.value_net.2.bias": ("<f4", (h2,)),
        "value_net.weight": ("<f4", (1, h2)),
        "value_net.bias": ("<f4", (1,)),
    }


class InitError(ValueError):
    """No usable source run, or a source that is not a 10-input PPO balance snapshot."""


@dataclass(frozen=True)
class InitSource:
    """Where a fine-tune started (``train_meta.init_source``)."""

    run: str
    step: int
    onnx_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {"run": self.run, "step": self.step, "onnx_sha256": self.onnx_sha256}


@dataclass(frozen=True)
class ExtendedInit:
    """A 15-input policy's float32 state (SB3 ``ActorCriticPolicy`` names), its float64
    observation statistics and the source it was built from."""

    state: MappingProxyType[str, F32]
    obs_mean: F64
    obs_var: F64
    obs_count: float
    epsilon: float
    clip_obs: float
    source: InitSource

    def actions(self, obs: npt.ArrayLike) -> F64:
        """The deterministic action for each row of ``obs`` (``[n, 15]``): normalized and
        clipped as ``VecNormalize`` does, the tanh actor, the mean clipped to ±1."""
        x = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        z = np.clip(
            (x - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )
        s = {k: np.asarray(v, dtype=np.float64) for k, v in self.state.items()}
        h = np.tanh(z @ s[ACTOR_KEYS["W0"]].T + s[ACTOR_KEYS["b0"]])
        h = np.tanh(h @ s[ACTOR_KEYS["W1"]].T + s[ACTOR_KEYS["b1"]])
        return np.clip(h @ s[ACTOR_KEYS["W2"]].T + s[ACTOR_KEYS["b2"]], -1.0, 1.0)


def choose_source(init: InitConfig, seed: int, *, cwd: Path | None = None) -> InitSource:
    """§8.8: seed k starts from ``from_runs[k mod n]`` at its ``best.json`` step if that step is
    eligible and the run has no abort flag, else from the next run (k+1, k+2, ... mod n)."""
    from topple.export import ExportError

    root = Path.cwd() if cwd is None else cwd
    n = len(init.from_runs)
    reasons = []
    for i in range(n):
        name = init.from_runs[(seed + i) % n]
        try:
            source = _usable(root / name, name)
        except ExportError as exc:
            reasons.append(f"{name}: {exc}")
            continue
        if isinstance(source, str):
            reasons.append(f"{name}: {source}")
            continue
        return source
    raise InitError("no init.from_runs run is usable: " + "; ".join(reasons))


def extended_init(cfg: PpoTrainConfig, *, cwd: Path | None = None) -> ExtendedInit:
    """``extend_obs`` from ``choose_source`` for ``cfg``'s seed; every failure (a reader's
    ``ExportError`` included) is an ``InitError``."""
    from topple.export import ExportError

    if cfg.init is None:
        raise InitError("the config has no init")
    try:
        return extend_obs(cfg, choose_source(cfg.init, cfg.seed, cwd=cwd), cwd=cwd)
    except ExportError as exc:
        raise InitError(str(exc)) from None


def _usable(run_dir: Path, name: str) -> InitSource | str:
    """The run's best step as a source, or why it cannot be one."""
    from topple.export import _best_step, _flags, _history_row, _object, read_run_json

    if not run_dir.is_dir():
        return "not a run directory"
    step = _best_step(run_dir)
    meta = _object(read_run_json(run_dir / "train_meta.json"), "train_meta.json")
    blocked = sorted(set(_flags(meta)) & NOT_SHIPPABLE)
    if blocked:
        return f"flags {blocked}"
    row = _history_row(meta, step)
    if not row["eligible"]:
        return f"its best step {step} is not eligible"
    return InitSource(run=name, step=step, onnx_sha256=str(row["onnx_sha256"]))


def extend_obs(cfg: PpoTrainConfig, source: InitSource, *, cwd: Path | None = None) -> ExtendedInit:
    """The 15-input init of ``cfg`` (which has ``init``) from ``source`` (module docstring)."""
    from topple.challenge import load_challenge

    init = cfg.init
    if init is None:
        raise InitError("the config has no init")
    root = Path.cwd() if cwd is None else cwd
    snap, critic, log_std = _read_source(root / source.run, source)
    _check_arch(cfg, snap)
    if any(i >= SOURCE_DIM for i in init.zero_inputs):
        raise InitError(f"init.zero_inputs must be source features (< {SOURCE_DIM})")
    state: dict[str, F32] = {}
    for k, (w, b) in enumerate(snap.layers):
        state[ACTOR_KEYS[f"W{k}"]] = w
        state[ACTOR_KEYS[f"b{k}"]] = b
    state.update(critic)
    for key in FIRST_LAYERS:
        state[key] = _widen(state[key], init.zero_inputs)
    state["value_net.weight"] = _scaled(state["value_net.weight"], cfg)
    state["value_net.bias"] = _scaled(state["value_net.bias"], cfg)
    state["log_std"] = np.full_like(log_std, init.log_std, dtype=np.float32)
    leg = float(load_challenge(cfg.env.challenge).config.course.leg_m)
    mean = np.concatenate([snap.norm.mean, np.zeros(4), [FINAL_MEAN]])
    var = np.concatenate([snap.norm.var, np.full(4, leg * leg), [FINAL_VAR]])
    count = _count(root / source.run, source.step)
    return ExtendedInit(
        state=MappingProxyType({k: np.array(v, dtype=np.float32) for k, v in state.items()}),
        obs_mean=mean.astype(np.float64),
        obs_var=var.astype(np.float64),
        obs_count=count,
        epsilon=snap.norm.epsilon,
        clip_obs=snap.norm.clip,
        source=source,
    )


def _read_source(run_dir: Path, source: InitSource) -> tuple[Any, dict[str, F32], F32]:
    from topple.export import (
        _npz,
        _object,
        _read_snapshot,
        _Run,
        _step_dir,
        build_onnx,
        read_run_json,
    )

    meta = _object(read_run_json(run_dir / "train_meta.json"), "train_meta.json")
    step_dir = _step_dir(run_dir, source.step)
    run = _Run(source.step, step_dir, True, source.onnx_sha256, meta, False)
    snap = _read_snapshot(run)
    built = hashlib.sha256(
        build_onnx(
            snap.layers, snap.norm, activation=snap.meta.activation, challenge=snap.challenge
        )
    ).hexdigest()
    if not built == snap.meta.onnx_sha256 == snap.parity_sha256 == source.onnx_sha256:
        raise InitError(
            f"{source.run}: step {source.step}'s weights do not rebuild the ONNX its selection "
            "eval scored"
        )
    if snap.cfg.algo != "ppo" or snap.cfg.env.challenge != "balance-v1":
        raise InitError(f"{source.run}: the source must be a PPO balance-v1 run")
    if snap.task.obs_dim != SOURCE_DIM:
        raise InitError(f"{source.run}: the source policy has {snap.task.obs_dim} inputs")
    vf1, vf2 = snap.cfg.policy.net_arch.vf
    critic = _npz(step_dir / "critic.npz", critic_schema((vf1, vf2), SOURCE_DIM))[0]
    dist = _npz(step_dir / "dist.npz", {"log_std": ("<f4", (snap.task.action_dim,))})[0]
    return snap, {k: np.asarray(v, dtype=np.float32) for k, v in critic.items()}, dist["log_std"]


def _check_arch(cfg: PpoTrainConfig, snap: Any) -> None:
    src = snap.cfg.policy
    if (tuple(src.net_arch.pi), tuple(src.net_arch.vf), src.activation) != (
        tuple(cfg.policy.net_arch.pi),
        tuple(cfg.policy.net_arch.vf),
        cfg.policy.activation,
    ):
        raise InitError("the source run's net_arch or activation differ from the config's")
    if (snap.norm.epsilon, snap.norm.clip) != (cfg.normalize.epsilon, cfg.normalize.clip_obs):
        raise InitError("the source run's normalization epsilon or clip_obs differ")


def _widen(w: F32, zero_inputs: tuple[int, ...]) -> F32:
    """``[h, 10]`` -> ``[h, 15]``: new columns zero, ``zero_inputs`` columns zeroed."""
    out = np.zeros((w.shape[0], TARGET_DIM), dtype=np.float32)
    out[:, :SOURCE_DIM] = w
    out[:, list(zero_inputs)] = 0.0
    return out


def _scaled(a: F32, cfg: PpoTrainConfig) -> F32:
    assert cfg.env.waypoint_reward is not None  # init requires a waypoint challenge
    return (a * np.float32(cfg.env.waypoint_reward.balance_scale)).astype(np.float32)


def _count(run_dir: Path, step: int) -> float:
    from topple.export import NormalizeFile, _model, _step_dir, read_run_json

    raw = read_run_json(_step_dir(run_dir, step) / "normalize.json")
    return float(_model(NormalizeFile, raw, "normalize.json").count)


def sanity_check(ext: ExtendedInit, seeds: range) -> float:
    """§8.9 item 7: the share of ``seeds`` on balance-v1 ``clean`` where ``ext`` survives the
    whole episode, given the 10 balance features, 5 zeros appended, and +100 rad added to both
    wheel positions."""
    from topple.challenge import load_challenge
    from topple.envs.balance import BalanceEnv

    if len(seeds) == 0:
        raise ValueError("sanity_check needs at least one seed")
    env = BalanceEnv("clean", challenge=load_challenge("balance-v1"))
    survived = 0
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                x = np.concatenate([np.asarray(obs, dtype=np.float64), np.zeros(5)])
                x[list(WHEEL_POS)] += SANITY_WHEEL_OFFSET_RAD
                obs, _, terminated, truncated, _ = env.step(ext.actions(x)[0].astype(np.float32))
            survived += int(not terminated)
    finally:
        env.close()
    return survived / len(seeds)
