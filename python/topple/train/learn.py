"""``topple train``'s engine: PPO or SAC with selection evals and pickle-free snapshots
(docs/design/phase-2.md §1, §2; train extra, imported lazily by the CLI).

``train(cfg, out, *, config_file_sha256)`` does six things:
1. builds the vectorized, normalized training envs;
2. measures the gate baseline once on the selection seeds;
3. trains with ``SelectionCallback``, which handles evals, snapshots and hard limits;
4. writes ``train_meta.json`` (§2.3);
5. writes ``selection_scores.npz``;
6. writes ``best.json``.

The run directory holds only run-relative paths. ``resume_from`` is the one place that loads
SB3's pickles.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from topple.challenge import ENV_ALGO_VERSION, load_challenge
from topple.evaluate import cpu_isa, platform_tag
from topple.export import _npz_bytes, read_run_json
from topple.pool import eval_pool
from topple.train.callbacks import HealthCallback, SelectionCallback, SelectionContext
from topple.train.config import (
    MAX_TRAIN_META_BYTES,
    PpoTrainConfig,
    SacTrainConfig,
    config_doc,
    config_sha256,
    lr_fn,
    validate_train_config,
)
from topple.train.resources import ResourceSampler
from topple.train.selection import (
    FLAGS,
    EvalTrigger,
    choose_best,
    measure_gate_baseline,
    run_flags,
    selection_protocol,
)
from topple.train.surgery import ExtendedInit, InitError, extended_init
from topple.train.wrappers import make_train_env

REPO = Path(__file__).resolve().parents[3]
PID_SUBMISSION = REPO / "submissions" / "pid_example.py"
SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
VERSION_DISTS = {
    "topple": "topple-sim",
    "torch": "torch",
    "stable_baselines3": "stable-baselines3",
    "gymnasium": "gymnasium",
    "mujoco": "mujoco",
    "numpy": "numpy",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "protobuf": "protobuf",
    "psutil": "psutil",
}

Config = PpoTrainConfig | SacTrainConfig


def train(cfg: Config, out: Path, *, config_file_sha256: str) -> Path:
    """Train ``cfg`` into the run directory ``out`` (missing or empty) and return it. Exits
    normally after a hard limit too: the abort flag is in ``train_meta.flags``, and the CLI
    exits 4."""
    out = Path(out)
    ext = _extended_init(cfg)  # first: an unusable init source fails before anything is made
    out.mkdir(parents=True, exist_ok=True)
    for name in SINGLE_THREAD_ENV:  # inherited by the env workers and the eval pool
        os.environ[name] = "1"
    torch.set_num_threads(1)
    started = time.monotonic()
    git_state = _git()  # the code that runs: read before training, never after
    load_start = os.getloadavg()  # the 1, 5 and 15 minute averages (§2.3, phase2.json)
    challenge = load_challenge(cfg.env.challenge)
    e_ref_j = float(challenge.config.score.e_ref_j)
    protocol = selection_protocol(cfg.selection)
    if cfg.selection.gate == "pid" and not PID_SUBMISSION.is_file():
        raise FileNotFoundError(f"the pid gate needs {PID_SUBMISSION.relative_to(REPO)}")
    trigger = (
        EvalTrigger(cfg.selection.every_steps)
        if cfg.algo == "ppo"
        else EvalTrigger(cfg.selection.every_steps, cfg.total_timesteps)
    )
    sampler = ResourceSampler()
    sampler.start()
    vecnorm: Any = None
    try:
        with eval_pool(cfg.selection.workers) as pool:
            t0 = time.monotonic()
            baseline = measure_gate_baseline(
                cfg.selection.gate, protocol, pool, pid_submission=PID_SUBMISSION
            )
            baseline_s = time.monotonic() - t0
            vecnorm = _vec_env(cfg)
            model = _model(cfg, vecnorm, out)
            if ext is not None:
                apply_init(model, vecnorm, ext)
            ctx = SelectionContext(
                cfg=cfg,
                out=out,
                vecnorm=vecnorm,
                pool=pool,
                protocol=protocol,
                baseline=baseline,
                challenge=challenge,
                e_ref_j=e_ref_j,
                sampler=sampler,
                started=started,
                trigger=trigger,
            )
            callbacks: list[Any] = [SelectionCallback(ctx)]
            if cfg.algo == "ppo":
                callbacks.append(HealthCallback())
            model.learn(
                total_timesteps=cfg.total_timesteps,
                callback=CallbackList(callbacks),
                tb_log_name="tb",
                progress_bar=False,
            )
            num_timesteps = int(model.num_timesteps)
    finally:
        if vecnorm is not None:
            vecnorm.close()
        sampler.stop()
    wall = time.monotonic() - started
    _write_outputs(
        cfg,
        out,
        ctx,
        num_timesteps=num_timesteps,
        wall_clock_s=wall,
        baseline_s=baseline_s,
        load_avg_start=load_start,
        config_file_sha256=config_file_sha256,
        git_state=git_state,
        protocol_hash=protocol.protocol_hash(),
        bundle_hash=challenge.bundle_hash(),
        e_ref_j=e_ref_j,
        init_source=None if ext is None else ext.source.to_json(),
    )
    return out


def _extended_init(cfg: Config) -> ExtendedInit | None:
    if cfg.algo != "ppo" or cfg.init is None:
        return None
    return extended_init(cfg)


def apply_init(model: Any, vecnorm: VecNormalize, ext: ExtendedInit) -> None:
    """Load ``ext`` into a fresh PPO ``model`` (its optimizer stays fresh) and freeze
    ``vecnorm``'s observation statistics at ``ext``'s (§8.8 steps 1-5)."""
    state = model.policy.state_dict()
    if set(state) != set(ext.state):
        raise InitError(f"the policy's parameters {sorted(set(state) ^ set(ext.state))} differ")
    for key, value in ext.state.items():
        if tuple(state[key].shape) != value.shape:
            raise InitError(f"{key}: shape {value.shape}, the policy has {tuple(state[key].shape)}")
    tensors = {
        key: torch.from_numpy(np.array(value, dtype=np.float32)) for key, value in ext.state.items()
    }
    model.policy.load_state_dict(tensors, strict=True)
    rms: Any = vecnorm.obs_rms  # a RunningMeanStd for a Box observation space
    if rms.mean.shape != ext.obs_mean.shape:
        raise InitError(f"the env observes {rms.mean.shape}, the init {ext.obs_mean.shape}")
    rms.mean = ext.obs_mean.copy()
    rms.var = ext.obs_var.copy()
    rms.count = ext.obs_count
    vecnorm.training = False


def _vec_env(cfg: Config) -> VecNormalize:
    fns: list[Any] = [partial(make_train_env, cfg, i) for i in range(cfg.n_envs)]
    venv = (
        SubprocVecEnv(fns, start_method=cfg.start_method)
        if cfg.vec_env == "subproc"
        else DummyVecEnv(fns)
    )
    gamma = cfg.ppo.gamma if cfg.algo == "ppo" else cfg.sac.gamma
    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=cfg.normalize.reward,
        clip_obs=cfg.normalize.clip_obs,
        epsilon=cfg.normalize.epsilon,
        gamma=gamma,
    )


def _model(cfg: Config, vecnorm: VecNormalize, out: Path) -> Any:
    tb = str(out / "tb")
    if cfg.algo == "ppo":
        p = cfg.ppo
        return PPO(
            "MlpPolicy",
            vecnorm,
            n_steps=p.n_steps,
            batch_size=p.batch_size,
            n_epochs=p.n_epochs,
            gamma=p.gamma,
            gae_lambda=p.gae_lambda,
            learning_rate=lr_fn(p.learning_rate),
            clip_range=p.clip_range,
            ent_coef=p.ent_coef,
            vf_coef=p.vf_coef,
            max_grad_norm=p.max_grad_norm,
            target_kl=p.target_kl,
            policy_kwargs={
                "net_arch": {
                    "pi": list(cfg.policy.net_arch.pi),
                    "vf": list(cfg.policy.net_arch.vf),
                },
                "activation_fn": torch.nn.Tanh,
                "log_std_init": cfg.policy.log_std_init,
            },
            seed=cfg.seed,
            device="cpu",
            tensorboard_log=tb,
            verbose=0,
        )
    s = cfg.sac
    return SAC(
        "MlpPolicy",
        vecnorm,
        buffer_size=s.buffer_size,
        learning_starts=s.learning_starts,
        batch_size=s.batch_size,
        gamma=s.gamma,
        tau=s.tau,
        train_freq=s.train_freq,
        gradient_steps=s.gradient_steps,
        ent_coef=s.ent_coef,
        learning_rate=lr_fn(s.learning_rate),
        policy_kwargs={
            "net_arch": {"pi": list(cfg.policy.net_arch.pi), "qf": list(cfg.policy.net_arch.qf)},
            "activation_fn": torch.nn.ReLU,
        },
        seed=cfg.seed,
        device="cpu",
        tensorboard_log=tb,
        verbose=0,
    )


# --- outputs ------------------------------------------------------------------------------------


def _git() -> tuple[str | None, bool | None]:
    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return (
        None if commit is None else commit.strip(),
        None if status is None else bool(status.strip()),
    )


def _versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": platform.python_version()}
    for key, dist in VERSION_DISTS.items():
        try:
            found[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            found[key] = None
    return found


def _write_outputs(
    cfg: Config,
    out: Path,
    ctx: SelectionContext,
    *,
    num_timesteps: int,
    wall_clock_s: float,
    baseline_s: float,
    load_avg_start: tuple[float, float, float],
    config_file_sha256: str,
    git_state: tuple[str | None, bool | None],
    protocol_hash: str,
    bundle_hash: str,
    e_ref_j: float,
    init_source: dict[str, Any] | None = None,
) -> None:
    history = ctx.history
    best = choose_best(history, ctx.challenge.kind)  # type: ignore[arg-type]
    if best is None:
        raise RuntimeError("training finished without a selection eval")
    raised = set(run_flags(history, health=getattr(cfg, "health", None), e_ref_j=e_ref_j))
    if ctx.abort is not None:
        raised.add(ctx.abort)
    reading = ctx.final_reading or ctx.sampler.reading()
    selection_s = ctx.selection_s + baseline_s
    train_s = max(wall_clock_s - selection_s, 0.0)
    commit, dirty = git_state
    meta: dict[str, Any] = {
        "config_sha256": config_sha256(cfg),
        "config_file_sha256": config_file_sha256,
        "config": config_doc(cfg),
        "git_commit": commit,
        "git_dirty": dirty,
        "versions": _versions(),
        "torch_threads": torch.get_num_threads(),
        "mp_start_method": cfg.start_method,
        "platform": platform_tag(),
        "cpu_isa": cpu_isa(),
        "e_ref_j": e_ref_j,
        "selection_protocol_hash": protocol_hash,
        "env_algo_version": ENV_ALGO_VERSION,
        "bundle_hash": bundle_hash,
        "load_avg_start": list(load_avg_start),
        "load_avg_end": list(os.getloadavg()),
        "wall_clock_s": wall_clock_s,
        "train_s": train_s,
        "selection_s": selection_s,
        "cpu_s": reading.cpu_s,
        "env_steps_per_s": num_timesteps / train_s if train_s > 0 else 0.0,
        "peak_tree_rss_mib": reading.peak_tree_rss_mib,
        "peak_tree_rss_at_step": reading.peak_tree_rss_at_step,
        "peak_procs": reading.peak_procs,
        "gate_baseline": ctx.baseline.to_json(),
        "selection_history": [row.to_json() for row in history],
        "best": {"step": best.step, "eligible": best.eligible, "onnx_sha256": best.onnx_sha256},
        "flags": [f for f in FLAGS if f in raised],
        "num_timesteps": num_timesteps,
    }
    if init_source is not None:
        meta["init_source"] = init_source
    data = (json.dumps(meta, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > MAX_TRAIN_META_BYTES:
        raise RuntimeError(f"train_meta.json is {len(data)} bytes: at most {MAX_TRAIN_META_BYTES}")
    (out / "train_meta.json").write_bytes(data)
    scores = np.array(ctx.scores, dtype=np.float64)
    steps = np.array([row.step for row in history], dtype=np.int64)
    (out / "selection_scores.npz").write_bytes(_npz_bytes({"scores": scores, "steps": steps}))


# --- resume -------------------------------------------------------------------------------------


def _owned_file(path: Path) -> Path:
    st = os.lstat(path)
    if not os.path.isfile(path) or os.path.islink(path):
        raise PermissionError(f"{path.name} is not a regular file")
    if st.st_uid != os.getuid():
        raise PermissionError(f"{path.name} is not owned by this user: refusing to unpickle it")
    return path


def _owned_directory(path: Path) -> None:
    st = os.lstat(path)
    if os.path.islink(path) or not os.path.isdir(path):
        raise PermissionError(f"{path.name} is not a plain directory")
    if st.st_uid != os.getuid():
        raise PermissionError(f"{path.name} is not owned by this user")


def resume_from(run_dir: Path, step: int | None = None) -> tuple[Any, VecNormalize]:
    """Reload a snapshot's SB3 model and ``VecNormalize`` (``training=False``) from
    ``model.zip`` and ``vecnormalize.pkl``: the only place pickles are loaded, and only files
    owned by the current user (§2.1). ``step`` defaults to ``best.json``'s."""
    run_dir = Path(run_dir)
    meta = read_run_json(run_dir / "train_meta.json")
    if not isinstance(meta, dict):
        raise ValueError("train_meta.json is not an object")
    cfg = validate_train_config(meta.get("config"))
    if step is None:
        best = read_run_json(run_dir / "best.json")
        if not isinstance(best, dict) or type(best.get("step")) is not int:
            raise ValueError("best.json has no integer step")
        step = int(best["step"])
    step_dir = run_dir / "ckpt" / f"step_{step}"
    for directory in (run_dir / "ckpt", step_dir):  # a swapped parent would redirect the loads
        _owned_directory(directory)
    model_zip = _owned_file(step_dir / "model.zip")
    pkl = _owned_file(step_dir / "vecnormalize.pkl")
    fns: list[Any] = [partial(make_train_env, cfg, 0)]
    venv = DummyVecEnv(fns)
    vecnorm = VecNormalize.load(str(pkl), venv)
    vecnorm.training = False
    vecnorm.norm_reward = False
    algo = PPO if cfg.algo == "ppo" else SAC
    model = algo.load(str(model_zip), env=vecnorm, device="cpu")
    return model, vecnorm
