"""The fine-tune init (docs/design/phase-2.md §8.8 arm b, §6 ``test_surgery``).

A smoke balance run is extended to the 15-input waypoint observation with its pickles deleted:
the extended policy acts as the balance policy whatever the new inputs and with yaw and the
wheel positions at their mean (or +100 rad off it), its value is ``balance_scale`` times the
balance value, ``log_std`` restarts at ``init.log_std``, a real PPO policy loads it, and a
fine-tune run records ``init_source``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")  # the train extra; the default CI jobs collect without it
pytest.importorskip("stable_baselines3")

import yaml
from typer.testing import CliRunner

from topple.cli import app
from topple.export import export_policy, read_run_json
from topple.train.config import PpoTrainConfig, parse_train_config, read_train_config
from topple.train.learn import _model, _vec_env, apply_init, train
from topple.train.surgery import (
    InitError,
    choose_source,
    extend_obs,
    sanity_check,
)

pytestmark = pytest.mark.train

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "train"
YAW, WHEELS = 2, (4, 5)


def _yaml(name: str) -> dict[str, Any]:
    doc = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    assert isinstance(doc, dict)
    return doc


def _tiny_ft(from_runs: list[str], **init: object) -> PpoTrainConfig:
    """ppo_waypoint_ft.yaml cut to a smoke run's size."""
    doc = _yaml("ppo_waypoint_ft")
    doc.update(total_timesteps=1024, n_envs=2, vec_env="dummy")
    doc["ppo"].update(n_steps=512, batch_size=256)
    doc["selection"].update(seeds={"first": 2000, "last": 2001}, every_steps=1024, workers=2)
    doc["limits"] = {"max_wall_clock_s": 600, "max_cpu_s": 1200, "max_tree_rss_mib": 4096}
    doc["init"] = {**doc["init"], "from_runs": from_runs, **init}
    cfg = parse_train_config(yaml.safe_dump(doc).encode())
    assert isinstance(cfg, PpoTrainConfig)
    return cfg


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding ``smoke/``: smoke_ppo trained, its SB3 pickles deleted."""
    base = tmp_path_factory.mktemp("surgery")
    loaded = read_train_config(CONFIGS / "smoke_ppo.yaml")
    run = train(loaded.config, base / "smoke", config_file_sha256=loaded.file_sha256)
    for pickle_file in [*run.rglob("model.zip"), *run.rglob("vecnormalize.pkl")]:
        pickle_file.unlink()
    return base


def _source_policy(run: Path, step: int) -> tuple[Any, Any, Any]:
    """The 10-input actor layers, critic arrays and normalization, read with numpy."""
    d = run / "ckpt" / f"step_{step}"
    actor = np.load(d / "actor.npz")
    critic = dict(np.load(d / "critic.npz"))
    norm = json.loads((d / "normalize.json").read_text())
    return actor, critic, norm


def _z(x: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float, clip: float) -> np.ndarray:
    return np.clip((x - mean) / np.sqrt(var + eps), -clip, clip)


def _mlp(z: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    h = z
    for w, b in layers[:-1]:
        h = np.tanh(h @ np.asarray(w, np.float64).T + np.asarray(b, np.float64))
    w, b = layers[-1]
    return h @ np.asarray(w, np.float64).T + np.asarray(b, np.float64)


def _rows(norm: dict[str, Any], n: int = 64) -> np.ndarray:
    """``n`` balance observations around the training statistics, with yaw and the wheel
    positions at their mean (normalized 0)."""
    rng = np.random.default_rng(7)
    mean, sd = np.asarray(norm["mean"]), np.sqrt(np.asarray(norm["var"]))
    x = mean + sd * rng.normal(size=(n, len(mean)))
    x[:, [YAW, *WHEELS]] = mean[[YAW, *WHEELS]]
    return x


def _copy_run(
    root: Path, dest: Path, *, flags: list[str] | None = None, eligible: bool = True
) -> None:
    shutil.copytree(root / "smoke", dest)
    meta = json.loads((dest / "train_meta.json").read_text())
    if flags is not None:
        meta["flags"] = flags
    if not eligible:
        for row in meta["selection_history"]:
            row["eligible"] = False
    (dest / "train_meta.json").write_text(json.dumps(meta))


def test_the_source_is_chosen_by_seed_with_fallback(root: Path, tmp_path: Path) -> None:
    """§8.8: seed k starts from from_runs[k mod n]; an unusable run passes to k+1, k+2 mod n."""
    for name in ("a", "b"):
        _copy_run(root, tmp_path / "runs" / name)
    init = _tiny_ft(["runs/a", "runs/b"]).init
    assert init is not None
    assert [choose_source(init, k, cwd=tmp_path).run for k in range(4)] == [
        "runs/a",
        "runs/b",
        "runs/a",
        "runs/b",
    ]
    shutil.rmtree(tmp_path / "runs" / "a")
    _copy_run(root, tmp_path / "runs" / "a", eligible=False)
    assert choose_source(init, 0, cwd=tmp_path).run == "runs/b"
    shutil.rmtree(tmp_path / "runs" / "b")
    _copy_run(root, tmp_path / "runs" / "b", flags=["wall_clock_abort"])
    with pytest.raises(InitError, match="runs/a: its best step"):
        choose_source(init, 1, cwd=tmp_path)  # b aborted, wraps to a (ineligible): none left
    cfg = _tiny_ft(["missing", "smoke"])
    assert cfg.init is not None
    assert choose_source(cfg.init, 0, cwd=root).run == "smoke"  # "missing" falls through
    with pytest.raises(InitError, match="missing: not a run directory"):
        choose_source(_tiny_ft(["missing"]).init, 0, cwd=root)  # type: ignore[arg-type]


def test_an_ineligible_or_aborted_source_is_skipped(root: Path, tmp_path: Path) -> None:
    for name, edit in (("aborted", "flags"), ("ineligible", "eligible")):
        run = tmp_path / name
        shutil.copytree(root / "smoke", run)
        meta = json.loads((run / "train_meta.json").read_text())
        if edit == "flags":
            meta["flags"] = ["cpu_abort"]
        else:
            for row in meta["selection_history"]:
                row["eligible"] = False
        (run / "train_meta.json").write_text(json.dumps(meta))
    cfg = _tiny_ft(["aborted", "ineligible"])
    with pytest.raises(InitError, match=r"flags \['cpu_abort'\].*not eligible"):
        choose_source(cfg.init, 0, cwd=tmp_path)  # type: ignore[arg-type]


def test_the_extended_policy_acts_as_the_balance_policy(root: Path) -> None:
    cfg = _tiny_ft(["smoke"])
    assert cfg.init is not None
    ext = extend_obs(cfg, choose_source(cfg.init, 0, cwd=root), cwd=root)
    actor, critic, norm = _source_policy(root / "smoke", ext.source.step)
    x10 = _rows(norm)
    z10 = _z(
        x10, np.asarray(norm["mean"]), np.asarray(norm["var"]), norm["epsilon"], norm["clip_obs"]
    )
    a10 = np.clip(_mlp(z10, [(actor[f"W{k}"], actor[f"b{k}"]) for k in range(3)]), -1.0, 1.0)
    rng = np.random.default_rng(11)
    for new in (np.zeros((len(x10), 5)), rng.normal(scale=3.0, size=(len(x10), 5))):
        x15 = np.concatenate([x10, new], axis=1)
        np.testing.assert_allclose(ext.actions(x15), a10, atol=1e-6)
        x15[:, list(WHEELS)] += 100.0  # +100 rad of wheel travel changes nothing
        np.testing.assert_allclose(ext.actions(x15), a10, atol=1e-6)
    assert ext.state["log_std"].tolist() == [-1.0, -1.0]
    assert ext.obs_var[10:].tolist() == pytest.approx([0.49] * 4 + [0.25])
    assert ext.obs_mean[10:].tolist() == [0.0, 0.0, 0.0, 0.0, 0.5]
    # V_15 = balance_scale * V_10
    v10 = _mlp(
        z10,
        [
            (critic["mlp_extractor.value_net.0.weight"], critic["mlp_extractor.value_net.0.bias"]),
            (critic["mlp_extractor.value_net.2.weight"], critic["mlp_extractor.value_net.2.bias"]),
            (critic["value_net.weight"], critic["value_net.bias"]),
        ],
    )
    z15 = _z(
        np.concatenate([x10, new], axis=1), ext.obs_mean, ext.obs_var, ext.epsilon, ext.clip_obs
    )
    s = ext.state
    v15 = _mlp(
        z15,
        [
            (s["mlp_extractor.value_net.0.weight"], s["mlp_extractor.value_net.0.bias"]),
            (s["mlp_extractor.value_net.2.weight"], s["mlp_extractor.value_net.2.bias"]),
            (s["value_net.weight"], s["value_net.bias"]),
        ],
    )
    np.testing.assert_allclose(v15, 0.1 * v10, atol=1e-6)


def test_a_ppo_policy_loads_the_init(root: Path, tmp_path: Path) -> None:
    cfg = _tiny_ft(["smoke"])
    assert cfg.init is not None
    ext = extend_obs(cfg, choose_source(cfg.init, 0, cwd=root), cwd=root)
    vecnorm = _vec_env(cfg)
    try:
        model = _model(cfg, vecnorm, tmp_path)
        apply_init(model, vecnorm, ext)
        assert vecnorm.training is False
        _, _, norm = _source_policy(root / "smoke", ext.source.step)
        x15 = np.concatenate([_rows(norm), np.ones((64, 5))], axis=1)
        actions, _ = model.predict(vecnorm.normalize_obs(x15), deterministic=True)
        np.testing.assert_allclose(np.clip(actions, -1, 1), ext.actions(x15), atol=1e-5)
        assert float(model.policy.log_std.detach()[0]) == -1.0
    finally:
        vecnorm.close()


def test_a_source_whose_weights_changed_is_refused(root: Path, tmp_path: Path) -> None:
    """The weights must rebuild the ONNX the source's selection eval scored."""
    _copy_run(root, tmp_path / "edited")
    cfg = _tiny_ft(["edited"])
    assert cfg.init is not None
    source = choose_source(cfg.init, 0, cwd=tmp_path)
    actor_path = tmp_path / "edited" / "ckpt" / f"step_{source.step}" / "actor.npz"
    arrays = dict(np.load(actor_path))
    arrays["b2"] = arrays["b2"] + np.float32(0.5)
    np.savez(actor_path, **arrays)
    with pytest.raises(InitError, match="do not rebuild"):
        extend_obs(cfg, source, cwd=tmp_path)


def test_a_source_with_another_architecture_is_refused(root: Path) -> None:
    doc = _tiny_ft(["smoke"]).model_dump(mode="json")
    doc["policy"]["net_arch"] = {"pi": [32, 32], "vf": [64, 64]}
    cfg = PpoTrainConfig.model_validate_json(json.dumps(doc), strict=True)
    assert cfg.init is not None
    with pytest.raises(InitError, match="net_arch"):
        extend_obs(cfg, choose_source(cfg.init, 0, cwd=root), cwd=root)


def test_the_sanity_check_is_a_survival_rate(root: Path) -> None:
    cfg = _tiny_ft(["smoke"])
    assert cfg.init is not None
    ext = extend_obs(cfg, choose_source(cfg.init, 0, cwd=root), cwd=root)
    rate = sanity_check(ext, range(2))
    assert rate in (0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="at least one seed"):
        sanity_check(ext, range(0))


def test_a_fine_tune_run_records_its_source(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(root)
    cfg = _tiny_ft(["smoke"])
    assert cfg.init is not None
    ext = extend_obs(cfg, choose_source(cfg.init, 0))
    run = train(cfg, tmp_path / "ft", config_file_sha256="0" * 64)
    meta = read_run_json(run / "train_meta.json")
    assert isinstance(meta, dict)
    source = meta["init_source"]
    best = read_run_json(root / "smoke" / "best.json")
    assert isinstance(best, dict)
    smoke_meta = read_run_json(root / "smoke" / "train_meta.json")
    assert isinstance(smoke_meta, dict)
    (row,) = [r for r in smoke_meta["selection_history"] if r["step"] == best["step"]]
    assert source == {"run": "smoke", "step": best["step"], "onnx_sha256": row["onnx_sha256"]}
    # the statistics stay frozen at the init's in every snapshot, and the run exports
    for norm_file in sorted((run / "ckpt").glob("step_*/normalize.json")):
        norm = json.loads(norm_file.read_text())
        assert norm["mean"] == ext.obs_mean.tolist()
        assert norm["var"] == ext.obs_var.tolist()
        assert norm["count"] == ext.obs_count
    export_policy(run, run / "report_policy", report_only=True)


def test_the_cli_refuses_an_unusable_init_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "ft.yaml"
    config.write_text(yaml.safe_dump(_tiny_ft(["runs/none"]).model_dump(mode="json")))
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["train", str(config), "--out", str(out)])
    assert result.exit_code == 2, result.output
    assert "init: no init.from_runs run is usable" in result.output
    assert not out.exists()
