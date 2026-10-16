"""``topple train`` end to end on the smoke configs (docs/design/phase-2.md §2.3, §2.4, §6).

Each smoke config trains twice with the same seed. Both runs must give the same arrays, the
same ONNX bytes at every label and the same selection history: training is deterministic per
(config, seed, platform, start method). The run exports torch-free, and its snapshots hold
exactly the bytes the selection eval scored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # the train extra; the default CI jobs collect without it
pytest.importorskip("stable_baselines3")

from typer.testing import CliRunner

from topple.cli import app
from topple.export import export_policy, read_run_json
from topple.train.config import config_sha256, load_train_config, read_train_config
from topple.train.learn import train
from topple.train.selection import eval_labels

pytestmark = pytest.mark.train

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "train"
runner = CliRunner()


def _train(name: str, out: Path) -> Path:
    loaded = read_train_config(CONFIGS / f"{name}.yaml")
    return train(loaded.config, out, config_file_sha256=loaded.file_sha256)


def _digests(run: Path) -> dict[str, str]:
    return {
        str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((run / "ckpt").rglob("*.npz"))
        if p.name in ("actor.npz", "critic.npz")
    }


def _meta(run: Path) -> dict[str, object]:
    meta = read_run_json(run / "train_meta.json")
    assert isinstance(meta, dict)
    return meta


@pytest.fixture(scope="module", params=["smoke_ppo", "smoke_sac"])
def twice(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    name = request.param
    root = tmp_path_factory.mktemp(name)
    return name, _train(name, root / "a"), _train(name, root / "b")


def test_snapshots_at_exactly_the_eval_labels(twice: tuple) -> None:
    name, run, _ = twice
    cfg = load_train_config(CONFIGS / f"{name}.yaml")
    steps = sorted(int(p.name.removeprefix("step_")) for p in (run / "ckpt").iterdir())
    assert tuple(steps) == eval_labels(cfg)
    assert len(steps) == 2
    meta = _meta(run)
    assert steps[-1] == meta["num_timesteps"]
    assert [row["step"] for row in meta["selection_history"]] == steps  # type: ignore[index]
    assert not list((run / "ckpt").glob(".step_*"))  # no temp directory left behind


def test_train_meta_records_the_run(twice: tuple) -> None:
    name, run, _ = twice
    cfg = load_train_config(CONFIGS / f"{name}.yaml")
    meta = _meta(run)
    assert meta["config_sha256"] == config_sha256(cfg)
    assert meta["mp_start_method"] == cfg.start_method
    assert meta["torch_threads"] == 1
    assert meta["flags"] == [] or set(meta["flags"]) <= {"no_eligible", "high_energy"}  # type: ignore[arg-type]
    assert (run / "train_meta.json").stat().st_size <= 64 * 1024
    best = json.loads((run / "best.json").read_text())
    assert best["step"] == meta["best"]["step"]  # type: ignore[index]
    scores = np.load(run / "selection_scores.npz")
    assert scores["scores"].shape == (2, len(cfg.selection.seeds.seeds))
    assert scores["steps"].tolist() == [row["step"] for row in meta["selection_history"]]  # type: ignore[index]


def test_no_absolute_paths_in_run_files(twice: tuple) -> None:
    _, run, _ = twice
    texts = [run / "train_meta.json", run / "best.json"]
    texts += sorted((run / "ckpt").rglob("*.json"))
    for path in texts:
        assert str(run) not in path.read_text(), path
        assert "/Users/" not in path.read_text(), path
        assert "/home/" not in path.read_text(), path


def test_snapshot_binds_the_scored_onnx(twice: tuple) -> None:
    _, run, _ = twice
    meta = _meta(run)
    for row in meta["selection_history"]:  # type: ignore[union-attr]
        step_dir = run / "ckpt" / f"step_{row['step']}"
        snap_meta = json.loads((step_dir / "meta.json").read_text())
        assert snap_meta["onnx_sha256"] == row["onnx_sha256"]
        with np.load(step_dir / "parity_ref.npz") as parity:
            assert str(parity["onnx_sha256"]) == row["onnx_sha256"]


def test_training_is_deterministic(twice: tuple) -> None:
    _, a, b = twice
    assert _digests(a) == _digests(b)
    hist_a = _meta(a)["selection_history"]
    hist_b = _meta(b)["selection_history"]
    strip = ("cpu_s", "wall_s")  # resource numbers, not results
    assert [{k: v for k, v in r.items() if k not in strip} for r in hist_a] == [  # type: ignore[union-attr]
        {k: v for k, v in r.items() if k not in strip}
        for r in hist_b  # type: ignore[union-attr]
    ]


def test_the_run_exports_torch_free_with_parity(twice: tuple) -> None:
    """A smoke policy is untrained, so it cannot meet export's saturated-row coverage bar;
    ``--report-only`` still rebuilds the ONNX, binds it to the scored bytes and checks parity
    row by row (§3.3)."""
    _, run, _ = twice
    manifest = export_policy(run, run / "report_policy", report_only=True, replace=True)
    assert (run / "report_policy" / "policy.onnx").is_file()
    assert manifest is not None


def test_cli_train_refuses_a_non_empty_out(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "x").write_text("x")
    done = runner.invoke(app, ["train", str(CONFIGS / "smoke_ppo.yaml"), "--out", str(out)])
    assert done.exit_code == 2


@pytest.mark.parametrize(
    ("limits", "flag"),
    [
        ({"max_wall_clock_s": 300, "max_cpu_s": 1, "max_tree_rss_mib": 4096}, "cpu_abort"),
        ({"max_wall_clock_s": 300, "max_cpu_s": 600, "max_tree_rss_mib": 1}, "rss_abort"),
    ],
)
def test_a_hard_limit_stops_with_a_final_snapshot_and_exit_4(
    tmp_path: Path, limits: dict[str, int], flag: str
) -> None:
    """§1: a hard limit stops training, the final snapshot and selection eval still run, the
    flag is recorded and ``topple train`` exits 4."""
    import yaml

    doc = yaml.safe_load((CONFIGS / "smoke_ppo.yaml").read_text())
    doc["limits"] = limits
    doc["total_timesteps"] = 1_000_000  # far more than the limit allows
    doc["selection"]["every_steps"] = 50_000
    config = tmp_path / "limited.yaml"
    config.write_text(yaml.safe_dump(doc))
    out = tmp_path / "run"
    done = runner.invoke(app, ["train", str(config), "--out", str(out)])
    assert done.exit_code == 4, done.output
    meta = _meta(out)
    assert flag in meta["flags"]  # type: ignore[operator]
    assert meta["num_timesteps"] < 1_000_000  # type: ignore[operator]
    final = out / "ckpt" / f"step_{meta['num_timesteps']}"
    assert (final / "actor.npz").is_file()


def test_subproc_forkserver_with_the_pid_gate_is_deterministic(tmp_path: Path) -> None:
    """The shipped runs' path: SubprocVecEnv with forkserver workers and the PID gate (baseline
    measured, eligibility applied in the loop). Two runs with the same seed give the same
    arrays and history (what --retry and G10 rely on)."""
    import yaml

    from topple.train.config import parse_train_config

    doc = yaml.safe_load((CONFIGS / "smoke_ppo.yaml").read_text())
    doc["vec_env"] = "subproc"
    doc["start_method"] = "forkserver"
    doc["n_envs"] = 2
    doc["selection"]["gate"] = "pid"
    doc["selection"]["max_saturation"] = 0.08
    data = yaml.safe_dump(doc).encode()
    cfg = parse_train_config(data)
    file_sha = hashlib.sha256(data).hexdigest()
    a = train(cfg, tmp_path / "a", config_file_sha256=file_sha)
    b = train(cfg, tmp_path / "b", config_file_sha256=file_sha)
    assert _digests(a) == _digests(b)
    strip = ("cpu_s", "wall_s")
    hist_a = [{k: v for k, v in r.items() if k not in strip} for r in _meta(a)["selection_history"]]  # type: ignore[union-attr]
    hist_b = [{k: v for k, v in r.items() if k not in strip} for r in _meta(b)["selection_history"]]  # type: ignore[union-attr]
    assert hist_a == hist_b
    baseline = _meta(a)["gate_baseline"]
    assert baseline["rule"] == "pid"  # type: ignore[index]
    assert baseline["reference"] is not None  # type: ignore[index]


def test_the_train_meta_is_a_valid_phase2_summary(twice: tuple) -> None:
    """Every §5.1 summary field comes from train_meta.json and validates as phase2.json's
    RunSummary, so phase2_results.py never has to invent a value."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from phase2_common import RunSummary

    _, run, _ = twice
    meta = _meta(run)
    cfg = meta["config"]
    summary = {
        "run": "smoke",
        "algo": cfg["algo"],  # type: ignore[index]
        "seed": cfg["seed"],  # type: ignore[index]
        "total_timesteps": cfg["total_timesteps"],  # type: ignore[index]
        "attempts": [],
        **{k: meta[k] for k in RunSummary.model_fields if k in meta},
        "best": {**meta["best"], "rule": cfg["selection"]["gate"]},  # type: ignore[dict-item,index]
    }
    RunSummary.model_validate_json(json.dumps(summary), strict=True)
