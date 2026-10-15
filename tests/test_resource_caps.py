"""Resource caps (docs/design/phase-2.md §1, §6 ``test_resource_caps``).

Every evaluation pool, compare's and shove's included, is ``min(workers, 4, os.cpu_count())``
spawn processes, whatever is asked. Every shipped training config uses at most 8 envs and writes
its three hard limits out. ``topple train`` validates its config and ``--out`` before it imports
the training code: a non-empty ``--out`` exits 2, and ``--retry`` moves only a run that stopped on
its wall-clock limit aside (``<out>.attempt1``, then ``.attempt2``). The training code itself (the
``train`` extra) is stood in for by a fake ``topple.train.learn`` here.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import json
import multiprocessing
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import topple.pool as pool_module
from pool_support import InlinePool
from topple.bench import shove_curve
from topple.cli import app
from topple.compare import compare
from topple.evaluate import EvalProtocol
from topple.pool import POOL_MAX_WORKERS, eval_pool, pool_size
from topple.train.config import PpoTrainConfig

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("workers", [4, 5, 16, 64, 10_000])
def test_pools_clamp_to_4(monkeypatch: pytest.MonkeyPatch, workers: int) -> None:
    monkeypatch.setattr(pool_module.os, "cpu_count", lambda: 64)
    pool = eval_pool(workers)  # no process starts before the first submit
    try:
        assert pool._max_workers == POOL_MAX_WORKERS == 4
        assert isinstance(pool._mp_context, multiprocessing.context.SpawnContext)
    finally:
        pool.shutdown()


def test_a_2_cpu_runner_gets_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI's 2-vCPU runner: ``--workers 4`` becomes 2 (§4)."""
    monkeypatch.setattr(pool_module.os, "cpu_count", lambda: 2)
    assert pool_size(4) == 2


class _RecordingExecutor(InlinePool):
    """Stands in for ``ProcessPoolExecutor``: records the size asked, runs inline."""

    sizes: list[int] = []  # noqa: RUF012

    def __init__(self, max_workers: int, mp_context: Any) -> None:
        super().__init__(max_workers)
        _RecordingExecutor.sizes.append(max_workers)
        assert isinstance(mp_context, multiprocessing.context.SpawnContext)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    _RecordingExecutor.sizes = []
    monkeypatch.setattr(pool_module.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(pool_module, "ProcessPoolExecutor", _RecordingExecutor)
    return _RecordingExecutor.sizes


def test_compare_and_shove_pools_are_clamped(recorded: list[int], tmp_path: Path) -> None:
    sub = tmp_path / "pid.py"
    shutil.copyfile(REPO / "submissions" / "pid_example.py", sub)
    protocol = EvalProtocol.from_id("balance-v1/ideal")
    compare([sub], protocol.__class__(protocol.challenge, "ideal", (1000,)), workers=64)
    shove_curve(sub, [0.0], [0], workers=64)
    assert recorded == [4, 4]


# --- training configs ---------------------------------------------------------------------------

CONFIGS = REPO / "configs" / "train"
SHIPPED = sorted(CONFIGS.glob("*.yaml"))
LIMIT_KEYS = {"max_wall_clock_s", "max_cpu_s", "max_tree_rss_mib"}
runner = CliRunner()


def test_every_config_is_checked() -> None:
    assert {p.stem for p in SHIPPED} >= {
        "ppo_balance",
        "sac_balance",
        "smoke_ppo",
        "smoke_sac",
        "ablate_nopenalty",
    }


@pytest.mark.parametrize("path", SHIPPED, ids=[p.stem for p in SHIPPED])
def test_shipped_configs_use_at_most_8_envs_and_write_every_limit(path: Path) -> None:
    raw = yaml.safe_load(path.read_text())
    assert 1 <= raw["n_envs"] <= 8
    limits = raw["limits"]
    assert set(limits) == LIMIT_KEYS  # written out: no limit is left to a default
    assert all(type(v) is int and v > 0 for v in limits.values())
    assert limits["max_tree_rss_mib"] <= 8192


# --- topple train ------------------------------------------------------------------------------


class FakeLearn:
    """Stands in for ``topple.train.learn``: records each call and writes the run's
    ``train_meta.json`` with ``flags``."""

    def __init__(self, flags: list[str]) -> None:
        self.flags = flags
        self.calls: list[tuple[Any, Path, str]] = []

    def train(self, cfg: Any, out: Path, *, config_file_sha256: str) -> Path:
        self.calls.append((cfg, out, config_file_sha256))
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_meta.json").write_text(json.dumps({"flags": self.flags}))
        return out


@pytest.fixture
def learn(monkeypatch: pytest.MonkeyPatch) -> FakeLearn:
    fake = FakeLearn([])
    module = ModuleType("topple.train.learn")
    module.train = fake.train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "topple.train.learn", module)
    return fake


def aborted_run(path: Path, flags: list[str]) -> Path:
    path.mkdir(parents=True)
    (path / "train_meta.json").write_text(json.dumps({"flags": flags}))
    (path / "ckpt").mkdir()
    return path


SMOKE = str(CONFIGS / "smoke_ppo.yaml")


def test_train_runs_the_effective_config(learn: FakeLearn, tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(app, ["train", SMOKE, "--out", str(out), "--seed", "2"])
    assert result.exit_code == 0, result.output
    ((cfg, got_out, file_sha),) = learn.calls
    assert isinstance(cfg, PpoTrainConfig)
    assert cfg.seed == 2
    assert got_out == out
    assert file_sha == hashlib.sha256((CONFIGS / "smoke_ppo.yaml").read_bytes()).hexdigest()


def test_train_into_an_empty_out(learn: FakeLearn, tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    result = runner.invoke(app, ["train", SMOKE, "--out", str(tmp_path / "run")])
    assert result.exit_code == 0, result.output
    assert len(learn.calls) == 1


def test_train_on_a_non_empty_out_exits_2(learn: FakeLearn, tmp_path: Path) -> None:
    out = aborted_run(tmp_path / "run", ["wall_clock_abort"])
    result = runner.invoke(app, ["train", SMOKE, "--out", str(out)])
    assert result.exit_code == 2
    assert "exists and is not empty" in result.output
    assert learn.calls == []
    assert (out / "ckpt").is_dir()
    assert not (tmp_path / "run.attempt1").exists()
    plain = tmp_path / "file"
    plain.write_text("x")
    assert runner.invoke(app, ["train", SMOKE, "--out", str(plain)]).exit_code == 2


def test_retry_moves_a_wall_clock_abort_aside(learn: FakeLearn, tmp_path: Path) -> None:
    out = tmp_path / "ppo_s0"
    for k in (1, 2):
        aborted_run(out, ["high_energy", "wall_clock_abort"])
        result = runner.invoke(app, ["train", SMOKE, "--out", str(out), "--retry"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / f"ppo_s0.attempt{k}" / "ckpt").is_dir()
        assert f"ppo_s0.attempt{k}" in result.output
        shutil.rmtree(out)
    aborted_run(out, ["wall_clock_abort"])
    result = runner.invoke(app, ["train", SMOKE, "--out", str(out), "--retry"])
    assert result.exit_code == 2
    assert "already retried 2 times" in result.output
    assert len(learn.calls) == 2
    assert (out / "ckpt").is_dir()


@pytest.mark.parametrize("flags", [["rss_abort"], ["cpu_abort"], [], ["nondeterministic"]])
def test_retry_moves_nothing_else(learn: FakeLearn, tmp_path: Path, flags: list[str]) -> None:
    out = aborted_run(tmp_path / "run", flags)
    result = runner.invoke(app, ["train", SMOKE, "--out", str(out), "--retry"])
    assert result.exit_code == 2
    assert "wall_clock_abort" in result.output
    assert learn.calls == []
    assert (out / "ckpt").is_dir()
    assert not (tmp_path / "run.attempt1").exists()


@pytest.mark.parametrize("meta", [None, b"{", b'{"flags": "wall_clock_abort"}', b"[]"])
def test_retry_needs_a_readable_train_meta(
    learn: FakeLearn, tmp_path: Path, meta: bytes | None
) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "notes").write_text("x")
    if meta is not None:
        (out / "train_meta.json").write_bytes(meta)
    result = runner.invoke(app, ["train", SMOKE, "--out", str(out), "--retry"])
    assert result.exit_code == 2
    assert learn.calls == []
    assert not (tmp_path / "run.attempt1").exists()


def test_retry_without_an_out_just_trains(learn: FakeLearn, tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", SMOKE, "--out", str(tmp_path / "run"), "--retry"])
    assert result.exit_code == 0, result.output
    assert len(learn.calls) == 1


@pytest.mark.parametrize("flag", ["rss_abort", "cpu_abort", "wall_clock_abort"])
def test_a_hard_limit_exits_4(learn: FakeLearn, tmp_path: Path, flag: str) -> None:
    learn.flags = ["no_eligible", flag]
    result = runner.invoke(app, ["train", SMOKE, "--out", str(tmp_path / "run")])
    assert result.exit_code == 4


def test_a_warning_flag_exits_0(learn: FakeLearn, tmp_path: Path) -> None:
    learn.flags = ["no_eligible", "high_energy"]
    assert runner.invoke(app, ["train", SMOKE, "--out", str(tmp_path / "run")]).exit_code == 0


def test_a_bad_config_exits_2_before_training(learn: FakeLearn, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("algo: ppo\nseed: 0\n")
    result = runner.invoke(app, ["train", str(bad), "--out", str(tmp_path / "run")])
    assert result.exit_code == 2
    assert result.output.startswith("error: bad.yaml: ")
    assert learn.calls == []
    assert not (tmp_path / "run").exists()
    seed = runner.invoke(app, ["train", SMOKE, "--out", str(tmp_path / "r"), "--seed", "-1"])
    assert seed.exit_code == 2


def test_without_the_training_module_train_exits_2(tmp_path: Path) -> None:
    """This build has no ``topple.train.learn`` (the torch half lands with the train extra)."""
    import importlib.util

    if importlib.util.find_spec("topple.train.learn") is not None:
        pytest.skip("topple.train.learn exists: the train extra's tests cover it")
    result = runner.invoke(app, ["train", SMOKE, "--out", str(tmp_path / "run")])
    assert result.exit_code == 2
    assert len(result.output.strip().splitlines()) == 1
    assert "topple.train.learn" in result.output or "train extra" in result.output


class _NoTorch(importlib.abc.MetaPathFinder):
    """``topple.train.learn`` as it imports without the train extra: torch is missing."""

    def find_spec(self, name: str, path: object = None, target: object = None) -> None:
        if name == "topple.train.learn":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")


def test_without_the_train_extra_train_exits_2_with_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "topple.train.learn", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_NoTorch(), *sys.meta_path])
    result = runner.invoke(app, ["train", SMOKE, "--out", str(tmp_path / "run")])
    assert result.exit_code == 2
    assert result.output.strip().splitlines() == [
        "error: topple train needs the train extra (torch is not installed): uv sync --extra train"
    ]
    assert not (tmp_path / "run").exists()
