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
from topple.train.config import PpoTrainConfig, config_sha256, load_train_config

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
        self.history: list[dict[str, Any]] = []  # selection_history the fake run writes

    def train(self, cfg: Any, out: Path, *, config_file_sha256: str) -> Path:
        self.calls.append((cfg, out, config_file_sha256))
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "flags": self.flags,
            "config_sha256": config_sha256(cfg),  # as learn writes it: a retry re-runs only it
            "selection_history": self.history,
        }
        (out / "train_meta.json").write_text(json.dumps(meta))
        return out


@pytest.fixture
def learn(monkeypatch: pytest.MonkeyPatch) -> FakeLearn:
    fake = FakeLearn([])
    module = ModuleType("topple.train.learn")
    module.train = fake.train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "topple.train.learn", module)
    return fake


SMOKE = str(CONFIGS / "smoke_ppo.yaml")
SMOKE_SHA = config_sha256(load_train_config(Path(SMOKE)))  # the effective config, seed 0


def aborted_run(
    path: Path,
    flags: list[str],
    config_sha: str = SMOKE_SHA,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    path.mkdir(parents=True)
    meta = {"flags": flags, "config_sha256": config_sha, "selection_history": history or []}
    (path / "train_meta.json").write_text(json.dumps(meta))
    (path / "ckpt").mkdir()
    return path


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


@pytest.mark.parametrize(
    ("args", "config_sha"),
    [
        (["--seed", "5"], SMOKE_SHA),  # the attempt ran seed 0
        ([], "0" * 64),  # the attempt ran another config
        ([], None),  # the attempt recorded none
    ],
)
def test_retry_reruns_only_the_attempts_own_config_and_seed(
    learn: FakeLearn, tmp_path: Path, args: list[str], config_sha: Any
) -> None:
    """§1: a retry starts afresh with the same seed and must reproduce the attempt's
    ``onnx_sha256``; another effective config would be an unrelated run."""
    out = aborted_run(tmp_path / "ppo_s0", ["wall_clock_abort"], config_sha)
    result = runner.invoke(app, ["train", SMOKE, "--out", str(out), "--retry", *args])
    assert result.exit_code == 2
    assert "config_sha256" in result.output
    assert learn.calls == []
    assert (out / "ckpt").is_dir()
    assert not (tmp_path / "ppo_s0.attempt1").exists()


def test_retry_with_the_same_seed_given_explicitly(learn: FakeLearn, tmp_path: Path) -> None:
    out = aborted_run(
        tmp_path / "ppo_s2",
        ["wall_clock_abort"],
        config_sha256(load_train_config(Path(SMOKE), seed=2)),
    )
    result = runner.invoke(app, ["train", SMOKE, "--out", str(out), "--retry", "--seed", "2"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ppo_s2.attempt1" / "ckpt").is_dir()


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


def rows(*pairs: tuple[int, str]) -> list[dict[str, Any]]:
    return [{"step": step, "onnx_sha256": sha} for step, sha in pairs]


@pytest.mark.parametrize(
    ("old", "new", "flagged"),
    [
        (rows((1024, "a" * 64)), rows((1024, "a" * 64), (2048, "b" * 64)), False),
        (rows((1024, "a" * 64), (1500, "c" * 64)), rows((1024, "a" * 64), (2048, "b" * 64)), False),
        (rows((1024, "a" * 64)), rows((1024, "f" * 64), (2048, "b" * 64)), True),
        ([{"step": "1024", "onnx_sha256": "a" * 64}], rows((1024, "a" * 64)), True),
    ],
    ids=["reproduces", "abort-label-not-shared", "differs", "malformed"],
)
def test_a_retry_that_does_not_reproduce_is_flagged(
    learn: FakeLearn,
    tmp_path: Path,
    old: list[dict[str, Any]],
    new: list[dict[str, Any]],
    flagged: bool,
) -> None:
    """§1: the re-run must give the aborted attempt's onnx_sha256 at every label both reached."""
    out = tmp_path / "ppo_s0"
    aborted_run(out, ["wall_clock_abort"], history=old)
    learn.history = new
    learn.flags = ["high_energy"]
    result = runner.invoke(app, ["train", SMOKE, "--out", str(out), "--retry"])
    assert result.exit_code == 0, result.output
    flags = json.loads((out / "train_meta.json").read_text())["flags"]
    expected = ["high_energy", "nondeterministic"] if flagged else ["high_energy"]
    assert flags == expected
    assert ("nondeterministic" in result.output) is flagged


A, B, C, F = ("a" * 64, "b" * 64, "c" * 64, "f" * 64)


def retry(out: Path) -> Any:
    return runner.invoke(app, ["train", SMOKE, "--out", str(out), "--retry"])


def test_a_retry_chain_compares_with_every_earlier_attempt(
    learn: FakeLearn, tmp_path: Path
) -> None:
    """§1: attempt 1 gave b at 2048; its retry (attempt 2) gave f there and was flagged. A
    second retry that reproduces attempt 2 still diverges from attempt 1, and inherits attempt
    2's flag: comparing with the latest attempt alone would ship a nondeterministic run."""
    out = tmp_path / "ppo_s0"
    aborted_run(out, ["wall_clock_abort"], history=rows((1024, A), (2048, B)))
    learn.history, learn.flags = rows((1024, A), (2048, F)), ["wall_clock_abort"]
    result = retry(out)
    assert result.exit_code == 4, result.output  # it stopped on its wall-clock limit again
    meta = json.loads((out / "train_meta.json").read_text())
    assert meta["flags"] == ["wall_clock_abort", "nondeterministic"]
    learn.history, learn.flags = rows((1024, A), (2048, F), (3072, C)), ["high_energy"]
    result = retry(out)
    assert result.exit_code == 0, result.output
    assert len(learn.calls) == 2
    assert json.loads((tmp_path / "ppo_s0.attempt2" / "train_meta.json").read_text()) == meta
    flags = json.loads((out / "train_meta.json").read_text())["flags"]
    assert flags == ["high_energy", "nondeterministic"]
    assert "ppo_s0.attempt1, ppo_s0.attempt2: flagged nondeterministic" in result.output


def test_a_retry_chain_that_reproduces_every_attempt_is_not_flagged(
    learn: FakeLearn, tmp_path: Path
) -> None:
    out = tmp_path / "ppo_s0"
    aborted_run(out, ["wall_clock_abort"], history=rows((1024, A)))
    learn.history, learn.flags = rows((1024, A), (2048, B)), ["wall_clock_abort"]
    assert retry(out).exit_code == 4
    learn.history, learn.flags = rows((1024, A), (2048, B), (3072, C)), ["high_energy"]
    result = retry(out)
    assert result.exit_code == 0, result.output
    assert json.loads((out / "train_meta.json").read_text())["flags"] == ["high_energy"]
    assert "nondeterministic" not in result.output


@pytest.mark.parametrize(
    ("first", "second", "second_flags", "diverged"),
    [
        # reproduces attempt 2 (clean), not attempt 1: only attempt 1 is named
        (rows((1024, A), (2048, B)), rows((1024, A), (2048, F)), [], ["ppo_s0.attempt1"]),
        # reproduces both, but attempt 2 was flagged: the flag is inherited
        (rows((1024, A)), rows((1024, A), (2048, F)), ["nondeterministic"], ["ppo_s0.attempt2"]),
        # reproduces both clean attempts
        (rows((1024, A)), rows((1024, A), (2048, F)), [], []),
    ],
    ids=["diverges-from-the-first", "inherits-the-flag", "reproduces-both"],
)
def test_each_earlier_attempt_counts_on_its_own(
    learn: FakeLearn,
    tmp_path: Path,
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    second_flags: list[str],
    diverged: list[str],
) -> None:
    """Attempt 1 is already aside; the aborted run in ``--out`` becomes attempt 2."""
    out = tmp_path / "ppo_s0"
    aborted_run(tmp_path / "ppo_s0.attempt1", ["wall_clock_abort"], history=first)
    aborted_run(out, ["wall_clock_abort", *second_flags], history=second)
    learn.history, learn.flags = rows((1024, A), (2048, F), (3072, C)), []
    result = retry(out)
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ppo_s0.attempt2" / "ckpt").is_dir()
    flags = json.loads((out / "train_meta.json").read_text())["flags"]
    assert flags == (["nondeterministic"] if diverged else [])
    if diverged:
        assert f"does not reproduce {', '.join(diverged)}: flagged" in result.output
    else:
        assert "nondeterministic" not in result.output
