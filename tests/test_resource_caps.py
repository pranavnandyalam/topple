"""Resource caps (docs/design/phase-2.md §1, §6 ``test_resource_caps``).

Slice 1c covers the "pools clamp to 4" line: every evaluation pool, compare's and shove's
included, is ``min(workers, 4, os.cpu_count())`` spawn processes, whatever is asked. The other
lines (shipped configs ``n_envs <= 8``, complete ``limits``, ``topple train`` refusing a
non-empty ``--out``, ``--retry``) belong to the training slice.
"""

from __future__ import annotations

import multiprocessing
import shutil
from pathlib import Path
from typing import Any

import pytest

import topple.pool as pool_module
from pool_support import InlinePool
from topple.bench import shove_curve
from topple.compare import compare
from topple.evaluate import EvalProtocol
from topple.pool import POOL_MAX_WORKERS, eval_pool, pool_size

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
