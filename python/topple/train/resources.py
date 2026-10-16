"""Resource sampler and hard limits (docs/design/phase-2.md §1 "Resource sampler", "Hard limits").

``ResourceSampler`` runs a daemon thread in the learner that samples, every ``interval_s``, the
root process (the learner by default) and ``children(recursive=True)`` (forkserver workers
included: they descend from the learner's forkserver child). Each sample:

- sums ``memory_info().rss`` over the tree → ``tree_rss_mib``, and keeps the largest such sum as
  ``peak_tree_rss_mib`` with the env step (``set_step``) and process count it was seen at;
- writes ``user + system`` of every process into a **ledger** keyed by ``(pid, create_time)``
  (last seen value, so a process that exits keeps counting and a reused pid is a new entry).
  ``cpu_s = sum(ledger.values())``; it is monotone and uses no rusage (``children_user`` and
  ``children_system`` are ignored: they would count reaped children twice).

A process that vanishes mid-sample (``NoSuchProcess``, ``AccessDenied``, ``ZombieProcess``) is
skipped. No sampling error ever reaches the caller: it is logged and the thread carries on.

``abort_flag`` maps a reading to the first hard limit it exceeds (strictly), in the order RSS,
CPU, wall clock. Needs ``psutil`` (the ``train`` extra).
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Self

import psutil

from topple.train.config import LimitsConfig

__all__ = [
    "CPU_ABORT",
    "RSS_ABORT",
    "WALL_CLOCK_ABORT",
    "ResourceReading",
    "ResourceSampler",
    "abort_flag",
]

log = logging.getLogger(__name__)

RSS_ABORT: Final = "rss_abort"
CPU_ABORT: Final = "cpu_abort"
WALL_CLOCK_ABORT: Final = "wall_clock_abort"
_MIB: Final = float(1 << 20)
_GONE: Final = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)
_THREAD_NAME: Final = "topple-resources"


@dataclass(frozen=True)
class ResourceReading:
    cpu_s: float
    """Sum of the ledger: user + system of every process ever seen in the tree (dead ones keep
    their last value)."""
    tree_rss_mib: float
    """Summed RSS of the latest sample."""
    peak_tree_rss_mib: float
    peak_tree_rss_at_step: int
    peak_procs: int


@dataclass(frozen=True)
class _Proc:
    key: tuple[int, float]
    cpu_s: float
    rss_bytes: int


def _read(proc: psutil.Process) -> _Proc | None:
    """One process's ledger key, CPU and RSS, or ``None`` if it vanished meanwhile."""
    try:
        with proc.oneshot():
            key = (int(proc.pid), float(proc.create_time()))
            times = proc.cpu_times()
            rss = int(proc.memory_info().rss)
    except _GONE:
        return None
    return _Proc(key=key, cpu_s=float(times.user) + float(times.system), rss_bytes=rss)


def _tree(root_pid: int) -> list[_Proc]:
    """The root and its descendants, as one sample; empty if the root is gone."""
    try:
        root = psutil.Process(root_pid)
        procs = [root, *root.children(recursive=True)]
    except _GONE:
        return []
    return [p for p in map(_read, procs) if p is not None]


class ResourceSampler:
    """Samples a process tree on a daemon thread; see the module docstring.

    ``reading`` and ``stop`` are thread-safe and take one synchronous sample first, so the final
    reading includes everything up to the call. After ``stop`` the reading is frozen: further
    ``stop`` and ``reading`` calls return it unchanged. Usable as a context manager (start on
    enter, stop on exit). A sampler starts at most once.
    """

    def __init__(self, *, interval_s: float = 1.0, root_pid: int | None = None) -> None:
        if not (math.isfinite(interval_s) and interval_s > 0.0):
            raise ValueError(f"interval_s must be finite and > 0, got {interval_s!r}")
        if root_pid is not None and root_pid <= 0:
            raise ValueError(f"root_pid must be > 0, got {root_pid!r}")
        self._interval_s = interval_s
        self._root_pid = os.getpid() if root_pid is None else root_pid
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._final: ResourceReading | None = None
        self._step = 0
        self._ledger: dict[tuple[int, float], float] = {}
        self._cpu_s = 0.0
        self._tree_rss_mib = 0.0
        self._peak_rss_mib = 0.0
        self._peak_step = 0
        self._peak_procs = 0

    # --- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        """Takes a first sample and starts the sampling thread."""
        with self._lock:
            if self._started:
                raise RuntimeError("ResourceSampler already started")
            self._started = True
            self._sample_locked()
        self._thread = threading.Thread(target=self._run, name=_THREAD_NAME, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceReading:
        """Takes the final sample, stops and joins the thread; idempotent."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self._started = True  # a stopped sampler never starts
            if self._final is None:
                self._sample_locked()
                self._final = self._reading_locked()
            return self._final

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # --- use ---------------------------------------------------------------------------------

    def set_step(self, step: int) -> None:
        """The current env step, recorded with a new RSS peak."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step!r}")
        with self._lock:
            self._step = step

    def reading(self) -> ResourceReading:
        """Takes one synchronous sample (unless stopped) and returns the totals."""
        with self._lock:
            if self._final is not None:
                return self._final
            self._sample_locked()
            return self._reading_locked()

    # --- internals ---------------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            with self._lock:
                self._sample_locked()

    def _sample_locked(self) -> None:
        """One sample into the ledger and peaks; never raises (the caller trains)."""
        try:
            procs = _tree(self._root_pid)
        except Exception:
            log.warning("resource sample of pid %d failed", self._root_pid, exc_info=True)
            return
        for proc in procs:
            last = self._ledger.get(proc.key, 0.0)
            self._ledger[proc.key] = max(last, proc.cpu_s)
        self._cpu_s = max(self._cpu_s, sum(self._ledger.values()))
        self._tree_rss_mib = sum(p.rss_bytes for p in procs) / _MIB
        if self._tree_rss_mib > self._peak_rss_mib:
            self._peak_rss_mib = self._tree_rss_mib
            self._peak_step = self._step
        self._peak_procs = max(self._peak_procs, len(procs))

    def _reading_locked(self) -> ResourceReading:
        return ResourceReading(
            cpu_s=self._cpu_s,
            tree_rss_mib=self._tree_rss_mib,
            peak_tree_rss_mib=self._peak_rss_mib,
            peak_tree_rss_at_step=self._peak_step,
            peak_procs=self._peak_procs,
        )


def abort_flag(reading: ResourceReading, limits: LimitsConfig, elapsed_s: float) -> str | None:
    """The first hard limit ``reading`` exceeds (strictly): RSS, then CPU, then wall clock."""
    if reading.tree_rss_mib > limits.max_tree_rss_mib:
        return RSS_ABORT
    if reading.cpu_s > limits.max_cpu_s:
        return CPU_ABORT
    if elapsed_s > limits.max_wall_clock_s:
        return WALL_CLOCK_ABORT
    return None
