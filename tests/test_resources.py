"""Resource sampler and hard limits (docs/design/phase-2.md §1 "Resource sampler", "Hard limits").

``ResourceSampler`` samples a process tree (the learner and ``children(recursive=True)``) on a
daemon thread. ``cpu_s`` sums a ledger keyed by ``(pid, create_time)`` holding each process's
last seen ``user + system``, so a worker that exits keeps counting; ``peak_tree_rss_mib`` is the
largest summed RSS, with the env step it was seen at. ``abort_flag`` maps a reading to the first
limit it exceeds. Real trees here are small (one ~100 MiB child at most); the error paths use a
stand-in for ``psutil.Process``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import threading
import time
from collections import namedtuple
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import psutil
import pytest

from topple.train.config import LimitsConfig
from topple.train.resources import ResourceReading, ResourceSampler, abort_flag
from topple.train.selection import ABORT_FLAGS

pytestmark = pytest.mark.train

MIB = 1 << 20
FAST = 0.02  # sampling interval for the real-tree tests, seconds
BURN_S = 0.5  # CPU seconds the burning child spends
ALLOC_MIB = 100


# --- real process trees --------------------------------------------------------------------------

# A tree root: spawns `n` copies of CHILD with `mode`, prints "up" once they are running, waits for
# a line, closes their stdin (they exit), reaps them, prints "down" and waits for EOF.
ROOT = """
import subprocess, sys
child, mode, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
kids = [subprocess.Popen([sys.executable, "-c", child, mode], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True) for _ in range(n)]
for k in kids:
    assert k.stdout.readline().strip() == "ready"
print("up", flush=True)
sys.stdin.readline()
for k in kids:
    k.stdin.close()
    k.wait()
print("down", flush=True)
sys.stdin.read()
"""
CHILD = """
import os, sys, time
mode = sys.argv[1]
if mode == "burn":
    end = time.process_time() + float(os.environ["BURN_S"])
    while time.process_time() < end:
        pass
    blob = b""
elif mode == "alloc":
    blob = os.urandom(int(os.environ["ALLOC_MIB"]) << 20)  # incompressible, so it stays resident
else:
    blob = b""
print("ready", flush=True)
sys.stdin.read()
"""


@dataclass
class Tree:
    proc: subprocess.Popen[str]

    @property
    def pid(self) -> int:
        return self.proc.pid

    def expect(self, line: str) -> None:
        assert self.proc.stdout is not None
        assert self.proc.stdout.readline().strip() == line

    def release(self) -> None:
        """Lets the children exit and waits until the root has reaped them."""
        assert self.proc.stdin is not None
        self.proc.stdin.write("go\n")
        self.proc.stdin.flush()
        self.expect("down")


@contextlib.contextmanager
def tree(mode: str, n: int = 1) -> Iterator[Tree]:
    env = {**os.environ, "BURN_S": str(BURN_S), "ALLOC_MIB": str(ALLOC_MIB)}
    proc = subprocess.Popen(
        [sys.executable, "-c", ROOT, CHILD, mode, str(n)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        yield Tree(proc)
    finally:
        with contextlib.suppress(OSError):
            assert proc.stdin is not None
            proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            for kid in psutil.Process(proc.pid).children(recursive=True):
                kid.kill()
            proc.kill()
            proc.wait()


def test_a_child_that_burned_cpu_and_exited_still_counts() -> None:
    with tree("burn") as t, ResourceSampler(interval_s=FAST, root_pid=t.pid) as sampler:
        t.expect("up")  # the child burned BURN_S of CPU and is idle; the thread has seen it
        time.sleep(5 * FAST)
        t.release()  # the child has exited and been reaped
        after = sampler.reading()
    # the root alone (an idle interpreter) uses far less than BURN_S
    assert after.cpu_s >= 0.8 * BURN_S


def test_cpu_s_counts_a_dead_child_seen_only_by_a_synchronous_reading() -> None:
    with tree("burn") as t:
        sampler = ResourceSampler(interval_s=3600.0, root_pid=t.pid)  # the thread never fires
        sampler.start()
        t.expect("up")
        seen = sampler.reading()
        t.release()
        final = sampler.stop()
    assert seen.cpu_s >= 0.8 * BURN_S
    assert final.cpu_s >= seen.cpu_s  # monotone: the dead child keeps its last value
    assert final.peak_procs == 2


def test_an_allocating_child_raises_the_peak_rss_at_its_step() -> None:
    with tree("idle") as t:
        t.expect("up")
        base = ResourceSampler(interval_s=3600.0, root_pid=t.pid).stop()
        t.release()
    with tree("alloc") as t, ResourceSampler(interval_s=FAST, root_pid=t.pid) as sampler:
        sampler.set_step(7)
        t.expect("up")
        high = sampler.reading()
        sampler.set_step(99)
        t.release()
        low = sampler.reading()
    assert high.tree_rss_mib >= base.tree_rss_mib + 0.8 * ALLOC_MIB
    assert high.tree_rss_mib < base.tree_rss_mib + 3 * ALLOC_MIB  # bounded
    assert high.peak_tree_rss_mib >= high.tree_rss_mib
    assert low.tree_rss_mib < high.tree_rss_mib - 0.5 * ALLOC_MIB  # the child is gone
    assert low.peak_tree_rss_mib == high.peak_tree_rss_mib
    assert low.peak_tree_rss_at_step == 7


def test_peak_procs_counts_the_whole_tree() -> None:
    with tree("idle", n=3) as t, ResourceSampler(interval_s=FAST, root_pid=t.pid) as sampler:
        t.expect("up")
        during = sampler.reading()
        t.release()
        after = sampler.reading()
    assert during.peak_procs == 4  # the root and its three children
    assert after.peak_procs == 4


def test_the_default_root_is_this_process() -> None:
    with ResourceSampler(interval_s=3600.0) as sampler:
        reading = sampler.reading()
    me = psutil.Process()
    assert reading.peak_procs >= 1
    assert reading.cpu_s >= sum(me.cpu_times()[:2]) * 0.5
    assert reading.tree_rss_mib >= me.memory_info().rss / MIB * 0.5


# --- lifecycle -----------------------------------------------------------------------------------


def samplers() -> int:
    return sum(t.name == "topple-resources" for t in threading.enumerate())


def test_stop_is_idempotent_and_freezes_the_reading() -> None:
    before = samplers()
    sampler = ResourceSampler(interval_s=FAST)
    sampler.start()
    assert samplers() == before + 1
    first = sampler.stop()
    assert isinstance(first, ResourceReading)
    assert sampler.stop() == first
    assert sampler.reading() == first
    assert samplers() == before


def test_stop_without_start_takes_one_reading() -> None:
    reading = ResourceSampler().stop()
    assert reading.peak_procs >= 1
    assert reading.cpu_s > 0.0


def test_start_twice_is_an_error() -> None:
    sampler = ResourceSampler(interval_s=3600.0)
    sampler.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            sampler.start()
    finally:
        sampler.stop()
    with pytest.raises(RuntimeError, match="already started"):
        sampler.start()


def test_the_context_manager_starts_and_stops() -> None:
    before = samplers()
    with ResourceSampler(interval_s=FAST) as sampler:
        assert samplers() == before + 1
        assert all(t.daemon for t in threading.enumerate() if t.name == "topple-resources")
    assert samplers() == before
    final = sampler.stop()
    assert sampler.reading() == final


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
def test_a_bad_interval_is_refused(interval: float) -> None:
    with pytest.raises(ValueError, match="interval_s"):
        ResourceSampler(interval_s=interval)


def test_a_bad_root_pid_is_refused() -> None:
    with pytest.raises(ValueError, match="root_pid"):
        ResourceSampler(root_pid=0)


def test_a_negative_step_is_refused() -> None:
    with pytest.raises(ValueError, match="step"):
        ResourceSampler().set_step(-1)


# --- the ledger and vanishing processes (a stand-in for psutil.Process) ---------------------------

CpuTimes = namedtuple("CpuTimes", "user system children_user children_system")
MemInfo = namedtuple("MemInfo", "rss vms")


@dataclass
class FakeProc:
    pid: int
    create: float = 1.0
    user: float = 0.0
    system: float = 0.0
    rss_mib: float = 0.0
    error: Callable[[int], Exception] | None = None
    kids: list[FakeProc] = field(default_factory=list)

    def _check(self) -> None:
        if self.error is not None:
            raise self.error(self.pid)

    def oneshot(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def create_time(self) -> float:
        self._check()
        return self.create

    def cpu_times(self) -> CpuTimes:
        self._check()
        return CpuTimes(self.user, self.system, 100.0, 100.0)  # children_* must be ignored

    def memory_info(self) -> MemInfo:
        self._check()
        return MemInfo(int(self.rss_mib * MIB), 0)

    def children(self, recursive: bool = False) -> list[FakeProc]:
        assert recursive
        return list(self.kids)


@dataclass
class World:
    root: FakeProc | None
    calls: int = 0
    fail: Exception | None = None

    def process(self, pid: int) -> FakeProc:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        if self.root is None or pid != self.root.pid:
            raise psutil.NoSuchProcess(pid)
        return self.root


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    w = World(root=FakeProc(pid=100, user=1.0, system=0.5, rss_mib=10.0))
    monkeypatch.setattr(psutil, "Process", w.process)
    return w


def sampler_for(world: World) -> ResourceSampler:
    assert world.root is not None
    return ResourceSampler(interval_s=3600.0, root_pid=world.root.pid)


def test_the_ledger_sums_user_and_system_only(world: World) -> None:
    assert world.root is not None
    world.root.kids = [FakeProc(pid=101, user=2.0, system=1.0, rss_mib=5.0)]
    reading = sampler_for(world).reading()
    assert reading.cpu_s == pytest.approx(4.5)
    assert reading.tree_rss_mib == pytest.approx(15.0)
    assert reading.peak_procs == 2


def test_dead_processes_keep_their_last_value(world: World) -> None:
    assert world.root is not None
    kid = FakeProc(pid=101, user=2.0)
    world.root.kids = [kid]
    sampler = sampler_for(world)
    sampler.reading()
    kid.user = 3.0
    sampler.reading()
    world.root.kids = []
    reading = sampler.reading()
    assert reading.cpu_s == pytest.approx(1.5 + 3.0)


def test_a_reused_pid_is_a_new_ledger_entry(world: World) -> None:
    assert world.root is not None
    world.root.kids = [FakeProc(pid=101, create=1.0, user=2.0)]
    sampler = sampler_for(world)
    sampler.reading()
    world.root.kids = [FakeProc(pid=101, create=2.0, user=0.25)]
    assert sampler.reading().cpu_s == pytest.approx(1.5 + 2.0 + 0.25)


def test_cpu_s_never_decreases(world: World) -> None:
    assert world.root is not None
    sampler = sampler_for(world)
    first = sampler.reading().cpu_s
    world.root.user = 0.0  # a process's counter going backwards must not lower cpu_s
    assert sampler.reading().cpu_s == pytest.approx(first)


@pytest.mark.parametrize("error", [psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess])
def test_a_process_that_vanishes_mid_sample_is_skipped(
    world: World, error: Callable[[int], Exception]
) -> None:
    assert world.root is not None
    world.root.kids = [
        FakeProc(pid=101, user=2.0, rss_mib=5.0, error=error),
        FakeProc(pid=102, user=1.0, rss_mib=3.0),
    ]
    reading = sampler_for(world).reading()
    assert reading.cpu_s == pytest.approx(1.5 + 1.0)
    assert reading.tree_rss_mib == pytest.approx(13.0)
    assert reading.peak_procs == 2


def test_a_vanished_root_gives_an_empty_sample(world: World) -> None:
    sampler = sampler_for(world)
    first = sampler.reading()
    world.root = None
    reading = sampler.reading()
    assert reading.tree_rss_mib == 0.0
    assert reading.cpu_s == pytest.approx(first.cpu_s)
    assert reading.peak_tree_rss_mib == pytest.approx(10.0)


def test_the_peak_keeps_its_step(world: World) -> None:
    assert world.root is not None
    sampler = sampler_for(world)
    sampler.set_step(5)
    sampler.reading()
    sampler.set_step(6)
    world.root.rss_mib = 30.0
    sampler.reading()
    sampler.set_step(7)
    world.root.rss_mib = 30.0  # a tie is not a new peak
    sampler.reading()
    sampler.set_step(8)
    world.root.rss_mib = 20.0
    reading = sampler.reading()
    assert reading.peak_tree_rss_mib == pytest.approx(30.0)
    assert reading.peak_tree_rss_at_step == 6
    assert reading.tree_rss_mib == pytest.approx(20.0)


def test_sampling_errors_never_reach_the_caller(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    world.fail = RuntimeError("boom")
    sampler = ResourceSampler(interval_s=0.005, root_pid=100)
    with caplog.at_level(logging.WARNING, logger="topple.train.resources"):
        sampler.start()
        deadline = time.monotonic() + 5.0
        while world.calls < 5 and time.monotonic() < deadline:
            time.sleep(0.005)
        reading = sampler.reading()  # the training thread's call does not raise either
        world.fail = None
        calls = world.calls
        deadline = time.monotonic() + 5.0
        while world.calls < calls + 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        healthy = sampler.stop()
    assert world.calls >= calls + 3  # the thread kept sampling after the errors
    assert reading.cpu_s == 0.0
    assert healthy.cpu_s == pytest.approx(1.5)
    assert "boom" in caplog.text


# --- hard limits ---------------------------------------------------------------------------------

LIMITS = LimitsConfig(max_wall_clock_s=100, max_cpu_s=200, max_tree_rss_mib=300)


def reading(*, cpu_s: float = 0.0, rss: float = 0.0) -> ResourceReading:
    return ResourceReading(
        cpu_s=cpu_s,
        tree_rss_mib=rss,
        peak_tree_rss_mib=rss,
        peak_tree_rss_at_step=0,
        peak_procs=1,
    )


def test_the_flags_are_the_selection_abort_flags() -> None:
    flags = {
        abort_flag(reading(rss=301.0), LIMITS, 0.0),
        abort_flag(reading(cpu_s=201.0), LIMITS, 0.0),
        abort_flag(reading(), LIMITS, 101.0),
    }
    assert flags == set(ABORT_FLAGS)


@pytest.mark.parametrize(
    ("cpu_s", "rss", "elapsed", "flag"),
    [
        (0.0, 0.0, 0.0, None),
        (200.0, 300.0, 100.0, None),  # every limit reached exactly: none exceeded
        (0.0, 300.001, 0.0, "rss_abort"),
        (200.001, 0.0, 0.0, "cpu_abort"),
        (0.0, 0.0, 100.001, "wall_clock_abort"),
        (1e9, 1e9, 1e9, "rss_abort"),  # rss first
        (1e9, 0.0, 1e9, "cpu_abort"),  # then cpu
    ],
)
def test_abort_flag_order_and_boundaries(
    cpu_s: float, rss: float, elapsed: float, flag: str | None
) -> None:
    assert abort_flag(reading(cpu_s=cpu_s, rss=rss), LIMITS, elapsed) == flag


def test_abort_flag_reads_the_latest_rss_not_the_peak() -> None:
    r = ResourceReading(
        cpu_s=0.0, tree_rss_mib=10.0, peak_tree_rss_mib=1e6, peak_tree_rss_at_step=3, peak_procs=2
    )
    assert abort_flag(r, LIMITS, 0.0) is None


# A wrapper whose only child is ROOT, so ROOT's workers are the sampler root's grandchildren, as
# the forkserver env workers are the learner's (they descend from its forkserver child).
WRAPPER = """
import subprocess, sys
p = subprocess.Popen([sys.executable, "-c", *sys.argv[1:]], stdin=sys.stdin, stdout=sys.stdout)
sys.exit(p.wait())
"""


@pytest.mark.parametrize("mode", ["burn", "alloc"])
def test_grandchildren_are_counted(mode: str) -> None:
    """``children(recursive=True)``: a non-recursive walk would miss every env worker."""
    env = {**os.environ, "BURN_S": str(BURN_S), "ALLOC_MIB": str(ALLOC_MIB)}
    wrapper = subprocess.Popen(
        [sys.executable, "-c", WRAPPER, ROOT, CHILD, mode, "1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert wrapper.stdin is not None
    assert wrapper.stdout is not None
    try:
        with ResourceSampler(interval_s=FAST, root_pid=wrapper.pid) as sampler:
            assert wrapper.stdout.readline().strip() == "up"
            reading = sampler.reading()
            assert reading.peak_procs >= 3  # wrapper, ROOT, and ROOT's worker
            if mode == "burn":
                assert reading.cpu_s >= 0.9 * BURN_S
            else:
                assert reading.tree_rss_mib >= 0.9 * ALLOC_MIB
            wrapper.stdin.write("\n")
            wrapper.stdin.flush()
            assert wrapper.stdout.readline().strip() == "down"
            after = sampler.reading()
            if mode == "burn":  # the exited grandchild keeps counting
                assert after.cpu_s >= reading.cpu_s
    finally:
        with contextlib.suppress(OSError):
            wrapper.stdin.close()
        wrapper.wait(timeout=30)
