"""An in-process stand-in for ``topple.pool.eval_pool`` (tests of compare and shove).

A spawn pool costs seconds to start, so most tests run the very same pool tasks
(``run_episodes`` with the arguments ``submit_episodes`` checked and materialized) in this
process instead, and one test per command uses a real pool.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from typing import Any


class InlinePool(Executor):
    """Runs each submitted call at once, in this process, and records it."""

    instances: list[InlinePool] = []  # noqa: RUF012 - the pools a test created, in order

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.calls: list[tuple[Any, ...]] = []
        self.shut_down = False
        InlinePool.instances.append(self)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        self.calls.append((fn, *args))
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shut_down = True

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def guarded[T](items: Iterable[T], cap: int) -> Iterator[T]:
    """``items`` (an endless iterable, say), but at most ``cap + 1`` of them: a bounded read
    (``islice(items, cap + 1)``) takes exactly that many, and asking for one more raises
    ``AssertionError``. Unguarded, a regression that materializes the whole iterable would
    allocate until the machine falls over instead of failing fast."""
    for k, item in enumerate(items):
        if k > cap:
            raise AssertionError(f"read more than {cap + 1} items from an endless iterable")
        yield item
