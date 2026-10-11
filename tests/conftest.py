import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from topple._gl import rendering_disabled

_GL_VARS = frozenset({"MUJOCO_GL", "DISPLAY", "WAYLAND_DISPLAY"})


@pytest.fixture
def headless_env() -> dict[str, str]:
    """``os.environ`` minus GL/display variables: what a fresh headless CI process sees."""
    return {k: v for k, v in os.environ.items() if k not in _GL_VARS}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `render` tests when the user disabled rendering via MUJOCO_GL."""
    if not rendering_disabled(os.environ):
        return
    skip = pytest.mark.skip(reason=f"rendering disabled (MUJOCO_GL={os.environ['MUJOCO_GL']})")
    for item in items:
        if "render" in item.keywords:
            item.add_marker(skip)


def _keep_opening_for_write(path: Path, stop: threading.Event) -> None:
    """Open ``path`` for writing and close it again, over and over, until ``stop`` is set.

    A reader blocked opening the FIFO is released (and then reads EOF), so a loader that wrongly
    opens a FIFO fails the test instead of hanging the suite.
    """
    while not stop.wait(0.02):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError:  # ENXIO: nobody is reading right now
            continue
        os.close(fd)


@pytest.fixture
def make_fifo() -> Iterator[Callable[[Path], Path]]:
    """Create named pipes that can never hang a test (see ``_keep_opening_for_write``)."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes need a POSIX system")
    stop = threading.Event()
    threads: list[threading.Thread] = []

    def make(path: Path) -> Path:
        os.mkfifo(path)
        thread = threading.Thread(target=_keep_opening_for_write, args=(path, stop), daemon=True)
        thread.start()
        threads.append(thread)
        return path

    yield make
    stop.set()
    for thread in threads:
        thread.join(timeout=5)
