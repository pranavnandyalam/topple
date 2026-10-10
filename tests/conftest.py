import os

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
