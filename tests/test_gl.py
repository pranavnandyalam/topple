import pytest

from topple._gl import default_gl_backend, resolve_gl_backend


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("darwin", "cgl"), ("linux", "osmesa"), ("win32", "glfw")],
)
def test_default_backend_per_platform(platform: str, expected: str) -> None:
    assert default_gl_backend(platform) == expected


def test_resolve_respects_existing_env() -> None:
    assert resolve_gl_backend({"MUJOCO_GL": "egl"}, "linux") == "egl"


def test_resolve_ignores_blank_env() -> None:
    assert resolve_gl_backend({"MUJOCO_GL": "  "}, "darwin") == "cgl"


def test_resolve_normalizes_case() -> None:
    assert resolve_gl_backend({"MUJOCO_GL": "OSMesa"}, "linux") == "osmesa"


def test_resolve_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="MUJOCO_GL"):
        resolve_gl_backend({"MUJOCO_GL": "vulkan"}, "linux")


@pytest.mark.parametrize(
    ("backend", "platform"),
    [("osmesa", "darwin"), ("egl", "darwin"), ("cgl", "linux"), ("wgl", "linux")],
)
def test_resolve_rejects_backend_unavailable_on_platform(backend: str, platform: str) -> None:
    with pytest.raises(ValueError, match=platform):
        resolve_gl_backend({"MUJOCO_GL": backend}, platform)


@pytest.mark.parametrize("value", ["disable", "disabled", "off"])
def test_resolve_allows_disabling_rendering(value: str) -> None:
    assert resolve_gl_backend({"MUJOCO_GL": value}, "linux") == value
