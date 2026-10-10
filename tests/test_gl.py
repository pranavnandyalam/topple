import os
import subprocess
import sys

import pytest

from topple._gl import configure, default_gl_backend, rendering_disabled, resolve_gl_backend

# Values MuJoCo 3.15 itself accepts for MUJOCO_GL (mujoco/rendering/classic/gl_context.py).
MUJOCO_ACCEPTED = {
    "darwin": ["cgl", "glfw", "enable", "enabled", "on", "true", "1"],
    "linux": ["glx", "egl", "osmesa", "glfw", "enable", "enabled", "on", "true", "1"],
    "win32": ["wgl", "glfw", "enable", "enabled", "on", "true", "1"],
}
DISABLED = ["disable", "disabled", "off", "false", "0"]


def test_default_backend_macos_is_cgl() -> None:
    assert default_gl_backend("darwin", {}) == "cgl"


def test_default_backend_headless_linux_is_osmesa() -> None:
    assert default_gl_backend("linux", {}) == "osmesa"


@pytest.mark.parametrize("display_var", ["DISPLAY", "WAYLAND_DISPLAY"])
def test_default_backend_linux_desktop_is_glfw(display_var: str) -> None:
    assert default_gl_backend("linux", {display_var: ":0"}) == "glfw"


def test_default_backend_other_platforms_is_glfw() -> None:
    assert default_gl_backend("win32", {}) == "glfw"


def test_resolve_respects_existing_env() -> None:
    assert resolve_gl_backend({"MUJOCO_GL": "egl"}, "linux") == "egl"


def test_resolve_ignores_blank_env() -> None:
    assert resolve_gl_backend({"MUJOCO_GL": "  "}, "darwin") == "cgl"


def test_resolve_normalizes_case() -> None:
    assert resolve_gl_backend({"MUJOCO_GL": "OSMesa"}, "linux") == "osmesa"


@pytest.mark.parametrize(
    ("platform", "value"),
    [(p, v) for p, values in MUJOCO_ACCEPTED.items() for v in values],
)
def test_resolve_accepts_everything_mujoco_accepts(platform: str, value: str) -> None:
    assert resolve_gl_backend({"MUJOCO_GL": value}, platform) == value


@pytest.mark.parametrize("value", DISABLED)
@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_resolve_allows_disabling_rendering(platform: str, value: str) -> None:
    assert resolve_gl_backend({"MUJOCO_GL": value}, platform) == value


def test_resolve_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="MUJOCO_GL"):
        resolve_gl_backend({"MUJOCO_GL": "vulkan"}, "linux")


@pytest.mark.parametrize(
    ("backend", "platform"),
    [("osmesa", "darwin"), ("egl", "darwin"), ("cgl", "linux"), ("wgl", "linux"), ("glx", "win32")],
)
def test_resolve_rejects_backend_unavailable_on_platform(backend: str, platform: str) -> None:
    with pytest.raises(ValueError, match=platform):
        resolve_gl_backend({"MUJOCO_GL": backend}, platform)


@pytest.mark.parametrize("value", [*DISABLED, " OFF "])
def test_rendering_disabled_true(value: str) -> None:
    assert rendering_disabled({"MUJOCO_GL": value})


@pytest.mark.parametrize("env", [{}, {"MUJOCO_GL": "cgl"}, {"MUJOCO_GL": "egl"}])
def test_rendering_disabled_false(env: dict[str, str]) -> None:
    assert not rendering_disabled(env)


def test_configure_sets_platform_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.warns(RuntimeWarning, match="imported before topple"):
        backend = configure()  # mujoco is already imported in this test process
    assert backend == default_gl_backend(sys.platform, {})
    assert os.environ["MUJOCO_GL"] == backend


def test_configure_keeps_matching_env_without_warning(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    current = os.environ["MUJOCO_GL"]
    monkeypatch.setenv("MUJOCO_GL", current)
    assert configure() == current
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_import_topple_sets_default_before_mujoco(headless_env: dict[str, str]) -> None:
    """A fresh interpreter: `import topple` must set MUJOCO_GL without importing mujoco."""
    code = "import os, sys, topple; print(os.environ['MUJOCO_GL']); print('mujoco' in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=headless_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    backend, mujoco_loaded = proc.stdout.split()
    assert backend == default_gl_backend(sys.platform, {})
    assert mujoco_loaded == "False"
