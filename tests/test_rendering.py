from typing import Any

import mujoco
import pytest

from topple.rendering import RenderingUnavailableError, make_renderer
from topple.smoke import SMOKE_XML


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(SMOKE_XML)


@pytest.mark.parametrize("value", ["off", "disable", "0"])
def test_disabled_rendering_raises(
    model: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("MUJOCO_GL", value)
    with pytest.raises(RenderingUnavailableError, match=f"MUJOCO_GL={value}"):
        make_renderer(model, height=8, width=8)


class NullFunctionError(Exception):
    """Stand-in for OpenGL.error.NullFunctionError (not a RuntimeError/OSError subclass)."""


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("no context"),
        OSError("libOSMesa missing"),
        NullFunctionError("glGenFramebuffers"),
    ],
)
def test_context_failures_are_wrapped(
    model: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def broken_renderer(*_args: Any, **_kwargs: Any) -> None:
        raise error

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(mujoco, "Renderer", broken_renderer, raising=False)
    with pytest.raises(RenderingUnavailableError, match="MUJOCO_GL=osmesa") as excinfo:
        make_renderer(model, height=8, width=8)
    assert excinfo.value.__cause__ is error


@pytest.mark.render
def test_make_renderer_renders(model: mujoco.MjModel) -> None:
    renderer = make_renderer(model, height=12, width=16)
    try:
        renderer.update_scene(mujoco.MjData(model))
        assert renderer.render().shape == (12, 16, 3)
    finally:
        renderer.close()
