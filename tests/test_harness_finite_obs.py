"""Non-finite observations never reach a controller (docs/design/phase-2.md §3.5, §6
``test_harness_finite_obs``; contract §8, §17).

They are outside the policy domain (runtimes disagree on NaN inputs), and by construction the
env never emits one: ``sim_unstable`` rolls the bad substep back and a non-finite reset raises.
So one reaching the control loop is a harness bug: ``evaluate`` raises ``RuntimeError`` before
``act()`` is called, a real check rather than an ``assert``, so it holds under ``python -O``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

import topple.evaluate as evaluate_module
from topple.contract import FEATURES, BalancerObs
from topple.envs.balance import BalanceEnv
from topple.evaluate import EvalProtocol, evaluate
from topple.results import ControllerInfo

SPEC = BalanceEnv("ideal").task_spec
INFO = ControllerInfo(id="probe", sha256="0" * 64)


class FakeEnv:
    """Just enough of ``BalanceEnv`` for ``evaluate``: its observations come from ``rows``."""

    rows: ClassVar[list[np.ndarray[Any, Any]]] = []

    def __init__(self, preset: str, challenge: object, record: bool) -> None:
        self.task_spec = SPEC
        self.challenge = challenge
        self.termination_reason: str | None = None
        self._k = 0

    def reset(self, seed: int) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        self._k = 0
        return self.rows[0], {"step": 0, "t": 0.0}

    def step(self, action: object) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._k += 1
        info = {"step": self._k, "t": self._k * 0.01, "termination_reason": None}
        return self.rows[self._k], 0.0, False, False, info

    def close(self) -> None:
        pass


class Recorder:
    def __init__(self) -> None:
        self.observations: list[BalancerObs] = []

    def reset(self, rng_seed: int) -> None:
        pass

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        self.observations.append(obs)
        return (0.0, 0.0)


def finite() -> np.ndarray[Any, Any]:
    return np.zeros(len(FEATURES), dtype=np.float32)


def with_value(value: float, at: int = 0) -> np.ndarray[Any, Any]:
    row = finite()
    row[at] = value
    return row


def one_seed() -> EvalProtocol:
    protocol = EvalProtocol.from_id("balance-v1/ideal")
    return replace(protocol, seeds=protocol.seeds[:1])


def fake_make_env(challenge: object, preset: str, **kwargs: Any) -> FakeEnv:
    """``kinds.make_env`` for the fake: ``evaluate`` builds its env through it."""
    return FakeEnv(preset, challenge, kwargs["record"])


def run_with(monkeypatch: pytest.MonkeyPatch, rows: list[np.ndarray[Any, Any]]) -> Recorder:
    FakeEnv.rows = rows
    monkeypatch.setattr(evaluate_module, "make_env", fake_make_env)
    recorder = Recorder()
    with pytest.raises(RuntimeError, match="non-finite observation"):
        evaluate(one_seed(), lambda spec: recorder, INFO)
    return recorder


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_a_non_finite_first_observation_raises_before_act(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    recorder = run_with(monkeypatch, [with_value(value, at=3)])
    assert recorder.observations == []


def test_a_non_finite_observation_after_a_step_raises_before_the_next_act(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = run_with(monkeypatch, [finite(), finite(), with_value(np.nan, at=9)])
    assert len(recorder.observations) == 2  # steps 0 and 1; step 2's observation never arrived


def test_a_float64_beyond_float32_range_is_non_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    """The controller sees float32 values (contract §6): 1e39 would reach it as inf."""
    row = np.zeros(len(FEATURES), dtype=np.float64)
    row[0] = 1e39
    recorder = run_with(monkeypatch, [row])
    assert recorder.observations == []


def test_the_message_names_the_step_and_blames_the_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeEnv.rows = [finite(), with_value(np.nan)]
    monkeypatch.setattr(evaluate_module, "make_env", fake_make_env)
    with pytest.raises(RuntimeError) as info:
        evaluate(one_seed(), lambda spec: Recorder(), INFO)
    assert str(info.value) == (
        "the env emitted a non-finite observation at step 1 (a harness bug, not the "
        "controller's: contract §17)"
    )


def test_the_check_holds_under_optimized_python(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import sys
        assert False, "this line is stripped under -O"
        sys.path.insert(0, sys.argv[1])
        import numpy as np
        import topple.evaluate as ev
        from test_harness_finite_obs import FakeEnv, Recorder, one_seed, INFO, with_value
        FakeEnv.rows = [with_value(float("nan"))]
        ev.make_env = lambda challenge, preset, **kw: FakeEnv(preset, challenge, kw["record"])
        recorder = Recorder()
        try:
            ev.evaluate(one_seed(), lambda spec: recorder, INFO)
        except RuntimeError as exc:
            print("raised:", exc, "acts:", len(recorder.observations))
        else:
            print("no error")
        """
    )
    tests_dir = str(Path(__file__).resolve().parent)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-O", "-c", script, tests_dir],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "raised: the env emitted a non-finite observation at step 0" in proc.stdout
    assert "acts: 0" in proc.stdout


def test_finite_observations_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check never fires on an honest episode (the real env, ideal preset)."""
    seen: list[BalancerObs] = []

    class Counting(Recorder):
        def act(self, obs: BalancerObs) -> tuple[float, float]:
            seen.append(obs)
            return (0.0, 0.0)

    results = evaluate(one_seed(), lambda spec: Counting(), INFO)
    assert results.episodes is not None
    assert results.episodes[0].error is None
    assert seen
