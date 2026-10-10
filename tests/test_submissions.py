import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from topple.contract import BalancerObs, Controller, TaskSpec, validate_action
from topple.envs.balance import BalanceEnv

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"
EXAMPLES = sorted(SUBMISSIONS.glob("*.py"))
BALANCERS = ("pid_example", "lqr_example")


@pytest.fixture(scope="module")
def env() -> BalanceEnv:
    return BalanceEnv("ideal")


@pytest.fixture(scope="module")
def spec(env: BalanceEnv) -> TaskSpec:
    return env.task_spec


def load(path: Path) -> ModuleType:
    """Import a submission by file path under a unique module name, as the loader does."""
    name = f"topple_example_{path.stem}"
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def example(stem: str) -> ModuleType:
    return load(SUBMISSIONS / f"{stem}.py")


def drive(
    env: BalanceEnv, controller: Controller[Any], steps: int, seed: int = 0
) -> tuple[list[tuple[float, ...]], int | None]:
    """Run ``steps`` control steps (starting a new episode after each end).

    Returns every action (validated against the contract) and the step of the first fall.
    """
    actions: list[tuple[float, ...]] = []
    first_fall: int | None = None
    obs_arr, info = env.reset(seed=seed)
    controller.reset(seed)
    for _ in range(steps):
        raw = controller.act(BalancerObs.from_array(obs_arr, info["t"]))
        actions.append(validate_action(raw, dim=env.task_spec.action.dim))
        obs_arr, _, terminated, truncated, info = env.step(raw)
        if terminated and first_fall is None:
            first_fall = info["step"]
        if terminated or truncated:
            seed += 1
            obs_arr, info = env.reset(seed=seed)
            controller.reset(seed)
    return actions, first_fall


def test_shipped_examples_are_present() -> None:
    stems = {p.stem for p in EXAMPLES}
    assert stems >= {"pid_example", "lqr_example", "zero_example", "random_example"}
    assert (SUBMISSIONS / "README.md").is_file()


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_follows_the_contract(path: Path, env: BalanceEnv, spec: TaskSpec) -> None:
    controller = load(path).make_controller(spec)
    assert isinstance(controller, Controller)
    actions, _ = drive(env, controller, steps=50)
    assert len(actions) == 50
    assert all(len(a) == 2 for a in actions)


@pytest.mark.parametrize("stem", BALANCERS)
def test_baseline_examples_balance(stem: str, env: BalanceEnv, spec: TaskSpec) -> None:
    _, first_fall = drive(env, example(stem).make_controller(spec), steps=300)
    assert first_fall is None


def test_zero_example_falls_within_two_seconds(env: BalanceEnv, spec: TaskSpec) -> None:
    for seed in (0, 1, 2):
        actions, first_fall = drive(env, example("zero_example").make_controller(spec), 200, seed)
        assert first_fall is not None, f"seed {seed}: still upright after 2 s"
        assert set(actions) == {(0.0, 0.0)}


def test_zero_example_needs_no_spec() -> None:
    controller = example("zero_example").make_controller()
    assert isinstance(controller, Controller)


def test_random_example_is_deterministic_per_seed(spec: TaskSpec) -> None:
    module = example("random_example")
    a, b = module.make_controller(spec), module.make_controller(spec)
    o = BalancerObs(0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0))

    def sequence(controller: Controller[Any], seed: int) -> list[tuple[float, ...]]:
        controller.reset(seed)
        return [tuple(controller.act(o)) for _ in range(20)]

    a.reset(7)
    b.reset(7)
    interleaved = [(tuple(a.act(o)), tuple(b.act(o))) for _ in range(20)]
    assert all(x == y for x, y in interleaved)  # instances share no RNG
    first = sequence(a, 7)
    assert first == [x for x, _ in interleaved]
    assert sequence(b, 7) == first
    assert sequence(a, 8) != first
    assert all(-1.0 <= v <= 1.0 for action in first for v in action)
    assert len({v for action in first for v in action}) == 40  # not a constant
