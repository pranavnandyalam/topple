import json
import math
import shutil
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, overload

import pytest
from pydantic import ValidationError

import topple.contract as contract_module
import topple.evaluate as evaluate_module
from topple.challenge import load_challenge_dir
from topple.contract import BalancerObs, TaskSpec, controller_obs
from topple.controllers.pid import PIDController
from topple.envs.balance import BalanceEnv
from topple.evaluate import (
    RUNNER_ERROR_CODES,
    EpisodeRun,
    EvalProtocol,
    RunnerError,
    evaluate,
    run_episode,
    summarize,
    wants_epilogue,
)
from topple.results import (
    ControllerInfo,
    EpisodeResult,
    ErrorInfo,
    EvalResults,
    read_results,
    write_results,
)
from topple.rng import controller_seed
from topple.robot import challenge_dir

INFO = ControllerInfo(id="test", sha256="0" * 64)


def pid(_spec: TaskSpec) -> PIDController:
    return PIDController()


class Zero:
    def reset(self, rng_seed: int) -> None:
        pass

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        return (0.0, 0.0)


class Recorder:
    """Records what the harness hands the controller."""

    seeds: list[int] = []  # noqa: RUF012
    specs: list[TaskSpec] = []  # noqa: RUF012

    def __init__(self, spec: TaskSpec) -> None:
        Recorder.specs.append(spec)

    def reset(self, rng_seed: int) -> None:
        Recorder.seeds.append(rng_seed)

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        return (0.0, 0.0)


class Explodes:
    def __init__(self, at_step: int, payload: Any) -> None:
        self.at_step, self.payload, self.k = at_step, payload, 0

    def reset(self, rng_seed: int) -> None:
        self.k = 0

    def act(self, obs: BalancerObs) -> Any:
        self.k += 1
        if self.k > self.at_step:
            if isinstance(self.payload, BaseException):
                raise self.payload
            return self.payload
        return (0.0, 0.0)


def protocol(preset: str = "clean", n: int = 3) -> EvalProtocol:
    base = EvalProtocol.from_id(f"balance-v1/{preset}")
    return replace(base, seeds=tuple(range(n)))


SALT = 2**127 + 99  # a private protocol's 128-bit secret


def test_protocol_from_id_and_alias() -> None:
    p = EvalProtocol.from_id("eval_v1")
    assert p.id == "balance-v1/standard"
    assert len(p.seeds) == 20
    assert EvalProtocol.from_id("eval_v1_clean").preset == "clean"
    with pytest.raises(ValueError, match="no preset"):
        EvalProtocol.from_id("balance-v1/nope")


def test_protocol_hash_depends_on_preset() -> None:
    assert (
        EvalProtocol.from_id("balance-v1/clean").protocol_hash()
        != EvalProtocol.from_id("balance-v1/standard").protocol_hash()
    )


def test_protocol_hash_covers_the_seed_list_and_the_salt() -> None:
    p = protocol("ideal", 3)
    expected = p.challenge.protocol_hash(p.preset, seeds=p.seeds, salt=p.salt)
    assert p.protocol_hash() == expected
    others = {
        replace(p, seeds=p.seeds[:-1]).protocol_hash(),
        replace(p, seeds=(p.seeds[1], p.seeds[0], p.seeds[2])).protocol_hash(),
        replace(p, seed_set="private", salt=2**64).protocol_hash(),
        replace(p, seed_set="private", salt=2**127 + 1).protocol_hash(),
    }
    assert expected not in others
    assert len(others) == 4


def test_results_carry_the_protocols_full_hash() -> None:
    p = replace(protocol("ideal", 1), seed_set="private", salt=SALT)
    results = evaluate(p, pid, INFO)
    assert results.protocol_hash == p.challenge.protocol_hash(p.preset, seeds=p.seeds, salt=SALT)
    assert results.protocol_hash != replace(p, seed_set="public", salt=0).protocol_hash()


@pytest.mark.parametrize(
    ("seed_set", "salt"),
    [
        ("private", 0),
        ("private", 99),
        ("private", 2**64 - 1),
        ("private", 2**128),
        ("private", -(2**100)),
        ("private", True),
        ("private", float(2**100)),
        ("public", 1),
        ("public", 2**127),
        ("public", False),
    ],
)
def test_a_protocol_needs_the_salt_its_seed_set_calls_for(seed_set: str, salt: object) -> None:
    """§8, §9: public seeds use salt 0; private ones a 128-bit secret. A private protocol whose
    salt was forgotten (0) or hand-typed (99) would let a controller recognize its episodes."""
    with pytest.raises(ValueError, match="salt"):
        replace(protocol("ideal", 1), seed_set=seed_set, salt=salt)  # type: ignore[arg-type]


def test_salts_in_range_are_accepted() -> None:
    p = protocol("ideal", 1)
    assert p.salt == 0
    for salt in (2**64, SALT, 2**128 - 1):
        assert replace(p, seed_set="private", salt=salt).salt == salt


def test_the_seed_set_is_public_or_private() -> None:
    with pytest.raises(ValueError, match="seed_set"):
        replace(protocol("ideal", 1), seed_set="secret")  # type: ignore[arg-type]


def test_pid_on_clean_scores_high(tmp_path: Path) -> None:
    results = evaluate(protocol("clean", 3), pid, INFO, traj_dir=tmp_path)
    assert results.complete
    assert results.aggregate.success_rate == 1.0
    assert results.aggregate.score > 85
    assert results.episodes is not None
    assert all(e.error is None for e in results.episodes)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "traj_000.npz",
        "traj_001.npz",
        "traj_002.npz",
    ]


def test_zero_controller_scores_below_10() -> None:
    results = evaluate(protocol("clean", 3), lambda spec: Zero(), INFO)
    assert results.aggregate.success_rate == 0.0
    assert results.aggregate.score < 10


def test_controller_never_sees_episode_seed() -> None:
    Recorder.seeds.clear()
    Recorder.specs.clear()
    evaluate(protocol("ideal", 2), Recorder, INFO)
    assert len(Recorder.seeds) == 2
    assert not set(Recorder.seeds) & {0, 1}
    assert Recorder.specs[0].challenge == "balance-v1"
    assert Recorder.specs[0].robot.wheel_radius_m == pytest.approx(0.034)


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (RuntimeError("boom"), "exception"),
        ((float("nan"), 0.0), "invalid_action"),
        ((0.1,), "invalid_action"),
        ("left", "invalid_action"),
    ],
)
def test_controller_errors_are_recorded_not_raised(payload: Any, code: str) -> None:
    env = BalanceEnv("ideal", record=True)
    run = run_episode(env, lambda spec: Explodes(5, payload), seed=0, index=0)
    assert run.result.error is not None
    assert run.result.error.code == code
    assert run.result.error.step == 5
    assert run.result.score == 0.0
    assert not run.result.success
    assert run.trajectory.steps == 5


class FailsLate:
    """Balances with the PID, then raises at step ``at``."""

    def __init__(self, at: int) -> None:
        self.at, self.pid, self.k = at, PIDController(), 0

    def reset(self, rng_seed: int) -> None:
        self.pid.reset(rng_seed)

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        if self.k == self.at:
            raise RuntimeError("late failure")
        self.k += 1
        return self.pid.act(obs)


def test_an_errored_episode_did_not_survive() -> None:
    """The trajectory stops at the error with reason "timeout" (§14), so ``survived`` hinges on
    ``T == max_steps``: an episode that raised at step 400 must never report survival."""
    env = BalanceEnv("ideal", record=True)
    run = run_episode(env, lambda spec: FailsLate(400), seed=0, index=0)
    assert run.result.error is not None
    assert (run.result.error.code, run.result.error.step) == ("exception", 400)
    assert run.trajectory.meta["termination"] == {"step": 400, "reason": "timeout"}
    assert run.result.metrics is not None
    assert run.result.metrics["survived"] is False
    assert (run.result.score, run.result.success) == (0.0, False)


def test_factory_errors_are_recorded() -> None:
    def bad_factory(spec: TaskSpec) -> Zero:
        raise ImportError("missing dependency")

    env = BalanceEnv("ideal", record=True)
    run = run_episode(env, bad_factory, seed=0, index=0)
    assert run.result.error is not None
    assert run.result.error.code == "exception"
    assert run.trajectory.steps == 0


def test_long_error_detail_is_truncated() -> None:
    env = BalanceEnv("ideal", record=True)
    run = run_episode(env, lambda spec: Explodes(0, RuntimeError("x" * 5000)), seed=0, index=0)
    assert run.result.error is not None
    assert len(run.result.error.detail) == 300


def test_results_round_trip_and_atomic_write(tmp_path: Path) -> None:
    results = evaluate(protocol("ideal", 2), pid, INFO)
    path = tmp_path / "out" / "results.json"
    write_results(results, path)
    assert read_results(path) == results
    data = json.loads(path.read_text())
    assert data["format"] == "topple.results/1"
    assert data["isolation"] == "inprocess"
    assert [p.name for p in path.parent.iterdir()] == ["results.json"]


def test_results_reject_nan_and_extra_fields() -> None:
    results = evaluate(protocol("ideal", 1), pid, INFO)
    data = results.model_dump()
    data["aggregate"]["score"] = math.nan
    with pytest.raises(ValidationError):
        EvalResults.model_validate(data)
    data = results.model_dump()
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        EvalResults.model_validate(data)


def test_private_seed_set_redacts_episodes() -> None:
    p = replace(protocol("ideal", 2), seed_set="private", salt=SALT)
    results = evaluate(p, pid, INFO)
    assert results.seed_set == "private"
    assert results.episodes is None
    assert results.aggregate.episodes == 2


def test_error_info_codes_are_closed() -> None:
    with pytest.raises(ValidationError):
        ErrorInfo.model_validate({"code": "teleported", "step": 0})


# ------------------------------------------------------------------ controller misbehaviour (B)


READ_CAP = 16  # far above dim + 1: reading this many items means the islice guard is gone


class Endless:
    """Never runs out, but refuses to serve more than ``READ_CAP`` items.

    Unbounded, a regression that materializes the action (``list(action)``) would grow until the
    machine runs out of memory instead of failing.
    """

    def __iter__(self) -> Iterator[float]:
        for _ in range(READ_CAP):
            yield 0.0
        raise AssertionError(f"the harness read more than {READ_CAP} action items")


class Huge(Sequence[float]):
    """Claims 10**12 items (like ``range(10**12)``) but only ever serves ``READ_CAP`` of them."""

    def __len__(self) -> int:
        return 10**12

    @overload
    def __getitem__(self, index: int) -> float: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[float]: ...
    def __getitem__(self, index: int | slice) -> float | Sequence[float]:
        if not isinstance(index, int) or index >= READ_CAP:
            raise AssertionError(f"the harness read more than {READ_CAP} action items")
        return 0.0


class IterRaises:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def __iter__(self) -> Iterator[float]:
        raise self.exc


class BadStrError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("cannot describe myself")


class _NamelessMeta(type):
    """A metaclass whose ``__name__`` property wins over ``type.__name__`` and raises."""

    @property
    def __name__(cls) -> str:  # type: ignore[override]
        raise ZeroDivisionError("no name for you")


class NamelessError(Exception, metaclass=_NamelessMeta):
    pass


@dataclass(frozen=True)
class Deferred:
    """Builds a payload when the test runs. pytest renders test arguments with ``reprlib``, which
    reads ``type(x).__name__`` and would itself crash on a ``NamelessError`` argument."""

    build: Callable[[], Any]
    label: str

    def __repr__(self) -> str:
        return self.label


def materialize(payload: Any) -> Any:
    return payload.build() if isinstance(payload, Deferred) else payload


NAMELESS = Deferred(NamelessError, "NamelessError()")


class ClasslessError(Exception):
    """``isinstance(exc, str)`` consults ``exc.__class__`` after the real type: this one raises."""

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # type: ignore[override]
        raise ZeroDivisionError("no class for you")


CLASSLESS = Deferred(ClasslessError, "ClasslessError()")


class FormatBomb(str):
    """A ``str`` subclass whose own methods raise, or concatenate to something that is not a
    ``str`` (Python tries a subclass's reflected ``__radd__`` first, so ``": " + bomb`` runs it)."""

    def __format__(self, spec: str) -> str:
        raise RuntimeError("cannot format myself")

    def __str__(self) -> str:
        raise RuntimeError("cannot str myself")

    def __add__(self, other: object) -> Any:
        return {"not": "a str"}

    def __radd__(self, other: object) -> Any:
        return {"not": "a str"}


class FormatBombError(Exception):
    """``str(exc)`` hands back a ``str`` subclass whose own methods raise."""

    def __str__(self) -> str:
        return FormatBomb("boom")


def _renamed_error() -> type[Exception]:
    """An exception class whose real ``__name__`` is a hostile ``str`` subclass."""

    class RenamedError(Exception):
        pass

    RenamedError.__name__ = FormatBomb("Renamed")
    return RenamedError


RENAMED = Deferred(lambda: _renamed_error()(), "RenamedError()")


def lazy_action() -> Iterator[float]:
    yield from (0.1, 0.2)


def run_contained(factory: Any) -> EpisodeRun:
    """``run_episode`` on ``ideal`` seed 0, failing cleanly if anything escapes it.

    The failure is raised outside the ``except`` block, so pytest never has to render the escaped
    exception (a hostile one can break pytest's own reporting).
    """
    try:
        return run_episode(BalanceEnv("ideal", record=True), factory, seed=0, index=0)
    except KeyboardInterrupt:
        raise
    except BaseException:
        pass
    pytest.fail("run_episode raised for controller misbehaviour")


MISBEHAVIOUR: dict[str, tuple[Any, str]] = {
    "act raises SystemExit": (SystemExit(3), "exception"),
    "act raises GeneratorExit": (GeneratorExit(), "exception"),
    "act raises StopIteration": (StopIteration(), "exception"),
    "exception with a broken __str__": (BadStrError(), "exception"),
    "exception type whose __name__ raises": (NAMELESS, "exception"),
    "__str__ returns a str subclass that cannot be formatted": (FormatBombError(), "exception"),
    "exception type named by a hostile str subclass": (RENAMED, "exception"),
    "exception whose __class__ raises": (CLASSLESS, "exception"),
    "action iteration raises SystemExit": (IterRaises(SystemExit(1)), "exception"),
    "action iteration raises ValueError": (IterRaises(ValueError("bad")), "invalid_action"),
    "action iteration raises an exception whose class name raises": (
        Deferred(lambda: IterRaises(NamelessError()), "IterRaises(NamelessError())"),
        "invalid_action",
    ),
    "generator": (lazy_action(), "invalid_action"),
    "huge int": ([10**400, 0.0], "invalid_action"),
    "endless iterable": (Endless(), "invalid_action"),
    "huge sequence": (Huge(), "invalid_action"),
    "set": ({0.1, 0.2}, "invalid_action"),
}


@pytest.mark.parametrize(("payload", "code"), MISBEHAVIOUR.values(), ids=MISBEHAVIOUR.keys())
def test_misbehaving_controllers_never_crash_the_harness(payload: Any, code: str) -> None:
    run = run_contained(lambda spec: Explodes(5, materialize(payload)))
    assert run.result.error is not None
    assert run.result.error.code == code
    assert run.result.error.step == 5
    assert run.result.score == 0.0
    assert not run.result.success
    assert run.trajectory.steps == 5


@pytest.mark.parametrize(
    "payload",
    [Endless(), Huge(), Deferred(lambda: IterRaises(NamelessError()), "IterRaises(Nameless)")],
    ids=["endless iterable", "huge sequence", "iteration raises a nameless exception"],
)
def test_invalid_actions_are_described_by_what_was_wrong(payload: Any) -> None:
    """The detail tells an over-long action (one item too many read) from one that could not be
    read at all: a regression that reads past ``READ_CAP`` changes the detail, not just memory."""
    run = run_contained(lambda spec: Explodes(5, materialize(payload)))
    assert run.result.error is not None
    expected = (
        "ActionError: action could not be read: NamelessError"
        if isinstance(payload, Deferred)
        else "ActionError: action must have exactly 2 values, got more than 2"
    )
    assert (run.result.error.code, run.result.error.detail) == ("invalid_action", expected)


@pytest.mark.parametrize(
    ("exc", "detail"),
    [
        (BadStrError(), "BadStrError: <str() failed>"),
        (NAMELESS, "NamelessError: "),
        (FormatBombError(), "FormatBombError: boom"),
        (Deferred(lambda: _renamed_error()("why"), "RenamedError('why')"), "Renamed: why"),
        (CLASSLESS, "ClasslessError: "),
    ],
    ids=[
        "broken __str__",
        "raising __name__",
        "str subclass message",
        "str subclass name",
        "raising __class__",
    ],
)
def test_hostile_exceptions_are_described_by_their_real_name(exc: Any, detail: str) -> None:
    run = run_contained(lambda spec: Explodes(2, materialize(exc)))
    assert run.result.error is not None
    assert (run.result.error.code, run.result.error.detail) == ("exception", detail)
    assert type(run.result.error.detail) is str


def test_an_undescribable_exception_still_becomes_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever describing the exception does, the harness records an error and carries on."""

    def explode(exc: BaseException) -> str:
        raise MemoryError

    monkeypatch.setattr(contract_module, "_exception_detail", explode)
    run = run_contained(lambda spec: Explodes(2, RuntimeError("x")))
    assert run.result.error is not None
    assert run.result.error.code == "exception"
    assert run.result.error.detail == "<exception could not be described>"


@pytest.mark.parametrize("where", ["factory", "reset"])
@pytest.mark.parametrize(
    "exc",
    [SystemExit(2), GeneratorExit(), BadStrError(), NAMELESS, FormatBombError(), CLASSLESS],
    ids=["SystemExit", "GeneratorExit", "BadStr", "Nameless", "FormatBomb", "Classless"],
)
def test_setup_failures_are_exceptions(where: str, exc: Any) -> None:
    class Fails(Zero):
        def reset(self, rng_seed: int) -> None:
            raise materialize(exc)

    def factory(spec: TaskSpec) -> Zero:
        if where == "factory":
            raise materialize(exc)
        return Fails()

    run = run_contained(factory)
    assert run.result.error is not None
    assert (run.result.error.code, run.result.error.step) == ("exception", 0)
    assert run.trajectory.steps == 0


class InterruptingStrError(Exception):
    def __str__(self) -> str:
        raise KeyboardInterrupt


def test_keyboard_interrupt_while_describing_an_exception_still_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(KeyboardInterrupt):
        run_episode(
            BalanceEnv("ideal", record=True),
            lambda spec: Explodes(2, InterruptingStrError()),
            seed=0,
            index=0,
        )

    def interrupted(exc: BaseException) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(contract_module, "_exception_detail", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_episode(
            BalanceEnv("ideal", record=True),
            lambda spec: Explodes(2, RuntimeError("x")),
            seed=0,
            index=0,
        )


def test_a_controller_whose_class_attribute_raises_still_runs() -> None:
    """The harness never asks the controller object what class it is (``isinstance`` would)."""

    class Classless(Zero):
        @property  # type: ignore[misc]
        def __class__(self) -> type:  # type: ignore[override]
            raise ZeroDivisionError("no class for you")

    run = run_contained(lambda spec: Classless())
    assert run.result.error is None
    assert run.trajectory.meta["termination"]["reason"] == "fell"


@pytest.mark.parametrize("where", ["factory", "reset", "act", "action"])
def test_keyboard_interrupt_still_stops_the_run(where: str) -> None:
    class Interrupts(Zero):
        def reset(self, rng_seed: int) -> None:
            if where == "reset":
                raise KeyboardInterrupt

        def act(self, obs: BalancerObs) -> Any:
            if where == "act":
                raise KeyboardInterrupt
            return IterRaises(KeyboardInterrupt()) if where == "action" else (0.0, 0.0)

    def factory(spec: TaskSpec) -> Zero:
        if where == "factory":
            raise KeyboardInterrupt
        return Interrupts()

    with pytest.raises(KeyboardInterrupt):
        run_episode(BalanceEnv("ideal", record=True), factory, seed=0, index=0)


def test_sim_unstable_scores_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def blows_up(self: BalanceEnv) -> bool:
        calls["n"] += 1
        return calls["n"] > 37  # the 38th substep: inside control step 7

    monkeypatch.setattr(BalanceEnv, "_unstable", blows_up)
    run = run_episode(BalanceEnv("ideal", record=True), pid, seed=0, index=0)
    assert run.result.error is not None
    assert (run.result.error.code, run.result.error.step) == ("sim_unstable", 8)
    assert run.result.score == 0.0
    assert not run.result.success
    assert run.trajectory.meta["termination"]["reason"] == "sim_unstable"
    assert run.trajectory.epilogue_qpos.shape[0] == 0  # nothing to show after a blow-up


# ------------------------------------------------------------------ private seeds (B, §8)


def test_private_seeds_use_the_salt() -> None:
    Recorder.seeds.clear()
    Recorder.specs.clear()
    salt = 2**127 + 12345
    p = replace(protocol("ideal", 2), seed_set="private", salt=salt)
    evaluate(p, Recorder, INFO)
    assert Recorder.seeds == [controller_seed(s, salt) for s in p.seeds]
    assert not set(Recorder.seeds) & {controller_seed(s, 0) for s in p.seeds}


def test_private_seeds_never_save_trajectories(tmp_path: Path) -> None:
    p = replace(protocol("ideal", 2), seed_set="private", salt=SALT)
    results = evaluate(p, pid, INFO, traj_dir=tmp_path / "traj")
    assert not (tmp_path / "traj").exists() or not any((tmp_path / "traj").iterdir())
    assert results.episodes is None
    assert results.aggregate.episodes == 2
    text = results.model_dump_json()
    assert all(f'"seed":{s}' not in text.replace(" ", "") for s in p.seeds)


def test_run_episode_can_hide_the_seed() -> None:
    run = run_episode(BalanceEnv("ideal", record=True), pid, seed=7, index=0, report_seed=False)
    assert run.result.seed is None
    assert run_episode(BalanceEnv("ideal", record=True), pid, seed=7, index=0).result.seed == 7


# ------------------------------------------------------------------ epilogue (C)


@pytest.mark.parametrize(
    ("reason", "has_error", "expected"),
    [
        ("fell", False, True),
        ("floor_contact", False, True),
        ("timeout", False, False),
        ("timeout", True, True),
        ("fell", True, True),
        ("sim_unstable", True, False),
        ("sim_unstable", False, False),
        ("drift", False, False),
    ],
)
def test_wants_epilogue(reason: str, has_error: bool, expected: bool) -> None:
    assert wants_epilogue(reason, has_error) is expected


def test_surviving_episode_has_no_epilogue() -> None:
    run = run_episode(BalanceEnv("ideal", record=True), pid, seed=0, index=0)
    assert run.trajectory.meta["termination"]["reason"] == "timeout"
    assert run.trajectory.epilogue_qpos.shape == (0, run.trajectory.qpos.shape[1])


def test_fallen_and_errored_episodes_get_a_one_second_epilogue() -> None:
    fell = run_episode(BalanceEnv("ideal", record=True), lambda spec: Zero(), seed=0, index=0)
    assert fell.trajectory.meta["termination"]["reason"] == "fell"
    assert fell.trajectory.epilogue_qpos.shape[0] == 100
    errored = run_episode(
        BalanceEnv("ideal", record=True),
        lambda spec: Explodes(5, RuntimeError("x")),
        seed=0,
        index=0,
    )
    assert errored.trajectory.epilogue_qpos.shape[0] == 100


# ------------------------------------------------------------------ aggregates (E)


def episode(index: int, score: float, success: bool, upright_s: float) -> EpisodeResult:
    return EpisodeResult(
        index=index,
        seed=index,
        success=success,
        score=score,
        metrics={"time_upright_s": upright_s},
        error=None,
    )


def test_summarize_divides_by_the_expected_episode_count() -> None:
    present = [episode(0, 90.0, True, 10.0), episode(1, 60.0, False, 4.0)]
    agg = summarize(present, expected=4)
    assert agg.score == 37.5  # missing episodes count as 0
    assert agg.success_rate == 0.25
    assert agg.mean_time_upright_s == 7.0  # over the episodes that are present
    assert (agg.episodes, agg.errors) == (2, 0)


def test_summarize_uses_exact_summation() -> None:
    # Python 3.12's sum() is already compensated (Neumaier), so it takes an awkward case to tell
    # it from the correctly rounded fsum: the subnormal throws the compensation off.
    scores = [1.0, 1.0, 5e-324, 2.0**-52]
    assert sum(scores) / 4 != math.fsum(scores) / 4
    present = [episode(i, v, False, 1.0) for i, v in enumerate(scores)]
    assert summarize(present, expected=4).score == math.fsum(scores) / 4


def test_summarize_rejects_more_episodes_than_expected() -> None:
    with pytest.raises(ValueError, match="expected"):
        summarize([episode(0, 1.0, True, 1.0), episode(1, 1.0, True, 1.0)], expected=1)


def test_summarize_of_nothing_is_zero() -> None:
    agg = summarize([], expected=3)
    assert (agg.score, agg.success_rate, agg.mean_time_upright_s, agg.episodes) == (0, 0, 0, 0)


# ------------------------------------------------------------------ observation path (F)


def test_controllers_receive_controller_obs_of_the_recorded_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def spy(obs_arr: Any, t: float) -> BalancerObs:
        calls.append(t)
        return controller_obs(obs_arr, t)

    seen: list[BalancerObs] = []

    class Sees(Zero):
        def act(self, obs: BalancerObs) -> tuple[float, float]:
            seen.append(obs)
            return PIDController().act(obs)

    monkeypatch.setattr(evaluate_module, "controller_obs", spy)
    run = run_episode(BalanceEnv("clean", record=True), lambda spec: Sees(), seed=1, index=0)
    traj = run.trajectory
    assert len(calls) == len(seen) == traj.steps
    for k, obs in enumerate(seen):
        assert obs.to_wire() == controller_obs(traj.obs[k], k * 0.01).to_wire()
        assert BalancerObs.from_wire(obs.to_wire()) == obs


# ------------------------------------------------------------------ one bundle per protocol (I)


def test_evaluate_runs_on_the_protocols_challenge_object(tmp_path: Path) -> None:
    root = tmp_path / "balance-v1"
    shutil.copytree(challenge_dir("balance-v1"), root)
    yaml_path = root / "challenge.yaml"
    yaml_path.write_text(yaml_path.read_text().replace("e_ref_j: 13.0", "e_ref_j: 0.001"))
    base = protocol("clean", 1)
    custom = replace(base, challenge=load_challenge_dir(root))
    a = evaluate(base, pid, INFO)
    b = evaluate(custom, pid, INFO)
    assert a.episodes is not None
    assert b.episodes is not None
    assert b.episodes[0].score < a.episodes[0].score  # scored with the custom E_ref
    assert b.protocol_hash != a.protocol_hash


# ------------------------------------------------------------------ runner errors (§15)


class RaisesAt:
    """A controller (or proxy) that raises ``error`` from ``where``: factory, reset or act."""

    def __init__(self, where: str, error: BaseException, at_step: int = 5) -> None:
        self.where, self.error, self.at_step, self.k = where, error, at_step, 0
        if where == "factory":
            raise error

    def reset(self, rng_seed: int) -> None:
        if self.where == "reset":
            raise self.error

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        self.k += 1
        if self.where == "act" and self.k > self.at_step:
            raise self.error
        return PIDController().act(obs)


@pytest.mark.parametrize("code", sorted(RUNNER_ERROR_CODES))
@pytest.mark.parametrize(("where", "step"), [("factory", 0), ("reset", 0), ("act", 5)])
def test_a_runner_error_ends_the_episode_with_its_code(code: str, where: str, step: int) -> None:
    """Phase 3's controller proxy reports its own failures (a reply that never came, a crashed
    subprocess, a malformed or oversized reply) with the §15 code, not as ``exception``."""
    error = RunnerError(code, "no reply within 20 ms")  # type: ignore[arg-type]
    env = BalanceEnv("ideal", record=True)
    run = run_episode(env, lambda spec: RaisesAt(where, error), seed=0, index=0)
    assert run.result.error == ErrorInfo(code=code, step=step, detail="no reply within 20 ms")
    assert run.result.score == 0.0
    assert not run.result.success
    assert run.trajectory.steps == step
    assert run.trajectory.epilogue_qpos.shape[0] == 100  # a controller error shows what follows


def test_runner_error_codes_are_the_runners_only() -> None:
    expected = {"timeout", "crash", "protocol", "output_limit", "load_error"}
    assert expected == RUNNER_ERROR_CODES
    for code in ("exception", "invalid_action", "sim_unstable", "missing", "incomplete", "x"):
        with pytest.raises(ValueError, match="runner error code"):
            RunnerError(code, "detail")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="detail"):
        RunnerError("timeout", 5)  # type: ignore[arg-type]


class HostileStr(str):
    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("compared")

    def __hash__(self) -> int:
        raise RuntimeError("hashed")


def subclassed() -> BaseException:
    class MineError(RunnerError):
        pass

    return MineError("timeout", "from a subclass")


def tampered_code() -> BaseException:
    error = RunnerError("timeout", "tampered")
    error.code = HostileStr("timeout")  # type: ignore[assignment]
    return error


def tampered_detail() -> BaseException:
    error = RunnerError("crash", "tampered")
    error.detail = HostileStr("x")
    return error


def deleted_code() -> BaseException:
    error = RunnerError("protocol", "deleted")
    del error.code
    return error


@pytest.mark.parametrize(
    ("make", "detail"),
    [
        (subclassed, "MineError: ('timeout', 'from a subclass')"),
        (tampered_code, "RunnerError: ('timeout', 'tampered')"),
        (tampered_detail, "RunnerError: ('crash', 'tampered')"),
        (deleted_code, "RunnerError: ('protocol', 'deleted')"),
    ],
    ids=["subclass", "tampered code", "tampered detail", "deleted code"],
)
def test_only_the_harness_runner_error_names_a_code(
    make: Callable[[], BaseException], detail: str
) -> None:
    """Anything but ``RunnerError`` itself, holding an exact-``str`` runner code and an
    exact-``str`` detail, is an ordinary ``exception``: in-process code cannot smuggle a hostile
    value into the record."""
    run = run_episode(
        BalanceEnv("ideal", record=True), lambda spec: RaisesAt("act", make()), seed=0, index=0
    )
    assert run.result.error == ErrorInfo(code="exception", step=5, detail=detail)


def test_a_runner_error_names_the_code_it_holds_when_raised() -> None:
    """§15: what counts is the exact ``str`` code and detail the instance holds when it is
    raised, reassigned or not; a valid code only picks another zero-scoring error."""
    error = RunnerError("timeout", "built with")
    error.code = "crash"
    error.detail = "reassigned"
    run = run_episode(
        BalanceEnv("ideal", record=True), lambda spec: RaisesAt("act", error), seed=0, index=0
    )
    assert run.result.error == ErrorInfo(code="crash", step=5, detail="reassigned")
    assert run.result.score == 0.0
