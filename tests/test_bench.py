"""``topple shove``: survival against one calibrated push (docs/design/phase-2.md §5.2 item 3,
§5.3, §6 ``test_bench``).

Preset ``clean`` (sensor noise and model randomization, no pushes of its own), one push at
2.0 s lasting 0.1 s with no lateral part, injected through the trusted
``reset(options={"pushes": ...})`` hook (contract §9), at every impulse level and in each
direction. ``impulse_at_50`` is the first downward crossing of 50% survival, linearly
interpolated between levels.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import topple.bench as bench_module
from policy_support import write_policy_dir
from pool_support import InlinePool
from topple.bench import (
    PULSE_S,
    PUSH_TIME_S,
    SHOVE_PROTOCOL,
    ShoveCurve,
    impulse_at_50,
    parse_impulses,
    shove_curve,
    shove_curves,
    shove_push,
)
from topple.cli import app
from topple.disturbances import Push
from topple.evaluate import EvalProtocol
from topple.loader import LoadError, submission_sha256
from topple.pool import run_episodes

REPO = Path(__file__).resolve().parents[1]
PID = REPO / "submissions" / "pid_example.py"
runner = CliRunner()
CALM, HARD = 0.0, 3.0  # N·s: PID survives no push and falls to a 3 N·s shove on clean seeds


@pytest.fixture
def inline(monkeypatch: pytest.MonkeyPatch) -> list[InlinePool]:
    InlinePool.instances = []
    monkeypatch.setattr(bench_module, "eval_pool", InlinePool)
    return InlinePool.instances


@pytest.fixture(scope="module")
def pid(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sub = tmp_path_factory.mktemp("subs") / "pid_example.py"
    shutil.copyfile(PID, sub)
    return sub


def curve(survival: list[float], impulses: list[float] | None = None) -> ShoveCurve:
    """A curve with the given survival fractions over 4 seeds."""
    levels = impulses or [0.3 + 0.025 * k for k in range(len(survival))]
    survived = tuple(tuple(k < round(s * 4) for k in range(4)) for s in survival)
    return ShoveCurve(
        id="bot",
        sha256="a" * 64,
        policy_sha256=None,
        protocol=SHOVE_PROTOCOL,
        protocol_hash="b" * 64,
        forward_sign=1,
        impulses=tuple(levels),
        seeds=(0, 1, 2, 3),
        survived=survived,
    )


# --- impulse levels ---------------------------------------------------------------------------


def test_the_default_levels() -> None:
    levels = parse_impulses("0.30:0.65:0.025")
    assert len(levels) == 15
    assert levels[0] == 0.3
    assert levels[-1] == 0.65
    assert levels == tuple(float(f"{0.3 + 0.025 * k:.3f}") for k in range(15))  # exact decimals


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0:3:3", (0.0, 3.0)), ("0.5:0.5:0.1", (0.5,)), ("1:2:0.5", (1.0, 1.5, 2.0))],
)
def test_levels(text: str, expected: tuple[float, ...]) -> None:
    assert parse_impulses(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "0.3:0.65",
        "0.3:0.65:0",
        "0.65:0.3:0.025",
        "0.3:0.65:0.03",  # not a whole number of steps
        "-0.1:0.3:0.1",
        "0:6:1",  # beyond 5 N·s, the force cap for a 0.1 s pulse (contract §9)
        "0:5:0.001",  # more levels than a bench runs
        "a:b:c",
        "1e-1:1:0.1",
        "nan:1:0.1",
        " 0:1:1",
    ],
)
def test_bad_levels(text: str) -> None:
    with pytest.raises(ValueError, match="impulses"):
        parse_impulses(text)


# --- impulse_at_50 ----------------------------------------------------------------------------


def test_interpolates_the_first_downward_crossing() -> None:
    c = curve([1.0, 1.0, 0.75, 0.25, 0.0])
    # between 0.35 (75%) and 0.375 (25%): 0.35 + (0.75 - 0.5) / (0.75 - 0.25) * 0.025
    assert impulse_at_50(c) == pytest.approx(0.3625, abs=1e-12)


def test_a_level_at_exactly_50_percent_is_the_crossing() -> None:
    assert impulse_at_50(curve([1.0, 0.5, 0.25])) == pytest.approx(0.325, abs=1e-12)
    assert impulse_at_50(curve([1.0, 0.75, 0.5, 0.0])) == pytest.approx(0.35, abs=1e-12)


def test_only_the_first_crossing_counts() -> None:
    assert impulse_at_50(curve([0.75, 0.25, 1.0, 0.0])) == pytest.approx(0.3125, abs=1e-12)


@pytest.mark.parametrize(
    "survival",
    [[1.0, 1.0, 0.75], [0.25, 0.0], [0.5, 0.5], [0.25, 0.75, 1.0], [1.0]],
    ids=["never-below", "starts-below", "never-below-half", "rising", "one-level"],
)
def test_none_without_a_downward_crossing(survival: list[float]) -> None:
    assert impulse_at_50(curve(survival)) is None


def test_uneven_levels_interpolate_on_impulse() -> None:
    c = curve([1.0, 0.0], impulses=[0.2, 1.0])
    assert impulse_at_50(c) == pytest.approx(0.6, abs=1e-12)


def test_survival_is_the_fraction_of_seeds() -> None:
    assert curve([1.0, 0.75, 0.0]).survival == (1.0, 0.75, 0.0)


@pytest.mark.parametrize(
    "fields",
    [
        {"impulses": (0.4, 0.3)},
        {"impulses": (0.3, 0.3)},
        {"impulses": ()},
        {"survived": ((True, True, True, True),)},  # one row for two levels
        {"survived": ((True,), (False,))},  # one seed of four
        {"forward_sign": 0},
        {"seeds": ()},
    ],
)
def test_a_curve_is_consistent(fields: dict[str, Any]) -> None:
    base = curve([1.0, 0.0])
    with pytest.raises(ValueError):  # noqa: PT011
        replace(base, **fields)


# --- the push ---------------------------------------------------------------------------------


@pytest.mark.parametrize("sign", [1, -1])
def test_the_push(sign: int) -> None:
    protocol = EvalProtocol.from_id(SHOVE_PROTOCOL)
    push = shove_push(0.45, sign, protocol)
    assert push == Push(
        start_step=200,  # 2.0 s at 100 Hz
        n_substeps=50,  # 0.1 s at 500 Hz
        magnitude_n=push.magnitude_n,
        forward_sign=float(sign),
        lateral_ratio=0.0,
    )
    assert push.magnitude_n * push.n_substeps * 0.002 == pytest.approx(0.45, abs=1e-15)
    assert (PUSH_TIME_S, PULSE_S) == (2.0, 0.1)


# --- shove_curve ------------------------------------------------------------------------------


def test_shove_curve_runs_every_level_on_clean_with_the_push(
    pid: Path, inline: list[InlinePool]
) -> None:
    c = shove_curve(pid, [CALM, HARD], [0, 1], forward_sign=-1, workers=2)
    assert c.protocol == "balance-v1/clean"
    assert c.seeds == (0, 1)
    assert c.impulses == (CALM, HARD)
    assert c.forward_sign == -1
    assert c.survival == (1.0, 0.0)
    assert impulse_at_50(c) == pytest.approx(1.5, abs=1e-12)
    assert (c.id, c.sha256, c.policy_sha256) == ("pid_example", submission_sha256(pid), None)
    protocol = replace(EvalProtocol.from_id(SHOVE_PROTOCOL), seeds=(0, 1))
    assert c.protocol_hash == protocol.protocol_hash()
    (pool,) = inline
    assert [(call[2], call[3], call[4]) for call in pool.calls] == [
        (protocol, (0, 1), (shove_push(CALM, -1, protocol),)),
        (protocol, (0, 1), (shove_push(HARD, -1, protocol),)),
    ]
    assert pool.shut_down


def test_survival_comes_from_the_injected_episodes(pid: Path, inline: list[InlinePool]) -> None:
    c = shove_curve(pid, [0.45], [0, 1], workers=1)
    protocol = replace(EvalProtocol.from_id(SHOVE_PROTOCOL), seeds=(0, 1))
    from topple.pool import policy_ref

    ref = policy_ref(pid, challenge=protocol.challenge)
    episodes = run_episodes(ref, protocol, [0, 1], (shove_push(0.45, 1, protocol),))
    assert c.survived == (tuple(bool(e.metrics and e.metrics["survived"]) for e in episodes),)


def test_shove_curves_share_one_pool_for_both_directions(
    pid: Path, inline: list[InlinePool]
) -> None:
    forward, backward = shove_curves(pid, [CALM, HARD], [0, 1], workers=2)
    assert (forward.forward_sign, backward.forward_sign) == (1, -1)
    assert forward.survival == backward.survival == (1.0, 0.0)
    (pool,) = inline
    assert [call[4][0].forward_sign for call in pool.calls] == [1.0, 1.0, -1.0, -1.0]


def test_shove_curve_chunks_the_seeds(
    pid: Path, inline: list[InlinePool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench_module, "CHUNK_EPISODES", 2)
    c = shove_curve(pid, [CALM], [5, 6, 7], workers=2)
    (pool,) = inline
    assert [call[3] for call in pool.calls] == [(0, 1), (2,)]
    assert c.seeds == (5, 6, 7)
    assert len(c.survived[0]) == 3


def test_a_real_pool_gives_the_same_curve(pid: Path, tmp_path: Path) -> None:
    policy = write_policy_dir(tmp_path / "tiny_ppo")
    pid_curve, ppo_curve = (
        shove_curve(sub, [CALM, HARD], [0], forward_sign=1, workers=2) for sub in (pid, policy)
    )
    assert pid_curve.survival == (1.0, 0.0)
    assert ppo_curve.survival == (0.0, 0.0)  # random weights fall without any push
    assert ppo_curve.policy_sha256 is not None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"impulses": []}, "impulse"),
        ({"impulses": [0.4, 0.3]}, "impulse"),
        ({"impulses": [0.3, math.nan]}, "impulse"),
        ({"impulses": [-0.1]}, "impulse"),
        ({"impulses": [5.5]}, "impulse"),
        ({"impulses": [True]}, "impulse"),
        ({"seeds": []}, "seed"),
        ({"seeds": [-1]}, "seed"),
        ({"seeds": [2**53]}, "seed"),
        ({"seeds": range(10_001)}, "seed"),
        ({"forward_sign": 0}, "forward_sign"),
        ({"forward_sign": True}, "forward_sign"),
        ({"forward_sign": 1.0}, "forward_sign"),
        ({"workers": 0}, "workers"),
    ],
)
def test_shove_curve_refuses(
    pid: Path, inline: list[InlinePool], kwargs: dict[str, Any], match: str
) -> None:
    args: dict[str, Any] = {"impulses": [CALM], "seeds": [0], "forward_sign": 1, "workers": 1}
    args.update(kwargs)
    impulses, seeds = args.pop("impulses"), args.pop("seeds")
    with pytest.raises(ValueError, match=match):
        shove_curve(pid, impulses, seeds, **args)
    assert inline == []  # refused before any pool exists


def test_an_endless_iterable_is_refused_after_a_bounded_read(
    pid: Path, inline: list[InlinePool]
) -> None:
    import itertools

    with pytest.raises(ValueError, match="seed"):
        shove_curve(pid, [CALM], itertools.count())
    with pytest.raises(ValueError, match="impulse"):
        shove_curve(pid, itertools.count(), [0])


def test_shove_curve_propagates_a_load_error(tmp_path: Path, inline: list[InlinePool]) -> None:
    with pytest.raises(LoadError):
        shove_curve(tmp_path / "absent.py", [CALM], [0])


# --- CLI --------------------------------------------------------------------------------------


def test_cli_shove_prints_both_directions(pid: Path, inline: list[InlinePool]) -> None:
    result = runner.invoke(
        app, ["shove", str(pid), "--impulses", "0:3:3", "--seeds", "0..1", "--direction", "both"]
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "pid\\_example" in out
    assert "| 0.000 | 1.00 | 1.00 |" in out
    assert "| 3.000 | 0.00 | 0.00 |" in out
    assert "i50: forward 1.500 N·s, backward 1.500 N·s, min 1.500 N·s" in out
    assert len(inline) == 1  # one pool for every submission and direction


@pytest.mark.parametrize(("direction", "sign"), [("forward", 1.0), ("backward", -1.0)])
def test_cli_shove_one_direction(
    pid: Path, inline: list[InlinePool], direction: str, sign: float
) -> None:
    args = ["shove", str(pid), "--impulses", "3:3:1", "--seeds", "0..0", "--direction", direction]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert f"i50: {direction} none" in result.output
    (pool,) = inline
    assert [call[4][0].forward_sign for call in pool.calls] == [sign]


def test_cli_shove_defaults(pid: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(subs: Any, impulses: Any, seeds: Any, signs: Any, workers: int) -> Any:
        seen.update(impulses=impulses, seeds=seeds, signs=signs, workers=workers)
        return [tuple(curve([1.0, 0.0], [0.3, 0.325]) for _ in signs) for _ in subs]

    monkeypatch.setattr(bench_module, "run_shoves", fake)
    result = runner.invoke(app, ["shove", str(pid)])
    assert result.exit_code == 0, result.output
    assert seen["impulses"] == parse_impulses("0.30:0.65:0.025")
    assert seen["seeds"] == tuple(range(100))
    assert seen["signs"] == (1, -1)
    assert seen["workers"] == 4


@pytest.mark.parametrize(
    "args",
    [
        ["--impulses", "0.3:0.2:0.1"],
        ["--seeds", "9..1"],
        ["--direction", "sideways"],
        ["--workers", "0"],
    ],
)
def test_cli_shove_bad_options_exit_2(pid: Path, args: list[str]) -> None:
    result = runner.invoke(app, ["shove", str(pid), *args])
    assert result.exit_code == 2


def test_cli_shove_load_error_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["shove", str(tmp_path / "absent.py"), "--seeds", "0..0"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_shove_a_dead_worker_exits_2(pid: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from concurrent.futures.process import BrokenProcessPool

    def dead(*args: object, **kwargs: object) -> None:
        raise BrokenProcessPool("a child process terminated abruptly")

    monkeypatch.setattr(bench_module, "run_shoves", dead)
    result = runner.invoke(app, ["shove", str(pid), "--seeds", "0..0"])
    assert result.exit_code == 2
    assert "an evaluation worker died" in result.output
