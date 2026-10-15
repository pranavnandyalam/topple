"""The paired bootstrap that every gate and compare row uses (docs/design/phase-2.md §5.3, §6
``test_stats``).

§5.3 gives the algorithm as normative code; ``REFERENCE`` below is that code verbatim, and the
library must agree with it bit for bit. The known answer pins (mean, lo, hi) for the committed
200-element vectors in ``tests/data/stats_bootstrap_200.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from topple.stats import MAX_BOOTSTRAP_ELEMENTS, bootstrap_ci, paired_bootstrap

DATA = json.loads((Path(__file__).parent / "data" / "stats_bootstrap_200.json").read_text())
A: list[float] = DATA["a"]
B: list[float] = DATA["b"]


def REFERENCE(  # noqa: N802 - the §5.3 code, verbatim
    a: Any, b: Any, *, n_resamples: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)  # seed order
    idx = np.random.default_rng(seed).integers(0, d.size, size=(n_resamples, d.size))  # one call
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2], method="linear")
    return float(d.mean()), float(lo), float(hi)


def test_the_committed_vectors() -> None:
    assert len(A) == len(B) == 200
    assert all(0.0 <= x <= 100.0 for x in A + B)


def test_known_answer_paired() -> None:
    """Pinned for the committed vectors (seed 0, 10,000 resamples, 95%). The tolerance only
    absorbs a last-bit difference in numpy's summation across platforms; any change of seed,
    resample count, quantile method or pairing moves these by far more."""
    mean, lo, hi = paired_bootstrap(A, B)
    assert mean == pytest.approx(5.6484000000000005, abs=1e-12)
    assert lo == pytest.approx(0.17215000000000108, abs=1e-12)
    assert hi == pytest.approx(11.097112499999996, abs=1e-12)


def test_known_answer_one_sample() -> None:
    mean, lo, hi = bootstrap_ci(A)
    assert mean == pytest.approx(50.38985, abs=1e-12)
    assert lo == pytest.approx(46.312465, abs=1e-12)
    assert hi == pytest.approx(54.3360675, abs=1e-12)


def test_known_answer_with_other_arguments() -> None:
    mean, lo, hi = paired_bootstrap(A, B, seed=1, n_resamples=2000, level=0.9)
    assert mean == pytest.approx(5.6484000000000005, abs=1e-12)
    assert lo == pytest.approx(1.1010575000000014, abs=1e-12)
    assert hi == pytest.approx(9.934685, abs=1e-12)


@pytest.mark.parametrize(
    ("n", "kwargs"),
    [
        (200, {}),
        (20, {}),
        (7, {"seed": 3}),
        (200, {"seed": 12345, "n_resamples": 500, "level": 0.8}),
        (1, {}),
        (2, {"n_resamples": 1}),
    ],
)
def test_equals_the_normative_code_bit_for_bit(n: int, kwargs: dict[str, Any]) -> None:
    a, b = A[:n], B[:n]
    assert paired_bootstrap(a, b, **kwargs) == REFERENCE(a, b, **kwargs)
    assert bootstrap_ci(a, **kwargs) == REFERENCE(a, np.zeros(n), **kwargs)


def test_n_equals_one_is_a_point() -> None:
    assert paired_bootstrap([7.5], [2.25]) == (5.25, 5.25, 5.25)
    assert bootstrap_ci([3.0]) == (3.0, 3.0, 3.0)


@pytest.mark.parametrize("value", [0.0, 0.1, 67.89, -3.3])
def test_all_equal_differences_have_a_zero_width_interval(value: float) -> None:
    a = [value + 10.0] * 200
    b = [10.0] * 200
    mean, lo, hi = paired_bootstrap(a, b)
    assert mean == lo == hi  # every resample is the same multiset, summed the same way
    assert mean == pytest.approx(value, abs=1e-12)


def test_identical_inputs_give_zero() -> None:
    assert paired_bootstrap(A, A) == (0.0, 0.0, 0.0)


def test_pairing_follows_seed_order() -> None:
    """Paired, not two independent samples: shuffling one side changes the interval."""
    shuffled = B[1:] + B[:1]
    assert paired_bootstrap(A, B) != paired_bootstrap(A, shuffled)
    assert paired_bootstrap(A, B)[0] == pytest.approx(paired_bootstrap(A, shuffled)[0], abs=1e-9)


def test_swapping_sides_negates_the_interval() -> None:
    mean, lo, hi = paired_bootstrap(A, B)
    r_mean, r_lo, r_hi = paired_bootstrap(B, A)
    assert r_mean == pytest.approx(-mean, abs=1e-12)
    assert r_lo == pytest.approx(-hi, abs=1e-9)
    assert r_hi == pytest.approx(-lo, abs=1e-9)


def test_deterministic_per_seed() -> None:
    assert paired_bootstrap(A, B, seed=5) == paired_bootstrap(A, B, seed=5)
    assert paired_bootstrap(A, B, seed=5)[1:] != paired_bootstrap(A, B, seed=6)[1:]


def test_accepts_arrays_and_tuples() -> None:
    expected = paired_bootstrap(A, B)
    assert paired_bootstrap(np.asarray(A), tuple(B)) == expected


def test_returns_python_floats() -> None:
    assert all(type(x) is float for x in paired_bootstrap(A, B))


@pytest.mark.parametrize(
    ("a", "b", "kwargs", "match"),
    [
        ([], [], {}, "at least one"),
        ([1.0, 2.0], [1.0], {}, "same length"),
        ([[1.0, 2.0]], [[1.0, 2.0]], {}, "1-D"),
        ([1.0, float("nan")], [1.0, 2.0], {}, "finite"),
        ([1.0, float("inf")], [1.0, 2.0], {}, "finite"),
        ([1.0], [1.0], {"n_resamples": 0}, "n_resamples"),
        ([1.0], [1.0], {"n_resamples": True}, "n_resamples"),
        ([1.0], [1.0], {"n_resamples": 2.0}, "n_resamples"),
        ([1.0], [1.0], {"seed": -1}, "seed"),
        ([1.0], [1.0], {"seed": 1.5}, "seed"),
        ([1.0], [1.0], {"level": 0.0}, "level"),
        ([1.0], [1.0], {"level": 1.0}, "level"),
        ([1.0], [1.0], {"level": float("nan")}, "level"),
        ([1.0], [1.0], {"level": "0.95"}, "level"),
        (["a", "b"], [1.0, 2.0], {}, "numbers"),
        ([object()], [1.0], {}, "numbers"),
    ],
)
def test_bad_arguments_raise(
    a: list[Any], b: list[Any], kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        paired_bootstrap(a, b, **kwargs)


def test_the_resample_matrix_is_bounded() -> None:
    """One call draws n_resamples x n indices (§5.3); the product is capped before any draw,
    so a huge request fails at once instead of allocating."""
    n = MAX_BOOTSTRAP_ELEMENTS // 10_000 + 1
    with pytest.raises(ValueError, match="resample"):
        paired_bootstrap(np.zeros(n), np.zeros(n))
