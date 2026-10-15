"""Paired bootstrap confidence intervals (docs/design/phase-2.md §5.3, normative).

Every paired comparison Topple reports (``topple compare``'s rows, the Phase 2 gates G2 and
G2b) comes from ``paired_bootstrap``, and the gates recompute it from the per-seed arrays a
comparison stores, so the algorithm is fixed: the per-seed differences ``a - b`` in seed order,
``n_resamples`` resamples of them drawn with one ``integers`` call of
``default_rng(seed)``, the mean of each resample, and the ``linear`` quantiles at
``(1 - level) / 2`` and ``(1 + level) / 2``. The point estimate is the plain mean of the
differences.

Arguments are checked before anything is drawn: the resample matrix (``n_resamples`` x
``len(a)`` indices, then as many values) is the one allocation that grows with the input, so
its size is capped.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

N_RESAMPLES: Final = 10_000
SEED: Final = 0
LEVEL: Final = 0.95
MAX_BOOTSTRAP_ELEMENTS: Final = 10**8
"""Most entries of the resample matrix: 10,000 resamples of the 10,000 episodes a protocol may
hold (contract §8), about 0.8 GB of indices plus as much of values."""


def paired_bootstrap(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> tuple[float, float, float]:
    """(mean, lo, hi) of ``a - b``, paired by position (seed order): the mean difference and
    its ``level`` bootstrap interval. ``ValueError`` for anything but two finite 1-D sequences
    of one length (at least 1), a positive int ``n_resamples``, an int ``seed >= 0`` and
    ``0 < level < 1``."""
    _check(n_resamples, seed, level)
    x = _vector(a, "a")
    y = _vector(b, "b")
    if x.shape != y.shape:
        raise ValueError(f"a and b must have the same length, got {x.size} and {y.size}")
    if n_resamples * x.size > MAX_BOOTSTRAP_ELEMENTS:
        raise ValueError(
            f"{n_resamples} resamples of {x.size} values exceed the {MAX_BOOTSTRAP_ELEMENTS} "
            "resample entries one bootstrap may draw"
        )
    # §5.3, verbatim: the gates recompute every stored row with this code.
    d = x - y  # seed order
    idx = np.random.default_rng(seed).integers(0, d.size, size=(n_resamples, d.size))  # one call
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2], method="linear")
    return float(d.mean()), float(lo), float(hi)


def bootstrap_ci(
    x: npt.ArrayLike,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> tuple[float, float, float]:
    """(mean, lo, hi) of ``x`` alone: ``paired_bootstrap(x, zeros)``."""
    values = _vector(x, "x")
    return paired_bootstrap(
        values, np.zeros(values.shape), n_resamples=n_resamples, seed=seed, level=level
    )


def _vector(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a sequence of numbers") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {arr.ndim} dimensions")
    if arr.size == 0:
        raise ValueError(f"{name} must hold at least one value")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _check(n_resamples: object, seed: object, level: object) -> None:
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError(f"n_resamples must be a positive int, got {n_resamples!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError(f"seed must be an int >= 0, got {seed!r}")
    if not isinstance(level, float | int) or isinstance(level, bool):
        raise ValueError(f"level must be a number in (0, 1), got {level!r}")
    if not (math.isfinite(level) and 0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
