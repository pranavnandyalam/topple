"""``topple shove``: survival against one calibrated push (docs/design/phase-2.md §5.2 item 3,
§5.3).

Each episode runs on preset ``clean`` of balance-v1 (sensor noise and model randomization, no
pushes of its own) with exactly one push in place of the scenario's: at 2.0 s, a 0.1 s pulse
along the heading (``forward_sign`` +1) or against it (-1), no lateral part, of a given impulse.
The push goes in through the trusted ``reset(options={"pushes": ...})`` hook (contract §9), so
everything else about each seed is its ``clean`` scenario. A curve is the fraction of seeds
that survive (``metrics.survived``: the episode ran to ``max_steps``) at each impulse level;
``impulse_at_50`` is where it first falls through 50%.

Submissions are loaded and validated once here and run by reference in one evaluation pool
(``topple.pool``), like ``topple compare``.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final

from topple.disturbances import MAX_PUSH_IMPULSE_NS, Push
from topple.evaluate import EvalProtocol
from topple.pool import eval_pool, pool_size, submission_ref, submit_episodes
from topple.results import MAX_EPISODES, MAX_ID_CHARS, sanitize_id
from topple.rng import MAX_SEED

if TYPE_CHECKING:
    from topple.loader import Submission
    from topple.results import EpisodeResult

SHOVE_PROTOCOL: Final = "balance-v1/clean"
PUSH_TIME_S: Final = 2.0
PULSE_S: Final = 0.1
LATERAL_RATIO: Final = 0.0
DEFAULT_IMPULSES: Final = "0.30:0.65:0.025"
MAX_IMPULSES: Final = 200
DIRECTIONS: Final = {1: "forward", -1: "backward"}
CHUNK_EPISODES = 25
"""Episodes per pool task (as in ``topple.compare``); results never depend on it."""
_DECIMAL: Final = r"[0-9]{1,3}(?:\.[0-9]{1,6})?"


@dataclass(frozen=True)
class ShoveCurve:
    """Survival of one submission against one push direction: ``survived[k][s]`` is whether
    seed ``seeds[s]`` survived impulse ``impulses[k]`` (N·s, strictly increasing).
    ``protocol_hash`` is the ``clean`` protocol's over ``seeds``; the push is the same for
    every curve (``PUSH_TIME_S``, ``PULSE_S``, no lateral part)."""

    id: str
    sha256: str
    policy_sha256: str | None
    protocol: str
    protocol_hash: str
    forward_sign: int
    impulses: tuple[float, ...]
    seeds: tuple[int, ...]
    survived: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        if type(self.forward_sign) is not int or self.forward_sign not in DIRECTIONS:
            raise ValueError(f"forward_sign must be 1 or -1, got {self.forward_sign!r}")
        if not self.impulses or any(b <= a for a, b in itertools.pairwise(self.impulses)):
            raise ValueError("impulses must be non-empty and strictly increasing")
        if not self.seeds:
            raise ValueError("a curve needs at least one seed")
        if len(self.survived) != len(self.impulses) or any(
            len(row) != len(self.seeds) for row in self.survived
        ):
            raise ValueError("survived must hold one row per impulse and one value per seed")

    @property
    def direction(self) -> str:
        return DIRECTIONS[self.forward_sign]

    @property
    def survival(self) -> tuple[float, ...]:
        """The fraction of seeds that survived, per impulse level."""
        return tuple(sum(row) / len(row) for row in self.survived)


def impulse_at_50(curve: ShoveCurve) -> float | None:
    """The impulse at which survival first falls through 50%: for the first pair of adjacent
    levels whose survival goes from at least 0.5 to below 0.5, linearly interpolated on
    impulse between them. A level at exactly 50% followed by one below is the crossing itself,
    so a plateau at exactly 50% crosses at its last level (``[1.0, 0.5, 0.5, 0.0]`` gives the
    third impulse). ``None`` if no such pair exists: survival never goes from 50% or more to
    below 50% between adjacent levels."""
    s, x = curve.survival, curve.impulses
    for k in range(len(s) - 1):
        if s[k] >= 0.5 > s[k + 1]:
            return x[k] + (s[k] - 0.5) / (s[k] - s[k + 1]) * (x[k + 1] - x[k])
    return None


def parse_impulses(text: str) -> tuple[float, ...]:
    """``START:STOP:STEP`` (decimal N·s, ``STOP`` included): ``START + k·STEP`` computed exactly
    in decimal, then rounded once to float, so ``0.30:0.65:0.025`` gives 0.3, 0.325, ..., 0.65
    (15 levels). ``ValueError`` unless STEP > 0, STOP - START is a whole number of steps, there
    are at most ``MAX_IMPULSES`` levels and STOP is at most ``MAX_PUSH_IMPULSE_NS``."""

    usage = (
        f"impulses must be START:STOP:STEP in N·s with 0 <= START <= STOP <= "
        f"{MAX_PUSH_IMPULSE_NS:g}, STEP > 0 dividing STOP - START, at most {MAX_IMPULSES} levels"
    )
    match = re.fullmatch(f"({_DECIMAL}):({_DECIMAL}):({_DECIMAL})", text)
    if match is None:
        raise ValueError(f"{usage}; got {text!r}")
    try:
        start, stop, step = (Decimal(part) for part in match.groups())
    except InvalidOperation as exc:  # unreachable for the pattern above
        raise ValueError(f"{usage}; got {text!r}") from exc
    if step <= 0 or stop < start or stop > Decimal(repr(MAX_PUSH_IMPULSE_NS)):
        raise ValueError(f"{usage}; got {text!r}")
    count = (stop - start) / step
    if count != count.to_integral_value() or count + 1 > MAX_IMPULSES:
        raise ValueError(f"{usage}; got {text!r}")
    return tuple(float(start + k * step) for k in range(int(count) + 1))


def shove_push(impulse: float, forward_sign: int, protocol: EvalProtocol) -> Push:
    """The bench's push for ``protocol``'s timing: at ``PUSH_TIME_S`` on the control grid, a
    ``PULSE_S`` rectangular pulse of ``impulse`` N·s along ``forward_sign`` times the heading,
    no lateral part. The force is ``impulse / (n_substeps · timestep)``, as the ``standard``
    preset draws it (``disturbances._draw_pushes``)."""
    timestep = _physics_timestep(protocol.challenge.id)
    n_substeps = round(PULSE_S / timestep)
    return Push(
        start_step=round(PUSH_TIME_S / protocol.challenge.config.control_dt),
        n_substeps=n_substeps,
        magnitude_n=impulse / (n_substeps * timestep),
        forward_sign=float(forward_sign),
        lateral_ratio=LATERAL_RATIO,
    )


@lru_cache(maxsize=8)
def _physics_timestep(challenge_id: str) -> float:
    """The challenge's MuJoCo timestep (its scene's ``<option timestep>``)."""
    from topple.challenge import load_challenge
    from topple.robot import build_model

    ch = load_challenge(challenge_id)
    return float(build_model(ch.robot_xml, ch.scene_xml, ch.params).opt.timestep)


def shove_curve(
    sub: Path,
    impulses: Iterable[float],
    seeds: Iterable[int],
    *,
    forward_sign: int = 1,
    workers: int = 4,
) -> ShoveCurve:
    """The survival curve of ``sub`` against pushes of ``impulses`` (N·s, increasing) along
    ``forward_sign`` on the ``clean`` seeds ``seeds``. ``LoadError`` if ``sub`` does not load;
    ``ValueError`` for bad arguments, before anything is loaded or run."""
    _check_sign(forward_sign)
    ((curve,),) = run_shoves([sub], impulses, seeds, (forward_sign,), workers)
    return curve


def shove_curves(
    sub: Path,
    impulses: Iterable[float],
    seeds: Iterable[int],
    *,
    forward_signs: Sequence[int] = (1, -1),
    workers: int = 4,
) -> tuple[ShoveCurve, ...]:
    """``shove_curve`` for each of ``forward_signs`` (default: both directions), in one pool."""
    for sign in forward_signs:
        _check_sign(sign)
    (curves,) = run_shoves([sub], impulses, seeds, tuple(forward_signs), workers)
    return curves


def run_shoves(
    subs: Sequence[Path],
    impulses: Iterable[float],
    seeds: Iterable[int],
    signs: Sequence[int],
    workers: int,
) -> list[tuple[ShoveCurve, ...]]:
    """Every submission's curve per sign of ``signs``, in one evaluation pool. Arguments are
    checked and every submission is loaded before the pool starts."""
    levels = _checked_impulses(impulses)
    seed_list = _checked_seeds(seeds)
    for sign in signs:
        _check_sign(sign)
    pool_size(workers)
    protocol = replace(EvalProtocol.from_id(SHOVE_PROTOCOL), seeds=seed_list)
    loaded = [submission_ref(sub, challenge=protocol.challenge) for sub in subs]
    pushes = {(sign, x): shove_push(x, sign, protocol) for sign in signs for x in levels}
    chunks = [
        range(start, min(start + CHUNK_EPISODES, len(seed_list)))
        for start in range(0, len(seed_list), CHUNK_EPISODES)
    ]
    pool = eval_pool(workers)
    try:
        futures = [
            [
                [
                    [
                        submit_episodes(pool, ref, protocol, c, pushes=(pushes[sign, x],))
                        for c in chunks
                    ]
                    for x in levels
                ]
                for sign in signs
            ]
            for _, ref in loaded
        ]
        return [
            tuple(
                _curve(sub, protocol, sign, levels, by_level)
                for sign, by_level in zip(signs, per_sign, strict=True)
            )
            for (sub, _), per_sign in zip(loaded, futures, strict=True)
        ]
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _curve(
    sub: Submission,
    protocol: EvalProtocol,
    sign: int,
    levels: tuple[float, ...],
    by_level: Sequence[Sequence[Future[list[EpisodeResult]]]],
) -> ShoveCurve:
    survived = []
    for futures in by_level:
        episodes = list(itertools.chain.from_iterable(f.result() for f in futures))
        if [e.index for e in episodes] != list(range(len(protocol.seeds))):
            raise RuntimeError("the pool returned episodes out of seed order")
        survived.append(tuple(bool(e.metrics and e.metrics["survived"]) for e in episodes))
    return ShoveCurve(
        id=sanitize_id(sub.id or "unknown")[:MAX_ID_CHARS],
        sha256=sub.sha256,
        policy_sha256=sub.policy_sha256,
        protocol=protocol.id,
        protocol_hash=protocol.protocol_hash(),
        forward_sign=sign,
        impulses=levels,
        seeds=tuple(protocol.seeds),
        survived=tuple(survived),
    )


def _check_sign(sign: object) -> None:
    if type(sign) is not int or sign not in DIRECTIONS:
        raise ValueError(f"forward_sign must be the int 1 or -1, got {sign!r}")


def _checked_impulses(impulses: Iterable[float]) -> tuple[float, ...]:
    """At most ``MAX_IMPULSES`` levels (read once, one past the cap at most), each a finite
    number in [0, ``MAX_PUSH_IMPULSE_NS``], strictly increasing."""
    levels = tuple(itertools.islice(impulses, MAX_IMPULSES + 1))
    if not 1 <= len(levels) <= MAX_IMPULSES:
        raise ValueError(f"a bench runs 1 to {MAX_IMPULSES} impulse levels")
    for x in levels:
        if isinstance(x, bool) or not isinstance(x, int | float):
            raise ValueError(f"impulse levels must be numbers, got {x!r}")
        if not (math.isfinite(x) and 0.0 <= x <= MAX_PUSH_IMPULSE_NS):
            raise ValueError(f"impulse levels must lie in [0, {MAX_PUSH_IMPULSE_NS:g}] N·s")
    if any(b <= a for a, b in itertools.pairwise(levels)):
        raise ValueError("impulse levels must be strictly increasing")
    return tuple(float(x) for x in levels)


def _checked_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    """At most ``MAX_EPISODES`` seeds (a protocol's limit, read once), each an int in
    [0, 2**53)."""
    seed_list = tuple(itertools.islice(seeds, MAX_EPISODES + 1))
    if not 1 <= len(seed_list) <= MAX_EPISODES:
        raise ValueError(f"a bench runs 1 to {MAX_EPISODES} seeds")
    if not all(type(s) is int and 0 <= s < MAX_SEED for s in seed_list):
        raise ValueError("seeds must be ints in [0, 2**53)")
    return seed_list


def to_markdown(curves: Sequence[ShoveCurve]) -> str:
    """One submission's curves (one per direction, same levels and seeds) as a Markdown table
    of survival per impulse, then each direction's ``impulse_at_50`` and their minimum. The id
    is escaped (every ASCII punctuation character backslash-escaped)."""
    from topple.compare import contiguous, escape_markdown

    first = curves[0]
    seeds = first.seeds
    span = f"{seeds[0]}..{seeds[-1]}" if contiguous(seeds) else ""
    lines = [
        f"## {escape_markdown(first.id)}: survival vs push impulse",
        "",
        f"{first.protocol}, {len(seeds)} seeds {span}, one push at {PUSH_TIME_S:g} s "
        f"({PULSE_S:g} s pulse, no lateral part)",
        "",
        "| Impulse (N·s) | " + " | ".join(c.direction for c in curves) + " |",
        "|---" * (len(curves) + 1) + "|",
    ]
    for k, x in enumerate(first.impulses):
        cells = " | ".join(f"{c.survival[k]:.2f}" for c in curves)
        lines.append(f"| {x:.3f} | {cells} |")
    i50 = [impulse_at_50(c) for c in curves]
    parts = [f"{c.direction} {_ns(v)}" for c, v in zip(curves, i50, strict=True)]
    if len(curves) > 1:
        found = [v for v in i50 if v is not None]
        parts.append(f"min {_ns(min(found)) if len(found) == len(i50) else 'none'}")
    lines += ["", "i50: " + ", ".join(parts)]
    return "\n".join(lines) + "\n"


def _ns(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f} N·s"
