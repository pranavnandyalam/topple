"""Named, append-only random streams (docs/contract.md §9).

Every source of randomness in an episode draws from its own stream, derived from the episode seed
with ``SeedSequence(seed, spawn_key=(agent, stream))``. Adding a stream never shifts another, and
consuming one stream never changes the draws of another.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

MAX_SEED = 2**53  # seeds must round-trip exactly through JSON / JavaScript numbers


class Stream(IntEnum):
    """Stream ids. Append only: never reorder or reuse a value."""

    INIT = 0
    PUSHES = 1
    SENSOR = 2
    MODEL = 3
    DELAY = 4
    CONTROLLER = 5
    TARGETS = 6  # waypoint courses (topple.course)


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < MAX_SEED:
        raise ValueError(f"seed must be an int in [0, 2**53), got {seed!r}")
    return seed


def stream_rng(episode_seed: int, stream: Stream, agent: int = 0) -> np.random.Generator:
    validate_seed(episode_seed)
    return np.random.default_rng(
        np.random.SeedSequence(episode_seed, spawn_key=(agent, int(stream)))
    )


def controller_seed(episode_seed: int, salt: int = 0, agent: int = 0) -> int:
    """Controller-private seed: carries no usable information about the episode seed.

    With a secret ``salt`` (private protocols) it cannot be inverted to recover the seed.
    """
    validate_seed(episode_seed)
    seq = np.random.SeedSequence(
        entropy=(episode_seed, salt), spawn_key=(agent, int(Stream.CONTROLLER))
    )
    return int(seq.generate_state(1, np.uint64)[0]) >> 11
