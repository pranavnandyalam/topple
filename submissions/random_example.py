"""Uniformly random motor commands: the floor any real controller should beat.

It also shows the randomness rule: draw only from the ``rng_seed`` handed to ``reset`` (never the
global ``random`` module or the clock), so a seeded episode replays exactly.
"""

from __future__ import annotations

import random

from topple.contract import BalancerObs, TaskSpec


class RandomController:
    def __init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, rng_seed: int) -> None:
        self._rng = random.Random(rng_seed)

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        return (self._rng.uniform(-1.0, 1.0), self._rng.uniform(-1.0, 1.0))


def make_controller(spec: TaskSpec) -> RandomController:
    return RandomController()
