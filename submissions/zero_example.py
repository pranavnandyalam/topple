"""Lesson 1: the smallest valid controller. It never drives the wheels, so the robot falls over.

A controller is any object with ``reset(rng_seed)`` and ``act(obs)``. ``act`` runs 100 times a
second and returns the (left, right) motor commands in [-1, 1]. Watch this one topple with
``topple run submissions/zero_example.py``, then make ``act`` drive the wheels toward the lean.
"""

from __future__ import annotations

from topple.contract import BalancerObs, TaskSpec


class ZeroController:
    def reset(self, rng_seed: int) -> None:
        """Called once at the start of every episode; nothing to prepare here."""

    def act(self, obs: BalancerObs) -> tuple[float, float]:
        return (0.0, 0.0)


def make_controller(spec: TaskSpec | None = None) -> ZeroController:
    return ZeroController()
