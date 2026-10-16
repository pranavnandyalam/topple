"""``ONNXPolicy``: a ``topple.policy/1`` policy as a Controller (docs/design/phase-2.md §3.5).

The policy is a pure function of the current observation, so ``reset`` does nothing and one
session serves every episode. ``act`` feeds the observation's float32 feature row (what the
env emits and the trajectory stores, contract §6) at batch 1, checks that the output is float32
``(1, action_dim)`` and returns Python floats; the harness validates them like any controller's
action (a NaN from a runtime that propagates one is ``invalid_action``). Non-finite
observations never get here: they are outside the policy domain (contract §17) and the harness
raises before calling ``act``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NoReturn

import numpy as np

from topple.contract import BalancerObs, TaskSpec
from topple.policy.validate import INPUT_NAME, OUTPUT_NAME

if TYPE_CHECKING:
    import onnxruntime as ort

    from topple.policy.runtime import LoadedPolicy


class ONNXPolicy:
    """A Controller over an ONNX Runtime session (from ``make_session``) mapping ``obs_dim``
    features to ``action_dim`` actions."""

    def __init__(self, session: ort.InferenceSession, obs_dim: int, action_dim: int) -> None:
        self._session = session
        self._obs_dim = obs_dim
        self._action_dim = action_dim

    def reset(self, rng_seed: int) -> None:
        """Nothing to reset: the policy is stateless and deterministic."""

    def act(self, obs: BalancerObs) -> tuple[float, ...]:
        row = obs.to_array()[None]
        if row.shape != (1, self._obs_dim):
            raise ValueError(
                f"the policy takes {self._obs_dim} features, the observation has {row.shape[1]}"
            )
        outputs = self._session.run([OUTPUT_NAME], {INPUT_NAME: row})
        out = outputs[0] if isinstance(outputs, list) and len(outputs) == 1 else None
        if (
            not isinstance(out, np.ndarray)
            or out.dtype != np.float32
            or out.shape != (1, self._action_dim)
        ):
            raise RuntimeError(f"the policy must return float32 (1, {self._action_dim})")
        return tuple(float(x) for x in out[0])

    def __reduce__(self) -> NoReturn:
        raise TypeError("an ONNXPolicy is never pickled: send a topple.pool.PolicyRef")


def policy_factory(loaded: LoadedPolicy) -> Callable[[TaskSpec], ONNXPolicy]:
    """A ``make_controller`` for ``loaded``: one ``ONNXPolicy`` per episode over its single
    session. A task with other dims than the policy's is refused (the episode records the
    error) rather than fed rows of the wrong width."""
    session = loaded.session
    obs_dim, action_dim = loaded.validated.obs_dim, loaded.validated.action_dim

    def make_controller(spec: TaskSpec) -> ONNXPolicy:
        if len(spec.obs_features) != obs_dim or spec.action.dim != action_dim:
            raise ValueError(
                f"the policy maps {obs_dim} features to {action_dim} actions; the task has "
                f"{len(spec.obs_features)} and {spec.action.dim}"
            )
        return ONNXPolicy(session, obs_dim, action_dim)

    return make_controller
