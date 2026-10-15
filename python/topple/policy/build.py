"""Build a first-party policy's ONNX graph without torch (docs/design/phase-2.md §3.2).

``torch.onnx`` is not used: its default exporter needs onnxscript, which is not in the lock, and
the legacy one is deprecated. The graph is assembled with ``onnx.helper`` from the actor's
weights and the observation statistics, so export is pickle-free and the bytes are canonical as
built (§3.4): two builds from the same inputs are byte-identical.

PPO (9 nodes): ``Sub(obs, mean) → Mul(inv_std) → Clip(lo, hi) → Gemm → Tanh → Gemm → Tanh →
Gemm → Clip(-1, 1)``. SAC (10 nodes): the same with ``Relu`` hidden layers and ``Gemm → Tanh →
Clip`` as the head. Normalization is SB3's ``normalize_obs``, ``z = clip((x - mean) /
sqrt(var + eps), -clip, clip)``, with its constants computed in float64 and rounded to float32
once; the head bounds come from the challenge's action spec, never from the training action
space (Box(-2, 2) under the action-excess penalty).
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt
import onnx
from onnx import TensorProto, helper

from topple.contract import ActionSpec, TaskSpec
from topple.policy.validate import (
    BATCH,
    CANONICAL_IR,
    GRAPH_NAME,
    INPUT_NAME,
    MAX_METADATA_VALUE_BYTES,
    METADATA_FIELDS,
    METADATA_KEYS,
    OUTPUT_NAME,
    POLICY_FORMAT,
    PRODUCER,
)

ActionRule = Literal["clip(mean)", "tanh(mean)"]
Activation = Literal["tanh", "relu"]
ACTION_RULES: tuple[str, ...] = get_args(ActionRule)
OPSET = 17
TINY_STD = 1e-6  # warn when sqrt(var + eps) < TINY_STD * max(1, |mean|)
_HIDDEN_OP: Mapping[str, str] = {"tanh": "Tanh", "relu": "Relu"}

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ObsNorm:
    """SB3 ``VecNormalize`` observation statistics (``obs_rms``, ``epsilon``, ``clip_obs``)."""

    mean: F64
    var: F64
    epsilon: float
    clip: float


@dataclass(frozen=True)
class MlpWeights:
    """The actor's (weight ``[out, in]``, bias ``[out]``) float32 pairs: two hidden layers, then
    the head producing the action mean."""

    layers: tuple[tuple[F32, F32], ...]
    activation: Activation


@dataclass(frozen=True)
class _NormConstants:
    """The float32 constants the graph holds for the normalization."""

    mean: F32
    inv_std: F32
    clip: np.float32


@dataclass
class _Graph:
    """Nodes and initializers in creation order, named as §3.2 says: node i is ``n{i}`` with
    output ``v{i}``, initializer j is ``c{j}``; calling ``constant`` inside a ``node`` call's
    argument list numbers initializers in first-use order."""

    nodes: list[onnx.NodeProto] = field(default_factory=list)
    initializers: list[TensorProto] = field(default_factory=list)

    def constant(self, values: npt.ArrayLike) -> str:
        name = f"c{len(self.initializers)}"
        arr = np.asarray(values, dtype="<f4")
        raw = arr.tobytes()
        self.initializers.append(
            helper.make_tensor(name, TensorProto.FLOAT, arr.shape, raw, raw=True)
        )
        return name

    def node(self, op: str, inputs: list[str], **attributes: Any) -> str:
        i = len(self.nodes)
        self.nodes.append(helper.make_node(op, inputs, [f"v{i}"], name=f"n{i}", **attributes))
        return f"v{i}"


def policy_metadata(spec: TaskSpec, action_rule: str) -> dict[str, str]:
    """The 7 ``topple.*`` metadata values for a policy of ``spec``: each manifest field as
    ``json.dumps`` (ASCII), so the metadata is fully derived from the manifest."""
    fields: dict[str, object] = {
        "format": POLICY_FORMAT,
        "challenge": spec.challenge,
        "obs_schema": spec.obs_schema,
        "obs_features": list(spec.obs_features),
        "action_spec": spec.action.id,
        "control_dt": spec.control_dt,
        "action_rule": action_rule,
    }
    return {f"topple.{name}": json.dumps(fields[name]) for name in METADATA_FIELDS}


def build_policy_onnx(
    w: MlpWeights,
    norm: ObsNorm,
    *,
    action_rule: ActionRule,
    metadata: Mapping[str, str],
    action: ActionSpec,
) -> bytes:
    """The canonical ONNX bytes of the policy ``w`` behind ``norm``, with head ``action_rule``
    clipped to ``action``'s bounds (the challenge's ``TaskSpec.action``) and the 7 metadata
    values (see ``policy_metadata``). ``ValueError`` for inconsistent inputs.

    IR ``CANONICAL_IR[17]`` = 8, opset 17, weights inline, ``producer_name`` only (never
    ``producer_version`` or ``doc_string``), deterministic serialization.
    """
    layers = _checked_layers(w, action)
    obs_dim = layers[0][0].shape[1]
    constants = _norm_constants(norm, obs_dim)
    rule = _checked_rule(action_rule)
    _check_metadata(metadata, rule, action, obs_dim)
    _warn_tiny_std(norm)
    g = _Graph()
    x = g.node("Sub", [INPUT_NAME, g.constant(constants.mean)])
    x = g.node("Mul", [x, g.constant(constants.inv_std)])
    x = g.node("Clip", [x, g.constant(-constants.clip), g.constant(constants.clip)])
    for weight, bias in layers[:-1]:
        x = g.node("Gemm", [x, g.constant(weight), g.constant(bias)], transB=1)
        x = g.node(_HIDDEN_OP[w.activation], [x])
    weight, bias = layers[-1]
    x = g.node("Gemm", [x, g.constant(weight), g.constant(bias)], transB=1)
    if rule == "tanh(mean)":
        x = g.node("Tanh", [x])
    low, high = _action_bounds(action)
    g.node("Clip", [x, g.constant(low), g.constant(high)])
    g.nodes[-1].output[0] = OUTPUT_NAME
    graph = helper.make_graph(
        g.nodes,
        GRAPH_NAME,
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [BATCH, obs_dim])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [BATCH, action.dim])],
        g.initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=CANONICAL_IR[OPSET],
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name=PRODUCER,
    )
    for key in sorted(metadata):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = metadata[key]
    data: bytes = model.SerializeToString(deterministic=True)
    return data


def reference_actions(
    w: MlpWeights,
    norm: ObsNorm,
    action_rule: str,
    obs: npt.NDArray[np.float32],
    *,
    action: ActionSpec,
) -> F64:
    """What the built graph computes for float32 rows ``obs`` ``[N, D]``: the graph's float32
    constants widened to float64, then float64 arithmetic (the parity reference, §3.3)."""
    layers = _checked_layers(w, action)
    obs_dim = layers[0][0].shape[1]
    constants = _norm_constants(norm, obs_dim)
    rule = _checked_rule(action_rule)
    if not isinstance(obs, np.ndarray) or obs.dtype != np.float32 or obs.ndim != 2:
        raise ValueError("obs must be a float32 array of rows [N, D]")
    if obs.shape[1] != obs_dim:
        raise ValueError(f"obs rows have {obs.shape[1]} features, the policy takes {obs_dim}")
    clip = float(constants.clip)
    centered = obs.astype(np.float64) - constants.mean.astype(np.float64)
    h = np.clip(centered * constants.inv_std.astype(np.float64), -clip, clip)
    for weight, bias in layers[:-1]:
        h = _activate(w.activation, h @ weight.astype(np.float64).T + bias.astype(np.float64))
    weight, bias = layers[-1]
    out = h @ weight.astype(np.float64).T + bias.astype(np.float64)
    if rule == "tanh(mean)":
        out = np.tanh(out)
    low, high = _action_bounds(action)
    result: F64 = np.clip(out, float(low), float(high))
    return result


def _activate(activation: str, x: F64) -> F64:
    out: F64 = np.tanh(x) if activation == "tanh" else np.maximum(x, 0.0)
    return out


def _checked_rule(action_rule: str) -> ActionRule:
    if action_rule == "clip(mean)":
        return "clip(mean)"
    if action_rule == "tanh(mean)":
        return "tanh(mean)"
    raise ValueError(f"action_rule must be one of {ACTION_RULES}")


def _checked_layers(w: MlpWeights, action: ActionSpec) -> tuple[tuple[F32, F32], ...]:
    """``w``'s layers if they are two hidden layers and a head of matching float32 shapes,
    finite, with as many outputs as ``action`` has dimensions; else ``ValueError``."""
    if w.activation not in _HIDDEN_OP:
        raise ValueError(f"activation must be one of {tuple(_HIDDEN_OP)}")
    if len(w.layers) != 3:
        raise ValueError("expected two hidden layers and a head")
    width: int | None = None
    for k, (weight, bias) in enumerate(w.layers):
        for arr in (weight, bias):
            if not isinstance(arr, np.ndarray) or arr.dtype != np.float32:
                raise ValueError(f"layer {k}: weights must be float32 arrays")
            if not np.isfinite(arr).all():
                raise ValueError(f"layer {k}: weights must be finite")
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(f"layer {k}: weight [out, in] and bias [out] do not match")
        if width is not None and weight.shape[1] != width:
            raise ValueError(f"layer {k} takes {weight.shape[1]} inputs, not {width}")
        width = weight.shape[0]
    if width != action.dim:
        raise ValueError(f"the head has {width} outputs; {action.id} has {action.dim}")
    return w.layers


def _norm_constants(norm: ObsNorm, obs_dim: int) -> _NormConstants:
    """``f32(mean)``, ``f32(1 / sqrt(var + eps))`` and ``f32(clip)``, each computed in float64
    and rounded once; ``ValueError`` for statistics SB3 could not have produced."""
    mean = np.asarray(norm.mean, dtype=np.float64)
    var = np.asarray(norm.var, dtype=np.float64)
    if mean.shape != (obs_dim,) or var.shape != (obs_dim,):
        raise ValueError(f"mean and var must have {obs_dim} entries, one per feature")
    if not (np.isfinite(mean).all() and np.isfinite(var).all() and (var >= 0).all()):
        raise ValueError("mean and var must be finite, var non-negative")
    if not (math.isfinite(norm.epsilon) and norm.epsilon > 0):
        raise ValueError("epsilon must be finite and positive")
    clip = np.float32(norm.clip)
    if not (np.isfinite(clip) and clip > 0):
        raise ValueError("clip must be finite and positive in float32")
    inv_std = 1.0 / np.sqrt(var + norm.epsilon)
    return _NormConstants(mean.astype(np.float32), inv_std.astype(np.float32), clip)


def _action_bounds(action: ActionSpec) -> tuple[np.float32, np.float32]:
    return np.float32(action.low), np.float32(action.high)


def _check_metadata(
    metadata: Mapping[str, str], rule: str, action: ActionSpec, obs_dim: int
) -> None:
    """``ValueError`` unless ``metadata`` holds exactly the 7 keys with short ASCII JSON values
    that agree with what is being built (format, action rule, action spec, feature count)."""
    if set(metadata) != set(METADATA_KEYS):
        raise ValueError(f"metadata must have exactly the keys {METADATA_KEYS}")
    values: dict[str, object] = {}
    for key in METADATA_KEYS:
        value = metadata[key]
        if not value.isascii() or len(value) > MAX_METADATA_VALUE_BYTES:
            raise ValueError(f"{key} must be ASCII and at most 1024 bytes")
        values[key] = json.loads(value)
    if values["topple.format"] != POLICY_FORMAT:
        raise ValueError(f"topple.format must be {POLICY_FORMAT!r}")
    if values["topple.action_rule"] != rule:
        raise ValueError(f"topple.action_rule must be {rule!r}")
    if values["topple.action_spec"] != action.id:
        raise ValueError(f"topple.action_spec must be {action.id!r}")
    features = values["topple.obs_features"]
    if not isinstance(features, list) or len(features) != obs_dim:
        raise ValueError(f"topple.obs_features must list {obs_dim} features")


def _warn_tiny_std(norm: ObsNorm) -> None:
    """Warn on stderr for each feature whose ``sqrt(var + eps)`` is below ``TINY_STD ·
    max(1, |mean|)``: its normalized value turns float32 rounding of the input into noise."""
    mean = np.asarray(norm.mean, dtype=np.float64)
    std = np.sqrt(np.asarray(norm.var, dtype=np.float64) + norm.epsilon)
    bar = TINY_STD * np.maximum(1.0, np.abs(mean))
    for i in np.flatnonzero(std < bar):
        print(
            f"warning: feature {i} has std {std[i]:.3g} below {bar[i]:.3g} "
            "(1e-6 * max(1, |mean|)); its normalized value is mostly float32 rounding",
            file=sys.stderr,
        )
