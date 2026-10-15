"""Shared builders for the policy tests (docs/design/phase-2.md §3.2-§3.5).

Weights come from integer arithmetic and one correctly rounded division, so the tiny fixtures
are bit-identical on every platform and numpy version (no RNG stream, no libm call), which lets
the golden tests pin the sha256 of the bytes built from them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
from onnx import TensorProto, helper

from topple.challenge import load_challenge
from topple.contract import ActionSpec, TaskSpec
from topple.policy.build import MlpWeights, ObsNorm, build_policy_onnx, policy_metadata
from topple.policy.runtime import make_session
from topple.policy.validate import (
    CANONICAL_IR,
    CHECK_OF,
    ValidatedPolicy,
    canonicalize,
    validate_policy_bytes,
)

D = 10  # balance-v1 obs width
A = 2  # balance-v1 action width
F32 = np.float32


def balance_spec() -> TaskSpec:
    return load_challenge("balance-v1").task_spec(wheel_radius_m=0.05, track_width_m=0.2)


def det64(shape: Sequence[int], salt: int) -> npt.NDArray[np.float64]:
    """Values in [-1, 1] from integer arithmetic and one division: the same on every platform."""
    i = np.arange(prod(shape), dtype=np.int64)
    v = ((i * 7919 + salt * 104729) % 2003 - 1001).astype(np.float64) / 1001.0
    return v.reshape(tuple(shape))


def det(shape: Sequence[int], salt: int, scale: float = 1.0) -> npt.NDArray[np.float32]:
    return (det64(shape, salt) * scale).astype(F32)


def tiny_weights(
    activation: str = "tanh",
    hidden: tuple[int, int] = (8, 8),
    d: int = D,
    a: int = A,
    head_scale: float = 3.0,
) -> MlpWeights:
    h1, h2 = hidden
    # scaled by 1/sqrt(fan_in) like a trained MLP, so pre-activations stay O(1) at any width
    layers = (
        (det((h1, d), 1, 1.5 / d**0.5), det((h1,), 2, 0.1)),
        (det((h2, h1), 3, 1.5 / h1**0.5), det((h2,), 4, 0.1)),
        (det((a, h2), 5, head_scale / h2**0.5), det((a,), 6, 0.2)),
    )
    return MlpWeights(layers=layers, activation=activation)  # type: ignore[arg-type]


def tiny_norm(d: int = D, clip: float = 10.0, epsilon: float = 1e-8) -> ObsNorm:
    mean = det64((d,), 7) * 0.5 + 0.75  # positive, so "mean and var swapped" stays a valid var
    var = (det64((d,), 8) * 0.4 + 0.6) ** 2
    return ObsNorm(mean=mean, var=var, epsilon=epsilon, clip=clip)


RULE_OF = {"tanh": "clip(mean)", "relu": "tanh(mean)"}  # PPO and SAC


def build(
    weights: MlpWeights | None = None,
    norm: ObsNorm | None = None,
    *,
    action_rule: str | None = None,
    metadata: Mapping[str, str] | None = None,
    action: ActionSpec | None = None,
) -> bytes:
    """``build_policy_onnx`` with balance-v1's metadata and action spec unless given."""
    weights = weights or tiny_weights()
    rule = action_rule or RULE_OF[weights.activation]
    spec = balance_spec()
    return build_policy_onnx(
        weights,
        norm or tiny_norm(),
        action_rule=rule,  # type: ignore[arg-type]
        metadata=policy_metadata(spec, rule) if metadata is None else metadata,
        action=action or spec.action,
    )


def ppo_bytes() -> bytes:
    return build(tiny_weights("tanh"))


def sac_bytes() -> bytes:
    return build(tiny_weights("relu"))


def fixture_rows(norm: ObsNorm | None = None, n: int = 96) -> npt.NDArray[np.float32]:
    """Float32 rows spread to ±14 standard deviations: some inside the normalization clip,
    some between 5 and 10, some beyond it; with ``tiny_weights`` some actions saturate."""
    norm = norm or tiny_norm()
    d = norm.mean.shape[0]
    k = det64((n, d), 11) * 14.0
    return (norm.mean + np.sqrt(norm.var + norm.epsilon) * k).astype(F32)


def parse(data: bytes) -> onnx.ModelProto:
    proto: onnx.ModelProto = onnx.ModelProto.FromString(data)
    return proto


def serialize(proto: onnx.ModelProto) -> bytes:
    data: bytes = proto.SerializeToString(deterministic=True)
    return data


def validated(data: bytes, obs_dim: int = D, action_dim: int = A) -> ValidatedPolicy:
    result = validate_policy_bytes(data, obs_dim, action_dim)
    assert isinstance(result, ValidatedPolicy), result
    return result


def run(data: bytes, rows: npt.NDArray[np.float32], action_dim: int = A) -> npt.NDArray[Any]:
    """Actions for ``rows`` through the real runtime (validation, then ``make_session``)."""
    session = make_session(validated(data, rows.shape[1], action_dim))
    out: npt.NDArray[Any] = session.run(None, {"obs": rows})[0]
    return out


def rejected_at(result: object) -> tuple[int, str]:
    """(check, code) of the first message of a rejected validation."""
    assert isinstance(result, list), result
    assert result, "an empty rejection list"
    code = result[0].split(":", 1)[0]
    return CHECK_OF[code], code


def tensor(name: str, values: object, *, raw: bool = True) -> TensorProto:
    arr = np.asarray(values, dtype="<f4")
    if raw:
        return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.tobytes(), raw=True)
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.reshape(-1).tolist())


def value_info(name: str, dims: Sequence[str | int]) -> onnx.ValueInfoProto:
    info: onnx.ValueInfoProto = helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
    return info


def graph_model(
    nodes: Sequence[onnx.NodeProto],
    initializers: Sequence[TensorProto] = (),
    *,
    obs_dim: int = D,
    action_dim: int = A,
    opset: int = 17,
    ir: int | None = None,
    metadata: Mapping[str, str] | None = None,
) -> onnx.ModelProto:
    """A model with a valid header, I/O ``obs``/``action`` and the 7 metadata keys around
    ``nodes``; the node and value names are the caller's (canonicalize to get §3.2 names)."""
    graph = helper.make_graph(
        list(nodes),
        "topple_policy",
        [value_info("obs", ["batch", obs_dim])],
        [value_info("action", ["batch", action_dim])],
        list(initializers),
    )
    model: onnx.ModelProto = helper.make_model(
        graph,
        ir_version=CANONICAL_IR[opset] if ir is None else ir,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="topple",
    )
    entries = metadata if metadata is not None else policy_metadata(balance_spec(), "clip(mean)")
    for key in sorted(entries):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = entries[key]
    return model


def node(op: str, inputs: Sequence[str], output: str, **attrs: Any) -> onnx.NodeProto:
    made: onnx.NodeProto = helper.make_node(
        op, list(inputs), [output], name=f"node_{output}", **attrs
    )
    return made


def chain_model(
    ops: Sequence[str],
    width: int,
    *,
    obs_dim: int = D,
    action_dim: int = A,
    inputs_per_node: int = 1,
) -> onnx.ModelProto:
    """obs -> Gemm(width) -> ops (each fed ``inputs_per_node`` copies of the previous value)
    -> Gemm(action_dim) -> action: a compute chain at a chosen width."""
    nodes = [node("Gemm", ["obs", "w_in"], "h0", transB=1)]
    for i, op in enumerate(ops):
        nodes.append(node(op, [f"h{i}"] * inputs_per_node, f"h{i + 1}"))
    nodes.append(node("Gemm", [f"h{len(ops)}", "w_out"], "action", transB=1))
    inits = [
        tensor("w_in", det((width, obs_dim), 21, 0.1)),
        tensor("w_out", det((action_dim, width), 22, 0.1)),
    ]
    return graph_model(nodes, inits, obs_dim=obs_dim, action_dim=action_dim)


def canonical(proto: onnx.ModelProto, *, pack: bool = False) -> bytes:
    return canonicalize(proto, pack=pack)


# --- topple.policy/1 directories (§3.1) ------------------------------------------------------


def first_party_claims() -> dict[str, Any]:
    """Claims as the exporter writes them (every key present; self-reported)."""
    return {
        "name": "tiny ppo",
        "algo": "ppo",
        "activation": "tanh",
        "net_arch": [8, 8],
        "total_timesteps": 2048,
        "selected_step": 1024,
        "train_seed": 0,
        "train_config_sha256": "ab" * 32,
        "git_commit": "c" * 40,
        "git_dirty": False,
        "selection_eligible": True,
        "versions": {"topple": "0.1.0", "onnxruntime": "1.31.0"},
        "export_parity": {
            "reference": "sb3.predict",
            "n": 1074,
            "n_onpolicy": 10,
            "n_onpolicy_seeds": 2,
            "n_unsaturated": 400,
            "n_saturated": 40,
            "max_abs_diff": 2.4e-7,
            "tolerance": 1e-5,
            "vacuous": False,
        },
    }


def first_party_manifest(
    data: bytes, *, action_rule: str = "clip(mean)", claims: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The ``policy.json`` of balance-v1 policy ``data`` (§3.1), as a JSON-ready dict."""
    spec = balance_spec()
    proto = parse(data)
    dim = [d.dim_value for d in proto.graph.input[0].type.tensor_type.shape.dim][1]
    out = [d.dim_value for d in proto.graph.output[0].type.tensor_type.shape.dim][1]
    return {
        "format": "topple.policy/1",
        "kind": "onnx",
        "challenge": spec.challenge,
        "obs_schema": spec.obs_schema,
        "obs_features": list(spec.obs_features),
        "action_spec": spec.action.id,
        "action_dim": spec.action.dim,
        "control_dt": spec.control_dt,
        "action_rule": action_rule,
        "onnx": {
            "file": "policy.onnx",
            "sha256": hashlib.sha256(data).hexdigest(),
            "ir_version": proto.ir_version,
            "opset": proto.opset_import[0].version,
            "input": {"name": "obs", "dtype": "float32", "shape": ["batch", dim]},
            "output": {"name": "action", "dtype": "float32", "shape": ["batch", out]},
        },
        "claims": dict(first_party_claims() if claims is None else claims),
    }


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2) + "\n").encode("ascii")


def write_policy_dir(
    root: Path,
    data: bytes | None = None,
    *,
    manifest: Mapping[str, Any] | bytes | None = None,
    readme: bytes | None = None,
) -> Path:
    """A ``topple.policy/1`` directory at ``root``: ``policy.onnx`` (default: the tiny PPO
    build), its ``policy.json`` (default: ``first_party_manifest``) and optionally README.md."""
    data = ppo_bytes() if data is None else data
    root.mkdir(parents=True, exist_ok=True)
    (root / "policy.onnx").write_bytes(data)
    if manifest is None:
        manifest = first_party_manifest(data)
    raw = manifest if isinstance(manifest, bytes) else manifest_bytes(manifest)
    (root / "policy.json").write_bytes(raw)
    if readme is not None:
        (root / "README.md").write_bytes(readme)
    return root


def sha_of_files(files: Mapping[str, bytes]) -> str:
    """Contract §8's ``submission_sha256`` recipe over ``files`` (name -> bytes)."""
    digest = hashlib.sha256(b"topple.submission/1\n")
    for name in sorted(files):
        for part in (name.encode("utf-8"), files[name]):
            digest.update(len(part).to_bytes(8, "big") + part)
    return digest.hexdigest()
