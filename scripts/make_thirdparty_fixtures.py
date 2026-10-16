"""Write ``tests/data/thirdparty_legacy.onnx`` and ``tests/data/thirdparty_dynamo.onnx``: one
small balance-v1 policy exported by both of torch's ONNX exporters, as a third party would
export it (docs/design/phase-2.md §3.3 "Third-party recipes", §6 ``test_policy_pack``,
decision log B26).

The network is the §3.2 PPO shape written as a user writes it, not as ``topple export`` builds
it: ``clamp((obs - mean) * scale, -10, 10)``, ``Linear Tanh Linear Tanh Linear`` (10 -> 8 -> 8
-> 2), ``clamp(-1, 1)``, with ``mean`` a ``[D]`` buffer and ``scale`` one Python float. Every
weight comes from integer arithmetic and one division (no RNG stream, no libm), so the weights
are the same on every platform; what the exporters write from them is pinned only by the
committed bytes.

- ``thirdparty_legacy.onnx``: ``torch.onnx.export(dynamo=False)`` (the TorchScript exporter,
  deprecated by torch 2.14 with "will be removed"): IR 9, opset 20, a static batch of 1, five
  rank-0 ``Constant`` nodes interleaved with the compute nodes (the scale and the four Clip
  bounds), parameter names such as ``net.0.weight``. It proves pack folds Constants into
  initializers, turns the static batch into ``batch`` and keeps IR 9 (``CANONICAL_IR[20]``).
- ``thirdparty_dynamo.onnx``: ``torch.onnx.export(dynamo=True)`` (the default exporter; it
  needs onnxscript): IR 10, opset 20, ``Sub Mul Clip Gemm Tanh Gemm Tanh Gemm Clip``, a
  ``value_info`` for every value (20), rank-0 initializers and per-input metadata. It proves
  pack drops ``value_info`` and metadata and lowers IR 10 to 9 (``CANONICAL_IR[20]``). The
  exporter records each node's Python stack trace in the model; its absolute paths are made
  relative (``relative_paths``), so these bytes also depend on this script's line numbers.

Both pack to ``action_rule: other`` (the scale is rank 0, not the §3.2 ``[D]``); the dynamo
export with the scale broadcast to ``[D]`` packs to ``clip(mean)`` (§3.4 structure match).

Before writing anything the script checks each export has those properties, packs it for
balance-v1 (``pack_policy``: canonicalize, checks 1-9, load, smoke test) and compares ONNX
Runtime on the packed model with torch on 64 observations (max |diff| <= 1e-5).

Run once from the repository root, then commit both files (onnxscript is needed only here and
is deliberately not in Topple's lock; the default CI job only reads the committed bytes):

    uv run --extra train --with onnxscript python scripts/make_thirdparty_fixtures.py

``--check`` writes nothing and exits 1 if either committed file differs from a fresh export
(a different torch or onnxscript release may write different bytes; then regenerate, rerun
``tests/test_policy_pack.py`` and commit both together).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from math import prod
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import onnx
import torch
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message
from torch import nn

from topple.challenge import load_challenge
from topple.loader import LoadError
from topple.policy.manifest import policy_task
from topple.policy.pack import pack_policy
from topple.policy.runtime import load_policy_files
from topple.policy.validate import CANONICAL_IR

REPO: Final = Path(__file__).resolve().parents[1]
DATA: Final = REPO / "tests" / "data"
LEGACY: Final = "thirdparty_legacy.onnx"
DYNAMO: Final = "thirdparty_dynamo.onnx"
CHALLENGE: Final = load_challenge("balance-v1")
TASK: Final = policy_task(CHALLENGE)
D: Final = TASK.obs_dim
A: Final = TASK.action_dim
HIDDEN: Final = 8
CLIP_OBS: Final = 10.0
SCALE: Final = 0.5  # one scale for every feature: rank 0 in both exports
OPSET: Final = 20
TOLERANCE: Final = 1e-5
SEQUENCE: Final = ("Sub", "Mul", "Clip", "Gemm", "Tanh", "Gemm", "Tanh", "Gemm", "Clip")
ONNX_FLOAT: Final = 1  # TensorProto.FLOAT


def det(shape: tuple[int, ...], salt: int, scale: float) -> npt.NDArray[np.float32]:
    """Values in [-scale, scale] from integer arithmetic and one division (as
    ``tests/policy_support.det``): the same on every platform and numpy version."""
    i = np.arange(prod(shape), dtype=np.int64)
    v = ((i * 7919 + salt * 104729) % 2003 - 1001).astype(np.float64) / 1001.0
    return (v * scale).astype(np.float32).reshape(shape)


class ThirdPartyPolicy(nn.Module):  # type: ignore[misc, unused-ignore]  # Any without torch
    """The policy a user writes by hand: normalize, clamp, a tanh MLP, clamp the action."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(det((D,), 7, 0.5) + np.float32(0.75)))
        linears = [nn.Linear(D, HIDDEN), nn.Linear(HIDDEN, HIDDEN), nn.Linear(HIDDEN, A)]
        with torch.no_grad():
            for k, linear in enumerate(linears):
                fan_out, fan_in = linear.out_features, linear.in_features
                weight = det((fan_out, fan_in), 2 * k + 1, 1.5 / fan_in**0.5)
                linear.weight.copy_(torch.from_numpy(weight))
                linear.bias.copy_(torch.from_numpy(det((fan_out,), 2 * k + 2, 0.1)))
        self.net = nn.Sequential(linears[0], nn.Tanh(), linears[1], nn.Tanh(), linears[2])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean")
        z = torch.clamp((obs - mean) * SCALE, -CLIP_OBS, CLIP_OBS)
        out: torch.Tensor = torch.clamp(self.net(z), -1.0, 1.0)
        return out


def export_legacy(net: ThirdPartyPolicy) -> bytes:
    """The TorchScript exporter with its defaults and the I/O names a user would pick."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        torch.onnx.export(
            net,
            (torch.zeros(1, D),),
            path,
            dynamo=False,
            opset_version=OPSET,
            input_names=["obs"],
            output_names=["action"],
        )
        return path.read_bytes()


def export_dynamo(net: ThirdPartyPolicy) -> bytes:
    """The §3.3 primary recipe plus ``external_data=False``: torch 2.14.1 otherwise moves the
    80-float first weight into ``model.onnx.data``, which pack refuses (external data)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        torch.onnx.export(
            net,
            (torch.zeros(1, D),),
            path,
            dynamo=True,
            opset_version=OPSET,
            input_names=["obs"],
            output_names=["action"],
            dynamic_shapes=({0: torch.export.Dim("batch")},),
            external_data=False,  # torch 2.14 writes model.onnx.data beside it by default
            verbose=False,
        )
        extra = sorted(p.name for p in Path(tmp).iterdir() if p != path)
        if extra:  # external data would be refused by pack anyway
            raise SystemExit(f"the dynamo exporter wrote more than one file: {extra}")
        return path.read_bytes()


def relative_paths(data: bytes) -> bytes:
    """``data`` with every absolute path prefix in its strings removed: the dynamo exporter
    records each node's Python stack trace (doc strings, ``pkg.torch.onnx.stack_trace``
    metadata) with absolute file names, which would commit this machine's home directory and
    make the bytes depend on where the checkout lives. Rewritten field by field (a byte-level
    replace would break the length prefixes); the structure is unchanged."""
    site = str(Path(torch.__file__).resolve().parents[1]) + "/"
    prefixes = ((site, "site-packages/"), (str(REPO) + "/", ""))
    proto = onnx.ModelProto.FromString(data)
    _rewrite(proto, prefixes)
    out: bytes = proto.SerializeToString(deterministic=True)
    return out


def _rewrite(message: Message, prefixes: tuple[tuple[str, str], ...]) -> None:
    def fix(text: str) -> str:
        for old, new in prefixes:
            text = text.replace(old, new)
        return text

    for field, value in message.ListFields():
        repeated = field.is_repeated  # protobuf 7 has no `label`
        if field.type == FieldDescriptor.TYPE_STRING:
            if repeated:
                value[:] = [fix(v) for v in value]
            else:
                setattr(message, field.name, fix(value))
        elif field.type == FieldDescriptor.TYPE_MESSAGE:
            for child in value if repeated else [value]:
                _rewrite(child, prefixes)


def check_legacy(proto: onnx.ModelProto) -> list[str]:
    ops = [n.op_type for n in proto.graph.node]
    constants = [n for n in proto.graph.node if n.op_type == "Constant"]
    problems = _header(proto, ir=9)
    if [op for op in ops if op != "Constant"] != list(SEQUENCE):
        problems.append(f"compute nodes {ops}")
    if len(constants) != 5 or ops[0] == "Constant" or ops[-1] == "Constant":
        problems.append(f"expected 5 Constants between compute nodes, got {ops}")
    if any(list(n.attribute[0].t.dims) for n in constants):
        problems.append("a Constant is not rank 0")
    dims = proto.graph.input[0].type.tensor_type.shape.dim
    if [d.dim_value for d in dims] != [1, D]:
        problems.append("the legacy input is not a static [1, D]")
    return problems


def check_dynamo(proto: onnx.ModelProto) -> list[str]:
    ops = [n.op_type for n in proto.graph.node]
    problems = _header(proto, ir=10)
    if ops != list(SEQUENCE):
        problems.append(f"nodes {ops}")
    if not proto.graph.value_info:
        problems.append("no value_info")
    rank0 = [t.name for t in proto.graph.initializer if not t.dims]
    if len(rank0) < 5:
        problems.append(f"expected rank-0 scale and Clip bounds, got {rank0}")
    scale = _mul_scale(proto)
    if scale is None or list(scale.dims):
        problems.append("the Mul scale is not a rank-0 initializer")
    return problems


def _header(proto: onnx.ModelProto, *, ir: int) -> list[str]:
    data = proto.SerializeToString()
    if any(word in data for word in (b"/Users/", b"/home/", str(REPO).encode())):
        return ["an absolute path is left in the model"]
    opsets = [(o.domain, o.version) for o in proto.opset_import if o.domain in ("", "ai.onnx")]
    problems = [] if proto.ir_version == ir else [f"IR {proto.ir_version}, expected {ir}"]
    if opsets != [("", OPSET)]:
        problems.append(f"opsets {opsets}")
    if any(t.HasField("data_location") and t.data_location for t in proto.graph.initializer):
        problems.append("external data")
    return problems


def _mul_scale(proto: onnx.ModelProto) -> onnx.TensorProto | None:
    mul = next(n for n in proto.graph.node if n.op_type == "Mul")
    by_name = {t.name: t for t in proto.graph.initializer}
    return by_name.get(mul.input[1])


def with_vector_scale(data: bytes) -> bytes:
    """The dynamo export with its rank-0 Mul scale broadcast to ``[D]`` (same values)."""
    proto = onnx.ModelProto.FromString(data)
    scale = _mul_scale(proto)
    assert scale is not None
    assert scale.data_type == ONNX_FLOAT
    payload = scale.raw_data or np.float32(scale.float_data[0]).tobytes()
    scale.ClearField("float_data")
    scale.raw_data = payload * D
    scale.dims[:] = [D]
    out: bytes = proto.SerializeToString(deterministic=True)
    return out


def action_rule(packed: dict[str, bytes]) -> str:
    rule: str = json.loads(packed["policy.json"])["action_rule"]
    return rule


def parity(net: ThirdPartyPolicy, packed: dict[str, bytes]) -> float:
    """max |ORT(packed) - torch| over 64 deterministic observations, some far past the clip."""
    obs = det((64, D), 11, 30.0)
    with torch.no_grad():
        want = net(torch.from_numpy(obs)).numpy()
    session = load_policy_files(packed, challenge=CHALLENGE).session
    got = np.asarray(session.run(["action"], {"obs": obs})[0])
    return float(np.max(np.abs(got - want)))


def build(
    net: ThirdPartyPolicy,
    export: Callable[[ThirdPartyPolicy], bytes],
    check: Callable[[onnx.ModelProto], list[str]],
    expected_rule: str,
) -> bytes:
    data = relative_paths(export(net))
    problems = check(onnx.ModelProto.FromString(data))
    try:
        packed = pack_policy(data, challenge=CHALLENGE)
    except LoadError as exc:
        problems.append(f"pack refused it: {exc}")
    else:
        proto = onnx.ModelProto.FromString(packed["policy.onnx"])
        if (proto.ir_version, proto.opset_import[0].version) != (CANONICAL_IR[OPSET], OPSET):
            problems.append("the packed model is not IR 9 / opset 20")
        if (rule := action_rule(packed)) != expected_rule:
            problems.append(f"the packed action_rule is {rule}, expected {expected_rule}")
        if (diff := parity(net, packed)) > TOLERANCE:
            problems.append(f"ORT differs from torch by {diff:.3g}")
    if problems:
        raise SystemExit(f"{export.__name__}: " + "; ".join(problems))
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="compare, write nothing")
    args = parser.parse_args(argv)
    torch.manual_seed(0)  # nothing here draws from it; set so nothing can draw differently
    net = ThirdPartyPolicy().eval()
    outputs = {
        LEGACY: build(net, export_legacy, check_legacy, "other"),
        DYNAMO: build(net, export_dynamo, check_dynamo, "other"),
    }
    vector = pack_policy(with_vector_scale(outputs[DYNAMO]), challenge=CHALLENGE)
    if action_rule(vector) != "clip(mean)":
        raise SystemExit("the dynamo export with a [D] scale is not clip(mean)")
    stale = []
    for name, data in outputs.items():
        target = DATA / name
        if args.check:
            if not target.is_file() or target.read_bytes() != data:
                stale.append(name)
            continue
        target.write_bytes(data)
        print(f"wrote tests/data/{name} ({len(data)} bytes)")
    if stale:
        print(f"differs from a fresh export: {', '.join(stale)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
