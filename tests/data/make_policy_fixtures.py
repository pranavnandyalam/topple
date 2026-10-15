"""Write ``noncanonical_policy.onnx``: the tiny first-party PPO policy as a third-party exporter
might write it (docs/design/phase-2.md §3.3 ``topple policy pack``, §6 ``test_policy_canonical``).

It holds every kind of non-canonical content pack mode removes: IR 10 under opset 17, a second
opset import, producer/doc strings, model domain and version, foreign metadata, ``value_info``,
I/O named ``x``/``y`` with a static batch of 1, empty and odd node names, a ``;`` in a value
name, node domains ``""`` and ``"ai.onnx"``, ``Constant`` nodes (a named value tensor with
``float_data``, ``value_float``), ``Identity`` nodes (one feeding the output), explicit default
Gemm attributes in reverse order, ``float_data`` initializers, initializers listed out of
first-use order and repeated as graph inputs. Packing it gives exactly the first-party build.

Run once from the repository root, then commit the output:

    uv run python tests/data/make_policy_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/, for policy_support

from policy_support import (  # noqa: E402 - after the path set-up
    balance_spec,
    tiny_norm,
    tiny_weights,
)
from topple.policy.build import policy_metadata  # noqa: E402


def tensor(name: str, values: np.ndarray, *, raw: bool) -> TensorProto:
    arr = np.asarray(values, dtype="<f4")
    if raw:
        return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.tobytes(), raw=True)
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.reshape(-1).tolist())


def noncanonical_policy() -> onnx.ModelProto:
    norm = tiny_norm()
    (w0, b0), (w1, b1), (w2, b2) = tiny_weights("tanh").layers
    d, a = w0.shape[1], w2.shape[0]
    mean = norm.mean.astype(np.float32)
    inv_std = (1.0 / np.sqrt(norm.var + norm.epsilon)).astype(np.float32)
    clip = float(np.float32(norm.clip))

    mean_const = helper.make_node(
        "Constant", [], ["mean_const"], name="", value=tensor("mean_tensor", mean, raw=False)
    )
    mean_const.doc_string = "running mean"
    sub = helper.make_node("Sub", ["x", "mean_const"], ["/norm/Sub_output_0"], name="")
    sub.domain = ""
    mul = helper.make_node("Mul", ["/norm/Sub_output_0", "inv_std"], ["a;b"], name="/norm/Mul")
    mul.domain = "ai.onnx"
    gemm0 = helper.make_node("Gemm", ["z_id", "layer0.weight", "layer0.bias"], ["h0"], name="g0")
    for name, value in (("transB", 1), ("transA", 0), ("beta", 1.0), ("alpha", 1.0)):
        gemm0.attribute.append(helper.make_attribute(name, value))
    nodes = [
        mean_const,
        sub,
        mul,
        helper.make_node("Constant", [], ["lo"], name="lo", value_float=-clip),
        helper.make_node("Constant", [], ["hi"], name="hi", value_float=clip),
        helper.make_node("Clip", ["a;b", "lo", "hi"], ["z"], name="clip_obs"),
        helper.make_node("Identity", ["z"], ["z_id"], name="identity_0"),
        gemm0,
        helper.make_node("Tanh", ["h0"], ["t0"], name="t0"),
        helper.make_node("Gemm", ["t0", "layer1.weight", "layer1.bias"], ["h1"], transB=1),
        helper.make_node("Tanh", ["h1"], ["t1"]),
        helper.make_node("Gemm", ["t1", "head.weight", "head.bias"], ["mu"], transB=1),
        helper.make_node("Clip", ["mu", "act_lo", "act_hi"], ["act"], name="clip_act"),
        helper.make_node("Identity", ["act"], ["y"], name="identity_out"),
    ]
    initializers = [
        tensor("head.bias", b2, raw=False),
        tensor("layer0.weight", w0, raw=False),
        tensor("inv_std", inv_std, raw=True),
        tensor("act_hi", np.float32(1.0), raw=False),
        tensor("layer1.bias", b1, raw=True),
        tensor("layer0.bias", b0, raw=True),
        tensor("head.weight", w2, raw=True),
        tensor("layer1.weight", w1, raw=False),
        tensor("act_lo", np.float32(-1.0), raw=True),
    ]
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, d]),
        helper.make_tensor_value_info("layer0.weight", TensorProto.FLOAT, list(w0.shape)),
    ]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch_size", a])]
    value_info = [helper.make_tensor_value_info("h0", TensorProto.FLOAT, [1, w0.shape[0]])]
    graph = helper.make_graph(
        nodes, "main_graph", inputs, outputs, initializers, "exported", value_info
    )
    model: onnx.ModelProto = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 3)],
        producer_name="pytorch",
        producer_version="2.14.1",
        doc_string="a third-party export",
        domain="ai.example",
        model_version=3,
    )
    entries = {"source": "torch.onnx", **policy_metadata(balance_spec(), "clip(mean)")}
    for key in sorted(entries, reverse=True):
        entry = model.metadata_props.add()
        entry.key, entry.value = key, entries[key]
    return model


def main() -> None:
    out = HERE / "noncanonical_policy.onnx"
    out.write_bytes(noncanonical_policy().SerializeToString(deterministic=True))
    print(f"wrote {out.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
