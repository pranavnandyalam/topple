"""The ONNX acceptance policy (docs/design/phase-2.md §3.4, §3.5, §6 ``test_policy_validate``).

Every rejection is pinned to the check that must catch it (checks run in order and the first
check with any message returns), and every message is fixed vocabulary plus ``safe`` text: no
traceback, no raw exception text, no markup.

Deferred to slice 1b: "every failure is a fixed-code ``LoadError``" needs ``load_policy_dir``
(here the fixed-code strings that it will wrap are checked), and the smoke-timer cases of the
§6 ``test_onnx_policy`` row are covered here because ``smoke_test`` is a slice 1a function.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import struct
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from google.protobuf.message import DecodeError
from onnx import AttributeProto, TensorProto, helper

import topple.policy.runtime as runtime
import topple.policy.validate as validate
from policy_support import (
    A,
    D,
    build,
    canonical,
    chain_model,
    det,
    graph_model,
    node,
    parse,
    ppo_bytes,
    rejected_at,
    sac_bytes,
    serialize,
    tensor,
    tiny_weights,
    validated,
    value_info,
)
from topple.contract import ActionSpec
from topple.policy.runtime import make_session
from topple.policy.validate import (
    CANONICAL_IR,
    CHECK_OF,
    MAX_COST,
    MAX_ONNX_BYTES,
    OP_WEIGHT,
    ValidatedPolicy,
    safe,
    smoke_run,
    smoke_test,
    static_cost,
    structure_match,
    validate_policy_bytes,
)

PPO = ppo_bytes()
SAC = sac_bytes()
FIXED = re.compile(r"[a-z0-9_]+(: [\x20-\x7e]*)?")
UNKNOWN_99 = b"\x98\x06\x01"  # field 99, varint 1
MARK = "QQQQ"
BAD_UTF8 = b"\xff\xfe\xfd\xfc"

Make = Callable[[], bytes]
CASES: dict[str, tuple[Make, int, str, tuple[int, int]]] = {}


def edit_case(check: int, code: str) -> Callable[[Callable[[onnx.ModelProto], None]], Any]:
    """Register an in-place edit of the first-party PPO build as a rejection case."""

    def register(edit: Callable[[onnx.ModelProto], None]) -> Callable[[onnx.ModelProto], None]:
        def make() -> bytes:
            proto = parse(PPO)
            edit(proto)
            return serialize(proto)

        CASES[edit.__name__] = (make, check, code, (D, A))
        return edit

    return register


def bytes_case(check: int, code: str, dims: tuple[int, int] = (D, A)) -> Callable[[Make], Make]:
    """Register a function returning the model bytes as a rejection case."""

    def register(make: Make) -> Make:
        CASES[make.__name__] = (make, check, code, dims)
        return make

    return register


def marked(edit: Callable[[onnx.ModelProto], None]) -> bytes:
    """PPO bytes where ``edit`` put ``MARK`` in a string, then those bytes made invalid UTF-8
    (same length, so every enclosing length prefix stays right)."""
    proto = parse(PPO)
    edit(proto)
    data = serialize(proto)
    assert MARK.encode() in data
    return data.replace(MARK.encode(), BAD_UTF8)


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        low, n = n & 0x7F, n >> 7
        out.append(low | (0x80 if n else 0))
        if not n:
            return bytes(out)


def field(number: int, payload: bytes) -> bytes:
    return varint(number << 3 | 2) + varint(len(payload)) + payload


def attr(name: str, kind: int, **payload: Any) -> AttributeProto:
    a = AttributeProto(name=name, type=kind)  # type: ignore[arg-type]
    for key, value in payload.items():
        if key == "floats":
            a.floats.extend(value)
        elif key in ("t", "g"):
            getattr(a, key).CopyFrom(value)
        else:
            setattr(a, key, value)
    return a


# --- check 1: size and parse -----------------------------------------------------------------


@bytes_case(1, "onnx_too_large")
def over_1_mib() -> bytes:
    return PPO + field(99, b"x" * MAX_ONNX_BYTES)


@bytes_case(1, "onnx_parse_failed")
def truncated_varint() -> bytes:
    return b"\x08\x80"


@bytes_case(1, "onnx_parse_failed")
def not_a_protobuf() -> bytes:
    return b"\xff\xff\xff\xff"


@bytes_case(1, "onnx_parse_failed")
def depth_40_nesting() -> bytes:
    """40 subgraphs nested through If attributes: about 120 message levels, past upb's 100."""
    graph = field(2, b"g")
    for _ in range(40):
        attribute = field(1, b"a") + field(6, graph)
        graph = field(1, field(4, b"If") + field(5, attribute))
    return field(7, graph)


# --- check 2: field allow-list ---------------------------------------------------------------


@bytes_case(2, "unknown_field")
def unknown_field_99_on_model() -> bytes:
    return PPO + UNKNOWN_99


@edit_case(2, "unknown_field")
def unknown_field_99_on_node(p: onnx.ModelProto) -> None:
    n = p.graph.node[0]
    data = n.SerializeToString() + UNKNOWN_99
    n.Clear()
    n.MergeFromString(data)


@edit_case(2, "unknown_field")
def unknown_field_99_on_tensor(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[4]
    data = t.SerializeToString() + UNKNOWN_99
    t.Clear()
    t.MergeFromString(data)


@edit_case(2, "field_not_allowed")
def producer_version_empty(p: onnx.ModelProto) -> None:
    p.producer_version = ""


@edit_case(2, "field_not_allowed")
def node_doc_string(p: onnx.ModelProto) -> None:
    p.graph.node[0].doc_string = "hi"


@edit_case(2, "field_not_allowed")
def model_doc_string(p: onnx.ModelProto) -> None:
    p.doc_string = "hi"


@edit_case(2, "field_not_allowed")
def configuration(p: onnx.ModelProto) -> None:
    p.configuration.add().name = "cfg"


@edit_case(2, "field_not_allowed")
def quantization_annotation(p: onnx.ModelProto) -> None:
    p.graph.quantization_annotation.add().tensor_name = "v0"


@edit_case(2, "field_not_allowed")
def device_configurations(p: onnx.ModelProto) -> None:
    p.graph.node[0].device_configurations.add().configuration_id = "cfg"


@edit_case(2, "field_not_allowed")
def overload(p: onnx.ModelProto) -> None:
    p.graph.node[0].overload = "x"


@edit_case(2, "field_not_allowed")
def type_denotation(p: onnx.ModelProto) -> None:
    p.graph.input[0].type.denotation = "TENSOR"


@edit_case(2, "field_not_allowed")
def dim_denotation(p: onnx.ModelProto) -> None:
    p.graph.input[0].type.tensor_type.shape.dim[0].denotation = "DATA_BATCH"


@edit_case(2, "field_not_allowed")
def input_metadata_props(p: onnx.ModelProto) -> None:
    entry = p.graph.input[0].metadata_props.add()
    entry.key, entry.value = "k", "v"


@edit_case(2, "field_not_allowed")
def initializer_metadata_props(p: onnx.ModelProto) -> None:
    entry = p.graph.initializer[0].metadata_props.add()
    entry.key, entry.value = "k", "v"


@edit_case(2, "field_not_allowed")
def ref_attr_name(p: onnx.ModelProto) -> None:
    p.graph.node[3].attribute[0].ref_attr_name = "x"


@edit_case(2, "field_not_allowed")
def node_domain(p: onnx.ModelProto) -> None:
    p.graph.node[0].domain = ""


@edit_case(2, "field_not_allowed")
def external_data(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[0]
    t.ClearField("raw_data")
    entry = t.external_data.add()
    entry.key, entry.value = "location", "weights.bin"
    t.data_location = TensorProto.EXTERNAL


@edit_case(2, "field_not_allowed")
def named_constant_value_tensor(p: onnx.ModelProto) -> None:
    value = tensor("named", [1.0])
    p.graph.node.insert(0, helper.make_node("Constant", [], ["k"], name="k", value=value))


@edit_case(2, "field_not_allowed")
def constant_tensor_name_1kib(p: onnx.ModelProto) -> None:
    value = tensor("a" * 1024, [1.0])
    p.graph.node.insert(0, helper.make_node("Constant", [], ["k"], name="k", value=value))


@edit_case(2, "field_not_allowed")
def spoofed_value_info(p: onnx.ModelProto) -> None:
    p.graph.value_info.append(value_info("v3", ["batch", 99]))


@edit_case(2, "field_not_allowed")
def function(p: onnx.ModelProto) -> None:
    f = p.functions.add()
    f.name, f.domain = "f", "local"


@edit_case(2, "field_not_allowed")
def sparse_initializer(p: onnx.ModelProto) -> None:
    sparse = p.graph.sparse_initializer.add()
    sparse.values.CopyFrom(tensor("s", [1.0]))
    sparse.dims.append(4)


@edit_case(2, "field_not_allowed")
def training_info(p: onnx.ModelProto) -> None:
    p.training_info.add().algorithm.name = "t"


@edit_case(2, "field_not_allowed")
def model_domain(p: onnx.ModelProto) -> None:
    p.domain = "x"


@edit_case(2, "field_not_allowed")
def if_with_subgraphs(p: onnx.ModelProto) -> None:
    branch = helper.make_graph([], "b", [], [value_info("v4", ["batch", 8])])
    n = p.graph.node[4]
    n.op_type = "If"
    n.attribute.extend([attr("then_branch", AttributeProto.GRAPH, g=branch)])


@edit_case(2, "field_not_allowed")
def constant_value_ints(p: onnx.ModelProto) -> None:
    p.graph.node.insert(0, helper.make_node("Constant", [], ["k"], name="k", value_ints=[1, 2]))


@edit_case(2, "field_not_allowed")
def int64_data_tensor(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[2]
    t.ClearField("raw_data")
    t.data_type = TensorProto.INT64
    t.int64_data.append(-10)


@edit_case(2, "opset_domain_not_allowed")
def opset_domain(p: onnx.ModelProto) -> None:
    p.opset_import[0].domain = "com.evil"


@bytes_case(2, "invalid_utf8")
def invalid_utf8_node_name() -> bytes:
    return marked(lambda p: setattr(p.graph.node[0], "name", MARK))


@bytes_case(2, "invalid_utf8")
def invalid_utf8_metadata_key() -> bytes:
    return marked(lambda p: setattr(p.metadata_props[0], "key", "topple." + MARK))


@bytes_case(2, "invalid_utf8")
def invalid_utf8_metadata_value() -> bytes:
    return marked(lambda p: setattr(p.metadata_props[0], "value", MARK))


@bytes_case(2, "invalid_utf8")
def invalid_utf8_initializer_name() -> bytes:
    def edit(p: onnx.ModelProto) -> None:
        p.graph.initializer[0].name = MARK
        p.graph.node[0].input[1] = MARK

    return marked(edit)


# --- check 3: header -------------------------------------------------------------------------


def header(opset: int, ir: int) -> Callable[[onnx.ModelProto], None]:
    def edit(p: onnx.ModelProto) -> None:
        p.opset_import[0].version = opset
        p.ir_version = ir

    return edit


CASES["ir8_opset21"] = (lambda: serialize_edit(header(21, 8)), 3, "bad_ir_version", (D, A))
CASES["ir10_opset20"] = (lambda: serialize_edit(header(20, 10)), 3, "bad_ir_version", (D, A))
CASES["ir14"] = (lambda: serialize_edit(header(17, 14)), 3, "bad_ir_version", (D, A))
CASES["opset16"] = (lambda: serialize_edit(header(16, 8)), 3, "unsupported_opset", (D, A))
CASES["opset22"] = (lambda: serialize_edit(header(22, 10)), 3, "unsupported_opset", (D, A))


def serialize_edit(edit: Callable[[onnx.ModelProto], None]) -> bytes:
    proto = parse(PPO)
    edit(proto)
    return serialize(proto)


@edit_case(3, "bad_opset_import")
def two_opset_imports(p: onnx.ModelProto) -> None:
    p.opset_import.append(helper.make_opsetid("", 18))


@edit_case(3, "bad_opset_import")
def no_opset_import(p: onnx.ModelProto) -> None:
    del p.opset_import[:]


# --- check 4: strings ------------------------------------------------------------------------


@edit_case(4, "bad_name")
def trailing_newline_node_name(p: onnx.ModelProto) -> None:
    p.graph.node[0].name = "n0\n"


@edit_case(4, "bad_metadata_key")
def trailing_newline_metadata_key(p: onnx.ModelProto) -> None:
    p.metadata_props[0].key += "\n"


@edit_case(4, "bad_op_type")
def trailing_newline_op_type(p: onnx.ModelProto) -> None:
    p.graph.node[4].op_type = "Tanh\n"


@edit_case(4, "bad_name")
def initializer_name_1kib(p: onnx.ModelProto) -> None:
    p.graph.initializer[0].name = "a" * 1024
    p.graph.node[0].input[1] = "a" * 1024


@edit_case(4, "bad_name")
def empty_node_name(p: onnx.ModelProto) -> None:
    p.graph.node[0].name = ""


@edit_case(4, "bad_name")
def semicolon_value_name(p: onnx.ModelProto) -> None:
    p.graph.node[0].output[0] = "v;0"
    p.graph.node[1].input[0] = "v;0"


@edit_case(4, "bad_name")
def graph_name(p: onnx.ModelProto) -> None:
    p.graph.name = "topple policy"


@edit_case(4, "bad_name")
def markup_node_name(p: onnx.ModelProto) -> None:
    p.graph.node[0].name = "<b>"


@edit_case(4, "bad_op_type")
def markup_op_type(p: onnx.ModelProto) -> None:
    p.graph.node[4].op_type = "<b>"


@edit_case(2, "field_not_allowed")
def markup_domain(p: onnx.ModelProto) -> None:
    p.graph.node[4].domain = "<b>"


@edit_case(4, "bad_attribute_name")
def markup_attribute_name(p: onnx.ModelProto) -> None:
    p.graph.node[3].attribute[0].name = "<b>"


def set_metadata(p: onnx.ModelProto, key: str, value: str | None) -> None:
    entries = {e.key: e.value for e in p.metadata_props}
    if value is None:
        del entries[key]
    else:
        entries[key] = value
    del p.metadata_props[:]
    for k in sorted(entries):
        entry = p.metadata_props.add()
        entry.key, entry.value = k, entries[k]


@edit_case(4, "bad_metadata_key")
def extra_topple_note_key(p: onnx.ModelProto) -> None:
    set_metadata(p, "topple.note", '"hi"')


@edit_case(4, "bad_metadata_key")
def foreign_metadata_key(p: onnx.ModelProto) -> None:
    set_metadata(p, "source", '"torch"')


@edit_case(4, "missing_metadata_key")
def missing_metadata_key(p: onnx.ModelProto) -> None:
    set_metadata(p, "topple.format", None)


@edit_case(4, "duplicate_metadata_key")
def duplicate_metadata_key(p: onnx.ModelProto) -> None:
    entry = p.metadata_props.add()
    entry.key, entry.value = "topple.format", '"topple.policy/1"'


@edit_case(4, "bad_metadata_value")
def non_ascii_metadata_value(p: onnx.ModelProto) -> None:
    set_metadata(p, "topple.challenge", '"bälance-v1"')


@edit_case(4, "bad_metadata_value")
def metadata_value_2kb(p: onnx.ModelProto) -> None:
    set_metadata(p, "topple.challenge", '"' + "x" * 2048 + '"')


@edit_case(4, "producer_name_too_long")
def producer_name_129(p: onnx.ModelProto) -> None:
    p.producer_name = "p" * 129


# --- check 5: inputs and outputs -------------------------------------------------------------


@bytes_case(5, "bad_input", dims=(11, A))
def width_11() -> bytes:
    return PPO


@bytes_case(5, "bad_output", dims=(D, 3))
def action_width_3() -> bytes:
    return PPO


@edit_case(5, "initializer_as_input")
def initializer_as_input(p: onnx.ModelProto) -> None:
    p.graph.input.append(value_info("c0", [D]))


@edit_case(5, "bad_input")
def input_named_x(p: onnx.ModelProto) -> None:
    p.graph.input[0].name = "x"
    p.graph.node[0].input[0] = "x"


@edit_case(5, "bad_input")
def static_batch_1(p: onnx.ModelProto) -> None:
    p.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1


@edit_case(5, "bad_output")
def different_batch_symbols(p: onnx.ModelProto) -> None:
    p.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "N"


@edit_case(5, "bad_input")
def int64_input(p: onnx.ModelProto) -> None:
    p.graph.input[0].type.tensor_type.elem_type = TensorProto.INT64


@edit_case(5, "bad_input")
def rank_3_input(p: onnx.ModelProto) -> None:
    p.graph.input[0].type.tensor_type.shape.dim.add().dim_value = 1


@edit_case(5, "bad_output")
def two_outputs(p: onnx.ModelProto) -> None:
    p.graph.output.append(value_info("v7", ["batch", A]))


# --- check 6: ops, arity, attributes ---------------------------------------------------------


def replace_op(index: int, op: str) -> Callable[[onnx.ModelProto], None]:
    def edit(p: onnx.ModelProto) -> None:
        p.graph.node[index].op_type = op

    return edit


for _op in ("Sin", "Flatten", "Identity", "If"):
    CASES[_op.lower()] = (
        lambda op=_op: serialize_edit(replace_op(4, op)),  # type: ignore[misc]
        6,
        "op_not_allowed",
        (D, A),
    )


@bytes_case(6, "bad_arity")
def variadic_max_245k() -> bytes:
    nodes = [node("Gemm", ["obs", "w"], "h", transB=1), node("Max", ["h"] * 245_000, "action")]
    data = serialize(graph_model(nodes, [tensor("w", det((A, D), 1))]))
    assert len(data) <= MAX_ONNX_BYTES
    return data


@edit_case(6, "bad_arity")
def clip_4_inputs(p: onnx.ModelProto) -> None:
    p.graph.node[2].input.append("c3")


@edit_case(6, "bad_attribute_value")
def gemm_transA_outer_product(p: onnx.ModelProto) -> None:  # noqa: N802 - the §6 case name
    p.graph.node[3].attribute.append(attr("transA", AttributeProto.INT, i=1))


@edit_case(6, "nonfinite_attribute")
def gemm_alpha_nan(p: onnx.ModelProto) -> None:
    p.graph.node[3].attribute.append(attr("alpha", AttributeProto.FLOAT, f=float("nan")))


@edit_case(6, "bad_attribute_type")
def transB_as_float(p: onnx.ModelProto) -> None:  # noqa: N802 - the §6 case name
    p.graph.node[3].ClearField("attribute")
    p.graph.node[3].attribute.append(attr("transB", AttributeProto.FLOAT, f=1.0))


@edit_case(6, "bad_attribute_type")
def gemm_beta_int(p: onnx.ModelProto) -> None:
    p.graph.node[3].attribute.append(attr("beta", AttributeProto.INT, i=1))


@edit_case(6, "bad_attribute_value")
def gemm_transB_2(p: onnx.ModelProto) -> None:  # noqa: N802
    p.graph.node[3].attribute[0].i = 2


@edit_case(6, "bad_attribute_value")
def gemm_alpha_with_two_payloads(p: onnx.ModelProto) -> None:
    p.graph.node[3].attribute.append(attr("alpha", AttributeProto.FLOAT, f=1.0, i=3))


@edit_case(6, "duplicate_attribute")
def duplicate_attribute(p: onnx.ModelProto) -> None:
    p.graph.node[3].attribute.append(attr("transB", AttributeProto.INT, i=1))


@edit_case(6, "attribute_not_allowed")
def relu_with_alpha(p: onnx.ModelProto) -> None:
    p.graph.node[4].attribute.append(attr("alpha", AttributeProto.FLOAT, f=0.5))


@edit_case(6, "nonfinite_attribute")
def leaky_relu_alpha_inf(p: onnx.ModelProto) -> None:
    n = p.graph.node[4]
    n.op_type = "LeakyRelu"
    n.attribute.append(attr("alpha", AttributeProto.FLOAT, f=float("inf")))


@edit_case(6, "weight_not_constant")
def gemm_weight_not_constant(p: onnx.ModelProto) -> None:
    p.graph.node[3].input[1] = "v2"


@edit_case(6, "weight_not_constant")
def matmul_weight_not_constant(p: onnx.ModelProto) -> None:
    n = p.graph.node[3]
    n.op_type = "MatMul"
    n.ClearField("attribute")
    del n.input[2]
    n.input[1] = "v1"


@edit_case(6, "empty_input_name")
def empty_input_name(p: onnx.ModelProto) -> None:
    p.graph.node[2].input[1] = ""


@edit_case(6, "bad_output_count")
def node_with_two_outputs(p: onnx.ModelProto) -> None:
    p.graph.node[4].output.append("extra")


@edit_case(6, "bad_attribute_value")
def constant_two_values(p: onnx.ModelProto) -> None:
    k = helper.make_node("Constant", [], ["k"], name="k", value_float=1.0)
    k.attribute.append(attr("value_floats", AttributeProto.FLOATS, floats=[1.0]))
    p.graph.node.insert(0, k)


@edit_case(6, "bad_attribute_value")
def constant_empty_value_floats(p: onnx.ModelProto) -> None:
    k = helper.make_node("Constant", [], ["k"], name="k")
    k.attribute.append(attr("value_floats", AttributeProto.FLOATS))
    p.graph.node.insert(0, k)


@bytes_case(6, "too_many_node_inputs")
def node_inputs_over_192() -> bytes:
    return serialize(chain_model(["Max"] * 48, 4, inputs_per_node=4))  # 2 + 192 + 2


# --- check 7: tensors ------------------------------------------------------------------------


@edit_case(7, "nonfinite_tensor")
def nan_weight(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[6]
    t.raw_data = struct.pack("<f", float("nan")) + t.raw_data[4:]


@edit_case(7, "nonfinite_tensor")
def inf_in_float_data(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[0]
    values = list(np.frombuffer(t.raw_data, "<f4"))
    t.ClearField("raw_data")
    t.float_data.extend([float("inf"), *values[1:]])


@edit_case(7, "bad_tensor_payload")
def both_payloads(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[0]
    t.float_data.extend(np.frombuffer(t.raw_data, "<f4").tolist())


@edit_case(7, "bad_tensor_payload")
def neither_payload(p: onnx.ModelProto) -> None:
    p.graph.initializer[0].ClearField("raw_data")


@edit_case(7, "bad_tensor_type")
def int64_tensor(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[2]
    t.data_type = TensorProto.INT64
    t.raw_data = struct.pack("<q", -10)


@edit_case(7, "bad_tensor_shape")
def rank_3_tensor(p: onnx.ModelProto) -> None:
    p.graph.initializer[4].dims.append(1)


@edit_case(7, "bad_tensor_shape")
def zero_dim_tensor(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[0]
    del t.dims[:]
    t.dims.extend([D, 0])
    t.raw_data = b""


@edit_case(7, "bad_tensor_shape")
def dim_4097(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[0]
    t.dims[0] = 4097
    t.raw_data = b"\x00" * 4 * 4097


@edit_case(7, "bad_tensor_size")
def raw_length_mismatch(p: onnx.ModelProto) -> None:
    p.graph.initializer[0].raw_data += b"\x00\x00\x00\x00"


@edit_case(7, "bad_tensor_size")
def float_data_count_mismatch(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[0]
    t.ClearField("raw_data")
    t.float_data.extend([0.0] * (D - 1))


@edit_case(7, "too_many_initializers")
def initializers_33(p: onnx.ModelProto) -> None:
    p.graph.initializer.extend(tensor(f"u{i}", [1.0]) for i in range(21))


@edit_case(7, "constant_too_large")
def constant_over_4096_elements(p: onnx.ModelProto) -> None:
    value = TensorProto(dims=[2, 2049], data_type=TensorProto.FLOAT, raw_data=b"\x00" * 4 * 4098)
    p.graph.node.insert(0, helper.make_node("Constant", [], ["k"], name="k", value=value))


@bytes_case(7, "too_many_nodes")
def nodes_65() -> bytes:
    return serialize(chain_model(["Relu"] * 63, 4))


@bytes_case(7, "too_many_elements", dims=(31, 1))
def elements_131073() -> bytes:
    nodes = [
        node("Gemm", ["obs", "w0"], "h", transB=1),
        node("Gemm", ["h", "w1", "b1"], "action", transB=1),
    ]
    inits = [
        tensor("w0", np.zeros((4096, 31))),
        tensor("w1", np.zeros((1, 4096))),
        tensor("b1", np.zeros(1)),
    ]
    return serialize(graph_model(nodes, inits, obs_dim=31, action_dim=1))


# --- check 8: structure and cost -------------------------------------------------------------


@edit_case(8, "unused_initializer")
def unused_initializer(p: onnx.ModelProto) -> None:
    p.graph.initializer.append(tensor("c12", [1.0]))


@edit_case(8, "dead_node")
def dead_node(p: onnx.ModelProto) -> None:
    p.graph.node.insert(5, helper.make_node("Relu", ["v4"], ["dead"], name="dead"))


@edit_case(8, "onnx_checker_failed")
def undefined_input(p: onnx.ModelProto) -> None:
    p.graph.node[4].input[0] = "nowhere"


@edit_case(8, "onnx_checker_failed")
def gemm_inner_dim_mismatch(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[4]
    t.dims[1] = 5
    t.dims[0] = 16


@bytes_case(8, "not_batch_major")
def max_bcast() -> bytes:
    nodes = [
        node("Gemm", ["obs", "w"], "action", transB=1),
        node("Max", ["ca", "cb"], "outer"),
    ]
    inits = [
        tensor("w", det((A, D), 1)),
        tensor("ca", np.ones((4096, 1))),
        tensor("cb", np.ones((1, 4096))),
    ]
    return serialize(graph_model(nodes, inits))


@bytes_case(8, "not_batch_major")
def bcast_matmul() -> bytes:
    nodes = [
        node("Gemm", ["obs", "w"], "action", transB=1),
        node("MatMul", ["ca", "cb"], "outer"),
    ]
    inits = [
        tensor("w", det((A, D), 1)),
        tensor("ca", np.ones((4096, 1))),
        tensor("cb", np.ones((1, 4096))),
    ]
    return serialize(graph_model(nodes, inits))


@bytes_case(8, "not_batch_major")
def matmul_rank1_weight() -> bytes:
    nodes = [
        node("MatMul", ["obs", "v"], "flat"),
        node("Gemm", ["obs", "w"], "action", transB=1),
    ]
    inits = [tensor("v", np.ones(D)), tensor("w", det((A, D), 1))]
    return serialize(graph_model(nodes, inits))


@bytes_case(8, "cost_exceeded")
def max4_width4096_x40() -> bytes:
    return serialize(chain_model(["Max"] * 40, 4096, inputs_per_node=4))


@bytes_case(8, "cost_exceeded")
def softplus_chain_w4096_x52() -> bytes:
    return serialize(chain_model(["Softplus"] * 52, 4096))


@bytes_case(8, "cost_exceeded")
def elu_chain_w4096_x52() -> bytes:
    return serialize(chain_model(["Elu"] * 52, 4096))


def reused_256(uses: int) -> onnx.ModelProto:
    """obs -> Gemm(256) -> Tanh -> [Gemm(W) -> Tanh] * uses -> Gemm(2): one 256x256 weight."""
    nodes = [node("Gemm", ["obs", "w_in"], "h0", transB=1), node("Tanh", ["h0"], "t0")]
    for i in range(uses):
        nodes.append(node("Gemm", [f"t{i}", "w"], f"h{i + 1}", transB=1))
        nodes.append(node("Tanh", [f"h{i + 1}"], f"t{i + 1}"))
    nodes.append(node("Gemm", [f"t{uses}", "w_out"], "action", transB=1))
    inits = [
        tensor("w_in", det((256, D), 1, 0.1)),
        tensor("w", det((256, 256), 2, 0.05)),
        tensor("w_out", det((A, 256), 3, 0.1)),
    ]
    return graph_model(nodes, inits)


@bytes_case(8, "cost_exceeded")
def weight_256_reused_by_2_gemms() -> bytes:
    return serialize(reused_256(2))


def cost_model(extra_neg: bool) -> onnx.ModelProto:
    """Static cost exactly 2^17: 8·4096 + 11 Tanh·2·4096 + Relu·4096 + 4096·1 (+1 for a Neg)."""
    nodes = [node("Gemm", ["obs", "w_in"], "h0", transB=1)]
    ops = ["Tanh"] * 11 + ["Relu"]
    for i, op in enumerate(ops):
        nodes.append(node(op, [f"h{i}"], f"h{i + 1}"))
    last = "pre" if extra_neg else "action"
    nodes.append(node("Gemm", [f"h{len(ops)}", "w_out"], last, transB=1))
    if extra_neg:
        nodes.append(node("Neg", ["pre"], "action"))
    inits = [tensor("w_in", det((4096, 8), 1, 0.01)), tensor("w_out", det((1, 4096), 2, 0.01))]
    return graph_model(nodes, inits, obs_dim=8, action_dim=1)


@bytes_case(8, "cost_exceeded", dims=(8, 1))
def static_cost_2_17_plus_1() -> bytes:
    return canonical(cost_model(extra_neg=True))


# --- check 9: canonical ----------------------------------------------------------------------


@edit_case(9, "not_canonical")
def constant_node(p: onnx.ModelProto) -> None:
    """A canonical-looking model whose normalization lower bound is a Constant node."""
    lo = p.graph.initializer[2]
    p.graph.node.insert(0, helper.make_node("Constant", [], ["k0"], name="k0", value=lo))
    p.graph.node[0].attribute[0].t.ClearField("name")
    p.graph.node[3].input[1] = "k0"
    del p.graph.initializer[2]


@edit_case(9, "not_canonical")
def explicit_default_alpha(p: onnx.ModelProto) -> None:
    p.graph.node[3].attribute.append(attr("alpha", AttributeProto.FLOAT, f=1.0))


@edit_case(9, "not_canonical")
def float_data_payload(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[0]
    values = np.frombuffer(t.raw_data, "<f4").tolist()
    t.ClearField("raw_data")
    t.float_data.extend(values)


@edit_case(9, "not_canonical")
def renamed_node(p: onnx.ModelProto) -> None:
    p.graph.node[0].name = "first"


@edit_case(9, "not_canonical")
def metadata_out_of_order(p: onnx.ModelProto) -> None:
    entries = [(e.key, e.value) for e in p.metadata_props][::-1]
    del p.metadata_props[:]
    for k, v in entries:
        entry = p.metadata_props.add()
        entry.key, entry.value = k, v


@edit_case(9, "not_canonical")
def producer_name_other(p: onnx.ModelProto) -> None:
    p.producer_name = "pytorch"


@edit_case(9, "not_canonical")
def graph_name_other(p: onnx.ModelProto) -> None:
    p.graph.name = "main_graph"


@edit_case(9, "not_canonical")
def opset_domain_ai_onnx(p: onnx.ModelProto) -> None:
    p.opset_import[0].domain = "ai.onnx"


@edit_case(9, "not_canonical")
def batch_symbol_n(p: onnx.ModelProto) -> None:
    p.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    p.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "N"


@edit_case(4, "bad_name")
def io_name_with_space(p: onnx.ModelProto) -> None:
    p.graph.input[0].name = "o bs"


@edit_case(5, "bad_input")
def input_without_type(p: onnx.ModelProto) -> None:
    p.graph.input[0].ClearField("type")


@edit_case(7, "nonfinite_tensor")
def nan_constant_value(p: onnx.ModelProto) -> None:
    value = tensor("", [float("nan")])
    value.ClearField("name")
    p.graph.node.insert(0, helper.make_node("Constant", [], ["k"], name="k", value=value))


@edit_case(9, "not_canonical")
def constant_value_float_node(p: onnx.ModelProto) -> None:
    """A Constant ``value_float`` passes checks 4-8 (its attribute name has an underscore)."""
    (hi,) = struct.unpack("<f", p.graph.initializer[3].raw_data)
    p.graph.node.insert(0, helper.make_node("Constant", [], ["k0"], name="k0", value_float=hi))
    p.graph.node[3].input[2] = "k0"
    del p.graph.initializer[3]


@edit_case(9, "not_canonical")
def constant_value_floats_node(p: onnx.ModelProto) -> None:
    mean = np.frombuffer(p.graph.initializer[0].raw_data, "<f4").tolist()
    p.graph.node.insert(0, helper.make_node("Constant", [], ["k0"], name="k0", value_floats=mean))
    p.graph.node[1].input[1] = "k0"
    del p.graph.initializer[0]


# --- the rejection table ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_rejected_at_its_check(name: str) -> None:
    make, check, code, (obs_dim, action_dim) = CASES[name]
    result = validate_policy_bytes(make(), obs_dim, action_dim)
    assert rejected_at(result) == (check, code), result
    assert isinstance(result, list)
    for message in result:
        assert FIXED.fullmatch(message), message
        assert "<" not in message
        assert "Traceback" not in message


def test_every_code_names_one_check() -> None:
    assert set(CHECK_OF.values()) == set(range(1, 10))
    assert all(FIXED.fullmatch(code) for code in CHECK_OF)


def test_markup_never_reaches_a_message() -> None:
    for name in ("markup_node_name", "markup_op_type", "markup_domain", "markup_attribute_name"):
        make, *_ = CASES[name]
        result = validate_policy_bytes(make(), D, A)
        assert isinstance(result, list)
        assert result
        assert not any("<" in m or ">" in m for m in result)


def test_messages_quote_names_through_safe() -> None:
    result = validate_policy_bytes(CASES["markup_node_name"][0](), D, A)
    assert result == ["bad_name: node '?b?'"]


def test_message_count_is_bounded() -> None:
    proto = parse(PPO)
    for n in proto.graph.node:
        n.name = "bad name"
    result = validate_policy_bytes(serialize(proto), D, A)
    assert isinstance(result, list)
    assert 1 <= len(result) <= validate.MAX_MESSAGES


# --- acceptance ------------------------------------------------------------------------------


@pytest.mark.parametrize(("data", "cost"), [(PPO, 236), (SAC, 224)], ids=["ppo", "sac"])
def test_accepts_first_party(data: bytes, cost: int) -> None:
    v = validate_policy_bytes(data, D, A)
    assert isinstance(v, ValidatedPolicy)
    assert v.canonical == data
    assert v.policy_sha256 == hashlib.sha256(data).hexdigest()
    assert (v.cost, v.obs_dim, v.action_dim) == (cost, D, A)


def test_first_party_64x64_cost_is_about_5200() -> None:
    v = validated(build(tiny_weights(hidden=(64, 64))))
    assert v.cost == 5164
    assert MAX_COST // v.cost >= 25


def test_static_cost_2_17_is_accepted() -> None:
    data = canonical(cost_model(extra_neg=False))
    v = validate_policy_bytes(data, 8, 1)
    assert isinstance(v, ValidatedPolicy)
    assert v.cost == MAX_COST == 1 << 17
    assert static_cost(parse(data)) == MAX_COST
    assert static_cost(parse(canonical(cost_model(extra_neg=True)))) == MAX_COST + 1


def test_one_use_of_a_256_weight_is_accepted() -> None:
    v = validate_policy_bytes(canonical(reused_256(1)), D, A)
    assert isinstance(v, ValidatedPolicy)
    assert v.cost == 2560 + 512 + 65536 + 512 + 512


def test_static_cost_formula() -> None:
    proto = parse(PPO)
    assert static_cost(proto) == 236
    # Gemm/MatMul: K·N; others: weight · max(1, inputs - 1) · width
    assert (OP_WEIGHT["Softplus"], OP_WEIGHT["Tanh"], OP_WEIGHT["Clip"]) == (48, 2, 1)


def test_static_cost_counts_constant_weights() -> None:
    proto = parse(PPO)
    w = proto.graph.initializer[6]
    proto.graph.node.insert(0, helper.make_node("Constant", [], ["k"], name="k", value=w))
    proto.graph.node[0].attribute[0].t.ClearField("name")
    proto.graph.node[6].input[1] = "k"
    del proto.graph.initializer[6]
    assert static_cost(proto) == 236


def test_static_cost_refuses_a_graph_that_is_not_batch_major() -> None:
    with pytest.raises(ValueError, match="not_batch_major"):
        static_cost(parse(CASES["max_bcast"][0]()))


def test_header_check_refuses_functions_even_past_the_allow_list() -> None:
    """Check 2 rejects these fields first; check 3 keeps the rule on its own as well."""
    proto = parse(PPO)
    proto.functions.add().name = "f"
    assert list(validate._check_header(proto)) == [
        "unsupported_feature: functions, training info and sparse initializers"
    ]


def test_lone_surrogates_are_not_utf8() -> None:
    """upb returns ``bytes`` for invalid UTF-8; a ``str`` must also have a UTF-8 form."""
    assert validate._is_utf8("n0")
    assert not validate._is_utf8("\ud800")
    assert not validate._is_utf8(b"n0")


def test_canonical_ir_table() -> None:
    assert CANONICAL_IR == {17: 8, 18: 8, 19: 9, 20: 9, 21: 10}


# --- exceptions become fixed codes -----------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (UnicodeDecodeError("utf-8", b"\xff<script>", 0, 1, "invalid start byte"), "invalid_utf8"),
        (RecursionError("<deep>"), "onnx_too_deep"),
        (MemoryError("<big>"), "out_of_memory"),
        (ValueError("<value>"), "onnx_invalid"),
        (TypeError("<type>"), "onnx_invalid"),
        (DecodeError("<decode>"), "onnx_parse_failed"),
        (KeyError("<key>"), "validator_error"),
    ],
)
def test_exceptions_become_fixed_codes(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, code: str
) -> None:
    def boom(*args: object, **kwargs: object) -> bytes:
        raise exc

    monkeypatch.setattr(validate, "canonicalize", boom)
    result = validate_policy_bytes(PPO, D, A)
    assert result == [f"{code}: raised during check 9"]


def test_checker_and_inference_exceptions_are_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise UnicodeDecodeError("utf-8", b"<script>", 0, 1, "x")

    monkeypatch.setattr(onnx.checker, "check_model", boom)
    assert validate_policy_bytes(PPO, D, A) == ["onnx_checker_failed"]
    monkeypatch.undo()
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", boom)
    assert validate_policy_bytes(PPO, D, A) == ["shape_inference_failed"]


def test_non_bytes_input_is_a_fixed_rejection() -> None:
    result = validate_policy_bytes("policy", D, A)  # type: ignore[arg-type]
    assert rejected_at(result) == (1, "onnx_parse_failed")


# --- external data red team ------------------------------------------------------------------


def test_external_data_is_rejected_before_any_tensor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_fifo: Callable[[Path], Path]
) -> None:
    (tmp_path / ".env").write_bytes(b"\x00" * 4 * D)  # exactly c0's size
    make_fifo(tmp_path / "fifo")
    monkeypatch.chdir(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("ran on bytes that failed checks 1-7")

    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    monkeypatch.setattr(onnx.checker, "check_model", forbidden)
    monkeypatch.setattr(onnx.numpy_helper, "to_array", forbidden)
    proto = parse(PPO)
    for t, location in zip(proto.graph.initializer[:2], (".env", "fifo"), strict=True):
        t.ClearField("raw_data")
        for key, value in (("location", location), ("offset", "0"), ("length", str(4 * D))):
            entry = t.external_data.add()
            entry.key, entry.value = key, value
        t.data_location = TensorProto.EXTERNAL
    result = validate_policy_bytes(serialize(proto), D, A)
    assert rejected_at(result) == (2, "field_not_allowed")
    with pytest.raises(ValueError, match="external_data"):
        validate.canonicalize(proto, pack=True)


# --- ValidatedPolicy -------------------------------------------------------------------------


def test_validated_policy_is_constructible_only_in_validate() -> None:
    with pytest.raises(TypeError):
        ValidatedPolicy(b"", "0" * 64, 1, D, A)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ValidatedPolicy(b"", "0" * 64, 1, D, A, object())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="validate_policy_bytes"):
        ValidatedPolicy(b"", "0" * 64, 1, D, A, seal=object())
    v = validated(PPO)
    with pytest.raises((TypeError, ValueError)):
        dataclasses.replace(v, canonical=b"evil")
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.canonical = b"evil"  # type: ignore[misc]


# --- safe ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("n0", "n0"),
        ("<b>", "?b?"),
        ("a\nb", "a?b"),
        (b"\xff<", "??"),
        ("x" * 100, "x" * 64),
        ("\x1b[31m", "??31m"),
        ("a/b:c.d-e_f", "a/b:c.d-e_f"),
        ("\ud800", "?"),
    ],
)
def test_safe(raw: str | bytes, shown: str) -> None:
    assert safe(raw) == shown


# --- structure match -------------------------------------------------------------------------


def test_structure_match_first_party() -> None:
    assert structure_match(parse(PPO)) == "clip(mean)"
    assert structure_match(parse(SAC)) == "tanh(mean)"


def edited(data: bytes, edit: Callable[[onnx.ModelProto], None]) -> onnx.ModelProto:
    proto = parse(data)
    edit(proto)
    return proto


def set_scalar(index: int, value: float) -> Callable[[onnx.ModelProto], None]:
    def edit(p: onnx.ModelProto) -> None:
        p.graph.initializer[index].raw_data = struct.pack("<f", value)

    return edit


def swap_sub_inputs(p: onnx.ModelProto) -> None:
    n = p.graph.node[0]
    n.input[0], n.input[1] = n.input[1], n.input[0]


def rank0_scale(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[1]
    del t.dims[:]
    t.raw_data = t.raw_data[:4]


def extra_gemm_attribute(p: onnx.ModelProto) -> None:
    p.graph.node[3].attribute.append(attr("alpha", AttributeProto.FLOAT, f=0.5))


def transb_0(p: onnx.ModelProto) -> None:
    p.graph.node[5].ClearField("attribute")


def float_data_head_bound(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[10]
    t.ClearField("raw_data")
    t.float_data.append(-1.0)


@pytest.mark.parametrize(
    "proto",
    [
        parse(build(action=ActionSpec(id="diffdrive.effort.v1", names=("l", "r"), low=-2, high=2))),
        edited(PPO, set_scalar(10, -0.999)),
        edited(PPO, set_scalar(11, float("nan"))),
        edited(PPO, set_scalar(2, -5.0)),
        edited(PPO, set_scalar(2, float("-inf"))),
        edited(PPO, swap_sub_inputs),
        edited(PPO, rank0_scale),
        edited(PPO, extra_gemm_attribute),
        edited(PPO, transb_0),
        edited(PPO, float_data_head_bound),
        parse(build(tiny_weights("relu"), action_rule="clip(mean)")),
        parse(build(tiny_weights("tanh"), action_rule="tanh(mean)")),
        parse(canonical(reused_256(1))),
    ],
    ids=[
        "head_clip_2",
        "head_bound_not_bitwise_1",
        "head_bound_nan",
        "asymmetric_normalization",
        "infinite_normalization",
        "sub_inputs_swapped",
        "rank0_scale",
        "extra_gemm_attribute",
        "transB_0",
        "float_data_bound",
        "relu_hidden_clip_head",
        "tanh_hidden_tanh_head",
        "other_graph",
    ],
)
def test_structure_match_other(proto: onnx.ModelProto) -> None:
    assert structure_match(proto) == "other"


@pytest.mark.parametrize("hidden", [(64, 32), (3, 17)])
def test_structure_match_reads_hidden_widths_from_the_weights(hidden: tuple[int, int]) -> None:
    assert structure_match(parse(build(tiny_weights("tanh", hidden=hidden)))) == "clip(mean)"
    assert structure_match(parse(build(tiny_weights("relu", hidden=hidden)))) == "tanh(mean)"


def test_structure_match_needs_one_input() -> None:
    proto = parse(PPO)
    proto.graph.input.append(value_info("extra", ["batch", 1]))
    assert structure_match(proto) == "other"


def test_structure_match_needs_a_static_io_width() -> None:
    proto = parse(PPO)
    proto.graph.input[0].type.tensor_type.shape.dim[1].dim_param = "d"
    assert structure_match(proto) == "other"


# --- make_session ----------------------------------------------------------------------------


def test_make_session_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnxruntime as ort

    session = make_session(validated(PPO))
    assert session.get_providers() == ["CPUExecutionProvider"]
    options = session.get_session_options()
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert options.log_severity_level == 3
    assert options.use_deterministic_compute
    assert options.get_session_config_entry("session.load_model_format") == "ONNX"
    assert options.get_session_config_entry("session.allow_released_opsets_only") == "1"
    assert options.get_session_config_entry("session.strict_shape_type_inference") == "1"


def test_make_session_requires_a_validated_policy() -> None:
    with pytest.raises(TypeError):
        make_session(PPO)  # type: ignore[arg-type]


def test_make_session_scrubs_ort_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORT_LOGGING_LEVEL", "0")
    monkeypatch.setenv("ORTX_KEEP", "1")
    make_session(validated(PPO))
    assert "ORT_LOGGING_LEVEL" not in os.environ
    assert os.environ["ORTX_KEEP"] == "1"


def forged(canonical_bytes: bytes) -> ValidatedPolicy:
    """A ValidatedPolicy that skipped validation (as if the seal were bypassed)."""
    v = ValidatedPolicy.__new__(ValidatedPolicy)
    for name, value in (
        ("canonical", canonical_bytes),
        ("policy_sha256", "0" * 64),
        ("cost", 1),
        ("obs_dim", D),
        ("action_dim", A),
    ):
        object.__setattr__(v, name, value)
    return v


def test_make_session_reasserts_no_external_data(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnxruntime as ort

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("session created")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    proto = parse(PPO)
    t = proto.graph.initializer[0]
    t.ClearField("raw_data")
    t.data_location = TensorProto.EXTERNAL
    with pytest.raises(ValueError, match="external"):
        make_session(forged(serialize(proto)))
    entry = t.external_data.add()
    entry.key, entry.value = "location", ".env"
    t.ClearField("data_location")
    with pytest.raises(ValueError, match="external"):
        make_session(forged(serialize(proto)))


def test_make_session_asserts_cpu_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(runtime.ort, "InferenceSession", Fake)
    with pytest.raises(RuntimeError, match="CPUExecutionProvider"):
        make_session(validated(PPO))


# --- smoke_test ------------------------------------------------------------------------------


@pytest.mark.parametrize("data", [PPO, SAC], ids=["ppo", "sac"])
def test_smoke_passes_first_party(data: bytes) -> None:
    v = validated(data)
    assert smoke_test(v) == []
    report = smoke_run(v)
    assert report.errors == ()
    assert report.latency_p99_us is not None
    assert report.latency_p99_us > 0


class FakeSession:
    """Records every run call; ``behaviour`` decides what a call returns or raises."""

    def __init__(self, behaviour: Callable[[Any, Any], Any]) -> None:
        self.behaviour = behaviour
        self.options: list[Any] = []
        self.feeds: list[Any] = []

    def run(self, names: Any, feeds: dict[str, Any], options: Any = None) -> Any:
        self.options.append(options)
        self.feeds.append(feeds["obs"].copy())
        return self.behaviour(feeds, options)


def with_fake(monkeypatch: pytest.MonkeyPatch, behaviour: Callable[[Any, Any], Any]) -> FakeSession:
    fake = FakeSession(behaviour)
    monkeypatch.setattr(runtime, "make_session", lambda v: fake)
    return fake


def good(feeds: Any, options: Any) -> Any:
    return [np.zeros((1, A), dtype=np.float32)]


def ort_raises(feeds: Any, options: Any) -> Any:
    raise RuntimeError("<ort>")


def test_smoke_feeds_zeros_then_rng1_normals_with_fresh_run_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timers: list[threading.Timer] = []
    real_timer = threading.Timer

    def recording_timer(*args: Any, **kwargs: Any) -> threading.Timer:
        timer = real_timer(*args, **kwargs)
        timers.append(timer)
        return timer

    monkeypatch.setattr(validate.threading, "Timer", recording_timer)
    fake = with_fake(monkeypatch, good)
    assert smoke_test(validated(PPO)) == []
    assert len(fake.feeds) == validate.SMOKE_CALLS == 32
    assert all(f.shape == (1, D) and f.dtype == np.float32 for f in fake.feeds)
    assert not fake.feeds[0].any()
    normals = np.random.default_rng(1).standard_normal((31, D)).astype(np.float32)
    assert np.array_equal(np.concatenate(fake.feeds[1:]), normals)
    assert len({id(o) for o in fake.options}) == 32  # a fresh RunOptions per call
    assert len(timers) == 32
    assert all(t.finished.is_set() for t in timers)  # every timer cancelled
    assert all(not t.is_alive() for t in timers)


def test_smoke_terminate_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validate, "SMOKE_TIMEOUT_S", 0.05)

    def hang(feeds: Any, options: Any) -> Any:
        for _ in range(200):  # bounded: at most about 2 s
            if options.terminate:
                raise RuntimeError("Exiting due to terminate flag being set to true.")
            threading.Event().wait(0.01)
        raise AssertionError("the timer never set terminate")

    with_fake(monkeypatch, hang)
    assert smoke_test(validated(PPO)) == ["smoke call exceeded 1 s"]


@pytest.mark.parametrize(
    ("behaviour", "message"),
    [
        (ort_raises, "smoke call failed: ort_run_failed"),
        (lambda f, o: [np.zeros((1, A + 1), np.float32)], "smoke call failed: bad_output"),
        (lambda f, o: [np.zeros((1, A), np.float64)], "smoke call failed: bad_output"),
        (lambda f, o: [], "smoke call failed: bad_output"),
        (
            lambda f, o: [np.full((1, A), np.nan, np.float32)],
            "smoke call failed: non_finite_output",
        ),
    ],
    ids=["raises", "width", "dtype", "empty", "nan"],
)
def test_smoke_failures(
    monkeypatch: pytest.MonkeyPatch, behaviour: Callable[[Any, Any], Any], message: str
) -> None:
    with_fake(monkeypatch, behaviour)
    report = smoke_run(validated(PPO))
    assert report.errors == (message,)
    assert report.latency_p99_us is None


def test_smoke_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(v: object) -> None:
        raise RuntimeError("<ort load>")

    monkeypatch.setattr(runtime, "make_session", fail)
    assert smoke_test(validated(PPO)) == ["ort_load_failed"]
