"""The acceptance policy for ONNX policies (docs/design/phase-2.md §3.4).

A ``policy.onnx`` is untrusted bytes. ``validate_policy_bytes`` runs nine checks in order and
returns at the first check that produces any message, so the C++ checker, shape inference and
ONNX Runtime only ever see bytes that passed checks 1-7 (sizes, an allow-list of fields, names,
I/O, ops and tensors): a crafted protobuf never reaches native code that was not written for
hostile input. Nothing here reads external data, calls ``numpy_helper`` or loads from a path.

Every message is a fixed code, then fixed words and names passed through ``safe``: never raw
exception text (an ``UnicodeDecodeError`` message holds the offending bytes) and never markup.

The canonical form makes ``policy_sha256 = sha256(policy.onnx)`` identify the policy: it is
rebuilt from allow-listed parts only, so two exports that differ in names, payload fields,
``Constant`` nodes, default attributes or metadata order give the same bytes, and the loader
accepts only bytes that already are canonical (check 9).
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import struct
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, InitVar, dataclass, field
from functools import partial
from math import prod
from typing import Any, Final, Literal

import numpy as np
import onnx
import onnx.checker
import onnx.shape_inference
from google.protobuf.message import DecodeError, Message
from google.protobuf.unknown_fields import UnknownFieldSet
from onnx import AttributeProto, TensorProto

MAX_ONNX_BYTES: Final = 1 << 20  # a [64, 64] export is about 21 KB
MAX_ELEMENTS: Final = 131_072  # initializers and Constant values together
MAX_COST: Final = 1 << 17  # static cost per batch row; first-party PPO is about 5,200
MAX_NODE_INPUTS: Final = 192  # sum over nodes: closes the variadic Max bomb
MAX_NODES: Final = 64
MAX_INITIALIZERS: Final = 32
MAX_CONSTANT_ELEMENTS: Final = 4096
MAX_DIM: Final = 4096
MAX_RANK: Final = 2
MAX_METADATA_VALUE_BYTES: Final = 1024
MAX_PRODUCER_NAME: Final = 128
MAX_MESSAGES: Final = 8  # per rejection: the first few say what is wrong

# IR <= 10 keeps ONNX Runtime Web >= 1.20 compatible; check 2 excludes every IR 9/10 feature.
CANONICAL_IR: Final[Mapping[int, int]] = {17: 8, 18: 8, 19: 9, 20: 9, 21: 10}
OP_WEIGHT: Final[Mapping[str, int]] = {
    **dict.fromkeys(
        (
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Neg",
            "Relu",
            "LeakyRelu",
            "Clip",
            "Min",
            "Max",
            "Sqrt",
            "Reciprocal",
            "Softsign",
        ),
        1,
    ),
    "Tanh": 2,
    "Sigmoid": 2,
    "Elu": 4,
    "Softplus": 48,
}
ARITY: Final[Mapping[str, tuple[int, int]]] = {
    "Gemm": (2, 3),
    **dict.fromkeys(("MatMul", "Add", "Sub", "Mul", "Div"), (2, 2)),
    **dict.fromkeys(
        (
            "Neg",
            "Sqrt",
            "Reciprocal",
            "Tanh",
            "Relu",
            "LeakyRelu",
            "Elu",
            "Sigmoid",
            "Softsign",
            "Softplus",
        ),
        (1, 1),
    ),
    "Clip": (1, 3),
    **dict.fromkeys(("Min", "Max"), (2, 4)),
    "Constant": (0, 0),
}
ALLOWED_OPS: Final = tuple(ARITY)

_FLOAT: Final = int(AttributeProto.FLOAT)
_INT: Final = int(AttributeProto.INT)
_TENSOR: Final = int(AttributeProto.TENSOR)
_FLOATS: Final = int(AttributeProto.FLOATS)
_PAYLOAD: Final[Mapping[int, str]] = {_FLOAT: "f", _INT: "i", _TENSOR: "t", _FLOATS: "floats"}
ATTRIBUTES: Final[Mapping[str, Mapping[str, int]]] = {
    "Gemm": {"alpha": _FLOAT, "beta": _FLOAT, "transA": _INT, "transB": _INT},
    "LeakyRelu": {"alpha": _FLOAT},
    "Elu": {"alpha": _FLOAT},
    "Constant": {"value": _TENSOR, "value_float": _FLOAT, "value_floats": _FLOATS},
}


def _f32(x: float) -> float:
    """``x`` rounded to float32 and widened back, as an ONNX FLOAT attribute holds it."""
    value: float = struct.unpack("<f", struct.pack("<f", x))[0]
    return value


# (op, attribute) -> (type, value) of every schema default among the allowed ops (opsets 17..21)
SCHEMA_DEFAULTS: Final[Mapping[tuple[str, str], tuple[int, float | int]]] = {
    ("Gemm", "alpha"): (_FLOAT, 1.0),
    ("Gemm", "beta"): (_FLOAT, 1.0),
    ("Gemm", "transA"): (_INT, 0),
    ("Gemm", "transB"): (_INT, 0),
    ("LeakyRelu", "alpha"): (_FLOAT, _f32(0.01)),
    ("Elu", "alpha"): (_FLOAT, 1.0),
}

# Check 2: the only fields a policy may set, per message type. Everything else (doc strings,
# producer_version, value_info, external data, functions, subgraphs, overloads, denotations,
# metadata outside the model, ...) is rejected at load and dropped by pack.
ALLOWED_FIELDS: Final[Mapping[str, frozenset[str]]] = {
    "onnx.ModelProto": frozenset(
        {"ir_version", "opset_import", "producer_name", "graph", "metadata_props"}
    ),
    "onnx.OperatorSetIdProto": frozenset({"domain", "version"}),
    "onnx.GraphProto": frozenset({"node", "name", "initializer", "input", "output"}),
    "onnx.NodeProto": frozenset({"input", "output", "name", "op_type", "attribute"}),
    "onnx.AttributeProto": frozenset({"name", "type", "f", "i", "t", "floats"}),
    "onnx.TensorProto": frozenset({"dims", "data_type", "raw_data", "float_data", "name"}),
    "onnx.ValueInfoProto": frozenset({"name", "type"}),
    "onnx.TypeProto": frozenset({"tensor_type"}),
    "onnx.TypeProto.Tensor": frozenset({"elem_type", "shape"}),
    "onnx.TensorShapeProto": frozenset({"dim"}),
    "onnx.TensorShapeProto.Dimension": frozenset({"dim_value", "dim_param"}),
    "onnx.StringStringEntryProto": frozenset({"key", "value"}),
}
DEFAULT_DOMAINS: Final = ("", "ai.onnx")

POLICY_FORMAT: Final = "topple.policy/1"
METADATA_FIELDS: Final = (
    "format",
    "challenge",
    "obs_schema",
    "obs_features",
    "action_spec",
    "control_dt",
    "action_rule",
)
METADATA_KEYS: Final = tuple(f"topple.{name}" for name in METADATA_FIELDS)
PRODUCER: Final = "topple"
GRAPH_NAME: Final = "topple_policy"
BATCH: Final = "batch"
INPUT_NAME: Final = "obs"
OUTPUT_NAME: Final = "action"

# Every pattern is used with ``fullmatch`` (``re.match`` with ``$`` accepts a trailing "\n").
NAME_RE: Final = re.compile(r"[A-Za-z0-9_.:/-]{1,128}")
OP_TYPE_RE: Final = re.compile(r"[A-Za-z]{1,32}")
# §3.4 says [A-Za-z]{1,32}; "_" is added so ``value_float``/``value_floats``, which check 6
# allows on Constant, reach check 9 ("run topple policy pack") instead of failing here.
ATTRIBUTE_NAME_RE: Final = re.compile(r"[A-Za-z_]{1,32}")
METADATA_KEY_RE: Final = re.compile(r"topple\.[a-z_]{1,32}")
_UNSAFE: Final = re.compile(r"[^A-Za-z0-9_.:/-]")

NOT_CANONICAL: Final = "not_canonical: policy.onnx is not canonical; run topple policy pack"
# The check each rejection code belongs to (codes are the text before the first ":").
CHECK_OF: Final[Mapping[str, int]] = {
    "onnx_too_large": 1,
    "onnx_parse_failed": 1,
    "unknown_field": 2,
    "field_not_allowed": 2,
    "invalid_utf8": 2,
    "opset_domain_not_allowed": 2,
    "bad_opset_import": 3,
    "unsupported_opset": 3,
    "bad_ir_version": 3,
    "unsupported_feature": 3,
    "bad_name": 4,
    "bad_op_type": 4,
    "bad_attribute_name": 4,
    "bad_metadata_key": 4,
    "duplicate_metadata_key": 4,
    "missing_metadata_key": 4,
    "bad_metadata_value": 4,
    "producer_name_too_long": 4,
    "initializer_as_input": 5,
    "bad_input": 5,
    "bad_output": 5,
    "op_not_allowed": 6,
    "bad_arity": 6,
    "bad_output_count": 6,
    "empty_input_name": 6,
    "attribute_not_allowed": 6,
    "duplicate_attribute": 6,
    "bad_attribute_type": 6,
    "bad_attribute_value": 6,
    "nonfinite_attribute": 6,
    "weight_not_constant": 6,
    "too_many_node_inputs": 6,
    "too_many_nodes": 7,
    "too_many_initializers": 7,
    "bad_tensor_type": 7,
    "bad_tensor_payload": 7,
    "bad_tensor_shape": 7,
    "bad_tensor_size": 7,
    "nonfinite_tensor": 7,
    "constant_too_large": 7,
    "too_many_elements": 7,
    "onnx_checker_failed": 8,
    "shape_inference_failed": 8,
    "not_batch_major": 8,
    "unused_initializer": 8,
    "dead_node": 8,
    "cost_exceeded": 8,
    "not_canonical": 9,
}
# An exception escaping a check becomes one of these codes, never its text.
_EXCEPTION_CODES: Final[tuple[tuple[type[BaseException], str], ...]] = (
    (DecodeError, "onnx_parse_failed"),
    (UnicodeError, "invalid_utf8"),
    (RecursionError, "onnx_too_deep"),
    (MemoryError, "out_of_memory"),
    (ValueError, "onnx_invalid"),
    (TypeError, "onnx_invalid"),
)

SMOKE_CALLS: Final = 32
SMOKE_TIMEOUT_S = 1.0  # not Final: tests shorten it
SMOKE_TIMEOUT: Final = "smoke call exceeded 1 s"

_SEAL: Final = object()


@dataclass(frozen=True)
class ValidatedPolicy:
    """Canonical policy bytes that passed checks 1-9, with their identity and static cost.

    Only ``validate_policy_bytes`` can create one (the module-private seal), so holding one
    proves the bytes were validated: ``make_session`` accepts nothing else.
    """

    canonical: bytes = field(repr=False)
    policy_sha256: str
    cost: int
    obs_dim: int
    action_dim: int
    _: KW_ONLY
    seal: InitVar[object]

    def __post_init__(self, seal: object) -> None:
        if seal is not _SEAL:
            raise TypeError("a ValidatedPolicy comes only from validate_policy_bytes")


@dataclass(frozen=True)
class SmokeReport:
    """The smoke test's verdict: ``errors`` empty on success, then the p99 latency of one
    batch-1 call in microseconds (recorded, never an acceptance criterion)."""

    errors: tuple[str, ...]
    latency_p99_us: float | None


def safe(s: str | bytes) -> str:
    """``s`` as at most 64 characters of ``[A-Za-z0-9_.:/-]``, every other character ``?``:
    the only form in which an untrusted name may appear in a message."""
    text = s.decode("utf-8", "replace") if isinstance(s, bytes) else s
    return _UNSAFE.sub("?", text[:64])


def _msg(code: str, detail: str = "") -> str:
    return f"{code}: {detail}" if detail else code


def _exception_message(exc: BaseException, check: int) -> str:
    code = next((c for kind, c in _EXCEPTION_CODES if isinstance(exc, kind)), "validator_error")
    return f"{code}: raised during check {check}"


def _bounded(messages: Iterable[str]) -> list[str]:
    return list(itertools.islice(messages, MAX_MESSAGES))


# --- validate_policy_bytes -------------------------------------------------------------------


def validate_policy_bytes(
    data: bytes, obs_dim: int, action_dim: int
) -> ValidatedPolicy | list[str]:
    """``data`` as a ``ValidatedPolicy`` if it passes checks 1-9 for a policy mapping
    ``obs_dim`` features to ``action_dim`` actions; else the first failing check's messages.

    Never raises for any ``data``: an exception inside a check becomes a fixed code.
    """
    check, cost = 1, 0
    try:
        if not _is_exact_bytes(data):  # a subclass could override the check-9 comparison
            return [_msg("onnx_parse_failed", "policy.onnx must be bytes")]
        if len(data) > MAX_ONNX_BYTES:
            return [_msg("onnx_too_large", f"policy.onnx has more than {MAX_ONNX_BYTES} bytes")]
        try:
            proto: onnx.ModelProto = onnx.ModelProto.FromString(data)
        except DecodeError:
            return [_msg("onnx_parse_failed", "policy.onnx is not an ONNX model")]
        checks: tuple[tuple[int, Callable[[], Iterator[str]]], ...] = (
            (2, partial(_check_fields, proto)),
            (3, partial(_check_header, proto)),
            (4, partial(_check_strings, proto)),
            (5, partial(_check_io, proto, obs_dim, action_dim)),
            (6, partial(_check_ops, proto)),
            (7, partial(_check_tensors, proto)),
        )
        for number, run in checks:
            check = number  # named in the message if the check raises
            errors = _bounded(run())
            if errors:
                return errors
        check = 8
        errors, cost = _check_structure(proto)
        if errors:
            return errors
        check = 9
        if canonicalize(proto) != data:
            return [NOT_CANONICAL]
    except Exception as exc:  # never a traceback or raw exception text (§3.4 "Error text")
        return [_exception_message(exc, check)]
    return ValidatedPolicy(
        canonical=data,
        policy_sha256=hashlib.sha256(data).hexdigest(),
        cost=cost,
        obs_dim=obs_dim,
        action_dim=action_dim,
        seal=_SEAL,
    )


def _is_exact_bytes(data: object) -> bool:
    return type(data) is bytes


# --- check 2: field allow-list ---------------------------------------------------------------


def _is_utf8(value: object) -> bool:
    """A proto2 string upb could decode (it returns ``bytes`` for invalid UTF-8) that also has
    a UTF-8 form (no lone surrogates)."""
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _check_fields(proto: Message) -> Iterator[str]:
    """Check 2: every message has no unknown fields and sets only allow-listed fields, every
    string is valid UTF-8, the opset domain is the default one, and no tensor inside an
    attribute is named. Disallowed fields are reported, never descended into."""
    stack: list[tuple[Message, bool]] = [(proto, False)]
    while stack:
        message, in_attribute = stack.pop()
        kind = message.DESCRIPTOR.full_name
        if len(UnknownFieldSet(message)) > 0:  # msg.UnknownFields() raises on upb protobuf 7
            yield _msg("unknown_field", kind)
        allowed = ALLOWED_FIELDS.get(kind, frozenset())
        inside = in_attribute or kind == "onnx.AttributeProto"
        children: list[Message] = []
        for fd, value in message.ListFields():
            where = f"{kind}.{fd.name}"
            named_in_attribute = in_attribute and kind == "onnx.TensorProto" and fd.name == "name"
            if fd.name not in allowed or named_in_attribute:
                yield _msg("field_not_allowed", where)
            elif fd.type == fd.TYPE_STRING:
                values = value if fd.is_repeated else (value,)
                if not all(_is_utf8(v) for v in values):
                    yield _msg("invalid_utf8", where)
                elif kind == "onnx.OperatorSetIdProto" and value not in DEFAULT_DOMAINS:
                    yield _msg("opset_domain_not_allowed", "only the default domain is allowed")
            elif fd.type == fd.TYPE_MESSAGE:
                children.extend(value if fd.is_repeated else (value,))
        stack.extend((child, inside) for child in reversed(children))


# --- check 3: header -------------------------------------------------------------------------


def _check_header(proto: onnx.ModelProto) -> Iterator[str]:
    """Check 3: one opset import of version 17..21 and the IR version ``CANONICAL_IR`` pins
    for it; no functions, training info or sparse initializers (check 2 already rejects them;
    this keeps the rule if the allow-list ever grows)."""
    if len(proto.opset_import) != 1:
        yield _msg("bad_opset_import", "exactly one opset import is allowed")
        return
    version = proto.opset_import[0].version
    ir = CANONICAL_IR.get(version)
    if ir is None:
        yield _msg("unsupported_opset", f"opset {version} (17..21 are accepted)")
        return
    if proto.ir_version != ir:
        yield _msg("bad_ir_version", f"IR {proto.ir_version} with opset {version} (needs {ir})")
    if len(proto.functions) or len(proto.training_info) or len(proto.graph.sparse_initializer):
        yield _msg("unsupported_feature", "functions, training info and sparse initializers")


# --- check 4: strings ------------------------------------------------------------------------


def _check_strings(proto: onnx.ModelProto) -> Iterator[str]:
    """Check 4: every name matches ``NAME_RE``, op types and attribute names their patterns,
    the metadata is exactly the 7 ``topple.*`` keys with short ASCII values, and the producer
    name is short. After this, no later message can carry markup or a line break."""
    graph = proto.graph
    if not NAME_RE.fullmatch(graph.name):
        yield _msg("bad_name", f"graph {safe(graph.name)!r}")
    for n in graph.node:
        if not NAME_RE.fullmatch(n.name):
            yield _msg("bad_name", f"node {safe(n.name)!r}")
        if not OP_TYPE_RE.fullmatch(n.op_type):
            yield _msg("bad_op_type", repr(safe(n.op_type)))
        for value in n.input:
            if value and not NAME_RE.fullmatch(value):  # an empty input name is check 6's
                yield _msg("bad_name", f"value {safe(value)!r}")
        for value in n.output:
            if not NAME_RE.fullmatch(value):
                yield _msg("bad_name", f"value {safe(value)!r}")
        for a in n.attribute:
            if not ATTRIBUTE_NAME_RE.fullmatch(a.name):
                yield _msg("bad_attribute_name", repr(safe(a.name)))
    for t in graph.initializer:
        if not NAME_RE.fullmatch(t.name):
            yield _msg("bad_name", f"initializer {safe(t.name)!r}")
    for info in (*graph.input, *graph.output):
        if not NAME_RE.fullmatch(info.name):
            yield _msg("bad_name", f"value {safe(info.name)!r}")
    yield from _check_metadata(proto)
    if len(proto.producer_name) > MAX_PRODUCER_NAME:
        yield _msg("producer_name_too_long", f"more than {MAX_PRODUCER_NAME} characters")


def _check_metadata(proto: onnx.ModelProto) -> Iterator[str]:
    seen: set[str] = set()
    for entry in proto.metadata_props:
        key = entry.key
        if not METADATA_KEY_RE.fullmatch(key) or key not in METADATA_KEYS:
            yield _msg("bad_metadata_key", repr(safe(key)))
        elif key in seen:
            yield _msg("duplicate_metadata_key", key)
        else:
            seen.add(key)
            value = entry.value
            if not value.isascii() or len(value) > MAX_METADATA_VALUE_BYTES:
                yield _msg("bad_metadata_value", f"{key} must be ASCII, at most 1024 bytes")
    for key in METADATA_KEYS:
        if key not in seen:
            yield _msg("missing_metadata_key", key)


# --- check 5: inputs and outputs -------------------------------------------------------------


def _batch_symbol(info: onnx.ValueInfoProto, width: int) -> str | None:
    """The symbolic batch dim of a float32 ``[batch, width]`` value, else ``None``."""
    if not info.type.HasField("tensor_type"):
        return None
    tensor = info.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT or len(tensor.shape.dim) != 2:
        return None
    batch, dim = tensor.shape.dim
    if batch.WhichOneof("value") != "dim_param" or not batch.dim_param:
        return None
    if dim.WhichOneof("value") != "dim_value" or dim.dim_value != width:
        return None
    symbol: str = batch.dim_param
    return symbol


def _check_io(proto: onnx.ModelProto, obs_dim: int, action_dim: int) -> Iterator[str]:
    """Check 5: one input ``obs`` float32 ``[sym, obs_dim]``, one output ``action`` float32
    ``[sym, action_dim]`` with the same symbol, and no initializer listed as an input."""
    graph = proto.graph
    initializers = {t.name for t in graph.initializer}
    if any(info.name in initializers for info in graph.input):
        yield _msg("initializer_as_input", "an initializer is listed as a graph input")
        return
    if len(graph.input) != 1 or graph.input[0].name != INPUT_NAME:
        yield _msg("bad_input", "exactly one input, named 'obs'")
        return
    if len(graph.output) != 1 or graph.output[0].name != OUTPUT_NAME:
        yield _msg("bad_output", "exactly one output, named 'action'")
        return
    symbol = _batch_symbol(graph.input[0], obs_dim)
    if symbol is None:
        yield _msg("bad_input", f"'obs' must be float32 [batch, {obs_dim}]")
        return
    if _batch_symbol(graph.output[0], action_dim) != symbol:
        yield _msg("bad_output", f"'action' must be float32 [batch, {action_dim}], batch of 'obs'")


# --- check 6: ops, arity, attributes ---------------------------------------------------------


def _check_ops(proto: onnx.ModelProto) -> Iterator[str]:
    """Check 6: allow-listed ops with their input counts, one output each, no empty input
    names, at most ``MAX_NODE_INPUTS`` inputs in all, allow-listed attributes of the schema
    type with finite floats, and Gemm/MatMul weights that are constants."""
    graph = proto.graph
    constants = {t.name for t in graph.initializer}
    constants |= {n.output[0] for n in graph.node if n.op_type == "Constant" and n.output}
    for n in graph.node:
        op = n.op_type
        arity = ARITY.get(op)
        if arity is None:
            yield _msg("op_not_allowed", repr(safe(op)))
            continue
        low, high = arity
        if not low <= len(n.input) <= high:
            yield _msg("bad_arity", f"{op} has {len(n.input)} inputs ({low}..{high} allowed)")
        if len(n.output) != 1:
            yield _msg("bad_output_count", f"{op} must have exactly one output")
        if not all(n.input):
            yield _msg("empty_input_name", f"{op} has an empty input name")
        yield from _check_attributes(n)
        if op in ("Gemm", "MatMul") and len(n.input) >= 2 and n.input[1] not in constants:
            yield _msg("weight_not_constant", f"{op}'s second input must be a constant")
    total = sum(len(n.input) for n in graph.node)
    if total > MAX_NODE_INPUTS:
        yield _msg("too_many_node_inputs", f"{total} node inputs (at most {MAX_NODE_INPUTS})")


def _check_attributes(n: onnx.NodeProto) -> Iterator[str]:
    op = n.op_type
    allowed = ATTRIBUTES.get(op, {})
    seen: set[str] = set()
    for a in n.attribute:
        where = f"{op}.{safe(a.name)}"
        kind = allowed.get(a.name)
        if a.name in seen:
            yield _msg("duplicate_attribute", where)
        elif kind is None:
            yield _msg("attribute_not_allowed", where)
        elif a.type != kind:
            yield _msg("bad_attribute_type", where)
        elif {fd.name for fd, _ in a.ListFields()} != {"name", "type", _PAYLOAD[kind]}:
            yield _msg("bad_attribute_value", f"{where} must hold exactly one value")
        elif not all(math.isfinite(x) for x in _attribute_floats(a)):
            yield _msg("nonfinite_attribute", where)
        elif op == "Gemm" and a.name == "transA" and a.i != 0:
            yield _msg("bad_attribute_value", "Gemm.transA must be 0")
        elif op == "Gemm" and a.name == "transB" and a.i not in (0, 1):
            yield _msg("bad_attribute_value", "Gemm.transB must be 0 or 1")
        seen.add(a.name)
    if op == "Constant" and len(n.attribute) != 1:
        yield _msg("bad_attribute_value", "a Constant holds exactly one value")


def _attribute_floats(a: AttributeProto) -> Sequence[float]:
    if a.type == _FLOAT:
        return (a.f,)
    if a.type == _FLOATS:
        return tuple(a.floats)
    return ()


# --- check 7: tensors ------------------------------------------------------------------------


def _check_tensors(proto: onnx.ModelProto) -> Iterator[str]:
    """Check 7: at most ``MAX_NODES`` nodes and ``MAX_INITIALIZERS`` initializers; every
    initializer and Constant value float32 with exactly one payload, rank <= 2, dims 1..4096
    and exactly prod(dims) finite values; ``MAX_ELEMENTS`` in all, ``MAX_CONSTANT_ELEMENTS``
    per Constant. Values are read from the raw fields only."""
    graph = proto.graph
    if len(graph.node) > MAX_NODES:
        yield _msg("too_many_nodes", f"{len(graph.node)} nodes (at most {MAX_NODES})")
    if len(graph.initializer) > MAX_INITIALIZERS:
        count = len(graph.initializer)
        yield _msg("too_many_initializers", f"{count} initializers (at most {MAX_INITIALIZERS})")
    total = 0
    for t in graph.initializer:
        problem = _tensor_problem(t)
        if problem:
            yield _msg(problem, f"initializer {safe(t.name)!r}")
            continue
        total += prod(t.dims)
    for n in graph.node:
        if n.op_type != "Constant":
            continue
        for a in n.attribute:
            if a.name == "value":
                problem = _tensor_problem(a.t)
                if problem:
                    yield _msg(problem, f"Constant {safe(n.name)!r}")
                    continue
                count = prod(a.t.dims)
            else:
                count = len(a.floats) if a.name == "value_floats" else 1
            if count > MAX_CONSTANT_ELEMENTS:
                yield _msg("constant_too_large", f"Constant {safe(n.name)!r} has {count} values")
            total += count
    if total > MAX_ELEMENTS:
        yield _msg("too_many_elements", f"{total} values (at most {MAX_ELEMENTS})")


def _tensor_problem(t: TensorProto) -> str | None:
    """The check-7 code for what is wrong with ``t``, else ``None``."""
    if t.data_type != TensorProto.FLOAT:
        return "bad_tensor_type"
    has_raw, has_floats = t.HasField("raw_data"), len(t.float_data) > 0
    if has_raw == has_floats:
        return "bad_tensor_payload"
    dims = list(t.dims)
    if len(dims) > MAX_RANK or not all(1 <= d <= MAX_DIM for d in dims):
        return "bad_tensor_shape"
    count = prod(dims)
    stored = len(t.raw_data) // 4 if has_raw else len(t.float_data)
    if stored != count or (has_raw and len(t.raw_data) % 4):
        return "bad_tensor_size"
    if has_raw:
        values = np.frombuffer(t.raw_data, dtype="<f4")
    else:
        values = np.fromiter(t.float_data, dtype=np.float64, count=count)
    if not np.isfinite(values).all():
        return "nonfinite_tensor"
    return None


# --- check 8: structure and cost -------------------------------------------------------------


def _check_structure(proto: onnx.ModelProto) -> tuple[list[str], int]:
    """Check 8: the ONNX checker and strict shape inference pass; every node output but a
    Constant's is float32 ``[batch, 1..4096]``; every initializer is used and every node
    reaches ``action``; the static cost is at most ``MAX_COST``. Returns (messages, cost)."""
    try:
        onnx.checker.check_model(proto, full_check=True)
    except Exception:
        return [_msg("onnx_checker_failed")], 0
    try:
        inferred = onnx.shape_inference.infer_shapes(proto, strict_mode=True, check_type=True)
    except Exception:
        return [_msg("shape_inference_failed")], 0
    widths = _widths(proto, inferred)
    errors = _bounded(_structure_errors(proto, widths))
    if errors:
        return errors, 0
    cost = _cost(proto.graph, {k: v for k, v in widths.items() if v is not None})
    if cost > MAX_COST:
        return [_msg("cost_exceeded", f"static cost {cost} (at most {MAX_COST})")], cost
    return [], cost


def _widths(proto: onnx.ModelProto, inferred: onnx.ModelProto) -> dict[str, int | None]:
    """dim1 of every non-Constant node output if it is float32 ``[batch, 1..4096]`` with the
    input's batch symbol, else ``None`` (from the shape-inferred copy of ``proto``)."""
    batch = proto.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    infos = {info.name: info for info in (*inferred.graph.value_info, *inferred.graph.output)}
    widths: dict[str, int | None] = {}
    for n in proto.graph.node:
        if n.op_type == "Constant":
            continue
        info = infos.get(n.output[0])
        widths[n.output[0]] = None if info is None else _width(info, batch)
    return widths


def _width(info: onnx.ValueInfoProto, batch: str) -> int | None:
    tensor = info.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT or len(tensor.shape.dim) != 2:
        return None
    first, second = tensor.shape.dim
    batch_major = first.WhichOneof("value") == "dim_param" and first.dim_param == batch
    width: int = second.dim_value if second.WhichOneof("value") == "dim_value" else 0
    return width if batch_major and 1 <= width <= MAX_DIM else None


def _structure_errors(proto: onnx.ModelProto, widths: Mapping[str, int | None]) -> Iterator[str]:
    graph = proto.graph
    for name, width in widths.items():
        if width is None:
            yield _msg("not_batch_major", f"{safe(name)!r} is not float32 [batch, 1..4096]")
    used = {name for n in graph.node for name in n.input}
    for t in graph.initializer:
        if t.name not in used:
            yield _msg("unused_initializer", repr(safe(t.name)))
    needed = {graph.output[0].name}
    live: set[int] = set()
    for i in reversed(range(len(graph.node))):  # topological order: the checker passed
        n = graph.node[i]
        if needed.intersection(n.output):
            live.add(i)
            needed.update(n.input)
    for i, n in enumerate(graph.node):
        if i not in live:
            yield _msg("dead_node", f"{safe(n.name)!r} does not reach 'action'")


def _constant_dims(n: onnx.NodeProto) -> list[int]:
    return list(_constant_value(n).dims)


def _cost(graph: onnx.GraphProto, widths: Mapping[str, int]) -> int:
    """Σ_{Gemm, MatMul} K·N (the weight's element count) + Σ_{other} OP_WEIGHT[op] ·
    max(1, inputs - 1) · width: per batch row, so a variadic Max or a chain of expensive
    unary ops at width 4096 costs what it computes."""
    shapes = {t.name: list(t.dims) for t in graph.initializer}
    shapes |= {n.output[0]: _constant_dims(n) for n in graph.node if n.op_type == "Constant"}
    total = 0
    for n in graph.node:
        op = n.op_type
        if op == "Constant":
            continue
        if op in ("Gemm", "MatMul"):
            total += prod(shapes[n.input[1]])
        else:
            total += OP_WEIGHT[op] * max(1, len(n.input) - 1) * widths[n.output[0]]
    return total


def static_cost(proto: onnx.ModelProto) -> int:
    """The §3.4 static cost of ``proto``, which must have passed checks 1-7 (this runs shape
    inference); ``ValueError`` if a node output is not batch-major."""
    inferred = onnx.shape_inference.infer_shapes(proto, strict_mode=True, check_type=True)
    widths = _widths(proto, inferred)
    known = {name: width for name, width in widths.items() if width is not None}
    if len(known) != len(widths):
        raise ValueError("not_batch_major: a node output is not float32 [batch, 1..4096]")
    return _cost(proto.graph, known)


# --- canonical form --------------------------------------------------------------------------


def _all_messages(proto: Message) -> Iterator[Message]:
    stack = [proto]
    while stack:
        message = stack.pop()
        yield message
        for fd, value in message.ListFields():
            if fd.type == fd.TYPE_MESSAGE:
                stack.extend(value if fd.is_repeated else (value,))


def reject_external_data(proto: onnx.ModelProto) -> None:
    """``ValueError`` if any tensor anywhere in ``proto`` points at external data."""
    for message in _all_messages(proto):
        if not isinstance(message, TensorProto):
            continue
        if message.data_location != TensorProto.DEFAULT or len(message.external_data):
            raise ValueError("external_data: tensors must hold their values inline")


def _text(value: object) -> str:
    """``value`` if it is a string with a UTF-8 form, else ``ValueError`` (pack copies strings
    from models that never passed check 2)."""
    if isinstance(value, str) and _is_utf8(value):
        return value
    raise ValueError("invalid_utf8: a copied string is not valid UTF-8")


def canonicalize(proto: onnx.ModelProto, *, pack: bool = False) -> bytes:
    """The canonical bytes of ``proto``: a fresh model built from allow-listed parts only.

    IR ``CANONICAL_IR[opset]`` with the single default-domain opset, producer ``topple``, graph
    ``topple_policy``, nodes ``n{i}`` with outputs ``v{i}`` (the graph output keeps its name),
    initializers ``c{j}`` in first-use order with every ``Constant`` folded in, each tensor as
    little-endian ``raw_data`` with no name inside attributes, schema-default attributes
    dropped and the rest sorted by name, batch ``dim_param "batch"``, metadata sorted by key;
    serialized deterministically. Node order is kept (ONNX requires topological order). It is a
    fixed point, and it never reads external data, calls ``numpy_helper`` or mutates ``proto``.

    ``pack`` (``topple policy pack``, third-party models) also drops every field outside the
    allow-list instead of refusing it, drops other opset imports and metadata keys other than
    the 7 ``topple.*`` ones, drops graph inputs that name initializers, removes ``Identity``
    nodes, renames the I/O to ``obs``/``action`` and turns a static batch of 1 into ``batch``.
    Either way external data is refused first; anything the form cannot represent (two
    outputs on a node, an undefined or twice-defined value, a node in another domain, an
    unfoldable ``Constant``) is a ``ValueError`` with a fixed code.
    """
    reject_external_data(proto)
    if not pack:
        problem = next(_check_fields(proto), None)
        if problem is not None:
            raise ValueError(problem)
    opset = _opset(proto, pack)
    graph = proto.graph
    initializers = _unique({t.name: t for t in graph.initializer}, len(graph.initializer))
    alias, folded, kept = _fold(graph.node, pack)
    inputs = [info for info in graph.input if not (pack and info.name in initializers)]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ValueError("io_count: exactly one graph input and one graph output")
    source, target = inputs[0], graph.output[0]
    _check_definitions(source.name, initializers, folded, alias, kept)

    def resolve(name: str) -> str:
        for _ in range(len(alias) + 1):
            if name not in alias:
                return name
            name = alias[name]
        raise ValueError("identity_cycle: Identity nodes form a cycle")

    in_name = INPUT_NAME if pack else _text(source.name)
    out_name = OUTPUT_NAME if pack else _text(target.name)
    produced = resolve(target.name)
    if produced not in {n.output[0] for n in kept}:
        raise ValueError("output_not_computed: no node computes the graph output")
    rename = {source.name: in_name}
    tensors: list[TensorProto] = []

    def use(name: str) -> str:
        if not name:
            return ""
        name = resolve(name)
        if name in rename:
            return rename[name]
        original = initializers[name] if name in initializers else folded.get(name)
        if original is None:
            raise ValueError("undefined_value: a node reads a value defined nowhere before it")
        rename[name] = f"c{len(tensors)}"
        tensors.append(_canonical_tensor(original, rename[name]))
        return rename[name]

    nodes = []
    for i, n in enumerate(kept):
        new_inputs = [use(name) for name in n.input]
        output = out_name if n.output[0] == produced else f"v{i}"
        rename[n.output[0]] = output
        nodes.append(_canonical_node(n, f"n{i}", new_inputs, output))
    for name in (*initializers, *folded):  # unused ones last, so check 8 still sees them
        use(name)

    model = onnx.ModelProto()
    model.ir_version = CANONICAL_IR[opset]
    opset_import = model.opset_import.add()
    opset_import.domain = ""
    opset_import.version = opset
    model.producer_name = PRODUCER
    model.graph.node.extend(nodes)
    model.graph.name = GRAPH_NAME
    model.graph.initializer.extend(tensors)
    model.graph.input.append(_canonical_value_info(source, in_name, pack))
    model.graph.output.append(_canonical_value_info(target, out_name, pack))
    for key, value in _canonical_metadata(proto, pack):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    data: bytes = model.SerializeToString(deterministic=True)
    return data


def _opset(proto: onnx.ModelProto, pack: bool) -> int:
    defaults = [o for o in proto.opset_import if o.domain in DEFAULT_DOMAINS]
    if len(defaults) != 1 or (not pack and len(proto.opset_import) != 1):
        raise ValueError("bad_opset_import: exactly one default-domain opset import")
    version: int = defaults[0].version
    if version not in CANONICAL_IR:
        raise ValueError("unsupported_opset: opsets 17..21 are accepted")
    return version


def _unique(by_name: dict[str, TensorProto], count: int) -> dict[str, TensorProto]:
    if len(by_name) != count:
        raise ValueError("duplicate_value: two initializers share a name")
    return by_name


def _fold(
    nodes: Iterable[onnx.NodeProto], pack: bool
) -> tuple[dict[str, str], dict[str, TensorProto], list[onnx.NodeProto]]:
    """(Identity output -> its input, Constant output -> its value, the other nodes)."""
    alias: dict[str, str] = {}
    folded: dict[str, TensorProto] = {}
    kept: list[onnx.NodeProto] = []
    for n in nodes:
        if n.domain not in DEFAULT_DOMAINS:  # non-pack: check 2 already refused any domain
            raise ValueError("custom_domain: a node is outside the default ONNX domain")
        if len(n.output) != 1:
            raise ValueError("node_outputs: every node must have exactly one output")
        if n.op_type == "Constant":
            folded[n.output[0]] = _constant_value(n)
        elif pack and n.op_type == "Identity" and len(n.input) == 1:
            alias[n.output[0]] = n.input[0]
        else:
            kept.append(n)
    return alias, folded, kept


def _check_definitions(
    graph_input: str,
    initializers: Mapping[str, TensorProto],
    folded: Mapping[str, TensorProto],
    alias: Mapping[str, str],
    kept: Sequence[onnx.NodeProto],
) -> None:
    """``ValueError`` unless every value is defined exactly once (graph input, initializer,
    Constant, Identity or node output)."""
    names = [graph_input, *initializers, *folded, *alias, *(n.output[0] for n in kept)]
    if len(set(names)) != len(names) or "" in names:
        raise ValueError("duplicate_value: a value is defined twice")


def _constant_value(n: onnx.NodeProto) -> TensorProto:
    """The value of a Constant node as a tensor (``value_float`` rank 0, ``value_floats``
    rank 1); ``ValueError`` for any other form."""
    if len(n.input) or len(n.attribute) != 1:
        raise ValueError("constant_value: a Constant holds exactly one value and no input")
    a = n.attribute[0]
    if a.name == "value" and a.type == _TENSOR:
        value: TensorProto = a.t
        return value
    if a.name == "value_float" and a.type == _FLOAT:
        return TensorProto(data_type=TensorProto.FLOAT, raw_data=struct.pack("<f", a.f))
    if a.name == "value_floats" and a.type == _FLOATS:
        raw = struct.pack(f"<{len(a.floats)}f", *a.floats)
        return TensorProto(dims=[len(a.floats)], data_type=TensorProto.FLOAT, raw_data=raw)
    raise ValueError("constant_value: only value, value_float and value_floats fold")


def _canonical_tensor(t: TensorProto, name: str | None) -> TensorProto:
    """``t``'s dims, type and values as little-endian ``raw_data``, named ``name`` (or not)."""
    if t.HasField("raw_data") and len(t.float_data):
        raise ValueError("both_payloads: a tensor sets both raw_data and float_data")
    out = TensorProto()
    out.dims.extend(t.dims)
    out.data_type = t.data_type
    if name is not None:
        out.name = name
    if t.HasField("raw_data"):
        out.raw_data = t.raw_data
    else:
        out.raw_data = struct.pack(f"<{len(t.float_data)}f", *t.float_data)
    return out


def _canonical_node(n: onnx.NodeProto, name: str, inputs: list[str], output: str) -> onnx.NodeProto:
    out = onnx.NodeProto()
    out.input.extend(inputs)
    out.output.append(output)
    out.name = name
    out.op_type = _text(n.op_type)
    out.attribute.extend(sorted(_canonical_attributes(n), key=lambda a: a.name))
    return out


def _canonical_attributes(n: onnx.NodeProto) -> Iterator[AttributeProto]:
    for a in n.attribute:
        if _is_default(n.op_type, a):
            continue
        out = AttributeProto()
        out.name = _text(a.name)
        out.type = a.type
        if a.HasField("f"):
            out.f = a.f
        if a.HasField("i"):
            out.i = a.i
        if a.HasField("t"):
            out.t.CopyFrom(_canonical_tensor(a.t, None))
        out.floats.extend(a.floats)
        yield out


def _is_default(op: str, a: AttributeProto) -> bool:
    """Whether ``a`` only restates its schema default (so canonical form leaves it out)."""
    default = SCHEMA_DEFAULTS.get((op, a.name))
    if default is None:
        return False
    kind, value = default
    payload = _PAYLOAD[kind]
    fields = {fd.name for fd, _ in a.ListFields()}
    return a.type == kind and fields == {"name", "type", payload} and getattr(a, payload) == value


def _canonical_value_info(info: onnx.ValueInfoProto, name: str, pack: bool) -> onnx.ValueInfoProto:
    out = onnx.ValueInfoProto()
    out.name = name
    if not info.type.HasField("tensor_type"):
        return out
    tensor = info.type.tensor_type
    out_tensor = out.type.tensor_type
    out_tensor.elem_type = tensor.elem_type
    if not tensor.HasField("shape"):
        return out
    out_tensor.shape.SetInParent()
    for k, dim in enumerate(tensor.shape.dim):
        out_dim = out_tensor.shape.dim.add()
        which = dim.WhichOneof("value")
        if which == "dim_param":
            out_dim.dim_param = BATCH if k == 0 else _text(dim.dim_param)
        elif which == "dim_value":
            if pack and k == 0 and dim.dim_value == 1:
                out_dim.dim_param = BATCH
            else:
                out_dim.dim_value = dim.dim_value
    return out


def _canonical_metadata(proto: onnx.ModelProto, pack: bool) -> list[tuple[str, str]]:
    entries = [(_text(e.key), _text(e.value)) for e in proto.metadata_props]
    if pack:
        entries = [(k, v) for k, v in entries if k in METADATA_KEYS]
    return sorted(entries, key=lambda kv: kv[0])


# --- structure match -------------------------------------------------------------------------

_SEQUENCES: Final[Mapping[str, tuple[str, ...]]] = {
    "clip(mean)": ("Sub", "Mul", "Clip", "Gemm", "Tanh", "Gemm", "Tanh", "Gemm", "Clip"),
    "tanh(mean)": ("Sub", "Mul", "Clip", "Gemm", "Relu", "Gemm", "Relu", "Gemm", "Tanh", "Clip"),
}
_CONSTANT_INPUTS: Final[Mapping[str, int]] = {"Sub": 1, "Mul": 1, "Clip": 2, "Gemm": 2}
_ONE: Final = struct.pack("<f", 1.0)
_MINUS_ONE: Final = struct.pack("<f", -1.0)


def structure_match(proto: onnx.ModelProto) -> Literal["clip(mean)", "tanh(mean)", "other"]:
    """The §3.2 sequence ``proto`` (canonical) is, else ``"other"``.

    A match needs the op types, the wiring (each node reads the previous value and the next
    initializers in §3.2 order), the attributes (only Gemm ``transB = 1``), the §3.2
    initializer shapes with D, H1, H2 and A read from the graph, head Clip bounds bitwise
    -1.0 / +1.0 and a symmetric, finite normalization Clip. The Phase 8 C path reads the
    initializers by position, so anything else is ``"other"``.
    """
    ops = tuple(n.op_type for n in proto.graph.node)
    if ops == _SEQUENCES["clip(mean)"] and _matches(proto):
        return "clip(mean)"
    if ops == _SEQUENCES["tanh(mean)"] and _matches(proto):
        return "tanh(mean)"
    return "other"


def _static_width(info: onnx.ValueInfoProto) -> int | None:
    dims = info.type.tensor_type.shape.dim
    if len(dims) != 2 or dims[1].WhichOneof("value") != "dim_value":
        return None
    width: int = dims[1].dim_value
    return width


def _matches(proto: onnx.ModelProto) -> bool:
    graph = proto.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        return False
    d, a = _static_width(graph.input[0]), _static_width(graph.output[0])
    tensors = list(graph.initializer)
    if d is None or a is None or [t.name for t in tensors] != [f"c{j}" for j in range(12)]:
        return False
    h1 = tensors[4].dims[0] if len(tensors[4].dims) == 2 else -1
    h2 = tensors[6].dims[0] if len(tensors[6].dims) == 2 else -1
    shapes = [[d], [d], [], [], [h1, d], [h1], [h2, h1], [h2], [a, h2], [a], [], []]
    if [list(t.dims) for t in tensors] != shapes:
        return False
    for t in tensors:
        if t.data_type != TensorProto.FLOAT or len(t.raw_data) != 4 * prod(t.dims):
            return False
    return _wired(graph) and _bounds_match(tensors)


def _wired(graph: onnx.GraphProto) -> bool:
    previous, j = graph.input[0].name, 0
    last = len(graph.node) - 1
    for i, n in enumerate(graph.node):
        count = _CONSTANT_INPUTS.get(n.op_type, 0)
        expected = [previous, *(f"c{j + k}" for k in range(count))]
        output = graph.output[0].name if i == last else f"v{i}"
        if list(n.input) != expected or list(n.output) != [output]:
            return False
        if not _attributes_match(n):
            return False
        previous, j = output, j + count
    return True


def _attributes_match(n: onnx.NodeProto) -> bool:
    if n.op_type != "Gemm":
        return len(n.attribute) == 0
    if len(n.attribute) != 1:
        return False
    a = n.attribute[0]
    fields = {fd.name for fd, _ in a.ListFields()}
    return a.name == "transB" and a.type == _INT and fields == {"name", "type", "i"} and a.i == 1


def _bounds_match(tensors: Sequence[TensorProto]) -> bool:
    if tensors[10].raw_data != _MINUS_ONE or tensors[11].raw_data != _ONE:
        return False
    (lo,) = struct.unpack("<f", tensors[2].raw_data)
    (hi,) = struct.unpack("<f", tensors[3].raw_data)
    return math.isfinite(lo) and math.isfinite(hi) and lo == -hi


# --- smoke test ------------------------------------------------------------------------------


def smoke_test(v: ValidatedPolicy) -> list[str]:
    """``smoke_run``'s errors: empty if the policy loads and its 32 smoke calls succeed."""
    return list(smoke_run(v).errors)


def smoke_run(v: ValidatedPolicy) -> SmokeReport:
    """32 batch-1 calls (zeros, then ``default_rng(1)`` normals) through ``make_session``;
    each output must be float32 ``(1, action_dim)`` and finite.

    Each call gets a fresh ``RunOptions`` and a ``threading.Timer`` that sets ``terminate``
    after ``SMOKE_TIMEOUT_S``, cancelled when the call returns. ORT checks ``terminate`` only
    between kernels, so this is not a bound: the static cost is the acceptance bound and the
    Phase 3 process kill the runtime one.
    """
    import onnxruntime as ort

    from topple.policy import runtime  # runtime imports this module

    try:
        session = runtime.make_session(v)
    except Exception:
        return SmokeReport(("ort_load_failed",), None)
    rows = np.zeros((SMOKE_CALLS, v.obs_dim), dtype=np.float32)
    rng = np.random.default_rng(1)
    rows[1:] = rng.standard_normal((SMOKE_CALLS - 1, v.obs_dim)).astype(np.float32)
    latencies: list[float] = []
    for row in rows:
        options = ort.RunOptions()
        timer = threading.Timer(SMOKE_TIMEOUT_S, _terminate, args=(options,))
        timer.daemon = True
        timer.start()
        try:
            start = time.perf_counter()
            outputs = session.run([OUTPUT_NAME], {INPUT_NAME: row[None, :]}, options)
            latencies.append(time.perf_counter() - start)
        except Exception:
            failed = SMOKE_TIMEOUT if options.terminate else "smoke call failed: ort_run_failed"
            return SmokeReport((failed,), None)
        finally:
            timer.cancel()
            timer.join()
        problem = _output_problem(outputs, v.action_dim)
        if problem:
            return SmokeReport((f"smoke call failed: {problem}",), None)
    return SmokeReport((), float(np.percentile(latencies, 99)) * 1e6)


def _terminate(options: Any) -> None:
    options.terminate = True


def _output_problem(outputs: object, action_dim: int) -> str | None:
    if not isinstance(outputs, list) or len(outputs) != 1:
        return "bad_output"
    out = outputs[0]
    if not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.shape != (1, action_dim):
        return "bad_output"
    if not np.isfinite(out).all():
        return "non_finite_output"
    return None
