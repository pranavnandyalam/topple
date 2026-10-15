"""Canonical form (docs/design/phase-2.md §3.4 "Canonical form", §6 ``test_policy_canonical``).

``policy_sha256`` identifies bytes, so every non-semantic choice an exporter makes (names, payload
field, Constant nodes, default attributes, field order, metadata order) must collapse to one
byte string, built constructively from allow-listed parts only.

Deferred to slice 1b: "non-canonical file rejected at load" through ``load_policy_dir`` (here
``validate_policy_bytes`` rejects it at check 9), and the third-party torch fixtures of
``test_policy_pack``.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import onnx.defs
import onnx.external_data_helper
import onnx.numpy_helper
import pytest
from onnx import AttributeProto, TensorProto, helper

from policy_support import (
    A,
    D,
    build,
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
from topple.policy.validate import (
    CANONICAL_IR,
    METADATA_KEYS,
    SCHEMA_DEFAULTS,
    canonicalize,
    structure_match,
    validate_policy_bytes,
)

PPO = ppo_bytes()
SAC = sac_bytes()
FIXTURE = Path(__file__).parent / "data" / "noncanonical_policy.onnx"
# pack(noncanonical_policy.onnx) is exactly the tiny PPO build (test_policy_build's golden)
GOLDEN_PACKED_SHA256 = "54b4a9f10cf126f62fbd9b7c53ec57267d05dc075a75e662d004a7e60c9364d8"


def edited(data: bytes, edit: Callable[[onnx.ModelProto], None]) -> onnx.ModelProto:
    proto = parse(data)
    edit(proto)
    return parse(serialize(proto))


def raw_to_float_data(t: TensorProto) -> None:
    values = np.frombuffer(t.raw_data, "<f4").tolist()
    t.ClearField("raw_data")
    t.float_data.extend(values)


# --- fixed point and payloads ----------------------------------------------------------------


@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("data", [PPO, SAC], ids=["ppo", "sac"])
def test_fixed_point(data: bytes, pack: bool) -> None:
    once = canonicalize(parse(data), pack=pack)
    assert once == data
    assert canonicalize(parse(once), pack=pack) == once


def test_fixed_point_of_a_packed_third_party_model() -> None:
    once = canonicalize(parse(FIXTURE.read_bytes()), pack=True)
    assert canonicalize(parse(once), pack=True) == once
    assert canonicalize(parse(once)) == once


def test_float_data_and_raw_data_give_equal_bytes() -> None:
    def to_float_data(p: onnx.ModelProto) -> None:
        for t in p.graph.initializer:
            raw_to_float_data(t)

    proto = edited(PPO, to_float_data)
    assert all(not t.HasField("raw_data") for t in proto.graph.initializer)
    assert canonicalize(proto) == PPO


def test_renamed_values_and_initializers_canonicalize_identically() -> None:
    def rename(p: onnx.ModelProto) -> None:
        names = {f"v{i}": f"/layers/{i}/out;x" for i in range(8)}
        names |= {f"c{j}": f"param.{11 - j}" for j in range(12)}
        for i, n in enumerate(p.graph.node):
            n.name = f"/node/{i}"
            n.input[:] = [names.get(x, x) for x in n.input]
            n.output[:] = [names.get(x, x) for x in n.output]
        for t in p.graph.initializer:
            t.name = names[t.name]
        reordered = sorted(p.graph.initializer, key=lambda t: t.name)
        del p.graph.initializer[:]
        p.graph.initializer.extend(reordered)

    proto = edited(PPO, rename)
    assert serialize(proto) != PPO
    assert canonicalize(proto) == PPO


def test_constants_fold_to_initializers_in_first_use_order() -> None:
    """mean as ``value`` (a named float_data tensor), the clip bounds as ``value_float``, the
    head bounds as rank-0 ``value`` tensors, all interleaved: canonical bytes are the build's."""

    def fold(p: onnx.ModelProto, inner_name: str | None) -> None:
        inits = {t.name: t for t in p.graph.initializer}
        mean = _unnamed(inits["c0"])
        if inner_name is not None:
            mean.name = inner_name
        raw_to_float_data(mean)
        lo = struct.unpack("<f", inits["c2"].raw_data)[0]
        hi = struct.unpack("<f", inits["c3"].raw_data)[0]
        constants = {
            "c0": helper.make_node("Constant", [], ["k_mean"], value=mean),
            "c2": helper.make_node("Constant", [], ["k_lo"], value_float=lo),
            "c3": helper.make_node("Constant", [], ["k_hi"], value_float=hi),
            "c10": helper.make_node("Constant", [], ["k_alo"], value=_unnamed(inits["c10"])),
            "c11": helper.make_node("Constant", [], ["k_ahi"], value=_unnamed(inits["c11"])),
        }
        nodes = list(p.graph.node)
        out: list[onnx.NodeProto] = [constants["c3"], constants["c11"]]  # defined early
        for n in nodes:
            for name in ("c0", "c2", "c10"):
                if name in n.input:
                    out.append(constants[name])
            n.input[:] = [constants[x].output[0] if x in constants else x for x in n.input]
            out.append(n)
        del p.graph.node[:]
        p.graph.node.extend(out)
        keep = [t for t in p.graph.initializer if t.name not in constants]
        del p.graph.initializer[:]
        p.graph.initializer.extend(keep)

    proto = edited(PPO, lambda p: fold(p, None))
    assert sum(n.op_type == "Constant" for n in proto.graph.node) == 5
    assert canonicalize(proto) == PPO
    named = edited(PPO, lambda p: fold(p, "inner"))  # a name inside an attribute: pack only
    with pytest.raises(ValueError, match="field_not_allowed"):
        canonicalize(named)
    assert canonicalize(named, pack=True) == PPO


def _unnamed(t: TensorProto) -> TensorProto:
    out = TensorProto()
    out.CopyFrom(t)
    out.ClearField("name")
    return out


def test_value_floats_constant_folds_to_a_rank_1_initializer() -> None:
    nodes = [
        helper.make_node("Constant", [], ["k"], value_floats=[1.0, 2.0]),
        node("Add", ["obs", "k"], "action"),
    ]
    proto = graph_model(nodes, obs_dim=2, action_dim=2)
    out = parse(canonicalize(proto))
    (init,) = out.graph.initializer
    assert (init.name, list(init.dims)) == ("c0", [2])
    assert init.raw_data == struct.pack("<2f", 1.0, 2.0)
    assert [n.op_type for n in out.graph.node] == ["Add"]


# --- attributes ------------------------------------------------------------------------------


def test_default_gemm_attributes_are_dropped() -> None:
    def explicit(p: onnx.ModelProto) -> None:
        for n in p.graph.node:
            if n.op_type == "Gemm":
                n.attribute.extend(
                    [
                        helper.make_attribute("alpha", 1.0),
                        helper.make_attribute("beta", 1.0),
                        helper.make_attribute("transA", 0),
                    ]
                )

    assert canonicalize(edited(PPO, explicit)) == PPO


def test_explicit_transb_0_equals_omitted() -> None:
    def make(explicit: bool) -> onnx.ModelProto:
        attrs = {"transB": 0} if explicit else {}
        nodes = [node("Gemm", ["obs", "w"], "action", **attrs)]
        return graph_model(nodes, [tensor("w", np.ones((D, A)))])

    assert canonicalize(make(True)) == canonicalize(make(False))
    assert not parse(canonicalize(make(True))).graph.node[0].attribute


def test_non_default_attributes_are_kept_and_sorted_by_name() -> None:
    gemm = node("Gemm", ["obs", "w"], "action")
    gemm.attribute.extend(
        [
            helper.make_attribute("transB", 1),
            helper.make_attribute("beta", 0.5),
            helper.make_attribute("alpha", 2.0),
        ]
    )
    out = parse(canonicalize(graph_model([gemm], [tensor("w", np.ones((A, D)))])))
    assert [(a.name, a.type) for a in out.graph.node[0].attribute] == [
        ("alpha", AttributeProto.FLOAT),
        ("beta", AttributeProto.FLOAT),
        ("transB", AttributeProto.INT),
    ]


@pytest.mark.parametrize(
    ("op", "value", "kept"),
    [("LeakyRelu", 0.01, False), ("LeakyRelu", 0.2, True), ("Elu", 1.0, False), ("Elu", 0.5, True)],
)
def test_activation_defaults(op: str, value: float, kept: bool) -> None:
    nodes = [node(op, ["obs"], "action", alpha=value)]
    out = parse(canonicalize(graph_model(nodes, obs_dim=A, action_dim=A)))
    assert bool(out.graph.node[0].attribute) is kept


@pytest.mark.parametrize("opset", sorted(CANONICAL_IR))
def test_schema_default_table_matches_onnx(opset: int) -> None:
    from topple.policy.validate import ALLOWED_OPS

    for op in ALLOWED_OPS:
        schema = onnx.defs.get_schema(op, opset)
        for name, attribute in schema.attributes.items():
            default = attribute.default_value
            if default.type == AttributeProto.UNDEFINED:
                assert (op, name) not in SCHEMA_DEFAULTS
                continue
            value = helper.get_attribute_value(default)
            assert SCHEMA_DEFAULTS[(op, name)] == (default.type, value), (op, name)


# --- header, I/O and metadata ----------------------------------------------------------------


@pytest.mark.parametrize("opset", sorted(CANONICAL_IR))
def test_ir_version_is_set_from_canonical_ir(opset: int) -> None:
    def header(p: onnx.ModelProto) -> None:
        p.opset_import[0].version = opset
        p.ir_version = 7

    out = parse(canonicalize(edited(PPO, header)))
    assert out.ir_version == CANONICAL_IR[opset]
    assert [(o.domain, o.version) for o in out.opset_import] == [("", opset)]


def test_metadata_is_sorted_by_key() -> None:
    def shuffle(p: onnx.ModelProto) -> None:
        entries = [(e.key, e.value) for e in p.metadata_props]
        del p.metadata_props[:]
        for k, v in entries[3:] + entries[:3]:
            entry = p.metadata_props.add()
            entry.key, entry.value = k, v

    assert canonicalize(edited(PPO, shuffle)) == PPO


def test_batch_symbol_becomes_batch() -> None:
    def rename(p: onnx.ModelProto) -> None:
        for info in (p.graph.input[0], p.graph.output[0]):
            info.type.tensor_type.shape.dim[0].dim_param = "N"

    assert canonicalize(edited(PPO, rename)) == PPO


def test_canonicalize_does_not_mutate_its_input() -> None:
    proto = parse(FIXTURE.read_bytes())
    before = serialize(proto)
    canonicalize(proto, pack=True)
    assert serialize(proto) == before


# --- golden and load -------------------------------------------------------------------------


def test_golden_noncanonical_fixture_packs_to_the_pinned_sha() -> None:
    data = FIXTURE.read_bytes()
    packed = canonicalize(parse(data), pack=True)
    assert hashlib.sha256(packed).hexdigest() == GOLDEN_PACKED_SHA256
    assert packed == PPO
    assert validated(packed).policy_sha256 == GOLDEN_PACKED_SHA256
    assert structure_match(parse(packed)) == "clip(mean)"


def test_the_noncanonical_fixture_itself_is_rejected() -> None:
    result = validate_policy_bytes(FIXTURE.read_bytes(), D, A)
    assert rejected_at(result)[0] == 2


def test_fixture_generator_reproduces_the_committed_file() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_policy_fixtures", FIXTURE.parent / "make_policy_fixtures.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.noncanonical_policy()
    assert model.SerializeToString(deterministic=True) == FIXTURE.read_bytes()


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: raw_to_float_data(p.graph.initializer[0]),
        lambda p: setattr(p.graph.node[0], "name", "first"),
        lambda p: setattr(p, "producer_name", "pytorch"),
    ],
    ids=["float_data", "node_name", "producer"],
)
def test_non_canonical_bytes_are_rejected_at_check_9(
    edit: Callable[[onnx.ModelProto], None],
) -> None:
    proto = edited(PPO, edit)
    assert canonicalize(proto) == PPO
    result = validate_policy_bytes(serialize(proto), D, A)
    assert rejected_at(result) == (9, "not_canonical")
    assert result == ["not_canonical: policy.onnx is not canonical; run topple policy pack"]


# --- pack mode -------------------------------------------------------------------------------


def test_pack_removes_foreign_content() -> None:
    out = parse(canonicalize(parse(FIXTURE.read_bytes()), pack=True))
    assert not out.HasField("producer_version")
    assert not out.HasField("doc_string")
    assert not out.HasField("domain")
    assert not out.graph.value_info
    assert [e.key for e in out.metadata_props] == sorted(METADATA_KEYS)
    assert [n.op_type for n in out.graph.node].count("Identity") == 0
    assert [(i.name, o.name) for i, o in zip(out.graph.input, out.graph.output, strict=True)] == [
        ("obs", "action")
    ]


def test_pack_renames_io_and_turns_a_static_batch_into_batch() -> None:
    nodes = [node("Gemm", ["x", "w"], "y", transB=1)]
    graph = helper.make_graph(
        nodes,
        "g",
        [value_info("x", [1, D])],
        [value_info("y", [1, A])],
        [tensor("w", np.ones((A, D)))],
    )
    proto = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])
    out = parse(canonicalize(proto, pack=True))
    for info, width in ((out.graph.input[0], D), (out.graph.output[0], A)):
        dims = [d.dim_param or d.dim_value for d in info.type.tensor_type.shape.dim]
        assert dims == ["batch", width]
    assert (out.graph.input[0].name, out.graph.output[0].name) == ("obs", "action")
    assert list(out.graph.node[0].input) == ["obs", "c0"]
    assert list(out.graph.node[0].output) == ["action"]
    assert not out.metadata_props  # pack adds the 7 keys afterwards (slice 1b)


def test_non_pack_keeps_a_static_batch_and_io_names() -> None:
    def static(p: onnx.ModelProto) -> None:
        p.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

    out = parse(canonicalize(edited(PPO, static)))
    assert out.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1


def test_pack_removes_identity_chains() -> None:
    nodes = [
        node("Identity", ["obs"], "i0"),
        node("Gemm", ["i0", "w"], "g", transB=1),
        node("Identity", ["g"], "i1"),
        node("Identity", ["i1"], "action"),
    ]
    out = parse(canonicalize(graph_model(nodes, [tensor("w", np.ones((A, D)))]), pack=True))
    assert [(n.op_type, list(n.input), list(n.output)) for n in out.graph.node] == [
        ("Gemm", ["obs", "c0"], ["action"])
    ]


def test_non_pack_rejects_foreign_fields() -> None:
    proto = edited(PPO, lambda p: setattr(p, "doc_string", "x"))
    with pytest.raises(ValueError, match="field_not_allowed"):
        canonicalize(proto)
    assert canonicalize(proto, pack=True) == PPO


def test_pack_rejects_external_data_before_any_tensor_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("tensor read")

    monkeypatch.setattr(onnx.numpy_helper, "to_array", forbidden)
    monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_model", forbidden)

    def external(p: onnx.ModelProto) -> None:
        p.graph.initializer[3].data_location = TensorProto.EXTERNAL

    with pytest.raises(ValueError, match="external_data"):
        canonicalize(edited(PPO, external), pack=True)


def test_pack_rejects_custom_domain_nodes() -> None:
    proto = edited(PPO, lambda p: setattr(p.graph.node[4], "domain", "com.evil"))
    with pytest.raises(ValueError, match="custom_domain"):
        canonicalize(proto, pack=True)


def test_pack_keeps_only_the_default_opset_import() -> None:
    proto = edited(PPO, lambda p: p.opset_import.append(helper.make_opsetid("ai.onnx.ml", 3)))
    assert canonicalize(proto, pack=True) == PPO


# --- what canonicalize refuses ---------------------------------------------------------------


def two_inputs(p: onnx.ModelProto) -> None:
    p.graph.input.append(value_info("extra", ["batch", 1]))


def two_outputs(p: onnx.ModelProto) -> None:
    p.graph.node[4].output.append("extra")


def duplicate_output(p: onnx.ModelProto) -> None:
    p.graph.node[4].output[0] = "v0"


def undefined_input(p: onnx.ModelProto) -> None:
    p.graph.node[4].input[0] = "nowhere"


def output_is_input(p: onnx.ModelProto) -> None:
    p.graph.output[0].name = "obs"


def both_payloads(p: onnx.ModelProto) -> None:
    t = p.graph.initializer[0]
    t.float_data.extend(np.frombuffer(t.raw_data, "<f4").tolist())


def empty_constant(p: onnx.ModelProto) -> None:
    p.graph.node.insert(0, helper.make_node("Constant", [], ["k"]))


def int_constant(p: onnx.ModelProto) -> None:
    p.graph.node.insert(0, helper.make_node("Constant", [], ["k"], value_int=3))


def opset_16(p: onnx.ModelProto) -> None:
    p.opset_import[0].version = 16


def no_default_opset(p: onnx.ModelProto) -> None:
    p.opset_import[0].domain = "com.evil"


def duplicate_initializer(p: onnx.ModelProto) -> None:
    p.graph.initializer.append(tensor("c0", np.zeros(D)))


@pytest.mark.parametrize(
    "edit",
    [
        two_inputs,
        two_outputs,
        duplicate_output,
        undefined_input,
        output_is_input,
        both_payloads,
        empty_constant,
        int_constant,
        opset_16,
        no_default_opset,
        duplicate_initializer,
    ],
)
@pytest.mark.parametrize("pack", [False, True])
def test_canonicalize_refuses(edit: Callable[[onnx.ModelProto], None], pack: bool) -> None:
    proto = edited(PPO, edit)
    with pytest.raises(ValueError):  # noqa: PT011 - fixed codes, checked per case below
        canonicalize(proto, pack=pack)


def test_pack_refuses_an_identity_cycle() -> None:
    proto = parse(PPO)
    proto.graph.node.extend([node("Identity", ["b"], "a"), node("Identity", ["a"], "b")])
    proto.graph.node[4].input[0] = "a"
    with pytest.raises(ValueError, match="identity_cycle"):
        canonicalize(proto, pack=True)


def test_an_empty_optional_input_is_kept_for_check_6() -> None:
    proto = parse(PPO)
    proto.graph.node[2].input[1] = ""
    out = parse(canonicalize(proto))
    assert list(out.graph.node[2].input) == ["v1", "", "c2"]
    assert rejected_at(validate_policy_bytes(serialize(out), D, A)) == (6, "empty_input_name")


def test_pack_keeps_a_tensor_attribute_without_its_name() -> None:
    """A tensor attribute on another op (rejected later at check 6) loses its name like a
    Constant's value would."""
    proto = parse(PPO)
    value = tensor("named", [0.0])
    proto.graph.node[4].attribute.append(helper.make_attribute("value", value))
    out = parse(canonicalize(proto, pack=True))
    (kept,) = out.graph.node[4].attribute
    assert kept.t.raw_data == struct.pack("<f", 0.0)
    assert not kept.t.HasField("name")


def test_a_default_valued_attribute_of_the_wrong_type_is_kept() -> None:
    proto = parse(PPO)
    proto.graph.node[3].attribute.append(helper.make_attribute("alpha", 1))  # INT, not FLOAT
    names = [a.name for a in parse(canonicalize(proto)).graph.node[3].attribute]
    assert names == ["alpha", "transB"]


def test_pack_copies_io_without_type_shape_or_dim() -> None:
    proto = parse(PPO)
    proto.graph.input[0].ClearField("type")
    proto.graph.output[0].type.tensor_type.ClearField("shape")
    out = parse(canonicalize(proto, pack=True))
    assert not out.graph.input[0].HasField("type")
    assert not out.graph.output[0].type.tensor_type.HasField("shape")
    proto = parse(PPO)
    proto.graph.output[0].type.tensor_type.shape.dim[1].Clear()
    out = parse(canonicalize(proto, pack=True))
    assert out.graph.output[0].type.tensor_type.shape.dim[1].WhichOneof("value") is None


def test_pack_refuses_invalid_utf8_it_would_copy() -> None:
    proto = parse(PPO)
    proto.graph.node[4].op_type = "QQQQ"
    data = serialize(proto).replace(b"QQQQ", b"\xff\xfe\xfd\xfc")
    with pytest.raises(ValueError, match="invalid_utf8"):
        canonicalize(parse(data), pack=True)


def test_the_build_and_a_rebuild_through_pack_agree_for_other_shapes() -> None:
    for hidden in ((64, 64), (5, 3)):
        data = build(tiny_weights("relu", hidden=hidden))
        assert canonicalize(parse(data), pack=True) == data
