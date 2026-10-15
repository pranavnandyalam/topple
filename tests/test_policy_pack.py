"""``topple policy pack``: third-party ONNX models to ``topple.policy/1`` directories
(docs/design/phase-2.md §3.3, decision log B8, §6 ``test_policy_pack``).

Pack reads the model (check 1), canonicalizes it in pack mode (drops doc strings,
``value_info``, foreign metadata and producer fields, removes ``Identity`` nodes, folds
``Constant`` nodes, renames the I/O to ``obs``/``action``, turns a static batch of 1 into
``batch``, sets ``CANONICAL_IR[opset]``; external data is refused before any tensor is read),
replaces the 7 ``topple.*`` metadata keys with the challenge's, runs checks 1-9 and the smoke
test, and writes ``policy.onnx`` + ``policy.json`` (``action_rule`` from the structure match,
every claim null) into a directory that did not exist, atomically.

The models here are built with ``onnx.helper`` the way the two torch exporters write them
(the legacy exporter's interleaved rank-0 ``Constant`` nodes; the dynamo exporter's
``value_info`` and rank-0 initializers). The committed torch-generated fixtures
``tests/data/thirdparty_{legacy,dynamo}.onnx`` and ``scripts/make_thirdparty_fixtures.py``
need torch and are DEFERRED until it installs (see
``test_committed_torch_fixtures_pack_and_load``).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx.checker
import onnx.external_data_helper
import onnx.numpy_helper
import onnxruntime
import pytest
from onnx import TensorProto, helper
from typer.testing import CliRunner

from policy_support import (
    A,
    D,
    balance_spec,
    parse,
    ppo_bytes,
    sac_bytes,
    serialize,
    tiny_norm,
    tiny_weights,
)
from topple.challenge import load_challenge
from topple.cli import app
from topple.loader import LoadError, load_submission
from topple.policy.build import policy_metadata
from topple.policy.manifest import read_manifest
from topple.policy.pack import pack_policy
from topple.policy.runtime import load_policy_dir
from topple.policy.validate import (
    CANONICAL_IR,
    MAX_ONNX_BYTES,
    METADATA_KEYS,
    identity_resolver,
    structure_match,
)

BALANCE = load_challenge("balance-v1")
DATA = Path(__file__).parent / "data"
runner = CliRunner()


def f32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="<f4")


def raw(name: str, values: Any) -> TensorProto:
    arr = f32(values)
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.tobytes(), raw=True)


def listed(name: str, values: Any) -> TensorProto:
    arr = f32(values)
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.reshape(-1).tolist())


def norm_constants() -> tuple[np.ndarray, np.ndarray, float]:
    norm = tiny_norm()
    mean = f32(norm.mean)
    inv_std = f32(1.0 / np.sqrt(norm.var + norm.epsilon))
    return mean, inv_std, float(np.float32(norm.clip))


def legacy_style(opset: int = 20) -> onnx.ModelProto:
    """The tiny PPO policy as ``torch.onnx.export(dynamo=False)`` writes an MLP: IR 9, a
    ``Constant`` node before each Clip, explicit default Gemm attributes, initializers named
    after the module, I/O ``x``/``y`` with a static batch of 1, empty node names, a ``;`` in a
    value name, doc strings, foreign metadata and an ``Identity`` on the output."""
    mean, inv_std, clip = norm_constants()
    (w0, b0), (w1, b1), (w2, b2) = tiny_weights("tanh").layers
    gemm = {"alpha": 1.0, "beta": 1.0, "transB": 1}
    nodes = [
        helper.make_node("Sub", ["x", "obs_rms.mean"], ["a;b"], name=""),
        helper.make_node("Mul", ["a;b", "obs_rms.inv_std"], ["/norm/Mul_output_0"], name=""),
        helper.make_node("Constant", [], ["/Constant_output_0"], value=raw("v0", -clip)),
        helper.make_node("Constant", [], ["/Constant_1_output_0"], value=raw("v1", clip)),
        helper.make_node(
            "Clip", ["/norm/Mul_output_0", "/Constant_output_0", "/Constant_1_output_0"], ["z"]
        ),
        helper.make_node("Gemm", ["z", "net.0.weight", "net.0.bias"], ["h0"], name="", **gemm),
        helper.make_node("Tanh", ["h0"], ["t0"], name=""),
        helper.make_node("Gemm", ["t0", "net.2.weight", "net.2.bias"], ["h1"], name="", **gemm),
        helper.make_node("Tanh", ["h1"], ["t1"], name=""),
        helper.make_node("Gemm", ["t1", "net.4.weight", "net.4.bias"], ["mu"], name="", **gemm),
        helper.make_node("Constant", [], ["/Constant_2_output_0"], value=raw("v2", -1.0)),
        helper.make_node("Constant", [], ["/Constant_3_output_0"], value_float=1.0),
        helper.make_node("Clip", ["mu", "/Constant_2_output_0", "/Constant_3_output_0"], ["act"]),
        helper.make_node("Identity", ["act"], ["y"], name=""),
    ]
    nodes[0].doc_string = "/home/someone/project/policy.py:12"
    initializers = [
        listed("net.4.bias", b2),
        raw("net.0.weight", w0),
        listed("obs_rms.inv_std", inv_std),
        raw("net.2.bias", b1),
        raw("net.0.bias", b0),
        raw("net.4.weight", w2),
        listed("net.2.weight", w1),
        raw("obs_rms.mean", mean),
    ]
    graph = helper.make_graph(
        nodes,
        "main_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, D])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, A])],
        initializers,
        doc_string="exported by torch",
    )
    model: onnx.ModelProto = helper.make_model(
        graph,
        ir_version=9,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="pytorch",
        producer_version="2.14.1",
        doc_string="a legacy export",
    )
    helper.set_model_props(model, {"exported_by": "someone", "torch.version": "2.14.1"})
    return model


def dynamo_style(scale: str = "rank0") -> onnx.ModelProto:
    """The same network as ``torch.onnx.export(dynamo=True)`` writes it: IR 10, opset 20,
    ``Sub Mul Clip Gemm Tanh Gemm Tanh Gemm Clip``, rank-0 initializers for every Clip bound,
    ``value_info`` for every value, and the normalization scale as a rank-0 initializer
    (``scale="rank0"``, one scale for every feature) or broadcast to ``[D]``
    (``scale="vector"``)."""
    mean, _, clip = norm_constants()
    (w0, b0), (w1, b1), (w2, b2) = tiny_weights("tanh").layers
    s = np.float32(0.5)
    scale_init = raw("scale", s) if scale == "rank0" else raw("scale", np.full(D, s))
    chain = [
        ("Sub", ["obs", "mean"], "sub"),
        ("Mul", ["sub", "scale"], "mul"),
        ("Clip", ["mul", "clip_lo", "clip_hi"], "clamp"),
        ("Gemm", ["clamp", "w0", "b0"], "linear"),
        ("Tanh", ["linear"], "tanh"),
        ("Gemm", ["tanh", "w1", "b1"], "linear_1"),
        ("Tanh", ["linear_1"], "tanh_1"),
        ("Gemm", ["tanh_1", "w2", "b2"], "linear_2"),
        ("Clip", ["linear_2", "act_lo", "act_hi"], "action"),
    ]
    nodes = [
        helper.make_node(
            op, ins, [out], name=f"node_{op}_{i}", **({"transB": 1} if op == "Gemm" else {})
        )
        for i, (op, ins, out) in enumerate(chain)
    ]
    initializers = [
        raw("mean", mean),
        scale_init,
        raw("clip_lo", -clip),
        raw("clip_hi", clip),
        raw("w0", w0),
        raw("b0", b0),
        raw("w1", w1),
        raw("b1", b1),
        raw("w2", w2),
        raw("b2", b2),
        raw("act_lo", -1.0),
        raw("act_hi", 1.0),
    ]
    widths = {"sub": D, "mul": D, "clamp": D, "linear": 8, "tanh": 8, "linear_1": 8}
    widths |= {"tanh_1": 8, "linear_2": A}
    value_info = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", width])
        for name, width in widths.items()
    ] + [
        helper.make_tensor_value_info(t.name, TensorProto.FLOAT, list(t.dims)) for t in initializers
    ]
    assert len(value_info) == 20
    graph = helper.make_graph(
        nodes,
        "main_graph",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", D])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", A])],
        initializers,
        value_info=value_info,
    )
    model: onnx.ModelProto = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 20)],
        producer_name="pytorch",
        producer_version="2.14.1",
    )
    return model


def pack(proto: onnx.ModelProto | bytes) -> dict[str, bytes]:
    data = proto if isinstance(proto, bytes) else serialize(proto)
    return pack_policy(data, challenge=BALANCE)


def manifest_of(files: dict[str, bytes]) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(files["policy.json"])
    return doc


def write_model(path: Path, proto: onnx.ModelProto | bytes) -> Path:
    path.write_bytes(proto if isinstance(proto, bytes) else serialize(proto))
    return path


# --- pack_policy ------------------------------------------------------------------------------


def test_a_legacy_export_packs_to_ir_9_opset_20_with_the_ppo_structure() -> None:
    files = pack(legacy_style())
    proto = parse(files["policy.onnx"])
    assert (proto.ir_version, [o.version for o in proto.opset_import]) == (9, [20])
    assert CANONICAL_IR[20] == 9
    assert [n.op_type for n in proto.graph.node] == [
        "Sub", "Mul", "Clip", "Gemm", "Tanh", "Gemm", "Tanh", "Gemm", "Clip"
    ]  # fmt: skip
    assert structure_match(proto) == "clip(mean)"
    assert manifest_of(files)["action_rule"] == "clip(mean)"
    assert [i.name for i in proto.graph.input] == ["obs"]
    assert [o.name for o in proto.graph.output] == ["action"]
    dims = proto.graph.input[0].type.tensor_type.shape.dim
    assert (dims[0].dim_param, dims[1].dim_value) == ("batch", D)
    assert not proto.HasField("doc_string")
    assert not proto.graph.HasField("doc_string")
    assert all(not n.HasField("doc_string") for n in proto.graph.node)
    assert proto.producer_name == "topple"
    assert not proto.HasField("producer_version")


def test_the_legacy_export_at_opset_17_packs_to_exactly_the_first_party_build() -> None:
    assert pack(legacy_style(opset=17))["policy.onnx"] == ppo_bytes()


def test_a_dynamo_export_with_a_rank_0_scale_is_other() -> None:
    files = pack(dynamo_style("rank0"))
    proto = parse(files["policy.onnx"])
    assert (proto.ir_version, proto.opset_import[0].version) == (9, 20)
    assert not proto.graph.value_info
    assert structure_match(proto) == "other"
    assert manifest_of(files)["action_rule"] == "other"
    assert [len(t.dims) for t in proto.graph.initializer][:4] == [1, 0, 0, 0]


def test_the_dynamo_export_with_the_scale_broadcast_to_d_is_clip_mean() -> None:
    files = pack(dynamo_style("vector"))
    assert structure_match(parse(files["policy.onnx"])) == "clip(mean)"
    assert manifest_of(files)["action_rule"] == "clip(mean)"


@pytest.mark.parametrize("data", [ppo_bytes, sac_bytes], ids=["ppo", "sac"])
def test_packing_the_first_party_build_is_the_identity(data: Callable[[], bytes]) -> None:
    files = pack(data())
    assert files["policy.onnx"] == data()
    rule = "clip(mean)" if data is ppo_bytes else "tanh(mean)"
    assert manifest_of(files)["action_rule"] == rule


def test_pack_is_deterministic_and_a_fixed_point() -> None:
    once = pack(legacy_style())
    assert pack(legacy_style()) == once
    assert pack(once["policy.onnx"]) == once


def test_foreign_metadata_is_dropped_and_the_seven_keys_replaced() -> None:
    proto = legacy_style()
    wrong = policy_metadata(balance_spec(), "tanh(mean)") | {
        "topple.challenge": json.dumps("other-v1"),
        "topple.note": "kept?",
        "author": "someone",
    }
    del proto.metadata_props[:]
    helper.set_model_props(proto, wrong)
    meta = {e.key: e.value for e in parse(pack(proto)["policy.onnx"]).metadata_props}
    assert sorted(meta) == sorted(METADATA_KEYS)
    assert meta == policy_metadata(balance_spec(), "clip(mean)")


def length_delimited(tag: int, payload: bytes) -> bytes:
    """One protobuf length-delimited field (``payload`` under 128 bytes)."""
    assert len(payload) < 128
    return bytes([(tag << 3) | 2, len(payload)]) + payload


INVALID_UTF8 = b"\xff\xfe"
# ModelProto field 14 is metadata_props (StringStringEntryProto: key = 1, value = 2); appended
# fields merge into the message.
FOREIGN_INVALID_UTF8 = {
    "value": length_delimited(
        14, length_delimited(1, b"author") + length_delimited(2, INVALID_UTF8)
    ),
    "key": length_delimited(
        14, length_delimited(1, b"a" + INVALID_UTF8) + length_delimited(2, b"x")
    ),
    "both": length_delimited(
        14, length_delimited(1, INVALID_UTF8) + length_delimited(2, INVALID_UTF8)
    ),
}


@pytest.mark.parametrize(
    "extra", list(FOREIGN_INVALID_UTF8.values()), ids=list(FOREIGN_INVALID_UTF8)
)
def test_foreign_metadata_that_is_not_utf8_is_dropped_too(extra: bytes) -> None:
    """Pack drops all foreign metadata (§3.3), so a key or value that is not valid UTF-8 is
    dropped unread, like an invalid doc string, not refused as ``invalid_utf8`` (quality-F5)."""
    assert pack(ppo_bytes() + extra)["policy.onnx"] == ppo_bytes()


class CountingAliases(dict[str, str]):
    """An alias map that counts its lookups."""

    lookups = 0

    def __contains__(self, key: object) -> bool:
        self.lookups += 1
        return super().__contains__(key)

    def __getitem__(self, key: str) -> str:
        self.lookups += 1
        return super().__getitem__(key)


def test_an_identity_chain_is_walked_once_however_often_it_is_read() -> None:
    """A chain of n Identity nodes read m times costs O(n + m) lookups, not O(n·m): a 1 MiB
    model (17,000 Identity nodes, 27,000 readers) took 35 s of CPU in pack before any node cap
    could refuse it (quality-F4)."""
    n = m = 2_000
    aliases = CountingAliases({f"i{k}": f"i{k - 1}" if k else "obs" for k in range(n)})
    resolve = identity_resolver(aliases)
    assert all(resolve(f"i{n - 1}") == "obs" for _ in range(m))
    assert all(resolve(f"i{k}") == "obs" for k in range(n))
    assert resolve("obs") == "obs"
    assert aliases.lookups < 4 * (n + m)
    with pytest.raises(ValueError, match="identity_cycle"):
        identity_resolver({"a": "b", "b": "c", "c": "a"})("a")
    with pytest.raises(ValueError, match="identity_cycle"):
        identity_resolver({"t": "a", "a": "b", "b": "a"})("t")


def test_a_long_identity_chain_is_canonicalized_then_refused() -> None:
    """End to end: 4,000 Identity nodes read by 4,000 nodes are resolved in one pass and the
    canonical model is then refused by check 7."""
    nodes = [helper.make_node("Identity", ["x"], ["i0"])]
    nodes += [helper.make_node("Identity", [f"i{k - 1}"], [f"i{k}"]) for k in range(1, 4_000)]
    nodes += [helper.make_node("Relu", ["i3999"], [f"r{k}"]) for k in range(4_000)]
    nodes.append(helper.make_node("Relu", ["i3999"], ["y"]))
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, D])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, A])],
    )
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])
    with pytest.raises(LoadError, match="too_many_node"):
        pack(model)


def test_the_manifest_states_the_challenge_and_null_claims() -> None:
    files = pack(legacy_style())
    m = read_manifest(files["policy.json"])
    spec = balance_spec()
    assert (m.challenge, m.obs_schema, m.action_spec) == (
        spec.challenge,
        spec.obs_schema,
        spec.action.id,
    )
    assert m.obs_features == spec.obs_features
    assert (m.action_dim, m.control_dt) == (spec.action.dim, spec.control_dt)
    assert m.onnx.sha256 == hashlib.sha256(files["policy.onnx"]).hexdigest()
    assert (m.onnx.ir_version, m.onnx.opset) == (9, 20)
    assert m.onnx.input.shape == ("batch", D)
    assert m.onnx.output.shape == ("batch", A)
    assert m.claims.export_parity is None
    assert all(value is None for value in m.claims.model_dump().values())


def test_pack_refuses_external_data_before_any_tensor_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_fifo: Callable[[Path], Path]
) -> None:
    """The referenced file is a FIFO (reading it would block) and every reader of tensors,
    external data or sessions raises if called: pack stops at the reference itself."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not be called")

    proto = parse(ppo_bytes())
    tensor = proto.graph.initializer[0]
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (("location", "weights.bin"), ("offset", "0"), ("length", "40")):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    monkeypatch.chdir(tmp_path)
    make_fifo(tmp_path / "weights.bin")
    monkeypatch.setattr(onnx.numpy_helper, "to_array", forbidden)
    monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_model", forbidden)
    monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_tensor", forbidden)
    monkeypatch.setattr(onnx.checker, "check_model", forbidden)
    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    with pytest.raises(LoadError, match="external_data"):
        pack(proto)
    model = write_model(tmp_path / "model.onnx", proto)
    result = runner.invoke(
        app, ["policy", "pack", str(model), "--challenge", "balance-v1", "--out", "bot"]
    )
    assert result.exit_code == 2
    assert "external_data" in result.output
    assert not (tmp_path / "bot").exists()


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        (lambda p: p.graph.node.add(op_type="Sin", input=["y"], output=["s"]), "op_not_allowed"),
        (lambda p: p.graph.input.append(helper.make_tensor_value_info("u", 1, [1, 1])), "io_count"),
        (lambda p: p.opset_import[0].__setattr__("version", 16), "unsupported_opset"),
        (lambda p: p.graph.node[1].__setattr__("domain", "com.example"), "custom_domain"),
        (
            lambda p: p.graph.input[0].type.tensor_type.shape.dim[1].__setattr__("dim_value", 11),
            "bad_input",
        ),
    ],
    ids=["op", "two-inputs", "opset-16", "custom-domain", "width-11"],
)
def test_pack_refuses_with_a_fixed_code(edit: Callable[[onnx.ModelProto], Any], code: str) -> None:
    proto = legacy_style()
    edit(proto)
    with pytest.raises(LoadError, match=code):
        pack(proto)


HOSTILE = "<b>\x1b[31m\u202erm"


def hostile_op_type(proto: onnx.ModelProto) -> None:
    proto.graph.node[6].op_type = HOSTILE


def hostile_attribute_name(proto: onnx.ModelProto) -> None:
    proto.graph.node[6].attribute.append(helper.make_attribute(HOSTILE, 1))


def hostile_names(proto: onnx.ModelProto) -> None:
    proto.graph.node.add(op_type="Sin", input=["y"], output=[HOSTILE], name=HOSTILE)


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        (hostile_op_type, "bad_op_type"),
        (hostile_attribute_name, "bad_attribute_name"),
        (hostile_names, "op_not_allowed"),
    ],
    ids=["op-type", "attribute-name", "node-and-value-names"],
)
def test_pack_refusals_never_echo_the_model(
    edit: Callable[[onnx.ModelProto], None], code: str
) -> None:
    """Op types and attribute names survive canonicalization (node and value names do not:
    they become n{i}/v{i}), so hostile text there reaches the refusal; it shows as ``?``
    (quality-F8)."""
    proto = legacy_style()
    edit(proto)
    with pytest.raises(LoadError, match=code) as caught:
        pack(proto)
    message = str(caught.value)
    assert not {"<", "\x1b", "\u202e"} & set(message), message


def test_pack_never_echoes_an_unknown_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only ``canonicalize``'s own fixed codes pass through; a ``ValueError`` with any other
    text (it could quote the model) is ``pack_failed`` without it (quality-F8, M11)."""
    import topple.policy.pack as pack_module

    def weird(*args: object, **kwargs: object) -> bytes:
        raise ValueError("weird: <raw text from the model>")

    monkeypatch.setattr(pack_module, "canonicalize", weird)
    with pytest.raises(LoadError, match="pack_failed: raised while canonicalizing") as caught:
        pack(legacy_style())
    assert "<raw" not in str(caught.value)
    assert "weird" not in str(caught.value)


@pytest.mark.parametrize(
    ("data", "code"),
    [(b"\x08\x80", "onnx_parse_failed"), (b"\0" * (MAX_ONNX_BYTES + 1), "onnx_too_large")],
    ids=["garbage", "too-large"],
)
def test_pack_check_1(data: bytes, code: str) -> None:
    with pytest.raises(LoadError, match=code):
        pack(data)


def test_pack_wraps_unexpected_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    import topple.policy.pack as pack_module

    def boom(*args: object, **kwargs: object) -> bytes:
        raise KeyError("<raw text from somewhere>")

    monkeypatch.setattr(pack_module, "canonicalize", boom)
    with pytest.raises(LoadError, match="pack_failed: raised while canonicalizing") as caught:
        pack(legacy_style())
    assert "<raw" not in str(caught.value)


# --- the command ------------------------------------------------------------------------------


def invoke(model: Path, out: Path | str, challenge: str = "balance-v1") -> Any:
    return runner.invoke(
        app, ["policy", "pack", str(model), "--challenge", challenge, "--out", str(out)]
    )


@pytest.mark.parametrize(
    ("make", "rule"),
    [(legacy_style, "clip(mean)"), (dynamo_style, "other")],
    ids=["legacy", "dynamo"],
)
def test_cli_pack_writes_a_loadable_policy_directory(
    tmp_path: Path, make: Callable[[], onnx.ModelProto], rule: str
) -> None:
    model = write_model(tmp_path / "model.onnx", make())
    out = tmp_path / "packed" / "bot"
    result = invoke(model, out)
    assert result.exit_code == 0, result.output
    assert rule in result.output
    assert sorted(p.name for p in out.iterdir()) == ["policy.json", "policy.onnx"]
    assert oct(out.stat().st_mode & 0o777) == "0o755"
    assert all(oct(p.stat().st_mode & 0o777) == "0o644" for p in out.iterdir())
    loaded = load_policy_dir(out, challenge=BALANCE)
    assert loaded.manifest is not None
    assert loaded.manifest.action_rule == rule
    sub = load_submission(out, challenge=BALANCE)
    assert sub.policy_sha256 == hashlib.sha256((out / "policy.onnx").read_bytes()).hexdigest()
    assert sub.policy_sha256 is not None
    assert sub.policy_sha256[:12] in result.output
    assert [p.name for p in out.parent.iterdir()] == ["bot"]  # no temporary left


def test_cli_pack_never_writes_into_an_existing_directory(tmp_path: Path) -> None:
    model = write_model(tmp_path / "model.onnx", legacy_style())
    out = tmp_path / "bot"
    out.mkdir()
    result = invoke(model, out)
    assert result.exit_code == 2
    assert "already exists" in result.output
    assert list(out.iterdir()) == []
    (tmp_path / "file").write_text("x")
    assert invoke(model, tmp_path / "file").exit_code == 2
    assert (tmp_path / "file").read_text() == "x"


def test_cli_pack_refuses_a_symlinked_relative_output_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", tmp_path / "out")
    result = invoke(tmp_path / "absent.onnx", "out/bot")
    assert result.exit_code == 2
    assert "refusing to write through the symlink 'out'" in result.output
    assert list((tmp_path / "elsewhere").iterdir()) == []


def test_cli_pack_failure_leaves_nothing(tmp_path: Path) -> None:
    proto = legacy_style()
    proto.graph.node.add(op_type="Sin", input=["y"], output=["s"])
    model = write_model(tmp_path / "model.onnx", proto)
    result = invoke(model, tmp_path / "out" / "bot")
    assert result.exit_code == 2
    assert "op_not_allowed" in result.output
    assert not (tmp_path / "out" / "bot").exists()
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []


@pytest.mark.parametrize("out", ["bot", "new/deeper/bot"])
def test_cli_pack_a_failed_write_leaves_no_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, out: str
) -> None:
    """Nothing is left behind, not even the parent directories pack created (quality-F10)."""
    import topple._atomic as atomic

    calls: list[Path] = []

    def failing(path: Path, data: bytes) -> None:
        calls.append(path)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        real(path, data)

    real = atomic.write_new_file
    monkeypatch.setattr(atomic, "write_new_file", failing)
    model = write_model(tmp_path / "model.onnx", legacy_style())
    result = invoke(model, tmp_path / out)
    assert result.exit_code == 2
    assert "No space left on device" in result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.onnx"]


@pytest.mark.parametrize(
    "setup",
    ["absent", "directory", "fifo", "too-large", "garbage"],
)
def test_cli_pack_input_problems_exit_2(
    tmp_path: Path, setup: str, make_fifo: Callable[[Path], Path]
) -> None:
    model = tmp_path / "model.onnx"
    if setup == "directory":
        model.mkdir()
    elif setup == "fifo":
        make_fifo(model)
    elif setup == "too-large":
        with model.open("wb") as fh:
            fh.truncate(MAX_ONNX_BYTES + 1)
    elif setup == "garbage":
        model.write_bytes(b"\x08\x80")
    result = invoke(model, tmp_path / "bot")
    assert result.exit_code == 2
    assert result.output.startswith("error: ")
    assert not (tmp_path / "bot").exists()


def test_cli_pack_unknown_challenge_exits_2(tmp_path: Path) -> None:
    model = write_model(tmp_path / "model.onnx", legacy_style())
    for challenge in ("balance-v9", "Balance"):
        result = invoke(model, tmp_path / "bot", challenge=challenge)
        assert result.exit_code == 2
        assert not (tmp_path / "bot").exists()


# --- deferred: the committed torch-generated fixtures (§3.3) ----------------------------------

TORCH_FIXTURES = ("thirdparty_legacy.onnx", "thirdparty_dynamo.onnx")


@pytest.mark.parametrize("name", TORCH_FIXTURES)
def test_committed_torch_fixtures_pack_and_load(name: str, tmp_path: Path) -> None:
    """§3.3: ``tests/data/thirdparty_legacy.onnx`` (IR 9 / opset 20, interleaved Constants)
    and ``tests/data/thirdparty_dynamo.onnx`` (IR 10 / opset 20, value_info, rank-0
    initializers) pack to IR 9 / opset 20 and load. DEFERRED: they are generated once by
    ``scripts/make_thirdparty_fixtures.py`` (``uv run --extra train --with onnxscript``), which
    needs torch; this test runs as soon as they are committed."""
    path = DATA / name
    if not path.exists():
        pytest.skip(
            f"deferred until torch installs: tests/data/{name} comes from "
            "scripts/make_thirdparty_fixtures.py (decision log B12 blocks torch here)"
        )
    files = pack(path.read_bytes())
    proto = parse(files["policy.onnx"])
    assert (proto.ir_version, proto.opset_import[0].version) == (9, 20)
    result = invoke(path, tmp_path / "bot")
    assert result.exit_code == 0, result.output
    load_policy_dir(tmp_path / "bot", challenge=BALANCE)
