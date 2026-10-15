"""The torch-free ONNX graph builder (docs/design/phase-2.md §3.2, §6 ``test_policy_build``).

Deferred to later slices: "ε = 1e-6 caught by the export constant binding" needs ``topple
export`` (slice 2); here the build side of it is pinned instead (the ε changes the ``inv_std``
initializer bitwise while parity alone barely sees it).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import pytest

from policy_support import (
    F32,
    A,
    D,
    balance_spec,
    build,
    det,
    fixture_rows,
    parse,
    ppo_bytes,
    run,
    sac_bytes,
    serialize,
    tiny_norm,
    tiny_weights,
)
from topple.contract import ActionSpec
from topple.policy.build import (
    MlpWeights,
    ObsNorm,
    build_policy_onnx,
    policy_metadata,
    reference_actions,
)
from topple.policy.validate import (
    ALLOWED_FIELDS,
    METADATA_KEYS,
    POLICY_FORMAT,
    canonicalize,
    structure_match,
)

PPO_OPS = ("Sub", "Mul", "Clip", "Gemm", "Tanh", "Gemm", "Tanh", "Gemm", "Clip")
SAC_OPS = ("Sub", "Mul", "Clip", "Gemm", "Relu", "Gemm", "Relu", "Gemm", "Tanh", "Clip")
H1, H2 = 8, 8
SHAPES = [[D], [D], [], [], [H1, D], [H1], [H2, H1], [H2], [A, H2], [A], [], []]

# sha256 of the tiny fixtures' builds (protobuf 7.36 / onnx 1.23); a change here is a re-score.
GOLDEN_PPO_SHA256 = "54b4a9f10cf126f62fbd9b7c53ec57267d05dc075a75e662d004a7e60c9364d8"
GOLDEN_SAC_SHA256 = "f0a6ac377bf91486301de3a1f8f2d84b33b34a372f9007343c479fb26bb406a2"


def inits(data: bytes) -> list[onnx.TensorProto]:
    return list(parse(data).graph.initializer)


def f32_of(t: onnx.TensorProto) -> npt.NDArray[np.float32]:
    return np.frombuffer(t.raw_data, dtype="<f4").reshape(tuple(t.dims))


# --- graph shape -----------------------------------------------------------------------------


@pytest.mark.parametrize(("data", "ops"), [(ppo_bytes(), PPO_OPS), (sac_bytes(), SAC_OPS)])
def test_node_sequence_and_canonical_names(data: bytes, ops: tuple[str, ...]) -> None:
    graph = parse(data).graph
    assert tuple(n.op_type for n in graph.node) == ops
    assert [n.name for n in graph.node] == [f"n{i}" for i in range(len(ops))]
    outputs = [n.output[0] for n in graph.node]
    assert outputs == [f"v{i}" for i in range(len(ops) - 1)] + ["action"]
    assert [list(n.input) for n in graph.node[:3]] == [
        ["obs", "c0"],
        ["v0", "c1"],
        ["v1", "c2", "c3"],
    ]
    assert list(graph.node[3].input) == ["v2", "c4", "c5"]
    assert list(graph.node[-1].input)[1:] == ["c10", "c11"]


@pytest.mark.parametrize("data", [ppo_bytes(), sac_bytes()], ids=["ppo", "sac"])
def test_initializer_names_and_shapes(data: bytes) -> None:
    tensors = inits(data)
    assert [t.name for t in tensors] == [f"c{j}" for j in range(12)]
    assert [list(t.dims) for t in tensors] == SHAPES  # Clip bounds are rank 0
    assert all(t.data_type == onnx.TensorProto.FLOAT and t.HasField("raw_data") for t in tensors)
    assert not any(t.float_data for t in tensors)


@pytest.mark.parametrize("data", [ppo_bytes(), sac_bytes()], ids=["ppo", "sac"])
def test_header(data: bytes) -> None:
    proto = parse(data)
    assert proto.ir_version == 8
    assert [(o.domain, o.version) for o in proto.opset_import] == [("", 17)]
    assert proto.producer_name == "topple"
    assert not proto.HasField("producer_version")
    assert not proto.HasField("doc_string")
    assert proto.graph.name == "topple_policy"
    (inp,) = proto.graph.input
    (out,) = proto.graph.output
    assert (inp.name, out.name) == ("obs", "action")
    for info, width in ((inp, D), (out, A)):
        tt = info.type.tensor_type
        assert tt.elem_type == onnx.TensorProto.FLOAT
        assert [d.dim_param or d.dim_value for d in tt.shape.dim] == ["batch", width]


@pytest.mark.parametrize("rule", ["clip(mean)", "tanh(mean)"])
def test_metadata_props_are_the_seven_json_fields(rule: str) -> None:
    activation = "tanh" if rule == "clip(mean)" else "relu"
    proto = parse(build(tiny_weights(activation)))
    keys = [e.key for e in proto.metadata_props]
    assert keys == sorted(METADATA_KEYS)
    values = {e.key: json.loads(e.value) for e in proto.metadata_props}
    spec = balance_spec()
    assert values == {
        "topple.format": POLICY_FORMAT,
        "topple.challenge": spec.challenge,
        "topple.obs_schema": spec.obs_schema,
        "topple.obs_features": list(spec.obs_features),
        "topple.action_spec": spec.action.id,
        "topple.control_dt": spec.control_dt,
        "topple.action_rule": rule,
    }
    assert all(e.value.isascii() for e in proto.metadata_props)


def test_policy_metadata_is_json_dumps_of_each_field() -> None:
    spec = balance_spec()
    meta = policy_metadata(spec, "clip(mean)")
    assert meta["topple.obs_features"] == json.dumps(list(spec.obs_features))
    assert meta["topple.control_dt"] == json.dumps(spec.control_dt)
    assert meta["topple.format"] == json.dumps(POLICY_FORMAT)


def walk_fields(message: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    stack = [message]
    while stack:
        msg = stack.pop()
        for fd, value in msg.ListFields():
            found.append((msg.DESCRIPTOR.full_name, fd.name))
            if fd.type == fd.TYPE_MESSAGE:
                stack.extend(value if fd.is_repeated else [value])
    return found


@pytest.mark.parametrize("data", [ppo_bytes(), sac_bytes()], ids=["ppo", "sac"])
def test_every_set_field_is_within_the_check2_allow_list(data: bytes) -> None:
    for message, field in walk_fields(parse(data)):
        assert field in ALLOWED_FIELDS[message], (message, field)


@pytest.mark.parametrize("data", [ppo_bytes(), sac_bytes()], ids=["ppo", "sac"])
def test_no_paths_or_environment_leak_into_the_bytes(data: bytes) -> None:
    for needle in (b"/Users/", b"/home/", b"site-packages", b"\\Users\\"):
        assert needle not in data


@pytest.mark.parametrize("data", [ppo_bytes(), sac_bytes()], ids=["ppo", "sac"])
def test_build_output_is_canonical(data: bytes) -> None:
    assert canonicalize(parse(data)) == data
    assert canonicalize(parse(data), pack=True) == data


def test_rebuild_is_byte_identical() -> None:
    assert ppo_bytes() == ppo_bytes()
    assert sac_bytes() == sac_bytes()


def test_golden_sha256() -> None:
    assert hashlib.sha256(ppo_bytes()).hexdigest() == GOLDEN_PPO_SHA256
    assert hashlib.sha256(sac_bytes()).hexdigest() == GOLDEN_SAC_SHA256


def test_structure_matches_its_action_rule() -> None:
    assert structure_match(parse(ppo_bytes())) == "clip(mean)"
    assert structure_match(parse(sac_bytes())) == "tanh(mean)"


# --- constants -------------------------------------------------------------------------------


def test_constants_are_rounded_once_from_f64() -> None:
    norm = tiny_norm()
    tensors = inits(ppo_bytes())
    assert np.array_equal(f32_of(tensors[0]), norm.mean.astype(F32))
    inv_std = (1.0 / np.sqrt(norm.var + norm.epsilon)).astype(F32)
    assert f32_of(tensors[1]).tobytes() == inv_std.tobytes()
    assert f32_of(tensors[2]).tobytes() == F32(-10.0).tobytes()
    assert f32_of(tensors[3]).tobytes() == F32(10.0).tobytes()
    weights = tiny_weights()
    for k, (w, b) in enumerate(weights.layers):
        assert f32_of(tensors[4 + 2 * k]).tobytes() == w.tobytes()
        assert f32_of(tensors[5 + 2 * k]).tobytes() == b.tobytes()


def test_head_bounds_come_from_the_challenge_action_spec() -> None:
    """The penalty wrapper trains under Box(-2, 2), but the builder never sees the model's
    action space: the head Clip takes the challenge's ``TaskSpec.action`` bounds."""
    weights = tiny_weights(head_scale=50.0)  # pre-clip actions far outside ±2
    tensors = inits(build(weights))
    assert f32_of(tensors[10]).tobytes() == F32(-1.0).tobytes()
    assert f32_of(tensors[11]).tobytes() == F32(1.0).tobytes()
    actions = run(build(weights), fixture_rows())
    assert np.abs(actions).max() == 1.0


def test_tiny_std_warns_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    norm = tiny_norm()
    var = norm.var.copy()
    var[3] = 0.0  # sqrt(0 + 1e-8) = 1e-4 is not below 1e-6 * max(1, |mean|) ...
    mean = norm.mean.copy()
    mean[3] = 1e3  # ... until |mean| = 1e3 makes the bar 1e-3
    build(norm=ObsNorm(mean=mean, var=var, epsilon=1e-8, clip=10.0))
    err = capsys.readouterr().err
    assert "warning" in err
    assert "feature 3" in err
    build()
    assert capsys.readouterr().err == ""


def test_epsilon_changes_inv_std_bitwise_but_barely_the_actions() -> None:
    """ε = 1e-6 instead of 1e-8: the build side of the export constant binding (§3.3 step 3)."""
    good = build()
    wrong = build(norm=tiny_norm(epsilon=1e-6))
    assert f32_of(inits(good)[1]).tobytes() != f32_of(inits(wrong)[1]).tobytes()
    rows = fixture_rows()
    assert np.abs(run(good, rows) - run(wrong, rows)).max() < 1e-3


# --- reference_actions vs ORT ----------------------------------------------------------------


@pytest.mark.parametrize("var", [0.0, 1e-12, 1e-8, 1.0, 1e6])
@pytest.mark.parametrize("mu", [0.0, 1e3])
@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_reference_actions_match_ort(
    var: float, mu: float, activation: str, capsys: pytest.CaptureFixture[str]
) -> None:
    signs = np.where(np.arange(D) % 2 == 0, 1.0, -1.0)
    norm = ObsNorm(mean=signs * mu, var=np.full(D, var), epsilon=1e-8, clip=10.0)
    weights = tiny_weights(activation, hidden=(64, 64))
    rule = "clip(mean)" if activation == "tanh" else "tanh(mean)"
    rows = fixture_rows(norm, n=128)
    data = build(weights, norm)
    capsys.readouterr()
    ref = reference_actions(weights, norm, rule, rows, action=balance_spec().action)
    assert ref.dtype == np.float64
    assert ref.shape == (128, A)
    assert np.abs(run(data, rows) - ref).max() <= 1e-5


# --- negative controls: each bug is visible to the 1e-5 parity check -------------------------


def drop_normalization(data: bytes) -> bytes:
    """The graph with Sub/Mul/Clip removed: the first Gemm reads ``obs`` directly."""
    proto = parse(data)
    graph = proto.graph
    nodes = list(graph.node)[3:]
    nodes[0].input[0] = "obs"
    del graph.node[:]
    graph.node.extend(nodes)
    keep = [t for t in graph.initializer if t.name not in {"c0", "c1", "c2", "c3"}]
    del graph.initializer[:]
    graph.initializer.extend(keep)
    return canonicalize(parse(serialize(proto)))


def bug_cases() -> dict[str, tuple[bytes, bytes]]:
    """(correct, buggy) build pairs, one per §6 negative control."""
    ppo, sac = tiny_weights("tanh"), tiny_weights("relu")
    norm = tiny_norm()
    w1_t = MlpWeights(
        layers=(ppo.layers[0], (ppo.layers[1][0].T.copy(), ppo.layers[1][1]), ppo.layers[2]),
        activation="tanh",
    )
    return {
        "clip_obs_5": (build(ppo), build(ppo, tiny_norm(clip=5.0))),
        "normalization_omitted": (build(ppo), drop_normalization(build(ppo))),
        "mean_and_var_swapped": (
            build(ppo),
            build(ppo, ObsNorm(mean=norm.var, var=norm.mean, epsilon=1e-8, clip=10.0)),
        ),
        "w1_transposed": (build(ppo), build(w1_t)),
        "head_clip_2": (
            build(ppo),
            build(
                ppo,
                action=ActionSpec(id="diffdrive.effort.v1", names=("l", "r"), low=-2.0, high=2.0),
            ),
        ),
        "tanh_relu_swapped": (
            build(ppo),
            build(MlpWeights(layers=ppo.layers, activation="relu"), action_rule="clip(mean)"),
        ),
        "sac_head_tanh_omitted": (build(sac), build(sac, action_rule="clip(mean)")),
    }


@pytest.mark.parametrize("case", list(bug_cases()))
def test_negative_control_exceeds_parity_tolerance(case: str) -> None:
    good, bad = bug_cases()[case]
    rows = fixture_rows()
    assert np.abs(run(good, rows) - run(bad, rows)).max() > 1e-3


# --- input checks ----------------------------------------------------------------------------


def with_layer(k: int, w: npt.NDArray[np.float32] | None = None, b: Any = None) -> MlpWeights:
    base = tiny_weights()
    layers = list(base.layers)
    layers[k] = (layers[k][0] if w is None else w, layers[k][1] if b is None else b)
    return MlpWeights(layers=tuple(layers), activation="tanh")


def bad_metadata(**changes: str | None) -> dict[str, str]:
    meta = policy_metadata(balance_spec(), "clip(mean)")
    for key, value in changes.items():
        full = f"topple.{key}"
        if value is None:
            del meta[full]
        else:
            meta[full] = value
    return meta


BAD_INPUTS: dict[str, dict[str, Any]] = {
    "two_layers": {"weights": MlpWeights(layers=tiny_weights().layers[:2], activation="tanh")},
    "w0_width": {"weights": with_layer(0, det((8, D + 1), 1))},
    "b0_width": {"weights": with_layer(0, b=det((9,), 2))},
    "w1_width": {"weights": with_layer(1, det((8, 9), 3))},
    "head_dim_vs_action_spec": {"weights": with_layer(2, det((3, 8), 5), det((3,), 6))},
    "float64_weight": {"weights": with_layer(0, det((8, D), 1).astype(np.float64))},  # type: ignore[arg-type]
    "nan_weight": {"weights": with_layer(1, np.full((8, 8), np.nan, dtype=F32))},
    "bad_activation": {"weights": MlpWeights(layers=tiny_weights().layers, activation="gelu")},  # type: ignore[arg-type]
    "mean_width": {"norm": ObsNorm(np.zeros(D + 1), np.ones(D + 1), 1e-8, 10.0)},
    "negative_var": {"norm": ObsNorm(np.zeros(D), -np.ones(D), 1e-8, 10.0)},
    "nan_mean": {"norm": ObsNorm(np.full(D, np.nan), np.ones(D), 1e-8, 10.0)},
    "zero_epsilon": {"norm": ObsNorm(np.zeros(D), np.ones(D), 0.0, 10.0)},
    "zero_clip": {"norm": ObsNorm(np.zeros(D), np.ones(D), 1e-8, 0.0)},
    "infinite_clip": {"norm": ObsNorm(np.zeros(D), np.ones(D), 1e-8, float("inf"))},
    "bad_rule": {"action_rule": "other"},
    "missing_key": {"metadata": bad_metadata(challenge=None)},
    "extra_key": {"metadata": {**bad_metadata(), "topple.note": '"x"'}},
    "rule_mismatch": {"metadata": bad_metadata(action_rule='"tanh(mean)"')},
    "format_mismatch": {"metadata": bad_metadata(format='"topple.policy/2"')},
    "action_spec_mismatch": {"metadata": bad_metadata(action_spec='"other.effort.v1"')},
    "feature_count_mismatch": {"metadata": bad_metadata(obs_features='["pitch"]')},
    "non_json_value": {"metadata": bad_metadata(challenge="balance-v1")},
    "non_ascii_value": {"metadata": bad_metadata(challenge='"bälance"')},
    "long_value": {"metadata": bad_metadata(challenge=json.dumps("x" * 1100))},
}


@pytest.mark.parametrize("case", list(BAD_INPUTS))
def test_bad_inputs_raise_value_error(case: str) -> None:
    args = BAD_INPUTS[case]
    weights = args.get("weights", tiny_weights())
    norm = args.get("norm", tiny_norm())
    rule = args.get("action_rule", "clip(mean)")
    with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
        build_policy_onnx(
            weights,
            norm,
            action_rule=rule,
            metadata=args.get("metadata", policy_metadata(balance_spec(), "clip(mean)")),
            action=balance_spec().action,
        )


@pytest.mark.parametrize(
    "obs",
    [
        np.zeros((3, D), dtype=np.float64),
        np.zeros((D,), dtype=F32),
        np.zeros((3, D + 1), dtype=F32),
    ],
    ids=["float64", "rank1", "width"],
)
def test_reference_actions_rejects_bad_rows(obs: npt.NDArray[Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        reference_actions(
            tiny_weights(), tiny_norm(), "clip(mean)", obs, action=balance_spec().action
        )
