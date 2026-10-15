"""``policy.json``, the manifest of a ``topple.policy/1`` directory (docs/design/phase-2.md
§3.1, §6 ``test_policy_manifest``).

The manifest is untrusted JSON. Every way it can be malformed (size, BOM, duplicate keys,
NaN/Infinity, numbers beyond float range, lone surrogates, huge integers, wrong scalar types,
extra keys) is one ``LoadError`` with a fixed code whose text never echoes the input, and the
claims are held to a printable charset.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

import topple.policy.manifest as manifest_module
from policy_support import (
    first_party_claims,
    first_party_manifest,
    manifest_bytes,
    parse,
    ppo_bytes,
    sac_bytes,
    write_policy_dir,
)
from topple._inputs import read_regular_file
from topple.challenge import load_challenge
from topple.loader import LoadError
from topple.policy.manifest import (
    HEX64,
    MAX_MANIFEST_BYTES,
    PRINTABLE,
    PolicyManifest,
    check_against_challenge,
    check_against_model,
    dump_manifest,
    policy_task,
    read_manifest,
)

PPO = ppo_bytes()
BALANCE = load_challenge("balance-v1")
# a load error from the manifest: "policy.json: <code>: <fixed text and safe names>"
FIXED = re.compile(r"policy\.json: [a-z0-9_]+: [\x20-\x7e]*")


def good() -> dict[str, Any]:
    return first_party_manifest(PPO)


def rejected(data: bytes) -> str:
    """The text of the ``LoadError`` ``read_manifest(data)`` raises (asserting it is fixed)."""
    with pytest.raises(LoadError) as info:
        read_manifest(data)
    text = str(info.value)
    assert FIXED.fullmatch(text), text
    assert info.value.__cause__ is None  # no raw exception text rides along
    return text


def with_change(path: tuple[str | int, ...], value: object) -> bytes:
    m = good()
    target: Any = m
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return manifest_bytes(m)


def without(path: tuple[str, ...]) -> bytes:
    m = good()
    target: Any = m
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return manifest_bytes(m)


# --- round trip -------------------------------------------------------------------------------


def test_round_trip_from_bytes_on_disk(tmp_path: Path) -> None:
    root = write_policy_dir(tmp_path / "bot", PPO)
    data = read_regular_file(root / "policy.json", MAX_MANIFEST_BYTES)
    m = read_manifest(data)
    assert isinstance(m, PolicyManifest)
    assert m.model_dump(mode="json") == json.loads(data)
    assert dump_manifest(m) == data  # the writer's form is the reader's
    assert read_manifest(dump_manifest(m)) == m
    assert m.onnx.input.shape == ("batch", 10)
    assert m.claims.net_arch == (8, 8)  # JSON arrays become tuples


def test_the_model_is_frozen() -> None:
    m = read_manifest(manifest_bytes(good()))
    with pytest.raises(ValueError, match="frozen"):
        m.challenge = "other-v1"  # type: ignore[misc]


def test_claims_may_all_be_null() -> None:
    claims = dict.fromkeys(first_party_claims())
    m = read_manifest(manifest_bytes(first_party_manifest(PPO, claims=claims)))
    assert m.claims.export_parity is None
    assert m.claims.name is None


def test_patterns_are_full_matches() -> None:
    assert HEX64.fullmatch("a" * 64)
    assert not HEX64.fullmatch("a" * 64 + "\n")
    assert not HEX64.fullmatch("A" * 64)
    assert PRINTABLE.fullmatch("")
    assert not PRINTABLE.fullmatch("x" * 65)
    assert not PRINTABLE.fullmatch("café")


# --- parsing rules ----------------------------------------------------------------------------


def test_size_cap_is_16_kib() -> None:
    data = manifest_bytes(good())
    padded = data + b" " * (MAX_MANIFEST_BYTES - len(data))
    assert len(padded) == MAX_MANIFEST_BYTES
    read_manifest(padded)  # exactly at the cap: fine
    assert "manifest_too_large" in rejected(padded + b" ")


def test_oversize_file_is_refused_before_it_is_read(tmp_path: Path) -> None:
    from topple.policy.runtime import load_policy_dir

    root = write_policy_dir(tmp_path / "bot", PPO)
    with (root / "policy.json").open("ab") as fh:
        fh.write(b" " * MAX_MANIFEST_BYTES)
    with pytest.raises(LoadError, match=r"'policy\.json' is larger than 16384 bytes"):
        load_policy_dir(root, challenge=BALANCE)


def test_bom_is_rejected() -> None:
    assert "manifest_bom" in rejected(b"\xef\xbb\xbf" + manifest_bytes(good()))


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_are_rejected(token: str) -> None:
    data = manifest_bytes(good()).replace(b'"control_dt": 0.01', f'"control_dt": {token}'.encode())
    assert b"0.01" not in data
    assert "manifest_nonfinite_number" in rejected(data)


@pytest.mark.parametrize("literal", ["1e400", "-1e400"])
def test_a_float_beyond_range_is_rejected(literal: str) -> None:
    data = manifest_bytes(good()).replace(
        b'"control_dt": 0.01', f'"control_dt": {literal}'.encode()
    )
    assert "manifest_nonfinite_number" in rejected(data)


@pytest.mark.parametrize("where", ["top", "nested"])
def test_duplicate_keys_are_rejected(where: str) -> None:
    data = manifest_bytes(good())
    if where == "top":
        data = data.replace(
            b'"kind": "onnx",', b'"kind": "onnx", "secret_dup": 1, "secret_dup": 2,'
        )
    else:
        data = data.replace(b'"file": "policy.onnx",', b'"file": "policy.onnx", "file": "x.onnx",')
    text = rejected(data)
    assert "manifest_duplicate_key" in text
    assert "secret_dup" not in text


@pytest.mark.parametrize(
    "replacement",
    [
        b'"name": "\\ud800"',  # a lone surrogate in a value
        b'"name": "x", "\\udfff": 1',  # in a key
        b'"name": "x", "versions2": ["\\ud83d"]',  # inside an array
    ],
    ids=["value", "key", "array"],
)
def test_lone_surrogates_are_rejected(replacement: bytes) -> None:
    data = manifest_bytes(good()).replace(b'"name": "tiny ppo"', replacement)
    assert "manifest_invalid_string" in rejected(data)


def test_invalid_utf8_is_rejected_without_echoing_the_bytes() -> None:
    data = manifest_bytes(good()).replace(b"tiny ppo", b"tiny \xff\xfe ppo")
    text = rejected(data)
    assert "manifest_not_utf8" in text
    assert "0xff" not in text
    assert "\\xff" not in text


def test_a_5000_digit_integer_is_rejected() -> None:
    data = manifest_bytes(good()).replace(b'"train_seed": 0', b'"train_seed": ' + b"9" * 5000)
    assert "manifest_big_int" in rejected(data)


def test_integers_at_and_beyond_2_53() -> None:
    read_manifest(with_change(("claims", "train_seed"), 2**53 - 1))
    assert "manifest_invalid: claims.train_seed" in rejected(
        with_change(("claims", "train_seed"), 2**53)
    )


def test_a_float_for_an_int_is_rejected() -> None:
    text = rejected(with_change(("action_dim",), 2.0))
    assert "manifest_invalid: action_dim (int_type)" in text


def test_an_int_for_a_float_is_accepted() -> None:
    m = read_manifest(with_change(("claims", "export_parity", "tolerance"), 1))
    assert m.claims.export_parity is not None
    assert m.claims.export_parity.tolerance == 1.0
    assert isinstance(m.claims.export_parity.tolerance, float)
    assert read_manifest(with_change(("control_dt",), 1)).control_dt == 1.0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("action_dim",), "2"),
        (("action_dim",), True),
        (("control_dt",), "0.01"),
        (("claims", "git_dirty"), 0),
        (("obs_features",), "pitch"),
    ],
    ids=["str-for-int", "bool-for-int", "str-for-float", "int-for-bool", "str-for-array"],
)
def test_scalars_are_strict(path: tuple[str, ...], value: object) -> None:
    assert "manifest_invalid" in rejected(with_change(path, value))


def test_deep_nesting_is_a_fixed_error() -> None:
    data = manifest_bytes(good()).replace(b'"name": "tiny ppo"', b'"name": ' + b"[" * 8000)
    assert len(data) < MAX_MANIFEST_BYTES
    assert rejected(data) == "policy.json: manifest_too_deep: policy.json nests too deeply"


@pytest.mark.parametrize("opener", [b"[", b'{"a":'])
def test_deep_nesting_never_reaches_a_parser(
    monkeypatch: pytest.MonkeyPatch, opener: bytes
) -> None:
    """The standard library's JSON scanner recurses in C: 8,190 levels (what 16 KiB holds)
    raise no ``RecursionError`` but need over 1 MiB of stack, which kills a thread with a
    smaller one (musl, a low ``ulimit -s``). Nesting is bounded by a byte scan first."""
    closer = b"]" if opener == b"[" else b"}"
    depth = MAX_MANIFEST_BYTES // (len(opener) + 1)
    data = opener * depth + closer * depth
    assert len(data) <= MAX_MANIFEST_BYTES

    def never(*args: object, **kwargs: object) -> object:
        raise AssertionError("a parser saw the nesting")

    monkeypatch.setattr(manifest_module.json, "loads", never)
    monkeypatch.setattr(manifest_module.PolicyManifest, "model_validate_json", never)
    assert "manifest_too_deep" in rejected(data)


def test_brackets_inside_strings_do_not_count_as_nesting() -> None:
    m = good()
    m["claims"]["versions"] = {"a": "[" * 60, "b": "{" * 60, "c": '\\"[[[[' * 10}
    assert read_manifest(manifest_bytes(m)).claims.versions == m["claims"]["versions"]


@pytest.mark.parametrize("depth", [32, 33])
def test_nesting_is_bounded_at_32_levels(depth: int) -> None:
    data = b"[" * depth + b"]" * depth
    text = rejected(data)
    assert ("manifest_too_deep" in text) == (depth > 32), text


def test_a_small_thread_stack_survives_the_deepest_manifest(tmp_path: Path) -> None:
    """Before the byte scan, this child died of SIGSEGV (exit -11) instead of a LoadError."""
    import subprocess
    import sys

    script = tmp_path / "deep.py"
    script.write_text(
        "import threading\n"
        "from topple.loader import LoadError\n"
        "from topple.policy.manifest import read_manifest\n"
        "out = []\n"
        "def run():\n"
        "    try:\n"
        "        read_manifest(b'[' * 8190 + b']' * 8190)\n"
        "    except LoadError as exc:\n"
        "        out.append(str(exc))\n"
        "threading.stack_size(256 * 1024)\n"
        "t = threading.Thread(target=run)\n"
        "t.start()\n"
        "t.join()\n"
        "print(out[0])\n"
    )
    done = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, (done.returncode, done.stderr[-2000:])
    assert "manifest_too_deep" in done.stdout


def test_not_json_at_all() -> None:
    assert "manifest_invalid_json" in rejected(b"{not json")
    assert "manifest_invalid" in rejected(b"[]")  # valid JSON, not an object


def test_only_bytes_are_read() -> None:
    with pytest.raises(LoadError, match="manifest_type"):
        read_manifest(manifest_bytes(good()).decode())  # type: ignore[arg-type]


def test_markup_never_reaches_the_error() -> None:
    m = good()
    m["<script>alert(1)</script>"] = 1
    m["claims"]["name"] = "<img src=x onerror=steal()>"
    text = rejected(manifest_bytes(m))
    assert "<" not in text
    assert ">" not in text
    assert "?script?alert?1??/script?" in text  # a key is a location: shown through safe
    assert "steal" not in text  # a value is never shown
    assert "extra_forbidden" in text
    assert "claims.name" in text


def test_missing_keys_are_named() -> None:
    assert "manifest_invalid: claims (missing)" in rejected(without(("claims",)))
    assert "manifest_invalid: onnx.sha256 (missing)" in rejected(without(("onnx", "sha256")))
    assert "claims.export_parity (missing)" in rejected(without(("claims", "export_parity")))


def test_message_count_is_bounded() -> None:
    m = good()
    for k in range(40):
        m[f"extra{k}"] = k
    text = rejected(manifest_bytes(m))
    assert text.count("extra_forbidden") == 8


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (ValueError("raw \xff text"), "manifest_invalid_json"),
        (TypeError("raw text"), "manifest_unreadable"),
        (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "raw"), "manifest_not_utf8"),
        (RecursionError("raw text"), "manifest_too_deep"),
        (MemoryError("raw text"), "manifest_out_of_memory"),
        (KeyError("raw text"), "manifest_unreadable"),
    ],
    ids=["value", "type", "unicode", "recursion", "memory", "other"],
)
def test_every_exception_class_is_a_fixed_code(
    monkeypatch: pytest.MonkeyPatch, raised: BaseException, code: str
) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise raised

    monkeypatch.setattr(manifest_module.json, "loads", boom)
    text = rejected(manifest_bytes(good()))
    assert code in text
    assert "raw" not in text


def test_a_validation_error_from_pydantic_never_shows_its_input() -> None:
    data = with_change(("challenge",), "evil‮-v1")  # a bidi override in a canonical id
    text = rejected(data)
    assert "manifest_invalid: challenge" in text
    assert "evil" not in text


# --- claims charset ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("claims", "name"), ""),
        (("claims", "name"), "x" * 65),
        (("claims", "name"), "tiny\nppo"),
        (("claims", "name"), "tiny ppo\n"),
        (("claims", "name"), "café"),
        (("claims", "name"), "a;b"),
        (("claims", "algo"), "dqn"),
        (("claims", "activation"), "gelu"),
        (("claims", "net_arch"), [8] * 9),
        (("claims", "net_arch"), [0]),
        (("claims", "net_arch"), [4097]),
        (("claims", "total_timesteps"), -1),
        (("claims", "selected_step"), 2**53),
        (("claims", "train_config_sha256"), "AB" * 32),
        (("claims", "train_config_sha256"), "ab" * 32 + "\n"),
        (("claims", "git_commit"), "c" * 39),
        (("claims", "git_commit"), "g" * 40),
        (("claims", "versions"), {f"k{i}": "1" for i in range(17)}),
        (("claims", "versions"), {"Torch": "2.14"}),
        (("claims", "versions"), {"k" * 33: "1"}),
        (("claims", "versions"), {"torch": "2.14\x1b[2J"}),
        (("claims", "versions"), {"torch": "x" * 65}),
        (("claims", "versions"), {"torch": 2}),
        (("claims", "export_parity", "reference"), "torch"),
        (("claims", "export_parity", "max_abs_diff"), -1.0),
        (("claims", "export_parity", "tolerance"), 0.0),
        (("claims", "export_parity", "n"), -1),
    ],
)
def test_claims_charset_and_bounds(path: tuple[str, ...], value: object) -> None:
    text = rejected(with_change(path, value))
    assert "manifest_invalid: claims." in text


def test_sixteen_printable_versions_are_accepted() -> None:
    versions = {f"pkg-{i}.x_y": " ~!" + str(i) for i in range(16)}
    m = read_manifest(with_change(("claims", "versions"), versions))
    assert m.claims.versions == versions


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("format",), "topple.policy/2"),
        (("kind",), "python"),
        (("challenge",), "Balance-v1"),
        (("challenge",), "a" * 62 + "-v1"),  # the grammar holds; 65 characters do not
        (("obs_schema",), "a" * 60 + ".obs.v1"),
        (("challenge",), "balance-v1\n"),
        (("obs_schema",), "obs_v1"),
        (("action_spec",), "diffdrive.effort"),
        (("obs_features",), []),
        (("obs_features",), ["Pitch"]),
        (("action_dim",), 0),
        (("control_dt",), 0.0),
        (("control_dt",), -0.01),
        (("action_rule",), "softmax"),
        (("onnx", "file"), "model.onnx"),
        (("onnx", "sha256"), "0" * 63),
        (("onnx", "input", "dtype"), "float64"),
        (("onnx", "input", "name"), "x"),
        (("onnx", "input", "shape"), ["batch", 10, 1]),
        (("onnx", "input", "shape"), ["N", 10]),
        (("onnx", "input", "shape"), ["batch", 0]),
        (("onnx", "output", "shape"), ["batch", 4097]),
    ],
)
def test_manifest_fields_are_checked(path: tuple[str, ...], value: object) -> None:
    assert "manifest_invalid" in rejected(with_change(path, value))


# --- check_against_model ----------------------------------------------------------------------


def manifest_of(**changes: Any) -> PolicyManifest:
    m = good()
    m["onnx"] = {**m["onnx"], **changes}
    return read_manifest(manifest_bytes(m))


def test_check_against_model_accepts_the_build() -> None:
    assert check_against_model(read_manifest(manifest_bytes(good())), parse(PPO)) == []
    sac = first_party_manifest(sac_bytes(), action_rule="tanh(mean)")
    assert check_against_model(read_manifest(manifest_bytes(sac)), parse(sac_bytes())) == []


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"ir_version": 9}, "onnx.ir_version"),
        ({"opset": 18}, "onnx.opset"),
        ({"input": {"name": "action", "dtype": "float32", "shape": ["batch", 10]}}, "onnx.input"),
        ({"input": {"name": "obs", "dtype": "float32", "shape": ["batch", 11]}}, "onnx.input"),
        (
            {"output": {"name": "obs", "dtype": "float32", "shape": ["batch", 2]}},
            "onnx.output",
        ),
        (
            {"output": {"name": "action", "dtype": "float32", "shape": ["batch", 3]}},
            "onnx.output",
        ),
    ],
    ids=["ir", "opset", "input-name", "input-width", "output-name", "output-width"],
)
def test_check_against_model_per_field(changes: dict[str, Any], named: str) -> None:
    problems = check_against_model(manifest_of(**changes), parse(PPO))
    assert problems == [f"manifest_model_mismatch: {named} differs from policy.onnx"]


def test_check_against_model_needs_one_opset_and_typed_io() -> None:
    proto = parse(PPO)
    proto.opset_import.add(domain="ai.onnx.ml", version=3)
    proto.graph.input[0].type.tensor_type.elem_type = 11  # double
    proto.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "N"
    problems = check_against_model(read_manifest(manifest_bytes(good())), proto)
    assert problems == [
        "manifest_model_mismatch: onnx.opset differs from policy.onnx",
        "manifest_model_mismatch: onnx.input differs from policy.onnx",
        "manifest_model_mismatch: onnx.output differs from policy.onnx",
    ]


def test_check_against_model_needs_every_dim_set() -> None:
    proto = parse(PPO)
    proto.graph.input[0].type.tensor_type.shape.dim[1].Clear()  # neither dim_value nor dim_param
    problems = check_against_model(read_manifest(manifest_bytes(good())), proto)
    assert problems == ["manifest_model_mismatch: onnx.input differs from policy.onnx"]


def test_check_against_model_survives_a_model_without_io() -> None:
    proto = parse(PPO)
    del proto.graph.input[:]
    proto.graph.output[0].type.Clear()
    problems = check_against_model(read_manifest(manifest_bytes(good())), proto)
    assert "manifest_model_mismatch: onnx.input differs from policy.onnx" in problems
    assert "manifest_model_mismatch: onnx.output differs from policy.onnx" in problems


# --- check_against_challenge ------------------------------------------------------------------


def test_check_against_challenge_accepts_balance() -> None:
    assert check_against_challenge(read_manifest(manifest_bytes(good())), BALANCE) == []


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("challenge",), "other-v1", "challenge must be 'balance-v1'"),
        (("obs_schema",), "balancer-target.obs.v1", "obs_schema must be 'balancer.obs.v1'"),
        (
            ("obs_features",),
            ["pitch"] * 10,
            "obs_features must be balance-v1's 10 features in order",
        ),
        (
            ("action_spec",),
            "diffdrive.effort.v2",
            "action_spec must be 'diffdrive.effort.v1'",
        ),
        (("action_dim",), 3, "action_dim must be 2"),
        (("control_dt",), 0.02, "control_dt must be 0.01"),
    ],
    ids=["challenge", "obs_schema", "obs_features", "action_spec", "action_dim", "control_dt"],
)
def test_challenge_mismatches_are_named(path: tuple[str, ...], value: object, message: str) -> None:
    m = read_manifest(with_change(path, value))
    assert check_against_challenge(m, BALANCE) == [f"challenge_mismatch: {message}"]


def test_policy_task_reads_dims_and_bounds_from_the_challenge() -> None:
    task = policy_task(BALANCE)
    assert task.challenge == "balance-v1"
    assert task.obs_dim == 10
    assert task.action_dim == 2
    assert (task.action.low, task.action.high) == (-1.0, 1.0)
    assert task.control_dt == BALANCE.config.control_dt


def test_dump_is_ascii_json_with_a_newline() -> None:
    m = read_manifest(manifest_bytes(good()))
    data = dump_manifest(m)
    assert data.endswith(b"}\n")
    assert data.isascii()


def test_manifest_copies_compare_by_value() -> None:
    a = read_manifest(manifest_bytes(good()))
    b = read_manifest(manifest_bytes(copy.deepcopy(good())))
    assert a == b
