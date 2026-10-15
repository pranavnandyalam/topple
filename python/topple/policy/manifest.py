"""``policy.json``, the manifest of a ``topple.policy/1`` directory (docs/design/phase-2.md §3.1).

The manifest is the normative description of a policy (ONNX Runtime Web cannot read
``metadata_props``), so it is parsed as hostile input. Parsing is two passes over at most
``MAX_MANIFEST_BYTES``: a rejection-only pass with the standard library (no BOM, valid UTF-8,
no duplicate keys, which pydantic would silently resolve to the last one, no ``NaN`` or
``Infinity`` tokens, no number beyond float range, no integer literal longer than
``MAX_INT_DIGITS``, no lone surrogate in any key or string), then ``model_validate_json`` on
strict, frozen models (JSON arrays become tuples, an ``int`` is accepted for a ``float``, never
the reverse). Every failure is one ``LoadError`` with a fixed code: a message never quotes the
input, only fixed words and ``safe`` field names (an ``UnicodeDecodeError`` holds the offending
bytes, a pydantic message holds the offending value).

``claims`` are self-reported by whoever wrote the file: they are displayed as such and never
used for ranking, divisions or anything called verified.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, NoReturn

import onnx
from onnx import TensorProto
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from topple.contract import ActionSpec, validate_schema_id
from topple.ids import validate_challenge_id
from topple.loader import LoadError
from topple.policy.validate import MAX_DIM, MAX_MESSAGES, POLICY_FORMAT, safe
from topple.rng import MAX_SEED

if TYPE_CHECKING:
    from topple.challenge import Challenge

MANIFEST_FILE: Final = "policy.json"
ONNX_FILE: Final = "policy.onnx"
MAX_MANIFEST_BYTES: Final = 16 * 1024
MAX_INT_DIGITS: Final = 20  # far beyond 2**53 (16 digits); never reaches int()'s 4300-digit cap
MAX_ID_CHARS: Final = 64
MAX_FEATURES: Final = MAX_DIM
MAX_NET_ARCH: Final = 8
MAX_VERSIONS: Final = 16
MAX_IR_VERSION: Final = 64
MAX_OPSET: Final = 64
_BOM: Final = b"\xef\xbb\xbf"

# Every pattern is used with ``fullmatch`` (``re.match`` with ``$`` accepts a trailing "\n").
HEX64: Final = re.compile(r"[0-9a-f]{64}")
PRINTABLE: Final = re.compile(r"[\x20-\x7e]{0,64}")
GIT_COMMIT: Final = re.compile(r"[0-9a-f]{40}")
CLAIM_NAME: Final = re.compile(r"[A-Za-z0-9 _.-]{1,64}")
VERSION_KEY: Final = re.compile(r"[a-z0-9_.-]{1,32}")
FEATURE_NAME: Final = re.compile(r"[a-z][a-z0-9_]{0,63}")

ActionRule = Literal["clip(mean)", "tanh(mean)", "other"]


def _matching(pattern: re.Pattern[str]) -> AfterValidator:
    """A validator for strings that ``pattern`` matches in full. Its message never quotes the
    value (and is not shown anyway: only a field's location and error type are)."""

    def check(value: str) -> str:
        if not pattern.fullmatch(value):
            raise ValueError("does not match the required pattern")
        return value

    return AfterValidator(check)


def _challenge_id(value: str) -> str:
    if len(value) > MAX_ID_CHARS:
        raise ValueError("too long")
    return validate_challenge_id(value)


def _schema_id(value: str) -> str:
    if len(value) > MAX_ID_CHARS:
        raise ValueError("too long")
    return validate_schema_id(value)


Hex64 = Annotated[str, _matching(HEX64)]
SafeInt = Annotated[int, Field(ge=0, lt=MAX_SEED)]  # exact as a JSON / JavaScript number
Dim = Annotated[int, Field(ge=1, le=MAX_DIM)]
ChallengeId = Annotated[str, AfterValidator(_challenge_id)]
SchemaId = Annotated[str, AfterValidator(_schema_id)]
FeatureName = Annotated[str, _matching(FEATURE_NAME)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class TensorSig(_Strict):
    """An I/O tensor of the graph: ``obs`` or ``action``, float32 ``["batch", dim]``."""

    name: Literal["obs", "action"]
    dtype: Literal["float32"]
    shape: tuple[Literal["batch"], Dim]


class OnnxInfo(_Strict):
    """The graph file: ``sha256`` is sha256(policy.onnx), the policy's identity
    (``policy_sha256``), and the header and I/O must equal the parsed model's."""

    file: Literal["policy.onnx"]
    sha256: Hex64
    ir_version: Annotated[int, Field(ge=1, le=MAX_IR_VERSION)]
    opset: Annotated[int, Field(ge=1, le=MAX_OPSET)]
    input: TensorSig
    output: TensorSig


class ExportParity(_Strict):
    """The exporter's own parity self-check (§3.3 step 4): never shown as verified."""

    reference: Literal["sb3.predict"]
    n: SafeInt
    n_onpolicy: SafeInt
    n_onpolicy_seeds: SafeInt
    n_unsaturated: SafeInt
    n_saturated: SafeInt
    max_abs_diff: Annotated[float, Field(ge=0.0)]
    tolerance: Annotated[float, Field(gt=0.0)]
    vacuous: bool


class Claims(_Strict):
    """SELF-REPORTED: displayed as such; never used for ranking, divisions or "verified".
    Every key is required (``null`` when unknown): loosening is additive, tightening would be a
    format bump."""

    name: Annotated[str, _matching(CLAIM_NAME)] | None
    algo: Literal["ppo", "sac", "other"] | None
    activation: Literal["tanh", "relu", "other"] | None
    net_arch: (
        Annotated[
            tuple[Annotated[int, Field(ge=1, le=MAX_DIM)], ...], Field(max_length=MAX_NET_ARCH)
        ]
        | None
    )
    total_timesteps: SafeInt | None
    selected_step: SafeInt | None
    train_seed: SafeInt | None
    train_config_sha256: Hex64 | None
    git_commit: Annotated[str, _matching(GIT_COMMIT)] | None
    git_dirty: bool | None
    selection_eligible: bool | None
    versions: (
        Annotated[
            dict[Annotated[str, _matching(VERSION_KEY)], Annotated[str, _matching(PRINTABLE)]],
            Field(max_length=MAX_VERSIONS),
        ]
        | None
    )
    export_parity: ExportParity | None


class PolicyManifest(_Strict):
    """``policy.json``: what the policy is for (canonical ids, features, action spec, control
    period), its head (``action_rule``), its graph file and the author's claims."""

    format: Literal["topple.policy/1"]
    kind: Literal["onnx"]
    challenge: ChallengeId
    obs_schema: SchemaId
    obs_features: Annotated[tuple[FeatureName, ...], Field(min_length=1, max_length=MAX_FEATURES)]
    action_spec: SchemaId
    action_dim: Dim
    control_dt: Annotated[float, Field(gt=0.0)]
    action_rule: ActionRule
    onnx: OnnxInfo
    claims: Claims


# --- reading ----------------------------------------------------------------------------------


class _RejectedError(Exception):
    """A fixed-code rejection raised inside the pre-pass (never shown as exception text)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


# An exception escaping the parse becomes one of these codes, never its text. Order matters:
# ``UnicodeDecodeError`` and ``json.JSONDecodeError`` are ``ValueError``s.
_EXCEPTION_CODES: Final[tuple[tuple[type[BaseException], str, str], ...]] = (
    (UnicodeError, "manifest_not_utf8", "policy.json is not valid UTF-8"),
    (RecursionError, "manifest_too_deep", "policy.json nests too deeply"),
    (MemoryError, "manifest_out_of_memory", "policy.json could not be parsed in memory"),
    (ValueError, "manifest_invalid_json", "policy.json is not valid JSON"),
    (TypeError, "manifest_unreadable", "raised while parsing"),
)


def _fail(code: str, detail: str) -> NoReturn:
    raise LoadError(f"{MANIFEST_FILE}: {code}: {detail}")


def read_manifest(data: bytes) -> PolicyManifest:
    """``data`` (the exact bytes of ``policy.json``) as a ``PolicyManifest``; anything else is
    one ``LoadError`` with a fixed code (§3.1 "Parsing", §3.4 "Error text")."""
    if type(data) is not bytes:
        _fail("manifest_type", "policy.json must be read as bytes")
    if len(data) > MAX_MANIFEST_BYTES:
        _fail("manifest_too_large", f"policy.json has more than {MAX_MANIFEST_BYTES} bytes")
    if data.startswith(_BOM):
        _fail("manifest_bom", "policy.json starts with a byte order mark")
    message: str
    try:
        _prepass(data)
        manifest = PolicyManifest.model_validate_json(data)
    except _RejectedError as exc:
        message = f"{exc.code}: {exc.detail}"
    except ValidationError as exc:
        message = _validation_message(exc)
    except Exception as exc:  # never a traceback or raw exception text (§3.4 "Error text")
        code, detail = next(
            ((c, d) for kind, c, d in _EXCEPTION_CODES if isinstance(exc, kind)),
            ("manifest_unreadable", "raised while parsing"),
        )
        message = f"{code}: {detail}"
    else:
        return manifest
    raise LoadError(f"{MANIFEST_FILE}: {message}") from None  # no raw text in the chain


def _prepass(data: bytes) -> None:
    """Reject what ``model_validate_json`` would accept or resolve silently: duplicate keys,
    ``NaN``/``Infinity``, numbers beyond float range, huge integer literals and keys or strings
    with no UTF-8 form (lone surrogates written as ``\\ud800`` escapes)."""
    value = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        parse_int=_bounded_int,
    )
    for text in _strings(value):
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise _RejectedError(
                "manifest_invalid_string", "a key or string of policy.json has no UTF-8 form"
            ) from None


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise _RejectedError("manifest_duplicate_key", "policy.json repeats a key in an object")
    return dict(pairs)


def _reject_constant(token: str) -> NoReturn:
    raise _RejectedError("manifest_nonfinite_number", "policy.json holds NaN or Infinity")


def _finite_float(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):
        raise _RejectedError(
            "manifest_nonfinite_number", "policy.json holds a number beyond float range"
        )
    return value


def _bounded_int(literal: str) -> int:
    if len(literal.lstrip("-")) > MAX_INT_DIGITS:
        raise _RejectedError(
            "manifest_big_int", f"an integer has more than {MAX_INT_DIGITS} digits"
        )
    return int(literal)


def _strings(value: object) -> Iterator[str]:
    """Every key and string inside ``value`` (at most 16 KiB of JSON), without recursion."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for key, child in item.items():
                yield key
                stack.append(child)
        elif isinstance(item, list):
            stack.extend(item)


def _validation_message(exc: ValidationError) -> str:
    """At most ``MAX_MESSAGES`` field errors as ``manifest_invalid: <loc> (<type>)``: only the
    location (through ``safe``) and the error type, never pydantic's ``msg``, ``input`` or
    ``ctx``, which quote the value."""
    parts = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False)[
        :MAX_MESSAGES
    ]:
        loc = ".".join(safe(str(p)) for p in error["loc"]) or "(root)"
        parts.append(f"manifest_invalid: {loc} ({safe(error['type'])})")
    return "; ".join(parts) or "manifest_invalid: (root) (invalid)"


def dump_manifest(m: PolicyManifest) -> bytes:
    """The bytes ``policy.json`` is written as: indented ASCII JSON in field order, then a
    newline. ``read_manifest(dump_manifest(m)) == m``."""
    text = json.dumps(m.model_dump(mode="json"), indent=2, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("ascii")


# --- what the manifest must match -------------------------------------------------------------


@dataclass(frozen=True)
class PolicyTask:
    """What a policy of a challenge maps: the challenge's ids, its observation features in
    order, its action spec (dims and head bounds) and its control period."""

    challenge: str
    obs_schema: str
    obs_features: tuple[str, ...]
    action: ActionSpec
    control_dt: float

    @property
    def obs_dim(self) -> int:
        return len(self.obs_features)

    @property
    def action_dim(self) -> int:
        return self.action.dim


# ``Challenge.task_spec`` also takes the robot geometry, which changes neither the features nor
# the action spec; a caller without an env (the loader, a pool's parent) has none to give.
_ANY_GEOMETRY: Final = 1.0


def policy_task(ch: Challenge) -> PolicyTask:
    """The challenge's dims and bounds, from its ``TaskSpec`` (§3.5): never from a manifest,
    a model or a training action space."""
    spec = ch.task_spec(wheel_radius_m=_ANY_GEOMETRY, track_width_m=_ANY_GEOMETRY)
    return PolicyTask(
        challenge=spec.challenge,
        obs_schema=spec.obs_schema,
        obs_features=spec.obs_features,
        action=spec.action,
        control_dt=spec.control_dt,
    )


def check_against_challenge(m: PolicyManifest, ch: Challenge) -> list[str]:
    """Each field of ``m`` that differs from the challenge's, as ``challenge_mismatch: <field>
    must be <the challenge's value>`` (the expected value is trusted; the manifest's is never
    quoted)."""
    task = policy_task(ch)
    problems: list[str] = []
    if m.challenge != task.challenge:
        problems.append(f"challenge must be {task.challenge!r}")
    if m.obs_schema != task.obs_schema:
        problems.append(f"obs_schema must be {task.obs_schema!r}")
    if m.obs_features != task.obs_features:
        problems.append(f"obs_features must be {task.challenge}'s {task.obs_dim} features in order")
    if m.action_spec != task.action.id:
        problems.append(f"action_spec must be {task.action.id!r}")
    if m.action_dim != task.action_dim:
        problems.append(f"action_dim must be {task.action_dim}")
    if m.control_dt != task.control_dt:
        problems.append(f"control_dt must be {task.control_dt!r}")
    return [f"challenge_mismatch: {p}" for p in problems]


def check_against_model(m: PolicyManifest, proto: onnx.ModelProto) -> list[str]:
    """Each ``onnx`` field of ``m`` that differs from ``proto``: ``ir_version``, the single
    opset version, and the input and output (name, float32, ``("batch", dim)``). Never raises
    for any parsed ``proto``."""
    problems: list[str] = []
    if proto.ir_version != m.onnx.ir_version:
        problems.append("onnx.ir_version")
    opsets = [o.version for o in proto.opset_import]
    if opsets != [m.onnx.opset]:
        problems.append("onnx.opset")
    graph = proto.graph
    if [_signature(info) for info in graph.input] != [_expected(m.onnx.input)]:
        problems.append("onnx.input")
    if [_signature(info) for info in graph.output] != [_expected(m.onnx.output)]:
        problems.append("onnx.output")
    return [f"manifest_model_mismatch: {field} differs from policy.onnx" for field in problems]


def _expected(sig: TensorSig) -> tuple[str, int, tuple[tuple[str, str | int], ...]]:
    batch, dim = sig.shape
    return sig.name, TensorProto.FLOAT, (("dim_param", batch), ("dim_value", dim))


def _signature(info: onnx.ValueInfoProto) -> tuple[str, int, tuple[tuple[str, str | int], ...]]:
    """(name, element type, dims as (kind, value)) of a graph input or output."""
    tensor = info.type.tensor_type
    dims: list[tuple[str, str | int]] = []
    for dim in tensor.shape.dim:
        which = dim.WhichOneof("value")
        if which == "dim_param":
            dims.append(("dim_param", str(dim.dim_param)))
        elif which == "dim_value":
            dims.append(("dim_value", int(dim.dim_value)))
        else:
            dims.append(("none", 0))
    name: str = info.name
    elem_type: int = tensor.elem_type
    return name, elem_type, tuple(dims)


# --- the 7 ``topple.*`` metadata values -------------------------------------------------------

METADATA_SOURCE_FIELDS: Final = (
    "format",
    "challenge",
    "obs_schema",
    "obs_features",
    "action_spec",
    "control_dt",
    "action_rule",
)


def task_fields(task: PolicyTask, action_rule: str) -> dict[str, object]:
    """The manifest fields the 7 metadata values restate, as the challenge fixes them."""
    return {
        "format": POLICY_FORMAT,
        "challenge": task.challenge,
        "obs_schema": task.obs_schema,
        "obs_features": list(task.obs_features),
        "action_spec": task.action.id,
        "control_dt": task.control_dt,
        "action_rule": action_rule,
    }


def _normalized(value: object) -> str:
    """Canonical JSON text of ``value``: equal texts mean equal JSON values *and* types (in
    Python ``1 == 1.0 == True``; in JSON they are three values)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def metadata_mismatches(proto: onnx.ModelProto, expected: dict[str, object]) -> list[str]:
    """The ``topple.<field>`` metadata keys of ``proto`` whose JSON value is not
    ``expected[field]`` (both sides normalized, type-exact). The values passed check 4 (ASCII,
    at most 1024 bytes); one that is not strict JSON (``NaN``, a duplicate key) mismatches."""
    metadata = {entry.key: entry.value for entry in proto.metadata_props}
    mismatched: list[str] = []
    for field in METADATA_SOURCE_FIELDS:
        key = f"topple.{field}"
        try:
            value = _strict_json(metadata[key])
            same = _normalized(value) == _normalized(expected[field])
        except Exception:  # a missing key, bad JSON, NaN: all simply "differs"
            same = False
        if not same:
            mismatched.append(key)
    return mismatched


def _strict_json(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        parse_int=_bounded_int,
    )
