"""``topple policy pack``: a third-party ONNX model as a ``topple.policy/1`` directory
(docs/design/phase-2.md §3.3, decision log B8).

A model exported by someone else's toolchain (``torch.onnx.export``, legacy or dynamo) carries
doc strings, ``value_info``, producer fields, ``Identity`` and ``Constant`` nodes, its own I/O
names and often a static batch of 1: the loader refuses all of that (check 9, "run topple
policy pack"). Packing rebuilds it in canonical form and states what it is for:

1. check 1: at most 1 MiB, parsed with ``onnx.ModelProto.FromString`` only;
2. ``canonicalize(pack=True)`` (§3.4): external data refused before any tensor is read; every
   field off the allow-list dropped; ``Identity`` removed, ``Constant`` folded, I/O renamed
   ``obs``/``action``, a static batch of 1 made ``batch``, IR set to ``CANONICAL_IR[opset]``;
3. the 7 ``topple.*`` metadata keys set from the challenge, replacing any the model carried
   (B8), with ``action_rule`` from the structure match; checks 1-9;
4. ``policy.json`` with the challenge's ids, dims and control period, the graph's header and
   I/O, and every claim null (``export_parity`` included: nothing was exported here).

The two files are then loaded exactly as a submission is (``load_policy_files``: manifest,
sha256 binding, checks 1-9, model and metadata against the manifest, the manifest against the
challenge, the smoke test) before anything is written. Every failure is one ``LoadError`` of
fixed vocabulary: a message never echoes the model.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final

import onnx
from google.protobuf.message import DecodeError

from topple.loader import LoadError
from topple.policy.manifest import (
    MANIFEST_FILE,
    ONNX_FILE,
    Claims,
    OnnxInfo,
    PolicyManifest,
    PolicyTask,
    TensorSig,
    dump_manifest,
    policy_task,
    task_fields,
)
from topple.policy.runtime import load_policy_files
from topple.policy.validate import (
    BATCH,
    INPUT_NAME,
    MAX_ONNX_BYTES,
    OUTPUT_NAME,
    POLICY_FORMAT,
    ActionRule,
    canonicalize,
    structure_match,
    validate_policy_bytes,
)

if TYPE_CHECKING:
    from topple.challenge import Challenge

# The codes ``canonicalize`` raises (each with fixed text); any other exception is reported as
# ``pack_failed`` without its text, which could quote the model.
CANONICAL_CODES: Final = frozenset(
    {
        "bad_opset_import",
        "both_payloads",
        "constant_value",
        "custom_domain",
        "duplicate_attribute",
        "duplicate_value",
        "external_data",
        "identity_cycle",
        "invalid_utf8",
        "io_count",
        "node_outputs",
        "output_not_computed",
        "undefined_value",
        "unsupported_opset",
    }
)
NULL_CLAIMS: Final = Claims(
    name=None,
    algo=None,
    activation=None,
    net_arch=None,
    total_timesteps=None,
    selected_step=None,
    train_seed=None,
    train_config_sha256=None,
    git_commit=None,
    git_dirty=None,
    selection_eligible=None,
    versions=None,
    export_parity=None,
)


def pack_policy(data: bytes, *, challenge: Challenge) -> dict[str, bytes]:
    """The files of a ``topple.policy/1`` directory (``policy.onnx``, ``policy.json``) for the
    model bytes ``data`` and ``challenge``, already loaded once as the loader loads a
    submission; ``LoadError`` with a fixed code for anything else (§3.3 pack steps 1-4)."""
    task = policy_task(challenge)
    canonical = _canonical(_parsed(data))
    proto: onnx.ModelProto = onnx.ModelProto.FromString(canonical)
    rule = structure_match(proto)
    del proto.metadata_props[:]
    for field, value in sorted(task_fields(task, rule).items()):  # canonical: sorted by key
        entry = proto.metadata_props.add()
        entry.key, entry.value = f"topple.{field}", json.dumps(value)
    packed: bytes = proto.SerializeToString(deterministic=True)
    result = validate_policy_bytes(packed, task.obs_dim, task.action_dim)
    if isinstance(result, list):
        raise LoadError(f"{ONNX_FILE}: " + "; ".join(result))
    files = {ONNX_FILE: packed, MANIFEST_FILE: dump_manifest(_manifest(proto, packed, rule, task))}
    load_policy_files(files, challenge=challenge)  # exactly what loading the directory runs
    return files


def _parsed(data: bytes) -> onnx.ModelProto:
    """Check 1: at most ``MAX_ONNX_BYTES``, then ``FromString`` (never a path or a loader that
    could follow external data)."""
    if type(data) is not bytes:
        raise LoadError("model: onnx_parse_failed: the model must be bytes")
    if len(data) > MAX_ONNX_BYTES:
        raise LoadError(f"model: onnx_too_large: the model has more than {MAX_ONNX_BYTES} bytes")
    try:
        proto: onnx.ModelProto = onnx.ModelProto.FromString(data)
    except (DecodeError, RecursionError, ValueError):
        raise LoadError("model: onnx_parse_failed: the file is not an ONNX model") from None
    return proto


def _canonical(proto: onnx.ModelProto) -> bytes:
    """``canonicalize(proto, pack=True)``, its refusals as ``LoadError`` (fixed text only)."""
    try:
        return canonicalize(proto, pack=True)
    except ValueError as exc:
        code = str(exc).split(":", 1)[0]
        if code in CANONICAL_CODES:
            raise LoadError(f"model: {exc}") from None
        raise LoadError("model: pack_failed: raised while canonicalizing") from None
    except Exception:
        raise LoadError("model: pack_failed: raised while canonicalizing") from None


def _manifest(
    proto: onnx.ModelProto, packed: bytes, rule: ActionRule, task: PolicyTask
) -> PolicyManifest:
    """``policy.json`` for the packed bytes: the challenge's ids, features, action spec and
    control period, the graph's header and I/O, and no claims."""
    return PolicyManifest(
        format=POLICY_FORMAT,
        kind="onnx",
        challenge=task.challenge,
        obs_schema=task.obs_schema,
        obs_features=task.obs_features,
        action_spec=task.action.id,
        action_dim=task.action_dim,
        control_dt=task.control_dt,
        action_rule=rule,
        onnx=OnnxInfo(
            file="policy.onnx",
            sha256=hashlib.sha256(packed).hexdigest(),
            ir_version=proto.ir_version,
            opset=proto.opset_import[0].version,
            input=TensorSig(name=INPUT_NAME, dtype="float32", shape=(BATCH, task.obs_dim)),
            output=TensorSig(name=OUTPUT_NAME, dtype="float32", shape=(BATCH, task.action_dim)),
        ),
        claims=NULL_CLAIMS,
    )
