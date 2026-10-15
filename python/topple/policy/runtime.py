"""ONNX Runtime sessions for validated policies (docs/design/phase-2.md §3.5).

``make_session`` is the only place a session is created, and it takes only a
``ValidatedPolicy``: ONNX Runtime sees canonical bytes that passed every acceptance check, never
a path (a path would let the model name external-data files) and never options a caller or the
environment could loosen. One thread, sequential execution, no graph optimizations (measured
bitwise equal to the basic level, and the smaller optimizer surface), deterministic compute,
the ONNX format pinned, released opsets only, strict shape and type inference, and the CPU
provider, asserted after creation.
"""

from __future__ import annotations

import os

import onnx
import onnxruntime as ort

from topple.policy.validate import ValidatedPolicy, reject_external_data

PROVIDERS = ("CPUExecutionProvider",)


def make_session(v: ValidatedPolicy) -> ort.InferenceSession:
    """An ``InferenceSession`` for ``v``'s canonical bytes with the §3.5 settings.

    ``TypeError`` for anything but a ``ValidatedPolicy``; ``ValueError`` if its bytes point at
    external data (re-asserted here, though validation already refuses it); ``RuntimeError`` if
    ONNX Runtime did not settle on the CPU provider alone.
    """
    if not isinstance(v, ValidatedPolicy):
        raise TypeError("make_session takes a ValidatedPolicy from validate_policy_bytes")
    reject_external_data(onnx.ModelProto.FromString(v.canonical))
    _scrub_ort_environment()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    options.use_deterministic_compute = True
    options.add_session_config_entry("session.load_model_format", "ONNX")
    options.add_session_config_entry("session.allow_released_opsets_only", "1")
    options.add_session_config_entry("session.strict_shape_type_inference", "1")
    session = ort.InferenceSession(v.canonical, sess_options=options, providers=list(PROVIDERS))
    if tuple(session.get_providers()) != PROVIDERS:
        raise RuntimeError("ONNX Runtime must run policies on CPUExecutionProvider only")
    return session


def _scrub_ort_environment() -> None:
    """Remove every ``ORT_*`` variable before a session exists: ONNX Runtime reads some of them
    when it creates its environment, and a policy run must not depend on the caller's shell."""
    for key in [k for k in os.environ if k.upper().startswith("ORT_")]:
        del os.environ[key]
