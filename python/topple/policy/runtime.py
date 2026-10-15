"""Loading ``topple.policy/1`` directories and running their sessions
(docs/design/phase-2.md §3.1, §3.5).

``make_session`` is the only place a session is created, and it takes only a
``ValidatedPolicy``: ONNX Runtime sees canonical bytes that passed every acceptance check, never
a path (a path would let the model name external-data files) and never options a caller or the
environment could loosen. One thread, sequential execution, no graph optimizations (measured
bitwise equal to the basic level, and the smaller optimizer surface), deterministic compute,
the ONNX format pinned, released opsets only, strict shape and type inference, and the CPU
provider, asserted after creation.

``load_policy_dir`` turns a directory into a ``LoadedPolicy``: the directory rules (exact names,
a bounded listing, regular single-link files read once), the manifest, the sha256 binding, the
nine acceptance checks, the manifest against the model, the structure against
``action_rule``, the graph's metadata against the manifest, the manifest against the
challenge, then the smoke test. Dims and action bounds come only from the challenge. Any failure
is one ``LoadError`` whose text is fixed vocabulary plus ``safe`` names.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn

import onnx
import onnxruntime as ort

from topple._inputs import read_regular_file
from topple.loader import LoadError
from topple.policy.manifest import (
    MANIFEST_FILE,
    MAX_MANIFEST_BYTES,
    ONNX_FILE,
    PolicyManifest,
    check_against_challenge,
    check_against_model,
    metadata_mismatches,
    policy_task,
    read_manifest,
    task_fields,
)
from topple.policy.validate import (
    MAX_ONNX_BYTES,
    ValidatedPolicy,
    reject_external_data,
    safe,
    smoke_test,
    structure_match,
    validate_policy_bytes,
)

if TYPE_CHECKING:
    from topple.challenge import Challenge

PROVIDERS = ("CPUExecutionProvider",)
README_FILE: Final = "README.md"
MAX_README_BYTES: Final = 64 * 1024
# The only entries a policy directory may hold, with the most bytes each may have.
POLICY_FILES: Final[Mapping[str, int]] = {
    ONNX_FILE: MAX_ONNX_BYTES,
    MANIFEST_FILE: MAX_MANIFEST_BYTES,
    README_FILE: MAX_README_BYTES,
}
REQUIRED_FILES: Final = (ONNX_FILE, MANIFEST_FILE)
PYTHON_ENTRY: Final = "controller.py"
AMBIGUOUS: Final = "ambiguous submission kind"


def make_session(v: ValidatedPolicy) -> ort.InferenceSession:
    """An ``InferenceSession`` for ``v``'s canonical bytes with the §3.5 settings.

    ``TypeError`` for anything but a ``ValidatedPolicy``; ``ValueError`` if its bytes point at
    external data (re-asserted here, though validation already refuses it); ``RuntimeError`` if
    ONNX Runtime did not settle on the CPU provider alone.
    """
    if type(v) is not ValidatedPolicy:  # exactly: a subclass could have skipped the seal
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


@dataclass(frozen=True)
class LoadedPolicy:
    """A policy that passed every load check: its manifest (``None`` for bare bytes), its
    validated canonical bytes and a session over them.

    Never pickled, copied or sent to another process: a pool sends a ``topple.pool.PolicyRef``
    and the worker loads its own (§3.5).
    """

    manifest: PolicyManifest | None
    validated: ValidatedPolicy
    session: ort.InferenceSession = field(repr=False, compare=False)

    def __reduce__(self) -> NoReturn:
        raise TypeError("a LoadedPolicy is never pickled: send a topple.pool.PolicyRef")


# --- the directory ----------------------------------------------------------------------------


def read_policy_dir(path: Path) -> dict[str, bytes]:
    """Each file of the policy directory ``path``, read once, by name.

    ``path`` must be a real directory (not a symlink) holding exactly ``policy.onnx``,
    ``policy.json`` and optionally ``README.md``, named exactly as ``os.scandir`` returns them.
    The listing stops at the fourth entry, so a directory of a million files costs four. Any
    other entry is ``unexpected entry: '<safe name>'``, a name that is not UTF-8 is refused,
    ``controller.py`` or any ``.py`` among the entries listed is ``ambiguous submission
    kind``; each file is read with ``read_regular_file`` (no symlink, one link, its size cap,
    never blocking). A path that cannot be examined is a ``LoadError``, never an ``OSError``.
    """
    try:
        before = os.lstat(path)
        _check_directory(path, before)
        names = _entries(path)
        files = {name: read_regular_file(path / name, POLICY_FILES[name]) for name in names}
        after = os.lstat(path)
    except OSError as exc:
        raise LoadError(
            f"cannot read {safe(path.name)!r}: {exc.strerror or type(exc).__name__}"
        ) from exc
    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        raise LoadError(f"{safe(path.name)!r} changed while it was read")
    return files


def _check_directory(path: Path, info: os.stat_result) -> None:
    if stat.S_ISLNK(info.st_mode):
        raise LoadError(f"{safe(path.name)!r} is a symlink; submit the directory itself")
    if not stat.S_ISDIR(info.st_mode):
        raise LoadError(f"{safe(path.name)!r} is not a directory")
    if _exists(path / PYTHON_ENTRY):  # the usual ambiguous case, wherever the listing stops
        raise LoadError(AMBIGUOUS)


def _exists(path: Path) -> bool:
    """Whether anything (a symlink included) is at ``path``; ``OSError`` if it cannot tell."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _entries(path: Path) -> list[str]:
    """The names in ``path``, sorted, if they are a valid policy directory's; else
    ``LoadError``. At most ``len(POLICY_FILES) + 1`` entries are listed: names are unique, so
    a further entry is always one too many."""
    names: list[str] = []
    problems: list[str] = []
    with os.scandir(path) as listing:
        for entry in itertools.islice(listing, len(POLICY_FILES) + 1):
            problem = _entry_problem(entry.name)
            if problem is None:
                names.append(entry.name)
            else:
                problems.append(problem)
    if AMBIGUOUS in problems:  # the more useful message, whatever the listing order
        raise LoadError(AMBIGUOUS)
    if problems:
        raise LoadError(problems[0])
    for required in REQUIRED_FILES:
        if required not in names:
            raise LoadError(f"missing {required!r}")
    return sorted(names)


def _entry_problem(name: str) -> str | None:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:  # a non-UTF-8 name decodes to lone surrogates (os.fsdecode)
        return f"file name {safe(name)!r} is not valid UTF-8"
    if name.casefold().endswith(".py"):
        return AMBIGUOUS
    if name not in POLICY_FILES:  # exact: POLICY.JSON is not policy.json, on any file system
        return f"unexpected entry: {safe(name)!r}"
    return None


# --- loading ----------------------------------------------------------------------------------


def load_policy_dir(path: Path, *, challenge: Challenge) -> LoadedPolicy:
    """The policy directory ``path`` for ``challenge``, fully validated and smoke-tested (§3.5);
    any failure is one ``LoadError``."""
    _check_challenge(challenge)
    return load_policy_files(read_policy_dir(path), challenge=challenge)


def load_policy_files(files: Mapping[str, bytes], *, challenge: Challenge) -> LoadedPolicy:
    """``load_policy_dir`` on files already read by ``read_policy_dir`` (the loader hashes those
    same bytes for the submission sha256, so what is hashed is what runs)."""
    _check_challenge(challenge)
    manifest = read_manifest(files[MANIFEST_FILE])
    data = files[ONNX_FILE]
    if hashlib.sha256(data).hexdigest() != manifest.onnx.sha256:
        raise LoadError(
            f"{MANIFEST_FILE}: policy_sha256_mismatch: onnx.sha256 is not the sha256 of {ONNX_FILE}"
        )
    return _load(data, challenge, manifest)


def policy_from_bytes(
    data: bytes, *, challenge: Challenge, expected_sha256: str | None = None
) -> LoadedPolicy:
    """Policy bytes without a manifest (a pool worker's ``PolicyRef``, a selection eval's
    in-memory export) for ``challenge``: the sha256 check when ``expected_sha256`` is given,
    then the same checks as ``load_policy_dir`` with the challenge standing in for the
    manifest (metadata against the challenge, ``topple.action_rule`` against the structure)."""
    _check_challenge(challenge)
    if type(data) is not bytes:
        raise LoadError(f"{ONNX_FILE}: onnx_parse_failed: policy.onnx must be bytes")
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise LoadError(
            f"{ONNX_FILE}: policy_sha256_mismatch: the bytes are not the expected policy"
        )
    return _load(data, challenge, None)


def _check_challenge(challenge: object) -> None:
    from topple.challenge import Challenge

    if not isinstance(challenge, Challenge):
        raise TypeError("a policy is loaded for a Challenge (its dims and bounds come from it)")


def _load(data: bytes, challenge: Challenge, manifest: PolicyManifest | None) -> LoadedPolicy:
    try:
        return _checked(data, challenge, manifest)
    except LoadError:
        raise
    except Exception:  # never a traceback or raw exception text (§3.4 "Error text")
        raise LoadError(f"{ONNX_FILE}: policy_load_failed: raised while loading") from None


def _checked(data: bytes, challenge: Challenge, manifest: PolicyManifest | None) -> LoadedPolicy:
    task = policy_task(challenge)
    result = validate_policy_bytes(data, task.obs_dim, task.action_dim)
    if isinstance(result, list):
        raise _error(ONNX_FILE, result)
    proto: onnx.ModelProto = onnx.ModelProto.FromString(result.canonical)
    rule = structure_match(proto)
    if manifest is not None:
        _reject(MANIFEST_FILE, check_against_model(manifest, proto))
        if manifest.action_rule != rule:
            _reject(
                MANIFEST_FILE,
                [
                    f"action_rule_mismatch: policy.json says {manifest.action_rule!r}; "
                    f"the graph is {rule!r}"
                ],
            )
        expected = manifest.model_dump(mode="json")
        source = MANIFEST_FILE
    else:
        expected = task_fields(task, rule)
        source = task.challenge
    mismatched = metadata_mismatches(proto, expected)
    _reject(ONNX_FILE, [f"metadata_mismatch: {key} differs from {source}" for key in mismatched])
    if manifest is not None:
        _reject(MANIFEST_FILE, check_against_challenge(manifest, challenge))
    _reject(ONNX_FILE, smoke_test(result))
    return LoadedPolicy(manifest=manifest, validated=result, session=make_session(result))


def _reject(file: str, messages: list[str]) -> None:
    """Raise ``_error(file, messages)`` if there are any messages."""
    if messages:
        raise _error(file, messages)


def _error(file: str, messages: list[str]) -> LoadError:
    """One ``LoadError`` naming ``file`` and every message (fixed codes and ``safe`` text)."""
    return LoadError(f"{file}: " + "; ".join(messages))
