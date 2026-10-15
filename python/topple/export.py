"""``topple export``: a training checkpoint as a ``topple.policy/1`` directory, without torch
or pickle (docs/design/phase-2.md §3.3).

A run directory is untrusted (it may come from someone else's machine), so export reads only
these files, each a regular single-link file read once (``read_regular_file``): ``best.json``
(without ``--step``) and ``train_meta.json`` in the run, and ``meta.json``, ``normalize.json``,
``actor.npz`` and ``parity_ref.npz`` in ``ckpt/step_<N>/``. JSON (at most 64 KiB) is parsed
strictly (no BOM, NaN, Infinity, duplicate keys, integers over 20 digits or nesting beyond 32);
``.npz`` (at most 8 MiB) through ``topple._npz``: the canonical zip layout, exactly the expected
members, dtypes and shapes, every header checked before any data is read. ``model.zip`` and
``vecnormalize.pkl`` are never opened.

1. **Resolve.** The step is ``--step`` or ``best.json``'s integer ``step``; ``ckpt/step_<N>`` must
   be a real directory under the run. Without ``--report-only`` the step's ``selection_history``
   row must be eligible, the run may carry no abort flag and no ``nondeterministic``, and the
   output must not lie under ``runs/`` (the working directory's ``runs/``, or the ``runs/`` the run
   itself lives in) or inside the run; with it the output is the run's ``report_policy/``,
   replaced if present.
2. **Read** (above), and bind the files to each other: ``train_meta.config`` validates as a
   ``TrainConfig`` whose ``config_sha256`` is the one ``train_meta.json`` and ``meta.json``
   record; ``meta.json`` agrees with it (algo, activation, actor widths) and with the step;
   ``normalize.json`` agrees with its ``normalize`` block.
3. **Build and bind.** ``build_onnx`` (§3.2) from ``actor.npz`` and ``normalize.json``; its
   sha256 must be ``meta.json``'s and ``parity_ref.npz``'s ``onnx_sha256`` (the bytes the
   selection eval scored) and the history row's. **Constant binding:** each initializer, by its
   §3.2 position, must equal bit for bit the value recomputed here from the inputs (a builder
   that used the wrong epsilon passes parity, but not this). Then ``validate_policy_bytes`` and
   ``smoke_test``.
4. **Parity.** Every ``parity_ref`` row at batch 1 through ``make_session``: ``max |a_onnx -
   a_ref| <= 1e-5`` always. The rows must be the §2.4 sets in order (on-policy rows, then exactly
   ``random_rows`` and ``edge_rows`` of ``normalize.json``). Coverage: at least 300 rows with
   ``max_i |a_ref_i| < 0.99``, at least 32 with ``>= 0.999``, ``n_onpolicy == 5 *
   n_onpolicy_seeds`` and ``1 <= n_onpolicy_seeds <=`` the run's number of selection seeds; with
   ``--report-only`` a miss marks the parity ``vacuous`` instead.
5. **Write.** ``policy.onnx`` and ``policy.json`` (claims from ``meta.json`` and
   ``train_meta.json`` through a whitelist), checked by loading them exactly as a submission is
   loaded, into a new directory; ``--replace`` (or ``--report-only``) swaps it in for an
   existing policy directory. ``--test-fixture NAME`` also writes ``tests/data/NAME_parity.npz``
   (a copy), ``.bin`` (layout v1) and ``.json`` for the C and JS parity readers: written to temp
   names before ``--out`` is swapped and renamed over their targets after it, so a failure to
   write them changes nothing. Removing the old policy directory after the swap is cleanup: a
   failure there is an ``ExportWarning``, not a failed export.

Training writes the snapshot files with ``snapshot_files`` and ``write_snapshot`` below (no
torch), so the format has one writer and one reader, tested together.

Every refusal is one ``ExportError`` line of fixed words, safe names and escaped, bounded text.
"""

from __future__ import annotations

import errno
import hashlib
import io
import itertools
import json
import math
import os
import re
import secrets
import shutil
import stat
import struct
import tempfile
import warnings
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, NoReturn

import numpy as np
import numpy.typing as npt
import onnx
from onnx import TensorProto
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from topple._atomic import (
    DIRECTORY_MODE,
    SymlinkedOutputError,
    check_output_directory,
    new_directory,
    replacement,
    write_new_file,
)
from topple._inputs import read_regular_file
from topple._npz import read_npz_bytes
from topple.loader import LoadError
from topple.policy.build import Activation, MlpWeights, ObsNorm, build_policy_onnx
from topple.policy.manifest import (
    GIT_COMMIT,
    HEX64,
    PRINTABLE,
    VERSION_KEY,
    Claims,
    ExportParity,
    Hex64,
    OnnxInfo,
    PolicyManifest,
    PolicyTask,
    TensorSig,
    dump_manifest,
    policy_task,
    task_fields,
)
from topple.policy.runtime import POLICY_FILES, load_policy_files, make_session
from topple.policy.validate import (
    BATCH,
    INPUT_NAME,
    OUTPUT_NAME,
    POLICY_FORMAT,
    ValidatedPolicy,
    safe,
    smoke_test,
    validate_policy_bytes,
)
from topple.pool import ONPOLICY_ROWS
from topple.rng import MAX_SEED
from topple.train.config import MAX_WIDTH, PpoTrainConfig, SacTrainConfig, config_sha256
from topple.train.config import validate_train_config as _validate_config
from topple.train.selection import (
    N_RANDOM_ROWS,
    NOT_SHIPPABLE,
    SET_EDGE,
    SET_ONPOLICY,
    SET_RANDOM,
    edge_row_count,
    edge_rows,
    onpolicy_seed_count,
    parity_obs,
    random_rows,
)
from topple.yamlsafe import ConfigError

if TYPE_CHECKING:
    from topple.challenge import Challenge

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

MAX_JSON_BYTES: Final = 64 * 1024
MAX_NPZ_BYTES: Final = 8 * 1024 * 1024
MAX_JSON_DEPTH: Final = 32
MAX_INT_DIGITS: Final = 20
PARITY_TOLERANCE: Final = 1e-5
MIN_UNSATURATED: Final = 300
MIN_SATURATED: Final = 32
UNSATURATED_BELOW: Final = 0.99
SATURATED_FROM: Final = 0.999
MAX_ONPOLICY_SEEDS: Final = 10_000
PARITY_REFERENCE: Final = "sb3.predict"
REPORT_DIR: Final = "report_policy"
RUNS_DIR: Final = "runs"
FIXTURE_DIR: Final = Path("tests/data")
FIXTURE_NAME: Final = re.compile(r"[a-z0-9_]{1,32}")
BIN_MAGIC: Final = b"TPAR"
BIN_VERSION: Final = 1
BIN_HEADER: Final = struct.Struct("<4sIIII")
RULE_OF: Final[Mapping[str, Literal["clip(mean)", "tanh(mean)"]]] = {
    "tanh": "clip(mean)",
    "relu": "tanh(mean)",
}
ACTOR_KEYS: Final = ("W0", "b0", "W1", "b1", "W2", "b2")
CONSTANT_NAMES: Final = (
    "mean",
    "inv_std",
    "obs_lo",
    "obs_hi",
    *ACTOR_KEYS,
    "act_lo",
    "act_hi",
)
"""The §3.2 initializers in position order (PPO and SAC alike)."""
_BOM: Final = b"\xef\xbb\xbf"


class ExportError(Exception):
    """A run directory that cannot be exported, or an output that cannot be written: one line
    of fixed words, safe names and escaped, bounded text (``topple export`` exits 2 with it)."""


class ExportWarning(UserWarning):
    """The export happened, but cleanup after it did not (``topple export`` prints it)."""


def _fail(message: str) -> NoReturn:
    raise ExportError(message)


# --- strict JSON --------------------------------------------------------------------------------


class _RejectedError(Exception):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise _RejectedError("repeats a key in an object")
    return dict(pairs)


def _no_constant(token: str) -> NoReturn:
    raise _RejectedError("holds NaN or Infinity")


def _finite_float(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):
        raise _RejectedError("holds a number beyond float range")
    return value


def _bounded_int(literal: str) -> int:
    if len(literal.lstrip("-")) > MAX_INT_DIGITS:
        raise _RejectedError(f"holds an integer of more than {MAX_INT_DIGITS} digits")
    return int(literal)


def _check_depth(data: bytes) -> None:
    """Nesting deeper than ``MAX_JSON_DEPTH`` refused in one pass over the bytes, before the
    standard library's scanner (which recurses in C) sees them."""
    depth = 0
    in_string = escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in b"[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise _RejectedError("nests too deeply")
        elif byte in b"]}":
            depth -= 1


def parse_run_json(data: bytes, name: str) -> object:
    """``data`` (a run file called ``name``) as strict JSON; ``ExportError`` otherwise."""
    if data.startswith(_BOM):
        _fail(f"{name}: starts with a byte order mark")
    try:
        _check_depth(data)
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_no_constant,
            parse_float=_finite_float,
            parse_int=_bounded_int,
        )
    except _RejectedError as exc:
        _fail(f"{name}: {exc}")
    except UnicodeDecodeError:
        _fail(f"{name}: is not valid UTF-8")
    except (ValueError, RecursionError):
        _fail(f"{name}: is not valid JSON")


def read_run_file(path: Path, cap: int) -> bytes:
    """``read_regular_file`` (no symlink, one link, at most ``cap`` bytes, never blocking), its
    ``LoadError`` as an ``ExportError``."""
    try:
        return read_regular_file(path, cap)
    except LoadError as exc:
        _fail(f"{path.parent.name}/{path.name}: {exc}")


def read_run_json(path: Path, cap: int = MAX_JSON_BYTES) -> object:
    """A run directory's JSON file, read and parsed strictly (also ``topple train --retry``'s
    reader of ``train_meta.json``)."""
    return parse_run_json(read_run_file(path, cap), path.name)


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name}: must be a JSON object")
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


StepInt = Annotated[int, Field(ge=0, lt=MAX_SEED)]


class SnapshotMeta(_Strict):
    """``ckpt/step_<N>/meta.json`` (§2.4): ``net_arch`` is the actor's two hidden widths."""

    step: StepInt
    algo: Literal["ppo", "sac"]
    action_rule: Literal["clip(mean)", "tanh(mean)"]
    net_arch: Annotated[
        list[Annotated[int, Field(ge=1, le=MAX_WIDTH)]], Field(min_length=2, max_length=2)
    ]
    activation: Literal["tanh", "relu"]
    config_sha256: Hex64
    onnx_sha256: Hex64
    n_onpolicy: Annotated[int, Field(ge=0, le=ONPOLICY_ROWS * MAX_ONPOLICY_SEEDS)]
    n_onpolicy_seeds: Annotated[int, Field(ge=0, le=MAX_ONPOLICY_SEEDS)]


class NormalizeFile(_Strict):
    """``ckpt/step_<N>/normalize.json``: SB3 ``VecNormalize``'s ``obs_rms`` (float64 written
    with ``repr``), ``epsilon`` and ``clip_obs``."""

    mean: Annotated[list[float], Field(min_length=1, max_length=4096)]
    var: Annotated[list[Annotated[float, Field(ge=0.0)]], Field(min_length=1, max_length=4096)]
    count: Annotated[float, Field(ge=0.0)]
    epsilon: Annotated[float, Field(gt=0.0)]
    clip_obs: Annotated[float, Field(gt=0.0)]


def _model[M: BaseModel](cls: type[M], value: object, name: str) -> M:
    try:
        return cls.model_validate(value, strict=True)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)[:8]
        where = "; ".join(
            f"{'.'.join(safe(str(p)) for p in e['loc']) or '(root)'} ({safe(e['type'])})"
            for e in errors
        )
        _fail(f"{name}: {where}")


# --- the build shared with training -------------------------------------------------------------


def build_onnx(
    layers: tuple[tuple[F32, F32], ...],
    norm: ObsNorm,
    *,
    activation: Activation,
    challenge: Challenge,
) -> bytes:
    """The canonical ONNX bytes of a checkpoint (§3.2): the actor's ``(W [out, in], b)``
    float32 layers behind ``norm``, with the head §3.2 pairs with ``activation`` and the
    challenge's 7 metadata values and action bounds. The selection eval scores these exact
    bytes (training builds them with this function) and export rebuilds them from the files."""
    task = policy_task(challenge)
    rule = RULE_OF[activation]
    metadata = {f"topple.{k}": json.dumps(v) for k, v in task_fields(task, rule).items()}
    return build_policy_onnx(
        MlpWeights(layers=layers, activation=activation),
        norm,
        action_rule=rule,
        metadata=metadata,
        action=task.action,
    )


def expected_constants(
    layers: tuple[tuple[F32, F32], ...], norm: ObsNorm, task: PolicyTask
) -> list[tuple[str, F32]]:
    """The §3.2 initializers by position, recomputed from the inputs: ``f32(mean)``,
    ``f32(1 / sqrt(var + eps))`` (float64, rounded once), ``-f32(clip_obs)``, ``f32(clip_obs)``,
    the actor's arrays as stored, and the challenge's ``f32(low)``, ``f32(high)``."""
    mean = np.asarray(norm.mean, dtype=np.float64)
    var = np.asarray(norm.var, dtype=np.float64)
    clip = np.float32(norm.clip)
    values: list[npt.ArrayLike] = [
        mean.astype(np.float32),
        (1.0 / np.sqrt(var + norm.epsilon)).astype(np.float32),
        -clip,
        clip,
        *(arr for pair in layers for arr in pair),
        np.float32(task.action.low),
        np.float32(task.action.high),
    ]
    # np.array, not np.ascontiguousarray: that would turn the rank-0 Clip bounds into rank 1
    return [
        (name, np.array(value, dtype="<f4", order="C"))
        for name, value in zip(CONSTANT_NAMES, values, strict=True)
    ]


def constant_mismatches(
    proto: onnx.ModelProto, layers: tuple[tuple[F32, F32], ...], norm: ObsNorm, task: PolicyTask
) -> list[str]:
    """The §3.2 positions whose initializer is not bit for bit ``expected_constants``."""
    initializers = list(proto.graph.initializer)
    expected = expected_constants(layers, norm, task)
    if len(initializers) != len(expected):
        return [f"{len(initializers)} initializers, not {len(expected)}"]
    problems = []
    for j, (tensor, (name, want)) in enumerate(zip(initializers, expected, strict=True)):
        same = (
            tensor.data_type == TensorProto.FLOAT
            and tuple(int(d) for d in tensor.dims) == want.shape
            and bytes(tensor.raw_data) == want.tobytes()
        )
        if not same:
            problems.append(f"c{j} ({name})")
    return problems


# --- the snapshot files, written by training ---------------------------------------------------

Schema = dict[str, tuple[str, tuple[int, ...]]]


def _actor_schema(h1: int, h2: int, d: int, a: int) -> Schema:
    """``actor.npz``: float32 ``[out, in]`` weights and biases, keys ``W0, b0, W1, b1, W2, b2``."""
    return {
        "W0": ("<f4", (h1, d)),
        "b0": ("<f4", (h1,)),
        "W1": ("<f4", (h2, h1)),
        "b1": ("<f4", (h2,)),
        "W2": ("<f4", (a, h2)),
        "b2": ("<f4", (a,)),
    }


def _parity_schema(n: int, d: int, a: int) -> Schema:
    """``parity_ref.npz``: ``obs`` f32 [N, D], ``action_ref`` f64 [N, A], ``set`` u1 [N],
    ``onnx_sha256`` a U64 scalar."""
    return {
        "obs": ("<f4", (n, d)),
        "action_ref": ("<f8", (n, a)),
        "set": ("|u1", (n,)),
        "onnx_sha256": ("<U64", ()),
    }


def _check_arrays(name: str, arrays: Mapping[str, npt.NDArray[Any]], schema: Schema) -> None:
    if set(arrays) != set(schema):
        raise ValueError(f"{name}: holds {sorted(arrays)}, not {sorted(schema)}")
    for key, (dtype, shape) in schema.items():
        arr = arrays[key]
        if arr.dtype.str != dtype or arr.shape != shape:
            raise ValueError(
                f"{name}: {key} must be {dtype} {shape}, not {arr.dtype.str} {arr.shape}"
            )


def _npz_bytes(arrays: Mapping[str, npt.NDArray[Any]]) -> bytes:
    """``np.savez``: numpy's canonical layout (the one ``topple._zip`` reads), members dated
    1980-01-01, so the same arrays give the same bytes on every run."""
    buf = io.BytesIO()
    np.savez(buf, **dict(arrays))  # type: ignore[arg-type]  # numpy's stub types **kwds poorly
    return buf.getvalue()


def _json_bytes(doc: object) -> bytes:
    """Floats as ``repr`` (float64 round-trips exactly), never NaN or infinity."""
    return (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode("ascii")


def snapshot_files(
    cfg: PpoTrainConfig | SacTrainConfig,
    step: int,
    layers: tuple[tuple[F32, F32], ...],
    norm: ObsNorm,
    *,
    count: float,
    onpolicy: F32,
    action_ref: F64,
    onnx_sha256: str,
) -> dict[str, bytes]:
    """The bytes of ``ckpt/step_<step>/``'s ``actor.npz``, ``normalize.json``,
    ``parity_ref.npz`` and ``meta.json`` (§2.4), exactly in the layout export reads: the actor's
    float32 layers, ``VecNormalize``'s ``obs_rms`` (``norm`` with its ``count``), the parity rows
    (``onpolicy`` (a), then ``random_rows`` and ``edge_rows`` of ``norm``, ``parity_obs``) with
    ``action_ref`` computed on them by the caller (SB3's ``predict``), and the metadata binding
    them to ``cfg`` and to ``onnx_sha256`` (of the ``build_onnx`` bytes the selection eval
    scored). Torch-free: training passes numpy arrays. ``ValueError`` for an array of the wrong
    dtype or shape (checked against the same schemas export reads with)."""
    from topple.challenge import load_challenge

    task = policy_task(load_challenge(cfg.env.challenge))
    d, a = task.obs_dim, task.action_dim
    h1, h2 = cfg.policy.net_arch.pi
    if len(layers) != len(ACTOR_KEYS) // 2:
        raise ValueError(f"actor.npz: {len(ACTOR_KEYS) // 2} layers, not {len(layers)}")
    actor = {
        f"{kind}{k}": np.asarray(arr)
        for k, (w, b) in enumerate(layers)
        for kind, arr in (("W", w), ("b", b))
    }
    _check_arrays("actor.npz", actor, _actor_schema(h1, h2, d, a))
    if not (isinstance(onnx_sha256, str) and HEX64.fullmatch(onnx_sha256)):
        raise ValueError("onnx_sha256 must be 64 lowercase hex digits")
    n_seeds = onpolicy_seed_count(onpolicy)
    obs, sets = parity_obs(onpolicy, norm)
    parity: dict[str, npt.NDArray[Any]] = {
        "obs": obs,
        "action_ref": np.asarray(action_ref),
        "set": sets,
        "onnx_sha256": np.asarray(np.str_(onnx_sha256)),
    }
    _check_arrays("parity_ref.npz", parity, _parity_schema(obs.shape[0], d, a))
    normalize = NormalizeFile(
        mean=[float(x) for x in np.asarray(norm.mean, dtype=np.float64)],
        var=[float(x) for x in np.asarray(norm.var, dtype=np.float64)],
        count=float(count),
        epsilon=float(norm.epsilon),
        clip_obs=float(norm.clip),
    )
    if len(normalize.mean) != d:
        raise ValueError(f"normalize.json: mean and var must have {d} entries ({task.challenge})")
    if (normalize.epsilon, normalize.clip_obs) != (cfg.normalize.epsilon, cfg.normalize.clip_obs):
        raise ValueError("normalize.json: epsilon or clip_obs differ from the run's config")
    meta = SnapshotMeta(
        step=step,
        algo=cfg.algo,
        action_rule=RULE_OF[cfg.policy.activation],
        net_arch=[h1, h2],
        activation=cfg.policy.activation,
        config_sha256=config_sha256(cfg),
        onnx_sha256=onnx_sha256,
        n_onpolicy=int(onpolicy.shape[0]),
        n_onpolicy_seeds=n_seeds,
    )
    return {
        "actor.npz": _npz_bytes(actor),
        "normalize.json": _json_bytes(normalize.model_dump(mode="json")),
        "parity_ref.npz": _npz_bytes(parity),
        "meta.json": _json_bytes(meta.model_dump(mode="json")),
    }


def write_snapshot(
    ckpt_dir: Path,
    step: int,
    files: Mapping[str, bytes],
    *,
    more: Callable[[Path], None] | None = None,
) -> Path:
    """Write the immutable ``ckpt_dir/step_<step>/`` (§2.4) and return it: ``files``
    (``snapshot_files``' and any other, such as PPO's ``critic.npz``), then ``more(directory)``
    (SB3's ``model.zip`` and ``vecnormalize.pkl``, resume only) into ``.step_<step>.tmp/`` (a
    stale one from a crash is removed first), renamed into place. ``FileExistsError`` if
    ``step_<step>`` exists; a failure leaves no snapshot and no temp directory."""
    if type(step) is not int or step < 0:
        raise ValueError(f"step must be an int >= 0, got {step!r}")
    if not all(name and Path(name).name == name and not name.startswith(".") for name in files):
        raise ValueError("snapshot files are plain names")
    final = ckpt_dir / f"step_{step}"
    if os.path.lexists(final):
        raise FileExistsError(
            errno.EEXIST, "the snapshot exists (snapshots are immutable)", str(final)
        )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / f".step_{step}.tmp"
    if os.path.lexists(tmp):
        _remove(tmp)
    os.mkdir(tmp)
    try:
        for name, data in files.items():
            write_new_file(tmp / name, data)
        if more is not None:
            more(tmp)
        os.chmod(tmp, DIRECTORY_MODE)
        if os.path.lexists(final):
            raise FileExistsError(errno.EEXIST, "the snapshot exists", str(final))
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


# --- inputs -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Run:
    """What step 1 resolved: the step, its directory, its history row's eligibility, the
    training metadata and where the output goes."""

    step: int
    step_dir: Path
    eligible: bool
    history_sha256: str
    train_meta: dict[str, Any]
    replace: bool


@dataclass(frozen=True)
class _Snapshot:
    """What step 2 read and bound together."""

    cfg: PpoTrainConfig | SacTrainConfig
    challenge: Challenge
    task: PolicyTask
    meta: SnapshotMeta
    norm: ObsNorm
    layers: tuple[tuple[F32, F32], ...]
    obs: F32
    action_ref: F64
    sets: npt.NDArray[np.uint8]
    parity_sha256: str
    parity_bytes: bytes


def _best_step(run_dir: Path) -> int:
    best = _object(read_run_json(run_dir / "best.json"), "best.json")
    step = best.get("step")
    if type(step) is not int or not 0 <= step < MAX_SEED:
        _fail("best.json: step must be an integer in [0, 2**53)")
    return step


def _step_dir(run_dir: Path, step: int) -> Path:
    """``run_dir/ckpt/step_<step>``: a real directory (not a symlink) that resolves inside the
    run directory."""
    path = run_dir / "ckpt" / f"step_{step}"
    try:
        info = os.lstat(path)
        inside = path.resolve(strict=True).is_relative_to(run_dir.resolve(strict=True))
    except OSError as exc:
        _fail(f"ckpt/step_{step}: {exc.strerror or type(exc).__name__}")
    if stat.S_ISLNK(info.st_mode):
        _fail(f"ckpt/step_{step} is a symlink")
    if not stat.S_ISDIR(info.st_mode):
        _fail(f"ckpt/step_{step} is not a directory")
    if not inside:
        _fail(f"ckpt/step_{step} resolves outside the run directory")
    return path


def _flags(train_meta: dict[str, Any]) -> list[str]:
    flags = train_meta.get("flags")
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        _fail("train_meta.json: flags must be a list of strings")
    return flags


def _history_row(train_meta: dict[str, Any], step: int) -> dict[str, Any]:
    history = train_meta.get("selection_history")
    if not isinstance(history, list):
        _fail("train_meta.json: selection_history must be a list")
    rows = [r for r in history if isinstance(r, dict) and type(r.get("step")) is int]
    if len(rows) != len(history):
        _fail("train_meta.json: every selection_history row is an object with an integer step")
    matching = [r for r in rows if r["step"] == step]
    if len(matching) != 1:
        _fail(f"train_meta.json: selection_history has {len(matching)} rows for step {step}")
    (found,) = matching
    if type(found.get("eligible")) is not bool:
        _fail(f"train_meta.json: the step {step} row has no boolean eligible")
    if not isinstance(found.get("onnx_sha256"), str):
        _fail(f"train_meta.json: the step {step} row has no onnx_sha256")
    return found


def _check_output(out_dir: Path, run_dir: Path, *, report_only: bool) -> None:
    """Without ``--report-only``: not under ``runs/`` (the working directory's, the checkout's
    gitignored run folder, or the one the run lives in) nor inside the run; a directory called
    ``runs`` elsewhere, such as an ancestor of the checkout, does not count. With it: exactly
    the run's ``report_policy/``."""
    try:
        check_output_directory(out_dir)
        run_root = run_dir.resolve(strict=True)
        out_parent = out_dir.parent.resolve()
        runs_dirs = [(Path.cwd() / RUNS_DIR).resolve()]
    except SymlinkedOutputError as exc:
        _fail(str(exc))
    except OSError as exc:
        _fail(f"cannot resolve the output: {exc.strerror or type(exc).__name__}")
    if report_only:
        if out_parent != run_root or out_dir.name != REPORT_DIR:
            _fail(f"--report-only writes the run's {REPORT_DIR}/ only (--out RUN_DIR/{REPORT_DIR})")
        return
    if run_root.parent.name == RUNS_DIR:
        runs_dirs.append(run_root.parent)
    resolved = out_parent / out_dir.name
    if any(resolved.is_relative_to(runs) for runs in runs_dirs):
        _fail(f"--out must not be under {RUNS_DIR}/ (runs are gitignored scratch)")
    if resolved.is_relative_to(run_root):
        _fail("--out must not be inside the run directory")


def _resolve(
    run_dir: Path, out_dir: Path, step: int | None, *, report_only: bool, replace: bool
) -> _Run:
    if not run_dir.is_dir():
        _fail(f"{run_dir} is not a run directory")
    _check_output(out_dir, run_dir, report_only=report_only)
    if step is None:
        step = _best_step(run_dir)
    elif type(step) is not int or not 0 <= step < MAX_SEED:
        _fail("--step must be an integer in [0, 2**53)")
    train_meta = _object(read_run_json(run_dir / "train_meta.json"), "train_meta.json")
    step_dir = _step_dir(run_dir, step)
    flags = _flags(train_meta)
    row = _history_row(train_meta, step)
    if not report_only:
        if not row["eligible"]:
            _fail(f"step {step} is not eligible (use --report-only for a report row)")
        blocked = sorted(set(flags) & NOT_SHIPPABLE)
        if blocked:
            _fail(f"the run has flags {blocked}: never shipped (use --report-only)")
    return _Run(
        step, step_dir, row["eligible"], row["onnx_sha256"], train_meta, replace or report_only
    )


def _config(train_meta: dict[str, Any]) -> PpoTrainConfig | SacTrainConfig:
    try:
        cfg = _validate_config(train_meta.get("config"))
    except ConfigError as exc:
        _fail(f"train_meta.json: config: {exc}")
    recorded = train_meta.get("config_sha256")
    if recorded != config_sha256(cfg):
        _fail("train_meta.json: config_sha256 is not the sha256 of its config")
    return cfg


def _read_snapshot(run: _Run) -> _Snapshot:
    cfg = _config(run.train_meta)
    meta = _model(SnapshotMeta, read_run_json(run.step_dir / "meta.json"), "meta.json")
    raw_norm = _model(
        NormalizeFile, read_run_json(run.step_dir / "normalize.json"), "normalize.json"
    )
    if meta.step != run.step:
        _fail(f"meta.json: step {meta.step} is not the step directory's {run.step}")
    if meta.config_sha256 != config_sha256(cfg):
        _fail("meta.json: config_sha256 is not train_meta.json's")
    pi = list(cfg.policy.net_arch.pi)
    if (meta.algo, meta.activation, meta.net_arch) != (cfg.algo, cfg.policy.activation, pi):
        _fail("meta.json: algo, activation or net_arch differ from the run's config")
    if meta.action_rule != RULE_OF[meta.activation]:
        _fail(f"meta.json: a {meta.activation} actor has action_rule {RULE_OF[meta.activation]}")
    if (raw_norm.epsilon, raw_norm.clip_obs) != (cfg.normalize.epsilon, cfg.normalize.clip_obs):
        _fail("normalize.json: epsilon or clip_obs differ from the run's config")
    from topple.challenge import load_challenge

    challenge = load_challenge(cfg.env.challenge)
    task = policy_task(challenge)
    d, a = task.obs_dim, task.action_dim
    if len(raw_norm.mean) != d or len(raw_norm.var) != d:
        _fail(f"normalize.json: mean and var must have {d} entries ({task.challenge})")
    norm = ObsNorm(
        mean=np.asarray(raw_norm.mean, dtype=np.float64),
        var=np.asarray(raw_norm.var, dtype=np.float64),
        epsilon=raw_norm.epsilon,
        clip=raw_norm.clip_obs,
    )
    h1, h2 = meta.net_arch
    actor = _npz(run.step_dir / "actor.npz", _actor_schema(h1, h2, d, a))[0]
    layers = tuple((actor[f"W{k}"], actor[f"b{k}"]) for k in range(3))
    n = meta.n_onpolicy + N_RANDOM_ROWS + edge_row_count(d)
    parity, parity_bytes = _npz(run.step_dir / "parity_ref.npz", _parity_schema(n, d, a))
    parity_sha = str(parity["onnx_sha256"][()])
    if not HEX64.fullmatch(parity_sha):
        _fail("parity_ref.npz: onnx_sha256 is not a sha256")
    return _Snapshot(
        cfg=cfg,
        challenge=challenge,
        task=task,
        meta=meta,
        norm=norm,
        layers=layers,
        obs=parity["obs"],
        action_ref=parity["action_ref"],
        sets=parity["set"],
        parity_sha256=parity_sha,
        parity_bytes=parity_bytes,
    )


def _npz(
    path: Path, schema: Mapping[str, tuple[str, tuple[int, ...]]]
) -> tuple[dict[str, npt.NDArray[Any]], bytes]:
    data = read_run_file(path, MAX_NPZ_BYTES)
    try:
        return read_npz_bytes(data, schema, max_bytes=MAX_NPZ_BYTES), data
    except ValueError as exc:
        _fail(f"{path.name}: {exc}")
    except Exception as exc:  # struct.error, zlib.error, ...: never a traceback
        _fail(f"{path.name}: not a readable npz file ({type(exc).__name__})")


# --- build, bind, parity ------------------------------------------------------------------------


def _bind(snap: _Snapshot, history_sha: str) -> tuple[bytes, ValidatedPolicy]:
    """Step 3: build, the sha256 and constant bindings, validation and the smoke test."""
    try:
        data = build_onnx(
            snap.layers, snap.norm, activation=snap.meta.activation, challenge=snap.challenge
        )
    except ValueError as exc:  # non-finite weights, a mean beyond float32, ...
        _fail(f"cannot build the policy: {exc}")
    sha = hashlib.sha256(data).hexdigest()
    for name, recorded in (
        ("meta.json", snap.meta.onnx_sha256),
        ("parity_ref.npz", snap.parity_sha256),
        ("train_meta.json selection_history", history_sha),
    ):
        if recorded != sha:
            _fail(f"onnx_sha256_mismatch: {name} does not record the sha256 of the rebuilt policy")
    proto: onnx.ModelProto = onnx.ModelProto.FromString(data)
    mismatched = constant_mismatches(proto, snap.layers, snap.norm, snap.task)
    if mismatched:
        _fail(f"constant_binding: initializers {', '.join(mismatched)} differ from the inputs")
    result = validate_policy_bytes(data, snap.task.obs_dim, snap.task.action_dim)
    if isinstance(result, list):
        _fail("policy.onnx: " + "; ".join(result))
    errors = smoke_test(result)
    if errors:
        _fail("policy.onnx: " + "; ".join(errors))
    return data, result


@dataclass(frozen=True)
class ParityReport:
    """Step 4's numbers (``claims.export_parity``) and the run's number of selection seeds,
    which bounds ``n_onpolicy_seeds``."""

    n: int
    n_onpolicy: int
    n_onpolicy_seeds: int
    n_unsaturated: int
    n_saturated: int
    max_abs_diff: float
    n_selection_seeds: int

    def claim(self, *, vacuous: bool) -> ExportParity:
        return ExportParity(
            reference=PARITY_REFERENCE,
            n=self.n,
            n_onpolicy=self.n_onpolicy,
            n_onpolicy_seeds=self.n_onpolicy_seeds,
            n_unsaturated=self.n_unsaturated,
            n_saturated=self.n_saturated,
            max_abs_diff=self.max_abs_diff,
            tolerance=PARITY_TOLERANCE,
            vacuous=vacuous,
        )


def _check_rows(snap: _Snapshot) -> None:
    """The rows are the §2.4 sets in order: on-policy, ``random_rows``, ``edge_rows``."""
    n_on = snap.meta.n_onpolicy
    expected_sets = np.concatenate(
        [
            np.full(n_on, SET_ONPOLICY, np.uint8),
            np.full(N_RANDOM_ROWS, SET_RANDOM, np.uint8),
            np.full(edge_row_count(snap.task.obs_dim), SET_EDGE, np.uint8),
        ]
    )
    if not np.array_equal(snap.sets, expected_sets):
        _fail("parity_ref.npz: set must label the on-policy, random and edge rows, in order")
    if not (np.isfinite(snap.obs).all() and np.isfinite(snap.action_ref).all()):
        _fail("parity_ref.npz: obs and action_ref must be finite")
    regenerated = np.concatenate([random_rows(snap.norm), edge_rows(snap.norm)])
    if not np.array_equal(snap.obs[n_on:], regenerated):
        _fail("parity_ref.npz: the random and edge rows are not those of normalize.json")


def _parity(snap: _Snapshot, validated: ValidatedPolicy) -> ParityReport:
    """Step 4: every row at batch 1 through ``make_session``, and the coverage counts."""
    _check_rows(snap)
    session = make_session(validated)
    n, a = snap.obs.shape[0], snap.task.action_dim
    worst = 0.0
    for i in range(n):
        out = session.run(None, {INPUT_NAME: snap.obs[i : i + 1]})[0]
        if not (isinstance(out, np.ndarray) and out.dtype == np.float32 and out.shape == (1, a)):
            _fail("policy.onnx: parity row output is not float32 (1, action_dim)")
        diff = float(np.max(np.abs(out[0].astype(np.float64) - snap.action_ref[i])))
        if math.isnan(diff) or diff > worst:  # a NaN sticks (nothing is > NaN) and fails below
            worst = diff
    if not worst <= PARITY_TOLERANCE:
        _fail(f"parity: max |a_onnx - a_ref| is {worst:.3g}, above {PARITY_TOLERANCE:g}")
    unsaturated, saturated = coverage_counts(snap.action_ref)
    return ParityReport(
        n,
        snap.meta.n_onpolicy,
        snap.meta.n_onpolicy_seeds,
        unsaturated,
        saturated,
        worst,
        len(snap.cfg.selection.seeds.seeds),
    )


def coverage_counts(action_ref: F64) -> tuple[int, int]:
    """``(n_unsaturated, n_saturated)``: the rows with ``max_i |a_ref_i| < 0.99`` and those with
    ``max_i |a_ref_i| >= 0.999`` (§3.3 step 4)."""
    amax = np.abs(action_ref).max(axis=1)
    return int((amax < UNSATURATED_BELOW).sum()), int((amax >= SATURATED_FROM).sum())


def _coverage_misses(report: ParityReport) -> list[str]:
    """Step 4's coverage rules that ``report`` misses. The on-policy rows (a) are bound to the
    run: 5 per selection episode of T >= 1, so at least one episode and at most the run's
    selection seeds (a snapshot writer that dropped or duplicated them would pass otherwise)."""
    misses = []
    if report.n_unsaturated < MIN_UNSATURATED:
        misses.append(f"{report.n_unsaturated} unsaturated rows (< {MIN_UNSATURATED})")
    if report.n_saturated < MIN_SATURATED:
        misses.append(f"{report.n_saturated} saturated rows (< {MIN_SATURATED})")
    if report.n_onpolicy != ONPOLICY_ROWS * report.n_onpolicy_seeds:
        misses.append(
            f"n_onpolicy {report.n_onpolicy} is not {ONPOLICY_ROWS} x n_onpolicy_seeds "
            f"{report.n_onpolicy_seeds}"
        )
    if report.n_onpolicy_seeds < 1:
        misses.append("no on-policy rows (n_onpolicy_seeds 0)")
    if report.n_onpolicy_seeds > report.n_selection_seeds:
        misses.append(
            f"n_onpolicy_seeds {report.n_onpolicy_seeds} is more than the "
            f"{report.n_selection_seeds} selection seeds"
        )
    return misses


# --- the manifest -------------------------------------------------------------------------------


def _versions(value: object) -> dict[str, str] | None:
    """``train_meta.versions`` through the claims' rules: string values only, keys and values
    that ``Claims`` accepts, at most 16 (sorted by key)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail("train_meta.json: versions must be an object")
    kept = {
        k: v
        for k, v in sorted(value.items())
        if isinstance(v, str) and VERSION_KEY.fullmatch(k) and PRINTABLE.fullmatch(v)
    }
    return dict(itertools.islice(kept.items(), 16))


def _optional(meta: dict[str, Any], key: str, kind: type) -> Any:
    value = meta.get(key)
    if value is not None and type(value) is not kind:
        _fail(f"train_meta.json: {key} must be a {kind.__name__} or null")
    return value


def _claims(run: _Run, snap: _Snapshot, parity: ExportParity) -> Claims:
    """The whitelist (§3.3 step 5): the actor's algo, activation and widths from
    ``meta.json``; budget, seed and config hash from the config; git and versions from
    ``train_meta.json``; the step, its eligibility and the parity self-check from here."""
    commit = _optional(run.train_meta, "git_commit", str)
    if commit is not None and not GIT_COMMIT.fullmatch(commit):
        _fail("train_meta.json: git_commit must be 40 lowercase hex digits or null")
    return Claims(
        name=None,
        algo=snap.meta.algo,
        activation=snap.meta.activation,
        net_arch=tuple(snap.meta.net_arch),
        total_timesteps=snap.cfg.total_timesteps,
        selected_step=run.step,
        train_seed=snap.cfg.seed,
        train_config_sha256=snap.meta.config_sha256,
        git_commit=commit,
        git_dirty=_optional(run.train_meta, "git_dirty", bool),
        selection_eligible=run.eligible,
        versions=_versions(run.train_meta.get("versions")),
        export_parity=parity,
    )


def _manifest(data: bytes, snap: _Snapshot, claims: Claims) -> PolicyManifest:
    proto: onnx.ModelProto = onnx.ModelProto.FromString(data)
    task = snap.task
    return PolicyManifest(
        format=POLICY_FORMAT,
        kind="onnx",
        challenge=task.challenge,
        obs_schema=task.obs_schema,
        obs_features=task.obs_features,
        action_spec=task.action.id,
        action_dim=task.action_dim,
        control_dt=task.control_dt,
        action_rule=snap.meta.action_rule,
        onnx=OnnxInfo(
            file="policy.onnx",
            sha256=hashlib.sha256(data).hexdigest(),
            ir_version=proto.ir_version,
            opset=proto.opset_import[0].version,
            input=TensorSig(name=INPUT_NAME, dtype="float32", shape=(BATCH, task.obs_dim)),
            output=TensorSig(name=OUTPUT_NAME, dtype="float32", shape=(BATCH, task.action_dim)),
        ),
        claims=claims,
    )


# --- writing ------------------------------------------------------------------------------------


def _replaceable(out: Path) -> None:
    """``--replace`` swaps out a symlink (the link, never its target) or a policy directory
    (only ``policy.onnx``, ``policy.json``, ``README.md``), never anything else."""
    try:
        info = os.lstat(out)
        if stat.S_ISLNK(info.st_mode):
            return
        if not stat.S_ISDIR(info.st_mode):
            _fail(f"{out} is not a directory: --replace replaces a policy directory only")
        with os.scandir(out) as listing:
            entries = list(itertools.islice(listing, len(POLICY_FILES) + 1))
        ok = all(e.name in POLICY_FILES and not e.is_dir(follow_symlinks=False) for e in entries)
    except OSError as exc:
        _fail(f"cannot examine {out}: {exc.strerror or type(exc).__name__}")
    if not ok:
        _fail(f"{out} holds more than a policy: --replace replaces a policy directory only")


def _remove(path: Path) -> None:
    if os.path.islink(path) or not os.path.isdir(path):
        os.unlink(path)
    else:
        shutil.rmtree(path)


def write_policy_dir(out: Path, files: Mapping[str, bytes], *, replace: bool) -> None:
    """Write ``files`` as the directory ``out``, complete or not at all. Without ``replace``
    ``out`` must not exist (``_atomic.new_directory``). With it the new directory is written as
    a temp sibling, the old one (``_replaceable``) renamed aside, the new one renamed in and the
    old one removed; a failure up to the swap puts the old one back. Once the new one is in,
    failing to remove the old one is an ``ExportWarning`` (the hidden ``.<name>.<hex>.old`` is
    left), not an error."""
    if not replace:
        try:
            with new_directory(out) as tmp:
                for name, data in files.items():
                    write_new_file(tmp / name, data)
        except FileExistsError:
            _fail(f"{out} already exists (pass --replace to replace it)")
        return
    if os.path.lexists(out):
        _replaceable(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out.parent, prefix=f".{out.name}.", suffix=".partial"))
    try:
        for name, data in files.items():
            write_new_file(tmp / name, data)
        os.chmod(tmp, DIRECTORY_MODE)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    aside: Path | None = None
    try:
        if os.path.lexists(out):
            aside = out.parent / f".{out.name}.{secrets.token_hex(8)}.old"
            os.rename(out, aside)
        os.rename(tmp, out)
    except BaseException:
        if aside is not None and not os.path.lexists(out):
            os.rename(aside, out)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if aside is not None:
        try:
            _remove(aside)
        except OSError as exc:
            warnings.warn(
                ExportWarning(
                    f"replaced {out}, but cannot remove the old one at {aside.name}: "
                    f"{exc.strerror or type(exc).__name__}"
                ),
                stacklevel=2,
            )


def parity_bin(obs: F32, action_ref: F64, sets: npt.NDArray[np.uint8]) -> bytes:
    """``.bin`` layout v1, little-endian: ``b"TPAR"``, u32 version 1, u32 N, u32 D, u32 A,
    f32 obs[N*D], f32 a_ref[N*A], u8 set[N], zero-padded to a multiple of 4 bytes."""
    n, d = obs.shape
    a = action_ref.shape[1]
    body = b"".join(
        (
            BIN_HEADER.pack(BIN_MAGIC, BIN_VERSION, n, d, a),
            np.ascontiguousarray(obs, dtype="<f4").tobytes(),
            np.ascontiguousarray(action_ref, dtype="<f4").tobytes(),
            np.ascontiguousarray(sets, dtype="u1").tobytes(),
        )
    )
    return body + b"\0" * (-len(body) % 4)


def _fixture_paths(fixture_dir: Path, name: str) -> dict[str, Path]:
    return {ext: fixture_dir / f"{name}_parity.{ext}" for ext in ("npz", "bin", "json")}


def _check_fixture_targets(paths: Mapping[str, Path], *, replace: bool) -> None:
    """Before anything is built or written: an existing fixture needs ``--replace``, and a
    target a file cannot be renamed over (a directory) is refused."""
    existing = [p.name for p in paths.values() if os.path.lexists(p)]
    if existing and not replace:
        _fail(f"fixture files {existing} exist (pass --replace to replace them)")
    for path in paths.values():
        if os.path.isdir(path) and not os.path.islink(path):
            _fail(f"{path} is a directory: --test-fixture replaces files only")


def _fixture_contents(snap: _Snapshot, report: ParityReport, sha: str) -> dict[str, bytes]:
    sidecar = {
        "onnx_sha256": snap.parity_sha256,
        "n": report.n,
        "d": snap.task.obs_dim,
        "a": snap.task.action_dim,
        "tolerance": PARITY_TOLERANCE,
        "n_unsaturated": report.n_unsaturated,
        "n_saturated": report.n_saturated,
        "policy_sha256": sha,
    }
    return {
        "npz": snap.parity_bytes,
        "bin": parity_bin(snap.obs, snap.action_ref, snap.sets),
        "json": (json.dumps(sidecar, indent=2) + "\n").encode("ascii"),
    }


def _write_outputs(
    out_dir: Path,
    files: Mapping[str, bytes],
    *,
    replace: bool,
    fixtures: Mapping[str, Path],
    contents: Mapping[str, bytes],
) -> None:
    """The policy directory and the fixture files: each fixture is written to a temp file beside
    its target first (``replacement``), then the policy directory is written and swapped in,
    then the temp files are renamed over their targets (checked up front, so the renames cannot
    meet a directory). A failure before the swap leaves ``--out`` and every fixture as they
    were; errors name the target, never a temp file."""
    swapped = False
    try:
        with ExitStack() as staged:
            for ext, path in fixtures.items():
                try:
                    tmp = staged.enter_context(replacement(path, suffix=".tmp"))
                    tmp.write_bytes(contents[ext])
                except OSError as exc:
                    _fail(f"cannot write {path}: {exc.strerror or type(exc).__name__}")
            try:
                write_policy_dir(out_dir, files, replace=replace)
            except OSError as exc:
                _fail(f"cannot write {out_dir}: {exc.strerror or type(exc).__name__}")
            swapped = True
    except OSError as exc:  # a rename of a staged fixture, after the swap
        if not swapped:
            raise
        names = ", ".join(path.name for path in fixtures.values())
        _fail(
            f"wrote {out_dir}, but cannot finish the fixture files {names}: "
            f"{exc.strerror or type(exc).__name__}"
        )


# --- the command --------------------------------------------------------------------------------


def export_policy(
    run_dir: Path,
    out_dir: Path,
    *,
    step: int | None = None,
    report_only: bool = False,
    replace: bool = False,
    test_fixture: str | None = None,
    fixture_dir: Path = FIXTURE_DIR,
) -> PolicyManifest:
    """Export ``run_dir``'s checkpoint ``step`` (default: ``best.json``'s) to the policy
    directory ``out_dir`` (§3.3 steps 1-5) and return its manifest; ``ExportError`` for every
    refusal. ``test_fixture`` names the parity fixture files written to ``fixture_dir``."""
    fixtures: dict[str, Path] = {}
    if test_fixture is not None:
        if not isinstance(test_fixture, str) or not FIXTURE_NAME.fullmatch(test_fixture):
            _fail("--test-fixture must match [a-z0-9_]{1,32}")
        if report_only:
            _fail("--test-fixture needs a shippable export, not --report-only")
        try:
            check_output_directory(fixture_dir)
        except OSError as exc:
            _fail(str(exc) if isinstance(exc, SymlinkedOutputError) else f"{fixture_dir}: {exc}")
        fixtures = _fixture_paths(fixture_dir, test_fixture)
        _check_fixture_targets(fixtures, replace=replace)
    run = _resolve(run_dir, out_dir, step, report_only=report_only, replace=replace)
    if not run.replace and os.path.lexists(out_dir):
        _fail(f"{out_dir} already exists (pass --replace to replace it)")
    snap = _read_snapshot(run)
    data, validated = _bind(snap, run.history_sha256)
    report = _parity(snap, validated)
    misses = _coverage_misses(report)
    if misses and not report_only:
        _fail("parity coverage: " + "; ".join(misses) + " (use --report-only for a report row)")
    manifest = _manifest(data, snap, _claims(run, snap, report.claim(vacuous=bool(misses))))
    files = {"policy.onnx": data, "policy.json": dump_manifest(manifest)}
    try:
        load_policy_files(files, challenge=snap.challenge)  # exactly what loading it runs
    except LoadError as exc:
        _fail(f"the exported policy does not load: {exc}")
    contents = _fixture_contents(snap, report, manifest.onnx.sha256) if fixtures else {}
    _write_outputs(out_dir, files, replace=run.replace, fixtures=fixtures, contents=contents)
    return manifest
