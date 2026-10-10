"""Episode trajectories: ``topple.traj/1`` (docs/contract.md §14).

A trajectory is the complete, replayable record of one episode. Files are plain ``.npz`` with a
small allow-list of dtypes so browsers can decode them, and loading never unpickles anything.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FORMAT = "topple.traj/1"
MAX_MEMBER_BYTES = 16 * 1024 * 1024
_ALLOWED_DTYPES = {np.dtype("<f8"), np.dtype("<f4"), np.dtype("<i4"), np.dtype("|u1")}
# name -> (dtype, trailing shape or None for model-dependent width)
_SCHEMA: dict[str, tuple[str, tuple[int, ...] | None]] = {
    "qpos": ("<f8", None),
    "qvel": ("<f8", None),
    "obs": ("<f4", None),
    "action": ("<f8", None),
    "ctrl": ("<f8", None),
    "enc_counts": ("<i4", (2,)),
    "gyro": ("<f4", (3,)),
    "accel": ("<f4", (3,)),
    "push_force": ("<f8", (3,)),
    "epilogue_qpos": ("<f8", None),
}


def _frozen(arr: npt.ArrayLike, dtype: str) -> npt.NDArray[Any]:
    out = np.ascontiguousarray(np.asarray(arr, dtype=np.dtype(dtype)))
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Trajectory:
    qpos: npt.NDArray[np.float64]  # (T+1, nq)
    qvel: npt.NDArray[np.float64]  # (T+1, nv)
    obs: npt.NDArray[np.float32]  # (T, n_obs)
    action: npt.NDArray[np.float64]  # (T, nu) raw controller output, NaN rows on error
    ctrl: npt.NDArray[np.float64]  # (T, nu) clipped command
    enc_counts: npt.NDArray[np.int32]  # (T, 2)
    gyro: npt.NDArray[np.float32]  # (T, 3)
    accel: npt.NDArray[np.float32]  # (T, 3)
    push_force: npt.NDArray[np.float64]  # (T, 3)
    epilogue_qpos: npt.NDArray[np.float64]  # (E, nq)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, (dtype, _) in _SCHEMA.items():
            object.__setattr__(self, name, _frozen(getattr(self, name), dtype))
        self._check_shapes()

    @property
    def steps(self) -> int:
        return int(self.ctrl.shape[0])

    def _check_shapes(self) -> None:
        t = self.steps
        if self.qpos.ndim != 2 or self.qpos.shape[0] != t + 1:
            raise ValueError(f"qpos must have T+1={t + 1} rows, got shape {self.qpos.shape}")
        if self.qvel.ndim != 2 or self.qvel.shape[0] != t + 1:
            raise ValueError(f"qvel must have T+1={t + 1} rows, got shape {self.qvel.shape}")
        for name, (_, trailing) in _SCHEMA.items():
            arr = getattr(self, name)
            if name in ("qpos", "qvel", "epilogue_qpos"):
                continue
            if arr.ndim != 2 or arr.shape[0] != t:
                raise ValueError(f"{name} must have T={t} rows, got shape {arr.shape}")
            if trailing is not None and arr.shape[1:] != trailing:
                raise ValueError(f"{name} must have shape (T, {trailing[0]}), got {arr.shape}")
        if self.epilogue_qpos.ndim != 2 or self.epilogue_qpos.shape[1] != self.qpos.shape[1]:
            raise ValueError(f"epilogue_qpos must have {self.qpos.shape[1]} columns")

    def save_npz(self, path: Path) -> None:
        meta = {"format": FORMAT, **self.meta}
        arrays = {name: getattr(self, name) for name in _SCHEMA}
        arrays["meta_json"] = np.frombuffer(
            json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
        for name, arr in arrays.items():
            if arr.dtype not in _ALLOWED_DTYPES or not arr.flags.c_contiguous:
                raise ValueError(f"{name}: dtype {arr.dtype} not allowed in {FORMAT}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)


def load_npz(path: Path, max_steps: int | None = None) -> Trajectory:
    """Load and validate a trajectory without trusting the file (no pickles, size caps)."""
    with zipfile.ZipFile(path) as zf:
        names = {info.filename for info in zf.infolist()}
        expected = {f"{n}.npy" for n in _SCHEMA} | {"meta_json.npy"}
        if names != expected:
            raise ValueError(f"unexpected members {sorted(names ^ expected)}")
        for info in zf.infolist():
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"{info.filename} is too large ({info.file_size} bytes)")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in (*_SCHEMA, "meta_json")}
    for name, (dtype, _) in _SCHEMA.items():
        if arrays[name].dtype != np.dtype(dtype):
            raise ValueError(f"{name}: expected dtype {dtype}, got {arrays[name].dtype}")
    if arrays["meta_json"].dtype != np.uint8 or arrays["meta_json"].ndim != 1:
        raise ValueError("meta_json must be a 1-D uint8 array")
    meta = json.loads(arrays["meta_json"].tobytes().decode("utf-8"))
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} file")
    meta.pop("format")
    traj = Trajectory(**{name: arrays[name] for name in _SCHEMA}, meta=meta)
    if max_steps is not None and traj.steps > max_steps:
        raise ValueError(f"trajectory has {traj.steps} steps, more than {max_steps}")
    return traj
