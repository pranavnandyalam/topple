import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from topple.traj import FORMAT, Trajectory, load_npz


def make_traj(t: int = 5, nq: int = 9, nv: int = 8, **overrides: object) -> Trajectory:
    rng = np.random.default_rng(0)
    fields: dict[str, object] = {
        "qpos": rng.normal(size=(t + 1, nq)),
        "qvel": rng.normal(size=(t + 1, nv)),
        "obs": rng.normal(size=(t, 10)).astype(np.float32),
        "action": rng.normal(size=(t, 2)),
        "ctrl": rng.uniform(-1, 1, size=(t, 2)),
        "enc_counts": rng.integers(-100, 100, size=(t, 2)),
        "gyro": rng.normal(size=(t, 3)),
        "accel": rng.normal(size=(t, 3)),
        "push_force": np.zeros((t, 3)),
        "epilogue_qpos": rng.normal(size=(3, nq)),
        "meta": {"challenge": "balance-v1", "episode_index": 0},
    }
    fields.update(overrides)
    return Trajectory(**fields)  # type: ignore[arg-type]


def test_round_trip(tmp_path: Path) -> None:
    traj = make_traj()
    path = tmp_path / "traj.npz"
    traj.save_npz(path)
    back = load_npz(path)
    for name in (
        "qpos",
        "qvel",
        "obs",
        "action",
        "ctrl",
        "enc_counts",
        "gyro",
        "accel",
        "push_force",
        "epilogue_qpos",
    ):
        np.testing.assert_array_equal(getattr(back, name), getattr(traj, name))
        assert getattr(back, name).dtype == getattr(traj, name).dtype
    assert back.meta == traj.meta


def test_arrays_are_read_only() -> None:
    traj = make_traj()
    with pytest.raises(ValueError, match="read-only"):
        traj.qpos[0, 0] = 1.0


def test_dtypes_are_canonical() -> None:
    traj = make_traj()
    assert traj.obs.dtype == np.float32
    assert traj.enc_counts.dtype == np.int32
    assert traj.qpos.dtype == np.float64


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="T\\+1"):
        make_traj(qpos=np.zeros((5, 9)))
    with pytest.raises(ValueError, match="obs"):
        make_traj(obs=np.zeros((4, 10)))
    with pytest.raises(ValueError, match="push_force"):
        make_traj(push_force=np.zeros((5, 2)))


def test_nan_actions_allowed() -> None:
    action = np.zeros((5, 2))
    action[-1] = np.nan
    assert np.isnan(make_traj(action=action).action[-1]).all()


def test_steps() -> None:
    assert make_traj(t=7).steps == 7


def test_load_rejects_extra_member(tmp_path: Path) -> None:
    path = tmp_path / "t.npz"
    make_traj().save_npz(path)
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("evil.npy", b"x")
    with pytest.raises(ValueError, match="unexpected members"):
        load_npz(path)


def test_load_rejects_pickled_object_array(tmp_path: Path) -> None:
    traj = make_traj()
    arrays = {
        n: getattr(traj, n)
        for n in (
            "qpos",
            "qvel",
            "obs",
            "action",
            "ctrl",
            "enc_counts",
            "gyro",
            "accel",
            "push_force",
            "epilogue_qpos",
        )
    }
    arrays["obs"] = np.array([{"x": 1}] * 5, dtype=object)
    arrays["meta_json"] = np.frombuffer(json.dumps({"format": FORMAT}).encode(), dtype=np.uint8)
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Object arrays"):
        load_npz(path)


def test_load_rejects_wrong_dtype(tmp_path: Path) -> None:
    traj = make_traj()
    arrays = {
        n: getattr(traj, n)
        for n in (
            "qpos",
            "qvel",
            "obs",
            "action",
            "ctrl",
            "enc_counts",
            "gyro",
            "accel",
            "push_force",
            "epilogue_qpos",
        )
    }
    arrays["enc_counts"] = arrays["enc_counts"].astype(np.int64)
    arrays["meta_json"] = np.frombuffer(json.dumps({"format": FORMAT}).encode(), dtype=np.uint8)
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="enc_counts"):
        load_npz(path)


def test_load_rejects_wrong_format(tmp_path: Path) -> None:
    traj = make_traj()
    arrays = {
        n: getattr(traj, n)
        for n in (
            "qpos",
            "qvel",
            "obs",
            "action",
            "ctrl",
            "enc_counts",
            "gyro",
            "accel",
            "push_force",
            "epilogue_qpos",
        )
    }
    arrays["meta_json"] = np.frombuffer(json.dumps({"format": "other/1"}).encode(), dtype=np.uint8)
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"not a topple\.traj/1"):
        load_npz(path)


def test_load_enforces_max_steps(tmp_path: Path) -> None:
    path = tmp_path / "t.npz"
    make_traj(t=5).save_npz(path)
    with pytest.raises(ValueError, match="more than 4"):
        load_npz(path, max_steps=4)


def test_meta_is_utf8_json_bytes(tmp_path: Path) -> None:
    path = tmp_path / "t.npz"
    make_traj(meta={"note": "héllo"}).save_npz(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["meta_json"].tobytes().decode("utf-8"))
    assert meta == {"format": FORMAT, "note": "héllo"}
