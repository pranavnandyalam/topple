import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import topple.traj as traj_module
from topple.traj import FORMAT, Trajectory, load_npz, read_meta


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
    back = load_npz(path, max_steps=5)
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
        load_npz(path, max_steps=1000)


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
        load_npz(path, max_steps=1000)


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
        load_npz(path, max_steps=1000)


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
        load_npz(path, max_steps=1000)


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


# ------------------------------------------------------------------ hardening (spec H)

ARRAY_NAMES = (
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


def npy_bytes(arr: np.ndarray, *, fortran: bool = False) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.asfortranarray(arr) if fortran else arr)
    return buf.getvalue()


def write_members(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def honest_members(traj: Trajectory | None = None, meta: object = None) -> dict[str, bytes]:
    traj = traj or make_traj()
    members = {f"{n}.npy": npy_bytes(getattr(traj, n)) for n in ARRAY_NAMES}
    payload = {"format": FORMAT} if meta is None else meta
    raw = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
    members["meta_json.npy"] = npy_bytes(np.frombuffer(raw, dtype=np.uint8))
    return members


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("obs", (5, 11)),
        ("obs", (5, 9)),
        ("action", (5, 3)),
        ("ctrl", (5, 1)),
        ("enc_counts", (5, 3)),
        ("gyro", (5, 4)),
        ("accel", (5, 2)),
        ("push_force", (5, 4)),
    ],
)
def test_fixed_trailing_shapes(name: str, shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match=name):
        make_traj(**{name: np.zeros(shape)})


def test_state_arrays_need_at_least_one_column() -> None:
    with pytest.raises(ValueError, match="qpos"):
        make_traj(qpos=np.zeros((6, 0)), epilogue_qpos=np.zeros((0, 0)))
    with pytest.raises(ValueError, match="qvel"):
        make_traj(qvel=np.zeros((6, 0)))


def test_epilogue_width_must_match_qpos() -> None:
    with pytest.raises(ValueError, match="epilogue_qpos"):
        make_traj(epilogue_qpos=np.zeros((2, 8)))


def test_load_requires_max_steps(tmp_path: Path) -> None:
    path = tmp_path / "t.npz"
    make_traj().save_npz(path)
    with pytest.raises(TypeError):
        load_npz(path)  # type: ignore[call-arg]


def test_load_enforces_max_steps_exactly(tmp_path: Path) -> None:
    path = tmp_path / "t.npz"
    make_traj(t=5).save_npz(path)
    assert load_npz(path, max_steps=5).steps == 5
    with pytest.raises(ValueError, match="more than 4"):
        load_npz(path, max_steps=4)


def test_per_member_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "t.npz"
    make_traj(t=50).save_npz(path)  # qpos alone is 51 * 9 * 8 = 3672 bytes plus the header
    monkeypatch.setattr(traj_module, "MAX_MEMBER_BYTES", 2000)
    with pytest.raises(ValueError, match="too large"):
        load_npz(path, max_steps=1000)
    monkeypatch.setattr(traj_module, "MAX_MEMBER_BYTES", 10_000)
    assert load_npz(path, max_steps=1000).steps == 50


def test_header_cannot_claim_more_data_than_the_member_holds(tmp_path: Path) -> None:
    """The npy header's shape is checked against the member size before anything is allocated."""
    members = honest_members()
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f4", "fortran_order": False, "shape": (2**20, 10)}
    )
    members["obs.npy"] = header.getvalue() + np.zeros((5, 10), np.float32).tobytes()
    path = tmp_path / "liar.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match="obs"):
        load_npz(path, max_steps=1000)


def test_negative_header_dimensions_are_rejected(tmp_path: Path) -> None:
    members = honest_members()
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f8", "fortran_order": False, "shape": (-1, 2)}
    )
    members["ctrl.npy"] = header.getvalue() + np.zeros((5, 2)).tobytes()
    path = tmp_path / "negative.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match="negative dimension"):
        load_npz(path, max_steps=1000)


def test_shape_errors_name_the_file(tmp_path: Path) -> None:
    members = honest_members()
    members["obs.npy"] = npy_bytes(np.zeros((5, 11), np.float32))
    path = tmp_path / "wide.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match=r"wide\.npz: obs must have shape"):
        load_npz(path, max_steps=1000)


def test_trailing_bytes_after_the_array_are_rejected(tmp_path: Path) -> None:
    members = honest_members()
    members["ctrl.npy"] += b"\x00" * 16
    path = tmp_path / "padded.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match="ctrl"):
        load_npz(path, max_steps=1000)


def test_fortran_order_is_rejected(tmp_path: Path) -> None:
    members = honest_members()
    members["qpos.npy"] = npy_bytes(make_traj().qpos, fortran=True)
    path = tmp_path / "fortran.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match="qpos"):
        load_npz(path, max_steps=1000)


def test_duplicate_members_are_rejected(tmp_path: Path) -> None:
    members = honest_members()
    path = tmp_path / "dup.npz"
    write_members(path, members)
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(path, "a") as zf,
    ):
        zf.writestr("qpos.npy", members["qpos.npy"])
    with pytest.raises(ValueError, match="duplicate"):
        load_npz(path, max_steps=1000)


@pytest.mark.parametrize(
    "meta",
    [
        b"[" * 100_000 + b"]" * 100_000,  # RecursionError inside json
        b'{"format": "topple.traj/1", "x": NaN}',
        b'{"format": "topple.traj/1", "x": Infinity}',
        b"\xff\xfe not utf-8",
        b'["topple.traj/1"]',
        b"",
    ],
    ids=["deep nesting", "NaN", "Infinity", "bad utf-8", "not an object", "empty"],
)
def test_bad_meta_is_a_value_error(tmp_path: Path, meta: bytes) -> None:
    path = tmp_path / "meta.npz"
    write_members(path, honest_members(meta=meta))
    with pytest.raises(ValueError, match="meta"):
        load_npz(path, max_steps=1000)


@pytest.mark.parametrize(
    "content",
    [b"", b"not a zip file at all", b"PK\x03\x04" + b"\x00" * 40],
    ids=["empty", "text", "broken zip"],
)
def test_unreadable_files_are_value_errors(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "junk.npz"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=r"junk\.npz"):
        load_npz(path, max_steps=1000)


def test_missing_file_and_directory_are_value_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"nope\.npz"):
        load_npz(tmp_path / "nope.npz", max_steps=1000)
    with pytest.raises(ValueError, match=re.escape(tmp_path.name)):
        load_npz(tmp_path, max_steps=1000)


def test_corrupted_member_crc_is_a_value_error(tmp_path: Path) -> None:
    path = tmp_path / "t.npz"
    write_members(path, honest_members())
    raw = bytearray(path.read_bytes())
    data = npy_bytes(make_traj().qpos)
    at = raw.find(data[-64:])  # stored uncompressed: flip a byte inside the qpos payload
    assert at > 0
    raw[at] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match=r"t\.npz"):
        load_npz(path, max_steps=1000)


def test_read_meta_reads_only_the_metadata(tmp_path: Path) -> None:
    path = tmp_path / "t.npz"
    make_traj(t=7, meta={"challenge": "balance-v1", "episode_index": 3}).save_npz(path)
    assert read_meta(path) == {"challenge": "balance-v1", "episode_index": 3}
    path.write_bytes(b"junk")
    with pytest.raises(ValueError, match=r"t\.npz"):
        read_meta(path)


@settings(max_examples=60, deadline=None)
@given(st.binary(max_size=2048))
def test_load_never_raises_anything_but_value_error(data: bytes) -> None:
    """Arbitrary bytes, and arbitrary bytes inside a valid zip, only ever give ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "fuzz.npz"
        raw.write_bytes(data)
        with pytest.raises(ValueError, match=r"fuzz\.npz"):
            load_npz(raw, max_steps=1000)
        members = honest_members()
        members["obs.npy"] = data
        zipped = Path(tmp) / "fuzz_member.npz"
        write_members(zipped, members)
        with pytest.raises(ValueError, match=r"fuzz_member\.npz"):
            load_npz(zipped, max_steps=1000)


@pytest.mark.parametrize("max_steps", [-1, True, 2.0, None])
def test_max_steps_must_be_a_non_negative_int(tmp_path: Path, max_steps: object) -> None:
    path = tmp_path / "t.npz"
    make_traj().save_npz(path)
    with pytest.raises(ValueError, match="max_steps"):
        load_npz(path, max_steps=max_steps)  # type: ignore[arg-type]


def test_npy_format_2_is_accepted_and_3_is_not(tmp_path: Path) -> None:
    traj = make_traj()
    members = honest_members(traj)
    for version, ok in (((2, 0), True), ((3, 0), False)):
        buf = io.BytesIO()
        np.lib.format.write_array(buf, traj.qpos, version=version)
        members["qpos.npy"] = buf.getvalue()
        path = tmp_path / f"v{version[0]}.npz"
        write_members(path, members)
        if ok:
            np.testing.assert_array_equal(load_npz(path, max_steps=5).qpos, traj.qpos)
        else:
            with pytest.raises(ValueError, match="npy version"):
                load_npz(path, max_steps=5)


def test_meta_must_be_one_dimensional(tmp_path: Path) -> None:
    members = honest_members()
    raw = np.frombuffer(json.dumps({"format": FORMAT}).encode(), dtype=np.uint8)
    members["meta_json.npy"] = npy_bytes(raw.reshape(1, -1))
    path = tmp_path / "meta2d.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match="1-D"):
        load_npz(path, max_steps=1000)
