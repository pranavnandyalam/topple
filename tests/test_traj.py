import io
import json
import os
import re
import stat
import struct
import tempfile
import threading
import time
import tracemalloc
import zipfile
from collections.abc import Callable, Iterable
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
    with pytest.raises(ValueError, match=r"obs: header shape .* needs more data"):
        load_npz(path, max_steps=2**21)  # within the step limit: only the member size refuses it
    with pytest.raises(ValueError, match="obs: trajectory has 1048576 steps, more than 1000"):
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
        b'{"format": "topple.traj/1", "x": 1e400}',
        b'{"format": "topple.traj/1", "x": [-1e400]}',
        b'{"format": "topple.traj/1", "x": {"y": 1E999}}',
        b"\xff\xfe not utf-8",
        b'["topple.traj/1"]',
        b"",
    ],
    ids=[
        "deep nesting",
        "NaN",
        "Infinity",
        "overflow",
        "negative overflow",
        "nested overflow",
        "bad utf-8",
        "not an object",
        "empty",
    ],
)
def test_bad_meta_is_a_value_error(tmp_path: Path, meta: bytes) -> None:
    path = tmp_path / "meta.npz"
    write_members(path, honest_members(meta=meta))
    with pytest.raises(ValueError, match="meta"):
        load_npz(path, max_steps=1000)


@pytest.mark.parametrize("text", ["1e400", "-1e400", "1E999", "2e308", "-1.8e308"])
def test_read_meta_rejects_numbers_that_overflow_to_infinity(tmp_path: Path, text: str) -> None:
    """``1e400`` is valid JSON syntax but parses to inf: the loader must refuse it (§14)."""
    path = tmp_path / "meta.npz"
    meta = b'{"format": "topple.traj/1", "x": ' + text.encode() + b"}"
    write_members(path, honest_members(meta=meta))
    # The whole message: the dedicated error must not be rewrapped as "not valid UTF-8 JSON".
    exact = rf"^meta\.npz: meta_json: non-finite number {re.escape(text)}$"
    with pytest.raises(ValueError, match=exact):
        read_meta(path)


def test_large_finite_numbers_in_meta_still_load(tmp_path: Path) -> None:
    path = tmp_path / "meta.npz"
    meta = b'{"format": "topple.traj/1", "x": 1.7976931348623157e308, "y": -5e-324, "z": 1e-400}'
    write_members(path, honest_members(meta=meta))
    assert read_meta(path) == {"x": 1.7976931348623157e308, "y": -5e-324, "z": 0.0}


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


def test_a_fifo_is_refused_without_blocking(
    tmp_path: Path, make_fifo: Callable[[Path], Path]
) -> None:
    """Opening a FIFO for reading blocks until a writer appears: never open one (§14)."""
    path = make_fifo(tmp_path / "traj_000.npz")
    with pytest.raises(ValueError, match=r"traj_000\.npz: not a regular file"):
        load_npz(path, max_steps=1000)
    with pytest.raises(ValueError, match=r"traj_000\.npz: not a regular file"):
        read_meta(path)


def _release_blocked_readers(path: Path) -> None:
    """Open ``path`` for writing once, which lets a reader stuck opening the FIFO go on."""
    for _ in range(100):
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:  # ENXIO: no reader has the FIFO open yet
            time.sleep(0.01)
        else:
            return


@pytest.mark.parametrize(
    "read",
    [lambda path: load_npz(path, max_steps=1000), read_meta],
    ids=["load_npz", "read_meta"],
)
def test_a_fifo_with_no_writer_is_refused_at_once(
    tmp_path: Path, read: Callable[[Path], object]
) -> None:
    """No writer ever appears, so a blocking open would hang forever: the loader must refuse the
    FIFO without waiting (§14). Unlike ``make_fifo``, nothing here releases a blocked reader
    until the verdict is in; the release afterwards only keeps a regressed loader from leaking a
    stuck thread."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes need a POSIX system")
    path = tmp_path / "traj_000.npz"
    os.mkfifo(path)
    outcome: list[BaseException | None] = []

    def attempt() -> None:
        try:
            read(path)
        except BaseException as exc:  # recorded and checked below
            outcome.append(exc)
        else:
            outcome.append(None)

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    blocked = thread.is_alive()
    if blocked:
        _release_blocked_readers(path)
        thread.join(timeout=5.0)
    assert not blocked, "opening a FIFO that has no writer blocked"
    assert len(outcome) == 1
    assert isinstance(outcome[0], ValueError)
    assert "traj_000.npz: not a regular file" in str(outcome[0])


def test_a_device_is_refused(tmp_path: Path) -> None:
    device = Path(os.devnull)
    if not device.exists() or device.is_file():
        pytest.skip("no character device to try")
    with pytest.raises(ValueError, match="not a regular file"):
        load_npz(device, max_steps=1000)


def test_a_symlink_to_a_regular_file_loads(tmp_path: Path) -> None:
    traj = make_traj()
    real = tmp_path / "real.npz"
    traj.save_npz(real)
    link = tmp_path / "link.npz"
    link.symlink_to(real)
    assert load_npz(link, max_steps=5).meta == traj.meta


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


# ------------------------------------------------------------------ size limits (§14)


def raw_members(t: int, nq: int, nv: int, e: int, meta: bytes | None = None) -> dict[str, bytes]:
    """Members of a file with the given sizes, built without ``Trajectory``'s own checks."""
    arrays = {
        "qpos": np.zeros((t + 1, nq)),
        "qvel": np.zeros((t + 1, nv)),
        "obs": np.zeros((t, 10), np.float32),
        "action": np.zeros((t, 2)),
        "ctrl": np.zeros((t, 2)),
        "enc_counts": np.zeros((t, 2), np.int32),
        "gyro": np.zeros((t, 3), np.float32),
        "accel": np.zeros((t, 3), np.float32),
        "push_force": np.zeros((t, 3)),
        "epilogue_qpos": np.zeros((e, nq)),
    }
    members = {f"{n}.npy": npy_bytes(a) for n, a in arrays.items()}
    raw = json.dumps({"format": FORMAT}).encode() if meta is None else meta
    members["meta_json.npy"] = npy_bytes(np.frombuffer(raw, dtype=np.uint8))
    return members


def write_deflated(path: Path, members: dict[str, bytes]) -> None:
    """Like ``write_members``, compressed: a bloated member costs next to nothing on disk."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def meta_of_size(size: int) -> bytes:
    """A valid ``meta_json`` document of exactly ``size`` bytes."""
    head = json.dumps({"format": FORMAT, "pad": ""}).encode()
    return head[:-2] + b"x" * (size - len(head)) + head[-2:]


def test_state_widths_are_capped() -> None:
    cap = traj_module.MAX_STATE_DIM
    assert make_traj(nq=cap, nv=cap).qpos.shape == (6, cap)
    with pytest.raises(ValueError, match=f"qpos must have at most {cap} columns"):
        make_traj(nq=cap + 1)
    with pytest.raises(ValueError, match=f"qvel must have at most {cap} columns"):
        make_traj(nv=cap + 1)


def test_a_wide_state_is_refused_at_load(tmp_path: Path) -> None:
    cap = traj_module.MAX_STATE_DIM
    for name, (nq, nv) in {"qpos": (cap + 1, 8), "qvel": (9, cap + 1)}.items():
        path = tmp_path / f"wide_{name}.npz"
        write_members(path, raw_members(t=2, nq=nq, nv=nv, e=0))
        with pytest.raises(ValueError, match=rf"wide_{name}\.npz: {name} must have at most"):
            load_npz(path, max_steps=1000)


def test_epilogue_rows_are_bounded_by_max_steps(tmp_path: Path) -> None:
    path = tmp_path / "t.npz"
    write_members(path, raw_members(t=2, nq=9, nv=8, e=5))
    assert load_npz(path, max_steps=5).epilogue_qpos.shape == (5, 9)
    with pytest.raises(ValueError, match="epilogue_qpos has 5 rows, more than 4"):
        load_npz(path, max_steps=4)


@pytest.mark.parametrize(
    "read",
    [lambda path: load_npz(path, max_steps=1000), read_meta],
    ids=["load_npz", "read_meta"],
)
def test_meta_json_is_capped(tmp_path: Path, read: Callable[[Path], object]) -> None:
    cap = traj_module.MAX_META_BYTES
    assert cap < traj_module.MAX_MEMBER_BYTES
    path = tmp_path / "meta.npz"
    write_deflated(path, raw_members(t=2, nq=9, nv=8, e=0, meta=meta_of_size(cap)))
    read(path)
    write_deflated(path, raw_members(t=2, nq=9, nv=8, e=0, meta=meta_of_size(cap + 1)))
    with pytest.raises(ValueError, match="meta_json"):
        read(path)


def test_bloated_members_are_refused_before_their_data_is_read(tmp_path: Path) -> None:
    """A few KB of deflated zeros can claim 16 MiB per member: a header shape beyond the format's
    limits (T <= max_steps, widths <= MAX_STATE_DIM, meta <= MAX_META_BYTES) is refused before
    any of its data is read, so loading never allocates more than an honest file could need."""
    width = 2_000_000  # 16 MB of f8 per state member: under the 16 MiB member cap
    members = raw_members(t=0, nq=width, nv=width, e=1)
    path = tmp_path / "bloat.npz"
    write_deflated(path, members)
    assert path.stat().st_size < 200_000
    del members
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=r"bloat\.npz: qpos"):
            load_npz(path, max_steps=1000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000


@settings(max_examples=40, deadline=None)
@given(
    t=st.integers(0, 4),
    nq=st.sampled_from([1, 9, 256, 257, 300]),
    nv=st.sampled_from([1, 8, 256, 257]),
    e=st.integers(0, 5),
)
def test_load_accepts_exactly_the_files_within_the_limits(t: int, nq: int, nv: int, e: int) -> None:
    """With max_steps = 3: T <= 3, E <= 3 and both widths <= MAX_STATE_DIM load, nothing else."""
    cap = traj_module.MAX_STATE_DIM
    within = t <= 3 and e <= 3 and nq <= cap and nv <= cap
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "limits.npz"
        write_members(path, raw_members(t=t, nq=nq, nv=nv, e=e))
        if within:
            traj = load_npz(path, max_steps=3)
            assert traj.steps == t
            assert traj.qpos.shape == (t + 1, nq)
            assert traj.epilogue_qpos.shape == (e, nq)
        else:
            with pytest.raises(ValueError, match=r"limits\.npz"):
                load_npz(path, max_steps=3)


@pytest.mark.parametrize(
    ("name", "arr"),
    [
        ("qpos", np.zeros((6, 9, 1))),  # a third axis would slip past the width limit
        ("epilogue_qpos", np.zeros(9)),
        ("ctrl", np.zeros((5, 2, 1))),
    ],
)
def test_array_members_must_be_two_dimensional(tmp_path: Path, name: str, arr: np.ndarray) -> None:
    members = honest_members()
    members[f"{name}.npy"] = npy_bytes(arr)
    path = tmp_path / "ndim.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match=rf"ndim\.npz: {name} must be 2-D"):
        load_npz(path, max_steps=1000)


# ------------------------------------------------------------------ hostile archives (§14)


def peak_traced(action: Callable[[], object]) -> int:
    """Peak bytes Python allocated while running ``action``, which must refuse its file with a
    ``ValueError`` naming it."""
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=r"\.npz: "):
            action()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def write_streamed(
    path: Path, members: dict[str, bytes], name: str, chunks: Iterable[bytes], method: int
) -> None:
    """``members`` stored, plus ``name`` compressed with ``method`` from ``chunks`` as they come,
    so a bomb never sits whole in the test's memory."""
    with zipfile.ZipFile(path, "w") as zf:
        for member, data in members.items():
            zf.writestr(member, data)
        info = zipfile.ZipInfo(name)
        info.compress_type = method
        with zf.open(info, "w") as fh:
            for chunk in chunks:
                fh.write(chunk)


def central_entry(raw: bytes | bytearray, name: str) -> int:
    """Offset of ``name``'s entry in the zip central directory of ``raw``."""
    at = raw.find(b"PK\x01\x02")
    while at >= 0:
        length = int.from_bytes(raw[at + 28 : at + 30], "little")
        if raw[at + 46 : at + 46 + length] == name.encode():
            return at
        at = raw.find(b"PK\x01\x02", at + 4)
    raise AssertionError(f"{name} is not in the central directory")


def declare_size(path: Path, name: str, size: int) -> None:
    """Make the archive lie about ``name``'s uncompressed size, consistently: in its central
    directory entry and in its local header (written by ``zipfile`` with the sizes in the
    header), so the lie passes the layout checks and reaches the member reader."""
    raw = bytearray(path.read_bytes())
    at = central_entry(raw, name)
    raw[at + 24 : at + 28] = size.to_bytes(4, "little")
    header = int.from_bytes(raw[at + 42 : at + 46], "little")
    assert raw[header : header + 4] == b"PK\x03\x04"
    raw[header + 22 : header + 26] = size.to_bytes(4, "little")
    path.write_bytes(bytes(raw))


def npy_with_header(header: str, data: bytes = b"", *, version: int = 1) -> bytes:
    """An npy member with a hand-written header (numpy would refuse to write a hostile one)."""
    raw = header.encode("latin1")
    length = struct.pack("<H", len(raw)) if version == 1 else struct.pack("<I", len(raw))
    return b"\x93NUMPY" + bytes([version, 0]) + length + raw + data


@pytest.mark.parametrize("method", [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA], ids=["bzip2", "lzma"])
def test_only_stored_and_deflated_members_are_read(tmp_path: Path, method: int) -> None:
    """bzip2 and lzma decompress a whole compressed chunk at once, with no output limit: such a
    member is refused before anything is read, as soon as the member list is checked."""
    path = tmp_path / "packed.npz"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in honest_members().items():
            zf.writestr(name, data, compress_type=method if name == "qpos.npy" else None)
    message = rf"packed\.npz: qpos\.npy: compression method {method} is not allowed"
    with pytest.raises(ValueError, match=message):
        load_npz(path, max_steps=1000)
    with pytest.raises(ValueError, match=message):
        read_meta(path)


def test_a_bzip2_bomb_with_a_lying_size_is_never_decompressed(tmp_path: Path) -> None:
    """The review's bomb: 32 MiB of zeros in a few hundred bytes of bzip2 whose directory entry
    declares 100 bytes. Decompressing even its first chunk would allocate all 32 MiB."""
    members = honest_members()
    del members["qpos.npy"]
    path = tmp_path / "bomb.npz"
    write_streamed(path, members, "qpos.npy", [bytes(1 << 20)] * 32, zipfile.ZIP_BZIP2)
    declare_size(path, "qpos.npy", 100)
    assert path.stat().st_size < 20_000
    assert peak_traced(lambda: load_npz(path, max_steps=1000)) < 2_000_000
    with pytest.raises(ValueError, match=r"bomb\.npz: qpos\.npy: compression method 12"):
        load_npz(path, max_steps=1000)


def test_a_huge_npy_header_length_is_refused_before_the_header_is_read(tmp_path: Path) -> None:
    """An npy 2.0 header length is 4 bytes the uploader picks: asking the zip stream for that
    many bytes at once would make deflate expand 64 KB of spaces into 64 MiB, even though the
    directory declares only 1 MiB for the member."""
    members = honest_members()
    del members["qpos.npy"]
    path = tmp_path / "header.npz"
    length = 64 << 20
    head = b"\x93NUMPY\x02\x00" + struct.pack("<I", length)
    write_streamed(
        path, members, "qpos.npy", [head, *[b" " * (1 << 20)] * 64], zipfile.ZIP_DEFLATED
    )
    declare_size(path, "qpos.npy", 1 << 20)  # within the 16 MiB member cap
    assert path.stat().st_size < 200_000
    assert peak_traced(lambda: load_npz(path, max_steps=1000)) < 2_000_000
    cap = traj_module.MAX_NPY_HEADER_BYTES
    with pytest.raises(ValueError, match=rf"qpos: npy header has {length} bytes, more than {cap}"):
        load_npz(path, max_steps=1000)


def test_a_lying_deflated_size_reads_only_what_it_declares(tmp_path: Path) -> None:
    """A deflated member that holds far more than its sizes declare: reading stops at the
    declared size, the stream must end there and does not, and no read asks the stream for more
    than a small chunk, so the 32 MiB behind it are never inflated."""
    members = honest_members()
    qpos = members.pop("qpos.npy")
    path = tmp_path / "liar.npz"
    write_streamed(path, members, "qpos.npy", [qpos, *[bytes(1 << 20)] * 32], zipfile.ZIP_DEFLATED)
    declare_size(path, "qpos.npy", len(qpos))
    assert peak_traced(lambda: load_npz(path, max_steps=1000)) < 2_000_000
    with pytest.raises(ValueError, match=r"qpos\.npy: the member inflates to more than its"):
        load_npz(path, max_steps=1000)


MINUS_CHAIN = "{'descr': '|u1', 'fortran_order': False, 'shape': (" + "-" * 9000 + "1,), }"
"""Under numpy's own 10,000-character header limit, but ``ast.literal_eval`` gives up on it with
``MemoryError: Parser stack overflowed``."""


@pytest.mark.parametrize(
    ("member", "read"),
    [
        ("meta_json.npy", read_meta),
        ("meta_json.npy", lambda path: load_npz(path, max_steps=1000)),
        ("qpos.npy", lambda path: load_npz(path, max_steps=1000)),
    ],
    ids=["read_meta", "load_npz meta", "load_npz qpos"],
)
def test_a_header_too_complex_to_parse_is_a_value_error(
    tmp_path: Path, member: str, read: Callable[[Path], object]
) -> None:
    members = honest_members()
    members[member] = npy_with_header(MINUS_CHAIN)
    path = tmp_path / "minus.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match=r"minus\.npz: (meta_json|qpos): npy header has"):
        read(path)


@pytest.mark.parametrize(
    "header",
    [
        "{'descr': '<f8', 'fortran_order': False, 'shape': (--6, 9), }",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (-" + "-" * 3000 + "6, 9), }",
        "{'descr': '<f8', 'fortran_order': 0, 'shape': (6, 9), }",
        "{'descr': ['<f8'], 'fortran_order': False, 'shape': (6, 9), }",
        "{'descr': '<f8', 'fortran_order': False, 'shape': 6, }",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (6), }",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (0x6, 9), }",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (6, 9,,), }",
        "{'descr': '<f8', 'fortran_order': False, 'shape': ((6,), 9), }",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (6, 9), 'extra': 1, }",
        "{'descr': '<f8', 'fortran_order': False, 'shape': (6, 9), } junk",
        "{'fortran_order': False, 'descr': '<f8', 'shape': (6, 9), }",
        "{'descr': '<f8', 'fortran_order': False}",
        "",
    ],
    ids=[
        "double minus",
        "long minus chain",
        "flag not a bool",
        "descr not a string",
        "shape not a tuple",
        "parenthesized int",
        "hex dimension",
        "empty dimension",
        "nested tuple",
        "extra key",
        "trailing junk",
        "keys out of order",
        "missing key",
        "empty",
    ],
)
def test_a_malformed_npy_header_is_refused_without_evaluating_it(
    tmp_path: Path, header: str
) -> None:
    """The loader parses the header text itself and never hands it to a Python parser."""
    members = honest_members()
    members["qpos.npy"] = npy_with_header(header, np.zeros((6, 9)).tobytes())
    path = tmp_path / "header.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match=r"header\.npz: qpos: malformed npy header"):
        load_npz(path, max_steps=1000)


@pytest.mark.parametrize(
    "header",
    [
        "{'descr': '<f8', 'fortran_order': False, 'shape': (6, 9), }" + " " * 69 + "\n",
        "{'descr':'<f8','fortran_order':False,'shape':(6,9)}",
        "{ 'descr' : '<f8' , 'fortran_order' : False , 'shape' : ( 6 , 9 , ) }\n",
    ],
    ids=["numpy", "compact", "spaced"],
)
@pytest.mark.parametrize("version", [1, 2])
def test_npy_headers_written_by_other_tools_load(tmp_path: Path, header: str, version: int) -> None:
    traj = make_traj(t=5, nq=9)
    members = honest_members(traj)
    members["qpos.npy"] = npy_with_header(header, traj.qpos.tobytes(), version=version)
    path = tmp_path / "other.npz"
    write_members(path, members)
    np.testing.assert_array_equal(load_npz(path, max_steps=5).qpos, traj.qpos)


def test_an_npy_header_must_not_be_cut_short(tmp_path: Path) -> None:
    members = honest_members()
    members["qpos.npy"] = npy_with_header("{'descr': '<f8'")[:-4]
    path = tmp_path / "short.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match=r"short\.npz: qpos: truncated npy header"):
        load_npz(path, max_steps=1000)
    members["qpos.npy"] = b"\x93NUMPY\x01\x00\x10"  # one byte of a two-byte header length
    write_members(path, members)
    with pytest.raises(ValueError, match=r"short\.npz: qpos: truncated npy header"):
        load_npz(path, max_steps=1000)
    members["qpos.npy"] = b"\x93NUMP"
    write_members(path, members)
    with pytest.raises(ValueError, match=r"short\.npz: qpos: not an npy member"):
        load_npz(path, max_steps=1000)


def test_an_encrypted_member_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "locked.npz"
    write_members(path, honest_members())
    raw = bytearray(path.read_bytes())
    at = central_entry(raw, "obs.npy")
    raw[at + 8] |= 0x01  # general purpose flag bit 0: encrypted
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match=r"locked\.npz: obs\.npy: encrypted members"):
        load_npz(path, max_steps=1000)


def test_a_file_over_the_size_cap_is_refused_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "t.npz"
    make_traj().save_npz(path)
    size = path.stat().st_size
    monkeypatch.setattr(traj_module, "MAX_FILE_BYTES", size - 1)
    message = rf"t\.npz: the file has {size} bytes, more than {size - 1}"
    with pytest.raises(ValueError, match=message):
        load_npz(path, max_steps=5)
    with pytest.raises(ValueError, match=message):
        read_meta(path)
    monkeypatch.setattr(traj_module, "MAX_FILE_BYTES", size)
    assert load_npz(path, max_steps=5).steps == 5


def test_the_real_size_cap_refuses_a_huge_sparse_file(tmp_path: Path) -> None:
    path = tmp_path / "huge.npz"
    path.write_bytes(b"")
    os.truncate(path, traj_module.MAX_FILE_BYTES + 1)  # sparse: costs no disk space
    with pytest.raises(ValueError, match=r"huge\.npz: the file has \d+ bytes, more than"):
        load_npz(path, max_steps=1000)


@pytest.mark.parametrize(
    "read",
    [lambda path: load_npz(path, max_steps=1000), read_meta],
    ids=["load_npz", "read_meta"],
)
def test_a_directory_of_many_entries_is_refused_before_it_is_parsed(
    tmp_path: Path, read: Callable[[Path], object]
) -> None:
    """``zipfile`` builds one object per directory entry before the member list can be checked
    (about 7 bytes of memory per byte of directory): the directory's size is checked first."""
    path = tmp_path / "entries.npz"
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(20_000):
            zf.writestr(str(i), b"")
    assert peak_traced(lambda: read(path)) < 1_000_000
    cap = traj_module.MAX_DIRECTORY_BYTES
    message = rf"entries\.npz: the zip central directory has \d+ bytes, more than {cap}"
    with pytest.raises(ValueError, match=message):
        read(path)


def test_the_end_record_must_end_the_file(tmp_path: Path) -> None:
    """No archive comment, so ``zipfile`` reads exactly the record that was checked."""
    path = tmp_path / "comment.npz"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in honest_members().items():
            zf.writestr(name, data)
        zf.comment = b"hello"
    with pytest.raises(ValueError, match=r"comment\.npz: .*end-of-central-directory"):
        load_npz(path, max_steps=1000)


def test_zip64_end_records_are_refused(tmp_path: Path) -> None:
    """A zip64 locator just before the end record would make ``zipfile`` read the directory size
    from the zip64 record instead of the one that was checked."""
    path = tmp_path / "z64.npz"
    write_members(path, honest_members())
    raw = path.read_bytes()
    body, end = raw[:-22], raw[-22:]
    _, _, _, _, entries, cd_size, cd_offset, _ = struct.unpack("<4s4H2LH", end)
    record = struct.pack(
        "<4sQ2H2L2Q2Q", b"PK\x06\x06", 44, 45, 45, 0, 0, entries, entries, cd_size, cd_offset
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, len(body), 1)
    path.write_bytes(body + record + locator + end)
    with pytest.raises(ValueError, match=r"z64\.npz: zip64 archives are not allowed"):
        load_npz(path, max_steps=1000)


def test_unexpected_member_names_are_quoted_briefly(tmp_path: Path) -> None:
    path = tmp_path / "names.npz"
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(40):
            zf.writestr(f"\x1b[2J{'x' * 100}{i}", b"")
    with pytest.raises(ValueError, match="unexpected members") as caught:
        load_npz(path, max_steps=1000)
    message = str(caught.value)
    assert len(message) < 600
    assert "\x1b" not in message  # names are shown escaped
    assert "and 46 more" in message


def test_a_multi_disk_archive_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "disks.npz"
    write_members(path, honest_members())
    raw = bytearray(path.read_bytes())
    raw[-18:-16] = (1).to_bytes(2, "little")  # end record: number of this disk
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match=r"disks\.npz: multi-disk zip archives are not allowed"):
        load_npz(path, max_steps=1000)


def test_a_zero_dimensional_member_is_refused(tmp_path: Path) -> None:
    members = honest_members()
    members["qpos.npy"] = npy_with_header(
        "{'descr': '<f8', 'fortran_order': False, 'shape': (), }", np.zeros(1).tobytes()
    )
    path = tmp_path / "scalar.npz"
    write_members(path, members)
    with pytest.raises(ValueError, match=r"scalar\.npz: qpos must be 2-D, got header shape \(\)"):
        load_npz(path, max_steps=1000)


# ------------------------------------------------------------------ writing (§14)


def test_saving_replaces_a_planted_symlink_instead_of_writing_through_it(tmp_path: Path) -> None:
    """A checkout can plant ``out/run/traj_000.npz -> ~/.zshrc``: saving must not clobber the
    link's target."""
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    out = tmp_path / "out"
    out.mkdir()
    (out / "traj_000.npz").symlink_to(victim)
    traj = make_traj()
    traj.save_npz(out / "traj_000.npz")
    assert victim.read_text() == "precious"
    assert not (out / "traj_000.npz").is_symlink()
    assert load_npz(out / "traj_000.npz", max_steps=5).meta == traj.meta
    assert [p.name for p in out.iterdir()] == ["traj_000.npz"]  # no temporary file left
    assert stat.S_IMODE((out / "traj_000.npz").stat().st_mode) == 0o644


def test_a_failed_save_leaves_the_old_file_and_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "traj_000.npz"
    make_traj(t=3).save_npz(path)
    before = path.read_bytes()

    def half_written(file: object, **arrays: object) -> None:
        file.write(b"PK\x03\x04 half")  # type: ignore[attr-defined]
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(traj_module.np, "savez_compressed", half_written)
    with pytest.raises(OSError, match="No space left"):
        make_traj(t=5).save_npz(path)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["traj_000.npz"]


def test_saving_creates_missing_directories(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "traj_000.npz"
    make_traj().save_npz(path)
    assert load_npz(path, max_steps=5).steps == 5
