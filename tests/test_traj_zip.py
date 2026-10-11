"""``topple.traj/1`` loads only the canonical zip layout (docs/contract.md §14).

Python's ``zipfile`` finds an archive from its end and silently skips whatever comes before it, a
browser decoder (fflate) takes the stored offsets as they are, and a streaming reader (libarchive
reading a pipe) walks the local headers from byte 0. On any layout but the one numpy writes they
can decode different trajectories from the same bytes, so a verified file could replay as a
fabricated one. Two minimal readers here stand in for those decoders: whatever ``load_npz``
accepts, they must decode exactly the members ``zipfile`` does.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from topple._zip import Archive
from topple.traj import FORMAT, Trajectory, load_npz, read_meta

LOCAL = struct.Struct("<4s5H3L2H")  # local file header, before its name and extra field
CENTRAL = struct.Struct("<4s6H3L5H2L")  # central directory entry, before name, extra, comment
END = struct.Struct("<4s4H2LH")  # end of central directory record
ZIP64_EXTRA = struct.Struct("<2H2Q")  # id 1, 16 bytes: uncompressed then compressed size
UNKNOWN = 0xFFFFFFFF
ARRAYS = (
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


def make_traj(t: int, *, seed: int = 0) -> Trajectory:
    rng = np.random.default_rng(seed)
    return Trajectory(
        qpos=rng.normal(size=(t + 1, 9)),
        qvel=rng.normal(size=(t + 1, 8)),
        obs=rng.normal(size=(t, 10)).astype(np.float32),
        action=rng.normal(size=(t, 2)),
        ctrl=rng.uniform(-1, 1, size=(t, 2)),
        enc_counts=rng.integers(-100, 100, size=(t, 2)),
        gyro=rng.normal(size=(t, 3)),
        accel=rng.normal(size=(t, 3)),
        push_force=np.zeros((t, 3)),
        epilogue_qpos=rng.normal(size=(2, 9)),
        meta={"challenge": "balance-v1", "episode_index": seed},
    )


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, arr)
    return buf.getvalue()


def honest_contents(traj: Trajectory) -> dict[str, bytes]:
    """The 11 members of ``traj``'s file, as uncompressed ``.npy`` bytes."""
    contents = {f"{name}.npy": npy_bytes(getattr(traj, name)) for name in ARRAYS}
    meta = json.dumps({"format": FORMAT, **traj.meta}, sort_keys=True).encode()
    contents["meta_json.npy"] = npy_bytes(np.frombuffer(meta, dtype=np.uint8))
    return contents


# ------------------------------------------------------------------ independent readers


def _decoded(data: bytes, method: int, size: int) -> bytes:
    if method == zipfile.ZIP_STORED:
        return data  # every compressed byte, as fflate and libarchive return a stored member
    if method != zipfile.ZIP_DEFLATED:
        raise ValueError(f"method {method}")
    return zlib.decompressobj(-15).decompress(data, size + 1)  # bounded: never a bomb


def streamed_members(raw: bytes) -> dict[bytes, bytes]:
    """What a streaming reader extracts: local headers in file order from byte 0, each member's
    sizes from its own header (or its zip64 extra field), until something else follows."""
    members: dict[bytes, bytes] = {}
    at = 0
    while raw[at : at + 4] == b"PK\x03\x04":
        fields = LOCAL.unpack_from(raw, at)
        method, compressed, size, name_length, extra_length = (*fields[3:4], *fields[7:])
        name = raw[at + LOCAL.size : at + LOCAL.size + name_length]
        extra = raw[at + LOCAL.size + name_length : at + LOCAL.size + name_length + extra_length]
        if (compressed, size) == (UNKNOWN, UNKNOWN):
            _, _, size, compressed = ZIP64_EXTRA.unpack(extra)
        start = at + LOCAL.size + name_length + extra_length
        members[name] = _decoded(raw[start : start + compressed], method, size)
        at = start + compressed
    return members


def directory_members(raw: bytes) -> dict[bytes, bytes]:
    """What a browser decoder (fflate's ``unzipSync``) extracts: the end record's entry count and
    directory offset taken as they are, each member read where its directory entry points."""
    *_, entries, _, offset, _ = END.unpack_from(raw, len(raw) - END.size)
    members: dict[bytes, bytes] = {}
    for _ in range(entries):
        fields = CENTRAL.unpack_from(raw, offset)
        method, compressed, size = fields[4], fields[8], fields[9]
        name_length, extra_length, comment_length, header = (*fields[10:13], fields[16])
        name = raw[offset + CENTRAL.size : offset + CENTRAL.size + name_length]
        local_name_length, local_extra_length = LOCAL.unpack_from(raw, header)[9:]
        start = header + LOCAL.size + local_name_length + local_extra_length
        members[name] = _decoded(raw[start : start + compressed], method, size)
        offset += CENTRAL.size + name_length + extra_length + comment_length
    return members


def zipfile_members(path: Path) -> dict[bytes, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {info.orig_filename.encode(): zf.read(info) for info in zf.infolist()}


def assert_decoded_alike(path: Path) -> None:
    """Every reader decodes the same members from ``path`` (which ``load_npz`` accepted)."""
    raw = path.read_bytes()
    expected = zipfile_members(path)
    assert streamed_members(raw) == expected
    assert directory_members(raw) == expected


# ------------------------------------------------------------------ honest writers


def _save_npz(traj: Trajectory, path: Path) -> None:
    traj.save_npz(path)


def _np_savez(compress: bool) -> Callable[[Trajectory, Path], None]:
    def write(traj: Trajectory, path: Path) -> None:
        arrays: dict[str, Any] = {name: getattr(traj, name) for name in ARRAYS}
        meta = json.dumps({"format": FORMAT, **traj.meta}, sort_keys=True).encode()
        arrays["meta_json"] = np.frombuffer(meta, dtype=np.uint8)
        with path.open("wb") as fh:
            (np.savez_compressed if compress else np.savez)(fh, **arrays)

    return write


def _zipfile_writestr(traj: Trajectory, path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in honest_contents(traj).items():
            zf.writestr(name, data)


WRITERS = {
    "Trajectory.save_npz": _save_npz,
    "np.savez": _np_savez(compress=False),
    "np.savez_compressed": _np_savez(compress=True),
    "zipfile.writestr": _zipfile_writestr,
}


@pytest.mark.parametrize("writer", WRITERS.values(), ids=WRITERS.keys())
def test_honest_files_decode_alike_in_every_reader(
    tmp_path: Path, writer: Callable[[Trajectory, Path], None]
) -> None:
    traj = make_traj(5)
    path = tmp_path / "traj_000.npz"
    writer(traj, path)
    loaded = load_npz(path, max_steps=5)
    np.testing.assert_array_equal(loaded.qpos, traj.qpos)
    assert loaded.meta == traj.meta
    assert_decoded_alike(path)
    assert set(streamed_members(path.read_bytes())) == {f"{n}.npy".encode() for n in ARRAYS} | {
        b"meta_json.npy"
    }


# ------------------------------------------------------------------ the gate's polyglot


def polyglot(fake: bytes, honest: bytes) -> bytes:
    """``fake``'s members, padded so its central directory lands where ``honest``'s end record
    says the directory is, then ``honest`` whole: ``zipfile`` skips the prefix and reads
    ``honest``, while fflate and a streaming reader read ``fake``."""
    *_, fake_directory_bytes, fake_offset, _ = END.unpack_from(fake, len(fake) - END.size)
    *_, honest_offset, _ = END.unpack_from(honest, len(honest) - END.size)
    assert fake_offset <= honest_offset
    body = fake[:fake_offset].ljust(honest_offset, b"\0")
    return body + fake[fake_offset : fake_offset + fake_directory_bytes] + honest


def test_a_fake_archive_prepended_to_an_honest_one_is_refused(tmp_path: Path) -> None:
    """The gate's reproduction: a short episode in front of a long one. Without the layout rule,
    ``zipfile`` (so verification) saw the honest episode, every other decoder the fake one."""
    fake, honest = make_traj(3, seed=1), make_traj(40, seed=2)
    fake.save_npz(tmp_path / "fake.npz")
    honest.save_npz(tmp_path / "honest.npz")
    raw = polyglot((tmp_path / "fake.npz").read_bytes(), (tmp_path / "honest.npz").read_bytes())
    path = tmp_path / "traj_000.npz"
    path.write_bytes(raw)
    # The parser differential the layout rule closes:
    honest_qpos = npy_bytes(honest.qpos)
    assert zipfile_members(path)[b"qpos.npy"] == honest_qpos
    for decoded in (streamed_members(raw), directory_members(raw)):
        assert decoded[b"qpos.npy"] == npy_bytes(fake.qpos) != honest_qpos
    for read in (lambda p: load_npz(p, max_steps=1000), read_meta):
        with pytest.raises(ValueError, match=r"traj_000\.npz: .*central directory is not where"):
            read(path)


# ------------------------------------------------------------------ crafted layouts


@dataclass
class Member:
    name: bytes
    content: bytes
    deflate: bool = False
    zip64: bool = False  # sizes in a local zip64 extra field, as numpy writes them
    flags: int = 0
    padding: bytes = b""  # stored after the content, inside the member's compressed size
    cut: int = 0  # data bytes dropped from the end of the (deflated) content
    unfinished: bool = False  # a deflate stream flushed without its final block
    hidden: bytes = b""  # deflated after the content, but in neither its size nor its CRC
    gap: bytes = b""  # bytes before the local header
    local: dict[str, Any] = field(default_factory=dict)  # local header overrides
    central: dict[str, Any] = field(default_factory=dict)  # directory entry overrides


def _deflated(data: bytes, *, unfinished: bool = False) -> bytes:
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    end = zlib.Z_SYNC_FLUSH if unfinished else zlib.Z_FINISH
    return compressor.compress(data) + compressor.flush(end)


def build(
    members: list[Member],
    *,
    prefix: bytes = b"",
    before_directory: bytes = b"",
    directory_tail: bytes = b"",
    end: dict[str, int] | None = None,
) -> bytes:
    """A zip archive written field by field, so every field can be made to lie. Offsets are
    relative to the archive's own start, after ``prefix`` (a concatenated archive)."""
    out = bytearray(prefix)
    base = len(prefix)
    entries: list[dict[str, Any]] = []
    for m in members:
        payload = m.content + m.hidden
        data = _deflated(payload, unfinished=m.unfinished) if m.deflate else payload
        data = data[: len(data) - m.cut] + m.padding
        common = {
            "flags": m.flags,
            "method": zipfile.ZIP_DEFLATED if m.deflate else zipfile.ZIP_STORED,
            "crc": zlib.crc32(m.content),
            "compressed": len(data),
            "size": len(m.content),
            "name": m.name,
        }
        out += m.gap
        offset = len(out) - base
        local = {**common, "signature": b"PK\x03\x04", "extra": b""}
        if m.zip64:
            extra = ZIP64_EXTRA.pack(1, 16, len(m.content), len(data))
            local.update(compressed=UNKNOWN, size=UNKNOWN, extra=extra)
        local.update(m.local)
        out += LOCAL.pack(
            local["signature"],
            local.get("version", 45 if m.zip64 else 20),
            local["flags"],
            local["method"],
            0,
            0x21,
            local["crc"],
            local["compressed"],
            local["size"],
            len(local["name"]),
            len(local["extra"]),
        )
        out += local["name"] + local["extra"] + data
        entries.append(
            {
                **common,
                "version": 45 if m.zip64 else 20,
                "offset": offset,
                "extra": b"",
                "comment": b"",
                **m.central,
            }
        )
    out += before_directory
    directory = bytearray()
    for e in entries:
        directory += CENTRAL.pack(
            b"PK\x01\x02",
            20,
            e["version"],
            e["flags"],
            e["method"],
            0,
            0x21,
            e["crc"],
            e["compressed"],
            e["size"],
            len(e["name"]),
            len(e["extra"]),
            len(e["comment"]),
            0,
            0,
            0,
            e["offset"],
        )
        directory += e["name"] + e["extra"] + e["comment"]
    directory += directory_tail
    record = {
        "disk_entries": len(entries),
        "entries": len(entries),
        "size": len(directory),
        "offset": len(out) - base,
        **(end or {}),
    }
    out += directory
    out += END.pack(
        b"PK\x05\x06",
        0,
        0,
        record["disk_entries"],
        record["entries"],
        record["size"],
        record["offset"],
        0,
    )
    return bytes(out)


def members_of(traj: Trajectory, **options: Any) -> list[Member]:
    return [Member(name.encode(), data, **options) for name, data in honest_contents(traj).items()]


def first(edit: Callable[[Member], None]) -> Callable[[list[Member]], None]:
    """Apply ``edit`` to the first member (``qpos.npy``)."""

    def apply(members: list[Member]) -> None:
        edit(members[0])

    return apply


def fourth(edit: Callable[[Member], None]) -> Callable[[list[Member]], None]:
    def apply(members: list[Member]) -> None:
        edit(members[3])

    return apply


def last(edit: Callable[[Member], None]) -> Callable[[list[Member]], None]:
    """Apply ``edit`` to the last member (``meta_json.npy``, which ``read_meta`` reads too)."""

    def apply(members: list[Member]) -> None:
        edit(members[-1])

    return apply


def _set(**changes: Any) -> Callable[[Member], None]:
    def edit(member: Member) -> None:
        for key, value in changes.items():
            setattr(member, key, value)

    return edit


def _override(where: str, **changes: Any) -> Callable[[Member], None]:
    def edit(member: Member) -> None:
        getattr(member, where).update(changes)

    return edit


def _stored_with_padding(member: Member) -> None:
    """A stored member with 4 bytes after its content, declared in its compressed size only:
    ``zipfile`` stops at the uncompressed size, fflate returns every compressed byte."""
    member.padding = b"\0\0\0\0"


def _foreign_extra(member: Member) -> None:
    """A local extra field other than zip64 (here Info-ZIP's Unicode Path, which libarchive
    honours as the member's name)."""
    member.zip64 = True
    member.local["extra"] = struct.pack("<2H", 0x7075, 16) + b"\x01" + b"\0" * 15


def _zip64_sizes(size_delta: int, compressed_delta: int) -> Callable[[Member], None]:
    def edit(member: Member) -> None:
        member.zip64 = True
        data = len(member.content)
        member.local["extra"] = ZIP64_EXTRA.pack(1, 16, data + size_delta, data + compressed_delta)

    return edit


def _data_past_the_end(members: list[Member]) -> None:
    """``qpos.npy`` claims a megabyte of deflated data, and ``qvel.npy`` sits where that would
    end, far past the end of the file."""
    claimed = 10**6
    members[0].deflate = True
    members[0].local["compressed"] = members[0].central["compressed"] = claimed
    members[1].central["offset"] = LOCAL.size + len(members[0].name) + claimed


def _wrong_crc(member: Member) -> None:
    """Both headers agree on a CRC-32 that the content does not have."""
    member.local["crc"] = member.central["crc"] = zlib.crc32(member.content) ^ 1


def _extra_with_known_sizes(member: Member) -> None:
    member.local["extra"] = ZIP64_EXTRA.pack(1, 16, len(member.content), len(member.content))


REFUSED: dict[str, tuple[Callable[[list[Member]], None], dict[str, Any], str]] = {
    "bytes before the first member": (
        first(_set(gap=b"\0" * 8)),
        {},
        r"qpos\.npy: its local header is not at the start of the file",
    ),
    "bytes between two members": (
        fourth(_set(gap=b"junk")),
        {},
        r"action\.npy: its local header is not where the previous member ends",
    ),
    "bytes before the central directory": (
        lambda members: None,
        {"before_directory": b"junk"},
        r"the zip central directory does not start where the last member ends",
    ),
    "an archive behind a prefix": (
        lambda members: None,
        {"prefix": b"\0" * 64},
        r"the zip central directory is not where its end record says",
    ),
    "a NUL in a name": (
        first(_set(name=b"qpos.npy\x00../../evil")),
        {},
        r"unexpected members \['qpos\.npy', 'qpos\.npy\\x00\.\./\.\./evil'\]",
    ),
    "a backslash in a name": (
        first(_set(name=b"..\\qpos.npy")),
        {},
        r"unexpected members",
    ),
    "a local name that differs": (
        first(_override("local", name=b"qvel.npy")),
        {},
        r"qpos\.npy: the local header does not match the central directory \(name\)",
    ),
    "local flags that differ": (
        first(_override("local", flags=0x800)),
        {},
        r"qpos\.npy: the local header does not match the central directory \(flags\)",
    ),
    "a local method that differs": (
        first(_override("local", method=zipfile.ZIP_DEFLATED)),
        {},
        r"qpos\.npy: the local header does not match the central directory \(method\)",
    ),
    "a local CRC that differs": (
        first(_override("local", crc=1)),
        {},
        r"qpos\.npy: the local header does not match the central directory \(CRC-32\)",
    ),
    "local sizes that differ": (
        first(_override("local", size=5)),
        {},
        r"qpos\.npy: the local header does not match the central directory \(size\)",
    ),
    "zip64 sizes that differ": (
        first(_zip64_sizes(1, 0)),
        {},
        r"qpos\.npy: the local header does not match the central directory \(size\)",
    ),
    "a zip64 compressed size that differs": (
        first(_zip64_sizes(0, -1)),
        {},
        r"qpos\.npy: the local header does not match the central directory \(compressed size\)",
    ),
    "a local extra field that is not zip64": (
        first(_foreign_extra),
        {},
        r"qpos\.npy: unexpected extra field in the local header",
    ),
    "an extra field beside sizes in the header": (
        first(_extra_with_known_sizes),
        {},
        r"qpos\.npy: unexpected extra field in the local header",
    ),
    "a version needed beyond 4.5": (
        first(_override("central", version=63)),
        {},
        r"qpos\.npy: needs zip 6\.3 to extract, more than 4\.5",
    ),
    "a local version needed beyond 4.5": (
        first(_override("local", version=46)),
        {},
        r"qpos\.npy: the local header needs zip 4\.6 to extract, more than 4\.5",
    ),
    "a data descriptor": (
        first(_set(flags=0x8)),
        {},
        r"qpos\.npy: data descriptors are not allowed",
    ),
    "an unknown flag": (
        first(_set(flags=0x10)),
        {},
        r"qpos\.npy: general purpose flags 0x0010 are not allowed",
    ),
    "a directory extra field": (
        first(_override("central", extra=ZIP64_EXTRA.pack(1, 16, 0, 0))),
        {},
        r"qpos\.npy: extra fields and comments in the central directory are not allowed",
    ),
    "a directory comment": (
        first(_override("central", comment=b"hi")),
        {},
        r"qpos\.npy: extra fields and comments in the central directory are not allowed",
    ),
    "a stored member longer than its size": (
        first(_stored_with_padding),
        {},
        r"qpos\.npy: a stored member's compressed and uncompressed sizes differ",
    ),
    "junk after the last directory entry": (
        lambda members: None,
        {"directory_tail": b"\0" * 10},
        r"truncated zip central directory",
    ),
    "a directory entry cut short": (
        lambda members: None,
        {"directory_tail": CENTRAL.pack(b"PK\x01\x02", 20, 20, *[0] * 7, 100, *[0] * 6)},
        r"truncated zip central directory",
    ),
    "a malformed directory entry": (
        lambda members: None,
        {"directory_tail": b"\0" * CENTRAL.size},
        r"malformed zip central directory",
    ),
    "no local header where the directory points": (
        first(_override("local", signature=b"PK\x03\x05")),
        {},
        r"qpos\.npy: no local header where the central directory points",
    ),
    "a member's data past the end of the file": (
        _data_past_the_end,
        {},
        r"qvel\.npy: truncated local header",
    ),
    "a deflate stream cut short": (
        last(_set(deflate=True, cut=4)),
        {},
        r"meta_json: the member holds less data than its size declares",
    ),
    "a deflate stream without its end": (
        last(_set(deflate=True, unfinished=True)),
        {},
        r"meta_json\.npy: the deflate stream is cut short",
    ),
    "bytes after a deflate stream": (
        last(_set(deflate=True, padding=b"junk")),
        {},
        r"meta_json\.npy: the deflate stream ends before the member's data does",
    ),
    "a deflate stream longer than its size": (
        last(_set(deflate=True, hidden=b"surprise")),
        {},
        r"meta_json\.npy: the member inflates to more than its declared size",
    ),
    "content that is not its CRC": (
        last(_wrong_crc),
        {},
        r"meta_json\.npy: the member's content does not match its CRC-32",
    ),
    "entry counts that disagree": (
        lambda members: None,
        {"end": {"disk_entries": 10}},
        r"the zip end record's entry counts disagree",
    ),
    "an entry count the directory does not hold": (
        lambda members: None,
        {"end": {"disk_entries": 12, "entries": 12}},
        r"the zip end record lists 12 members, its central directory 11",
    ),
}


@pytest.mark.parametrize(("edit", "options", "message"), REFUSED.values(), ids=list(REFUSED.keys()))
@pytest.mark.parametrize(
    "read",
    [lambda path: load_npz(path, max_steps=5), read_meta],
    ids=["load_npz", "read_meta"],
)
def test_every_departure_from_the_canonical_layout_is_refused(
    tmp_path: Path,
    edit: Callable[[list[Member]], None],
    options: dict[str, Any],
    message: str,
    read: Callable[[Path], object],
) -> None:
    members = members_of(make_traj(5))
    edit(members)
    path = tmp_path / "odd.npz"
    path.write_bytes(build(members, **options))
    with pytest.raises(ValueError, match=rf"odd\.npz: {message}"):
        read(path)


@pytest.mark.parametrize(
    "options",
    [{}, {"deflate": True}, {"zip64": True}, {"deflate": True, "zip64": True}, {"flags": 0x0806}],
    ids=["stored", "deflated", "zip64", "deflated zip64", "tolerated flags"],
)
def test_the_builder_writes_canonical_archives_that_load(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    """The crafted cases above differ from these in one field each: what the loader refuses is
    that field, not the builder. Bits 1-2 (deflate options) and 11 (UTF-8 names) change no
    decoder's reading of a member, so they are tolerated."""
    traj = make_traj(5)
    path = tmp_path / "built.npz"
    path.write_bytes(build(members_of(traj, **options)))
    np.testing.assert_array_equal(load_npz(path, max_steps=5).ctrl, traj.ctrl)
    assert_decoded_alike(path)


# ------------------------------------------------------------------ the property


@pytest.fixture(scope="module")
def small_archive(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("small") / "traj_000.npz"
    make_traj(2).save_npz(path)
    return path.read_bytes()


@settings(
    max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(position=st.floats(0.0, 1.0, exclude_max=True), value=st.integers(0, 255))
def test_whatever_still_loads_after_a_byte_change_decodes_alike(
    tmp_path: Path, small_archive: bytes, position: float, value: int
) -> None:
    """Change any one byte of an honest file: either the loader refuses it, or every reader
    still decodes exactly the members ``zipfile`` does."""
    raw = bytearray(small_archive)
    raw[int(position * len(raw))] = value
    path = tmp_path / "mutant.npz"
    path.write_bytes(bytes(raw))
    try:
        load_npz(path, max_steps=2)
    except ValueError:
        return
    assert_decoded_alike(path)


# ------------------------------------------------------------------ the member reader


def _archive(raw: bytes) -> tuple[io.BytesIO, Archive]:
    fh = io.BytesIO(raw)
    names = frozenset(name.decode() for name in streamed_members(raw))
    return fh, Archive(fh, len(raw), names, max_directory_bytes=16384, max_member_bytes=1 << 20)


def test_a_member_is_finished_only_once_read_to_its_end() -> None:
    _, archive = _archive(build(members_of(make_traj(5), deflate=True)))
    member = archive.open("qpos.npy")
    assert len(member.read(10)) == 10
    with pytest.raises(ValueError, match=r"qpos\.npy: the member was not read to its end"):
        member.finish()
    member.read(member.size)
    member.finish()


def test_a_member_whose_data_vanished_after_the_directory_was_read() -> None:
    """The layout is checked once; a file cut short later fails the read instead of yielding
    fewer bytes."""
    fh, archive = _archive(build(members_of(make_traj(5))))
    fh.truncate(archive.members["ctrl.npy"].start + 3)
    with pytest.raises(ValueError, match=r"ctrl\.npy: the member's data is cut short"):
        archive.open("ctrl.npy").read(64)
