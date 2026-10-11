"""Zip archives in the one layout numpy writes, read by hand (docs/contract.md §14).

A ``topple.traj/1`` file is a zip archive of ``.npy`` members that browsers and other tools decode
too, and zip readers find members in different ways. Python's ``zipfile`` reads the central
directory from the end of the file and silently skips data before the archive, a browser decoder
such as fflate takes the stored offsets as they are, and a streaming reader such as libarchive
walks the local headers from byte 0. On any layout but the canonical one they can decode
different contents from the same bytes, so a verified file could replay as another one.

This module accepts exactly the layout numpy writes: the members' local headers and data back to
back from byte 0, then the central directory, then the end record, which ends the file; every
local header agreeing with its directory entry; every member's data decoding to exactly its
declared size and CRC-32 (a deflated member is one deflate stream that ends exactly where its
data does). Every reader then sees the same members. Nothing here allocates more than a bounded
read at a time: the directory is size-capped before it is read, and members are inflated at most
``READ_CHUNK`` compressed bytes, and as many output bytes as a read asks for, at a time.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import IO

STORED = 0
DEFLATED = 8
METHODS = {STORED: "stored", DEFLATED: "deflated"}
READ_CHUNK = 64 * 1024
"""Most compressed bytes one read takes from the file."""
_ENCRYPTED = 0x1  # general purpose flag bit 0
_DATA_DESCRIPTOR = 0x8  # bit 3: sizes and CRC follow the data, so the local header has none
_TOLERATED_FLAGS = 0x0806  # bits 1-2 (deflate options) and 11 (UTF-8 names) change no reading
_MAX_VERSION = 45  # zip 4.5 (zip64 extra fields): the most any stored or deflated member needs
_END = struct.Struct("<4s4H2LH")  # end of central directory record, without its comment
_END_SIGNATURE = b"PK\x05\x06"
_ZIP64_LOCATOR = b"PK\x06\x07"
_ZIP64_LOCATOR_BYTES = 20
_ENTRY = struct.Struct("<4s6H3L5H2L")  # central directory entry, before name, extra, comment
_ENTRY_SIGNATURE = b"PK\x01\x02"
_LOCAL = struct.Struct("<4s5H3L2H")  # local file header, before its name and extra field
_LOCAL_SIGNATURE = b"PK\x03\x04"
_ZIP64_SIZES = struct.Struct("<2H2Q")  # the zip64 extra field numpy writes in a local header
_ZIP64_SIZES_ID = 0x0001
_SIZES_IN_ZIP64 = 0xFFFFFFFF  # a local header's sizes when its zip64 extra field holds them
_SHOWN_NAMES = 5  # unexpected member names quoted in an error


@dataclass(frozen=True)
class Member:
    """Where one member's data lies in the file, and what it must decode to."""

    name: str
    start: int  # offset of the member's first data byte
    compressed: int
    size: int
    method: int
    crc: int


@dataclass(frozen=True)
class _EndRecord:
    entries: int
    directory_bytes: int
    directory_offset: int


@dataclass(frozen=True)
class _Entry:
    """A central directory entry, as stored."""

    name: str  # the stored name bytes, decoded one byte per character (latin-1)
    version: int  # needed to extract
    flags: int
    method: int
    crc: int
    compressed: int
    size: int
    offset: int
    extra: bytes
    comment: bytes


class Archive:
    """The members of a canonical zip archive, read from ``fh`` (``size`` bytes), which must be
    exactly ``names``. Raises ``ValueError`` for anything else; reading a member may too."""

    def __init__(
        self,
        fh: IO[bytes],
        size: int,
        names: frozenset[str],
        *,
        max_directory_bytes: int,
        max_member_bytes: int,
    ) -> None:
        self._fh = fh
        end = _read_end_record(fh, size, max_directory_bytes)
        entries = _read_directory(fh, end)
        _check_entries(entries, names, max_member_bytes)
        self.members = _check_layout(fh, entries, end)

    def open(self, name: str) -> MemberReader:
        return MemberReader(self._fh, self.members[name])


# ------------------------------------------------------------------ the directory


def _read_end_record(fh: IO[bytes], size: int, max_directory_bytes: int) -> _EndRecord:
    """The end-of-central-directory record, which must end the file (no archive comment, no
    zip64 records, one disk) and place a directory of at most ``max_directory_bytes`` right
    before itself, so nothing can come before the archive or between its parts."""
    tail = min(size, _END.size + _ZIP64_LOCATOR_BYTES)
    fh.seek(size - tail)
    data = fh.read(tail)
    if len(data) < _END.size:
        raise ValueError("not a zip archive")
    (
        signature,
        disk,
        directory_disk,
        disk_entries,
        entries,
        directory_bytes,
        directory_offset,
        comment,
    ) = _END.unpack(data[-_END.size :])
    if signature != _END_SIGNATURE or comment:
        raise ValueError("not a zip archive whose end-of-central-directory record ends the file")
    if len(data) == _END.size + _ZIP64_LOCATOR_BYTES and data.startswith(_ZIP64_LOCATOR):
        raise ValueError("zip64 archives are not allowed")
    if disk or directory_disk:
        raise ValueError("multi-disk zip archives are not allowed")
    if directory_bytes > max_directory_bytes:
        raise ValueError(
            f"the zip central directory has {directory_bytes} bytes, more than "
            f"{max_directory_bytes}"
        )
    if disk_entries != entries:
        raise ValueError("the zip end record's entry counts disagree")
    if directory_offset + directory_bytes + _END.size != size:
        raise ValueError(
            "the zip central directory is not where its end record says (data before or "
            "inside the archive)"
        )
    return _EndRecord(entries, directory_bytes, directory_offset)


def _read_directory(fh: IO[bytes], end: _EndRecord) -> list[_Entry]:
    """Every entry of the central directory, which must hold exactly the end record's count."""
    fh.seek(end.directory_offset)
    data = fh.read(end.directory_bytes)
    entries: list[_Entry] = []
    at = 0
    while at < len(data):
        if at + _ENTRY.size > len(data):
            raise ValueError("truncated zip central directory")
        fields = _ENTRY.unpack_from(data, at)
        if fields[0] != _ENTRY_SIGNATURE:
            raise ValueError("malformed zip central directory")
        name_length, extra_length, comment_length = fields[10:13]
        name_at = at + _ENTRY.size
        extra_at = name_at + name_length
        comment_at = extra_at + extra_length
        at = comment_at + comment_length
        if at > len(data):
            raise ValueError("truncated zip central directory")
        entries.append(
            _Entry(
                name=data[name_at:extra_at].decode("latin-1"),
                version=fields[2],
                flags=fields[3],
                method=fields[4],
                crc=fields[7],
                compressed=fields[8],
                size=fields[9],
                offset=fields[16],
                extra=data[extra_at:comment_at],
                comment=data[comment_at:at],
            )
        )
    if len(entries) != end.entries:
        raise ValueError(
            f"the zip end record lists {end.entries} members, its central directory {len(entries)}"
        )
    return entries


def _check_entries(entries: list[_Entry], names: frozenset[str], max_member_bytes: int) -> None:
    """Exactly ``names``, by their exact stored bytes, each stored or deflated, unencrypted,
    without a data descriptor or anything in the directory beyond its name, and size-capped."""
    stored = [entry.name for entry in entries]
    if len(set(stored)) != len(stored):
        raise ValueError("duplicate members")
    if set(stored) != names:
        raise ValueError(f"unexpected members {_quoted(sorted(set(stored) ^ names))}")
    for entry in entries:
        name = entry.name
        if entry.method not in METHODS:
            raise ValueError(
                f"{name}: compression method {entry.method} is not allowed "
                f"(only {' or '.join(METHODS.values())})"
            )
        if entry.version > _MAX_VERSION:
            raise ValueError(f"{name}: {_needs(entry.version)}")
        if entry.flags & _ENCRYPTED:
            raise ValueError(f"{name}: encrypted members are not allowed")
        if entry.flags & _DATA_DESCRIPTOR:
            raise ValueError(f"{name}: data descriptors are not allowed")
        if entry.flags & ~_TOLERATED_FLAGS:
            raise ValueError(f"{name}: general purpose flags {entry.flags:#06x} are not allowed")
        if entry.extra or entry.comment:
            raise ValueError(
                f"{name}: extra fields and comments in the central directory are not allowed"
            )
        if entry.size > max_member_bytes:
            raise ValueError(f"{name} is too large ({entry.size} bytes)")
        if entry.method == STORED and entry.compressed != entry.size:
            raise ValueError(f"{name}: a stored member's compressed and uncompressed sizes differ")


def _needs(version: int) -> str:
    return f"needs zip {version // 10}.{version % 10} to extract, more than 4.5"


def _quoted(names: list[str]) -> str:
    """The first few ``names``, escaped and shortened: they come from the untrusted archive."""
    shown = ", ".join(_short_repr(name) for name in names[:_SHOWN_NAMES])
    more = len(names) - _SHOWN_NAMES
    return f"[{shown}]" + (f" and {more} more" if more > 0 else "")


def _short_repr(name: str) -> str:
    text = repr(name)
    return text if len(text) <= 40 else text[:37] + "..."


# ------------------------------------------------------------------ the layout


def _check_layout(fh: IO[bytes], entries: list[_Entry], end: _EndRecord) -> dict[str, Member]:
    """The members back to back from byte 0, in directory order, then the directory: every byte
    of the file has one meaning, the same for every reader."""
    members: dict[str, Member] = {}
    offset = 0
    for entry in entries:
        if entry.offset != offset:
            where = "where the previous member ends" if offset else "at the start of the file"
            raise ValueError(f"{entry.name}: its local header is not {where}")
        start = _check_local_header(fh, entry)
        members[entry.name] = Member(
            entry.name, start, entry.compressed, entry.size, entry.method, entry.crc
        )
        offset = start + entry.compressed
    if offset != end.directory_offset:
        raise ValueError("the zip central directory does not start where the last member ends")
    return members


def _check_local_header(fh: IO[bytes], entry: _Entry) -> int:
    """Check ``entry``'s local header against the directory entry; return where its data starts.

    The sizes are in the header itself or, as numpy writes them, read 0xFFFFFFFF there and sit
    in a zip64 extra field that holds nothing else (any other extra field, such as Info-ZIP's
    Unicode path that some readers take as the member's name, is refused).
    """
    name = entry.name
    fh.seek(entry.offset)
    (
        signature,
        version,
        flags,
        method,
        _,
        _,
        crc,
        compressed,
        size,
        name_length,
        extra_length,
    ) = _LOCAL.unpack(_read_exactly(fh, _LOCAL.size, name))
    if signature != _LOCAL_SIGNATURE:
        raise ValueError(f"{name}: no local header where the central directory points")
    if version > _MAX_VERSION:
        raise ValueError(f"{name}: the local header {_needs(version)}")
    if name_length != len(name):
        raise ValueError(f"{name}: the local header does not match the central directory (name)")
    if extra_length not in (0, _ZIP64_SIZES.size):
        raise ValueError(f"{name}: unexpected extra field in the local header")
    variable = _read_exactly(fh, name_length + extra_length, name)
    if variable[:name_length].decode("latin-1") != name:
        raise ValueError(f"{name}: the local header does not match the central directory (name)")
    if extra_length:
        if (compressed, size) != (_SIZES_IN_ZIP64, _SIZES_IN_ZIP64):
            raise ValueError(f"{name}: unexpected extra field in the local header")
        extra_id, extra_bytes, size, compressed = _ZIP64_SIZES.unpack(variable[name_length:])
        if (extra_id, extra_bytes) != (_ZIP64_SIZES_ID, _ZIP64_SIZES.size - 4):
            raise ValueError(f"{name}: unexpected extra field in the local header")
    local = (flags, method, crc, compressed, size)
    central = (entry.flags, entry.method, entry.crc, entry.compressed, entry.size)
    differing = [
        field
        for field, mine, theirs in zip(
            ("flags", "method", "CRC-32", "compressed size", "size"), local, central, strict=True
        )
        if mine != theirs
    ]
    if differing:
        raise ValueError(
            f"{name}: the local header does not match the central directory "
            f"({', '.join(differing)})"
        )
    start: int = entry.offset + _LOCAL.size + name_length + extra_length
    return start


def _read_exactly(fh: IO[bytes], n: int, name: str) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise ValueError(f"{name}: truncated local header")
    return data


# ------------------------------------------------------------------ member data


class MemberReader:
    """One member's content, decoded by hand from its data in the file.

    A read takes at most ``READ_CHUNK`` compressed bytes from the file at a time and never
    produces more than it asked for, nor anything past the member's declared size, so neither a
    lying size nor a deflate bomb can make it allocate more than the caller's read.
    """

    def __init__(self, fh: IO[bytes], member: Member) -> None:
        self._fh = fh
        self._member = member
        self._next = member.start  # the next data byte to take from the file
        self._left = member.compressed  # data bytes not taken yet
        self._produced = 0
        self._crc = 0
        self._inflater = zlib.decompressobj(-zlib.MAX_WBITS) if member.method == DEFLATED else None

    @property
    def size(self) -> int:
        """The member's declared (uncompressed) size."""
        return self._member.size

    def tell(self) -> int:
        """Bytes of content read so far."""
        return self._produced

    def read(self, n: int) -> bytes:
        """Up to ``n`` bytes of content: fewer only at the declared size or where the data ends."""
        want = min(n, self._member.size - self._produced)
        if want <= 0:
            return b""
        data = self._take(want) if self._inflater is None else self._inflate(self._inflater, want)
        self._produced += len(data)
        self._crc = zlib.crc32(data, self._crc)
        return data

    def finish(self) -> None:
        """Check that the member held exactly its declared content, all of it read: its CRC-32,
        and a deflate stream that ends exactly where the member's data does."""
        name = self._member.name
        if self._produced != self._member.size:
            raise ValueError(f"{name}: the member was not read to its end")
        if self._inflater is not None:
            self._drain(self._inflater)
        if self._crc != self._member.crc:
            raise ValueError(f"{name}: the member's content does not match its CRC-32")

    def _take(self, n: int) -> bytes:
        """The next ``min(n, left)`` data bytes from the file."""
        n = min(n, self._left)
        self._fh.seek(self._next)
        data = self._fh.read(n)
        if len(data) != n:
            raise ValueError(f"{self._member.name}: the member's data is cut short")
        self._next += n
        self._left -= n
        return data

    def _feed(self, inflater: zlib._Decompress) -> bytes:
        """Input for ``inflater``: what it left unconsumed, else the next chunk of data."""
        return inflater.unconsumed_tail or self._take(READ_CHUNK)

    def _inflate(self, inflater: zlib._Decompress, want: int) -> bytes:
        parts: list[bytes] = []
        while want > 0 and not inflater.eof:
            feed = self._feed(inflater)
            if not feed:
                break  # the data ran out before the stream ended: ``finish`` refuses it
            data = inflater.decompress(feed, want)
            parts.append(data)
            want -= len(data)
        return b"".join(parts)

    def _drain(self, inflater: zlib._Decompress) -> None:
        """Run the deflate stream to its end, which must produce nothing more and come exactly at
        the end of the member's data."""
        name = self._member.name
        while not inflater.eof:
            feed = self._feed(inflater)
            if not feed:
                raise ValueError(f"{name}: the deflate stream is cut short")
            if inflater.decompress(feed, 1):
                raise ValueError(f"{name}: the member inflates to more than its declared size")
        if inflater.unused_data or inflater.unconsumed_tail or self._left:
            raise ValueError(f"{name}: the deflate stream ends before the member's data does")
