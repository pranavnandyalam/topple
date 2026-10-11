"""``topple.results/1``: schema bounds, cross-field checks, error details, atomic writes (§15)."""

import errno
import os
import stat
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

import topple.results as results_module
from topple.results import (
    MAX_DETAIL_CHARS,
    ControllerInfo,
    EpisodeResult,
    ErrorInfo,
    EvalResults,
    read_results,
    sanitize_detail,
    write_results,
)

HEX = "ab" * 32


def episode(index: int, *, seed: int | None = 0, error: str | None = None) -> dict[str, Any]:
    return {
        "index": index,
        "seed": seed,
        "success": error is None,
        "score": 0.0 if error else 90.0,
        "metrics": None if error == "missing" else {"time_upright_s": 10.0},
        "error": None if error is None else {"code": error, "step": 0, "detail": ""},
    }


def results_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "challenge": "balance-v1",
        "protocol": "balance-v1/standard",
        "protocol_hash": HEX,
        "seed_set": "public",
        "versions": {
            "topple": "0.1.0",
            "mujoco": "3.15.0",
            "numpy": "2.5.3",
            "obs_schema": "balancer.obs.v1",
            "action_spec": "diffdrive.effort.v1",
        },
        "platform": "darwin-arm64",
        "robot_hash": HEX,
        "controller": {"id": "pid", "sha256": HEX},
        "isolation": "inprocess",
        "complete": True,
        "episodes": [episode(0), episode(1, seed=5)],
        "aggregate": {
            "score": 90.0,
            "success_rate": 1.0,
            "mean_time_upright_s": 10.0,
            "episodes": 2,
            "errors": 0,
        },
    }
    data.update(overrides)
    return data


def test_valid_results_parse() -> None:
    results = EvalResults.model_validate(results_dict())
    assert results.complete
    assert results.episodes is not None
    assert len(results.episodes) == 2


@pytest.mark.parametrize("code", ["incomplete", "not_reproducible", "missing"])
def test_verifier_error_codes_exist(code: str) -> None:
    assert ErrorInfo(code=code, step=3).code == code  # type: ignore[arg-type]


def test_rankable_is_gone() -> None:
    """Rankability is decided by which trusted job produced a run (Phase 3), not by the file."""
    assert not hasattr(EvalResults, "rankable")


# ------------------------------------------------------------------ string bounds


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("controller", "id"), "x" * 201),
        (("controller", "id"), ""),
        (("controller", "sha256"), "0" * 63),
        (("controller", "sha256"), "G" * 64),
        (("controller", "sha256"), HEX.upper()),
        (("platform",), "p" * 65),
        (("protocol_hash",), "not-a-hash"),
        (("robot_hash",), HEX + "0"),
        (("robot_hash",), HEX + "\n"),
        (("challenge",), "c" * 201),
        (("protocol",), "p" * 201),
        (("versions", "topple"), "1" * 65),
    ],
)
def test_string_bounds(path: tuple[str, ...], value: str) -> None:
    data = results_dict()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        EvalResults.model_validate(data)


def test_string_bounds_accept_the_limits() -> None:
    data = results_dict(platform="p" * 64)
    data["controller"]["id"] = "x" * 200
    assert EvalResults.model_validate(data).controller.id == "x" * 200


def test_controller_info_id_bound() -> None:
    with pytest.raises(ValidationError):
        ControllerInfo(id="x" * 201, sha256=HEX)


# ------------------------------------------------------------------ cross-field checks


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"episodes": None}, "public"),
        ({"seed_set": "private"}, "private"),
        ({"episodes": [episode(0)]}, "aggregate.episodes"),
        ({"episodes": [episode(1), episode(0)]}, "index"),
        ({"episodes": [episode(0), episode(0)]}, "index"),
        ({"episodes": [episode(0), episode(1, seed=None)]}, "seed"),
        ({"episodes": [episode(0), episode(1, error="exception")]}, "errors"),
    ],
)
def test_cross_field_checks(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EvalResults.model_validate(results_dict(**overrides))


def test_private_results_carry_no_episodes() -> None:
    results = EvalResults.model_validate(results_dict(seed_set="private", episodes=None))
    assert results.episodes is None


def test_errors_count_matches_errored_episodes() -> None:
    data = results_dict(episodes=[episode(0), episode(2, seed=9, error="missing")])
    data["aggregate"]["errors"] = 1
    results = EvalResults.model_validate(data)
    assert results.aggregate.errors == 1


# ------------------------------------------------------------------ error details (spec H)


def test_detail_replaces_lone_surrogates_and_control_characters() -> None:
    raw = "a\ud800b\x00c\x1bd\x7fe\x85f\ng\th"
    clean = ErrorInfo(code="exception", step=0, detail=raw).detail
    assert clean == "a�b�c�d�e�f\ng\th"


def test_detail_is_truncated() -> None:
    assert len(ErrorInfo(code="exception", step=0, detail="x" * 5000).detail) == MAX_DETAIL_CHARS


def _is_clean(text: str) -> bool:
    return (
        all(ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cs") for ch in text)
        and len(text) <= MAX_DETAIL_CHARS
    )


@settings(max_examples=200, deadline=None)
@given(st.lists(st.integers(0, 0x10FFFF).map(chr), max_size=400).map("".join))
def test_sanitize_detail_property(raw: str) -> None:
    clean = sanitize_detail(raw)
    assert _is_clean(clean)
    assert sanitize_detail(clean) == clean  # idempotent: re-validating a stored error is a no-op
    assert len(clean) == min(len(raw), MAX_DETAIL_CHARS)  # one character in, one character out
    info = ErrorInfo(code="exception", step=1, detail=raw)
    assert ErrorInfo.model_validate_json(info.model_dump_json()) == info


def test_hostile_detail_survives_write_and_read(tmp_path: Path) -> None:
    data = results_dict(episodes=[episode(0), episode(1, seed=5, error="exception")])
    data["episodes"][1]["error"]["detail"] = "\ud83d boom \x00 ‮"
    data["aggregate"]["errors"] = 1
    results = EvalResults.model_validate(data)
    path = tmp_path / "results.json"
    write_results(results, path)
    assert read_results(path) == results


# ------------------------------------------------------------------ atomic write (spec H)


def test_results_file_is_world_readable_whatever_the_umask(tmp_path: Path) -> None:
    old = os.umask(0o077)
    try:
        write_results(EvalResults.model_validate(results_dict()), tmp_path / "results.json")
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "results.json").stat().st_mode) == 0o644


def test_results_are_fsynced_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def replace(src: str, dst: Path) -> None:
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(results_module.os, "fsync", fsync)
    monkeypatch.setattr(results_module.os, "replace", replace)
    write_results(EvalResults.model_validate(results_dict()), tmp_path / "results.json")
    assert events[:2] == ["fsync", "replace"]


def test_the_directory_is_fsynced_after_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename itself is only durable once the directory entry reaches the disk."""
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        events.append("fsync dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync file")
        real_fsync(fd)

    def replace(src: str, dst: Path) -> None:
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(results_module.os, "fsync", fsync)
    monkeypatch.setattr(results_module.os, "replace", replace)
    write_results(EvalResults.model_validate(results_dict()), tmp_path / "results.json")
    assert events == ["fsync file", "replace", "fsync dir"]


@pytest.mark.parametrize(
    # ENOTSUP and EOPNOTSUPP are one number on Linux but two on macOS and the BSDs.
    "code",
    sorted({errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}),
)
def test_a_directory_fsync_the_filesystem_does_not_support_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(code, os.strerror(code))
        real_fsync(fd)

    monkeypatch.setattr(results_module.os, "fsync", fsync)
    path = tmp_path / "results.json"
    write_results(EvalResults.model_validate(results_dict()), path)
    assert read_results(path).challenge == results_dict()["challenge"]


@pytest.mark.parametrize("code", [errno.EACCES, errno.EPERM])
def test_a_directory_that_cannot_be_opened_is_not_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """The rename has happened by then: a directory we may not open cannot be synced, like one
    whose file system does not support it, and the write succeeds."""
    real_open = os.open

    def open_(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if os.path.isdir(path):
            raise OSError(code, os.strerror(code))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(results_module.os, "open", open_)
    path = tmp_path / "results.json"
    write_results(EvalResults.model_validate(results_dict()), path)
    assert read_results(path).challenge == results_dict()["challenge"]


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions that apply to this user",
)
def test_results_can_be_written_into_a_write_only_drop_directory(tmp_path: Path) -> None:
    drop = tmp_path / "drop"
    drop.mkdir()
    drop.chmod(0o300)  # write and search, but not read: the directory cannot be opened
    try:
        write_results(EvalResults.model_validate(results_dict()), drop / "results.json")
    finally:
        drop.chmod(0o700)
    assert read_results(drop / "results.json").challenge == results_dict()["challenge"]


def test_any_other_failure_to_open_the_directory_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only "cannot open" is skipped; any other failure to open the directory still raises."""
    real_open = os.open

    def open_(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if os.path.isdir(path):
            raise OSError(errno.EIO, "I/O error")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(results_module.os, "open", open_)
    with pytest.raises(OSError, match="I/O error"):
        write_results(EvalResults.model_validate(results_dict()), tmp_path / "results.json")


def test_results_are_written_where_os_has_no_fchmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python 3.12 has no ``os.fchmod`` on Windows: the mode is set through the path instead."""
    monkeypatch.delattr(results_module.os, "fchmod")
    old = os.umask(0o077)
    try:
        write_results(EvalResults.model_validate(results_dict()), tmp_path / "results.json")
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "results.json").stat().st_mode) == 0o644


def test_a_failing_directory_fsync_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "I/O error")
        real_fsync(fd)

    monkeypatch.setattr(results_module.os, "fsync", fsync)
    with pytest.raises(OSError, match="I/O error"):
        write_results(EvalResults.model_validate(results_dict()), tmp_path / "results.json")


def test_failed_write_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(results_module.os, "fsync", broken)
    with pytest.raises(OSError, match="disk full"):
        write_results(EvalResults.model_validate(results_dict()), tmp_path / "results.json")
    assert list(tmp_path.iterdir()) == []


def test_episode_result_score_bounds() -> None:
    with pytest.raises(ValidationError):
        EpisodeResult.model_validate({**episode(0), "score": 100.5})
