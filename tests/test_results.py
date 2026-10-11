"""``topple.results/1``: schema bounds, cross-field checks, error details, atomic writes (§15)."""

import errno
import json
import math
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
    MAX_EPISODES,
    MAX_METRIC_CHARS,
    MAX_METRICS,
    MAX_TIME_UPRIGHT_S,
    ControllerInfo,
    EpisodeResult,
    ErrorInfo,
    EvalResults,
    read_results,
    sanitize_detail,
    sanitize_id,
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
    data = results_dict(episodes=[episode(0), episode(2, seed=9, error="missing")], complete=False)
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
    data["episodes"][1]["error"]["detail"] = "\ud83d boom \x00 \u202e"
    data["aggregate"].update(score=45.0, success_rate=0.5, errors=1)
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
    """Only "cannot open" is skipped; any other failure to open the directory still raises, and
    it raises after the rename (§15): the complete file is in place, no temporary is left."""
    real_open = os.open

    def open_(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if os.path.isdir(path):
            raise OSError(errno.EIO, "I/O error")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(results_module.os, "open", open_)
    with pytest.raises(OSError, match="I/O error"):
        write_results(EvalResults.model_validate(results_dict()), tmp_path / "results.json")
    assert read_results(tmp_path / "results.json").challenge == results_dict()["challenge"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["results.json"]


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


# ------------------------------------------------------------------ forged content (§15)


@pytest.mark.parametrize("changes", [{"score": 100.0}, {"success": True}])
def test_an_episode_with_an_error_scores_0_and_is_no_success(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="an episode with an error scores 0"):
        EpisodeResult.model_validate({**episode(0, error="exception"), **changes})


def test_a_complete_run_has_no_missing_episode() -> None:
    data = results_dict(episodes=[episode(0), episode(1, seed=5, error="missing")])
    data["aggregate"].update(score=45.0, success_rate=0.5, errors=1)
    with pytest.raises(ValidationError, match="missing"):
        EvalResults.model_validate(data)
    data["complete"] = False
    assert not EvalResults.model_validate(data).complete


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", 100.0),
        ("score", 90.00000000000001),
        ("success_rate", 0.5),
        ("mean_time_upright_s", 9.0),
    ],
)
def test_a_complete_aggregate_is_the_aggregate_of_its_episodes(field: str, value: float) -> None:
    """§12: score and success rate over N = the episodes listed, mean time upright over those
    with metrics. A complete run's N is in the file, so its aggregate can be checked; an
    incomplete one's N is the protocol's, which the file does not hold."""
    data = results_dict()
    data["aggregate"][field] = value
    with pytest.raises(ValidationError, match="aggregate does not match"):
        EvalResults.model_validate(data)
    data["complete"] = False
    assert EvalResults.model_validate(data).aggregate.model_dump()[field] == value


def test_the_aggregate_is_checked_exactly_as_section_12_sums_it() -> None:
    """``fsum`` over N, as ``summarize`` computes it; Python 3.12's ``sum`` is compensated, so
    it takes an awkward case to tell the two apart (the same case as ``test_evaluate``)."""
    scores = [1.0, 1.0, 5e-324, 2.0**-52]
    assert math.fsum(scores) / 5 != sum(scores) / 5
    episodes = [{**episode(i, seed=i), "score": s} for i, s in enumerate(scores)]
    errored = {**episode(4, seed=4, error="exception"), "metrics": {"time_upright_s": 0.25}}
    data = results_dict(episodes=[*episodes, errored])
    data["aggregate"].update(
        score=sum(scores) / 5,
        success_rate=4 / 5,
        mean_time_upright_s=math.fsum([10.0] * 4 + [0.25]) / 5,
        episodes=5,
        errors=1,
    )
    with pytest.raises(ValidationError, match="aggregate does not match"):
        EvalResults.model_validate(data)
    data["aggregate"]["score"] = math.fsum(scores) / 5
    assert EvalResults.model_validate(data).aggregate.score == math.fsum(scores) / 5


@pytest.mark.parametrize(
    "metrics",
    [
        {"drift_m": 0.1},
        {"time_upright_s": True},
        {"time_upright_s": "10"},
        {"time_upright_s": 10**400},
    ],
    ids=["absent", "bool", "str", "huge int"],
)
def test_a_complete_aggregate_needs_every_time_upright(metrics: dict[str, Any]) -> None:
    data = results_dict()
    data["episodes"][1]["metrics"] = metrics
    with pytest.raises(ValidationError, match="time_upright_s"):
        EvalResults.model_validate(data)


@pytest.mark.parametrize(
    ("times", "mean"),
    [
        ((1e308, 1e308), 1e308),
        ((10**308, 10**308), 1e308),
        ((-5.0, 25.0), 10.0),
        ((2e9, 0.0), 1e9),
    ],
    ids=["huge floats", "huge ints", "negative", "above the bound"],
)
def test_a_complete_run_bounds_every_time_upright(
    tmp_path: Path, times: tuple[float, float], mean: float
) -> None:
    """Every ``time_upright_s`` of a complete run is a number of seconds in [0, 1e9]. Two near
    the float limit made §12's sum overflow, an ``OverflowError`` that pydantic passes through
    instead of a ``ValidationError``; a negative time could balance a forged one in the mean."""
    data = results_dict()
    for e, t in zip(data["episodes"], times, strict=True):
        e["metrics"]["time_upright_s"] = t
    data["aggregate"]["mean_time_upright_s"] = mean
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match=r"time_upright_s must be a number of seconds"):
        read_results(path)


@pytest.mark.parametrize("time", [0, 0.0, MAX_TIME_UPRIGHT_S, int(MAX_TIME_UPRIGHT_S)])
def test_a_complete_run_accepts_times_up_to_the_bound(time: float) -> None:
    data = results_dict()
    for e in data["episodes"]:
        e["metrics"]["time_upright_s"] = time
    data["aggregate"]["mean_time_upright_s"] = time
    assert EvalResults.model_validate(data).aggregate.mean_time_upright_s == time


def test_metrics_are_bounded() -> None:
    def with_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        return {**episode(0), "metrics": metrics}

    many = {f"m{i}": 1.0 for i in range(MAX_METRICS)}
    assert EpisodeResult.model_validate(with_metrics(many)).metrics == many
    for bad in (
        {**many, "one_more": 1.0},
        {"k" * (MAX_METRIC_CHARS + 1): 1.0},
        {"": 1.0},
        {"termination": "x" * (MAX_METRIC_CHARS + 1)},
    ):
        with pytest.raises(ValidationError):
            EpisodeResult.model_validate(with_metrics(bad))


def test_the_episode_list_is_bounded() -> None:
    data = results_dict(episodes=[episode(i, seed=i) for i in range(MAX_EPISODES + 1)])
    with pytest.raises(ValidationError, match="at most"):
        EvalResults.model_validate(data)


def test_read_results_reads_at_most_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "results.json"
    write_results(EvalResults.model_validate(results_dict()), path)
    size = path.stat().st_size
    monkeypatch.setattr(results_module, "MAX_RESULTS_BYTES", size - 1)
    with pytest.raises(ValueError, match=rf"results\.json has more than {size - 1} bytes"):
        read_results(path)
    monkeypatch.setattr(results_module, "MAX_RESULTS_BYTES", size)
    assert read_results(path).controller.id == "pid"


UNSAFE_TEXT = "\x1b]0;pwn\x07\u202e\n::error::x"  # a terminal escape, a bidi override, a CI command


@pytest.mark.parametrize(
    "path",
    [
        ("platform",),
        ("versions", "topple"),
        ("versions", "mujoco"),
        ("versions", "numpy"),
        ("versions", "obs_schema"),
        ("versions", "action_spec"),
    ],
)
@pytest.mark.parametrize("text", [UNSAFE_TEXT, "\t", "\u2028", "\u200b", "\x00"])
def test_platform_and_versions_hold_printable_text_only(path: tuple[str, ...], text: str) -> None:
    """§15: every string of an uploaded file is attacker-chosen; besides the sanitized
    ``controller.id`` and ``detail``, the free-text fields refuse what a terminal, a CI log or a
    page would act on (control, format and separator characters but the space)."""
    data = results_dict()
    setter = data
    for key in path[:-1]:
        setter = setter[key]
    setter[path[-1]] = text
    with pytest.raises(ValidationError, match="printable"):
        EvalResults.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("where", ["name", "value"])
@pytest.mark.parametrize("text", [UNSAFE_TEXT, "a\tb", "\u202e", "\x00"])
def test_metric_names_and_string_values_hold_printable_text_only(where: str, text: str) -> None:
    metrics: dict[str, Any] = {"time_upright_s": 10.0, "termination": "timeout"}
    if where == "name":
        metrics[text] = 1.0
    else:
        metrics["termination"] = text
    with pytest.raises(ValidationError, match="printable"):
        EpisodeResult.model_validate_json(json.dumps({**episode(0), "metrics": metrics}))


def test_printable_text_is_kept() -> None:
    data = results_dict(platform="linux-x86_64 (é ok)")
    data["episodes"][0]["metrics"] = {"time_upright_s": 10.0, "termination": "fell", "ünï": 1}
    results = EvalResults.model_validate_json(json.dumps(data))
    assert results.platform == "linux-x86_64 (é ok)"
    assert results.episodes is not None
    assert results.episodes[0].metrics == {"time_upright_s": 10.0, "termination": "fell", "ünï": 1}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("challenge", UNSAFE_TEXT),
        ("challenge", "Balance-v1"),
        ("challenge", "balance"),
        ("challenge", "balance-v1\n"),
        ("protocol", UNSAFE_TEXT),
        ("protocol", "balance-v1"),
        ("protocol", "balance-v1/Standard"),
        ("protocol", "balance-v1/standard/x"),
        ("protocol", "eval_v1"),
    ],
)
def test_challenge_and_protocol_follow_the_section_1_grammar(field: str, value: str) -> None:
    """Results name a challenge and a protocol by their canonical ids (§1), never an alias."""
    with pytest.raises(ValidationError, match=f"invalid {field} id"):
        EvalResults.model_validate_json(json.dumps(results_dict(**{field: value})))


def test_the_controller_id_is_untrusted_text() -> None:
    """The id is a submission's file name: control, format and line-separator characters (a
    bidi override, a line break before ``@maintainers``) become U+FFFD, one for one."""
    raw = "[click](https://evil.example)\n@maintainers\u202e\x1b[2J\u2028\udcff ok é"
    clean = ControllerInfo(id=raw, sha256=HEX).id
    expected = "[click](https://evil.example)\ufffd@maintainers\ufffd\ufffd[2J\ufffd\ufffd ok é"
    assert clean == expected
    assert sanitize_id(clean) == clean
    assert (
        ControllerInfo.model_validate_json(ControllerInfo(id=raw, sha256=HEX).model_dump_json()).id
        == clean
    )
