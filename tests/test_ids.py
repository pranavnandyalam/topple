import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from topple.ids import (
    PROTOCOL_ALIASES,
    gym_id,
    parse_protocol,
    resolve_protocol,
    validate_challenge_id,
)


@pytest.mark.parametrize(
    "cid", ["balance-v1", "waypoint-v1", "sumo-v12", "frc-auto-v1", "so101-reach-v1", "go1-walk-v3"]
)
def test_valid_challenge_ids(cid: str) -> None:
    assert validate_challenge_id(cid) == cid


@pytest.mark.parametrize(
    "cid",
    [
        "Balance-v1",
        "balance",
        "balance-v0",
        "balance_v1",
        "-balance-v1",
        "balance-v1 ",
        "b--x-v1",
        "",
        "balance-v1\n",
        "frc-2024-v1",
        "a-1b-v1",
        "foo-1bar-v1",
    ],
)
def test_invalid_challenge_ids(cid: str) -> None:
    with pytest.raises(ValueError, match="challenge id"):
        validate_challenge_id(cid)


@pytest.mark.parametrize(
    ("cid", "expected"),
    [
        ("balance-v1", "topple/Balance-v1"),
        ("frc-auto-v2", "topple/FrcAuto-v2"),
        ("so101-reach-v1", "topple/So101Reach-v1"),
    ],
)
def test_gym_id(cid: str, expected: str) -> None:
    assert gym_id(cid) == expected


def test_parse_protocol() -> None:
    assert parse_protocol("balance-v1/standard") == ("balance-v1", "standard")


@pytest.mark.parametrize(
    "pid", ["balance-v1", "balance-v1/", "/standard", "balance-v1/Standard", "x/y/z"]
)
def test_parse_protocol_rejects(pid: str) -> None:
    with pytest.raises(ValueError, match="protocol id"):
        parse_protocol(pid)


def test_aliases_resolve() -> None:
    assert resolve_protocol("eval_v1") == "balance-v1/standard"
    assert resolve_protocol("eval_v1_clean") == "balance-v1/clean"
    assert resolve_protocol("balance-v1/standard") == "balance-v1/standard"
    assert set(PROTOCOL_ALIASES) == {"eval_v1", "eval_v1_clean"}


CHALLENGE_GRAMMAR = r"[a-z][a-z0-9]*(-[a-z][a-z0-9]*)*-v[1-9][0-9]*"


def from_gym_id(gid: str) -> str:
    """Inverse of ``gym_id``: split the Pascal name at its capitals."""
    match = re.fullmatch(r"topple/([A-Za-z0-9]+)-(v[1-9][0-9]*)", gid)
    assert match is not None, gid
    name, version = match.groups()
    segments = re.findall(r"[A-Z][a-z0-9]*", name)
    assert "".join(segments) == name, gid
    return "-".join(s.lower() for s in segments) + "-" + version


@given(st.from_regex(CHALLENGE_GRAMMAR, fullmatch=True))
def test_gym_id_is_injective(cid: str) -> None:
    """Every valid id maps to a Gymnasium id that maps back to it, so no two ids collide."""
    assert validate_challenge_id(cid) == cid
    assert from_gym_id(gym_id(cid)) == cid


def test_digit_leading_segments_would_collide_so_they_are_rejected() -> None:
    assert gym_id("foo1bar-v1") == "topple/Foo1bar-v1"
    with pytest.raises(ValueError, match="challenge id"):
        gym_id("foo-1bar-v1")  # would also be topple/Foo1bar-v1


@pytest.mark.parametrize(
    "pid", ["balance-v1/standard\n", "balance-v1\n/standard", "balance-v1/stan\ndard"]
)
def test_parse_protocol_rejects_newlines(pid: str) -> None:
    with pytest.raises(ValueError, match="protocol id"):
        parse_protocol(pid)


def test_resolve_protocol_rejects_trailing_newline() -> None:
    with pytest.raises(ValueError, match="protocol id"):
        resolve_protocol("balance-v1/standard\n")
