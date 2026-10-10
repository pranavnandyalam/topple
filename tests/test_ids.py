import pytest

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
