"""Identifier grammar for challenges, protocols and Gymnasium ids (docs/contract.md §1)."""

from __future__ import annotations

import re
from collections.abc import Mapping

_CHALLENGE_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*-v[1-9][0-9]*$")
_PRESET_RE = re.compile(r"^[a-z][a-z0-9_]*$")

PROTOCOL_ALIASES: Mapping[str, str] = {
    "eval_v1": "balance-v1/standard",
    "eval_v1_clean": "balance-v1/clean",
}


def validate_challenge_id(challenge_id: str) -> str:
    if not _CHALLENGE_RE.match(challenge_id):
        raise ValueError(f"invalid challenge id {challenge_id!r}: expected e.g. 'balance-v1'")
    return challenge_id


def gym_id(challenge_id: str) -> str:
    """``balance-v1`` -> ``topple/Balance-v1``."""
    validate_challenge_id(challenge_id)
    name, _, version = challenge_id.rpartition("-")
    pascal = "".join(part[:1].upper() + part[1:] for part in name.split("-"))
    return f"topple/{pascal}-{version}"


def parse_protocol(protocol_id: str) -> tuple[str, str]:
    """``balance-v1/standard`` -> ``("balance-v1", "standard")``."""
    parts = protocol_id.split("/")
    if len(parts) != 2 or not _PRESET_RE.match(parts[1]):
        raise ValueError(f"invalid protocol id {protocol_id!r}: expected '<challenge>/<preset>'")
    try:
        validate_challenge_id(parts[0])
    except ValueError as exc:
        raise ValueError(f"invalid protocol id {protocol_id!r}: {exc}") from exc
    return parts[0], parts[1]


def resolve_protocol(protocol_id: str) -> str:
    """Expand permanent aliases (``eval_v1``) and validate the result."""
    resolved = PROTOCOL_ALIASES.get(protocol_id, protocol_id)
    parse_protocol(resolved)
    return resolved
