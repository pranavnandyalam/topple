"""YAML read as plain data, bounded in size and nesting (docs/design/phase-2.md §2.1).

Training configs are YAML. ``yaml.safe_load`` already refuses Python tags, but it still expands
aliases (a few hundred bytes of ``&a [*a, *a, ...]`` "billion laughs" grow without bound) and its
composer recurses once per nesting level, so 64 KiB of ``[`` exhausts the stack, and it keeps the
last of two equal keys without a word. Here the size is capped first, then one pass over the
parser's events (the parser itself is iterative) refuses every anchor and alias, any nesting
deeper than ``max_depth``, a key repeated in one mapping, a merge key (``<<``, or any key with an
explicit tag such as ``!!merge``, which ``SafeLoader`` merges whatever its text: a config is
complete) and a key that is not a plain scalar; only then does ``SafeLoader`` build the data.
Every failure is a ``ConfigError`` (never a ``RecursionError`` or a traceback) of one bounded
line: what the parser quotes from the input is escaped and cut (``clean_text``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import yaml

MAX_YAML_BYTES: Final = 64 * 1024
MAX_YAML_DEPTH: Final = 32
MAX_DETAIL_CHARS: Final = 160
"""Most characters of input-derived text (a parser's problem, a validator's message) quoted in
an error."""
_OPEN: Final = (yaml.SequenceStartEvent, yaml.MappingStartEvent)
_CLOSE: Final = (yaml.SequenceEndEvent, yaml.MappingEndEvent)


class ConfigError(ValueError):
    """A configuration that cannot be used: too large, not UTF-8, not plain YAML data (anchors,
    aliases, tags ``SafeLoader`` refuses), nested too deeply, or invalid for its model."""


def clean_text(text: str, limit: int = MAX_DETAIL_CHARS) -> str:
    """``text`` as one printable line of at most ``limit`` characters: every character that is
    not printable shown as its escape (``\\n``, ``\\x1b``, a lone surrogate as ``\\ud800``),
    so a newline followed by ``::error::`` or a terminal escape in a config never reaches a log
    raw, then cut with ``...``."""
    shown = "".join(
        ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii")
        for ch in text[: limit + 1]
    )
    return shown if len(shown) <= limit else shown[: limit - 3] + "..."


def safe_load_capped(
    data: bytes, cap: int = MAX_YAML_BYTES, max_depth: int = MAX_YAML_DEPTH
) -> object:
    """The YAML document ``data`` (at most ``cap`` bytes of UTF-8) as plain Python data:
    ``ConfigError`` if it is larger, not UTF-8, holds an anchor or alias, nests collections
    more than ``max_depth`` deep or is anything ``yaml.safe_load`` refuses."""
    if type(data) is not bytes:
        raise ConfigError("YAML is read from bytes")
    if len(data) > cap:
        raise ConfigError(f"the YAML has {len(data)} bytes, more than {cap}")
    try:
        text = data.decode("utf-8")
        _scan(text, max_depth)
        return yaml.safe_load(text)
    except ConfigError:
        raise
    except UnicodeDecodeError:
        raise ConfigError("the YAML is not valid UTF-8") from None
    except RecursionError:
        raise ConfigError("the YAML nests too deeply") from None
    except yaml.MarkedYAMLError as exc:
        raise ConfigError(f"not valid YAML data: {_where(exc)}") from None
    except Exception as exc:  # yaml.YAMLError, ValueError (a 5,000-digit int), ...
        raise ConfigError(f"not valid YAML data ({type(exc).__name__})") from None


@dataclass
class _Keys:
    """An open mapping: the keys seen so far and whether the next node is a key."""

    seen: set[str] = field(default_factory=set)
    expect_key: bool = True


def _scan(text: str, max_depth: int) -> None:
    """Refuse anchors, aliases, nesting deeper than ``max_depth`` and repeated, merge or
    non-scalar mapping keys from the event stream."""
    open_collections: list[_Keys | None] = []  # None: a sequence
    for event in yaml.parse(text, Loader=yaml.SafeLoader):
        if isinstance(event, yaml.AliasEvent):
            raise ConfigError("YAML aliases are not allowed")
        if getattr(event, "anchor", None) is not None:
            raise ConfigError("YAML anchors are not allowed")
        if isinstance(event, (yaml.ScalarEvent, *_OPEN)) and open_collections:
            _place(open_collections[-1], event)
        if isinstance(event, _OPEN):
            mapping = isinstance(event, yaml.MappingStartEvent)
            open_collections.append(_Keys() if mapping else None)
            if len(open_collections) > max_depth:
                raise ConfigError(f"the YAML nests more than {max_depth} levels deep")
        elif isinstance(event, _CLOSE):
            open_collections.pop()


def _place(keys: _Keys | None, event: yaml.Event) -> None:
    """A node inside a mapping is a key, then a value: keys are distinct plain scalars."""
    if keys is None:
        return
    if keys.expect_key:
        if not isinstance(event, yaml.ScalarEvent):
            raise ConfigError("YAML mapping keys must be plain scalars")
        if event.tag is not None:  # !!merge x: {...} merges whatever the key's text is
            raise ConfigError(
                "YAML mapping keys take no tag (!!merge, !!str, ...): a config is written out "
                "in full"
            )
        if event.value == "<<":
            raise ConfigError(
                "YAML merge keys (<<) are not allowed: a config is written out in full"
            )
        if event.value in keys.seen:
            raise ConfigError(f"the YAML repeats the key {event.value[:40]!r} in one mapping")
        keys.seen.add(event.value)
    keys.expect_key = not keys.expect_key


def _where(exc: yaml.MarkedYAMLError) -> str:
    """What the parser or constructor found (escaped and cut: it can quote a 60,000-character
    tag) and where, without the excerpt it would quote."""
    problem = clean_text(str(exc.problem or exc.context or "invalid"))
    mark = exc.problem_mark or exc.context_mark
    if mark is None:
        return problem
    return f"{problem} at line {mark.line + 1}, column {mark.column + 1}"
