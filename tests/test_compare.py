"""``topple compare`` and the ``topple.compare/1`` file (docs/design/phase-2.md §5.3, §6
``test_compare``).

Every submission is loaded and validated once in the parent; its episodes run in a pool and
come back as per-seed arrays in seed order; every ordered pair i > j gets a paired bootstrap
row (``a`` = entry i, ``b`` = entry j). The file is read like contract §15's ``results.json``:
a regular file of at most 16 MiB, strict JSON types, no NaN or infinity, every object closed,
``format`` required, every array as long as ``seeds``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import topple.compare as compare_module
from policy_support import write_policy_dir
from pool_support import InlinePool
from topple.cli import app
from topple.compare import (
    FORMAT,
    MAX_ENTRIES,
    MINUS,
    Comparison,
    compare,
    parse_comparison,
    read_comparison,
    to_json,
    to_markdown,
)
from topple.evaluate import EvalProtocol, iter_episodes
from topple.loader import LoadError, load_submission
from topple.results import MAX_RESULTS_BYTES
from topple.stats import paired_bootstrap

REPO = Path(__file__).resolve().parents[1]
PID = REPO / "submissions" / "pid_example.py"
LQR = REPO / "submissions" / "lqr_example.py"
runner = CliRunner()


def ideal(n: int = 2) -> EvalProtocol:
    protocol = EvalProtocol.from_id("balance-v1/ideal")
    return replace(protocol, seeds=protocol.seeds[:n])


@pytest.fixture
def inline(monkeypatch: pytest.MonkeyPatch) -> list[InlinePool]:
    """``compare`` runs its pool tasks in this process."""
    InlinePool.instances = []
    monkeypatch.setattr(compare_module, "eval_pool", InlinePool)
    return InlinePool.instances


@pytest.fixture(scope="module")
def subs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    root = tmp_path_factory.mktemp("subs")
    pid, lqr = root / "pid_example.py", root / "lqr_example.py"
    shutil.copyfile(PID, pid)
    shutil.copyfile(LQR, lqr)
    return [pid, lqr, write_policy_dir(root / "tiny_ppo")]


@pytest.fixture(scope="module")
def three(subs: list[Path]) -> Comparison:
    """PID, LQR and the tiny policy on 2 ideal seeds, through a real spawn pool of 2."""
    return compare(subs, ideal(), workers=2)


def scores_of(sub: Path, protocol: EvalProtocol) -> list[float]:
    from topple.challenge import load_challenge

    factory = load_submission(sub, challenge=load_challenge("balance-v1")).factory
    return [run.result.score for run in iter_episodes(protocol, factory)]


# --- compare ----------------------------------------------------------------------------------


def test_compare_records_each_submission_in_seed_order(three: Comparison, subs: list[Path]) -> None:
    protocol = ideal()
    assert three.format == FORMAT
    assert three.protocol == "balance-v1/ideal"
    assert three.protocol_hash == protocol.protocol_hash()
    assert three.seeds == protocol.seeds
    assert three.complete is True
    assert [e.id for e in three.entries] == ["pid_example", "lqr_example", "tiny_ppo"]
    for entry, sub in zip(three.entries, subs, strict=True):
        assert list(entry.scores) == scores_of(sub, protocol)
        assert len(entry.success) == len(entry.energy_j) == len(entry.termination) == 2
        assert entry.extras == {}
    assert three.entries[0].policy_sha256 is None
    loaded = load_submission(subs[2], challenge=protocol.challenge)
    assert three.entries[2].policy_sha256 == loaded.policy_sha256
    assert three.entries[2].sha256 == loaded.sha256


def test_compare_has_a_paired_row_for_every_ordered_pair(three: Comparison) -> None:
    ids = [e.id for e in three.entries]
    assert [(r.a, r.b) for r in three.paired] == [
        (ids[1], ids[0]),
        (ids[2], ids[0]),
        (ids[2], ids[1]),
    ]
    for row in three.paired:
        a = next(e for e in three.entries if e.id == row.a)
        b = next(e for e in three.entries if e.id == row.b)
        assert (row.mean, row.lo, row.hi) == paired_bootstrap(a.scores, b.scores)
        assert (row.n_resamples, row.seed) == (10_000, 0)


def test_compare_records_terminations_and_metrics(three: Comparison) -> None:
    pid, lqr, ppo = three.entries
    assert pid.termination == ("timeout", "timeout")  # PID balances on ideal seeds
    assert all(e > 0.0 for e in pid.energy_j)
    assert all(t in ("fell", "floor_contact") for t in ppo.termination)  # random weights fall
    assert all(0.0 <= s <= 1.0 for e in three.entries for s in e.saturation_fraction)
    assert lqr.success == (True, True)


def test_inline_compare_equals_the_pool_compare(
    three: Comparison, subs: list[Path], inline: list[InlinePool]
) -> None:
    """Chunking and process placement never change a result: episodes are independent."""
    assert compare(subs, ideal(), workers=2) == three


def test_compare_uses_one_clamped_pool_and_shuts_it_down(
    subs: list[Path], inline: list[InlinePool]
) -> None:
    compare(subs[:1], ideal(1), workers=3)
    (pool,) = inline
    assert pool.workers == 3  # eval_pool clamps it (test_resource_caps)
    assert pool.shut_down


def test_compare_chunks_long_protocols_and_keeps_seed_order(
    subs: list[Path], inline: list[InlinePool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compare_module, "CHUNK_EPISODES", 2)
    c = compare(subs[:1], ideal(5), workers=2)
    (pool,) = inline
    assert [call[3] for call in pool.calls] == [(0, 1), (2, 3), (4,)]
    assert c.seeds == ideal(5).seeds
    assert len(c.entries[0].scores) == 5


def test_compare_loads_each_submission_once(
    subs: list[Path], inline: list[InlinePool], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    real = compare_module.submission_ref

    def counting(sub: Path, **kwargs: Any) -> Any:
        calls.append(sub)
        return real(sub, **kwargs)

    monkeypatch.setattr(compare_module, "submission_ref", counting)
    compare(subs, ideal(1), workers=2)
    assert calls == subs


def test_compare_refuses_two_submissions_with_one_name(
    tmp_path: Path, inline: list[InlinePool]
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, second = tmp_path / "a" / "bot.py", tmp_path / "b" / "bot.py"
    shutil.copyfile(PID, first)
    shutil.copyfile(PID, second)
    with pytest.raises(ValueError, match="named 'bot'"):
        compare([first, second], ideal(1), workers=1)
    assert inline == []  # refused before any pool exists


def test_compare_refuses_a_private_protocol_and_bad_arguments(subs: list[Path]) -> None:
    private = replace(ideal(1), seed_set="private", salt=2**64 + 1)
    with pytest.raises(ValueError, match="public"):
        compare(subs[:1], private)
    with pytest.raises(ValueError, match="at least one"):
        compare([], ideal(1))
    with pytest.raises(ValueError, match="at most"):
        compare([subs[0]] * (MAX_ENTRIES + 1), ideal(1))
    with pytest.raises(ValueError, match="workers"):
        compare(subs[:1], ideal(1), workers=0)


def test_compare_propagates_a_load_error(tmp_path: Path, inline: list[InlinePool]) -> None:
    with pytest.raises(LoadError):
        compare([tmp_path / "nothing.py"], ideal(1))


def test_a_worker_failure_cancels_the_rest_and_raises(
    subs: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failing(InlinePool):
        def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
            return super().submit(lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))

    InlinePool.instances = []
    monkeypatch.setattr(compare_module, "eval_pool", Failing)
    with pytest.raises(RuntimeError, match="boom"):
        compare(subs[:1], ideal(1))
    assert InlinePool.instances[0].shut_down


# --- topple.compare/1 -------------------------------------------------------------------------


def test_json_round_trip(three: Comparison, tmp_path: Path) -> None:
    text = to_json(three)
    assert text.endswith("\n")
    assert json.loads(text)["format"] == FORMAT
    path = tmp_path / "cmp.json"
    path.write_text(text)
    assert read_comparison(path) == three
    assert to_json(read_comparison(path)) == text


def test_json_layout(three: Comparison) -> None:
    doc = json.loads(to_json(three))
    assert set(doc) == {
        "format",
        "protocol",
        "protocol_hash",
        "complete",
        "seeds",
        "platform",
        "versions",
        "entries",
        "paired",
    }
    assert set(doc["entries"][0]) == {
        "id",
        "policy_sha256",
        "sha256",
        "scores",
        "success",
        "energy_j",
        "saturation_fraction",
        "termination",
        "extras",
    }
    assert set(doc["paired"][0]) == {"a", "b", "mean", "lo", "hi", "n_resamples", "seed"}
    assert doc["versions"]["onnxruntime"]  # per-episode ONNX results depend on it (§3.5)


def test_markdown_lists_entries_and_paired_rows(three: Comparison) -> None:
    text = to_markdown(three)
    assert "balance-v1/ideal" in text
    for name in ("pid\\_example", "lqr\\_example", "tiny\\_ppo"):
        assert name in text
    assert f"lqr\\_example {MINUS} pid\\_example" in text
    pid = three.entries[0]
    assert f"{math.fsum(pid.scores) / 2:.2f}" in text


def test_markdown_escapes_untrusted_ids(three: Comparison) -> None:
    hostile = three.model_copy(
        update={
            "entries": (
                three.entries[0].model_copy(update={"id": "<img src=x>|*[a](b)`"}),
                *three.entries[1:],
            ),
            "paired": (),
        }
    )
    text = to_markdown(hostile)
    row = next(line for line in text.splitlines() if "img" in line)
    cells = re.split(r"(?<!\\)\|", row)
    assert len(cells) == 10  # 8 columns: the id's "|" adds no cell
    assert re.search(r"(?<!\\)[<>*`\[\]()=]", cells[2]) is None  # all escaped
    assert "\\<img src\\=x\\>\\|\\*\\[a\\]\\(b\\)\\`" in text


# --- the reader: red team ----------------------------------------------------------------------


def doc_of(c: Comparison) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(to_json(c))
    return data


def parse(doc: dict[str, Any] | str) -> Comparison:
    text = doc if isinstance(doc, str) else json.dumps(doc)
    return parse_comparison(text.encode("utf-8"))


def test_the_reader_accepts_the_writer(three: Comparison) -> None:
    assert parse(doc_of(three)) == three


EDITS: dict[str, Callable[[dict[str, Any]], None]] = {
    "extra-top-key": lambda d: d.update(note="hi"),
    "extra-entry-key": lambda d: d["entries"][0].update(rank=1),
    "extra-paired-key": lambda d: d["paired"][0].update(p=0.01),
    "extra-versions-key": lambda d: d["versions"].update(cuda="12"),
    "wrong-format": lambda d: d.update(format="topple.compare/2"),
    "no-format": lambda d: d.pop("format"),
    "alias-protocol": lambda d: d.update(protocol="eval_v1"),
    "bad-protocol-hash": lambda d: d.update(protocol_hash="A" * 64),
    "no-seeds": lambda d: d.update(seeds=[]),
    "seed-too-large": lambda d: d["seeds"].__setitem__(0, 2**53),
    "negative-seed": lambda d: d["seeds"].__setitem__(0, -1),
    "string-seed": lambda d: d["seeds"].__setitem__(0, "1000"),
    "bool-seed": lambda d: d["seeds"].__setitem__(0, True),
    "float-seed": lambda d: d["seeds"].__setitem__(0, 1000.0),
    "score-above-100": lambda d: d["entries"][0]["scores"].__setitem__(0, 100.5),
    "negative-score": lambda d: d["entries"][0]["scores"].__setitem__(0, -0.5),
    "string-score": lambda d: d["entries"][0]["scores"].__setitem__(0, "50"),
    "int-success": lambda d: d["entries"][0]["success"].__setitem__(0, 1),
    "negative-energy": lambda d: d["entries"][0]["energy_j"].__setitem__(0, -1.0),
    "saturation-above-1": lambda d: d["entries"][0]["saturation_fraction"].__setitem__(0, 1.5),
    "short-scores": lambda d: d["entries"][0]["scores"].pop(),
    "long-success": lambda d: d["entries"][0]["success"].append(True),
    "upper-termination": lambda d: d["entries"][0]["termination"].__setitem__(0, "FELL"),
    "long-termination": lambda d: d["entries"][0]["termination"].__setitem__(0, "x" * 33),
    "bad-sha": lambda d: d["entries"][0].update(sha256="z" * 64),
    "bad-policy-sha": lambda d: d["entries"][2].update(policy_sha256="0" * 63),
    "empty-id": lambda d: d["entries"][0].update(id=""),
    "long-id": lambda d: d["entries"][0].update(id="x" * 201),
    "duplicate-id": lambda d: d["entries"][1].update(id=d["entries"][0]["id"]),
    "no-entries": lambda d: d.update(entries=[], paired=[]),
    "extras-wrong-length": lambda d: d["entries"][0].update(extras={"docked": [True]}),
    "extras-bad-name": lambda d: d["entries"][0].update(extras={"a\nb": [1, 2]}),
    "extras-long-string": lambda d: d["entries"][0].update(extras={"cause": ["x" * 65, "y"]}),
    "paired-missing": lambda d: d["paired"].pop(),
    "paired-swapped": lambda d: d["paired"][0].update(a=d["paired"][0]["b"], b=d["paired"][0]["a"]),
    "paired-reordered": lambda d: d["paired"].reverse(),
    "paired-extra": lambda d: d["paired"].append(dict(d["paired"][0])),
    "paired-lo-above-hi": lambda d: d["paired"][0].update(lo=1.0, hi=0.0),
    "paired-zero-resamples": lambda d: d["paired"][0].update(n_resamples=0),
    "paired-mean-out-of-range": lambda d: d["paired"][0].update(mean=101.0),
    "complete-with-missing": lambda d: d["entries"][0]["termination"].__setitem__(0, "missing"),
    "incomplete-without-missing": lambda d: d.update(complete=False),
    "string-complete": lambda d: d.update(complete="true"),
    "bad-platform": lambda d: d.update(platform="linux\x1b[31m"),
}


@pytest.mark.parametrize("edit", list(EDITS.values()), ids=list(EDITS))
def test_the_reader_refuses(three: Comparison, edit: Callable[[dict[str, Any]], None]) -> None:
    doc = doc_of(three)
    edit(doc)
    with pytest.raises(ValueError):  # noqa: PT011 - ValidationError or the reader's own
        parse(doc)


def test_a_missing_episode_is_incomplete_and_scores_zero(three: Comparison) -> None:
    doc = doc_of(three)
    entry = doc["entries"][0]
    entry["termination"][1] = "missing"
    entry["scores"][1] = 0.0
    entry["success"][1] = False
    doc["complete"] = False
    assert parse(doc).complete is False
    entry["scores"][1] = 10.0
    with pytest.raises(ValueError, match="missing"):
        parse(doc)


MARK = 0.123456789  # a number that appears once in the file, replaced by a raw token


@pytest.mark.parametrize(
    "token", ["NaN", "Infinity", "-Infinity", "1e400"], ids=["nan", "inf", "-inf", "overflow"]
)
@pytest.mark.parametrize("where", ["paired-mean", "energy"])
def test_the_reader_refuses_non_finite_numbers(three: Comparison, token: str, where: str) -> None:
    doc = doc_of(three)
    if where == "paired-mean":
        doc["paired"][0]["mean"] = MARK
    else:
        doc["entries"][0]["energy_j"][0] = MARK
    text = json.dumps(doc)
    assert text.count(repr(MARK)) == 1
    assert parse(text.replace(repr(MARK), "0.5")) is not None  # the edit itself is valid
    with pytest.raises(ValueError):  # noqa: PT011
        parse(text.replace(repr(MARK), token))


def test_the_reader_survives_a_nesting_bomb(three: Comparison) -> None:
    """A deeply nested value is refused by the parser's own depth limit, never by a crash."""
    doc = doc_of(three)
    doc["entries"][0]["extras"] = {"deep": "BOMB"}
    text = json.dumps(doc).replace('"BOMB"', "[" * 100_000 + "]" * 100_000)
    with pytest.raises(ValueError):  # noqa: PT011
        parse(text)


def test_the_reader_sanitizes_ids_like_results(three: Comparison) -> None:
    doc = doc_of(three)
    old = doc["entries"][0]["id"]
    doc["entries"][0]["id"] = "bot\x1b[2J"
    for row in doc["paired"]:
        for side in ("a", "b"):
            if row[side] == old:
                row[side] = "bot\x1b[2J"
    c = parse(doc)
    assert c.entries[0].id == "bot\ufffd[2J"
    assert c.paired[0].b == "bot\ufffd[2J"


def test_the_reader_bounds_the_seed_count(three: Comparison) -> None:
    doc = doc_of(three)
    n = 10_001
    doc["seeds"] = list(range(n))
    for entry in doc["entries"]:
        for key in ("scores", "success", "energy_j", "saturation_fraction", "termination"):
            entry[key] = [entry[key][0]] * n
    with pytest.raises(ValueError):  # noqa: PT011
        parse(doc)


def test_read_comparison_refuses_what_is_not_a_small_regular_file(
    tmp_path: Path, make_fifo: Callable[[Path], Path]
) -> None:
    with pytest.raises(ValueError, match="regular"):
        read_comparison(make_fifo(tmp_path / "fifo.json"))
    with pytest.raises(ValueError, match="regular"):
        read_comparison(tmp_path)
    big = tmp_path / "big.json"
    with big.open("wb") as fh:
        fh.truncate(MAX_RESULTS_BYTES + 1)  # sparse: nothing is written or read
    with pytest.raises(ValueError, match="bytes"):
        read_comparison(big)
    with pytest.raises(OSError):  # noqa: PT011
        read_comparison(tmp_path / "absent.json")


def test_read_comparison_follows_a_symlink_like_read_results(
    three: Comparison, tmp_path: Path
) -> None:
    target = tmp_path / "real.json"
    target.write_text(to_json(three))
    link = tmp_path / "link.json"
    os.symlink(target, link)
    assert read_comparison(link) == three


# --- CLI --------------------------------------------------------------------------------------


def test_cli_compare_prints_markdown_and_writes_json(
    subs: list[Path], tmp_path: Path, inline: list[InlinePool], three: Comparison
) -> None:
    out = tmp_path / "out" / "cmp.json"
    args = ["compare", *map(str, subs), "-p", "balance-v1/ideal", "--seeds", "1000..1001"]
    result = runner.invoke(app, [*args, "--workers", "2", "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert f"tiny\\_ppo {MINUS} lqr\\_example" in result.output
    written = read_comparison(out)
    assert written.seeds == (1000, 1001)
    assert written.protocol == "balance-v1/ideal"
    assert written.protocol_hash == replace(ideal(), seeds=(1000, 1001)).protocol_hash()
    assert written == three  # ideal's first two public seeds are 1000 and 1001
    assert oct(out.stat().st_mode & 0o777) == "0o644"
    assert [p.name for p in out.parent.iterdir()] == ["cmp.json"]  # no temporary left


@pytest.mark.parametrize(
    "seeds",
    ["5..4", "0..10000", "a..b", "1..", "..2", "0...2", "-1..2", "1.5..2", f"0..{2**53}", "3"],
)
def test_cli_compare_refuses_bad_seed_ranges(subs: list[Path], seeds: str) -> None:
    result = runner.invoke(
        app, ["compare", str(subs[0]), "-p", "balance-v1/ideal", "--seeds", seeds]
    )
    assert result.exit_code == 2
    assert "--seeds" in result.output


def test_cli_compare_seed_range_is_inclusive_and_capped(
    subs: list[Path], inline: list[InlinePool], tmp_path: Path
) -> None:
    out = tmp_path / "c.json"
    result = runner.invoke(
        app, ["compare", str(subs[0]), "-p", "eval_v1", "--seeds", "7..7", "--json", str(out)]
    )
    assert result.exit_code == 0, result.output
    c = read_comparison(out)
    assert (c.protocol, c.seeds) == ("balance-v1/standard", (7,))


def test_cli_compare_refuses_a_symlinked_output_before_running(
    subs: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inline: list[InlinePool]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", tmp_path / "out")
    result = runner.invoke(
        app, ["compare", str(subs[0]), "-p", "balance-v1/ideal", "--json", "out/cmp.json"]
    )
    assert result.exit_code == 2
    assert "refusing to write through the symlink 'out'" in result.output
    assert inline == []
    assert list((tmp_path / "elsewhere").iterdir()) == []


def test_cli_compare_load_errors_exit_2_on_one_line(tmp_path: Path) -> None:
    bad = tmp_path / "bad\x1b[31m.py"
    bad.write_text("raise SystemExit('no')\n")
    result = runner.invoke(app, ["compare", str(bad), "-p", "balance-v1/ideal"])
    assert result.exit_code == 2
    assert "\x1b" not in result.output
    assert result.output.count("\n") == 1


def test_cli_compare_unknown_protocol_exits_2(subs: list[Path]) -> None:
    result = runner.invoke(app, ["compare", str(subs[0]), "-p", "balance-v9/standard"])
    assert result.exit_code == 2


def test_cli_compare_duplicate_names_exit_2(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    shutil.copyfile(PID, tmp_path / "a" / "bot.py")
    shutil.copyfile(PID, tmp_path / "b" / "bot.py")
    result = runner.invoke(
        app,
        [
            "compare",
            str(tmp_path / "a" / "bot.py"),
            str(tmp_path / "b" / "bot.py"),
            "-p",
            "eval_v1",
        ],
    )
    assert result.exit_code == 2
    assert "named 'bot'" in result.output


def test_cli_compare_a_dead_worker_exits_2(
    subs: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures.process import BrokenProcessPool

    def dead(*args: object, **kwargs: object) -> None:
        raise BrokenProcessPool("a child process terminated abruptly")

    monkeypatch.setattr(compare_module, "compare", dead)
    result = runner.invoke(app, ["compare", str(subs[0]), "-p", "balance-v1/ideal"])
    assert result.exit_code == 2
    assert "an evaluation worker died" in result.output


def test_cli_compare_an_unwritable_json_path_exits_2(
    subs: list[Path], tmp_path: Path, inline: list[InlinePool]
) -> None:
    (tmp_path / "file").write_text("x")
    out = tmp_path / "file" / "cmp.json"
    args = ["compare", str(subs[0]), "-p", "balance-v1/ideal", "--seeds", "1000..1000"]
    result = runner.invoke(app, [*args, "--json", str(out)])
    assert result.exit_code == 2
    assert "cannot write" in result.output


def test_markdown_names_any_seed_list(three: Comparison) -> None:
    def seeds_line(seeds: tuple[int, ...]) -> str:
        n = len(seeds)
        doc = doc_of(three)
        doc["seeds"] = list(seeds)
        for entry in doc["entries"]:
            for key in ("scores", "success", "energy_j", "saturation_fraction", "termination"):
                entry[key] = [entry[key][0]] * n
        return to_markdown(parse(doc)).splitlines()[0]

    assert seeds_line((5, 6, 7)).endswith("3 seeds (5..7)")
    assert seeds_line((9, 2)).endswith("2 seeds (9, 2)")
    assert seeds_line((9, 2, 4)).endswith("3 seeds (9, ..., 4)")


def test_parse_comparison_caps_its_input() -> None:
    with pytest.raises(ValueError, match="at most"):
        parse_comparison(b" " * (MAX_RESULTS_BYTES + 1))
