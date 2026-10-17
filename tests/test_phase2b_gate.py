"""``scripts/phase2b_gate.py`` (docs/design/phase-2.md §8.9): each item fails on the defect it
guards, and the committed ``phase2b.json`` and ``phase2b.md`` agree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import phase2b_common as wp
import phase2b_gate as gate
from test_phase2b_common import comparison, driver, ent, flat

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "docs" / "results" / "phase2b.json"
pytestmark = pytest.mark.skipif(not REPORT.is_file(), reason="phase2b.json is not committed yet")


def doc() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(REPORT.read_text())
    return data


def report(data: dict[str, Any]) -> wp.Phase2bReport:
    return wp.parse_report(json.dumps(data).encode())


def test_the_committed_report_parses_and_its_markdown_is_its_rendering() -> None:
    rep = report(doc())
    md = REPORT.with_name(wp.PHASE2B_MD).read_text(encoding="utf-8")
    assert gate.item5(rep, md) == []
    assert gate.item5(rep, md + "edited\n") == ["phase2b.md is not the rendering of phase2b.json"]
    assert gate.item5(rep, None) == ["phase2b.md is missing"]


def test_item5_wants_three_runs_per_arm() -> None:
    data = doc()
    data["runs"][0]["arm"] = "finetune"
    problems = gate.item5(report(data), None)
    assert "not 3 scratch runs" in problems
    assert "not 3 finetune runs" in problems


def test_item7_checks_each_fine_tune_against_its_own_source() -> None:
    data = doc()
    assert gate.item7(report(data)) == []
    data["sanity"][0]["survival"] = 0.9
    assert gate.item7(report(data))[0].endswith("survival 0.90 < 0.95")
    data = doc()
    data["sanity"][0]["source"]["step"] += 1
    assert "checks another source" in gate.item7(report(data))[0]
    data = doc()
    data["sanity"] = data["sanity"][1:]
    assert "no sanity row" in gate.item7(report(data))[0]


def test_item11_refuses_a_stale_run() -> None:
    data = doc()
    assert gate.item11(report(data)) == []
    data["runs"][2]["summary"]["env_algo_version"] += 1
    data["runs"][3]["summary"]["selection_protocol_hash"] = "0" * 64
    problems = gate.item11(report(data))
    assert len(problems) == 2
    assert "waypoint env" in problems[0]
    assert "another selection protocol" in problems[1]


def test_item4_names_each_missing_term() -> None:
    text = " ".join(gate.CONTRACT_TERMS)
    assert gate.item4(text) == []
    assert gate.item4(text.replace("fall_cause", "")) == [
        "docs/contract.md does not name 'fall_cause'"
    ]


def test_item8_checks_presence_caps_and_gif_sides(tmp_path: Path) -> None:
    assert len(gate.item8(tmp_path)) == len(gate.MEDIA)
    for name, (_, side) in gate.MEDIA.items():
        if side is None:
            (tmp_path / name).write_bytes(b"\x00" * 64)
        else:
            Image.new("RGB", (side, side)).save(tmp_path / name)
    assert gate.item8(tmp_path) == []
    Image.new("RGB", (320, 320)).save(tmp_path / "waypoint.gif")
    assert gate.item8(tmp_path) == ["waypoint.gif is not a 480x480 GIF"]


def test_item6_judges_the_shipped_policy_and_binds_every_comparison() -> None:
    """The synthetic comparisons are standard 0..49, so each is flagged as not its protocol;
    the policy binding and the entry checks are judged on top of that."""
    sha = "ab" * 32
    base = [driver(), ent(wp.CREEPER_ID, flat(60.0), docked=0, fell=0)]
    shipped = ent(gate.SHIPPED_ID, flat(79.0), docked=40, fell=5)
    good = comparison([*base, shipped.model_copy(update={"policy_sha256": sha})])
    not_protocol = [f"{n} is not its protocol at the current hash" for n in gate.EXPECTED_HASHES]
    assert gate.item6(good, good, good, sha) == not_protocol
    other = gate.item6(good, good, good, "cd" * 32)
    assert "wp_eval: ppo_waypoint is not the submission's policy" in other
    weak = comparison([*base, ent(gate.SHIPPED_ID, flat(79.0), docked=20, fell=5)])
    assert gate.item6(weak, good, good, sha)[0] == "a_docked"
    without = comparison(base)
    problems = gate.item6(good, without, good, sha)
    assert "wp_clean: waypoint-v1/standard: no single entry 'ppo_waypoint'" in problems


RUNS = REPORT.with_name("phase2b")


def test_item5_rows_accept_the_committed_report_and_comparisons() -> None:
    assert gate.item5_rows(report(doc()), RUNS) == []


def test_item5_rows_catch_an_edited_row_and_a_reordered_shipping_list() -> None:
    data = doc()
    data["fresh"][0]["standard"]["score"] += 1.0
    assert any("its standard row differs" in p for p in gate.item5_rows(report(data), RUNS))
    data = doc()
    data["fresh"][1]["item6"]["pass"] = not data["fresh"][1]["item6"]["pass"]
    assert any("its item 6 differs" in p for p in gate.item5_rows(report(data), RUNS))
    data = doc()
    data["shipped"]["attempts"][0]["run"] = "ppo_wp_s1"
    problems = gate.item5_rows(report(data), RUNS)
    assert "the shipping attempts are not the candidates in rank order" in problems


def test_main_exits_2_on_an_unreadable_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.json"
    inputs = [str(missing)] * 3 + [str(REPORT)]
    flags = ["--submission", str(tmp_path), "--fixture", str(missing)]
    code = gate.main([*inputs, *flags, "--results-dir", str(tmp_path)])
    assert code == 2
    assert capsys.readouterr().err.startswith("phase2b_gate: ")
