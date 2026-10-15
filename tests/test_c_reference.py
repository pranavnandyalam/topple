"""The C reference forward pass (docs/design/phase-2.md §6 ``test_c_reference``, §3.2, §3.3).

``tests/c/genc.py`` turns a first-party ``policy.onnx`` into a C header: its initializers read by
§3.2 position, every value a ``float.hex`` literal, and ``policy_sha256``; it refuses anything but
the two structure-matched sequences (``clip(mean)``, ``tanh(mean)``). ``tests/c/mlp.c`` runs that
forward pass in float32 on every row of a ``.bin`` v1 parity fixture and reports the largest
``|a_c - a_ref|``, which must stay within 1e-5 with the system ``cc -O2``, with and without
``-ffp-contract=off`` (FMA contraction is the usual source of float differences between
compilers and CPUs).

Until ``submissions/ppo_balance`` is committed, the policies are tiny PPO and SAC exports built
here from a synthetic run (``tests/run_support.py``) with ``topple export --test-fixture``'s code;
the shipped case is collected always and runs once the submission exists. Without ``cc`` the
tests skip, unless ``TOPPLE_REQUIRE_CC=1`` (CI's ``python`` job), where they fail.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from run_support import make_run
from topple.export import PARITY_TOLERANCE, export_policy

REPO = Path(__file__).resolve().parents[1]
C_DIR = REPO / "tests" / "c"
GENC = C_DIR / "genc.py"
MLP = C_DIR / "mlp.c"
SHIPPED = "ppo_balance"
TIMEOUT_S = 120
FLAG_SETS = {"default": (), "no_fp_contract": ("-ffp-contract=off",)}


@dataclass(frozen=True)
class Case:
    """A policy and its parity fixture files."""

    policy: Path
    bin: Path
    sidecar: Path


def compiler() -> str:
    """The system ``cc``; skip without one, unless ``TOPPLE_REQUIRE_CC=1`` makes that a
    failure."""
    cc = shutil.which("cc")
    if cc is None:
        if os.environ.get("TOPPLE_REQUIRE_CC") == "1":
            pytest.fail("TOPPLE_REQUIRE_CC=1, but there is no cc on PATH")
        pytest.skip("no C compiler: cc is not on PATH")
    return cc


def genc(policy: Path, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENC), str(policy), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )


def build(cc: str, header: Path, exe: Path, flags: tuple[str, ...]) -> None:
    """``cc -O2 [flags] mlp.c`` against ``header`` (copied in as ``policy.h``)."""
    include = exe.parent / f"{exe.name}_include"
    include.mkdir()
    shutil.copyfile(header, include / "policy.h")
    result = subprocess.run(
        [
            cc,
            "-O2",
            *flags,
            "-Wall",
            "-Wextra",
            "-I",
            str(include),
            "-o",
            str(exe),
            str(MLP),
            "-lm",
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "warning" not in result.stderr, result.stderr


def run_c(exe: Path, fixture: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(exe), str(fixture), repr(PARITY_TOLERANCE)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Case]:
    """Tiny PPO (tanh, ``clip(mean)``) and SAC (relu, ``tanh(mean)``) exports with their
    ``--test-fixture`` files."""
    root = tmp_path_factory.mktemp("c_reference")
    cases = {}
    for algo in ("ppo", "sac"):
        run = make_run(root / "runs", algo=algo, name=f"{algo}_s0")
        name = f"tiny_{algo}"
        export_policy(run.path, root / name, test_fixture=name, fixture_dir=root / "data")
        cases[name] = Case(
            root / name / "policy.onnx",
            root / "data" / f"{name}_parity.bin",
            root / "data" / f"{name}_parity.json",
        )
    return cases


def resolve(name: str, tiny: dict[str, Case]) -> Case:
    if name in tiny:
        return tiny[name]
    sub = REPO / "submissions" / name
    if not sub.exists():
        pytest.skip(f"submissions/{name} is not committed yet")
    data = REPO / "tests" / "data"
    return Case(sub / "policy.onnx", data / f"{name}_parity.bin", data / f"{name}_parity.json")


@pytest.mark.parametrize("flags", FLAG_SETS.values(), ids=FLAG_SETS.keys())
@pytest.mark.parametrize("name", ["tiny_ppo", "tiny_sac", SHIPPED])
def test_the_c_forward_pass_matches_every_parity_row(
    name: str, flags: tuple[str, ...], tiny: dict[str, Case], tmp_path: Path
) -> None:
    cc = compiler()
    case = resolve(name, tiny)
    sidecar = json.loads(case.sidecar.read_text())
    policy_sha256 = hashlib.sha256(case.policy.read_bytes()).hexdigest()
    assert sidecar["policy_sha256"] == policy_sha256
    header = tmp_path / "policy.h"
    made = genc(case.policy, header)
    assert made.returncode == 0, made.stderr
    exe = tmp_path / "mlp"
    build(cc, header, exe, flags)
    result = run_c(exe, case.bin)
    assert result.returncode == 0, result.stdout + result.stderr
    sha, rows, worst = result.stdout.split()
    assert sha == policy_sha256
    assert int(rows) == sidecar["n"]
    assert 0.0 <= float(worst) <= PARITY_TOLERANCE


def test_the_header_holds_hex_literals_and_the_sha(tiny: dict[str, Case], tmp_path: Path) -> None:
    case = tiny["tiny_ppo"]
    header = tmp_path / "policy.h"
    assert genc(case.policy, header).returncode == 0
    text = header.read_text()
    assert hashlib.sha256(case.policy.read_bytes()).hexdigest() in text
    assert "#define POLICY_SAC 0" in text
    assert "0x1." in text  # float.hex literals
    assert "p+" in text
    assert "p-" in text
    assert "e-" not in text.lower().replace("0x", "")  # no decimal exponents


def test_a_different_policys_rows_fail(tiny: dict[str, Case], tmp_path: Path) -> None:
    """Negative control: the PPO header on the SAC fixture (same D and A) is far off."""
    cc = compiler()
    header = tmp_path / "policy.h"
    assert genc(tiny["tiny_ppo"].policy, header).returncode == 0
    exe = tmp_path / "mlp"
    build(cc, header, exe, ())
    result = run_c(exe, tiny["tiny_sac"].bin)
    assert result.returncode == 1, result.stdout + result.stderr
    assert float(result.stdout.split()[2]) > 1e-3


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda b: b"XPAR" + b[4:], "magic"),
        (lambda b: b[:4] + (2).to_bytes(4, "little") + b[8:], "version"),
        (lambda b: b[:12] + (11).to_bytes(4, "little") + b[16:], "D and A"),
        (lambda b: b[:40], "truncated"),
    ],
    ids=["magic", "version", "dims", "truncated"],
)
def test_the_c_reader_refuses_a_bad_fixture(
    edit: object, message: str, tiny: dict[str, Case], tmp_path: Path
) -> None:
    cc = compiler()
    header = tmp_path / "policy.h"
    assert genc(tiny["tiny_ppo"].policy, header).returncode == 0
    exe = tmp_path / "mlp"
    build(cc, header, exe, ())
    bad = tmp_path / "bad.bin"
    bad.write_bytes(edit(tiny["tiny_ppo"].bin.read_bytes()))  # type: ignore[operator]
    result = run_c(exe, bad)
    assert result.returncode == 2
    assert message in result.stderr


def test_genc_refuses_a_graph_that_is_not_a_section_3_2_sequence(
    tiny: dict[str, Case], tmp_path: Path
) -> None:
    """``Sub(c0, obs)`` (mean - obs) computes something else: ``structure_match`` says
    ``other``, so the positional read would be wrong and genc refuses."""
    import onnx

    proto = onnx.ModelProto.FromString(tiny["tiny_ppo"].policy.read_bytes())
    proto.graph.node[0].input.reverse()
    swapped = tmp_path / "swapped.onnx"
    swapped.write_bytes(proto.SerializeToString(deterministic=True))
    header = tmp_path / "policy.h"
    result = genc(swapped, header)
    assert result.returncode == 2
    assert "structure_match: other" in result.stderr
    assert not header.exists()


def test_genc_refuses_bytes_that_are_not_onnx(tmp_path: Path) -> None:
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"\xff" * 64)
    result = genc(junk, tmp_path / "policy.h")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_without_cc_the_tests_skip_or_fail_as_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.delenv("TOPPLE_REQUIRE_CC", raising=False)
    with pytest.raises(pytest.skip.Exception):
        compiler()
    monkeypatch.setenv("TOPPLE_REQUIRE_CC", "1")
    with pytest.raises(pytest.fail.Exception, match="TOPPLE_REQUIRE_CC"):
        compiler()
