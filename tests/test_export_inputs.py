"""``topple export`` reads untrusted run directories (docs/design/phase-2.md §3.3, §6
``test_export_inputs``).

Synthetic runs (``run_support.make_run``: tiny deterministic weights, ``np.savez`` and
``json``) export to loadable policies; every input rule refuses its bad case with one clean
line (exit 2 from the CLI), never a traceback, a big allocation or a write; the parity fixture
files have the ``.bin`` v1 layout and the JSON sidecar.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import topple.export as export_module
from policy_support import run as onnx_run
from run_support import GIT_COMMIT, Run, make_run
from topple._npz import read_npz_bytes
from topple.challenge import load_challenge
from topple.cli import app
from topple.export import (
    BIN_HEADER,
    CONSTANT_NAMES,
    MAX_NPZ_BYTES,
    ExportError,
    ExportWarning,
    constant_mismatches,
    coverage_counts,
    export_policy,
    parity_bin,
    read_run_json,
    snapshot_files,
    write_snapshot,
)
from topple.policy.build import ObsNorm, build_policy_onnx
from topple.policy.manifest import read_manifest
from topple.policy.runtime import load_policy_dir
from topple.train.config import validate_train_config

runner = CliRunner()


@pytest.fixture
def ppo(tmp_path: Path) -> Run:
    return make_run(tmp_path)


def refused(run: Run, out: Path, match: str, **kwargs: Any) -> None:
    with pytest.raises(ExportError, match=match):
        export_policy(run.path, out, **kwargs)
    assert not os.path.lexists(out) or kwargs.get("report_only") or kwargs.get("replace")


# --- the happy path -----------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_a_run_exports_to_a_loadable_policy(tmp_path: Path, algo: str) -> None:
    run = make_run(tmp_path, algo=algo)
    out = tmp_path / "policy"
    manifest = export_policy(run.path, out)
    assert sorted(os.listdir(out)) == ["policy.json", "policy.onnx"]
    assert (
        manifest.onnx.sha256
        == run.sha256
        == hashlib.sha256((out / "policy.onnx").read_bytes()).hexdigest()
    )
    assert read_manifest((out / "policy.json").read_bytes()) == manifest
    loaded = load_policy_dir(out, challenge=load_challenge("balance-v1"))
    assert loaded.policy_sha256 == run.sha256
    assert manifest.action_rule == ("clip(mean)" if algo == "ppo" else "tanh(mean)")
    claims = manifest.claims
    assert (claims.algo, claims.activation, claims.net_arch) == (
        algo,
        "tanh" if algo == "ppo" else "relu",
        (8, 8),
    )
    assert (claims.total_timesteps, claims.selected_step, claims.train_seed) == (2048, 1024, 0)
    assert claims.train_config_sha256 == run.train_meta["config_sha256"]
    assert (claims.git_commit, claims.git_dirty, claims.selection_eligible) == (
        GIT_COMMIT,
        False,
        True,
    )
    # the versions whitelist: strings with valid keys and printable values only
    assert claims.versions == {
        "onnx": "1.23.2",
        "python": "3.12.3",
        "stable_baselines3": "2.9.0",
        "topple": "0.1.0",
        "torch": "2.14.1",
    }
    assert claims.name is None
    parity = claims.export_parity
    assert parity is not None
    assert (parity.reference, parity.n, parity.n_onpolicy, parity.n_onpolicy_seeds) == (
        "sb3.predict",
        1000 + 1000 + 64,
        1000,
        200,
    )
    assert parity.n_unsaturated >= 300
    assert parity.n_saturated >= 32
    assert 0.0 <= parity.max_abs_diff <= 1e-5
    assert parity.tolerance == 1e-5
    assert parity.vacuous is False


def test_parity_is_against_the_onnx_actions(ppo: Run, tmp_path: Path) -> None:
    out = tmp_path / "policy"
    manifest = export_policy(ppo.path, out)
    actions = onnx_run((out / "policy.onnx").read_bytes(), ppo.obs)
    worst = float(np.abs(actions.astype(np.float64) - ppo.action_ref).max())
    assert manifest.claims.export_parity is not None
    assert manifest.claims.export_parity.max_abs_diff == pytest.approx(worst, rel=0, abs=1e-12)


def test_the_cli_exports(ppo: Run, tmp_path: Path) -> None:
    out = tmp_path / "policy"
    result = runner.invoke(app, ["export", str(ppo.path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert f"exported step 1024 -> {out}: policy_sha256 {ppo.sha256[:12]}" in result.output
    assert (out / "policy.onnx").exists()


# --- step 1: resolve ----------------------------------------------------------------------------


def test_best_json_extra_keys_such_as_dir_are_ignored(ppo: Run, tmp_path: Path) -> None:
    ppo.rewrite_json("best.json", lambda d: d.update(dir="../../elsewhere", note=[1, 2]))
    assert export_policy(ppo.path, tmp_path / "p").claims.selected_step == 1024


def test_step_overrides_best_json(ppo: Run, tmp_path: Path) -> None:
    (ppo.path / "best.json").unlink()
    assert export_policy(ppo.path, tmp_path / "p", step=1024).claims.selected_step == 1024
    with pytest.raises(ExportError, match=r"best\.json"):
        export_policy(ppo.path, tmp_path / "q")


@pytest.mark.parametrize("step", [2**53, -1, 1.5, True, "1024", None])
def test_best_json_step_must_be_an_integer_below_2_53(
    ppo: Run, tmp_path: Path, step: object
) -> None:
    ppo.rewrite_json("best.json", lambda d: d.update(step=step))
    refused(ppo, tmp_path / "p", "best.json: step")


@pytest.mark.parametrize("step", [2**53, -1])
def test_the_step_option_is_bounded(ppo: Run, tmp_path: Path, step: int) -> None:
    with pytest.raises(ExportError, match="--step"):
        export_policy(ppo.path, tmp_path / "p", step=step)
    result = runner.invoke(
        app, ["export", str(ppo.path), "--out", str(tmp_path / "q"), "--step", str(step)]
    )
    assert result.exit_code == 2
    assert not (tmp_path / "q").exists()


def test_a_missing_step_dir_is_refused(ppo: Run, tmp_path: Path) -> None:
    refused(ppo, tmp_path / "p", "step_2048", step=2048)


def test_a_symlinked_step_dir_is_refused(ppo: Run, tmp_path: Path) -> None:
    real = ppo.path / "ckpt" / "real"
    ppo.step_dir.rename(real)
    ppo.step_dir.symlink_to(real)
    refused(ppo, tmp_path / "p", "is a symlink")


def test_an_escaping_step_dir_is_refused(ppo: Run, tmp_path: Path) -> None:
    """``ckpt`` itself a symlink out of the run: the step dir resolves outside it."""
    elsewhere = tmp_path / "elsewhere"
    (ppo.path / "ckpt").rename(elsewhere)
    (ppo.path / "ckpt").symlink_to(elsewhere)
    refused(ppo, tmp_path / "p", "outside the run")


def test_an_ineligible_step_needs_report_only(tmp_path: Path) -> None:
    run = make_run(tmp_path, eligible=False)
    refused(run, tmp_path / "p", "not eligible")
    out = run.path / "report_policy"
    manifest = export_policy(run.path, out, report_only=True)
    assert manifest.claims.selection_eligible is False


@pytest.mark.parametrize("flag", ["rss_abort", "cpu_abort", "wall_clock_abort", "nondeterministic"])
def test_a_flagged_run_is_never_shipped(tmp_path: Path, flag: str) -> None:
    run = make_run(tmp_path, flags=(flag, "high_energy"))
    refused(run, tmp_path / "p", flag)
    export_policy(run.path, run.path / "report_policy", report_only=True)


def test_warning_flags_do_not_block(tmp_path: Path) -> None:
    run = make_run(tmp_path, flags=("high_energy", "late_regression", "std_collapse"))
    export_policy(run.path, tmp_path / "p")


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda d: d.update(flags="none"), "flags"),
        (lambda d: d.update(flags=[1]), "flags"),
        (lambda d: d.pop("flags"), "flags"),
        (lambda d: d.update(selection_history={}), "selection_history"),
        (lambda d: d["selection_history"].append({"step": "x"}), "integer step"),
        (lambda d: d["selection_history"].append(dict(d["selection_history"][1])), "2 rows"),
        (lambda d: d["selection_history"].pop(), "0 rows"),
        (lambda d: d["selection_history"][1].update(eligible=1), "eligible"),
        (lambda d: d["selection_history"][1].pop("onnx_sha256"), "onnx_sha256"),
        (lambda d: d["selection_history"][1].update(onnx_sha256="0" * 64), "onnx_sha256_mismatch"),
        (lambda d: d["config"].update(seed=5), "config_sha256"),
        (lambda d: d["config"].update(n_envs=99), "config"),
        (lambda d: d.update(config_sha256="2" * 64), "config_sha256"),
        (lambda d: d.update(git_commit="HEAD"), "git_commit"),
        (lambda d: d.update(git_dirty="no"), "git_dirty"),
        (lambda d: d.update(versions=["torch"]), "versions"),
    ],
)
def test_train_meta_is_checked(ppo: Run, tmp_path: Path, edit: Any, match: str) -> None:
    ppo.rewrite_json("train_meta.json", edit)
    refused(ppo, tmp_path / "p", match)


def test_null_git_fields_are_null_claims(ppo: Run, tmp_path: Path) -> None:
    ppo.rewrite_json(
        "train_meta.json", lambda d: d.update(git_commit=None, git_dirty=None, versions=None)
    )
    claims = export_policy(ppo.path, tmp_path / "p").claims
    assert (claims.git_commit, claims.git_dirty, claims.versions) == (None, None, None)


# --- the output ---------------------------------------------------------------------------------


def test_an_existing_out_needs_replace(ppo: Run, tmp_path: Path) -> None:
    out = tmp_path / "policy"
    export_policy(ppo.path, out)
    first = (out / "policy.onnx").read_bytes()
    (out / "README.md").write_text("old")
    result = runner.invoke(app, ["export", str(ppo.path), "--out", str(out)])
    assert result.exit_code == 2
    assert "already exists" in result.output
    assert (out / "README.md").exists()
    result = runner.invoke(app, ["export", str(ppo.path), "--out", str(out), "--replace"])
    assert result.exit_code == 0, result.output
    assert sorted(os.listdir(out)) == ["policy.json", "policy.onnx"]  # the old one is gone
    assert (out / "policy.onnx").read_bytes() == first
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".policy")] == []


def test_replace_never_removes_what_is_not_a_policy(ppo: Run, tmp_path: Path) -> None:
    out = tmp_path / "project"
    out.mkdir()
    (out / "notes.txt").write_text("precious")
    refused(ppo, out, "more than a policy", replace=True)
    assert (out / "notes.txt").read_text() == "precious"
    nested = tmp_path / "nested"
    (nested / "policy.json").mkdir(parents=True)
    refused(ppo, nested, "more than a policy", replace=True)
    plain = tmp_path / "file"
    plain.write_text("x")
    refused(ppo, plain, "not a directory", replace=True)
    assert plain.read_text() == "x"


def test_replace_swaps_a_symlink_never_its_target(ppo: Run, tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(target)
    export_policy(ppo.path, link, replace=True)
    assert not link.is_symlink()
    assert (link / "policy.onnx").exists()
    assert (target / "keep.txt").read_text() == "keep"


def test_out_must_not_be_under_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``runs/`` is the checkout's gitignored run folder: the one under the working directory,
    and the one a run lives in."""
    run = make_run(tmp_path / "runs", name="ppo_s0")
    monkeypatch.chdir(tmp_path)
    refused(run, Path("runs/shipped"), "under runs/")
    with pytest.raises(ExportError, match="under runs/"):
        export_policy(run.path, Path("runs"), replace=True)
    refused(run, tmp_path / "runs" / "shipped", "under runs/")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # the run's own runs/ still counts
    refused(run, tmp_path / "runs" / "shipped", "under runs/")
    refused(run, run.path / "shipped", "under runs/")  # inside the run, and under its runs/
    other = make_run(tmp_path / "scratch")  # a run outside any runs/: only the run itself
    refused(other, other.path / "shipped", "inside the run")
    export_policy(other.path, tmp_path / "x" / "runs" / "p")  # neither the cwd's nor the run's


def test_a_checkout_under_a_directory_named_runs_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clone at ``.../runs/topple`` ships ``runs/ppo_s0`` to ``submissions/ppo_balance``: only
    the checkout's own ``runs/`` is refused, not every path with a component called runs."""
    checkout = tmp_path / "runs" / "topple"
    make_run(checkout / "runs", name="ppo_s0")
    monkeypatch.chdir(checkout)
    manifest = export_policy(Path("runs/ppo_s0"), Path("submissions/ppo_balance"))
    assert (checkout / "submissions" / "ppo_balance" / "policy.onnx").exists()
    assert manifest.claims.selected_step == 1024


def test_report_only_writes_report_policy_and_is_idempotent(ppo: Run, tmp_path: Path) -> None:
    out = ppo.path / "report_policy"
    for _ in range(2):
        result = runner.invoke(app, ["export", str(ppo.path), "--out", str(out), "--report-only"])
        assert result.exit_code == 0, result.output
    assert sorted(os.listdir(out)) == ["policy.json", "policy.onnx"]
    refused(ppo, tmp_path / "elsewhere", "report_policy", report_only=True)
    refused(ppo, ppo.path / "other", "report_policy", report_only=True)


def test_vacuous_coverage_is_recorded_with_report_only_not_failed(tmp_path: Path) -> None:
    run = make_run(tmp_path, head_scale=0.05)  # no saturated rows at all
    with pytest.raises(ExportError, match="0 saturated rows"):
        export_policy(run.path, tmp_path / "p")
    manifest = export_policy(run.path, run.path / "report_policy", report_only=True)
    parity = manifest.claims.export_parity
    assert parity is not None
    assert parity.vacuous is True
    assert parity.n_saturated == 0


@pytest.mark.parametrize(
    ("onpolicy", "selection", "match"),
    [
        (201, 200, "n_onpolicy_seeds 201 is more than the 200 selection seeds"),
        (0, 200, "no on-policy rows"),
    ],
)
def test_onpolicy_seeds_are_bound_to_the_selection_seeds(
    tmp_path: Path, onpolicy: int, selection: int, match: str
) -> None:
    """(a) has 5 rows per selection seed with T >= 1: a snapshot that dropped every on-policy row
    or claims more episodes than the run's selection seeds is a coverage miss."""
    run = make_run(tmp_path, onpolicy_seeds=onpolicy, selection_seeds=selection)
    refused(run, tmp_path / "p", match)
    manifest = export_policy(run.path, run.path / "report_policy", report_only=True)
    assert manifest.claims.export_parity is not None
    assert manifest.claims.export_parity.vacuous is True
    assert manifest.claims.export_parity.n_onpolicy_seeds == onpolicy


def test_fewer_onpolicy_seeds_than_selection_seeds_are_fine(tmp_path: Path) -> None:
    """Episodes with T = 0 send no rows: 199 of 200 is covered."""
    run = make_run(tmp_path, onpolicy_seeds=199, selection_seeds=200)
    parity = export_policy(run.path, tmp_path / "p").claims.export_parity
    assert parity is not None
    assert (parity.n_onpolicy_seeds, parity.vacuous) == (199, False)


def test_onpolicy_rows_must_be_five_per_seed(ppo: Run, tmp_path: Path) -> None:
    ppo.rewrite_json("ckpt/step_1024/meta.json", lambda d: d.update(n_onpolicy_seeds=199))
    refused(ppo, tmp_path / "p", "n_onpolicy 1000 is not 5 x n_onpolicy_seeds 199")
    manifest = export_policy(ppo.path, ppo.path / "report_policy", report_only=True)
    assert manifest.claims.export_parity is not None
    assert manifest.claims.export_parity.vacuous is True


def test_a_relative_out_through_a_symlink_exits_2_before_any_write(
    ppo: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "out").symlink_to(tmp_path / "elsewhere")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["export", str(ppo.path), "--out", "out/policy"])
    assert result.exit_code == 2
    assert "refusing to write through the symlink 'out'" in result.output
    assert os.listdir(tmp_path / "elsewhere") == []
    with pytest.raises(ExportError, match="symlink"):
        export_policy(ppo.path, Path("out/policy"))


@pytest.mark.parametrize(
    "args",
    [
        ["compare", "submissions/pid.py", "-p", "balance-v1/ideal", "--json", "out/c.json"],
        ["policy", "pack", "m.onnx", "--challenge", "balance-v1", "--out", "out/p"],
        ["video", "--side-by-side", "a.npz", "b.npz", "--labels", "a", "b", "--out", "out/v.mp4"],
        ["train", "c.yaml", "--out", "out/run"],
        ["export", "run", "--out", "out/p"],
    ],
    ids=["compare", "pack", "video", "train", "export"],
)
def test_every_writing_command_refuses_a_symlinked_relative_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "out").symlink_to(tmp_path / "elsewhere")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "refusing to write through the symlink 'out'" in result.output
    assert os.listdir(tmp_path / "elsewhere") == []


# --- step 2: the files --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["best.json", "train_meta.json", "meta.json", "normalize.json", "actor.npz", "parity_ref.npz"],
)
def test_a_symlinked_input_is_refused(ppo: Run, tmp_path: Path, name: str) -> None:
    path = (ppo.path if name.endswith(("best.json", "train_meta.json")) else ppo.step_dir) / name
    real = tmp_path / f"real_{name}"
    path.rename(real)
    path.symlink_to(real)
    refused(ppo, tmp_path / "p", "symlink")


def test_a_hard_linked_input_is_refused(ppo: Run, tmp_path: Path) -> None:
    os.link(ppo.step_dir / "meta.json", tmp_path / "meta_copy.json")
    refused(ppo, tmp_path / "p", "hard links")


def test_a_fifo_input_is_refused_without_blocking(ppo: Run, tmp_path: Path, make_fifo: Any) -> None:
    (ppo.step_dir / "normalize.json").unlink()
    make_fifo(ppo.step_dir / "normalize.json")
    refused(ppo, tmp_path / "p", "not a regular file")


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (b'{"mean": [NaN]}', "NaN"),
        (b'{"mean": [Infinity]}', "NaN or Infinity"),
        (b'{"mean": [1e400]}', "beyond float range"),
        (b'{"a": 1, "a": 2}', "repeats a key"),
        (b"\xef\xbb\xbf{}", "byte order mark"),
        (b"[" * 40 + b"]" * 40, "nests too deeply"),
        (b'{"n": ' + b"9" * 21 + b"}", "more than 20 digits"),
        (b'{"a": "\xff"}', "UTF-8"),
        (b"{", "not valid JSON"),
    ],
)
def test_json_inputs_are_strict(ppo: Run, tmp_path: Path, text: bytes, match: str) -> None:
    (ppo.step_dir / "normalize.json").write_bytes(text)
    refused(ppo, tmp_path / "p", match)


def test_json_inputs_are_capped(ppo: Run, tmp_path: Path) -> None:
    (ppo.path / "train_meta.json").write_bytes(b" " * (64 * 1024 + 1))
    refused(ppo, tmp_path / "p", "larger than")


@pytest.mark.parametrize(
    ("file", "edit", "match"),
    [
        ("normalize.json", lambda d: d.update(mean=d["mean"][:9]), "10 entries"),
        ("normalize.json", lambda d: d.update(var=[-1.0] * 10), "var"),
        ("normalize.json", lambda d: d.update(epsilon=1e-6), "epsilon or clip_obs"),
        ("normalize.json", lambda d: d.update(clip_obs=5.0), "epsilon or clip_obs"),
        ("normalize.json", lambda d: d.update(extra=1), "extra"),
        ("normalize.json", lambda d: d.update(count=True), "count"),
        ("meta.json", lambda d: d.update(step=2048), "step"),
        ("meta.json", lambda d: d.update(algo="sac"), "algo"),
        ("meta.json", lambda d: d.update(net_arch=[8, 16]), "net_arch"),
        ("meta.json", lambda d: d.update(net_arch=[8, 8, 8]), "net_arch"),
        ("meta.json", lambda d: d.update(action_rule="tanh(mean)"), "action_rule"),
        ("meta.json", lambda d: d.update(config_sha256="3" * 64), "config_sha256"),
        ("meta.json", lambda d: d.update(onnx_sha256="4" * 64), "onnx_sha256_mismatch"),
        ("meta.json", lambda d: d.update(onnx_sha256="A" * 64), "onnx_sha256"),
        ("meta.json", lambda d: d.update(n_onpolicy=995), "shape"),
    ],
)
def test_snapshot_json_is_bound_to_the_run(
    ppo: Run, tmp_path: Path, file: str, edit: Any, match: str
) -> None:
    ppo.rewrite_json(f"ckpt/step_1024/{file}", edit)
    refused(ppo, tmp_path / "p", match)


def write_zip(path: Path, members: dict[str, bytes], *, prefix: bytes = b"") -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    path.write_bytes(prefix + buf.getvalue())


def npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, arr)
    return buf.getvalue()


def actor_members(run: Run) -> dict[str, bytes]:
    with np.load(run.step_dir / "actor.npz") as old:
        return {f"{k}.npy": npy(old[k]) for k in old.files}


def test_npz_with_a_ninth_member_is_refused(ppo: Run, tmp_path: Path) -> None:
    with np.load(ppo.step_dir / "parity_ref.npz") as old:
        members = {f"{k}.npy": npy(old[k]) for k in old.files}
    members.update({f"x{k}.npy": npy(np.zeros(1)) for k in range(5)})  # 9 members
    write_zip(ppo.step_dir / "parity_ref.npz", members)
    refused(ppo, tmp_path / "p", "unexpected members")


def test_npz_with_a_bad_name_is_refused(ppo: Run, tmp_path: Path) -> None:
    members = actor_members(ppo)
    members["W3.npy"] = members.pop("W2.npy")
    write_zip(ppo.step_dir / "actor.npz", members)
    refused(ppo, tmp_path / "p", "unexpected members")


def test_an_oversize_npz_file_is_refused_before_reading(ppo: Run, tmp_path: Path) -> None:
    (ppo.step_dir / "actor.npz").write_bytes(b"\0" * (MAX_NPZ_BYTES + 1))
    refused(ppo, tmp_path / "p", "larger than")


def test_members_declaring_more_than_8_mib_in_total_are_refused(ppo: Run, tmp_path: Path) -> None:
    big = np.zeros(1_500_000, np.float32)  # 6 MB each, deflated to a few KB
    members = {f"{k}.npy": npy(big) for k in ("W0", "b0", "W1", "b1", "W2", "b2")}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    assert len(buf.getvalue()) < 1 << 20
    (ppo.step_dir / "actor.npz").write_bytes(buf.getvalue())
    refused(ppo, tmp_path / "p", "in total")


def test_an_object_dtype_is_refused(ppo: Run, tmp_path: Path) -> None:
    members = actor_members(ppo)
    members["b0.npy"] = npy(np.array([1, "a"], dtype=object))
    write_zip(ppo.step_dir / "actor.npz", members)
    refused(ppo, tmp_path / "p", "Object arrays")


@pytest.mark.parametrize("shape", ["(4398046511104,)", "(2097152, 2097152)"])
def test_a_header_declaring_2_42_elements_allocates_nothing(
    ppo: Run, tmp_path: Path, shape: str
) -> None:
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }"
    header += " " * (63 - len(header) % 64) + "\n"
    lie = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + b"\0" * 16
    members = actor_members(ppo)
    members["W0.npy"] = lie
    write_zip(ppo.step_dir / "actor.npz", members)
    refused(ppo, tmp_path / "p", "must have shape")


def test_a_wrong_dtype_is_refused(ppo: Run, tmp_path: Path) -> None:
    members = actor_members(ppo)
    members["W1.npy"] = npy(np.zeros((8, 8), np.float64))
    write_zip(ppo.step_dir / "actor.npz", members)
    refused(ppo, tmp_path / "p", "expected dtype <f4")


def test_bytes_prepended_to_the_archive_are_refused(ppo: Run, tmp_path: Path) -> None:
    write_zip(ppo.step_dir / "actor.npz", actor_members(ppo), prefix=b"PK\x03\x04junk")
    refused(ppo, tmp_path / "p", "actor.npz")


def test_garbage_npz_is_a_clean_error(ppo: Run, tmp_path: Path) -> None:
    (ppo.step_dir / "parity_ref.npz").write_bytes(b"not a zip at all")
    refused(ppo, tmp_path / "p", "parity_ref.npz")


def test_a_bad_parity_sha_string_is_refused(ppo: Run, tmp_path: Path) -> None:
    ppo.rewrite_npz("parity_ref.npz", onnx_sha256=np.str_("Z" * 64))
    refused(ppo, tmp_path / "p", "onnx_sha256 is not a sha256")
    ppo.rewrite_npz("parity_ref.npz", onnx_sha256=np.str_("5" * 64))
    refused(ppo, tmp_path / "p", "parity_ref.npz does not record")


# --- step 3: build and bind ---------------------------------------------------------------------


def test_a_changed_weight_breaks_the_sha_binding(ppo: Run, tmp_path: Path) -> None:
    w = ppo.weights.layers[1][0].copy()
    w[0, 0] = np.nextafter(w[0, 0], np.float32(10))  # one ulp
    ppo.rewrite_npz("actor.npz", W1=w)
    refused(ppo, tmp_path / "p", "onnx_sha256_mismatch")


def test_the_constant_binding_catches_a_wrong_epsilon_that_parity_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A builder that used epsilon 1e-6 instead of VecNormalize's 1e-8 (§3.3): with small heads
    its actions differ from SB3's by about 1e-6, under the parity tolerance, so only the
    bitwise constant binding catches it."""
    run = make_run(tmp_path, head_scale=0.05)

    def wrong_epsilon(w: Any, norm: ObsNorm, **kwargs: Any) -> bytes:
        return build_policy_onnx(w, ObsNorm(norm.mean, norm.var, 1e-6, norm.clip), **kwargs)

    monkeypatch.setattr(export_module, "build_policy_onnx", wrong_epsilon)
    data = export_module.build_onnx(
        run.weights.layers, run.norm, activation="tanh", challenge=load_challenge("balance-v1")
    )
    sha = hashlib.sha256(data).hexdigest()
    worst = float(np.abs(onnx_run(data, run.obs).astype(np.float64) - run.action_ref).max())
    assert 0 < worst < 1e-5  # parity alone would pass
    run.rewrite_json("ckpt/step_1024/meta.json", lambda d: d.update(onnx_sha256=sha))
    run.rewrite_json("train_meta.json", lambda d: d["selection_history"][1].update(onnx_sha256=sha))
    run.rewrite_npz("parity_ref.npz", onnx_sha256=np.str_(sha))
    with pytest.raises(ExportError, match=r"constant_binding: initializers c1 \(inv_std\)"):
        export_policy(run.path, run.path / "report_policy", report_only=True)


def test_constant_mismatches_name_each_position(ppo: Run) -> None:
    import onnx

    task = export_module.policy_task(load_challenge("balance-v1"))
    data = export_module.build_onnx(
        ppo.weights.layers, ppo.norm, activation="tanh", challenge=load_challenge("balance-v1")
    )
    proto = onnx.ModelProto.FromString(data)
    assert constant_mismatches(proto, ppo.weights.layers, ppo.norm, task) == []
    clipped = ObsNorm(ppo.norm.mean, ppo.norm.var, ppo.norm.epsilon, 5.0)
    assert constant_mismatches(proto, ppo.weights.layers, clipped, task) == [
        "c2 (obs_lo)",
        "c3 (obs_hi)",
    ]
    del proto.graph.initializer[-1]
    assert constant_mismatches(proto, ppo.weights.layers, ppo.norm, task) == [
        "11 initializers, not 12"
    ]
    assert CONSTANT_NAMES[4:10] == ("W0", "b0", "W1", "b1", "W2", "b2")


# --- step 4: parity -----------------------------------------------------------------------------


def test_a_parity_miss_fails_even_for_a_report(ppo: Run, tmp_path: Path) -> None:
    action_ref = ppo.action_ref.copy()
    action_ref[1500, 1] += 2e-5
    ppo.rewrite_npz("parity_ref.npz", action_ref=action_ref)
    with pytest.raises(ExportError, match=r"max \|a_onnx - a_ref\| is 2e-05"):
        export_policy(ppo.path, ppo.path / "report_policy", report_only=True)


def test_a_nan_onnx_output_on_a_middle_row_fails_parity(
    ppo: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NaN is never <= anything: a NaN action on row 1,500 must become the worst difference,
    not be skipped by a ``diff > worst`` comparison."""
    real = export_module.make_session

    class NanOnRow1500:
        def __init__(self, validated: Any) -> None:
            self.session = real(validated)
            self.calls = 0

        def run(self, names: Any, feeds: Any) -> Any:
            out = self.session.run(names, feeds)
            self.calls += 1
            if self.calls == 1501:
                out[0][0, 1] = np.nan
            return out

    monkeypatch.setattr(export_module, "make_session", NanOnRow1500)
    with pytest.raises(ExportError, match=r"max \|a_onnx - a_ref\| is nan"):
        export_policy(ppo.path, ppo.path / "report_policy", report_only=True)


def test_nan_parity_rows_are_refused(ppo: Run, tmp_path: Path) -> None:
    action_ref = ppo.action_ref.copy()
    action_ref[3, 0] = np.nan
    ppo.rewrite_npz("parity_ref.npz", action_ref=action_ref)
    refused(ppo, tmp_path / "p", "finite")


def test_coverage_counts_at_the_thresholds() -> None:
    """Unsaturated: max_i |a_ref_i| < 0.99; saturated: >= 0.999 (§3.3 step 4)."""
    amax = [0.0, 0.98999999, 0.99, 0.995, 0.99899999, 0.999, 1.0]
    action_ref = np.array([[a, -a / 2] for a in amax])
    assert coverage_counts(action_ref) == (2, 2)
    assert coverage_counts(-action_ref[:, ::-1]) == (2, 2)  # either sign, either action


@pytest.mark.parametrize("where", [1000, 2000, 2063])
def test_random_and_edge_rows_must_be_the_spec_sets(ppo: Run, tmp_path: Path, where: int) -> None:
    obs = ppo.obs.copy()
    obs[where, 0] = np.nextafter(obs[where, 0], np.float32(np.inf))
    ppo.rewrite_npz("parity_ref.npz", obs=obs)
    refused(ppo, tmp_path / "p", "random and edge rows")


def test_set_labels_must_be_in_order(ppo: Run, tmp_path: Path) -> None:
    sets = ppo.sets.copy()
    sets[0] = 1
    ppo.rewrite_npz("parity_ref.npz", set=sets)
    refused(ppo, tmp_path / "p", "set must label")


# --- step 5: fixtures ---------------------------------------------------------------------------


def test_the_fixture_files(ppo: Run, tmp_path: Path) -> None:
    fixtures = tmp_path / "data"
    manifest = export_policy(
        ppo.path, tmp_path / "p", test_fixture="ppo_tiny", fixture_dir=fixtures
    )
    assert sorted(os.listdir(fixtures)) == [
        "ppo_tiny_parity.bin",
        "ppo_tiny_parity.json",
        "ppo_tiny_parity.npz",
    ]
    assert (fixtures / "ppo_tiny_parity.npz").read_bytes() == (
        ppo.step_dir / "parity_ref.npz"
    ).read_bytes()
    raw = (fixtures / "ppo_tiny_parity.bin").read_bytes()
    magic, version, n, d, a = BIN_HEADER.unpack_from(raw)
    assert (magic, version, n, d, a) == (b"TPAR", 1, 2064, 10, 2)
    at = BIN_HEADER.size
    obs = np.frombuffer(raw, "<f4", n * d, at).reshape(n, d)
    at += 4 * n * d
    a_ref = np.frombuffer(raw, "<f4", n * a, at).reshape(n, a)
    at += 4 * n * a
    sets = np.frombuffer(raw, "u1", n, at)
    at += n
    assert np.array_equal(obs, ppo.obs)
    assert np.array_equal(a_ref, ppo.action_ref.astype(np.float32))
    assert np.array_equal(sets, ppo.sets)
    assert len(raw) % 4 == 0
    assert raw[at:] == b"\0" * (len(raw) - at)
    assert len(raw) - at < 4
    sidecar = json.loads((fixtures / "ppo_tiny_parity.json").read_text())
    parity = manifest.claims.export_parity
    assert parity is not None
    assert sidecar == {
        "onnx_sha256": ppo.sha256,
        "n": 2064,
        "d": 10,
        "a": 2,
        "tolerance": 1e-5,
        "n_unsaturated": parity.n_unsaturated,
        "n_saturated": parity.n_saturated,
        "policy_sha256": manifest.onnx.sha256,
    }
    assert list(sidecar) == [
        "onnx_sha256",
        "n",
        "d",
        "a",
        "tolerance",
        "n_unsaturated",
        "n_saturated",
        "policy_sha256",
    ]


def test_a_fixture_target_that_is_a_directory_is_refused_before_anything_is_written(
    ppo: Run, tmp_path: Path
) -> None:
    out = tmp_path / "p"
    fixtures = tmp_path / "data"
    export_policy(ppo.path, out, test_fixture="t", fixture_dir=fixtures)
    before = {name: os.stat(out / name).st_ino for name in os.listdir(out)}
    npz_before = (fixtures / "t_parity.npz").stat().st_ino
    (fixtures / "t_parity.bin").unlink()
    (fixtures / "t_parity.bin").mkdir()
    with pytest.raises(ExportError, match=r"t_parity\.bin is a directory") as info:
        export_policy(ppo.path, out, test_fixture="t", fixture_dir=fixtures, replace=True)
    assert ".tmp" not in str(info.value)
    assert {name: os.stat(out / name).st_ino for name in os.listdir(out)} == before
    assert (fixtures / "t_parity.npz").stat().st_ino == npz_before
    assert sorted(os.listdir(fixtures)) == ["t_parity.bin", "t_parity.json", "t_parity.npz"]


def test_a_fixture_that_cannot_be_written_leaves_out_and_the_fixtures_alone(
    ppo: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture files are written (to temp names) before ``--out`` is swapped: a failure
    there replaces nothing, and the error names the fixture, not its temp file."""
    out = tmp_path / "p"
    fixtures = tmp_path / "data"
    export_policy(ppo.path, out, test_fixture="t", fixture_dir=fixtures)
    before = {name: os.stat(out / name).st_ino for name in os.listdir(out)}
    fixture_inodes = {name: os.stat(fixtures / name).st_ino for name in os.listdir(fixtures)}
    real = Path.write_bytes

    def full_disk(self: Path, data: Any) -> int:
        if ".t_parity.json." in self.name:
            raise OSError(28, "No space left on device", str(self))
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", full_disk)
    with pytest.raises(ExportError, match=r"cannot write .*t_parity\.json: No space") as info:
        export_policy(ppo.path, out, test_fixture="t", fixture_dir=fixtures, replace=True)
    assert ".tmp" not in str(info.value)
    assert {name: os.stat(out / name).st_ino for name in os.listdir(out)} == before
    assert {name: os.stat(fixtures / name).st_ino for name in os.listdir(fixtures)} == (
        fixture_inodes
    )
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_a_fixture_rename_that_fails_after_the_swap_says_what_was_written(
    ppo: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The renames after the swap only fail on a race (the targets are checked first): the
    error then says --out was written, and no temp file is left."""
    out = tmp_path / "p"
    fixtures = tmp_path / "data"
    real = os.replace

    def racing(src: Any, dst: Any) -> None:
        if str(dst).endswith("t_parity.json"):
            raise IsADirectoryError(21, "Is a directory", str(dst))
        real(src, dst)

    monkeypatch.setattr(os, "replace", racing)
    with pytest.raises(ExportError, match=r"wrote .*p, but cannot finish the fixture files"):
        export_policy(ppo.path, out, test_fixture="t", fixture_dir=fixtures)
    assert (out / "policy.onnx").exists()
    assert [p.name for p in fixtures.iterdir() if p.name.startswith(".")] == []


def test_a_failed_cleanup_after_the_swap_is_a_warning(
    ppo: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the new policy is renamed in, the export has happened: an old directory that
    cannot be removed is reported as a warning (the CLI prints it), not as a failed export."""
    out = tmp_path / "p"
    export_policy(ppo.path, out)
    old = os.stat(out / "policy.onnx").st_ino

    def stuck(path: Path) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(export_module, "_remove", stuck)
    with pytest.warns(ExportWarning, match=r"cannot remove the old one .*Permission denied"):
        export_policy(ppo.path, out, replace=True)
    assert os.stat(out / "policy.onnx").st_ino != old
    result = runner.invoke(app, ["export", str(ppo.path), "--out", str(out), "--replace"])
    assert result.exit_code == 0, result.output
    assert "warning: replaced" in result.output


def test_the_bin_layout_pads_to_4_bytes() -> None:
    for n in (1, 2, 3, 4):
        raw = parity_bin(np.zeros((n, 3), np.float32), np.zeros((n, 2)), np.zeros(n, np.uint8))
        assert len(raw) == 20 + 12 * n + 8 * n + n + (-n % 4)


def test_fixtures_need_replace_to_overwrite(ppo: Run, tmp_path: Path) -> None:
    fixtures = tmp_path / "data"
    export_policy(ppo.path, tmp_path / "p", test_fixture="t", fixture_dir=fixtures)
    refused(ppo, tmp_path / "q", "exist", test_fixture="t", fixture_dir=fixtures)
    export_policy(ppo.path, tmp_path / "p", test_fixture="t", fixture_dir=fixtures, replace=True)


@pytest.mark.parametrize("name", ["", "Bad", "a-b", "../x", "x" * 33, "ok\n"])
def test_a_bad_fixture_name_is_refused(ppo: Run, tmp_path: Path, name: str) -> None:
    refused(ppo, tmp_path / "p", "test-fixture", test_fixture=name, fixture_dir=tmp_path / "d")
    assert not (tmp_path / "d").exists()


def test_the_cli_refuses_a_bad_fixture_name(ppo: Run, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["export", str(ppo.path), "--out", str(tmp_path / "p"), "--test-fixture", "Bad-Name"]
    )
    assert result.exit_code == 2
    assert "test-fixture" in result.output
    assert not (tmp_path / "p").exists()


def test_a_fixture_needs_a_shippable_export(ppo: Run, tmp_path: Path) -> None:
    refused(
        ppo,
        ppo.path / "report_policy",
        "shippable",
        report_only=True,
        test_fixture="x",
        fixture_dir=tmp_path / "d",
    )


def test_read_run_json_is_the_strict_reader(tmp_path: Path) -> None:
    path = tmp_path / "train_meta.json"
    path.write_text('{"flags": ["wall_clock_abort"]}')
    assert read_run_json(path) == {"flags": ["wall_clock_abort"]}
    path.write_text('{"flags": NaN}')
    with pytest.raises(ExportError, match="NaN"):
        read_run_json(path)


def test_what_is_not_a_run_is_refused(ppo: Run, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="not a run directory"):
        export_policy(tmp_path / "missing", tmp_path / "p")
    (ppo.path / "best.json").write_text("[1024]")
    refused(ppo, tmp_path / "p", "best.json: must be a JSON object")


def test_a_step_path_that_is_a_file_is_refused(ppo: Run, tmp_path: Path) -> None:
    (ppo.path / "ckpt" / "step_7").write_text("x")
    refused(ppo, tmp_path / "p", "not a directory", step=7)


def test_an_output_path_that_cannot_be_examined_is_refused(
    ppo: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("plain").write_text("x")
    with pytest.raises(ExportError, match="cannot resolve the output"):
        export_policy(ppo.path, Path("plain/policy"))


def test_too_few_unsaturated_rows_is_a_coverage_miss(tmp_path: Path) -> None:
    run = make_run(tmp_path, head_scale=20.0)  # 11 of 2,064 rows stay unsaturated
    with pytest.raises(ExportError, match="unsaturated rows"):
        export_policy(run.path, tmp_path / "p")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_weights_are_a_clean_error(ppo: Run, tmp_path: Path, bad: float) -> None:
    w = ppo.weights.layers[0][0].copy()
    w[0, 0] = bad
    ppo.rewrite_npz("actor.npz", W0=w)
    refused(ppo, tmp_path / "p", "cannot build the policy: layer 0: weights must be finite")


def test_a_mean_beyond_float32_is_a_clean_error(ppo: Run, tmp_path: Path) -> None:
    ppo.rewrite_json("ckpt/step_1024/normalize.json", lambda d: d["mean"].__setitem__(0, 1e39))
    refused(ppo, tmp_path / "p", "cannot build the policy")


def test_integers_are_read_as_floats_in_normalize_json(ppo: Run, tmp_path: Path) -> None:
    ppo.rewrite_json("ckpt/step_1024/normalize.json", lambda d: d.update(count=2048))
    export_policy(ppo.path, tmp_path / "p")


def test_at_most_16_versions_become_claims(ppo: Run, tmp_path: Path) -> None:
    versions = {f"k{i:02d}": f"{i}.0" for i in range(17)}
    ppo.rewrite_json("train_meta.json", lambda d: d.update(versions=versions))
    claims = export_policy(ppo.path, tmp_path / "p").claims
    assert claims.versions == {f"k{i:02d}": f"{i}.0" for i in range(16)}


def test_a_config_error_in_train_meta_is_one_bounded_clean_line(ppo: Run, tmp_path: Path) -> None:
    evil = "x\n::error::pwned\x1b[2J" + "A" * 20_000
    ppo.rewrite_json("train_meta.json", lambda d: d["config"].update(algo=evil))
    with pytest.raises(ExportError) as info:
        export_policy(ppo.path, tmp_path / "p")
    message = str(info.value)
    assert message.startswith("train_meta.json: config: ")
    assert len(message) <= 2000
    assert message.isprintable()


def test_the_npz_reader_takes_export_dtypes_only() -> None:
    buf = io.BytesIO()
    np.savez(buf, n=np.zeros(3, np.int64))
    with pytest.raises(ValueError, match="n: dtype <i8 is not an export input dtype"):
        read_npz_bytes(buf.getvalue(), {"n": ("<i8", (3,))}, max_bytes=MAX_NPZ_BYTES)
    good = io.BytesIO()
    np.savez(good, n=np.zeros(3, np.float64))
    assert read_npz_bytes(good.getvalue(), {"n": ("<f8", (3,))}, max_bytes=1 << 20)["n"].shape == (
        3,
    )


# --- the snapshot writer ------------------------------------------------------------------------


def test_the_snapshot_writer_round_trips_through_export(ppo: Run, tmp_path: Path) -> None:
    """``run_support`` writes every run with ``snapshot_files``; its files are exactly what export
    reads, and writing the same snapshot twice gives the same bytes (training is deterministic)."""
    cfg = validate_train_config(ppo.config)
    onpolicy = ppo.obs[: int((ppo.sets == 0).sum())]
    args = (cfg, ppo.step, ppo.weights.layers, ppo.norm)
    kwargs: dict[str, Any] = {
        "count": 2048.0001,
        "onpolicy": onpolicy,
        "action_ref": ppo.action_ref,
        "onnx_sha256": ppo.sha256,
    }
    files = snapshot_files(*args, **kwargs)
    assert sorted(files) == ["actor.npz", "meta.json", "normalize.json", "parity_ref.npz"]
    assert files == snapshot_files(*args, **kwargs)
    for name, data in files.items():
        assert (ppo.step_dir / name).read_bytes() == data, name
    meta = json.loads(files["meta.json"])
    assert (meta["n_onpolicy"], meta["n_onpolicy_seeds"], meta["net_arch"]) == (1000, 200, [8, 8])
    with pytest.raises(FileExistsError):
        write_snapshot(ppo.path / "ckpt", ppo.step, files)
    with pytest.raises(ValueError, match="action_ref"):
        snapshot_files(*args, **{**kwargs, "action_ref": ppo.action_ref[:-1]})
    with pytest.raises(ValueError, match="onnx_sha256"):
        snapshot_files(*args, **{**kwargs, "onnx_sha256": "x"})
    with pytest.raises(ValueError, match="groups of 5"):
        snapshot_files(*args, **{**kwargs, "onpolicy": onpolicy[:-1]})
    wide = (
        (np.zeros((9, 10), np.float32), np.zeros(9, np.float32)),
        *ppo.weights.layers[1:],
    )
    with pytest.raises(ValueError, match="W0"):
        snapshot_files(cfg, ppo.step, wide, ppo.norm, **kwargs)


def test_write_snapshot_is_all_or_nothing(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    (ckpt / ".step_7.tmp").mkdir(parents=True)  # a stale one from a crash
    (ckpt / ".step_7.tmp" / "junk").write_text("x")

    def broken(directory: Path) -> None:
        (directory / "model.zip").write_bytes(b"half")
        raise OSError(28, "No space left on device")

    with pytest.raises(OSError, match="No space"):
        write_snapshot(ckpt, 7, {"meta.json": b"{}"}, more=broken)
    assert os.listdir(ckpt) == []

    def sb3(directory: Path) -> None:
        (directory / "model.zip").write_bytes(b"zip")

    path = write_snapshot(ckpt, 7, {"meta.json": b"{}"}, more=sb3)
    assert path == ckpt / "step_7"
    assert sorted(os.listdir(path)) == ["meta.json", "model.zip"]
    assert os.listdir(ckpt) == ["step_7"]
