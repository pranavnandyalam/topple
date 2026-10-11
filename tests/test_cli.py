import json
import tomllib
from collections.abc import Callable
from pathlib import Path

import mujoco
import pytest
from typer.testing import CliRunner

from topple import __version__
from topple.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"topple {__version__}"


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert __version__ == pyproject["project"]["version"]


@pytest.mark.render
def test_smoke_command(tmp_path: Path) -> None:
    out = tmp_path / "smoke.png"
    result = runner.invoke(app, ["smoke", "--out", str(out), "--steps", "10"])
    assert result.exit_code == 0, result.output
    assert "smoke ok: 10 steps" in result.output
    assert out.exists()


def test_smoke_command_rejects_zero_steps() -> None:
    result = runner.invoke(app, ["smoke", "--steps", "0"])
    assert result.exit_code == 2


def test_smoke_command_reports_rendering_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delattr(mujoco, "Renderer", raising=False)
    result = runner.invoke(app, ["smoke", "--out", str(tmp_path / "s.png"), "--steps", "1"])
    assert result.exit_code == 1
    assert "headless rendering unavailable" in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


PID_SUBMISSION = """
from topple.controllers.pid import PIDController

def make_controller(spec):
    return PIDController()
"""


def _submission(tmp_path: Path, text: str = PID_SUBMISSION) -> Path:
    path = tmp_path / "bot.py"
    path.write_text(text)
    return path


def test_run_writes_results_and_trajectory(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "run",
            str(_submission(tmp_path)),
            "--protocol",
            "balance-v1/clean",
            "--seed",
            "7",
            "--out",
            str(out),
            "--no-video",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "results.json").exists()
    assert (out / "traj_000.npz").exists()
    assert "balance-v1/clean: score" in result.output
    assert "seed      7" in result.output


def test_eval_alias_protocol_all_seeds(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    result = runner.invoke(
        app,
        [
            "eval",
            str(_submission(tmp_path)),
            "--protocol",
            "eval_v1_clean",
            "--out",
            str(out),
            "--videos",
            "none",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("traj_*.npz"))) == 20
    assert "balance-v1/clean: score" in result.output
    assert json.loads((out / "results.json").read_text())["complete"] is True


def test_verify_and_score_commands(tmp_path: Path) -> None:
    out = tmp_path / "run"
    runner.invoke(
        app,
        [
            "run",
            str(_submission(tmp_path)),
            "--protocol",
            "balance-v1/standard",
            "--out",
            str(out),
            "--no-video",
        ],
    )
    traj = out / "traj_000.npz"
    verified = runner.invoke(app, ["verify", str(traj)])
    assert verified.exit_code == 0, verified.output
    assert verified.output.startswith("OK: reproduced bit-exactly")
    assert "UNBOUND" not in verified.output
    scored = runner.invoke(app, ["score", str(traj)])
    assert scored.exit_code == 0, scored.output
    payload = json.loads(scored.stdout)
    results = json.loads((out / "results.json").read_text())
    assert payload["ok"] is True
    assert payload["bound"] is True
    assert payload["score"] == results["episodes"][0]["score"]
    assert payload["metrics"] == results["episodes"][0]["metrics"]


def test_verify_detects_tampering(tmp_path: Path) -> None:
    from topple.traj import Trajectory, load_npz

    out = tmp_path / "run"
    runner.invoke(
        app,
        [
            "run",
            str(_submission(tmp_path)),
            "--protocol",
            "balance-v1/clean",
            "--out",
            str(out),
            "--no-video",
        ],
    )
    traj = load_npz(out / "traj_000.npz", max_steps=1000)
    qpos = traj.qpos.copy()
    qpos[-1, 0] += 0.01
    Trajectory(**{**traj.__dict__, "qpos": qpos}).save_npz(out / "forged.npz")
    result = runner.invoke(app, ["verify", str(out / "forged.npz")])
    assert result.exit_code == 1
    assert result.output.startswith("FAIL")


def test_load_error_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(_submission(tmp_path, "x = 1\n")), "--no-video"])
    assert result.exit_code == 2
    assert "make_controller" in result.output


@pytest.mark.parametrize(
    ("text", "detail"),
    [
        ("import sys\nsys.exit(7)\n", "SystemExit: 7"),
        (
            "class Meta(type):\n"
            "    @property\n"
            "    def __name__(cls):\n"
            "        raise RuntimeError('no name for you')\n"
            "class Boom(Exception, metaclass=Meta):\n"
            "    pass\n"
            "raise Boom('at import')\n",
            "Boom: at import",
        ),
    ],
    ids=["sys.exit at import", "exception whose class name raises"],
)
def test_a_hostile_import_exits_2_without_a_traceback(
    tmp_path: Path, text: str, detail: str
) -> None:
    result = runner.invoke(app, ["run", str(_submission(tmp_path, text)), "--no-video"])
    assert result.exit_code == 2
    assert f"importing bot.py failed: {detail}" in result.output
    assert "Traceback" not in result.output


def test_bad_protocol_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", str(_submission(tmp_path)), "--protocol", "balance-v1/hard", "--no-video"]
    )
    assert result.exit_code == 2
    assert "no preset" in result.output


# ------------------------------------------------------------------ run: seeds and completeness


@pytest.mark.parametrize("seed", ["-1", str(2**53), "abc"])
def test_run_rejects_seeds_outside_the_contract(tmp_path: Path, seed: str) -> None:
    result = runner.invoke(
        app,
        ["run", str(_submission(tmp_path)), "--seed", seed, "--no-video", "--out", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert "--seed" in result.output
    assert not (tmp_path / "results.json").exists()


def test_a_single_seed_run_is_not_complete(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "run",
            str(_submission(tmp_path)),
            "-p",
            "balance-v1/clean",
            "--seed",
            str(2**53 - 1),
            "--out",
            str(out),
            "--no-video",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads((out / "results.json").read_text())
    assert data["complete"] is False
    assert data["episodes"][0]["seed"] == 2**53 - 1


def test_run_into_an_unwritable_place_exits_2(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    result = runner.invoke(
        app, ["run", str(_submission(tmp_path)), "--no-video", "--out", str(blocker / "run")]
    )
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert "error:" in result.output


def test_unknown_challenge_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", str(_submission(tmp_path)), "-p", "nope-v1/standard", "--no-video"]
    )
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert "nope-v1" in result.output


# ------------------------------------------------------------------ verify / score (spec A)


def _run(tmp_path: Path, *args: str) -> Path:
    out = tmp_path / "run"
    result = runner.invoke(
        app, ["run", str(_submission(tmp_path)), "--out", str(out), "--no-video", *args]
    )
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture(scope="module")
def eval_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A full `topple eval` of balance-v1/clean (all 20 public seeds)."""
    root = tmp_path_factory.mktemp("eval")
    out = root / "eval"
    result = runner.invoke(
        app,
        [
            "eval",
            str(_submission(root)),
            "-p",
            "balance-v1/clean",
            "--out",
            str(out),
            "--videos",
            "none",
        ],
    )
    assert result.exit_code == 0, result.output
    return out


def _copy_run(src: Path, dst: Path) -> Path:
    import shutil

    shutil.copytree(src, dst)
    return dst


def test_verify_a_run_directory(eval_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "trusted.json"
    result = runner.invoke(app, ["verify", str(eval_dir), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "OK: all 20 episodes reproduced" in result.output
    trusted = json.loads(out.read_text())
    original = json.loads((eval_dir / "results.json").read_text())
    assert trusted["complete"] is True
    assert trusted["aggregate"] == original["aggregate"]
    assert trusted["episodes"] == original["episodes"]
    assert trusted["controller"] == {"id": "unknown", "sha256": "0" * 64}


def test_verify_a_directory_records_the_submission_hash(eval_dir: Path, tmp_path: Path) -> None:
    from topple.loader import submission_sha256

    sub = _submission(tmp_path)
    out = tmp_path / "trusted.json"
    result = runner.invoke(
        app,
        [
            "verify",
            str(eval_dir),
            "--out",
            str(out),
            "--submission",
            str(sub),
            "--isolation",
            "container",
        ],
    )
    assert result.exit_code == 0, result.output
    trusted = json.loads(out.read_text())
    assert trusted["controller"] == {"id": "bot", "sha256": submission_sha256(sub)}
    assert trusted["isolation"] == "container"


def test_verify_a_directory_with_a_missing_episode(eval_dir: Path, tmp_path: Path) -> None:
    run = _copy_run(eval_dir, tmp_path / "copy")
    (run / "traj_004.npz").unlink()
    out = tmp_path / "trusted.json"
    result = runner.invoke(app, ["verify", str(run), "--out", str(out)])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
    trusted = json.loads(out.read_text())
    assert trusted["complete"] is False
    assert trusted["episodes"][4]["error"]["code"] == "missing"


def test_verify_a_directory_with_a_forged_episode(eval_dir: Path, tmp_path: Path) -> None:
    from dataclasses import replace

    from topple.traj import load_npz

    run = _copy_run(eval_dir, tmp_path / "copy")
    traj = load_npz(run / "traj_002.npz", max_steps=1000)
    replace(traj, meta={**traj.meta, "robot_hash": "0" * 64}).save_npz(run / "traj_002.npz")
    result = runner.invoke(app, ["verify", str(run)])
    assert result.exit_code == 1
    assert "FAIL: 1 of 20 episodes failed verification" in result.output


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["traj_020.npz"], "traj_020.npz"),
        (["traj_1.npz", "traj_001.npz"], "episode 1"),
    ],
)
def test_verify_a_directory_with_bad_names(
    eval_dir: Path, tmp_path: Path, names: list[str], message: str
) -> None:
    run = _copy_run(eval_dir, tmp_path / "copy")
    for name in names:
        (run / name).write_bytes((eval_dir / "traj_001.npz").read_bytes())
    result = runner.invoke(app, ["verify", str(run)])
    assert result.exit_code == 2
    assert message in result.output
    assert "Traceback" not in result.output


def test_verify_an_empty_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["verify", str(tmp_path)])
    assert result.exit_code == 2
    assert "traj_" in result.output


def test_verify_a_directory_rejects_index_and_seed(eval_dir: Path) -> None:
    result = runner.invoke(app, ["verify", str(eval_dir), "--index", "0"])
    assert result.exit_code == 2


def test_verify_a_local_seed_is_unbound(tmp_path: Path) -> None:
    out = _run(tmp_path, "-p", "balance-v1/clean", "--seed", "7")
    result = runner.invoke(app, ["verify", str(out / "traj_000.npz")])
    assert result.exit_code == 0, result.output
    # The first line alone tells a protocol-bound pass from an unbound one.
    assert result.output.startswith("OK (UNBOUND): reproduced bit-exactly")
    assert "UNBOUND (seed taken from file)" in result.output
    explicit = runner.invoke(app, ["verify", str(out / "traj_000.npz"), "--seed", "7"])
    assert explicit.exit_code == 0, explicit.output
    assert explicit.output.startswith("OK (UNBOUND)")
    assert "UNBOUND (seed from --seed)" in explicit.output
    scored = runner.invoke(app, ["score", str(out / "traj_000.npz"), "--seed", "7"])
    assert scored.exit_code == 0, scored.output
    payload = json.loads(scored.stdout)
    assert (payload["ok"], payload["bound"], payload["seed"]) == (True, False, 7)
    bound = runner.invoke(app, ["verify", str(out / "traj_000.npz"), "--index", "0"])
    assert bound.exit_code == 1
    assert "scenario" in bound.output


def test_verify_writes_single_episode_results(tmp_path: Path) -> None:
    out = _run(tmp_path, "-p", "balance-v1/clean")
    trusted = tmp_path / "trusted.json"
    result = runner.invoke(app, ["verify", str(out / "traj_000.npz"), "--out", str(trusted)])
    assert result.exit_code == 0, result.output
    data = json.loads(trusted.read_text())
    original = json.loads((out / "results.json").read_text())
    assert data["complete"] is False
    assert data["episodes"] == original["episodes"]


@pytest.mark.parametrize(
    "args",
    [["--index", "0", "--seed", "1"], ["--index", "20"], ["--index", "-1"], ["--seed", "-3"]],
)
def test_verify_binding_flags_are_validated(tmp_path: Path, args: list[str]) -> None:
    out = _run(tmp_path, "-p", "balance-v1/clean")
    result = runner.invoke(app, ["verify", str(out / "traj_000.npz"), *args])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "content", [b"", b"junk", b"PK\x03\x04garbage"], ids=["empty", "text", "broken zip"]
)
@pytest.mark.parametrize("command", ["verify", "score"])
def test_unreadable_trajectories_exit_2(tmp_path: Path, content: bytes, command: str) -> None:
    path = tmp_path / "traj_000.npz"
    path.write_bytes(content)
    result = runner.invoke(app, [command, str(path)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert "traj_000.npz" in result.output


@pytest.mark.parametrize("command", ["verify", "score"])
def test_missing_path_exits_2(tmp_path: Path, command: str) -> None:
    result = runner.invoke(app, [command, str(tmp_path / "nope.npz")])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "meta",
    [
        {"challenge": "nope-v1", "preset": "standard"},
        {"challenge": "../../etc", "preset": "standard"},
        {"challenge": "balance-v1", "preset": "hard"},
        {"challenge": 5, "preset": "standard"},
        {},
    ],
    ids=["unknown challenge", "path traversal", "unknown preset", "not a string", "absent"],
)
def test_bad_protocol_in_a_file_exits_2(tmp_path: Path, meta: dict[str, object]) -> None:
    from dataclasses import replace

    from topple.traj import load_npz

    out = _run(tmp_path, "-p", "balance-v1/clean")
    traj = load_npz(out / "traj_000.npz", max_steps=1000)
    kept = {k: v for k, v in traj.meta.items() if k not in ("challenge", "preset")}
    replace(traj, meta={**kept, **meta}).save_npz(out / "bad.npz")
    result = runner.invoke(app, ["verify", str(out / "bad.npz")])
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert "error:" in result.output


def test_a_crash_inside_verification_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import topple.verify

    out = _run(tmp_path, "-p", "balance-v1/clean")

    def crash(*args: object, **kwargs: object) -> None:
        raise RuntimeError("verifier bug")

    monkeypatch.setattr(topple.verify, "verify_episode", crash)
    for command in ("verify", "score"):
        result = runner.invoke(app, [command, str(out / "traj_000.npz")])
        assert result.exit_code == 2
        assert "verification crashed: RuntimeError: verifier bug" in result.output
        assert "Traceback" not in result.output


def test_score_a_forgery_exits_1_with_score_0(tmp_path: Path) -> None:
    from dataclasses import replace

    import numpy as np

    from topple.traj import load_npz

    out = _run(tmp_path, "-p", "balance-v1/standard")
    traj = load_npz(out / "traj_000.npz", max_steps=1000)
    replace(traj, push_force=np.zeros_like(traj.push_force)).save_npz(out / "forged.npz")
    result = runner.invoke(app, ["score", str(out / "forged.npz")])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["code"] == "not_reproducible"
    assert payload["score"] == 0.0
    assert payload["error"]["code"] == "not_reproducible"
    assert "push_force" in payload["reason"]


def test_score_an_errored_episode_prints_0(tmp_path: Path) -> None:
    text = (
        "class Bot:\n"
        "    def reset(self, rng_seed):\n        self.k = 0\n"
        "    def act(self, obs):\n"
        "        self.k += 1\n"
        "        if self.k > 30:\n            raise RuntimeError('bug')\n"
        "        return (0.0, 0.0)\n"
        "def make_controller(spec):\n    return Bot()\n"
    )
    out = tmp_path / "run"
    ran = runner.invoke(
        app, ["run", str(_submission(tmp_path, text)), "--out", str(out), "--no-video"]
    )
    assert ran.exit_code == 0, ran.output
    result = runner.invoke(app, ["score", str(out / "traj_000.npz")])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["score"] == 0.0
    assert payload["error"]["code"] == "exception"


def test_score_takes_one_file(eval_dir: Path) -> None:
    result = runner.invoke(app, ["score", str(eval_dir)])
    assert result.exit_code == 2
    assert "verify" in result.output


def test_a_crash_inside_rescore_exits_2(eval_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import topple.verify

    def crash(*args: object, **kwargs: object) -> None:
        raise MemoryError("verifier ran out")

    monkeypatch.setattr(topple.verify, "rescore", crash)
    result = runner.invoke(app, ["verify", str(eval_dir)])
    assert result.exit_code == 2
    assert "verification crashed: MemoryError: verifier ran out" in result.output


def test_a_directory_with_an_unloadable_file_exits_2(eval_dir: Path, tmp_path: Path) -> None:
    run = _copy_run(eval_dir, tmp_path / "copy")
    (run / "traj_007.npz").write_bytes(b"junk")
    result = runner.invoke(app, ["verify", str(run)])
    assert result.exit_code == 2
    assert "traj_007.npz" in result.output
    assert "Traceback" not in result.output


def test_verify_with_an_unhashable_submission_exits_2(eval_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["verify", str(eval_dir), "--submission", str(tmp_path / "missing.py")]
    )
    assert result.exit_code == 2
    assert "cannot hash" in result.output


def test_verify_with_an_explicit_protocol(eval_dir: Path) -> None:
    """--protocol overrides the files' own labels; a clean run is no standard run."""
    result = runner.invoke(app, ["verify", str(eval_dir), "-p", "balance-v1/standard"])
    assert result.exit_code == 1
    assert "FAIL: 20 of 20 episodes failed verification" in result.output


def _edited_file(out: Path, name: str, **changes: object) -> Path:
    """``traj_000.npz`` of a run with meta keys replaced (or removed when the value is None)."""
    from dataclasses import replace

    from topple.traj import load_npz

    traj = load_npz(out / "traj_000.npz", max_steps=1000)
    meta = {k: v for k, v in {**traj.meta, **changes}.items() if v is not None}
    replace(traj, meta=meta).save_npz(out / name)
    return out / name


def test_a_file_that_names_no_episode_exits_2(tmp_path: Path) -> None:
    out = _run(tmp_path, "-p", "balance-v1/clean")
    path = _edited_file(out, "anon.npz", episode_index=None, scenario=None)
    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == 2
    assert "cannot tell which episode this is; pass --index or --seed" in result.output
    assert "Traceback" not in result.output


def test_long_untrusted_text_in_an_error_is_cut(tmp_path: Path) -> None:
    out = _run(tmp_path, "-p", "balance-v1/clean")
    path = _edited_file(out, "long.npz", challenge="x" * 10_000)
    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == 2
    line = result.output.strip()
    assert line.startswith("error: ")
    assert line.endswith("...")
    assert len(line) == len("error: ") + 500


def test_a_fifo_in_a_run_directory_exits_2_without_hanging(
    tmp_path: Path, make_fifo: Callable[[Path], Path]
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    make_fifo(run / "traj_000.npz")
    result = runner.invoke(app, ["verify", str(run)])
    assert result.exit_code == 2
    assert "traj_000.npz: not a regular file" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["verify", "score"])
def test_a_fifo_given_as_the_path_exits_2(
    tmp_path: Path, make_fifo: Callable[[Path], Path], command: str
) -> None:
    path = make_fifo(tmp_path / "traj_000.npz")
    result = runner.invoke(app, [command, str(path)])
    assert result.exit_code == 2
    assert "not a regular file" in result.output
