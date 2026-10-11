import json
import math
import re
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import mujoco
import pytest
from typer.testing import CliRunner

import topple.cli as cli_module
import topple.video
from topple import __version__
from topple.cli import app
from topple.traj import load_npz

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]
# Typer's rich help forces styled output under CI (GITHUB_ACTIONS, FORCE_COLOR): drop the escapes
# before matching text that rich renders, such as usage errors.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """``text`` without ANSI style escapes."""
    return _ANSI.sub("", text)


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


ZERO_SUBMISSION = """
class Zero:
    def reset(self, rng_seed):
        pass

    def act(self, obs):
        return (0.0, 0.0)


def make_controller(spec):
    return Zero()
"""


def _eval_zero(tmp_path: Path, *args: str) -> tuple[Any, Path, list[dict[str, Any]]]:
    """`topple eval` of a zero-torque bot on balance-v1/clean: every episode falls within a
    second, so the whole protocol is cheap."""
    out = tmp_path / "eval"
    bot = _submission(tmp_path, ZERO_SUBMISSION)
    result = runner.invoke(
        app, ["eval", str(bot), "--protocol", "balance-v1/clean", "--out", str(out), *args]
    )
    assert result.exit_code == 0, result.output
    episodes: list[dict[str, Any]] = json.loads((out / "results.json").read_text())["episodes"]
    return result, out, episodes


def _worst(episodes: list[dict[str, Any]]) -> int:
    worst: int = min(episodes, key=lambda e: (e["score"], e["index"]))["index"]
    return worst


@pytest.mark.parametrize("videos", ["worst", "first", "all", "none", None])
def test_eval_renders_the_episodes_it_is_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, videos: str | None
) -> None:
    """`--videos worst` (the default) renders the lowest-scoring episode (ties: the lowest
    index), `first` episode 0, `all` every episode in order and `none` nothing, each from its
    own traj_<index>.npz to episode_<index>."""
    rendered: list[tuple[str, str, str]] = []

    def spy(traj_path: Path, out: Path, fmt: Any, protocol: Any) -> None:
        rendered.append((traj_path.name, out.name, fmt.value))

    monkeypatch.setattr(cli_module, "_render", spy)
    _, _, episodes = _eval_zero(tmp_path, *([] if videos is None else ["--videos", videos]))
    worst = _worst(episodes)
    best = max(episodes, key=lambda e: e["score"])["index"]
    assert worst not in (0, best)  # worst, first and best are three different episodes here
    chosen = {
        "worst": [worst],
        None: [worst],
        "first": [0],
        "all": [e["index"] for e in episodes],
        "none": [],
    }[videos]
    assert rendered == [(f"traj_{i:03d}.npz", f"episode_{i:03d}", "mp4") for i in chosen]


@pytest.mark.render
def test_eval_renders_the_worst_episode_as_a_gif(tmp_path: Path) -> None:
    """The PR-comment path (Phase 3): one GIF, the worst episode's, showing every recorded row
    at 50 fps plus the second after the fall."""
    _, out, episodes = _eval_zero(tmp_path, "--video-format", "gif")
    worst = _worst(episodes)
    path = out / f"episode_{worst:03d}.gif"
    assert sorted(p.name for p in out.glob("episode_*")) == [path.name]
    traj = load_npz(out / f"traj_{worst:03d}.npz", max_steps=1000)
    assert traj.epilogue_qpos.shape[0] == 100
    frames = math.ceil((traj.qpos.shape[0] + traj.epilogue_qpos.shape[0]) / 2)
    stored = iio.improps(path, index=None).n_images
    durations = [iio.immeta(path, index=i)["duration"] for i in range(stored)]
    assert sum(durations) == frames * 20  # Pillow merges identical frames: count frame time


def test_a_video_that_cannot_be_made_is_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The results are already written when a video fails: a ValueError, OSError or missing GL
    backend becomes one escaped warning line and the run still exits 0."""

    def broken(traj: Any, path: Path, **kwargs: Any) -> int:
        raise ValueError("no \x1b[31mframes")

    monkeypatch.setattr(topple.video, "render_trajectory", broken)
    result, out, episodes = _eval_zero(tmp_path)
    assert "warning: video skipped: no \\x1b[31mframes" in result.output
    assert len(episodes) == 20
    assert not list(out.glob("episode_*"))


def test_an_unexpected_render_failure_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the expected failures become warnings: a bug in the renderer must fail the run."""

    def broken(traj: Any, path: Path, **kwargs: Any) -> int:
        raise RuntimeError("renderer bug")

    monkeypatch.setattr(topple.video, "render_trajectory", broken)
    bot = _submission(tmp_path, ZERO_SUBMISSION)
    result = runner.invoke(
        app, ["eval", str(bot), "--protocol", "balance-v1/clean", "--out", str(tmp_path / "e")]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)


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
    first_line = "OK: balance-v1/standard episode 0 (seed 1000): reproduced bit-exactly\n"
    assert verified.output.startswith(first_line)
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
    assert result.output.startswith(
        "FAIL (not_reproducible): balance-v1/clean episode 0 (seed 1000): qpos differs"
    )


def test_verify_names_the_protocol_a_file_was_verified_against(tmp_path: Path) -> None:
    """§16: without --protocol a file is verified against its own labels, so a run on the
    push-free clean preset passes. The first line names that protocol: nothing else would show
    that the pass is not a standard one."""
    out = _run(tmp_path, "-p", "balance-v1/clean")
    result = runner.invoke(app, ["verify", str(out / "traj_000.npz")])
    assert result.exit_code == 0, result.output
    first = result.output.splitlines()[0]
    assert first == "OK: balance-v1/clean episode 0 (seed 1000): reproduced bit-exactly"
    standard = runner.invoke(app, ["verify", str(out / "traj_000.npz"), "-p", "eval_v1"])
    assert standard.exit_code == 1, standard.output
    assert standard.output.splitlines()[0] == (
        "FAIL (not_reproducible): balance-v1/standard episode 0 (seed 1000): "
        "meta.preset is 'clean', expected 'standard'"
    )


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
    assert f"importing 'bot.py' failed: {detail}" in result.output
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
    assert "--seed" in plain(result.output)
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


@pytest.mark.parametrize("spelling", [".", "inner/.."])
def test_verify_records_the_resolved_submission_name(
    eval_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """`--submission .` records the directory's own name, as `topple eval .` does (§16)."""
    from topple.loader import ENTRY_FILE, load_submission, submission_sha256

    bot = tmp_path / "mybot"
    (bot / "inner").mkdir(parents=True)
    (bot / ENTRY_FILE).write_text(PID_SUBMISSION)
    out = tmp_path / "trusted.json"
    monkeypatch.chdir(bot)
    result = runner.invoke(
        app, ["verify", str(eval_dir), "--submission", spelling, "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    recorded = json.loads(out.read_text())["controller"]
    assert recorded == {"id": "mybot", "sha256": submission_sha256(bot)}
    assert recorded["id"] == load_submission(Path(spelling)).id


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


def test_verify_a_directory_with_an_incomplete_episode(eval_dir: Path, tmp_path: Path) -> None:
    """A file cut short with no error report is incomplete (§16, forgeries C and I): the run
    did not verify (exit 1), and the trusted results say why."""
    from topple.traj import load_npz
    from verify_support import truncated

    run = _copy_run(eval_dir, tmp_path / "copy")
    traj = load_npz(run / "traj_003.npz", max_steps=1000)
    assert traj.meta["error"] is None
    truncated(traj, 500).save_npz(run / "traj_003.npz")
    out = tmp_path / "trusted.json"
    result = runner.invoke(app, ["verify", str(run), "--out", str(out)])
    assert result.exit_code == 1, result.output
    assert "FAIL: 1 of 20 episodes failed verification" in result.output
    error = json.loads(out.read_text())["episodes"][3]["error"]
    assert (error["code"], error["step"]) == ("incomplete", 500)


@pytest.mark.parametrize("target", ["directory", "file"])
def test_verify_out_that_cannot_be_written_exits_2(
    eval_dir: Path, tmp_path: Path, target: str
) -> None:
    """Exit 1 means verification failed (§16). A run that reproduced but whose --out cannot be
    written is a usage error: exit 2, one error line, no traceback."""
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    path = eval_dir if target == "directory" else eval_dir / "traj_000.npz"
    result = runner.invoke(app, ["verify", str(path), "--out", str(blocker / "results.json")])
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    errors = [line for line in result.output.splitlines() if line.startswith("error:")]
    assert len(errors) == 1
    assert errors[0].startswith(f"error: cannot write {blocker / 'results.json'}: ")
    assert result.output.splitlines()[-1] == errors[0]


def test_verify_a_directory_holds_one_upload_at_a_time(
    eval_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uploads are loaded only as `rescore` reaches them and dropped once verified, so a run
    directory never costs more memory than its largest single file (§16)."""
    import gc
    import weakref

    import topple.traj
    import topple.verify

    loaded: list[weakref.ref[object]] = []
    live_at_verify: list[int] = []
    real_load, real_verify = topple.traj.load_npz, topple.verify._verify

    def load(path: Path, *, max_steps: int) -> object:
        traj = real_load(path, max_steps=max_steps)
        loaded.append(weakref.ref(traj))
        return traj

    def verify(*args: Any, **kwargs: Any) -> Any:
        gc.collect()
        live_at_verify.append(sum(ref() is not None for ref in loaded))
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(topple.traj, "load_npz", load)
    monkeypatch.setattr(topple.verify, "_verify", verify)
    result = runner.invoke(app, ["verify", str(eval_dir)])
    assert result.exit_code == 0, result.output
    assert len(loaded) == 20
    assert live_at_verify == [1] * 20


def _bloated_file(path: Path) -> None:
    """The review's bloat: T = 0, but state members 2,000,000 wide, deflated to a few KB."""
    import io
    import zipfile

    import numpy as np

    from topple.traj import FORMAT

    width = 2_000_000
    arrays: dict[str, Any] = {
        "qpos": np.zeros((1, width)),
        "qvel": np.zeros((1, width)),
        "epilogue_qpos": np.zeros((1, width)),
        "obs": np.zeros((0, 10), np.float32),
        "action": np.zeros((0, 2)),
        "ctrl": np.zeros((0, 2)),
        "enc_counts": np.zeros((0, 2), np.int32),
        "gyro": np.zeros((0, 3), np.float32),
        "accel": np.zeros((0, 3), np.float32),
        "push_force": np.zeros((0, 3)),
        "meta_json": np.frombuffer(json.dumps({"format": FORMAT}).encode(), np.uint8),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arr)
            zf.writestr(f"{name}.npy", buf.getvalue())


def test_verify_a_directory_with_a_bloated_file_exits_2(eval_dir: Path, tmp_path: Path) -> None:
    """A file whose state widths are beyond the format's limits does not load (§14)."""
    run = _copy_run(eval_dir, tmp_path / "copy")
    _bloated_file(run / "traj_005.npz")
    assert (run / "traj_005.npz").stat().st_size < 200_000
    result = runner.invoke(app, ["verify", str(run)])
    assert result.exit_code == 2, result.output
    assert "traj_005.npz: qpos must have at most 256 columns" in result.output
    assert "Traceback" not in result.output


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
    assert result.output.startswith("OK (UNBOUND): balance-v1/clean seed 7: reproduced bit-exactly")
    assert "UNBOUND (seed taken from file)" in result.output
    explicit = runner.invoke(app, ["verify", str(out / "traj_000.npz"), "--seed", "7"])
    assert explicit.exit_code == 0, explicit.output
    assert explicit.output.startswith("OK (UNBOUND): balance-v1/clean seed 7: ")
    assert "UNBOUND (seed from --seed)" in explicit.output
    scored = runner.invoke(app, ["score", str(out / "traj_000.npz"), "--seed", "7"])
    assert scored.exit_code == 0, scored.output
    payload = json.loads(scored.stdout)
    assert (payload["ok"], payload["bound"], payload["seed"]) == (True, False, 7)
    bound = runner.invoke(app, ["verify", str(out / "traj_000.npz"), "--index", "0"])
    assert bound.exit_code == 1
    assert "scenario" in bound.output


@pytest.mark.parametrize("episode_index", [10_000, 123_456, 2**53 - 1])
def test_a_file_index_no_results_file_holds_verifies_unbound(
    tmp_path: Path, episode_index: int
) -> None:
    """§16: an upload chooses its own meta.episode_index. One no results file can hold (an index
    is below 10,000, §15) once made the default binding, --seed and score print "verification
    crashed: ValidationError" and exit 2, the message reserved for verifier bugs. Like any index
    outside the protocol it is verified unbound on its scenario's seed, labelled 0."""
    out = _run(tmp_path, "-p", "balance-v1/clean")
    seed = load_npz(out / "traj_000.npz", max_steps=1000).meta["scenario"]["seed"]
    path = _edited_file(out, "far.npz", episode_index=episode_index)
    trusted = tmp_path / "trusted.json"
    for args in ([], ["--seed", str(seed)]):
        result = runner.invoke(app, ["verify", str(path), *args, "--out", str(trusted)])
        assert result.exit_code == 0, result.output
        line = f"OK (UNBOUND): balance-v1/clean seed {seed}: reproduced bit-exactly"
        assert result.output.startswith(line), result.output
        assert json.loads(trusted.read_text())["episodes"][0]["index"] == 0
    scored = runner.invoke(app, ["score", str(path)])
    assert scored.exit_code == 0, scored.output
    payload = json.loads(scored.stdout)
    assert (payload["ok"], payload["bound"], payload["index"], payload["seed"]) == (
        True,
        False,
        0,
        seed,
    )


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

    monkeypatch.setattr(topple.verify, "verify_run", crash)
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


def _with_member(src: Path, dst: Path, member: str, data: bytes) -> Path:
    """A copy of the trajectory ``src`` whose zip member ``member`` holds ``data`` instead."""
    import zipfile

    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            zout.writestr(info.filename, data if info.filename == member else zin.read(info))
    return dst


# Under numpy's own 10,000-character header limit, but ``ast.literal_eval`` gives up on it with
# ``MemoryError: Parser stack overflowed``.
_MINUS_HEADER = (
    "{'descr': '|u1', 'fortran_order': False, 'shape': (" + "-" * 9000 + "1,), }"
).encode("latin1")
_MINUS_NPY = b"\x93NUMPY\x01\x00" + len(_MINUS_HEADER).to_bytes(2, "little") + _MINUS_HEADER


@pytest.mark.parametrize("member", ["meta_json.npy", "qpos.npy"])
@pytest.mark.parametrize("command", ["verify", "score"])
def test_a_header_too_complex_to_parse_exits_2(tmp_path: Path, command: str, member: str) -> None:
    """Such a header used to escape the loader as MemoryError: a traceback and exit 1, the code
    that means "verification failed" (§16)."""
    out = _run(tmp_path, "-p", "balance-v1/clean")
    path = _with_member(out / "traj_000.npz", out / "minus.npz", member, _MINUS_NPY)
    result = runner.invoke(app, [command, str(path)])
    assert result.exit_code == 2, result.output
    assert f"minus.npz: {member.removesuffix('.npy')}: npy header has" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["verify", "score"])
def test_a_file_whose_meta_reads_but_whose_arrays_do_not_load_exits_2(
    tmp_path: Path, command: str
) -> None:
    """Forgery L through the CLI: the metadata binds the file to an episode, then loading it
    refuses its 1001 steps. A clean exit 2, never a traceback with exit 1."""
    import numpy as np

    from topple.traj import load_npz

    out = _run(tmp_path, "-p", "balance-v1/clean")
    traj = load_npz(out / "traj_000.npz", max_steps=1000)
    assert traj.steps == 1000

    def longer(arr: np.ndarray) -> np.ndarray:
        return np.vstack([arr, arr[-1:]])

    per_step = ("qpos", "qvel", "obs", "action", "ctrl", "enc_counts", "gyro", "accel")
    changes = {name: longer(getattr(traj, name)) for name in (*per_step, "push_force")}
    from dataclasses import replace

    replace(traj, **changes).save_npz(out / "long.npz")
    result = runner.invoke(app, [command, str(out / "long.npz")])
    assert result.exit_code == 2, result.output
    assert "long.npz: qpos: trajectory has 1001 steps, more than 1000" in result.output
    assert "Traceback" not in result.output


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


SURROGATE_NAME = "ctl\udcff.py"  # how Linux decodes the file name b"ctl\xff.py" (not UTF-8)


@pytest.mark.parametrize("target", ["directory", "file"])
def test_verify_with_a_submission_name_that_is_not_utf8_exits_2(
    eval_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """The name cannot be a controller id: a clean usage error before any verdict, never a
    traceback and never exit 1 (which means "verification failed")."""
    import topple.loader

    monkeypatch.setattr(topple.loader, "submission_sha256", lambda path: "a" * 64)
    path = eval_dir if target == "directory" else eval_dir / "traj_000.npz"
    out = tmp_path / "trusted.json"
    result = runner.invoke(
        app,
        ["verify", str(path), "--submission", str(tmp_path / SURROGATE_NAME), "--out", str(out)],
    )
    assert result.exit_code == 2, result.output
    assert "not valid UTF-8" in result.output
    assert "Traceback" not in result.output
    assert "OK" not in result.output
    assert not out.exists()


@pytest.mark.parametrize("command", ["run", "eval"])
def test_a_submission_name_that_is_not_utf8_exits_2_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """A submission directory named by non-UTF-8 bytes hashes fine (names inside it are relative)
    but its name cannot be the controller id."""
    from dataclasses import replace

    import topple.loader
    from topple.loader import Submission

    real = topple.loader.load_submission

    def load(path: Path) -> Submission:
        return replace(real(path), id=SURROGATE_NAME.removesuffix(".py"))

    monkeypatch.setattr(topple.loader, "load_submission", load)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [command, str(_submission(tmp_path)), "-p", "balance-v1/clean", "--out", str(out)]
        + (["--no-video"] if command == "run" else ["--videos", "none"]),
    )
    assert result.exit_code == 2, result.output
    assert "not valid UTF-8" in result.output
    assert "Traceback" not in result.output
    assert not (out / "results.json").exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("command", ["run", "eval"])
def test_an_unreadable_part_of_a_submission_exits_2(
    tmp_path: Path, command: str, kind: str
) -> None:
    """Contract §8: a submission that cannot be read is a load error (exit 2 with one line), as
    it already was for `verify --submission`; never a PermissionError traceback with exit 1."""
    import os

    root = tmp_path / "bot"
    (root / "data").mkdir(parents=True)
    (root / "controller.py").write_text(PID_SUBMISSION)
    locked = root / "asset.bin" if kind == "file" else root / "data"
    if kind == "file":
        locked.write_text("x")
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("file permissions are not enforced for this user")
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [command, str(root), "-p", "balance-v1/clean", "--out", str(out)]
            + (["--no-video"] if command == "run" else ["--videos", "none"]),
        )
    finally:
        locked.chmod(0o755)
    assert result.exit_code == 2, result.output
    assert f"cannot read 'bot/{locked.name}': Permission denied" in result.output
    assert "Traceback" not in result.output
    assert not (out / "results.json").exists()


TOO_LONG = "a" * 260  # over the 255-byte name limit of Linux and macOS file systems
VIDEOS_OFF = {"run": ["--no-video"], "eval": ["--videos", "none"]}


def _locked(path: Path, mode: int, probe: Path) -> None:
    """``path`` set to ``mode``; skips where permissions do not stop this user (root)."""
    import os

    path.chmod(mode)
    if os.access(probe, os.R_OK):
        path.chmod(0o755)
        pytest.skip("file permissions are not enforced for this user")


@pytest.mark.parametrize("command", ["run", "eval"])
def test_a_submission_name_too_long_for_the_file_system_exits_2(
    tmp_path: Path, command: str
) -> None:
    """Contract §8: the submission's own path is probed inside the loader's guard too, so the
    name is a load error (exit 2, one line), never an OSError traceback with exit 1."""
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [command, str(tmp_path / f"{TOO_LONG}.py"), "-p", "balance-v1/clean", "--out", str(out)]
        + VIDEOS_OFF[command],
    )
    assert result.exit_code == 2, result.output
    assert f"error: cannot read '{TOO_LONG}.py': File name too long" in result.output
    assert "Traceback" not in result.output
    assert not out.exists()


@pytest.mark.parametrize("where", ["parent", "submission"])
@pytest.mark.parametrize("command", ["run", "eval"])
def test_a_submission_that_cannot_be_searched_exits_2(
    tmp_path: Path, command: str, where: str
) -> None:
    """``parent``: a ``.py`` in a directory without search permission; ``submission``: a
    submission directory that can be listed but not searched (mode 0444)."""
    if where == "parent":
        locked = tmp_path / "locked"
        locked.mkdir()
        submission = _submission(locked)
        mode, shown = 0, "bot.py"
    else:
        locked = submission = tmp_path / "bot"
        locked.mkdir()
        (locked / "controller.py").write_text(PID_SUBMISSION)
        mode, shown = 0o444, "bot/controller.py"
    probe = submission if where == "parent" else submission / "controller.py"
    _locked(locked, mode, probe)
    out = tmp_path / "out"
    try:
        result = runner.invoke(
            app,
            [command, str(submission), "-p", "balance-v1/clean", "--out", str(out)]
            + VIDEOS_OFF[command],
        )
    finally:
        locked.chmod(0o755)
    assert result.exit_code == 2, result.output
    assert f"error: cannot read '{shown}': Permission denied" in result.output
    assert "Traceback" not in result.output
    assert not out.exists()


@pytest.mark.parametrize("where", ["name too long", "parent not searchable"])
@pytest.mark.parametrize("command", ["verify", "score"])
def test_a_trajectory_path_that_cannot_be_examined_exits_2(
    tmp_path: Path, command: str, where: str
) -> None:
    """Exit 1 means "verification failed" (§16): a path that cannot even be examined is a
    usage error, exit 2 with one line."""
    locked = tmp_path / "locked"
    locked.mkdir()
    if where == "name too long":
        path, reason = locked / f"{TOO_LONG}.npz", "File name too long"
    else:
        path, reason = locked / "traj_000.npz", "Permission denied"
        path.write_bytes(b"")
        _locked(locked, 0, path)
    try:
        result = runner.invoke(app, [command, str(path)])
    finally:
        locked.chmod(0o755)
    assert result.exit_code == 2, result.output
    assert result.output.startswith("error: cannot read ")
    assert result.output.rstrip().endswith(f": {reason}")
    assert "Traceback" not in result.output


def test_verify_a_run_directory_that_cannot_be_listed_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's own check refuses a directory without read permission, but macOS privacy
    protection (TCC) refuses to list folders that ``os.access`` calls readable: the listing's
    ``OSError`` must still be a usage error."""
    import errno

    run = tmp_path / "run"
    run.mkdir()
    real_iterdir = Path.iterdir

    def iterdir(self: Path) -> Iterator[Path]:
        if self == run:
            raise PermissionError(errno.EPERM, "Operation not permitted", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    result = runner.invoke(app, ["verify", str(run)])
    assert result.exit_code == 2, result.output
    assert result.output == f"error: cannot read {run}: Operation not permitted\n"


@pytest.mark.slow
def test_a_private_directory_rescore_fails_on_a_forged_episode(
    eval_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§16: pass/fail comes from the verdicts. A private protocol's results are redacted (no
    episodes), so counting failures in them printed "OK: all 0 episodes reproduced"."""
    from dataclasses import replace

    import topple.cli as cli
    from topple.traj import load_npz

    run = _copy_run(eval_dir, tmp_path / "copy")
    traj = load_npz(run / "traj_002.npz", max_steps=1000)
    replace(traj, meta={**traj.meta, "robot_hash": "0" * 64}).save_npz(run / "traj_002.npz")
    public_protocol = cli._protocol

    def private(protocol_id: str) -> object:
        return replace(public_protocol(protocol_id), seed_set="private", salt=2**127 + 1)

    monkeypatch.setattr(cli, "_protocol", private)
    forged = runner.invoke(app, ["verify", str(run)])
    assert forged.exit_code == 1, forged.output
    assert "FAIL: 1 of 20 episodes failed verification" in forged.output
    honest = runner.invoke(app, ["verify", str(eval_dir)])
    assert honest.exit_code == 0, honest.output
    assert "OK: all 20 episodes reproduced" in honest.output
    assert "seed" not in honest.output  # a private run prints no episode lines


def test_untrusted_text_in_an_error_is_escaped(eval_dir: Path, tmp_path: Path) -> None:
    """Terminal escapes, or a line break then ``::error::`` (a GitHub Actions workflow command),
    in a path or a file name reach the error line escaped: one line, nothing raw."""
    evil = tmp_path / "\x1b]0;pwned\x07\x1b[2J\n::error::evil.py"
    result = runner.invoke(app, ["verify", str(eval_dir), "--submission", str(evil)])
    assert result.exit_code == 2, result.output
    assert result.output.startswith("error: cannot hash")
    assert result.output.count("\n") == 1
    for raw in ("\x1b", "\x07"):
        assert raw not in result.output
    assert "\\x1b]0;pwned\\x07\\x1b[2J\\n::error::evil.py" in result.output


def test_printable_escapes_everything_a_terminal_or_ci_log_would_act_on() -> None:
    from topple.cli import _printable

    text = "ok é \t\n\r\x1b[31m\x07\u202e\u2028\udcff\x00"
    assert _printable(text) == "ok é \\t\\n\\r\\x1b[31m\\x07\\u202e\\u2028\\udcff\\x00"


@pytest.mark.parametrize("command", ["run", "eval"])
def test_run_and_eval_say_the_controller_runs_in_process(command: str) -> None:
    """Contract §8: in-process runs are for trusted code; the help says so where users look."""
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    text = " ".join(plain(result.output).replace("│", " ").split())
    assert "runs the controller's code in this process" in text
    assert "only run code you trust" in text


def test_isolation_is_documented_as_an_unchecked_label() -> None:
    result = runner.invoke(app, ["verify", "--help"])
    text = " ".join(plain(result.output).replace("│", " ").split())
    assert "a label that nothing checks" in text


def test_a_bad_submission_fails_before_any_verdict_is_printed(tmp_path: Path) -> None:
    """File mode resolves --submission first, as directory mode does: exit 2 never follows an
    `OK` line, with or without --out."""
    out = _run(tmp_path, "-p", "balance-v1/clean")
    for extra in ([], ["--out", str(tmp_path / "trusted.json")]):
        result = runner.invoke(
            app,
            [
                "verify",
                str(out / "traj_000.npz"),
                "--submission",
                str(tmp_path / "missing.py"),
                *extra,
            ],
        )
        assert result.exit_code == 2, result.output
        assert "cannot hash" in result.output
        assert "OK" not in result.output
    assert not (tmp_path / "trusted.json").exists()


def test_score_never_prints_a_private_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A private protocol's seeds are secret (§8, §15): `score` reports the seed as null."""
    from dataclasses import replace

    import topple.cli as cli

    out = _run(tmp_path, "-p", "balance-v1/clean")
    public_protocol = cli._protocol

    def private(protocol_id: str) -> object:
        return replace(public_protocol(protocol_id), seed_set="private", salt=2**127 + 99)

    monkeypatch.setattr(cli, "_protocol", private)
    result = runner.invoke(app, ["score", str(out / "traj_000.npz"), "--index", "0"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    # A real, reproduced private score is printed without its seed.
    assert payload["ok"] is True
    assert payload["code"] is None
    assert payload["score"] > 0
    assert payload["bound"] is True
    assert payload["seed"] is None
    verified = runner.invoke(app, ["verify", str(out / "traj_000.npz"), "--index", "0"])
    assert verified.exit_code == 0, verified.output
    assert verified.output.splitlines()[0] == (
        "OK: balance-v1/clean episode 0: reproduced bit-exactly"
    )
    assert "1000" not in verified.output


# ------------------------------------------------------------- planted output directories (§14)


def _checkout_with_planted_out(tmp_path: Path, link: str, victim_files: list[str]) -> Path:
    """A checkout whose ``link`` (``out`` or ``out/run``) is a committed symlink to another
    project holding ``victim_files``."""
    victim = tmp_path / "victim"
    victim.mkdir()
    for name in victim_files:
        (victim / name).parent.mkdir(parents=True, exist_ok=True)
        (victim / name).write_text("precious")
    checkout = tmp_path / "checkout"
    (checkout / link).parent.mkdir(parents=True, exist_ok=True)
    (checkout / link).symlink_to(victim)
    return checkout


def _untouched(tmp_path: Path, victim_files: list[str]) -> None:
    victim = tmp_path / "victim"
    assert sorted(str(p.relative_to(victim)) for p in victim.rglob("*") if p.is_file()) == sorted(
        victim_files
    )
    assert all((victim / name).read_text() == "precious" for name in victim_files)


@pytest.mark.parametrize("link", ["out", "out/{command}"])
@pytest.mark.parametrize("command", ["run", "eval"])
def test_a_default_output_directory_never_writes_through_a_planted_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, link: str
) -> None:
    """A branch can commit ``out`` (or ``out/run``) as a symlink to another project: the default
    ``out/run`` and ``out/eval`` would drop results.json and trajectories there."""
    link = link.format(command=command)
    files = [f"{command}/results.json"] if link == "out" else ["results.json"]
    checkout = _checkout_with_planted_out(tmp_path, link, files)
    sub = _submission(tmp_path)
    monkeypatch.chdir(checkout)
    result = runner.invoke(app, [command, str(sub), *VIDEOS_OFF[command]])
    assert result.exit_code == 2, result.output
    assert result.output.startswith(f"error: refusing to write through the symlink '{link}'")
    assert "Traceback" not in result.output
    _untouched(tmp_path, files)


def test_verify_out_never_writes_through_a_planted_symlink(
    eval_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _checkout_with_planted_out(tmp_path, "trusted", ["results.json"])
    monkeypatch.chdir(checkout)
    result = runner.invoke(app, ["verify", str(eval_dir), "--out", "trusted/results.json"])
    assert result.exit_code == 2, result.output
    assert result.output.startswith("error: refusing to write through the symlink 'trusted'")
    _untouched(tmp_path, ["results.json"])


def test_an_absolute_output_path_is_used_as_given(tmp_path: Path) -> None:
    """An absolute path is the caller's own choice: symlinks on it (macOS's /tmp, a symlinked
    home directory) are followed."""
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    out = tmp_path / "link" / "run"
    result = runner.invoke(
        app, ["run", str(_submission(tmp_path)), "--no-video", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (real / "run" / "results.json").is_file()


def test_smoke_never_writes_through_a_planted_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused before anything renders, so this needs no GL backend."""
    checkout = _checkout_with_planted_out(tmp_path, "shots", ["smoke.png"])
    monkeypatch.chdir(checkout)
    result = runner.invoke(app, ["smoke", "--out", "shots/smoke.png", "--steps", "1"])
    assert result.exit_code == 2, result.output
    assert result.output.startswith("error: refusing to write through the symlink 'shots'")
    _untouched(tmp_path, ["smoke.png"])


def test_the_default_output_directory_works_in_a_clean_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``out`` already there as a real directory (a previous run) is fine."""
    checkout = tmp_path / "checkout"
    (checkout / "out" / "eval").mkdir(parents=True)
    sub = _submission(tmp_path)
    monkeypatch.chdir(checkout)
    result = runner.invoke(app, ["run", str(sub), "--no-video"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (checkout / "out" / "run").iterdir()) == [
        "results.json",
        "traj_000.npz",
    ]


def test_a_relative_output_directory_that_cannot_be_examined_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "file").write_text("not a directory")
    sub = _submission(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", str(sub), "--no-video", "--out", "file/run"])
    assert result.exit_code == 2, result.output
    assert result.output == "error: cannot write to file/run: Not a directory\n"
