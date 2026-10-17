"""Side-by-side replays and the GIF ladder (docs/design/phase-2.md §5.2 item 1 and the GIF
ladders, §5.3, §6 ``test_video_sbs``).

Panes play stored trajectories on one clock. A pane whose episode fell or errored plays its
epilogue (contract §11), then holds its last frame stamped ``FELL at 4.31 s`` (or ``ERROR
at``); the video ends one hold (``STAMP_HOLD_S``) after the last stamped pane ends, or when the
longest pane ends, whichever is later. Labels sit on each pane, trimmed to fit it. Sizes are
multiples of 16 within the scene's offscreen buffer and every fps divides the 100 Hz control
rate. A GIF ladder tries its rungs in order, keeps the first that fits the cap and fails only
if the last one does not.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import imageio.v2 as iio2
import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner

import topple.traj as traj_module
import topple.video as video
from topple.cli import app
from topple.contract import TaskSpec
from topple.envs import BalanceEnv
from topple.evaluate import run_episode
from topple.loader import load_submission
from topple.traj import Trajectory
from topple.video import (
    GALLERY_LADDER,
    HERO_LADDER,
    STAMP_HOLD_S,
    WAYPOINT_LADDER,
    GifRung,
    GifTooLargeError,
    fit_text,
    gif_ladder,
    render_gif_ladder,
    render_side_by_side,
    side_by_side_frames,
    side_by_side_timeline,
)

REPO = Path(__file__).resolve().parents[1]
PID = REPO / "submissions" / "pid_example.py"
runner = CliRunner()
Frame = npt.NDArray[np.uint8]


class Zero:
    def reset(self, rng_seed: int) -> None:
        pass

    def act(self, obs: object) -> tuple[float, float]:
        return (0.0, 0.0)


class Raises(Zero):
    """Raises at step 30."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, obs: object) -> tuple[float, float]:
        self.calls += 1
        if self.calls > 30:
            raise RuntimeError("boom")
        return (0.0, 0.0)


def episode(factory: Callable[[TaskSpec], Any], seed: int) -> Trajectory:
    env = BalanceEnv("ideal", record=True)
    try:
        return run_episode(env, factory, seed, 0).trajectory
    finally:
        env.close()


@pytest.fixture(scope="module")
def survivor() -> Trajectory:
    traj = episode(load_submission(PID).factory, 1000)
    assert traj.meta["termination"]["reason"] == "timeout"
    return traj


@pytest.fixture(scope="module")
def fallen() -> Trajectory:
    traj = episode(lambda spec: Zero(), 1)
    assert traj.meta["termination"]["reason"] in ("fell", "floor_contact")
    assert traj.epilogue_qpos.shape[0] == 100
    return traj


@pytest.fixture(scope="module")
def errored() -> Trajectory:
    traj = episode(lambda spec: Raises(), 1000)
    assert traj.meta["error"]["code"] == "exception"
    assert traj.steps == 30
    return traj


def unstable(traj: Trajectory) -> Trajectory:
    """``traj`` relabelled as a ``sim_unstable`` episode: an error, and no epilogue."""
    meta = dict(traj.meta)
    meta["termination"] = {"step": traj.steps, "reason": "sim_unstable"}
    meta["error"] = {"code": "sim_unstable", "step": traj.steps, "detail": "physics"}
    return replace(traj, meta=meta, epilogue_qpos=traj.epilogue_qpos[:0])


# --- timeline ---------------------------------------------------------------------------------


def test_a_survivor_plays_to_the_end_unstamped(survivor: Trajectory) -> None:
    (pane,), total = side_by_side_timeline([survivor], fps=50)
    assert pane.frames == math.ceil(1001 / 2)
    assert pane.stamp is None
    assert total == pane.frames  # trimmed: nothing after the last frame


def test_a_fall_plays_its_epilogue_then_holds_stamped(fallen: Trajectory) -> None:
    (pane,), total = side_by_side_timeline([fallen], fps=50)
    rows = fallen.qpos.shape[0] + 100
    assert pane.frames == math.ceil(rows / 2)
    assert pane.stamp == f"FELL at {fallen.steps * 0.01:.2f} s"
    assert total == pane.frames + round(STAMP_HOLD_S * 50)


def test_an_error_is_stamped_error(errored: Trajectory) -> None:
    (pane,), _ = side_by_side_timeline([errored], fps=25)
    assert pane.stamp == "ERROR at 0.30 s"
    assert pane.frames == math.ceil((31 + 100) / 4)  # the epilogue follows an error too


def test_sim_unstable_is_an_error_without_epilogue(fallen: Trajectory) -> None:
    traj = unstable(fallen)
    (pane,), _ = side_by_side_timeline([traj], fps=50)
    assert pane.stamp == f"ERROR at {traj.steps * 0.01:.2f} s"
    assert pane.frames == math.ceil(traj.qpos.shape[0] / 2)


def test_panes_share_one_clock(survivor: Trajectory, fallen: Trajectory) -> None:
    (a, b), total = side_by_side_timeline([fallen, survivor], fps=50)
    assert total == b.frames == 501  # the fall's hold ends long before the survivor does
    assert a.frames + round(STAMP_HOLD_S * 50) < total


def test_the_last_stamped_pane_gets_its_hold(fallen: Trajectory, errored: Trajectory) -> None:
    (a, b), total = side_by_side_timeline([errored, fallen], fps=10)
    assert total == max(a.frames, b.frames) + round(STAMP_HOLD_S * 10)


def test_a_timeline_needs_a_pane() -> None:
    with pytest.raises(ValueError, match="pane"):
        side_by_side_timeline([], fps=50)


def test_timeline_follows_fps(fallen: Trajectory) -> None:
    rows = fallen.qpos.shape[0] + 100
    for fps in (100, 50, 25, 20, 10, 5, 4, 2, 1):
        (pane,), _ = side_by_side_timeline([fallen], fps=fps)
        assert pane.frames == math.ceil(rows / (100 // fps))


ODD_TERMINATIONS: dict[str, object] = {
    "reason-list": {"step": 0, "reason": ["fell"]},
    "reason-dict": {"step": 0, "reason": {"a": 1}},
    "reason-int": {"step": 0, "reason": 5},
    "reason-missing": {"step": 0},
    "termination-list": ["fell"],
    "termination-string": "fell",
}


def odd_meta(traj: Trajectory, termination: object) -> Trajectory:
    """``traj`` with an untrusted ``termination`` of another JSON shape (contract §14) and no
    error."""
    return replace(traj, meta={**traj.meta, "error": None, "termination": termination})


@pytest.mark.parametrize("termination", list(ODD_TERMINATIONS.values()), ids=list(ODD_TERMINATIONS))
def test_a_reason_that_is_not_a_string_names_no_fall(
    fallen: Trajectory, termination: object
) -> None:
    """A reason of another JSON type is no recorded reason (as ``_shows_epilogue`` reads it):
    the pane plays its rows without the epilogue and is never stamped; nothing raises."""
    (pane,), total = side_by_side_timeline([odd_meta(fallen, termination)], fps=50)
    assert pane.stamp is None
    assert pane.frames == math.ceil(fallen.qpos.shape[0] / 2)
    assert total == pane.frames


# --- labels and frame rule --------------------------------------------------------------------


def test_fit_text_trims_to_the_width() -> None:
    assert fit_text("PPO", 200, 20) == "PPO"
    long = fit_text("a very long label that cannot fit in a small pane", 120, 20)
    assert long.endswith("...")
    assert video.text_width(long, 20) <= 120
    assert fit_text("x" * 50, 1, 20) == ""


def test_fit_text_shows_untrusted_text_as_printable_ascii() -> None:
    assert fit_text("bot\x1b[2J\nnext‮é", 1000, 20) == "bot?[2J?next??"


def blank_frames(play: Any) -> Iterator[Frame]:
    """Stands in for ``video._frames`` (no GL): one black pane per row."""
    for _ in play.rows:
        yield np.zeros((play.height, play.width, 3), np.uint8)


def test_each_label_is_fitted_once_not_per_frame(
    fallen: Trajectory, survivor: Trajectory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fitting a label measures up to one candidate per character; the label and the pane
    never change, so a long label costs that once, not on every frame (quality-F11)."""
    calls = 0
    real = video.text_width

    def counting(text: str, px: int) -> float:
        nonlocal calls
        calls += 1
        return real(text, px)

    monkeypatch.setattr(video, "_frames", blank_frames)
    monkeypatch.setattr(video, "text_width", counting)
    label = "F11" + "x" * (video.MAX_LABEL_CHARS - 3)  # fitted by no other test
    with closing(side_by_side_frames([fallen, survivor], [label, "PPO"], pane=64, fps=10)) as it:
        count = sum(1 for _ in it)
    assert count > 50
    assert calls < 4 * video.MAX_LABEL_CHARS  # per frame, it was about 200 calls each


def test_annotate_draws_inside_the_pane() -> None:
    frame = np.zeros((64, 64, 3), np.uint8)
    labelled = video.annotate(frame, "PID", stamp=None)
    assert labelled.shape == frame.shape
    assert (labelled[:16] > 0).any()  # the label, top left
    assert not (labelled[48:] > 0).any()
    stamped = video.annotate(frame, "PID", stamp="FELL at 4.31 s")
    assert (stamped[40:] > 0).any()  # the stamp, at the bottom
    assert (frame == 0).all()  # never in place


@pytest.mark.parametrize("pane", [63, 0, -16, 736])
def test_pane_must_be_a_multiple_of_16_within_the_framebuffer(
    survivor: Trajectory, tmp_path: Path, pane: int
) -> None:
    with pytest.raises(ValueError, match="size"):
        render_side_by_side([survivor], ["A"], tmp_path / "x.mp4", pane=pane)
    assert not (tmp_path / "x.mp4").exists()


@pytest.mark.parametrize("fps", [0, 3, 30, 200])
def test_fps_must_divide_the_control_rate(survivor: Trajectory, tmp_path: Path, fps: int) -> None:
    with pytest.raises(ValueError, match="fps"):
        render_side_by_side([survivor], ["A"], tmp_path / "x.mp4", pane=64, fps=fps)


def test_gif_fps_is_capped(survivor: Trajectory, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GIF"):
        render_side_by_side([survivor], ["A"], tmp_path / "x.gif", pane=64, fps=100)


@pytest.mark.parametrize(
    ("trajs", "labels", "match"),
    [(0, [], "pane"), (5, ["a"] * 5, "pane"), (2, ["a"], "label")],
)
def test_pane_and_label_counts(
    survivor: Trajectory, tmp_path: Path, trajs: int, labels: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        render_side_by_side([survivor] * trajs, labels, tmp_path / "x.mp4", pane=64)


def test_unsupported_suffix(survivor: Trajectory, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="suffix"):
        render_side_by_side([survivor], ["A"], tmp_path / "x.avi", pane=64)


# --- rendering (GL) ---------------------------------------------------------------------------


def frames_of(trajs: list[Trajectory], labels: list[str], **kwargs: Any) -> list[Frame]:
    with closing(side_by_side_frames(trajs, labels, **kwargs)) as frames:
        return list(frames)


@pytest.mark.render
def test_render_64px_smoke(errored: Trajectory, fallen: Trajectory, tmp_path: Path) -> None:
    """Panes side by side on one clock: each plays its epilogue, then holds its last frame,
    stamped; the video ends one hold after the later pane. (Software GL renders about 10
    frames a second here, so the smoke uses two short episodes.)"""
    frames = frames_of([errored, fallen], ["PID", "PPO"], pane=64, fps=10)
    (a, b), total = side_by_side_timeline([errored, fallen], fps=10)
    assert (a.frames, b.frames, total) == (14, 16, 26)
    assert len(frames) == total
    assert all(f.shape == (64, 128, 3) and f.dtype == np.uint8 for f in frames)
    for pane, timeline in ((slice(0, 64), a), (slice(64, 128), b)):
        shown = [f[:, pane] for f in frames]
        played = timeline.frames - 1
        assert (shown[played + 1] != shown[played]).any()  # the stamp appears
        assert (shown[played + 1][40:] != shown[played][40:]).any()  # at the bottom
        assert (shown[played + 1][:24] == shown[played][:24]).all()  # the label stays
        assert all((shown[k] == shown[played + 1]).all() for k in range(played + 1, total))
        assert any((shown[k] != shown[0]).any() for k in range(1, played))  # it moved
    out = tmp_path / "sbs.mp4"
    written = render_side_by_side([errored, fallen], ["PID", "PPO"], out, pane=64, fps=10)
    assert written == total
    assert mp4_size_and_frames(out) == ((128, 64), total)
    assert oct(out.stat().st_mode & 0o777) == "0o644"
    assert [p.name for p in tmp_path.iterdir()] == ["sbs.mp4"]  # no temporary left


def mp4_size_and_frames(path: Path) -> tuple[tuple[int, int], int]:
    with iio2.get_reader(path, format="FFMPEG") as reader:
        size = tuple(reader.get_meta_data()["size"])
    return (size[0], size[1]), imageio_ffmpeg.count_frames_and_secs(str(path))[0]


@pytest.mark.render
def test_labels_differ_only_in_the_label_band(fallen: Trajectory) -> None:
    ab = frames_of([fallen, fallen], ["A", "B"], pane=64, fps=1)
    ba = frames_of([fallen, fallen], ["B", "A"], pane=64, fps=1)
    differ = np.any(ab[0] != ba[0], axis=2)
    rows = np.flatnonzero(differ.any(axis=1))
    assert rows.size > 0
    assert rows.max() < 24  # only the label band at the top
    assert (ab[0][:, :64] == ba[0][:, 64:]).all()  # same pane, same label: same pixels


# --- GIF ladder -------------------------------------------------------------------------------


def test_the_ladders_of_section_5_2() -> None:
    assert (
        GifRung(480, 20),
        GifRung(480, 10),
        GifRung(480, 5),
        GifRung(432, 5),
        GifRung(389, 5),
        GifRung(350, 5),
    ) == HERO_LADDER
    assert [(r.width, r.fps) for r in GALLERY_LADDER] == [
        (320, 10), (320, 5), (320, 4), (288, 4), (259, 4), (233, 4)
    ]  # fmt: skip
    assert [(r.width, r.fps) for r in WAYPOINT_LADDER] == [
        (480, 10), (480, 5), (480, 4), (432, 4), (389, 4), (350, 4)
    ]  # fmt: skip
    assert gif_ladder(480, (20, 10, 5), 5) == HERO_LADDER
    assert all(100 % r.fps == 0 for r in HERO_LADDER + GALLERY_LADDER + WAYPOINT_LADDER)


@pytest.fixture
def fake_frames(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """``side_by_side_frames`` replaced by noise frames (big GIFs at high fps and size), and a
    record of the (pane, fps) each rung asked for."""
    asked: list[tuple[int, int]] = []

    def frames(trajs: Any, labels: Any, *, pane: int, fps: int) -> Iterator[Frame]:
        asked.append((pane, fps))
        rng = np.random.default_rng(0)
        for _ in range(fps // 5):  # a fifth of a second of noise per rung: sizes still order
            yield rng.integers(0, 256, size=(pane, pane * len(trajs), 3), dtype=np.uint8)

    monkeypatch.setattr(video, "side_by_side_frames", frames)
    return asked


def test_the_first_rung_that_fits_is_used(
    survivor: Trajectory, tmp_path: Path, fake_frames: list[tuple[int, int]]
) -> None:
    out = tmp_path / "hero.gif"
    rung = render_gif_ladder([survivor, survivor], ["A", "B"], out, HERO_LADDER, max_bytes=10**9)
    assert rung == HERO_LADDER[0]
    assert fake_frames == [(240, 20)]
    assert iio.imread(out).shape[1:3] == (240, 480)


def test_the_ladder_steps_down_in_order(
    survivor: Trajectory, tmp_path: Path, fake_frames: list[tuple[int, int]]
) -> None:
    sizes = {}
    for rung in HERO_LADDER:  # measure each rung alone
        out = tmp_path / f"{rung.width}_{rung.fps}.gif"
        render_gif_ladder([survivor] * 2, ["A", "B"], out, (rung,), max_bytes=10**9)
        sizes[rung] = out.stat().st_size
    assert sizes[HERO_LADDER[0]] > sizes[HERO_LADDER[1]] > sizes[HERO_LADDER[2]]
    assert sizes[HERO_LADDER[2]] > sizes[HERO_LADDER[3]] > sizes[HERO_LADDER[5]]
    fake_frames.clear()
    out = tmp_path / "hero.gif"
    cap = sizes[HERO_LADDER[4]]
    rung = render_gif_ladder([survivor] * 2, ["A", "B"], out, HERO_LADDER, max_bytes=cap)
    assert rung == HERO_LADDER[4]
    assert out.stat().st_size <= cap
    # 389 px is no multiple of 16: rendered at 2 x 208 and resampled to 389 x 194
    assert fake_frames[-1] == (208, 5)
    assert iio.imread(out).shape[1:3] == (194, 389)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_only_the_last_rung_over_the_cap_fails(
    survivor: Trajectory, tmp_path: Path, fake_frames: list[tuple[int, int]]
) -> None:
    out = tmp_path / "hero.gif"
    out.write_bytes(b"old")
    with pytest.raises(GifTooLargeError, match="350 px at 5 fps"):
        render_gif_ladder([survivor] * 2, ["A", "B"], out, HERO_LADDER, max_bytes=1000)
    assert len(fake_frames) == len(HERO_LADDER)  # every rung was tried, in order
    assert [f for _, f in fake_frames] == [r.fps for r in HERO_LADDER]
    assert out.read_bytes() == b"old"  # left as it was
    assert sorted(p.name for p in tmp_path.iterdir()) == ["hero.gif"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rungs": ()}, "rung"),
        ({"rungs": (GifRung(480, 30),)}, "fps"),
        ({"rungs": (GifRung(480, 100),)}, "GIF"),
        ({"rungs": (GifRung(8, 5),)}, "width"),
        ({"rungs": (GifRung(2000, 5),)}, "size"),
        ({"max_bytes": 0}, "max_bytes"),
        ({"out": "x.mp4"}, "gif"),
    ],
)
def test_ladder_arguments(
    survivor: Trajectory, tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    args: dict[str, Any] = {"rungs": HERO_LADDER, "max_bytes": 10**6, "out": "x.gif"}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        render_gif_ladder(
            [survivor] * 2,
            ["A", "B"],
            tmp_path / args["out"],
            args["rungs"],
            max_bytes=args["max_bytes"],
        )


# --- CLI --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def npzs(
    tmp_path_factory: pytest.TempPathFactory, errored: Trajectory, fallen: Trajectory
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("npz")
    a, b = root / "pid.npz", root / "ppo.npz"
    errored.save_npz(a)
    fallen.save_npz(b)
    return a, b


def invoke(a: Path, b: Path, out: Path | str, *extra: str) -> Any:
    args = ["video", "--side-by-side", str(a), str(b), "--labels", "PID", "PPO", "--out", str(out)]
    return runner.invoke(app, [*args, *extra])


@pytest.mark.render
def test_cli_video_side_by_side(npzs: tuple[Path, Path], tmp_path: Path) -> None:
    out = tmp_path / "videos" / "hero.mp4"
    result = invoke(*npzs, out, "--fps", "10", "--size", "64")
    assert result.exit_code == 0, result.output
    assert mp4_size_and_frames(out) == ((128, 64), 26)
    assert "hero.mp4" in result.output


def test_cli_video_loads_with_the_challenge_bound(
    npzs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, int]] = []
    real = traj_module.load_npz

    def recording(path: Path, *, max_steps: int, **kwargs: Any) -> Trajectory:
        seen.append((Path(path).name, max_steps))
        return real(path, max_steps=max_steps, **kwargs)

    monkeypatch.setattr(traj_module, "load_npz", recording)
    monkeypatch.setattr(video, "render_side_by_side", lambda *a, **k: 0)
    result = invoke(*npzs, tmp_path / "x.mp4")
    assert result.exit_code == 0, result.output
    assert seen == [("pid.npz", 1000), ("ppo.npz", 1000)]


def test_cli_video_refuses_a_symlinked_output_before_loading(
    npzs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", tmp_path / "out")
    result = invoke(*npzs, "out/hero.mp4")
    assert result.exit_code == 2
    assert "refusing to write through the symlink 'out'" in result.output


@pytest.mark.parametrize(
    "extra", [["--size", "63"], ["--fps", "30"], ["--size", "736"]], ids=["size", "fps", "big"]
)
def test_cli_video_bad_options_exit_2(
    npzs: tuple[Path, Path], tmp_path: Path, extra: list[str]
) -> None:
    result = invoke(*npzs, tmp_path / "x.mp4", *extra)
    assert result.exit_code == 2
    assert not (tmp_path / "x.mp4").exists()


def test_cli_video_bad_inputs_exit_2(npzs: tuple[Path, Path], tmp_path: Path) -> None:
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a zip")
    result = invoke(bad, npzs[1], tmp_path / "x.mp4")
    assert result.exit_code == 2
    assert result.output.count("\n") == 1
    result = invoke(tmp_path / "absent.npz", npzs[1], tmp_path / "x.mp4")
    assert result.exit_code == 2
    result = invoke(*npzs, tmp_path / "x.avi")
    assert result.exit_code == 2


def test_cli_video_needs_two_files_and_two_labels(npzs: tuple[Path, Path], tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["video", "--side-by-side", str(npzs[0]), "--labels", "A", "--out", "x.mp4"]
    )
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "challenge", [5, "balance-v9", "Balance"], ids=["number", "unknown", "bad"]
)
def test_cli_video_refuses_a_file_naming_no_known_challenge(
    npzs: tuple[Path, Path], tmp_path: Path, challenge: object
) -> None:
    traj = traj_module.load_npz(npzs[0], max_steps=1000)
    odd = tmp_path / "odd.npz"
    replace(traj, meta={**traj.meta, "challenge": challenge}).save_npz(odd)
    result = invoke(odd, npzs[1], tmp_path / "x.mp4")
    assert result.exit_code == 2
    assert "odd.npz" in result.output


@pytest.mark.parametrize("termination", list(ODD_TERMINATIONS.values()), ids=list(ODD_TERMINATIONS))
def test_cli_video_survives_odd_termination_meta(
    npzs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination: object,
) -> None:
    """A crafted file whose ``termination`` (or its reason) is another JSON type (quality-F2:
    a list or dict reason raised ``TypeError: unhashable``, exit 1 with a traceback) renders
    unstamped (no GL here: the panes are blank)."""
    traj = traj_module.load_npz(npzs[1], max_steps=1000)
    odd = tmp_path / "odd.npz"
    odd_meta(traj, termination).save_npz(odd)
    monkeypatch.setattr(video, "_frames", blank_frames)
    out = tmp_path / "x.mp4"
    result = invoke(odd, npzs[0], out, "--fps", "10", "--size", "64")
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert out.is_file()


def test_cli_video_a_file_that_does_not_load_exits_2(
    npzs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: Path, *, max_steps: int, **kwargs: Any) -> Trajectory:
        raise ValueError(f"{Path(path).name}: T exceeds max_steps")

    monkeypatch.setattr(traj_module, "load_npz", refuse)
    result = invoke(*npzs, tmp_path / "x.mp4")
    assert result.exit_code == 2
    assert "pid.npz: T exceeds max_steps" in result.output


def test_cli_video_a_failed_write_exits_2(
    npzs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(*args: object, **kwargs: object) -> int:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(video, "render_side_by_side", full)
    result = invoke(*npzs, tmp_path / "x.mp4")
    assert result.exit_code == 2
    assert "No space left on device" in result.output
