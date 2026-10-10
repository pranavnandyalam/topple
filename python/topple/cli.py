"""`topple` command-line interface."""

from __future__ import annotations

import json
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from topple import __version__

if TYPE_CHECKING:
    from topple.evaluate import EvalProtocol
    from topple.loader import Submission
    from topple.results import EpisodeResult, EvalResults

app = typer.Typer(help="Topple: CI for robots.", no_args_is_help=True)


class Videos(StrEnum):
    none = "none"
    first = "first"
    worst = "worst"
    all = "all"


class VideoFormat(StrEnum):
    mp4 = "mp4"
    gif = "gif"


ProtocolOpt = Annotated[
    str, typer.Option("--protocol", "-p", help="Protocol id, e.g. balance-v1/standard or eval_v1.")
]
OutOpt = Annotated[Path, typer.Option("--out", "-o", help="Output directory.")]
FormatOpt = Annotated[VideoFormat, typer.Option("--video-format", help="Video container.")]


@app.command()
def version() -> None:
    """Print the installed Topple version."""
    typer.echo(f"topple {__version__}")


@app.command()
def smoke(
    out: Annotated[Path, typer.Option(help="Where to write the rendered PNG.")] = Path("smoke.png"),
    steps: Annotated[int, typer.Option(min=1, help="Physics steps before rendering.")] = 1000,
) -> None:
    """Check that MuJoCo simulates and renders headlessly on this machine."""
    from topple.rendering import RenderingUnavailableError
    from topple.smoke import run_smoke

    try:
        result = run_smoke(out, steps=steps)
    except RenderingUnavailableError as exc:
        typer.echo(f"smoke failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"smoke ok: {result.steps} steps -> {result.image_path}")


def _load(path: Path) -> Submission:
    from topple.loader import LoadError, load_submission

    try:
        return load_submission(path)
    except LoadError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _protocol(protocol_id: str) -> EvalProtocol:
    from topple.evaluate import EvalProtocol

    try:
        return EvalProtocol.from_id(protocol_id)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _episode_line(e: EpisodeResult) -> str:
    m = e.metrics or {}
    status = (
        f"error:{e.error.code}"
        if e.error
        else ("success" if e.success else str(m.get("termination")))
    )
    upright = float(m.get("time_upright_s", 0))
    drift = float(m.get("drift_m", 0))
    energy = float(m.get("energy_j", 0))
    return (
        f"  #{e.index:<3d} seed {e.seed!s:>6}  score {e.score:6.2f}  {status:<14} "
        f"upright {upright:5.2f}s  drift {drift:.3f} m  energy {energy:6.2f} J"
    )


def _print_results(results: EvalResults) -> None:
    for e in results.episodes or []:
        typer.echo(_episode_line(e))
    a = results.aggregate
    typer.echo(
        f"{results.protocol}: score {a.score:.2f}/100  success {a.success_rate * 100:.0f}%  "
        f"mean upright {a.mean_time_upright_s:.2f}s  errors {a.errors}/{a.episodes}"
    )


def _render(traj_path: Path, out: Path, fmt: VideoFormat) -> None:
    from topple.rendering import RenderingUnavailableError
    from topple.traj import load_npz
    from topple.video import render_trajectory

    try:
        render_trajectory(load_npz(traj_path), out.with_suffix(f".{fmt.value}"))
    except RenderingUnavailableError as exc:
        typer.echo(f"warning: video skipped: {exc}", err=True)


def _evaluate(sub: Submission, protocol: EvalProtocol, out: Path) -> EvalResults:
    from topple.evaluate import evaluate
    from topple.results import ControllerInfo, write_results

    results = evaluate(
        protocol, sub.factory, ControllerInfo(id=sub.id, sha256=sub.sha256), traj_dir=out
    )
    write_results(results, out / "results.json")
    return results


@app.command()
def run(
    controller: Annotated[Path, typer.Argument(help="Submission .py file or directory.")],
    protocol: ProtocolOpt = "balance-v1/standard",
    seed: Annotated[
        int | None, typer.Option(help="Episode seed (default: first public seed).")
    ] = None,
    out: OutOpt = Path("out/run"),
    video: Annotated[bool, typer.Option("--video/--no-video", help="Render the episode.")] = True,
    video_format: FormatOpt = VideoFormat.mp4,
) -> None:
    """Run one episode: writes results.json, traj_000.npz and a video."""
    sub = _load(controller)
    proto = _protocol(protocol)
    proto = replace(proto, seeds=(proto.seeds[0] if seed is None else seed,))
    results = _evaluate(sub, proto, out)
    _print_results(results)
    if video:
        _render(out / "traj_000.npz", out / "episode", video_format)


@app.command("eval")
def eval_cmd(
    controller: Annotated[Path, typer.Argument(help="Submission .py file or directory.")],
    protocol: ProtocolOpt = "balance-v1/standard",
    out: OutOpt = Path("out/eval"),
    videos: Annotated[Videos, typer.Option(help="Which episodes to render.")] = Videos.worst,
    video_format: FormatOpt = VideoFormat.mp4,
) -> None:
    """Evaluate on every seed of a protocol: results.json, trajectories, videos."""
    sub = _load(controller)
    results = _evaluate(sub, _protocol(protocol), out)
    _print_results(results)
    episodes = results.episodes or []
    chosen: list[int] = []
    if videos is Videos.all:
        chosen = [e.index for e in episodes]
    elif videos is Videos.first and episodes:
        chosen = [episodes[0].index]
    elif videos is Videos.worst and episodes:
        chosen = [min(episodes, key=lambda e: (e.score, e.index)).index]
    for index in chosen:
        _render(out / f"traj_{index:03d}.npz", out / f"episode_{index:03d}", video_format)


@app.command()
def verify(
    trajectory: Annotated[Path, typer.Argument(help="traj_*.npz to re-simulate.")],
) -> None:
    """Re-simulate a trajectory open loop and check it reproduces exactly."""
    from topple.traj import load_npz
    from topple.verify import verify_episode

    result = verify_episode(load_npz(trajectory))
    typer.echo(f"{'OK' if result.ok else 'FAIL'}: {result.reason}")
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def score(
    trajectory: Annotated[Path, typer.Argument(help="traj_*.npz to score.")],
) -> None:
    """Recompute an episode's metrics and score from its trajectory."""
    from topple.challenge import load_challenge
    from topple.scoring import episode_metrics, episode_score, episode_success
    from topple.traj import load_npz

    traj = load_npz(trajectory)
    challenge = load_challenge(traj.meta["challenge"])
    metrics = episode_metrics(traj, challenge.params.motor)
    cfg = challenge.config.score
    payload = {
        "score": episode_score(metrics, cfg),
        "success": episode_success(metrics, cfg),
        "metrics": metrics.as_dict(),
    }
    typer.echo(json.dumps(payload, indent=2))
