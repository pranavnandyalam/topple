"""`topple` command-line interface."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn

import typer

from topple import __version__
from topple.rng import MAX_SEED

if TYPE_CHECKING:
    from topple.evaluate import EvalProtocol
    from topple.loader import Submission
    from topple.results import ControllerInfo, EpisodeResult, EvalResults, Isolation
    from topple.traj import Trajectory
    from topple.verify import EpisodeVerdict, RunVerdict

app = typer.Typer(help="Topple: CI for robots.", no_args_is_help=True)


class Videos(StrEnum):
    none = "none"
    first = "first"
    worst = "worst"
    all = "all"


class VideoFormat(StrEnum):
    mp4 = "mp4"
    gif = "gif"


class IsolationChoice(StrEnum):
    inprocess = "inprocess"
    subprocess = "subprocess"
    container = "container"


MAX_MESSAGE_CHARS = 500  # error messages can quote untrusted file contents
UNKNOWN_CONTROLLER_ID = "unknown"
UNKNOWN_SHA256 = "0" * 64
TRAJ_NAME = re.compile(r"traj_([0-9]{1,6})\.npz")


ProtocolOpt = Annotated[
    str, typer.Option("--protocol", "-p", help="Protocol id, e.g. balance-v1/standard or eval_v1.")
]
OutOpt = Annotated[Path, typer.Option("--out", "-o", help="Output directory.")]
FormatOpt = Annotated[VideoFormat, typer.Option("--video-format", help="Video container.")]
SeedOpt = Annotated[
    int | None,
    typer.Option(min=0, max=MAX_SEED - 1, help="Episode seed, an integer in [0, 2**53)."),
]
IndexOpt = Annotated[
    int | None, typer.Option(min=0, help="Episode index in the protocol (default: from the file).")
]
BindProtocolOpt = Annotated[
    str | None,
    typer.Option("--protocol", "-p", help="Protocol id (default: the file's challenge/preset)."),
]


def _printable(text: str) -> str:
    """``text`` with every character that is not printable shown as its escape (``\\x1b``).

    Error text quotes paths, file names and upload contents. Raw, a control character could
    drive the terminal (retitle it, clear it), a line break followed by ``::error::`` would be a
    workflow command in a GitHub Actions log, and a lone surrogate (a file name that is not valid
    UTF-8) could not even be encoded.
    """
    return "".join(
        ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii") for ch in text
    )


def _fail(message: str) -> NoReturn:
    """Usage, load or internal error: a clean one-line message (escaped) and exit code 2."""
    printable = _printable(message)
    if len(printable) > MAX_MESSAGE_CHARS:
        printable = printable[: MAX_MESSAGE_CHARS - 3] + "..."
    typer.echo(f"error: {printable}", err=True)
    raise typer.Exit(code=2)


def _check_output(directory: Path) -> None:
    """Exit 2, before anything runs, if the output ``directory`` is a relative path through a
    symlink (a checkout can commit ``out`` as one, contract §14) or cannot be examined."""
    from topple._atomic import SymlinkedOutputError, check_output_directory

    try:
        check_output_directory(directory)
    except SymlinkedOutputError as exc:
        _fail(str(exc))
    except OSError as exc:
        _fail(f"cannot write to {directory}: {exc.strerror or type(exc).__name__}")


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

    _check_output(out.parent)
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
        _fail(str(exc))


def _protocol(protocol_id: str) -> EvalProtocol:
    from topple.evaluate import EvalProtocol

    try:
        return EvalProtocol.from_id(protocol_id)
    except (ValueError, OSError) as exc:  # bad grammar or preset; unknown challenge bundle
        _fail(str(exc))


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
        f"  #{e.index:<3d} seed {e.seed!s:>6}  score {e.score:6.2f}  {status:<22} "
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


def _render(traj_path: Path, out: Path, fmt: VideoFormat, protocol: EvalProtocol) -> None:
    from topple.rendering import RenderingUnavailableError
    from topple.traj import load_npz
    from topple.verify import protocol_max_steps
    from topple.video import render_trajectory

    try:
        traj = load_npz(traj_path, max_steps=protocol_max_steps(protocol))
        render_trajectory(traj, out.with_suffix(f".{fmt.value}"))
    except (RenderingUnavailableError, ValueError, OSError) as exc:
        typer.echo(f"warning: video skipped: {_printable(str(exc))}", err=True)


def _evaluate(
    sub: Submission, protocol: EvalProtocol, out: Path, *, complete: bool = True
) -> EvalResults:
    """Run, save the trajectories and ``results.json``; ``complete=False`` marks a partial run."""
    from topple.evaluate import evaluate

    info = _controller(sub.id, sub.sha256)
    try:
        results = evaluate(protocol, sub.factory, info, traj_dir=out)
    except OSError as exc:
        _fail(f"cannot write to {out}: {exc}")
    if not complete:
        results = results.model_copy(update={"complete": False})
    _write(results, out / "results.json")
    return results


def _write(results: EvalResults, path: Path) -> None:
    from topple.results import write_results

    try:
        write_results(results, path)
    except OSError as exc:
        _fail(f"cannot write {path}: {exc}")


@app.command()
def run(
    controller: Annotated[Path, typer.Argument(help="Submission .py file or directory.")],
    protocol: ProtocolOpt = "balance-v1/standard",
    seed: SeedOpt = None,
    out: OutOpt = Path("out/run"),
    video: Annotated[bool, typer.Option("--video/--no-video", help="Render the episode.")] = True,
    video_format: FormatOpt = VideoFormat.mp4,
) -> None:
    """Run one episode (default: the first public seed): writes results.json (complete: false),
    traj_000.npz and a video.

    It runs the controller's code in this process, with your privileges: only run code you trust.
    """
    _check_output(out)
    sub = _load(controller)
    proto = _protocol(protocol)
    proto = replace(proto, seeds=(proto.seeds[0] if seed is None else seed,))
    results = _evaluate(sub, proto, out, complete=False)
    _print_results(results)
    if video:
        _render(out / "traj_000.npz", out / "episode", video_format, proto)


@app.command("eval")
def eval_cmd(
    controller: Annotated[Path, typer.Argument(help="Submission .py file or directory.")],
    protocol: ProtocolOpt = "balance-v1/standard",
    out: OutOpt = Path("out/eval"),
    videos: Annotated[Videos, typer.Option(help="Which episodes to render.")] = Videos.worst,
    video_format: FormatOpt = VideoFormat.mp4,
) -> None:
    """Evaluate on every seed of a protocol: results.json, trajectories, videos.

    It runs the controller's code in this process, with your privileges: only run code you trust.
    """
    _check_output(out)
    sub = _load(controller)
    proto = _protocol(protocol)
    results = _evaluate(sub, proto, out)
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
        _render(out / f"traj_{index:03d}.npz", out / f"episode_{index:03d}", video_format, proto)


# ------------------------------------------------------------------ trusted verification


@dataclass(frozen=True)
class _Binding:
    """Which episode a trajectory file is verified as (contract §16)."""

    protocol: EvalProtocol
    index: int
    seed: int
    bound: bool  # True: episode ``index`` of ``protocol``; False: an explicit or file seed
    note: str  # printed when the verdict is not bound to the protocol


def _read_meta(path: Path) -> dict[str, Any]:
    from topple.traj import read_meta

    try:
        return read_meta(path)
    except ValueError as exc:
        _fail(str(exc))


def _load_upload(path: Path, protocol: EvalProtocol) -> Trajectory:
    from topple.traj import load_npz
    from topple.verify import protocol_max_steps

    try:
        return load_npz(path, max_steps=protocol_max_steps(protocol))
    except ValueError as exc:
        _fail(str(exc))


def _file_protocol(meta: dict[str, Any], protocol_id: str | None) -> EvalProtocol:
    """``--protocol``, else the challenge and preset the file names."""
    if protocol_id is not None:
        return _protocol(protocol_id)
    challenge, preset = meta.get("challenge"), meta.get("preset")
    if not isinstance(challenge, str) or not isinstance(preset, str):
        _fail("the trajectory does not name its challenge and preset; pass --protocol")
    return _protocol(f"{challenge}/{preset}")


def _meta_int(value: object) -> int | None:
    return value if type(value) is int and 0 <= value < MAX_SEED else None


def _bind(
    meta: dict[str, Any], protocol: EvalProtocol, index: int | None, seed: int | None
) -> _Binding:
    """Explicit flags win; otherwise the file's episode index, unless its seed disagrees."""
    n = len(protocol.seeds)
    file_index = _meta_int(meta.get("episode_index"))
    if index is not None and seed is not None:
        _fail("pass either --index or --seed, not both")
    if seed is not None:
        return _Binding(protocol, file_index or 0, seed, False, "UNBOUND (seed from --seed)")
    if index is not None:
        if index >= n:
            _fail(f"--index {index} is outside {protocol.id}'s {n} episodes")
        return _Binding(protocol, index, protocol.seeds[index], True, "")
    scenario = meta.get("scenario")
    file_seed = _meta_int(scenario.get("seed")) if isinstance(scenario, dict) else None
    if (
        file_index is not None
        and file_index < n
        and file_seed in (None, protocol.seeds[file_index])
    ):
        return _Binding(protocol, file_index, protocol.seeds[file_index], True, "")
    if file_seed is None:
        _fail("cannot tell which episode this is; pass --index or --seed")
    return _Binding(protocol, file_index or 0, file_seed, False, "UNBOUND (seed taken from file)")


def _verify_file(
    path: Path, protocol_id: str | None, index: int | None, seed: int | None
) -> tuple[_Binding, EpisodeVerdict]:
    from topple import verify as trusted
    from topple.contract import describe_exception

    meta = _read_meta(path)
    binding = _bind(meta, _file_protocol(meta, protocol_id), index, seed)
    upload = _load_upload(path, binding.protocol)
    try:
        if binding.bound:
            return binding, trusted.verify_episode(upload, binding.protocol, binding.index)
        return binding, trusted.verify_with_seed(
            upload, binding.protocol, binding.seed, index=binding.index
        )
    except Exception as exc:  # a verifier bug must never look like a verification failure
        _fail(f"verification crashed: {describe_exception(exc)}")


def _controller(name: str, sha256: str) -> ControllerInfo:
    """The recorded identity of a submission; exit 2 if its name cannot be a controller id."""
    from topple.results import MAX_ID_CHARS, ControllerInfo

    try:
        name.encode("utf-8")
    except UnicodeEncodeError:  # a file name that is not valid UTF-8 decodes to lone surrogates
        _fail(f"the submission name {name!r} is not valid UTF-8; rename it")
    return ControllerInfo(id=(name or UNKNOWN_CONTROLLER_ID)[:MAX_ID_CHARS], sha256=sha256)


def _controller_info(submission: Path | None) -> ControllerInfo:
    """The submission's hash (computed, never imported) and its id as ``topple eval`` records it,
    or the ``unknown`` placeholder."""
    from topple.loader import LoadError, submission_id, submission_sha256
    from topple.results import ControllerInfo

    if submission is None:
        return ControllerInfo(id=UNKNOWN_CONTROLLER_ID, sha256=UNKNOWN_SHA256)
    try:
        sha = submission_sha256(submission)  # refuses a symlink before anything is resolved
        name = submission_id(submission)
    except (LoadError, OSError, ValueError, RuntimeError) as exc:
        _fail(f"cannot hash {submission}: {exc}")
    return _controller(name, sha)


def _cannot_read(path: Path, exc: OSError) -> NoReturn:
    _fail(f"cannot read {path}: {exc.strerror or type(exc).__name__}")


def _traj_files(directory: Path) -> dict[int, Path]:
    """The ``traj_<index>.npz`` files of a run directory, by index, in index order."""
    try:
        paths = sorted(directory.iterdir())
    except OSError as exc:  # e.g. macOS privacy protection, which ``os.access`` does not see
        _cannot_read(directory, exc)
    files: dict[int, Path] = {}
    for path in paths:
        match = TRAJ_NAME.fullmatch(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in files:
            _fail(f"two files for episode {index}: {files[index].name} and {path.name}")
        files[index] = path
    if not files:
        _fail(f"no traj_<index>.npz files in {directory}")
    return dict(sorted(files.items()))


class _UploadLoadError(Exception):
    """A run directory's file did not load while ``rescore`` was reading it."""


class _UploadFiles(Mapping[int, "Trajectory"]):
    """The uploads of a run directory, by index, each loaded only when it is looked up.

    ``rescore`` looks each index up once, in order, and keeps only the verdict's result, so one
    upload at a time is in memory however many files the directory holds. Nothing is cached.
    """

    def __init__(self, files: dict[int, Path], max_steps: int) -> None:
        self._files = files
        self._max_steps = max_steps

    def __getitem__(self, index: int) -> Trajectory:
        from topple.traj import load_npz

        path = self._files[index]  # KeyError: no file for this index (``missing``)
        try:
            return load_npz(path, max_steps=self._max_steps)
        except ValueError as exc:
            raise _UploadLoadError(str(exc)) from exc

    def __iter__(self) -> Iterator[int]:
        return iter(self._files)

    def __len__(self) -> int:
        return len(self._files)


def _verify_directory(
    directory: Path, protocol_id: str | None, controller: ControllerInfo, isolation: Isolation
) -> RunVerdict:
    """Re-score every trajectory of a run directory against its protocol (``verify_run``),
    loading the files one at a time as they are verified."""
    from topple import verify as trusted
    from topple.contract import describe_exception

    files = _traj_files(directory)
    protocol = _file_protocol(_read_meta(next(iter(files.values()))), protocol_id)
    n = len(protocol.seeds)
    for index, path in files.items():
        if index >= n:
            _fail(f"{path.name}: index outside {protocol.id}'s {n} episodes")
    uploads = _UploadFiles(files, trusted.protocol_max_steps(protocol))
    try:
        return trusted.verify_run(protocol, uploads, controller, isolation)
    except _UploadLoadError as exc:
        _fail(str(exc))
    except Exception as exc:  # a verifier bug must never look like a verification failure
        _fail(f"verification crashed: {describe_exception(exc)}")


def _kind(path: Path) -> Literal["directory", "file"]:
    """Whether ``path`` is a directory or a regular file; exit 2 for anything else (a FIFO or
    device is never opened), including a path that cannot even be examined: Python 3.12's
    ``Path.is_dir``/``is_file`` raise for a name too long or a directory without search
    permission on the way, and exit 1 would claim that verification failed (§16)."""
    try:
        if path.is_dir():
            return "directory"
        if path.is_file():
            return "file"
        reason = f"{path}: not a regular file" if path.exists() else f"{path} does not exist"
    except OSError as exc:
        _cannot_read(path, exc)
    _fail(reason)


def _report_directory(run: RunVerdict) -> bool:
    """Print the trusted results; True if every expected episode reproduced.

    The verdict comes from ``run.failed``, counted before redaction: a private protocol's
    results list no episodes, so they cannot tell a forged run from a reproduced one.
    """
    _print_results(run.results)
    n = run.results.aggregate.episodes
    if not run.ok:
        typer.echo(f"FAIL: {run.failed} of {n} episodes failed verification")
        return False
    typer.echo(f"OK: all {n} episodes reproduced")
    return True


def _report_file(binding: _Binding, verdict: EpisodeVerdict) -> None:
    """The verdict. The first line names the protocol and the episode, and alone tells a
    protocol-bound pass from an unbound one (contract §16): without ``--protocol`` the protocol
    is the file's own label, so its id is what shows that a pass is, say, a ``clean`` one."""
    head = ("OK" if binding.bound else "OK (UNBOUND)") if verdict.ok else f"FAIL ({verdict.code})"
    typer.echo(f"{head}: {_episode_label(binding, verdict)}: {verdict.reason}")
    if not binding.bound:
        typer.echo(
            f"{binding.note}: seed {binding.seed} is checked against {binding.protocol.id}'s "
            "challenge and preset, not as one of its episodes"
        )
    typer.echo(_episode_line(verdict.result))


def _episode_label(binding: _Binding, verdict: EpisodeVerdict) -> str:
    """``<protocol> episode <i> (seed <s>)`` when bound (a private protocol's seed is secret and
    left out), ``<protocol> seed <s>`` when unbound (the seed came from the caller or the file)."""
    if not binding.bound:
        return f"{binding.protocol.id} seed {binding.seed}"
    seed = verdict.result.seed  # None for a private protocol
    shown = f" (seed {seed})" if seed is not None else ""
    return f"{binding.protocol.id} episode {binding.index}{shown}"


def _single_results(
    binding: _Binding, verdict: EpisodeVerdict, controller: ControllerInfo, isolation: Isolation
) -> EvalResults:
    from topple.evaluate import build_results

    protocol = replace(binding.protocol, seeds=(binding.seed,))
    results = build_results(protocol, [verdict.result], controller, isolation)
    return results.model_copy(update={"complete": False})


@app.command()
def verify(
    path: Annotated[Path, typer.Argument(help="A traj_*.npz file, or a run directory of them.")],
    protocol: BindProtocolOpt = None,
    index: IndexOpt = None,
    seed: SeedOpt = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the trusted results.json here.")
    ] = None,
    submission: Annotated[
        Path | None,
        typer.Option(help="Submission to record in --out results (hashed, never imported)."),
    ] = None,
    isolation: Annotated[
        IsolationChoice,
        typer.Option(
            help="How the controller ran, recorded in --out results (a label that nothing checks)."
        ),
    ] = IsolationChoice.inprocess,
) -> None:
    """Re-simulate uploaded trajectories open loop and re-score them from the replay only.

    Exit 0: reproduced. Exit 1: verification failed. Exit 2: usage or load error.
    """
    if out is not None:
        _check_output(out.parent)
    iso: Isolation = isolation.value
    if _kind(path) == "directory":
        if index is not None or seed is not None:
            _fail("--index and --seed apply to a single trajectory file")
        run = _verify_directory(path, protocol, _controller_info(submission), iso)
        ok = _report_directory(run)
        results = run.results
    else:
        controller = _controller_info(submission)  # a bad --submission fails before any verdict
        binding, verdict = _verify_file(path, protocol, index, seed)
        _report_file(binding, verdict)
        ok = verdict.ok
        results = _single_results(binding, verdict, controller, iso)
    if out is not None:
        _write(results, out)
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def score(
    trajectory: Annotated[Path, typer.Argument(help="traj_*.npz to verify and score.")],
    protocol: BindProtocolOpt = None,
    index: IndexOpt = None,
    seed: SeedOpt = None,
) -> None:
    """Verify a trajectory, then print the replay's metrics and score as JSON.

    Exit 1 if verification fails (the score is then 0). An errored episode scores 0.
    """
    if _kind(trajectory) == "directory":
        _fail("topple score takes one trajectory file; use topple verify for a run directory")
    binding, verdict = _verify_file(trajectory, protocol, index, seed)
    result = verdict.result
    payload = {
        "ok": verdict.ok,
        "code": verdict.code,
        "reason": verdict.reason,
        "bound": binding.bound,
        "protocol": binding.protocol.id,
        "index": verdict.index,
        "seed": result.seed,  # null for a private protocol: its seeds are secret
        "score": result.score,
        "success": result.success,
        "error": result.error.model_dump() if result.error else None,
        "metrics": result.metrics,
    }
    typer.echo(json.dumps(payload, indent=2))
    if not verdict.ok:
        raise typer.Exit(code=1)
