"""`topple` command-line interface."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
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
    from topple.traj import Trajectory, TrajLayout
    from topple.verify import EpisodeVerdict, RunVerdict

app = typer.Typer(help="Topple: CI for robots.", no_args_is_help=True)
policy_app = typer.Typer(
    help="topple.policy/1 policies (ONNX networks as data).", no_args_is_help=True
)
app.add_typer(policy_app, name="policy")


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


class Direction(StrEnum):
    both = "both"
    forward = "forward"
    backward = "backward"


FORWARD_SIGNS: Mapping[Direction, tuple[int, ...]] = {
    Direction.both: (1, -1),
    Direction.forward: (1,),
    Direction.backward: (-1,),
}


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
RequiredProtocolOpt = Annotated[
    str, typer.Option("--protocol", "-p", help="Protocol id, e.g. balance-v1/standard or eval_v1.")
]
SubmissionsArg = Annotated[
    list[Path],
    typer.Argument(help="Submissions: .py files, directories or topple.policy/1 directories."),
]
WorkersOpt = Annotated[
    int, typer.Option(min=1, help="Evaluation processes (at most 4 and the CPU count are used).")
]
SEED_RANGE = re.compile(r"([0-9]{1,16})\.\.([0-9]{1,16})")


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


def _write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (``_atomic.replacement``), or exit 2."""
    from topple._atomic import replacement

    try:
        with replacement(path, suffix=".tmp") as tmp:
            tmp.write_text(text, encoding="utf-8")
    except OSError as exc:
        _fail(f"cannot write {path}: {exc.strerror or type(exc).__name__}")


def _seed_range(text: str) -> tuple[int, ...]:
    """``A..B``: the seeds A to B inclusive, each in [0, 2**53), at most 10,000 of them (a
    protocol's limit, contract §8); exit 2 for anything else."""
    from topple.results import MAX_EPISODES

    match = SEED_RANGE.fullmatch(text)
    first, last = (int(match.group(1)), int(match.group(2))) if match else (1, 0)
    if not (match and first <= last < MAX_SEED and last - first < MAX_EPISODES):
        _fail(
            f"--seeds must be A..B with 0 <= A <= B < 2**53 and at most {MAX_EPISODES} seeds, "
            f"got {text!r}"
        )
    return tuple(range(first, last + 1))


def _with_seeds(protocol: EvalProtocol, seeds: str | None) -> EvalProtocol:
    """``protocol`` with ``--seeds`` in place of its own (``protocol_hash`` covers them)."""
    return protocol if seeds is None else replace(protocol, seeds=_seed_range(seeds))


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


def _load(path: Path, protocol: EvalProtocol) -> Submission:
    """The submission at ``path``, validated for ``protocol``'s challenge (a policy's dims and
    bounds come from it); exit 2 with one line if it does not load."""
    from topple.loader import LoadError, load_submission

    try:
        return load_submission(path, challenge=protocol.challenge)
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
    from topple.verify import protocol_max_steps, protocol_traj_layout
    from topple.video import render_trajectory

    try:
        traj = load_npz(
            traj_path,
            max_steps=protocol_max_steps(protocol),
            layout=protocol_traj_layout(protocol),
        )
        render_trajectory(traj, out.with_suffix(f".{fmt.value}"))
    except (RenderingUnavailableError, ValueError, OSError) as exc:
        typer.echo(f"warning: video skipped: {_printable(str(exc))}", err=True)


def _evaluate(
    sub: Submission, protocol: EvalProtocol, out: Path, *, complete: bool = True
) -> EvalResults:
    """Run, save the trajectories and ``results.json``; ``complete=False`` marks a partial run."""
    from topple.evaluate import evaluate

    info = _controller(sub.id, sub.sha256, sub.policy_sha256)
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
    controller: Annotated[
        Path, typer.Argument(help="Submission .py file, directory or topple.policy/1 directory.")
    ],
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
    proto = _protocol(protocol)
    sub = _load(controller, proto)
    proto = replace(proto, seeds=(proto.seeds[0] if seed is None else seed,))
    results = _evaluate(sub, proto, out, complete=False)
    _print_results(results)
    if video:
        _render(out / "traj_000.npz", out / "episode", video_format, proto)


@app.command("eval")
def eval_cmd(
    controller: Annotated[
        Path, typer.Argument(help="Submission .py file, directory or topple.policy/1 directory.")
    ],
    protocol: ProtocolOpt = "balance-v1/standard",
    out: OutOpt = Path("out/eval"),
    videos: Annotated[Videos, typer.Option(help="Which episodes to render.")] = Videos.worst,
    video_format: FormatOpt = VideoFormat.mp4,
) -> None:
    """Evaluate on every seed of a protocol: results.json, trajectories, videos.

    It runs the controller's code in this process, with your privileges: only run code you trust.
    """
    _check_output(out)
    proto = _protocol(protocol)
    sub = _load(controller, proto)
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
    from topple.verify import protocol_max_steps, protocol_traj_layout

    try:
        return load_npz(
            path, max_steps=protocol_max_steps(protocol), layout=protocol_traj_layout(protocol)
        )
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


def _unbound_label(file_index: int | None) -> int:
    """The index an unbound verdict is labelled with: the file's own, if a results file can hold
    it (contract §15), else 0. The upload chooses it, and it labels nothing but the verdict."""
    from topple.results import MAX_EPISODES

    return file_index if file_index is not None and file_index < MAX_EPISODES else 0


def _bind(
    meta: dict[str, Any], protocol: EvalProtocol, index: int | None, seed: int | None
) -> _Binding:
    """Explicit flags win; otherwise the file's episode index, unless its seed disagrees."""
    n = len(protocol.seeds)
    file_index = _meta_int(meta.get("episode_index"))
    if index is not None and seed is not None:
        _fail("pass either --index or --seed, not both")
    if seed is not None:
        label = _unbound_label(file_index)
        return _Binding(protocol, label, seed, False, "UNBOUND (seed from --seed)")
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
    label = _unbound_label(file_index)
    return _Binding(protocol, label, file_seed, False, "UNBOUND (seed taken from file)")


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


def _controller(name: str, sha256: str, policy_sha256: str | None = None) -> ControllerInfo:
    """The recorded identity of a submission (with its ``policy_sha256`` if it is a policy);
    exit 2 if its name cannot be a controller id."""
    from topple.results import MAX_ID_CHARS, ControllerInfo

    try:
        name.encode("utf-8")
    except UnicodeEncodeError:  # a file name that is not valid UTF-8 decodes to lone surrogates
        _fail(f"the submission name {name!r} is not valid UTF-8; rename it")
    return ControllerInfo(
        id=(name or UNKNOWN_CONTROLLER_ID)[:MAX_ID_CHARS],
        sha256=sha256,
        policy_sha256=policy_sha256,
    )


def _controller_info(submission: Path | None) -> ControllerInfo:
    """The submission's hashes (computed, never imported or validated) and its id as ``topple
    eval`` records them, ``policy_sha256`` included for a policy directory; or the ``unknown``
    placeholder."""
    from topple.loader import LoadError, submission_hashes, submission_id
    from topple.results import ControllerInfo

    if submission is None:
        return ControllerInfo(id=UNKNOWN_CONTROLLER_ID, sha256=UNKNOWN_SHA256)
    try:
        sha, policy_sha = submission_hashes(submission)  # refuses a symlink before resolving
        name = submission_id(submission)
    except (LoadError, OSError, ValueError, RuntimeError) as exc:
        _fail(f"cannot hash {submission}: {exc}")
    return _controller(name, sha, policy_sha)


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

    def __init__(self, files: dict[int, Path], max_steps: int, layout: TrajLayout) -> None:
        self._files = files
        self._max_steps = max_steps
        self._layout = layout

    def __getitem__(self, index: int) -> Trajectory:
        from topple.traj import load_npz

        path = self._files[index]  # KeyError: no file for this index (``missing``)
        try:
            return load_npz(path, max_steps=self._max_steps, layout=self._layout)
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
    uploads = _UploadFiles(
        files, trusted.protocol_max_steps(protocol), trusted.protocol_traj_layout(protocol)
    )
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


# ------------------------------------------------------------------ comparisons and benches


@app.command("compare")
def compare_cmd(
    submissions: SubmissionsArg,
    protocol: RequiredProtocolOpt,
    seeds: Annotated[
        str | None,
        typer.Option(help="Seeds A..B (inclusive) in place of the protocol's own."),
    ] = None,
    workers: WorkersOpt = 4,
    json_path: Annotated[
        Path | None, typer.Option("--json", help="Also write the topple.compare/1 file here.")
    ] = None,
) -> None:
    """Run several submissions on the same seeds and compare them pairwise (Markdown report;
    paired bootstrap 95% intervals of score differences).

    It runs the submissions' code on this machine, with your privileges: only run code you
    trust.
    """
    from concurrent.futures.process import BrokenProcessPool

    from topple.compare import compare, to_json, to_markdown
    from topple.loader import LoadError

    if json_path is not None:
        _check_output(json_path.parent)
    proto = _with_seeds(_protocol(protocol), seeds)
    try:
        result = compare(submissions, proto, workers=workers)
    except (LoadError, ValueError) as exc:
        _fail(str(exc))
    except BrokenProcessPool as exc:
        _fail(f"an evaluation worker died: {exc}")
    typer.echo(to_markdown(result), nl=False)
    if json_path is not None:
        _write_text(json_path, to_json(result))


@app.command()
def shove(
    submissions: SubmissionsArg,
    impulses: Annotated[
        str, typer.Option(help="Impulse levels START:STOP:STEP in N·s, STOP included.")
    ] = "0.30:0.65:0.025",
    seeds: Annotated[str, typer.Option(help="Seeds A..B (inclusive) of balance-v1/clean.")] = (
        "0..99"
    ),
    direction: Annotated[
        Direction, typer.Option(help="Push along the heading (forward), against it, or both.")
    ] = Direction.both,
    workers: WorkersOpt = 4,
) -> None:
    """Survival vs push impulse: one push at 2.0 s (0.1 s pulse, no lateral part) on every
    balance-v1/clean seed, at each impulse level, and the impulse where survival first falls
    through 50% (i50).

    It runs the submissions' code on this machine, with your privileges: only run code you
    trust.
    """
    from concurrent.futures.process import BrokenProcessPool

    from topple import bench
    from topple.loader import LoadError

    try:
        levels = bench.parse_impulses(impulses)
    except ValueError as exc:
        _fail(f"--{exc}")
    seed_list = _seed_range(seeds)
    try:
        results = bench.run_shoves(
            submissions, levels, seed_list, FORWARD_SIGNS[direction], workers
        )
    except (LoadError, ValueError) as exc:
        _fail(str(exc))
    except BrokenProcessPool as exc:
        _fail(f"an evaluation worker died: {exc}")
    typer.echo("\n".join(bench.to_markdown(curves) for curves in results), nl=False)


@app.command("video")
def video_cmd(
    side_by_side: Annotated[
        tuple[Path, Path],
        typer.Option("--side-by-side", help="Two traj_*.npz files: the left pane, then the right."),
    ],
    labels: Annotated[tuple[str, str], typer.Option(help="A label for each pane.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="The .mp4 or .gif to write.")],
    fps: Annotated[
        int, typer.Option(help="Frames per second; must divide the 100 Hz control rate.")
    ] = 50,
    size: Annotated[
        int, typer.Option(help="Pane size in px (each pane is SIZE x SIZE; a multiple of 16).")
    ] = 480,
) -> None:
    """Replay two stored trajectories side by side on one clock (nothing is re-simulated): a
    pane that fell or errored plays its epilogue, then holds its last frame, stamped."""
    from topple.rendering import RenderingUnavailableError
    from topple.video import render_side_by_side

    _check_output(out.parent)
    trajs = [_load_trajectory(path) for path in side_by_side]
    try:
        frames = render_side_by_side(trajs, list(labels), out, pane=size, fps=fps)
    except (ValueError, RenderingUnavailableError) as exc:
        _fail(str(exc))
    except OSError as exc:
        _fail(f"cannot write {out}: {exc.strerror or type(exc).__name__}")
    typer.echo(f"wrote {out}: {frames} frames at {fps} fps, {2 * size}x{size}")


def _load_trajectory(path: Path) -> Trajectory:
    """A stored trajectory, loaded with ``load_npz`` bounded by its own challenge's episode
    length (the challenge it names, a bundle Topple ships); exit 2 for anything else."""
    from topple.challenge import load_challenge
    from topple.traj import load_npz

    challenge_id = _read_meta(path).get("challenge")
    if not isinstance(challenge_id, str):
        _fail(f"{path.name}: the trajectory does not name its challenge")
    try:
        challenge = load_challenge(challenge_id)
    except (ValueError, OSError) as exc:
        _fail(f"{path.name}: {exc}")
    from topple.kinds import traj_layout

    cfg = challenge.config
    try:
        return load_npz(
            path,
            max_steps=round(cfg.episode_seconds / cfg.control_dt),
            layout=traj_layout(challenge),
        )
    except ValueError as exc:
        _fail(str(exc))


# ------------------------------------------------------------------ training and export

TRAIN_EXTRA_MODULES = frozenset({"torch", "stable_baselines3", "tensorboard", "psutil"})
MAX_RETRIES = 2
EXIT_HARD_LIMIT = 4


@app.command()
def train(
    config: Annotated[Path, typer.Argument(help="A configs/train/*.yaml file.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="The run directory to create (must be empty).")
    ],
    seed: Annotated[
        int | None,
        typer.Option(min=0, max=2**32 - 1, help="Replaces the config's seed (0, 1, 2, ...)."),
    ] = None,
    retry: Annotated[
        bool,
        typer.Option(
            "--retry",
            help="Re-run a run that stopped on its wall-clock limit: the old --out becomes "
            "<out>.attempt1 (then .attempt2).",
        ),
    ] = False,
) -> None:
    """Train a policy (PPO or SAC) with periodic selection evals and pickle-free snapshots.
    Needs the train extra (torch, stable-baselines3). Exits 4 when a hard limit stopped it."""
    from topple.train.config import config_sha256, read_train_config
    from topple.yamlsafe import ConfigError

    _check_output(out)
    try:
        loaded = read_train_config(config, seed=seed)
    except ConfigError as exc:
        _fail(f"{config.name}: {exc}")
    attempt = _retry_target(out, retry=retry, config_sha256=config_sha256(loaded.config))
    learn = _learn_module()
    if attempt is not None:
        try:
            os.rename(out, attempt)
        except OSError as exc:
            _fail(f"cannot move {out} to {attempt}: {exc.strerror or type(exc).__name__}")
        typer.echo(f"retry: moved the aborted run to {attempt}")
    try:
        run_dir = learn.train(loaded.config, out, config_file_sha256=loaded.file_sha256)
    except learn.InitError as exc:  # a fine-tune's source (§8.8): before anything is written
        _fail(f"{config.name}: init: {exc}")
    if attempt is not None:
        earlier = [
            out.with_name(f"{out.name}.attempt{k}")
            for k in range(1, MAX_RETRIES + 1)
            if os.path.isdir(out.with_name(f"{out.name}.attempt{k}"))
        ]
        _flag_if_not_reproduced(Path(run_dir), earlier)
    if _hard_limit_flags(Path(run_dir)):
        raise typer.Exit(code=EXIT_HARD_LIMIT)


def _flag_if_not_reproduced(run_dir: Path, attempts: Sequence[Path]) -> None:
    """§1: a retry must reproduce every earlier attempt's ``onnx_sha256`` at every label it
    shares with each (not just the latest, which may itself have diverged), and inherits a
    ``nondeterministic`` flag from any of them; otherwise its ``train_meta.json`` gains
    ``nondeterministic`` (in ``FLAGS`` order) and the run is never shipped or reported (G10)."""
    import json

    from topple._atomic import replacement
    from topple.export import ExportError, read_run_json
    from topple.train.selection import FLAGS, NONDETERMINISTIC, reproduces

    try:
        new = read_run_json(run_dir / "train_meta.json")
    except ExportError as exc:
        _fail(f"--retry: cannot read {run_dir}/train_meta.json: {exc}")
    if not isinstance(new, dict):
        _fail(f"--retry: train_meta.json of {run_dir} is not an object")
    new_rows = new.get("selection_history")
    diverged: list[str] = []
    for attempt in attempts:
        try:
            old = read_run_json(attempt / "train_meta.json")
        except ExportError as exc:
            _fail(f"--retry: cannot compare with {attempt.name}: {exc}")
        if not isinstance(old, dict):
            _fail(f"--retry: train_meta.json of {attempt.name} is not an object")
        old_rows, old_flags = old.get("selection_history"), old.get("flags")
        same = (
            isinstance(new_rows, list)
            and isinstance(old_rows, list)
            and reproduces(new_rows, old_rows)
            and not (isinstance(old_flags, list) and NONDETERMINISTIC in old_flags)
        )
        if not same:
            diverged.append(attempt.name)
    if not diverged:
        return
    raised = {*new.get("flags", []), NONDETERMINISTIC}
    new["flags"] = [f for f in FLAGS if f in raised]
    with replacement(run_dir / "train_meta.json", suffix=".tmp") as tmp:
        tmp.write_text(json.dumps(new, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    typer.echo(
        f"retry: the run does not reproduce {', '.join(diverged)}: flagged {NONDETERMINISTIC}"
    )


def _retry_target(out: Path, *, retry: bool, config_sha256: str) -> Path | None:
    """Where ``--retry`` moves the existing ``out`` (``None``: nothing to move). Exit 2 if
    ``out`` exists and is not an empty directory, unless ``--retry`` and its
    ``train_meta.json`` carries ``wall_clock_abort`` and the ``config_sha256`` of this run's
    effective config (the same config and seed: a retry re-runs the attempt and must reproduce
    it, §1), and fewer than 2 attempts are kept."""
    from topple.export import ExportError, read_run_json

    if not os.path.lexists(out) or _empty_directory(out):
        return None
    if not retry:
        _fail(
            f"--out {out} exists and is not empty (a run never writes into another; --retry "
            "re-runs a run that stopped on its wall-clock limit)"
        )
    try:
        meta = read_run_json(out / "train_meta.json")
    except ExportError as exc:
        _fail(f"--retry: {exc}")
    if not isinstance(meta, dict):
        meta = {}
    flags = meta.get("flags")
    if not isinstance(flags, list) or "wall_clock_abort" not in flags:
        _fail(
            f"--retry re-runs only a run whose train_meta.json carries wall_clock_abort; {out} "
            "does not (rss_abort and cpu_abort are bugs to fix, not to re-run)"
        )
    if meta.get("config_sha256") != config_sha256:
        _fail(
            f"--retry re-runs the attempt in {out} with its own config and seed: its "
            "train_meta.json config_sha256 is not this run's (check the config file and --seed)"
        )
    if out.name in ("", ".", ".."):
        _fail(f"--retry needs a named --out, not {out}")
    for k in range(1, MAX_RETRIES + 1):
        target = out.with_name(f"{out.name}.attempt{k}")
        if not os.path.lexists(target):
            return target
    _fail(f"{out} was already retried {MAX_RETRIES} times: wait for a lower load")


def _empty_directory(path: Path) -> bool:
    try:
        if not os.path.isdir(path) or os.path.islink(path):
            return False
        with os.scandir(path) as listing:
            return next(listing, None) is None
    except OSError as exc:
        _fail(f"cannot examine {path}: {exc.strerror or type(exc).__name__}")


def _learn_module() -> Any:
    """``topple.train.learn``, imported only here; exit 2 with one line without the train
    extra (or in a build that does not have the module)."""
    import importlib

    try:
        return importlib.import_module("topple.train.learn")
    except ImportError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing in TRAIN_EXTRA_MODULES:
            _fail(
                f"topple train needs the train extra ({missing} is not installed): "
                "uv sync --extra train"
            )
        if exc.name == "topple.train.learn":
            _fail("topple train is not available: this build has no topple.train.learn")
        raise


def _hard_limit_flags(run_dir: Path) -> list[str]:
    """The abort flags the finished run recorded (exit 4 when there are any)."""
    from topple.export import ExportError, read_run_json
    from topple.train.selection import ABORT_FLAGS

    try:
        meta = read_run_json(run_dir / "train_meta.json")
    except ExportError as exc:
        _fail(f"the run wrote no readable train_meta.json: {exc}")
    flags = meta.get("flags") if isinstance(meta, dict) else None
    if not isinstance(flags, list):
        _fail("the run's train_meta.json has no flags list")
    return [flag for flag in ABORT_FLAGS if flag in flags]


@app.command("export")
def export_cmd(
    run_dir: Annotated[Path, typer.Argument(help="A training run directory (runs/<name>).")],
    out: Annotated[Path, typer.Option("--out", "-o", help="The policy directory to write.")],
    step: Annotated[
        int | None,
        typer.Option(min=0, max=MAX_SEED - 1, help="The checkpoint (default: best.json's)."),
    ] = None,
    report_only: Annotated[
        bool,
        typer.Option(
            "--report-only",
            help="Export any checkpoint for a report row, to RUN_DIR/report_policy (replaced).",
        ),
    ] = False,
    test_fixture: Annotated[
        str | None,
        typer.Option(help="Also write tests/data/NAME_parity.{npz,bin,json}."),
    ] = None,
    replace_out: Annotated[
        bool, typer.Option("--replace", help="Replace an existing policy directory.")
    ] = False,
) -> None:
    """Export a training checkpoint as a topple.policy/1 directory, without torch or pickle:
    rebuild the ONNX from the checkpoint's arrays, bind it to the bytes training scored, check
    parity row by row, and write policy.onnx + policy.json."""
    import warnings

    from topple.export import FIXTURE_DIR, ExportError, ExportWarning, export_policy

    _check_output(out)
    if test_fixture is not None:
        _check_output(FIXTURE_DIR)
    error: str | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ExportWarning)
        try:
            manifest = export_policy(
                run_dir,
                out,
                step=step,
                report_only=report_only,
                replace=replace_out,
                test_fixture=test_fixture,
            )
        except ExportError as exc:
            error = str(exc)
    for w in caught:  # the export happened, cleanup after it did not; others pass through
        if issubclass(w.category, ExportWarning):
            typer.echo(f"warning: {_printable(str(w.message))}", err=True)
        else:
            warnings.showwarning(w.message, w.category, w.filename, w.lineno)
    if error is not None:
        _fail(error)
    parity = manifest.claims.export_parity
    if parity is None:  # export always records it
        _fail("internal error: the export recorded no parity")
    typer.echo(
        f"exported step {manifest.claims.selected_step} -> {out}: policy_sha256 "
        f"{manifest.onnx.sha256[:12]}, parity max |diff| {parity.max_abs_diff:.2e} over "
        f"{parity.n} rows" + (" (coverage vacuous: report only)" if parity.vacuous else "")
    )


# ------------------------------------------------------------------ policies


@policy_app.command("pack")
def policy_pack(
    model: Annotated[Path, typer.Argument(help="An ONNX model exported elsewhere (<= 1 MiB).")],
    challenge: Annotated[str, typer.Option(help="The challenge it is for, e.g. balance-v1.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="The policy directory to create (must not exist).")
    ],
) -> None:
    """Pack a third-party ONNX model into a topple.policy/1 directory: canonical policy.onnx
    (doc strings, value_info, foreign metadata, Identity and Constant nodes removed; I/O named
    obs/action, batch dimension symbolic) and policy.json, checked and smoke-tested exactly as
    loading it will be. Claims are left empty."""
    from topple import _atomic
    from topple._inputs import open_regular
    from topple.challenge import load_challenge
    from topple.loader import LoadError
    from topple.policy.pack import pack_policy
    from topple.policy.validate import MAX_ONNX_BYTES, safe

    _check_output(out)
    if os.path.lexists(out):
        _fail(f"{out} already exists: pack writes a new directory, never into an existing one")
    try:
        target = load_challenge(challenge)
    except (ValueError, OSError) as exc:
        _fail(str(exc))
    name = safe(model.name)
    try:
        fh, _ = open_regular(model, MAX_ONNX_BYTES)
        with fh:
            data = fh.read(MAX_ONNX_BYTES + 1)
    except ValueError as exc:
        _fail(f"cannot pack {name!r}: {exc}")
    except OSError as exc:
        _cannot_read(model, exc)
    try:
        files = pack_policy(data, challenge=target)
    except LoadError as exc:
        _fail(f"cannot pack {name!r}: {exc}")
    try:
        with _atomic.new_directory(out) as tmp:
            for file_name, content in files.items():
                _atomic.write_new_file(tmp / file_name, content)
    except OSError as exc:
        _fail(f"cannot write {out}: {exc.strerror or type(exc).__name__}")
    manifest = json.loads(files["policy.json"])
    typer.echo(
        f"packed {name} -> {out}: {manifest['challenge']}, action_rule {manifest['action_rule']}, "
        f"policy_sha256 {manifest['onnx']['sha256'][:12]}"
    )
