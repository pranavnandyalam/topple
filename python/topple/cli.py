"""`topple` command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from topple import __version__

app = typer.Typer(help="Topple: CI for robots.", no_args_is_help=True)


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
    from topple.smoke import run_smoke

    result = run_smoke(out, steps=steps)
    typer.echo(f"smoke ok: {result.steps} steps -> {result.image_path}")
