"""Initial command-line entry point."""

from __future__ import annotations

import typer

app = typer.Typer(help="Budget-aware autonomous ML competition research agent.")


@app.callback(invoke_without_command=True)
def root() -> None:
    """Show the initial v0 command surface."""

    return None


def main() -> None:
    app()
