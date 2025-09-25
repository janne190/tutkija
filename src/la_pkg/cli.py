"""CLI entrypoints for Tutkija."""

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Tutkija CLI-komennot."""


def _load_env_example() -> str:
    """Return the contents of the .env example template."""
    env_path = Path(".env.example")
    if not env_path.exists():
        raise FileNotFoundError("Puuttuva .env.example")
    return env_path.read_text(encoding="utf-8")


@app.command(name="hello")
def hello() -> None:
    """Tulosta konfiguraation mallisisältö."""
    typer.echo("Tutkija, konfiguraation malli alla")
    try:
        typer.echo(_load_env_example())
    except FileNotFoundError:
        typer.secho("Ei löydy .env.examplea – täydennä ennen ajoa", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
