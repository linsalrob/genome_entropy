"""Main CLI entry point for genome_entropy."""

from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    typer = None

from ..config import __version__, DEFAULT_LOG_LEVEL, VALID_LOG_LEVELS
from ..logging_config import configure_logging

# Create main app
if typer:
    app = typer.Typer(
        name="genome_entropy",
        help="DNA to 3Di pipeline: Convert DNA sequences to ORFs, proteins, and 3Di structural tokens with entropy analysis.",
        add_completion=False,
    )

    @app.callback(invoke_without_command=True)
    def main(
        ctx: typer.Context,
        version: bool = typer.Option(
            False, "--version", "-v", help="Show version and exit"
        ),
        log_level: str = typer.Option(
            DEFAULT_LOG_LEVEL,
            "--log-level",
            "-l",
            help=f"Logging level ({', '.join(VALID_LOG_LEVELS)})",
        ),
        log_file: Optional[Path] = typer.Option(
            None,
            "--log-file",
            help="Path to log file (default: log to STDOUT)",
        ),
    ) -> None:
        """DNA to 3Di pipeline with entropy analysis."""
        if version:
            typer.echo(f"genome_entropy version {__version__}")
            raise typer.Exit()

        # Configure logging before executing any commands
        if ctx.invoked_subcommand is not None:
            # Validate log level
            if log_level.upper() not in VALID_LOG_LEVELS:
                typer.echo(
                    f"Error: Invalid log level '{log_level}'. Must be one of: {', '.join(VALID_LOG_LEVELS)}",
                    err=True,
                )
                raise typer.Exit(2)

            configure_logging(level=log_level.upper(), log_file=log_file)

        if ctx.invoked_subcommand is None:
            typer.echo(ctx.get_help())
            raise typer.Exit()

    # Import and register commands
    try:
        from .commands import (
            download,
            encode3di,
            entropy,
            orf,
            run,
            translate,
            estimate_tokens,
        )

        app.command(name="download")(download.download_command)
        app.command(name="orf")(orf.orf_command)
        app.command(name="translate")(translate.translate_command)
        app.command(name="encode3di")(encode3di.encode3di_command)
        app.command(name="entropy")(entropy.entropy_command)
        app.command(name="run")(run.run_command)
        app.command(name="estimate-tokens")(estimate_tokens.estimate_token_size_command)
    except ImportError as e:
        # Commands not yet implemented
        pass

else:
    # Typer not installed
    def app() -> None:
        """Placeholder when typer is not installed."""
        print("Error: typer package is required for CLI functionality")
        print("Install with: pip install typer")
        exit(1)


if __name__ == "__main__":
    if typer:
        app()
    else:
        app()
