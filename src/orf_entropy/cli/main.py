"""Main CLI entry point for dna23di."""

try:
    import typer
except ImportError:
    typer = None

from ..config import __version__

# Create main app
if typer:
    app = typer.Typer(
        name="dna23di",
        help="DNA to 3Di pipeline: Convert DNA sequences to ORFs, proteins, and 3Di structural tokens with entropy analysis.",
        add_completion=False,
    )

    @app.callback(invoke_without_command=True)
    def main(
        ctx: typer.Context,
        version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    ) -> None:
        """DNA to 3Di pipeline with entropy analysis."""
        if version:
            typer.echo(f"dna23di version {__version__}")
            raise typer.Exit()
        
        if ctx.invoked_subcommand is None:
            typer.echo(ctx.get_help())
            raise typer.Exit()

    # Import and register commands
    try:
        from .commands import download, encode3di, entropy, orf, run, translate, estimate_tokens
        
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
