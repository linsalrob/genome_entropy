"""Token size estimation command."""

from pathlib import Path
from typing import Optional
import traceback

try:
    import typer
except ImportError:
    typer = None


def estimate_token_size_command(
    model: str = typer.Option(
        "gbouras13/modernprost-base",
        "--model",
        "-m",
        help="Model name (gbouras13/modernprost-base, gbouras13/modernprost-profiles, Rostlab/ProstT5, or Rostlab/ProstT5_fp16)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use (auto/cuda/mps/cpu)",
    ),
    start_length: int = typer.Option(
        3000,
        "--start",
        "-s",
        help="Starting total protein length to test (amino acids)",
    ),
    end_length: int = typer.Option(
        10000,
        "--end",
        "-e",
        help="Maximum total protein length to test (amino acids)",
    ),
    step: int = typer.Option(
        1000,
        "--step",
        help="Increment between test lengths (amino acids)",
    ),
    num_trials: int = typer.Option(
        3,
        "--trials",
        "-t",
        help="Number of trials per length for robustness",
    ),
    base_protein_length: int = typer.Option(
        100,
        "--base-length",
        "-b",
        help="Approximate length of individual proteins (amino acids)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Path to log file (default: log to STDOUT)",
    ),
) -> None:
    """Estimate optimal token size for GPU encoding.

    This command generates random protein sequences of increasing length
    and tests encoding to find the maximum that can be encoded on the
    available GPU without running out of memory.

    The recommended token size is returned as 90% of the maximum for safety.
    """
    try:
        from ...config import VALID_LOG_LEVELS, MODERNPROST_MODELS
        from ...logging_config import configure_logging
        from ...encode3di import ProstT5ThreeDiEncoder, ModernProstThreeDiEncoder, estimate_token_size

        # Validate and configure logging
        if log_level.upper() not in VALID_LOG_LEVELS:
            typer.echo(
                f"Error: Invalid log level '{log_level}'. Must be one of: {', '.join(VALID_LOG_LEVELS)}",
                err=True,
            )
            raise typer.Exit(2)

        configure_logging(level=log_level.upper(), log_file=log_file)

        typer.echo("=" * 60)
        typer.echo("Token Size Estimation")
        typer.echo("=" * 60)
        typer.echo(f"Model: {model}")
        typer.echo(f"Testing range: {start_length} to {end_length} AA (step: {step})")
        typer.echo(f"Trials per length: {num_trials}")
        typer.echo(f"Base protein length: {base_protein_length} AA")
        typer.echo("=" * 60)

        # Select encoder based on model name
        typer.echo("\nInitializing encoder...")
        if model in MODERNPROST_MODELS:
            encoder = ModernProstThreeDiEncoder(model_name=model, device=device)
        else:
            encoder = ProstT5ThreeDiEncoder(model_name=model, device=device)
        typer.echo(f"Using device: {encoder.device}")

        typer.echo("\nStarting estimation (this may take several minutes)...")
        typer.echo("Watch for Out of Memory errors to identify the limit.\n")

        results = estimate_token_size(
            encoder=encoder,
            start_length=start_length,
            end_length=end_length,
            step=step,
            num_trials=num_trials,
            base_protein_length=base_protein_length,
        )

        typer.echo("\n" + "=" * 60)
        typer.echo("RESULTS")
        typer.echo("=" * 60)
        typer.echo(f"Device: {results['device']}")
        typer.echo(f"Max successful length: {results['max_length']} amino acids")
        typer.echo(
            f"Recommended token size: {results['recommended_token_size']} amino acids"
        )
        typer.echo("\nTrials per length:")
        for length, trials in sorted(results["trials_per_length"].items()):
            status = "✓" if trials > 0 else "✗"
            typer.echo(f"  {status} {length} AA: {trials}/{num_trials} successful")
        typer.echo("=" * 60)
        typer.echo("\n✓ Estimation complete!")
        typer.echo(
            f"\nTo use this in encoding, add: --encoding-size {results['recommended_token_size']}"
        )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        traceback.print_exc()
        raise typer.Exit(3)
