"""Download command for pre-downloading models and datasets."""

from pathlib import Path

try:
    import typer
except ImportError:
    typer = None


def _is_model_cached(model_name: str) -> bool:
    """Check if a model is already cached locally.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        True if model is cached, False otherwise
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        # Try to find the model in the cache
        cache_path = try_to_load_from_cache(
            repo_id=model_name,
            filename="config.json",
        )

        # If we get a path (not None or _CACHED_NO_EXIST), model is cached
        if cache_path is not None and isinstance(cache_path, (str, Path)):
            return True

        return False
    except Exception:
        # If we can't check the cache, assume it's not cached
        return False


def download_command(
    model: str = typer.Option(
        "Rostlab/ProstT5_fp16",
        "--model",
        "-m",
        help="Model to download (Rostlab/ProstT5_fp16, gbouras13/modernprost-base, or gbouras13/modernprost-profiles)",
    ),
    test_data: bool = typer.Option(
        False,
        "--test-data",
        help="Download test datasets",
    ),
) -> None:
    """Pre-download models and optional test datasets.

    Downloads ProstT5 or ModernProst models from HuggingFace to local cache.
    Optionally downloads small reference datasets for testing.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        from ...config import MODERNPROST_MODELS

        typer.echo(f"Downloading model: {model}")

        # Check if model is already cached
        is_cached = _is_model_cached(model)
        if is_cached:
            typer.echo("Model found in cache, verifying...")
        else:
            typer.echo("This may take a few minutes on first run...")

        # Check if this is a ModernProst model
        is_modernprost = model in MODERNPROST_MODELS

        # Download tokenizer
        typer.echo("  - Downloading tokenizer...")
        if is_modernprost:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=True,
                local_files_only=is_cached,
                force_download=not is_cached,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                use_fast=False,
                legacy=True,
                local_files_only=is_cached,
                force_download=not is_cached,
            )

        # Download model
        typer.echo("  - Downloading model...")
        if is_modernprost:
            model_obj = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
                local_files_only=is_cached,
                force_download=not is_cached,
            )
        else:
            model_obj = AutoModel.from_pretrained(
                model,
                local_files_only=is_cached,
                force_download=not is_cached,
            )

        # Get cache location
        cache_dir = Path.home() / ".cache" / "huggingface"
        typer.echo(f"\n✓ Model downloaded successfully to: {cache_dir}")

        if test_data:
            typer.echo("\nTest data download not yet implemented.")
            typer.echo("Use examples/example_small.fasta for testing.")

        typer.echo("\n✓ Download complete!")

    except ImportError as e:
        error_msg = str(e)
        typer.echo(f"Error: Import Error loading AutoTokenizer: {error_msg}", err=True)

        # Check if it's a ModernBert-related error
        if "ModernBertModel" in error_msg or "ModernBert" in error_msg:
            typer.echo("\n" + "=" * 60, err=True)
            typer.echo("ModernProst models require transformers >= 4.47.0", err=True)
            typer.echo("Please upgrade transformers:", err=True)
            typer.echo("  pip install --upgrade 'transformers>=4.47.0'", err=True)
            typer.echo("=" * 60, err=True)
        else:
            typer.echo("Error: transformers package required", err=True)
            typer.echo("Install with: pip install transformers torch", err=True)
        raise typer.Exit(2)
    except Exception as e:
        typer.echo(f"Error downloading model: {e}", err=True)
        raise typer.Exit(3)
