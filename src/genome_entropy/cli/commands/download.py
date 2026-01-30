"""Download command for pre-downloading models and datasets."""

from pathlib import Path

try:
    import typer
except ImportError:
    typer = None


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
        
        typer.echo(f"Downloading model: {model}")
        typer.echo("This may take a few minutes on first run...")
        
        # Check if this is a ModernProst model
        is_modernprost = model in {
            "gbouras13/modernprost-base",
            "gbouras13/modernprost-profiles",
        }
        
        # Download tokenizer
        typer.echo("  - Downloading tokenizer...")
        if is_modernprost:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                use_fast=False,
                legacy=True
            )
        
        # Download model
        typer.echo("  - Downloading model...")
        if is_modernprost:
            model_obj = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
            )
        else:
            model_obj = AutoModel.from_pretrained(model)
        
        # Get cache location
        cache_dir = Path.home() / ".cache" / "huggingface"
        typer.echo(f"\n✓ Model downloaded successfully to: {cache_dir}")
        
        if test_data:
            typer.echo("\nTest data download not yet implemented.")
            typer.echo("Use examples/example_small.fasta for testing.")
        
        typer.echo("\n✓ Download complete!")
        
    except ImportError as e:
        typer.echo(f"Error: Import Error loading AutoTokenizer: {e}", err=True)
        typer.echo("Error: transformers package required", err=True)
        typer.echo("Install with: pip install transformers torch", err=True)
        raise typer.Exit(2)
    except Exception as e:
        typer.echo(f"Error downloading model: {e}", err=True)
        raise typer.Exit(3)
