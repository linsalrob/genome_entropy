"""Download command for pre-downloading models and datasets."""

from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    typer = None


def download_command(
    model: str = typer.Option(
        "Rostlab/ProstT5_fp16",
        "--model",
        "-m",
        help="ProstT5 model to download",
    ),
    test_data: bool = typer.Option(
        False,
        "--test-data",
        help="Download test datasets",
    ),
) -> None:
    """Pre-download models and optional test datasets.
    
    Downloads ProstT5 model from HuggingFace to local cache.
    Optionally downloads small reference datasets for testing.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        
        typer.echo(f"Downloading ProstT5 model: {model}")
        typer.echo("This may take a few minutes on first run...")
        
        # Download tokenizer
        typer.echo("  - Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model)
        
        # Download model
        typer.echo("  - Downloading model...")
        model_obj = AutoModel.from_pretrained(model)
        
        # Get cache location
        cache_dir = Path.home() / ".cache" / "huggingface"
        typer.echo(f"\n✓ Model downloaded successfully to: {cache_dir}")
        
        if test_data:
            typer.echo("\nTest data download not yet implemented.")
            typer.echo("Use examples/example_small.fasta for testing.")
        
        typer.echo("\n✓ Download complete!")
        
    except ImportError:
        typer.echo("Error: transformers package required", err=True)
        typer.echo("Install with: pip install transformers torch", err=True)
        raise typer.Exit(2)
    except Exception as e:
        typer.echo(f"Error downloading model: {e}", err=True)
        raise typer.Exit(3)
