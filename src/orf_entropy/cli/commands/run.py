"""End-to-end pipeline command."""

from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    typer = None


def run_command(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input FASTA file with DNA sequences",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSON file with complete pipeline results",
    ),
    table: int = typer.Option(
        11,
        "--table",
        "-t",
        help="NCBI genetic code table ID",
    ),
    min_aa: int = typer.Option(
        30,
        "--min-aa",
        help="Minimum protein length in amino acids",
    ),
    model: str = typer.Option(
        "Rostlab/ProstT5_fp16",
        "--model",
        "-m",
        help="ProstT5 model name",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use (auto/cuda/mps/cpu)",
    ),
    skip_entropy: bool = typer.Option(
        False,
        "--skip-entropy",
        help="Skip entropy calculation",
    ),
) -> None:
    """Run the complete DNA to 3Di pipeline.
    
    Executes all pipeline steps:
    1. Find ORFs in DNA sequences
    2. Translate ORFs to proteins
    3. Encode proteins to 3Di tokens
    4. Calculate entropy at all levels
    """
    try:
        from ...pipeline.runner import run_pipeline
        
        typer.echo(f"Starting DNA to 3Di pipeline...")
        typer.echo(f"  Input: {input}")
        typer.echo(f"  Output: {output}")
        typer.echo(f"  Genetic code table: {table}")
        typer.echo(f"  Minimum AA length: {min_aa}")
        typer.echo(f"  Model: {model}")
        
        typer.echo(f"\nRunning pipeline...")
        
        results = run_pipeline(
            input_fasta=input,
            table_id=table,
            min_aa_len=min_aa,
            model_name=model,
            compute_entropy=not skip_entropy,
            output_json=output,
            device=device,
        )
        
        typer.echo(f"\n✓ Pipeline complete!")
        typer.echo(f"  Processed {len(results)} sequence(s)")
        
        total_orfs = sum(len(r.orfs) for r in results)
        total_proteins = sum(len(r.proteins) for r in results)
        typer.echo(f"  Found {total_orfs} ORF(s)")
        typer.echo(f"  Translated {total_proteins} protein(s)")
        typer.echo(f"  Results saved to: {output}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(3)
