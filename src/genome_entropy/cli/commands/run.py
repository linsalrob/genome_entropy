"""End-to-end pipeline command."""

from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    typer = None


def run_command(
    input: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input FASTA file with DNA sequences. Required if --genbank is not provided.",
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
        help="Device to use (auto/cuda/mps/cpu). Ignored if --multi-gpu is set.",
    ),
    skip_entropy: bool = typer.Option(
        False,
        "--skip-entropy",
        help="Skip entropy calculation",
    ),
    multi_gpu: bool = typer.Option(
        False,
        "--multi-gpu",
        help="Use multi-GPU parallel encoding when available",
    ),
    gpu_ids: Optional[str] = typer.Option(
        None,
        "--gpu-ids",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). "
             "If not specified, auto-discovers available GPUs.",
    ),
    genbank: Optional[Path] = typer.Option(
        None,
        "--genbank",
        "-g",
        help="GenBank file with DNA sequences and CDS annotations. "
             "Can be used instead of --input to provide both sequences and annotations. "
             "ORFs will be matched to GenBank CDS features by C-terminal sequence.",
        exists=True,
        dir_okay=False,
    ),
    encoding_size: int = typer.Option(
        10000,
        "--encoding-size",
        "-e",
        help="Encoding size (approximates to amino acids)",
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
    """Run the complete DNA to 3Di pipeline.
    
    Executes all pipeline steps:
    1. Find ORFs in DNA sequences
    2. Translate ORFs to proteins
    3. Encode proteins to 3Di tokens
    4. Calculate entropy at all levels
    5. Optionally match ORFs to GenBank CDS annotations
    
    Input options:
    - Provide --input for FASTA file (standard usage)
    - Provide --genbank for GenBank file (extracts sequences and CDS annotations)
    - Provide both --input and --genbank to use FASTA sequences with GenBank annotations
    - At least one of --input or --genbank must be provided
    
    Multi-GPU encoding can significantly speed up 3Di encoding by distributing
    batches across multiple GPUs. Use --multi-gpu to enable, and optionally
    specify --gpu-ids to select specific GPUs.
    """
    try:
        from ...config import VALID_LOG_LEVELS
        from ...logging_config import configure_logging
        from ...pipeline.runner import run_pipeline
        
        # Validate and configure logging
        if log_level.upper() not in VALID_LOG_LEVELS:
            typer.echo(
                f"Error: Invalid log level '{log_level}'. Must be one of: {', '.join(VALID_LOG_LEVELS)}",
                err=True,
            )
            raise typer.Exit(2)
        
        configure_logging(level=log_level.upper(), log_file=log_file)
        
        # Validate that at least one input source is provided
        if not input and not genbank:
            typer.echo("Error: Must provide either --input or --genbank", err=True)
            typer.echo("Use --help for more information", err=True)
            raise typer.Exit(2)
        
        # Parse GPU IDs if provided
        parsed_gpu_ids = None
        if gpu_ids:
            try:
                parsed_gpu_ids = [int(x.strip()) for x in gpu_ids.split(",")]
            except ValueError:
                typer.echo(f"Error: Invalid GPU IDs format: {gpu_ids}", err=True)
                typer.echo("Expected comma-separated integers, e.g., '0,1,2'", err=True)
                raise typer.Exit(2)
        
        typer.echo(f"Starting DNA to 3Di pipeline...")
        
        if input:
            typer.echo(f"  Input FASTA: {input}")
        if genbank:
            typer.echo(f"  GenBank file: {genbank}")
            if not input:
                typer.echo(f"  (Using GenBank for DNA sequences)")
        
        typer.echo(f"  Output: {output}")
        typer.echo(f"  Genetic code table: {table}")
        typer.echo(f"  Minimum AA length: {min_aa}")
        typer.echo(f"  Model: {model}")
        typer.echo(f"  Encoding size: {encoding_size}")
        
        if multi_gpu:
            typer.echo(f"  Multi-GPU encoding: enabled")
            if parsed_gpu_ids:
                typer.echo(f"  GPU IDs: {parsed_gpu_ids}")
            else:
                typer.echo(f"  GPU IDs: auto-discover")
        else:
            typer.echo(f"  Device: {device if device else 'auto'}")
        
        typer.echo(f"\nRunning pipeline...")
        
        results = run_pipeline(
            input_fasta=input,
            table_id=table,
            min_aa_len=min_aa,
            model_name=model,
            compute_entropy=not skip_entropy,
            output_json=output,
            device=device,
            use_multi_gpu=multi_gpu,
            gpu_ids=parsed_gpu_ids,
            genbank_file=genbank,
            encoding_size=encoding_size,
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
