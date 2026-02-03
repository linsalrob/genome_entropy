"""3Di encoding command."""

from pathlib import Path
from typing import Optional
import traceback

try:
    import typer
except ImportError:
    typer = None


def encode3di_command(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input protein file (FASTA or JSON format)",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSON file with 3Di records",
    ),
    model: str = typer.Option(
        "Rostlab/ProstT5_fp16",
        "--model",
        "-m",
        help="Model name (Rostlab/ProstT5_fp16, gbouras13/modernprost-base, or gbouras13/modernprost-profiles)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use (auto/cuda/mps/cpu). Ignored if --multi-gpu is set.",
    ),
    encoding_size: int = typer.Option(
        10000,
        "--encoding-size",
        "-e",
        help="Encoding size (approximates to amino acids)",
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
) -> None:
    """Encode proteins to 3Di structural tokens.

    Uses ProstT5 or ModernProst models to predict 3Di structural alphabet tokens
    directly from amino acid sequences.
    
    Input formats:
    - FASTA file: Protein sequences in FASTA format (.fasta, .fa, .faa)
    - JSON file: Protein records in JSON format (output from translate or fasta-to-protein)
    
    Available models:
    - Rostlab/ProstT5_fp16 (default, original ProstT5 model)
    - gbouras13/modernprost-base (newer base model)
    - gbouras13/modernprost-profiles (newer model with profile support)
    
    Multi-GPU encoding can significantly speed up encoding by distributing
    batches across multiple GPUs. Use --multi-gpu to enable, and optionally
    specify --gpu-ids to select specific GPUs.
    """
    try:
        from ...config import MODERNPROST_MODELS
        from ...encode3di.prostt5 import ProstT5ThreeDiEncoder
        from ...encode3di.modernprost import ModernProstThreeDiEncoder
        from ...io.jsonio import read_json, write_json
        from ...io.fasta import read_fasta
        from ...orf.types import OrfRecord
        from ...translate.translator import ProteinRecord

        # Parse GPU IDs if provided
        parsed_gpu_ids = None
        if gpu_ids:
            try:
                parsed_gpu_ids = [int(x.strip()) for x in gpu_ids.split(",")]
            except ValueError:
                typer.echo(f"Error: Invalid GPU IDs format: {gpu_ids}", err=True)
                typer.echo("Expected comma-separated integers, e.g., '0,1,2'", err=True)
                raise typer.Exit(2)

        typer.echo(f"Reading proteins from: {input}")
        
        # Detect input format based on file extension
        input_str = str(input).lower()
        is_fasta = input_str.endswith(('.fasta', '.fa', '.faa'))
        
        if is_fasta:
            # Read FASTA file and convert to ProteinRecord objects
            typer.echo("  Detected FASTA format")
            sequences = read_fasta(input)
            
            # Convert to ProteinRecord objects (similar to fasta_to_protein)
            proteins = []
            for seq_id, aa_sequence in sequences.items():
                # Create a minimal OrfRecord for compatibility
                # These proteins are not from ORFs, so we use placeholder values
                orf = OrfRecord(
                    parent_id=seq_id,
                    orf_id=seq_id,
                    start=0,
                    end=len(aa_sequence) * 3,  # Approximate nucleotide length
                    strand="+",
                    frame=0,
                    nt_sequence="",  # No nucleotide sequence available
                    aa_sequence=aa_sequence,
                    table_id=11,  # Default genetic code table
                    has_start_codon=False,
                    has_stop_codon=False,
                    in_genbank=False,
                )
                
                protein = ProteinRecord(
                    orf=orf,
                    aa_sequence=aa_sequence,
                    aa_length=len(aa_sequence),
                )
                proteins.append(protein)
        else:
            # Read JSON file (original behavior)
            typer.echo("  Detected JSON format")
            protein_data = read_json(input)
            
            # Reconstruct ProteinRecord objects
            if isinstance(protein_data, list):
                proteins = []
                for p in protein_data:
                    orf = OrfRecord(**p["orf"])
                    protein = ProteinRecord(
                        orf=orf,
                        aa_sequence=p["aa_sequence"],
                        aa_length=p["aa_length"],
                    )
                    proteins.append(protein)
            else:
                raise ValueError("Invalid protein JSON format")

        typer.echo(f"  Loaded {len(proteins)} protein(s)")

        # Select encoder based on model name
        typer.echo(f"\nInitializing encoder (model: {model})...")
        if model in MODERNPROST_MODELS:
            encoder = ModernProstThreeDiEncoder(model_name=model, device=device)
        else:
            encoder = ProstT5ThreeDiEncoder(model_name=model, device=device)
        
        if multi_gpu:
            typer.echo(f"  Multi-GPU encoding: enabled")
            if parsed_gpu_ids:
                typer.echo(f"  GPU IDs: {parsed_gpu_ids}")
            else:
                typer.echo(f"  GPU IDs: auto-discover")
        else:
            typer.echo(f"  Using device: {encoder.device}")

        typer.echo(f"\nEncoding to 3Di tokens...")
        three_dis = encoder.encode_proteins(
            proteins,
            encoding_size,
            use_multi_gpu=multi_gpu,
            gpu_ids=parsed_gpu_ids,
        )
        typer.echo(f"  Encoded {len(three_dis)} sequence(s)")

        typer.echo(f"\nWriting results to: {output}")
        write_json(three_dis, output)

        typer.echo("✓ 3Di encoding complete!")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        traceback.print_exc()
        raise typer.Exit(3)
