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
        help="Input JSON file with protein records",
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
        help="ProstT5 model name",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use (auto/cuda/mps/cpu)",
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        "-b",
        help="Batch size for encoding",
    ),
) -> None:
    """Encode proteins to 3Di structural tokens.

    Uses ProstT5 model to predict 3Di structural alphabet tokens
    directly from amino acid sequences.
    """
    try:
        from ...encode3di.prostt5 import ProstT5ThreeDiEncoder
        from ...io.jsonio import read_json, write_json
        from ...orf.types import OrfRecord
        from ...translate.translator import ProteinRecord

        typer.echo(f"Reading proteins from: {input}")
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

        typer.echo(f"\nInitializing ProstT5 encoder (model: {model})...")
        encoder = ProstT5ThreeDiEncoder(model_name=model, device=device)
        typer.echo(f"  Using device: {encoder.device}")

        typer.echo(f"\nEncoding to 3Di tokens...")
        three_dis = encoder.encode_proteins(proteins, batch_size)
        typer.echo(f"  Encoded {len(three_dis)} sequence(s)")

        typer.echo(f"\nWriting results to: {output}")
        write_json(three_dis, output)

        typer.echo("✓ 3Di encoding complete!")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        traceback.print_exc()
        raise typer.Exit(3)
