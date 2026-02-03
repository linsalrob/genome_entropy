"""Translation command."""

from pathlib import Path
from typing import List

try:
    import typer
except ImportError:
    typer = None


def translate_command(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input JSON file with ORF records",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSON file with protein records",
    ),
    table: int = typer.Option(
        11,
        "--table",
        "-t",
        help="NCBI genetic code table ID",
    ),
) -> None:
    """Translate ORFs to protein sequences.

    Takes ORF records from JSON and translates them to amino acid
    sequences using the specified genetic code table.
    """
    try:
        from ...io.jsonio import read_json, write_json
        from ...orf.types import OrfRecord
        from ...translate.translator import translate_orfs

        typer.echo(f"Reading ORFs from: {input}")
        orf_data = read_json(input)

        # Reconstruct OrfRecord objects
        if isinstance(orf_data, list):
            orfs = [OrfRecord(**orf) for orf in orf_data]
        else:
            raise ValueError("Invalid ORF JSON format")

        typer.echo(f"  Loaded {len(orfs)} ORF(s)")

        typer.echo(f"\nTranslating ORFs (table: {table})...")
        proteins = translate_orfs(orfs, table_id=table)
        typer.echo(f"  Translated {len(proteins)} protein(s)")

        typer.echo(f"\nWriting results to: {output}")
        write_json(proteins, output)

        typer.echo("✓ Translation complete!")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(3)
