"""ORF extraction command."""

from pathlib import Path

try:
    import typer
except ImportError:
    typer = None


def orf_command(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input FASTA file",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSON file with ORF records",
    ),
    table: int = typer.Option(
        11,
        "--table",
        "-t",
        help="NCBI genetic code table ID",
    ),
    min_nt: int = typer.Option(
        90,
        "--min-nt",
        help="Minimum ORF length in nucleotides",
    ),
) -> None:
    """Extract ORFs from DNA sequences.

    Finds all Open Reading Frames (ORFs) in input FASTA sequences
    using the get_orfs binary. Outputs ORF records as JSON.
    """
    try:
        from ...io.fasta import read_fasta
        from ...io.jsonio import write_json
        from ...orf.finder import find_orfs

        typer.echo(f"Reading FASTA from: {input}")
        sequences = read_fasta(input)
        typer.echo(f"  Found {len(sequences)} sequence(s)")

        typer.echo(f"\nFinding ORFs (min length: {min_nt} nt, table: {table})...")
        orfs = find_orfs(sequences, table_id=table, min_nt_length=min_nt)
        typer.echo(f"  Found {len(orfs)} ORF(s)")

        typer.echo(f"\nWriting results to: {output}")
        write_json(orfs, output)

        typer.echo("✓ ORF extraction complete!")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(3)
