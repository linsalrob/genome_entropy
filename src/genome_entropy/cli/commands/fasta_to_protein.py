"""Convert protein FASTA to protein JSON format."""

from pathlib import Path

try:
    import typer
except ImportError:
    typer = None


def fasta_to_protein_command(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input protein FASTA file",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSON file with protein records",
    ),
) -> None:
    """Convert protein FASTA file to protein JSON format.
    
    Takes a FASTA file containing protein sequences and converts them
    to the protein JSON format required for input to encode3di.
    
    Since these are direct protein sequences (not translated from ORFs),
    minimal OrfRecord metadata is created for compatibility.
    """
    try:
        from ...io.fasta import read_fasta
        from ...io.jsonio import write_json
        from ...orf.types import OrfRecord
        from ...translate.translator import ProteinRecord
        
        typer.echo(f"Reading proteins from: {input}")
        sequences = read_fasta(input)
        
        typer.echo(f"  Loaded {len(sequences)} protein sequence(s)")
        
        # Convert to ProteinRecord objects
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
        
        typer.echo(f"\nConverted {len(proteins)} protein(s)")
        
        typer.echo(f"\nWriting results to: {output}")
        write_json(proteins, output)
        
        typer.echo("✓ Conversion complete!")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(3)
