"""Entropy calculation command."""

from pathlib import Path

try:
    import typer
except ImportError:
    typer = None


def entropy_command(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input JSON file with structural-state records",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSON file with entropy report",
    ),
) -> None:
    """Calculate Shannon entropy at all representation levels.

    Computes entropy for DNA, ORF nucleotides, proteins, 3Di, and 12-state tokens.
    """
    try:
        from ...entropy.shannon import (
            EntropyReport,
            calculate_entropies_for_sequences,
            mutual_information,
        )
        from ...io.jsonio import read_json, write_json
        from ...orf.types import OrfRecord
        from ...translate.translator import ProteinRecord
        from ...encode3di.prostt5 import ThreeDiRecord
        from ...config import (
            THREEDDI_ALPHABET_SIZE,
            TWELVE_STATE_ALPHABET_SIZE,
        )

        typer.echo(f"Reading 3Di records from: {input}")
        three_di_data = read_json(input)

        # Reconstruct ThreeDiRecord objects
        if isinstance(three_di_data, list):
            three_dis = []
            for td in three_di_data:
                orf = OrfRecord(**td["protein"]["orf"])
                protein = ProteinRecord(
                    orf=orf,
                    aa_sequence=td["protein"]["aa_sequence"],
                    aa_length=td["protein"]["aa_length"],
                )
                three_di = ThreeDiRecord(
                    protein=protein,
                    three_di=td["three_di"],
                    method=td["method"],
                    model_name=td["model_name"],
                    inference_device=td["inference_device"],
                    twelve_state=td.get("twelve_state"),
                )
                three_dis.append(three_di)
        else:
            raise ValueError("Invalid 3Di JSON format")

        typer.echo(f"  Loaded {len(three_dis)} 3Di record(s)")

        typer.echo("\nCalculating entropy...")

        # Calculate entropies at different levels
        orf_nt_seqs = {
            td.protein.orf.orf_id: td.protein.orf.nt_sequence for td in three_dis
        }
        protein_aa_seqs = {
            td.protein.orf.orf_id: td.protein.aa_sequence for td in three_dis
        }
        three_di_seqs = {td.protein.orf.orf_id: td.three_di for td in three_dis}
        twelve_state_seqs = {
            td.protein.orf.orf_id: td.twelve_state
            for td in three_dis
            if td.twelve_state is not None
        }

        orf_nt_entropy = calculate_entropies_for_sequences(orf_nt_seqs)
        protein_aa_entropy = calculate_entropies_for_sequences(protein_aa_seqs)
        three_di_entropy = calculate_entropies_for_sequences(three_di_seqs)
        twelve_state_entropy = (
            calculate_entropies_for_sequences(twelve_state_seqs)
            if twelve_state_seqs
            else None
        )
        three_di_twelve_state_mutual_information = (
            {
                td.protein.orf.orf_id: mutual_information(td.three_di, td.twelve_state)
                for td in three_dis
                if td.twelve_state is not None
            }
            if twelve_state_seqs
            else None
        )

        # Create report
        report = EntropyReport(
            dna_entropy_global=0.0,  # Not available from 3Di records alone
            orf_nt_entropy=orf_nt_entropy,
            protein_aa_entropy=protein_aa_entropy,
            three_di_entropy=three_di_entropy,
            alphabet_sizes={
                "dna": 4,
                "protein": 20,
                "three_di": THREEDDI_ALPHABET_SIZE,
                "twelve_state": TWELVE_STATE_ALPHABET_SIZE,
            },
            twelve_state_entropy=twelve_state_entropy,
            three_di_twelve_state_mutual_information=(
                three_di_twelve_state_mutual_information
            ),
        )

        typer.echo(f"  Calculated entropy for {len(orf_nt_entropy)} sequence(s)")

        typer.echo(f"\nWriting results to: {output}")
        write_json(report, output)

        typer.echo("✓ Entropy calculation complete!")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(3)
