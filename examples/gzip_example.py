#!/usr/bin/env python3
"""Example demonstrating automatic gzip support for all file types.

This example shows how to use gzipped files with genome_entropy I/O functions.
The library automatically detects gzipped files by their .gz extension and
handles compression/decompression transparently.
"""

from pathlib import Path
import tempfile

from genome_entropy.io.fasta import read_fasta, write_fasta
from genome_entropy.io.jsonio import read_json, write_json


def main():
    """Demonstrate gzip support with examples."""
    print("=" * 70)
    print("Genome Entropy - Gzip Support Example")
    print("=" * 70)
    print()

    # Create a temporary directory for examples
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Example sequences
        sequences = {
            "seq1": "ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTGA",
            "seq2": "ATGAAAAAAAAAAAAAAAAAATAA",
            "seq3": "ATGGGGGGGGGGGGGGGGGGTAG",
        }

        print("1. FASTA File Compression")
        print("-" * 70)

        # Write plain FASTA
        plain_fasta = tmpdir / "sequences.fasta"
        write_fasta(sequences, plain_fasta)
        print(f"   Wrote plain FASTA: {plain_fasta.name}")
        print(f"   Size: {plain_fasta.stat().st_size} bytes")

        # Write gzipped FASTA (just add .gz extension!)
        gzipped_fasta = tmpdir / "sequences.fasta.gz"
        write_fasta(sequences, gzipped_fasta)
        print(f"   Wrote gzipped FASTA: {gzipped_fasta.name}")
        print(f"   Size: {gzipped_fasta.stat().st_size} bytes")

        compression_ratio = (
            1 - gzipped_fasta.stat().st_size / plain_fasta.stat().st_size
        ) * 100
        print(f"   Compression: {compression_ratio:.1f}% smaller")
        print()

        # Read both formats
        seqs_plain = read_fasta(plain_fasta)
        seqs_gzipped = read_fasta(gzipped_fasta)
        print(f"   Read {len(seqs_plain)} sequences from plain FASTA")
        print(f"   Read {len(seqs_gzipped)} sequences from gzipped FASTA")
        assert seqs_plain == seqs_gzipped, "Sequences should match!"
        print("   ✓ Content matches perfectly")
        print()

        print("2. JSON File Compression")
        print("-" * 70)

        # Create sample JSON data
        json_data = {
            "metadata": {"version": "1.0", "type": "example"},
            "sequences": sequences,
            "features": [
                {"id": f"feat_{i}", "type": "ORF", "score": 0.95 + i * 0.01}
                for i in range(10)
            ],
        }

        # Write plain JSON
        plain_json = tmpdir / "data.json"
        write_json(json_data, plain_json)
        print(f"   Wrote plain JSON: {plain_json.name}")
        print(f"   Size: {plain_json.stat().st_size} bytes")

        # Write gzipped JSON (just add .gz extension!)
        gzipped_json = tmpdir / "data.json.gz"
        write_json(json_data, gzipped_json)
        print(f"   Wrote gzipped JSON: {gzipped_json.name}")
        print(f"   Size: {gzipped_json.stat().st_size} bytes")

        compression_ratio = (
            1 - gzipped_json.stat().st_size / plain_json.stat().st_size
        ) * 100
        print(f"   Compression: {compression_ratio:.1f}% smaller")
        print()

        # Read both formats
        data_plain = read_json(plain_json)
        data_gzipped = read_json(gzipped_json)
        print(f"   Read JSON data from plain file")
        print(f"   Read JSON data from gzipped file")
        assert data_plain == data_gzipped, "Data should match!"
        print("   ✓ Content matches perfectly")
        print()

        print("3. Usage in Pipelines")
        print("-" * 70)
        print("   The gzip support works automatically in all CLI commands:")
        print()
        print("   # Input files - automatically detect .gz")
        print("   $ dna23di orf --input sequences.fasta.gz --output orfs.json")
        print()
        print("   # Output files - automatically compress if .gz")
        print("   $ dna23di orf --input sequences.fasta --output orfs.json.gz")
        print()
        print("   # Mix and match as needed")
        print(
            "   $ dna23di run --input sequences.fasta.gz --output results.json.gz"
        )
        print()

        print("4. Benefits")
        print("-" * 70)
        print("   ✓ Transparent: No code changes needed")
        print("   ✓ Space efficient: 40-90% size reduction")
        print("   ✓ All file types: FASTA, JSON, GenBank, etc.")
        print("   ✓ Backward compatible: Plain files still work")
        print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
