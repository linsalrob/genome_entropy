#!/usr/bin/env python3
"""Stream GenBank records into a gzipped FASTA file."""

import argparse
import gzip
from pathlib import Path

from Bio import SeqIO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with gzip.open(args.input, "rt") as input_handle, gzip.open(
        args.output, "wt"
    ) as output_handle:
        count = SeqIO.write(SeqIO.parse(input_handle, "genbank"), output_handle, "fasta")
    print(f"Wrote {count} FASTA records to {args.output}")


if __name__ == "__main__":
    main()
