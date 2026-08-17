#!/usr/bin/env python3
"""Split a gzipped GenBank file into fixed-record gzipped chunks."""

import argparse
import gzip
from pathlib import Path

from Bio import SeqIO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--records-per-chunk", type=int, default=250)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunk_index = 0
    records = []
    with gzip.open(args.input, "rt") as handle:
        for record in SeqIO.parse(handle, "genbank"):
            records.append(record)
            if len(records) == args.records_per_chunk:
                write_chunk(args.output_dir, chunk_index, records)
                chunk_index += 1
                records = []
    if records:
        write_chunk(args.output_dir, chunk_index, records)


def write_chunk(output_dir: Path, index: int, records: list) -> None:
    path = output_dir / f"phold_{index:04d}.gbk.gz"
    with gzip.open(path, "wt") as handle:
        SeqIO.write(records, handle, "genbank")
    print(f"{path}\t{len(records)}")


if __name__ == "__main__":
    main()
