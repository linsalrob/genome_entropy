#!/usr/bin/env python3
"""Write selected per-ORF entropy and structural information to TSV.

The input is a gzipped unified genome_entropy JSON document. It is parsed one
record at a time to keep memory use bounded for large PHOLD datasets.
"""

import argparse
import csv
import gzip
import json
import math
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterator


HEADER = [
    "input_id",
    "orf_id",
    "dna_entropy",
    "protein_entropy",
    "three_di_entropy",
    "twelve_state_entropy",
    "three_di_twelve_state_mutual_information",
    "in_genbank",
]


def mutual_information(sequence_a: str, sequence_b: str) -> float:
    """Calculate empirical mutual information in bits for aligned strings."""
    if len(sequence_a) != len(sequence_b):
        raise ValueError("Mutual information requires aligned sequences of equal length")
    if not sequence_a:
        return 0.0

    total = len(sequence_a)
    counts_a = Counter(sequence_a)
    counts_b = Counter(sequence_b)
    joint_counts = Counter(zip(sequence_a, sequence_b))
    information = sum(
        (joint_count / total)
        * math.log2(
            (joint_count / total)
            / ((counts_a[state_a] / total) * (counts_b[state_b] / total))
        )
        for (state_a, state_b), joint_count in joint_counts.items()
    )
    return 0.0 if -1e-12 < information < 0 else information


def iter_records(input_path: Path) -> Iterator[dict]:
    """Yield record dictionaries from nested top-level JSON arrays.

    Pipeline output is commonly a list of batches, each containing unified
    records. This decoder consumes delimiters separately, so it never creates
    the full outer list in memory.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0

    with gzip.open(input_path, "rt", encoding="utf-8") as stream:
        while True:
            if position >= len(buffer):
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    return
                buffer = buffer[position:] + chunk
                position = 0

            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                continue

            if buffer[position] in "[, ]":
                position += 1
                continue

            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    raise
                buffer = buffer[position:] + chunk
                position = 0
                continue

            position = end
            if not isinstance(value, dict):
                raise ValueError("Expected unified record dictionaries in input JSON")
            yield value


def label_database(label_path: Path) -> tuple[tempfile.TemporaryDirectory[str], sqlite3.Connection]:
    """Create a disk-backed lookup of legacy PHOLD GenBank-match labels."""
    temporary_directory = tempfile.TemporaryDirectory()
    connection = sqlite3.connect(Path(temporary_directory.name) / "labels.sqlite")
    connection.execute(
        "CREATE TABLE labels (input_id, start, end, strand, frame, in_genbank, "
        "PRIMARY KEY (input_id, start, end, strand, frame))"
    )
    for record in iter_records(label_path):
        for feature in record.get("features", {}).values():
            location = feature["location"]
            connection.execute(
                "INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?, ?, ?)",
                (record["input_id"], location["start"], location["end"],
                 location["strand"], location["frame"],
                 feature.get("metadata", {}).get("in_genbank")),
            )
    connection.commit()
    return temporary_directory, connection


def write_tsv(input_path: Path, output_path: Path, label_path: Path | None) -> tuple[int, int]:
    """Extract one TSV row per ORF and return record and ORF counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    features_written = 0
    temporary_directory = None
    labels = None
    if label_path is not None:
        temporary_directory, labels = label_database(label_path)

    with output_path.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()

        for record in iter_records(input_path):
            records += 1
            input_id = record.get("input_id", "")
            for orf_id, feature in record.get("features", {}).items():
                entropy = feature.get("entropy", {})
                twelve_state = feature.get("twelve_state") or {}
                three_di = feature.get("three_di", {}).get("encoding", "")
                twelve_state_encoding = twelve_state.get("encoding")
                information = entropy.get(
                    "three_di_twelve_state_mutual_information"
                )
                if information is None and twelve_state_encoding is not None:
                    information = mutual_information(three_di, twelve_state_encoding)

                in_genbank = feature.get("metadata", {}).get("in_genbank", "")
                if labels is not None:
                    location = feature["location"]
                    matched = labels.execute(
                        "SELECT in_genbank FROM labels WHERE input_id = ? AND start = ? "
                        "AND end = ? AND strand = ? AND frame = ?",
                        (input_id, location["start"], location["end"],
                         location["strand"], location["frame"]),
                    ).fetchone()
                    in_genbank = "" if matched is None else bool(matched[0])
                writer.writerow(
                    {
                        "input_id": input_id,
                        "orf_id": orf_id,
                        "dna_entropy": entropy.get("dna_entropy", ""),
                        "protein_entropy": entropy.get("protein_entropy", ""),
                        "three_di_entropy": entropy.get("three_di_entropy", ""),
                        "twelve_state_entropy": entropy.get(
                            "twelve_state_entropy", ""
                        ),
                        "three_di_twelve_state_mutual_information": (
                            "" if information is None else information
                        ),
                        "in_genbank": in_genbank,
                    }
                )
                features_written += 1

    if labels is not None:
        labels.close()
    if temporary_directory is not None:
        temporary_directory.cleanup()
    return records, features_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Gzipped unified JSON input")
    parser.add_argument("output", type=Path, help="Output TSV path")
    parser.add_argument(
        "--labels", type=Path, help="Legacy PHOLD JSON supplying in_genbank labels"
    )
    args = parser.parse_args()

    records, features = write_tsv(args.input, args.output, args.labels)
    print(f"Wrote {features} ORFs from {records} records to {args.output}")


if __name__ == "__main__":
    main()
