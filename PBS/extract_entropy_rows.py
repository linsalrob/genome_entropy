#!/usr/bin/env python3
"""Extract per-ORF entropy rows from genome_entropy JSON into one TSV.

Run inside pipeline_array.pbs, on node-local jobfs, before the JSON is
packed into a chunk archive. The point is that downstream analysis reads
these TSVs and never has to unpack a multi-gigabyte JSON archive: the JSON
keeps the sequences and encodings, the TSV keeps the numbers.

Schema 2.2.0: the top level is a list of records, one per sequence record
in the input, each with input_id and a features dict keyed by ORF id. ORF
ids are only unique within a record, so genome and input_id are both
carried on every row.
"""
import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

ENTROPY_FIELDS = (
    "dna_entropy",
    "protein_entropy",
    "three_di_entropy",
    "twelve_state_entropy",
    "three_di_twelve_state_mutual_information",
)

COLUMNS = (
    "chunk",
    "genome",
    "input_id",
    "orf_id",
    "start",
    "end",
    "strand",
    "aa_length",
    "in_genbank",
    *ENTROPY_FIELDS,
)


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        data = json.load(fh)
    return [data] if isinstance(data, dict) else data


def rows_for(path, chunk):
    genome = os.path.basename(path).split(".json")[0]
    for record in load(path):
        input_id = record.get("input_id", "")
        for orf_id, feat in record.get("features", {}).items():
            if not isinstance(feat, dict):
                continue
            loc = feat.get("location") or {}
            protein = feat.get("protein") or {}
            meta = feat.get("metadata") or {}
            entropy = feat.get("entropy") or {}
            row = {
                "chunk": chunk,
                "genome": genome,
                "input_id": input_id,
                "orf_id": orf_id,
                "start": loc.get("start"),
                "end": loc.get("end"),
                "strand": loc.get("strand"),
                "aa_length": protein.get("length"),
                "in_genbank": meta.get("in_genbank"),
            }
            for field in ENTROPY_FIELDS:
                value = entropy.get(field)
                # Legacy encoders and schema 2.1 write these as null or omit
                # them. Leave the cell empty rather than writing 0, which
                # would silently bias any downstream mean.
                row[field] = "" if value is None else value
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--failures",
                    help="write the label of every unparseable genome here, "
                         "one per line, so the caller can fold them into the "
                         "chunk's failure state")
    args = ap.parse_args()

    files = sorted(Path(args.input_dir).glob("*.json"))
    if not files:
        print(f"ERROR: no JSON under {args.input_dir}", file=sys.stderr)
        return 1

    n_rows = 0
    bad = []
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for path in files:
            try:
                # Buffer the genome so a file that fails partway through
                # cannot leave half its ORFs in an otherwise complete TSV.
                genome_rows = list(rows_for(path, args.chunk))
            except (json.JSONDecodeError, OSError, ValueError) as e:
                # Count and name it rather than dropping it silently; a
                # chunk that quietly lost genomes is worse than one that
                # reports the loss.
                print(f"WARNING: could not parse {path}: {e}", file=sys.stderr)
                bad.append(os.path.basename(str(path)).split(".json")[0])
                continue
            for row in genome_rows:
                writer.writerow(row)
                n_rows += 1

    if args.failures:
        with open(args.failures, "w") as fh:
            for label in bad:
                fh.write(f"{label}\n")

    print(f"chunk {args.chunk}: {len(files) - len(bad)} genomes, {n_rows} ORF rows"
          f"{f', {len(bad)} unparseable' if bad else ''}")

    # Any unreadable genome means this TSV is missing science the chunk was
    # supposed to produce. Exit non-zero so the caller cannot publish it as
    # a clean result; the caller decides whether to keep the partial output.
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
