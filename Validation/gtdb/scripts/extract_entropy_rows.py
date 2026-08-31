#!/usr/bin/env python3
"""Extract per-ORF entropy rows from genome_entropy JSON into one TSV.

Run inside 04_run_entropy.pbs, on node-local jobfs, before the JSON is
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
    "domain",
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
    # Appended, not inserted: 07_count_orfs.pbs and
    # 11_genome_annotation_status.pbs address this file by column number.
    #
    # Length of the contig this ORF was called on. Needed downstream because
    # negative-strand start/end index the reverse complement, so placing an
    # ORF on the forward genomic axis is impossible without it -- see
    # 10_missed_genes.py.
    "contig_length",
)


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        data = json.load(fh)
    return [data] if isinstance(data, dict) else data


def rows_for(path, chunk, domain):
    genome = os.path.basename(path).split(".json")[0]
    for record in load(path):
        input_id = record.get("input_id", "")
        contig_length = record.get("input_dna_length")
        for orf_id, feat in record.get("features", {}).items():
            if not isinstance(feat, dict):
                continue
            loc = feat.get("location") or {}
            protein = feat.get("protein") or {}
            meta = feat.get("metadata") or {}
            entropy = feat.get("entropy") or {}
            row = {
                "domain": domain,
                "chunk": chunk,
                "genome": genome,
                "input_id": input_id,
                "orf_id": orf_id,
                "start": loc.get("start"),
                "end": loc.get("end"),
                "strand": loc.get("strand"),
                "aa_length": protein.get("length"),
                "in_genbank": meta.get("in_genbank"),
                "contig_length": "" if contig_length is None else contig_length,
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
    ap.add_argument("--domain", required=True,
                    help="bac or arc; carried on every row so the two "
                         "domains stay distinguishable after any merge")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    files = sorted(Path(args.input_dir).glob("*.json"))
    if not files:
        print(f"ERROR: no JSON under {args.input_dir}", file=sys.stderr)
        return 1

    n_rows = 0
    n_bad = 0
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for path in files:
            try:
                for row in rows_for(path, args.chunk, args.domain):
                    writer.writerow(row)
                    n_rows += 1
            except (json.JSONDecodeError, OSError, ValueError) as e:
                # Count and name it rather than dropping it silently; a
                # chunk that quietly lost genomes is worse than one that
                # reports the loss.
                print(f"WARNING: could not parse {path}: {e}", file=sys.stderr)
                n_bad += 1

    print(f"chunk {args.domain}_{args.chunk}: {len(files) - n_bad} genomes, {n_rows} ORF rows"
          f"{f', {n_bad} unparseable' if n_bad else ''}")
    return 1 if n_bad and n_bad == len(files) else 0


if __name__ == "__main__":
    sys.exit(main())
