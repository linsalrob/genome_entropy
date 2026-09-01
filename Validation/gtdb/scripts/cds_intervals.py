#!/usr/bin/env python3
"""Write the coordinates of every GenBank CDS to a TSV.

Called by 13_cds_intervals.pbs. Reads the records with Biopython, which is
the same parser genome_entropy uses, so the contig ids here are the same
``record.id`` values that appear as ``input_id`` in the entropy TSVs and the
coordinates use the same zero-based half-open forward-strand convention as
``normalise_orf_interval``. Those two agreements are the whole point: a
hand-rolled location parser would risk a join key or an off-by-one that
silently changes the shadow/intergenic split.

Compound locations contribute one row per contiguous part. An origin-crossing
CDS on a circular chromosome is two parts, and testing overlap against each
is correct where testing against min..max would swallow the whole
chromosome.
"""
import argparse
import csv
import gzip
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from Bio import SeqIO

COLUMNS = ("genome", "contig", "start", "end", "strand", "cds_id")


def cds_id_for(feature, seq_id):
    """Prefer a stable identifier, matching genome_entropy's preference."""
    for key in ("protein_id", "locus_tag", "gene"):
        values = feature.qualifiers.get(key)
        if values:
            return values[0]
    return f"{seq_id}:{feature.location}"


def rows_for(path):
    """Yield one row per contiguous CDS part in one GenBank file."""
    genome = os.path.basename(os.path.dirname(path))
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as handle:
        for record in SeqIO.parse(handle, "genbank"):
            for feature in record.features:
                if feature.type != "CDS":
                    continue
                cds_id = cds_id_for(feature, record.id)
                for part in feature.location.parts:
                    # Fuzzy ends (<1, >4567) still expose integer start/end.
                    rows.append({
                        "genome": genome,
                        "contig": record.id,
                        "start": int(part.start),
                        "end": int(part.end),
                        "strand": "-" if part.strand == -1 else "+",
                        "cds_id": cds_id,
                    })
    return rows


def safe_rows_for(path):
    try:
        return path, rows_for(path), None
    except Exception as exc:                      # noqa: BLE001 - reported
        return path, [], f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paths", required=True,
                    help="file listing one GenBank path per line")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    with open(args.paths) as fh:
        paths = [line.strip() for line in fh if line.strip()]
    if not paths:
        print(f"ERROR: no paths in {args.paths}", file=sys.stderr)
        return 1

    n_rows = 0
    failed = []
    with open(args.output, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for path, rows, error in pool.map(safe_rows_for, paths,
                                              chunksize=8):
                if error is not None:
                    print(f"WARNING: {path}: {error}", file=sys.stderr)
                    failed.append(path)
                    continue
                writer.writerows(rows)
                n_rows += len(rows)

    print(f"{len(paths) - len(failed)} genomes, {n_rows} CDS parts")

    # A genome whose records could not be read has no CDS intervals, and an
    # empty interval set would make every ORF in it look intergenic. Fail
    # rather than hand that to the analysis.
    if failed:
        print(f"ERROR: {len(failed)} genome(s) unreadable, e.g. {failed[0]}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
