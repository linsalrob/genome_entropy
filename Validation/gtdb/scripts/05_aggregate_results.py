#!/usr/bin/env python3
"""Aggregate per-chunk entropy TSVs into a per-genome summary.

Layout produced by 04_run_entropy.pbs, per chunk, on /g/data, with
bacteria and archaea in separate directories throughout:

    entropy_3di_results/bac/bac_NNN.tsv.gz    per-ORF entropy rows
    entropy_3di_results/bac/bac_NNN.tar.zst   all per-genome JSON
    entropy_3di_results/arc/arc_NNN.tsv.gz
    entropy_3di_results/arc/arc_NNN.tar.zst

One summary is written per domain, and the domain is also a column on
every row, so the two are never accidentally pooled. Compare them
deliberately, not by default: archaea and bacteria differ in genome size,
coding density and GC, so a mean taken across both is rarely the quantity
anyone wants.

Two files per chunk rather than two per genome, because gdata/ob80 has
roughly 520k inodes free and 200k genomes would otherwise want ~600k for
GenBank plus JSON alone.

This reads the TSVs, not the JSON archives. Measured on real output, the
TSV is about 42x smaller than the JSON it came from and holds every entropy
value, so there is no reason to unpack an archive to compute summaries. The
archives are there for sequences, 3Di strings and 12-state encodings.

Reading one archive directly, if you do need it:

    zstd -dc entropy_3di_results/bac/bac_000.tar.zst | tar -xO ./GCA_x.json

Writes summary_per_genome_<domain>.tsv. It does not re-emit a combined
per-ORF file:
that would be tens of GB duplicating the chunk TSVs, which already are the
per-ORF table and can be read directly or concatenated on demand.
"""
import argparse
import csv
import glob
import gzip
import math
import os
import sys
from collections import defaultdict

ENTROPY_FIELDS = (
    "dna_entropy",
    "protein_entropy",
    "three_di_entropy",
    "twelve_state_entropy",
    "three_di_twelve_state_mutual_information",
)

# Raw Shannon entropy in bits. Normalised entropy is deliberately not
# stored by genome_entropy and is derived here only for the alphabets whose
# sizes are fixed and documented in the schema.
ALPHABET_SIZES = {
    "dna_entropy": 4,
    "protein_entropy": 20,
    "three_di_entropy": 20,
    "twelve_state_entropy": 12,
}


class Accumulator:
    """Streaming mean/sd, so memory does not scale with ORF count."""

    __slots__ = ("n", "total", "sumsq", "lo", "hi")

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.sumsq = 0.0
        self.lo = None
        self.hi = None

    def add(self, x):
        self.n += 1
        self.total += x
        self.sumsq += x * x
        if self.lo is None or x < self.lo:
            self.lo = x
        if self.hi is None or x > self.hi:
            self.hi = x

    @property
    def mean(self):
        return self.total / self.n if self.n else None

    @property
    def sd(self):
        if self.n < 2:
            return None
        var = (self.sumsq - self.n * self.mean ** 2) / (self.n - 1)
        return math.sqrt(var) if var > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default="/g/data/ob80/re3494/gtdb_entropy/entropy_3di_results")
    ap.add_argument("--domain", required=True, choices=("bac", "arc"),
                    help="aggregate one domain at a time; bacteria and "
                         "archaea are kept separate end to end")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.output is None:
        args.output = f"summary_per_genome_{args.domain}.tsv"

    pattern = os.path.join(args.results_dir, args.domain, f"{args.domain}_*.tsv.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched {pattern} — has 04_run_entropy.pbs finished?",
              file=sys.stderr)
        return 1

    # genome -> metric -> Accumulator, kept twice: over all called ORFs,
    # and over only those matched to a GenBank CDS. Many GTDB
    # representatives are unannotated assemblies with no CDS at all, so
    # pooling the two would mix confirmed coding sequence with unvalidated
    # ORF calls.
    stats = defaultdict(lambda: defaultdict(Accumulator))
    stats_ing = defaultdict(lambda: defaultdict(Accumulator))
    in_genbank_counts = defaultdict(int)
    contigs = defaultdict(set)
    orf_counts = defaultdict(int)
    chunk_of = {}

    n_rows = 0
    n_bad_rows = 0
    bad_files = []

    for path in files:
        chunk = os.path.basename(path).split(".")[0]
        try:
            with gzip.open(path, "rt") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    genome = row.get("genome")
                    if not genome:
                        n_bad_rows += 1
                        continue
                    n_rows += 1
                    chunk_of[genome] = chunk
                    orf_counts[genome] += 1
                    if row.get("input_id"):
                        contigs[genome].add(row["input_id"])
                    is_ing = row.get("in_genbank") == "True"
                    if is_ing:
                        in_genbank_counts[genome] += 1
                    for metric in ENTROPY_FIELDS:
                        raw = row.get(metric, "")
                        if raw == "" or raw is None:
                            # Absent on 3Di-only models and schema 2.1.
                            # Skipping keeps it out of the mean instead of
                            # dragging it toward zero.
                            continue
                        try:
                            value = float(raw)
                        except ValueError:
                            n_bad_rows += 1
                            continue
                        stats[genome][metric].add(value)
                        if is_ing:
                            stats_ing[genome][metric].add(value)
        except (OSError, EOFError, csv.Error) as e:
            print(f"WARNING: could not read {path}: {e}", file=sys.stderr)
            bad_files.append(path)

    columns = ["genome", "domain", "chunk", "n_contigs", "n_orfs",
               "n_orfs_in_genbank", "frac_orfs_in_genbank", "genome_annotated"]
    for metric in ENTROPY_FIELDS:
        columns += [f"n_{metric}", f"mean_{metric}", f"sd_{metric}",
                    f"min_{metric}", f"max_{metric}"]
        if metric in ALPHABET_SIZES:
            columns.append(f"mean_normalised_{metric}")
        # Same statistic over the GenBank-confirmed subset only.
        columns.append(f"mean_{metric}_in_genbank")

    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for genome in sorted(orf_counts):
            row = {
                "genome": genome,
                "domain": args.domain,
                "chunk": chunk_of.get(genome, ""),
                "n_contigs": len(contigs.get(genome, ())),
                "n_orfs": orf_counts[genome],
            }
            n_ing = in_genbank_counts.get(genome, 0)
            row["n_orfs_in_genbank"] = n_ing
            row["frac_orfs_in_genbank"] = (
                f"{n_ing / orf_counts[genome]:.4f}" if orf_counts[genome] else "")
            # Distinguishes "assembly carries no CDS annotation at all" from
            # "annotated, but this ORF was not among the CDS".
            row["genome_annotated"] = "True" if n_ing else "False"
            for metric in ENTROPY_FIELDS:
                acc = stats[genome].get(metric)
                if acc and acc.n:
                    row[f"n_{metric}"] = acc.n
                    row[f"mean_{metric}"] = f"{acc.mean:.6f}"
                    row[f"sd_{metric}"] = "" if acc.sd is None else f"{acc.sd:.6f}"
                    row[f"min_{metric}"] = f"{acc.lo:.6f}"
                    row[f"max_{metric}"] = f"{acc.hi:.6f}"
                    if metric in ALPHABET_SIZES:
                        norm = acc.mean / math.log2(ALPHABET_SIZES[metric])
                        row[f"mean_normalised_{metric}"] = f"{norm:.6f}"
                else:
                    row[f"n_{metric}"] = 0
                acc_ing = stats_ing[genome].get(metric)
                if acc_ing and acc_ing.n:
                    row[f"mean_{metric}_in_genbank"] = f"{acc_ing.mean:.6f}"
            writer.writerow(row)

    print(f"domain           : {args.domain}")
    print(f"chunks read      : {len(files) - len(bad_files)} of {len(files)}")
    print(f"genomes          : {len(orf_counts)}")
    print(f"ORF rows         : {n_rows}")
    n_ing_total = sum(in_genbank_counts.values())
    n_annotated = sum(1 for g in orf_counts if in_genbank_counts.get(g))
    print(f"ORFs in GenBank  : {n_ing_total} "
          f"({n_ing_total / n_rows * 100:.1f}% of ORFs)" if n_rows else "")
    print(f"annotated genomes: {n_annotated} of {len(orf_counts)} "
          f"({n_annotated / len(orf_counts) * 100:.1f}%) carry any CDS annotation"
          if orf_counts else "")
    if n_bad_rows:
        print(f"unparseable rows : {n_bad_rows}", file=sys.stderr)
    if bad_files:
        print(f"unreadable chunks: {len(bad_files)}", file=sys.stderr)
        for p in bad_files:
            print(f"  {p}", file=sys.stderr)

    missing = [m for m in ENTROPY_FIELDS
               if not any(stats[g].get(m) and stats[g][m].n for g in stats)]
    if missing:
        print(f"NOTE: no values anywhere for {missing} — expected only if a "
              f"3Di-only model was used.", file=sys.stderr)

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
