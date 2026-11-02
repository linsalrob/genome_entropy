#!/usr/bin/env python3
"""Extract one genome for a gene-map case study: annotated CDS, every scored ORF, and the candidates' sequences.

Candidates are unannotated ORFs with out-of-fold p >= 0.8 in the searched
contexts (intergenic, antisense, frameshift); same_stop / same_frame ORFs are
pieces of an annotated gene and are not candidates.

Writes in --out: <genome>_cds.tsv, <genome>_orfs.tsv (every ORF: interval,
strand, length, 3Di entropy, p, label, context), <genome>_cand.{aa,3di}.fasta
for 15_build_query_db.py.

  11_case_study.py --work $PHAGE_WORK --genome NC_042013 --out $PHAGE_WORK/case_study
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SEARCHED = ["intergenic", "antisense", "frameshift"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--fs", default="intrinsic")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    G, T = a.genome, a.work / "tables"
    flt = [("input_id", "=", G)]
    cds = pq.read_table(T / "cds.parquet", filters=flt).to_pandas()
    orf = pq.read_table(T / "orfs.parquet", filters=flt).to_pandas()
    lab = pd.read_parquet(T / "labels.parquet", columns=["row", "in_genbank"])
    orf = orf.drop(columns="in_genbank").merge(lab, on="row")
    pr = pq.read_table(a.work / f"rank_orfs_relabelled_{a.fs}" / "all_orf_oof_predictions.parquet",
                       columns=["input_id", "orf_id", "probability_in_genbank"], filters=flt).to_pandas()
    orf = orf.merge(pr.rename(columns={"probability_in_genbank": "p"}), on=["input_id", "orf_id"])
    orf["candidate"] = (~orf.in_genbank) & (orf.p >= 0.8) & orf.context.isin(SEARCHED)
    cols = ["row", "orf_id", "g_start", "g_end", "strand", "protein_length", "three_di_entropy", "p", "in_genbank",
            "candidate", "context", "context_cds", "cds_overlap_frac"]
    orf[cols].sort_values("g_start").to_csv(a.out / f"{G}_orfs.tsv", sep="\t", index=False)
    cds[["cds_id", "g_start", "g_end", "strand", "aa_len", "product", "function", "phrog", "record_length"]] \
        .sort_values("g_start").to_csv(a.out / f"{G}_cds.tsv", sep="\t", index=False)
    cand = orf[orf.candidate]
    seqs = pq.read_table(T / "seqs.parquet", filters=[("row", "in", cand.row.tolist())]).to_pandas().set_index("row")
    for kind, col in (("aa", "aa"), ("3di", "three_di")):
        with open(a.out / f"{G}_cand.{kind}.fasta", "w") as fh:
            for r in cand.row:
                fh.write(f">{G}|{r}\n{seqs.loc[r, col]}\n")
    print(f"{G}: {len(cds)} CDS, {len(orf)} ORFs, {len(cand)} candidates "
          f"{cand.context.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
