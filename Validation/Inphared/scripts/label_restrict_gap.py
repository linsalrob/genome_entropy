#!/usr/bin/env python3
"""Bacteria-minus-phage gap in mean 3Di entropy of matched ORFs, with phage
positives restricted by how much of the ORF their matched CDS covers.

Answers the manuscript question whether short CDS inside long stop-to-stop
ORFs explain why annotated phage ORFs sit lower in 3Di entropy than annotated
bacterial ORFs (Results, "Phage genes sit lower in 3Di entropy").

Phage: the 31,200-genome ranking set (orfs/labels/cds parquet tables from the
corrected relabelling). A positive's CDS is the same-strand CDS on the same
record that ends at the ORF's stop; coverage = CDS length / ORF length.
Bacteria: the unrestricted GTDB matched curve, manuscript/data/figP04_gtdb_length_3di.tsv.gz.
Gaps are n-weighted over each length bin, as in figP04.

  label_restrict_gap.py --tables <ranking tables dir> --figp04 <figP04_gtdb_length_3di.tsv.gz> --out <tsv>
"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tables", required=True)
ap.add_argument("--figp04", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = a.tables.rstrip("/") + "/"

o = pd.read_parquet(R + "orfs.parquet", columns=["input_id", "orf_id", "strand", "g_start", "g_end",
                                                 "protein_length", "three_di_entropy"])
lab = pd.read_parquet(R + "labels.parquet", columns=["input_id", "orf_id", "in_genbank"])
pos = o.merge(lab, on=["input_id", "orf_id"])
pos = pos[pos.in_genbank].copy()
c = pd.read_parquet(R + "cds.parquet", columns=["input_id", "g_start", "g_end", "strand"])
c["cl"] = c.g_end - c.g_start
# stop end: + strand -> g_end, - strand -> g_start (0-based half-open coordinates)
pos["stop"] = np.where(pos.strand == "+", pos.g_end, pos.g_start)
c["stop"] = np.where(c.strand == "+", c.g_end, c.g_start)
m = pos.merge(c[["input_id", "strand", "stop", "cl"]], on=["input_id", "strand", "stop"], how="left")
m = (m.groupby(["input_id", "orf_id"], as_index=False).agg(cl=("cl", "max"))
     .merge(pos, on=["input_id", "orf_id"]))
m["cov"] = m.cl / (m.g_end - m.g_start)   # NaN when no CDS ends at the ORF's stop

g = pd.read_csv(a.figp04, sep="\t")
g = g[(g.dataset == "bacteria") & (g.cls == "matched")]
rows = []
for lo, hi in [(90, 149), (150, 299), (300, 599), (600, 999)]:
    gb = g[g.aa_length.between(lo, hi)]
    b = (gb.n * gb["mean"]).sum() / gb.n.sum()
    x = m[m.protein_length.between(lo, hi)]
    r = dict(bin=f"{lo}-{hi}", bacteria_mean=b, n_all=len(x), gap_all=b - x.three_di_entropy.mean())
    for t in (0.5, 0.8):
        y = x[x["cov"] >= t]
        r[f"n_cov{int(t * 100)}"] = len(y)
        r[f"gap_cov{int(t * 100)}"] = b - y.three_di_entropy.mean()
    rows.append(r)
rows.append(dict(bin="all lengths", n_all=len(m), n_cov50=int((m["cov"] >= 0.5).sum()),
                 n_cov80=int((m["cov"] >= 0.8).sum())))
out = pd.DataFrame(rows)
out.to_csv(a.out, sep="\t", index=False, float_format="%.4f")
print(out.to_string())
print("positives without a CDS ending at the ORF stop:", int(m["cov"].isna().sum()))
