#!/usr/bin/env python3
"""Group the Inphared genomes for homology-aware cross-validation: skani ANI clusters and NCBI genera.

1. Writes every genome in the Pharokka/phold GenBank chunks to one FASTA.
2. Runs `skani triangle --sparse --small-genomes -i` (every record against
   every other).
3. Clusters genomes single-linkage at ANI >= --ani over >= --af of the
   shorter genome (95 / 85: the MIUViG species-level vOTU standard).
4. Writes the grouping TSVs that 03_models.py --groups reads:
     groups_ani95.tsv   input_id, group = ANI cluster
     groups_genus.tsv   input_id, group = NCBI genus (09_ncbi_genus.py), or the
                        genome's ANI cluster when NCBI gives no genus. Genera that
                        share an ANI cluster are merged (connected components),
                        so no vOTU straddles two groups.
   plus cluster_summary.json and genome_nonACGT.tsv (input_id, count of
   non-ACGT bases such as IUPAC ambiguity codes; read by 12_all_genomes_tables.py).

  10_genome_groups.py --gbk-dir genbank_split --genus meta/ncbi_genus.tsv --out groups --threads 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gbk-dir", required=True, type=Path)
    ap.add_argument("--genus", required=True, type=Path, help="09_ncbi_genus.py output")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skani", default="skani")
    ap.add_argument("--ani", type=float, default=95.0)
    ap.add_argument("--af", type=float, default=85.0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    fa = a.out / "genomes.fa"
    lengths, nonacgt = {}, {}
    if not fa.exists():
        with open(fa.with_suffix(".fa.partial"), "w") as out:
            for g in sorted(a.gbk_dir.glob("*.gbk.gz")):
                with gzip.open(g, "rt") as fh:
                    for rec in SeqIO.parse(fh, "genbank"):
                        out.write(f">{rec.id}\n{str(rec.seq)}\n")
        fa.with_suffix(".fa.partial").rename(fa)
    for rec in SeqIO.parse(fa, "fasta"):
        lengths[rec.id] = len(rec.seq)
        seq = str(rec.seq).upper()
        nonacgt[rec.id] = len(seq) - sum(seq.count(b) for b in "ACGT")
    pd.Series(nonacgt, name="nonACGT").to_csv(a.out / "genome_nonACGT.tsv", sep="\t", index_label="input_id")
    print(f"{len(lengths):,} genomes, {sum(lengths.values()):,} bp", flush=True)

    tsv = a.out / "skani_triangle.tsv"
    if not tsv.exists():
        with open(tsv.with_suffix(".partial"), "w") as fh:
            subprocess.run([a.skani, "triangle", "--sparse", "--small-genomes", "-i", str(fa), "-t", str(a.threads)],
                           stdout=fh, check=True)
        tsv.with_suffix(".partial").rename(tsv)
    sk = pd.read_csv(tsv, sep="\t")
    # skani names a record by its FASTA header; keep the accession (first word)
    for c in ("Ref_name", "Query_name"):
        sk[c] = sk[c].astype(str).str.split().str[0]
    sk["af_max"] = sk[["Align_fraction_ref", "Align_fraction_query"]].max(axis=1)
    edges = sk[(sk.ANI >= a.ani) & (sk.af_max >= a.af)]
    print(f"{len(sk):,} skani pairs; {len(edges):,} at ANI >= {a.ani} over >= {a.af}% of the shorter genome", flush=True)

    d = DSU()
    for x in lengths:
        d.find(x)
    for r, q in zip(edges.Ref_name, edges.Query_name):
        d.union(r, q)
    ani = pd.DataFrame({"input_id": list(lengths)})
    ani["group"] = ["ani:" + d.find(x) for x in ani.input_id]
    ani.to_csv(a.out / "groups_ani95.tsv", sep="\t", index=False)

    gen = pd.read_csv(a.genus, sep="\t", dtype=str).fillna("")
    gmap = dict(zip(gen.input_id, gen.genus))
    d2 = DSU()
    for x, cl in zip(ani.input_id, ani.group):
        g = gmap.get(x, "")
        d2.union("genus:" + g if g else cl, cl)
    genus = pd.DataFrame({"input_id": ani.input_id,
                          "group": [d2.find("genus:" + gmap[x] if gmap.get(x) else cl)
                                    for x, cl in zip(ani.input_id, ani.group)]})
    genus.to_csv(a.out / "groups_genus.tsv", sep="\t", index=False)

    def sizes(s):
        v = s.value_counts()
        return {"groups": int(len(v)), "singletons": int((v == 1).sum()), "largest": int(v.iloc[0]),
                "largest_name": str(v.index[0]), "genomes_in_groups_of_2plus": int(v[v > 1].sum())}

    summ = {"genomes": len(lengths), "skani_pairs": int(len(sk)), "edges": int(len(edges)),
            "ani_threshold": a.ani, "af_threshold": a.af,
            "with_ncbi_genus": int(sum(bool(gmap.get(x)) for x in lengths)),
            "ani95": sizes(ani.group), "genus": sizes(genus.group)}
    json.dump(summ, open(a.out / "cluster_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
