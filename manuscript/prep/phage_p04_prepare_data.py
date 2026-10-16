#!/usr/bin/env python3
"""Reduce the Gadi Inphared run and the GTDB length references to P04's data.

P04 compares 3Di entropy of phage ORFs with bacterial and archaeal ORFs. Sources
(all on NCI Gadi; the phage run is recorded in agent-handoffs
projects/genome_entropy_logan/inphared_gadi/):

  phage     /g/data/ob80/re3494/inphared_entropy/entropy_3di_results/phg_*.tsv.gz
            (per-ORF rows, genome_entropy 0.3.0, 36,025 genomes, 6,754,361 ORFs)
  GTDB      /g/data/ob80/re3494/gtdb_entropy/entropy_null/empirical/length_entropy_{bac,arc}.tsv
            (stage 33: per-length n, mean and quantiles over the full population;
             stratum 3 = matched, 2 = unmatched in CDS-bearing genomes)
  density   /g/data/ob80/re3494/inphared_entropy/density/density_{phg,bac,arc}.tsv
            (46_matched_density: full-population protein x 3Di counts, 0.005-bit
             bins over 0-4.4 bits; stratum 1 = in_genbank True)

  phage_p04_prepare_data.py --only phage   # PBS job: reads 6.75 M rows
  phage_p04_prepare_data.py --only copy    # small files only
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
GZ = {"method": "gzip", "mtime": 0}
IPH = Path("/g/data/ob80/re3494/inphared_entropy")
GD = Path("/g/data/ob80/re3494/gtdb_entropy")
LMAX = 999  # 1 aa bins only, as the GTDB empirical reference


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def note(p):
    print(f"  in   {p}  sha256={sha256(p)}")


def write(df, name):
    p = OUT / name
    df.to_csv(p, sep="\t", index=False, compression=GZ, float_format="%.6g")
    print(f"  out  {p.name}  rows={len(df)}  sha256={sha256(p)}")


def prep_phage():
    files = sorted((IPH / "entropy_3di_results").glob("phg_*.tsv.gz"))
    if len(files) != 10:
        sys.exit(f"expected 10 phage chunks, found {len(files)}")
    parts = []
    for p in files:
        note(p)
        parts.append(pd.read_csv(p, sep="\t", usecols=["aa_length", "in_genbank",
                                                       "three_di_entropy"],
                                 dtype={"in_genbank": str}))
    d = pd.concat(parts, ignore_index=True)
    print(f"  phage rows {len(d):,}; in_genbank values {sorted(d.in_genbank.unique())}")
    d = d[d.aa_length <= LMAX]
    d["cls"] = np.where(d.in_genbank == "True", "matched", "unmatched")
    g = d.groupby(["cls", "aa_length"]).three_di_entropy
    t = pd.DataFrame({"n": g.size(), "mean": g.mean(), "q25": g.quantile(0.25),
                      "median": g.median(), "q75": g.quantile(0.75)}).reset_index()
    t.insert(0, "dataset", "phage")
    write(t, "figP04_phage_length_3di.tsv.gz")


def prep_copy():
    rows = []
    for dom, tag in (("bacteria", "bac"), ("archaea", "arc")):
        p = GD / f"entropy_null/empirical/length_entropy_{tag}.tsv"
        note(p)
        e = pd.read_csv(p, sep="\t", usecols=["stratum", "alphabet", "length_lo", "length_hi",
                                              "n", "mean", "qp25", "qp5", "qp75"])
        e = e[(e.alphabet == "three_di") & (e.length_lo == e.length_hi)
              & e.stratum.isin([2, 3])]
        rows.append(pd.DataFrame(dict(
            dataset=dom, cls=np.where(e.stratum == 3, "matched", "unmatched"),
            aa_length=e.length_lo.astype(int), n=e.n, mean=e["mean"], q25=e.qp25,
            median=e.qp5, q75=e.qp75)))
    write(pd.concat(rows, ignore_index=True), "figP04_gtdb_length_3di.tsv.gz")

    dens = []
    for dom, tag in (("phage", "phg"), ("bacteria", "bac"), ("archaea", "arc")):
        p = IPH / f"density/density_{tag}.tsv"
        note(p)
        d = pd.read_csv(p, sep="\t", comment="#")
        d = d[d.stratum == 1].drop(columns="stratum")
        d.insert(0, "dataset", dom)
        dens.append(d)
    write(pd.concat(dens, ignore_index=True), "figP04_matched_density.tsv.gz")


PREP = {"phage": prep_phage, "copy": prep_copy}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", choices=PREP, default=list(PREP))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for k in a.only:
        print(k)
        PREP[k]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
