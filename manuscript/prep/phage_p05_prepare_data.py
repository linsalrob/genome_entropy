#!/usr/bin/env python3
"""Copy the Gadi short-ORF phage analysis (10-90 aa) into P05's data.

P05 extends the phage analysis below the 90-aa floor (#103) and sets it beside
the GMSC smORFs. Sources (NCI Gadi; the run is recorded in agent-handoffs
projects/genome_entropy_logan/inphared_gadi/short_orfs/, commit 90f4fc6):

  /g/data/ob80/re3494/inphared_entropy/short_orfs_10_90/analysis_code11/
      52_short_orf_analysis.py --exclude recoded_genomes.txt: the 35,921
      genomes whose Pyrodigal-gv calls use genetic code 11
  /g/data/ob80/re3494/inphared_entropy/short_orfs_10_90/analysis/
      the same analysis over all 36,025 genomes, the CDS length bins
      (53_cds_lengths.py) and matched ORFs by genetic code (56)

Every value is copied; nothing is re-estimated here.

  phage_p05_prepare_data.py
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
GZ = {"method": "gzip", "mtime": 0}
SRC = Path("/g/data/ob80/re3494/inphared_entropy/short_orfs_10_90")
C11 = SRC / "analysis_code11"
ALL = SRC / "analysis"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    print(f"  in   {p}  sha256={sha256(p)}")
    return pd.read_csv(p, sep="\t")


def write(df, name):
    p = OUT / name
    df.to_csv(p, sep="\t", index=False, compression=GZ, float_format="%.6g")
    print(f"  out  {p.name}  rows={len(df)}  sha256={sha256(p)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # dataset rows from 52 are e.g. "phage_bins27" / "gmsc_gmsc_bins": split the
    # source from the binning scheme.
    parts = []
    for path, genomes in ((C11 / "auc_by_bin.tsv", "code11"),
                          (ALL / "auc_by_bin.tsv", "all"),
                          (C11 / "gmsc_auc_by_bin.tsv", "gmsc")):
        d = read(path)
        d.insert(1, "genomes", genomes)
        d.insert(2, "bins", d.dataset.str.split("_", n=1).str[1])
        d["dataset"] = d.dataset.str.split("_", n=1).str[0]
        parts.append(d)
    write(pd.concat(parts, ignore_index=True), "figP05_auc_by_bin.tsv.gz")

    phage = read(C11 / "length_curve.tsv.gz")
    gmsc = read(C11 / "gmsc_length_curve.tsv.gz")
    gmsc = gmsc.rename(columns={"dataset": "cls"})
    gmsc.insert(0, "dataset", "gmsc")
    write(pd.concat([phage, gmsc], ignore_index=True), "figP05_length_curve.tsv.gz")

    write(read(C11 / "three_di_hist.tsv.gz"), "figP05_three_di_hist.tsv.gz")
    write(read(C11 / "class_by_bin.tsv"), "figP05_class_by_bin.tsv.gz")

    cds = read(ALL / "cds_length_bins.tsv")
    # 56 sorted its output with the header, which therefore landed last.
    code = read(ALL / "matched_by_code.tsv")
    names = list(code.columns)
    code = code.T.reset_index().T if names[0] != "bin" else code
    code.columns = ["bin", "code", "n_orfs", "n_matched"]
    code = code[code["bin"] != "bin"].astype({"n_orfs": int, "n_matched": int})
    code = code.sort_values(["bin", "code"]).reset_index(drop=True)
    write(cds, "figP05_cds_length_bins.tsv.gz")
    write(code, "figP05_matched_by_code.tsv.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
