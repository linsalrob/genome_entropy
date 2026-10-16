#!/usr/bin/env python3
"""Reduce the GMSC smORF entropy run to the plotting-ready tables in ../data/.

Runs on NCI Gadi against /g/data/ob80/re3494/Projects/smORFs/results/ (mirrored
on Teams at genome_entropy:General/smORFs/results/). The run itself is recorded
in Validation/smorfs/ (genome_entropy #105). Nothing here re-estimates a
published number: AUC and model tables are copied, the 2-D density is the
saved 400 x 400 count array, and the histograms are plain counts over the
per-smORF table.

  smorfs_prepare_data.py --results <smORFs/results> --only hist   # PBS job: reads 913 MB
  smorfs_prepare_data.py --results <smORFs/results> --only copy   # small files only
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
GZ = {"method": "gzip", "mtime": 0}
CLASSES = ("positive", "negative")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def note(path):
    print(f"  in   {path}  sha256={sha256(path)}")


def write(df, name):
    p = OUT / name
    df.to_csv(p, sep="\t", index=False, compression=GZ)
    print(f"  out  {p.name}  rows={len(df)}  sha256={sha256(p)}")


def right_closed_bin(x, width):
    """Index i with x in (i*width, (i+1)*width]; x == 0 goes to bin 0."""
    return np.maximum(np.ceil(np.round(x / width, 9)).astype(np.int64) - 1, 0)


# (column, bin width, upper edge). 3Di and 12-state use right-closed 0.005-bit
# bins so strings exactly at log2(k) sit immediately below the ceiling.
HISTS = (("three_di_entropy", 0.005, 4.4), ("twelve_state_entropy", 0.005, 3.6),
         ("aa_entropy", 0.01, 4.4), ("orf_nt_entropy", 0.005, 2.0))


def prep_hist(res):
    p = res / "smorf_entropy.tsv.gz"
    note(p)
    nbins = {c: int(round(hi / w)) for c, w, hi in HISTS}
    ent = {(cls, c): np.zeros(nbins[c], np.int64) for cls in CLASSES for c, _, _ in HISTS}
    length = {cls: np.zeros(100, np.int64) for cls in CLASSES}
    n = {cls: 0 for cls in CLASSES}
    for chunk in pd.read_csv(p, sep="\t", chunksize=2_000_000,
                             usecols=["dataset", "aa_length"] + [c for c, _, _ in HISTS]):
        for cls in CLASSES:
            g = chunk[chunk.dataset == cls]
            n[cls] += len(g)
            length[cls] += np.bincount(g.aa_length.to_numpy(), minlength=100)[:100]
            for c, w, hi in HISTS:
                i = right_closed_bin(g[c].to_numpy(), w)
                if (i >= nbins[c]).any():
                    sys.exit(f"{c} exceeds {hi} bits in {cls}")
                ent[(cls, c)] += np.bincount(i, minlength=nbins[c])
    print("  rows:", n)
    # Every length from the shortest observed in either class to 99, zeros
    # kept, so a stairs plot never stretches a bin across a gap.
    lo = min(int(np.flatnonzero(length[c])[0]) for c in CLASSES)
    write(pd.DataFrame([dict(dataset=cls, aa_length=L, n=int(length[cls][L]))
                        for cls in CLASSES for L in range(lo, 100)]),
          "figS01_length_hist.tsv.gz")
    rows = []
    for (cls, c), h in ent.items():
        w = dict((a, b) for a, b, _ in HISTS)[c]
        rows.append(pd.DataFrame(dict(dataset=cls, feature=c,
                                      bin_lo=np.round(np.arange(len(h)) * w, 4),
                                      bin_hi=np.round((np.arange(len(h)) + 1) * w, 4),
                                      n=h, n_dataset=n[cls])))
    write(pd.concat(rows, ignore_index=True), "figS01_entropy_hist.tsv.gz")


def prep_copy(res):
    # Tables copied verbatim (written by manuscript_stats.py, cluster.pbs,
    # cluster_models.py and stripe_states.py).
    copies = {
        "manuscript/table2_single_feature_auc.tsv": "figS01_single_feature_auc.tsv.gz",
        "clusters/cluster_split_models.tsv": "figS01_models_by_split.tsv.gz",
        "stripes/stripe_bands.tsv": "figSS02_stripe_bands.tsv.gz",
        "manuscript/table1b_length_bins.tsv": "tabS01_length_bins.tsv.gz",
        "clusters/cluster_stats.tsv": "tabS02_cluster_stats.tsv.gz",
        "clusters/cluster_largest.tsv": "tabS03_cluster_largest.tsv.gz",
        "smorf_entropy_summary.tsv": "tabS04_dataset_summary.tsv.gz",
    }
    for src, dst in copies.items():
        p = res / src
        note(p)
        d = pd.read_csv(p, sep="\t")
        if src == "smorf_entropy_summary.tsv":
            # pandas describe() output: two unnamed index columns.
            d = d.rename(columns={"Unnamed: 0": "feature", "Unnamed: 1": "statistic"})
        elif src == "manuscript/table1b_length_bins.tsv":
            d = d.rename(columns={"aa_length": "aa_length_bin"})
        write(d, dst)

    p = res / "density_aa_vs_3di_counts.npz"
    note(p)
    z = np.load(p)
    xe, ye = z["x_edges"], z["y_edges"]
    # density_aa_vs_3di.py saves h.T: rows index 3Di (y_edges), columns amino
    # acid (x_edges). Checked: the row marginal has mean 3Di 1.857 (positives).
    rows = []
    for cls in CLASSES:
        h = z[cls]
        r, k = np.nonzero(h)
        rows.append(pd.DataFrame(dict(dataset=cls, i_aa=k, j_3di=r,
                                      n=h[r, k].astype(np.int64))))
    d = pd.concat(rows, ignore_index=True)
    d.insert(1, "aa_lo", np.round(xe[d.i_aa], 4))
    d.insert(2, "three_di_lo", np.round(ye[d.j_3di], 4))
    write(d, "figSS01_density_aa_3di.tsv.gz")


PREP = {"hist": prep_hist, "copy": prep_copy}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path,
                    default=Path("/g/data/ob80/re3494/Projects/smORFs/results"))
    ap.add_argument("--only", nargs="+", choices=PREP, default=list(PREP))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for k in a.only:
        print(k)
        PREP[k](a.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
