#!/usr/bin/env python3
"""Reduce step: exact AUC / AP from stratified histograms, overall and within length bins.

Sums analysis/v1/hist/*.npz (10_feature_histograms.py) and, for each
contrast of complete ORFs

  Prodigal+ (complete gene)  vs  all complete Prodigal-
                             vs  complete intergenic
                             vs  complete antisense shadows
                             vs  complete same-strand frameshifts

reports, for every feature (ORF length, AA / 3Di / 12-state entropy,
3Di-12st MI):

  * ROC AUC and average precision over all ORFs of the contrast;
  * the same within each ORF-length bin;
  * the length-stratified AUC: the probability that a random
    positive/negative pair from the SAME length bin is ranked correctly,
    i.e. sum_b P_b N_b AUC_b / sum_b P_b N_b. Length can contribute nothing
    to it beyond the bin, so it measures each feature's information at
    fixed length (to bin resolution);
  * the same overall AUC/AP by biome and by tranche.

AUC is computed exactly from per-class histograms with ties (same 0.01-bit
or length bin) counted as one half. AP treats each bin as one threshold
step.

  11_summarise_histograms.py --out qc/analysis
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logan_common as lc  # noqa: E402

_spec = importlib.util.spec_from_file_location("h", HERE / "10_feature_histograms.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

CONTRASTS = {
    "vs all Prodigal-": ["antisense_overlap", "frameshift_overlap", "intergenic", "marginal_overlap",
                         "same_frame_unmatched", "frame_undefined_overlap"],
    "vs intergenic": ["intergenic"],
    "vs antisense shadows": ["antisense_overlap"],
    "vs frameshift shadows": ["frameshift_overlap"],
}
LEN_LABELS = [f"{a}-{b - 1}" if b < 10**9 else f">={a}" for a, b in zip(H.LEN_EDGES[:-1], H.LEN_EDGES[1:])]


def auc_ap(pos: np.ndarray, neg: np.ndarray):
    """Exact ROC AUC (ties = 1/2) and AP from score histograms (higher bin = higher score)."""
    # float64: at LOGAN scale P * N (~1.5e19) overflows int64 and gave
    # negative AUCs in the first preview.
    pos, neg = pos.astype(np.float64), neg.astype(np.float64)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return np.nan, np.nan, int(P), int(N)
    neg_below = np.cumsum(neg) - neg
    auc = (pos * (neg_below + 0.5 * neg)).sum() / (P * N)
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    keep = (pos[::-1] + neg[::-1]) > 0
    tp, fp = tp[keep], fp[keep]
    prec = tp / (tp + fp)
    rec = tp / P
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    return float(auc), ap, int(P), int(N)


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--out", required=True)
    args = ap_.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(H.out_dir() / "*.npz")))
    tot, rows = None, 0
    for f in files:
        z = np.load(f)
        rows += int(z["rows"])
        if tot is None:
            tot = {k: z[k].astype(np.int64) for k in H.FEATURES}
        else:
            for k in H.FEATURES:
                tot[k] += z[k]
    print(f"{len(files)} shards, {rows:,} ORFs")
    shape = H.SHAPE + (-1,)
    T = {k: v.reshape(shape) for k, v in tot.items()}  # label, trunc, len, biome, batch, bin
    L = {name: i for i, name in enumerate(H.LABELS)}
    C = 0  # trunc index for complete

    # Class counts by truncation, for the record.
    counts = T["aa_len"].sum(axis=(2, 3, 4, 5))
    cnt = pd.DataFrame(counts, index=H.LABELS, columns=H.TRUNC)
    cnt.to_csv(out / "class_by_truncation.tsv", sep="\t")

    res, bylen, bystrat = [], [], []
    for cname, negs in CONTRASTS.items():
        for f in H.FEATURES:
            A = T[f]
            pos_all = A[L["prodigal_complete"], C]                  # len, biome, batch, bin
            neg_all = sum(A[L[n], C] for n in negs)
            a, p, P, N = auc_ap(pos_all.sum(axis=(0, 1, 2)), neg_all.sum(axis=(0, 1, 2)))
            res.append({"contrast": cname, "feature": f, "positives": P, "negatives": N,
                        "roc_auc": a, "avg_precision": p})
            num = den = 0.0
            for b, lab in enumerate(LEN_LABELS):
                pb, nb = pos_all[b].sum(axis=(0, 1)), neg_all[b].sum(axis=(0, 1))
                a_b, p_b, P_b, N_b = auc_ap(pb, nb)
                bylen.append({"contrast": cname, "feature": f, "length_bin": lab, "positives": P_b,
                              "negatives": N_b, "roc_auc": a_b, "avg_precision": p_b})
                if P_b and N_b and f != "aa_len":
                    num += P_b * N_b * a_b
                    den += P_b * N_b
            if f != "aa_len":
                res[-1]["length_stratified_auc"] = num / den if den else np.nan
            for d, names, axis_keep in (("biome", H.BIOMES + ["unassigned"], 1), ("batch", H.BATCHES, 2)):
                for i, name in enumerate(names):
                    if axis_keep == 1:
                        pb, nb = pos_all[:, i].sum(axis=(0, 1)), neg_all[:, i].sum(axis=(0, 1))
                    else:
                        pb, nb = pos_all[:, :, i].sum(axis=(0, 1)), neg_all[:, :, i].sum(axis=(0, 1))
                    a_s, p_s, P_s, N_s = auc_ap(pb, nb)
                    if P_s:
                        bystrat.append({"contrast": cname, "feature": f, "by": d, "stratum": name,
                                        "positives": P_s, "negatives": N_s, "roc_auc": a_s,
                                        "avg_precision": p_s})
    R = pd.DataFrame(res)
    R.to_csv(out / "separation_overall.tsv", sep="\t", index=False)
    pd.DataFrame(bylen).to_csv(out / "separation_by_length.tsv", sep="\t", index=False)
    pd.DataFrame(bystrat).to_csv(out / "separation_by_biome_batch.tsv", sep="\t", index=False)
    json.dump({"shards": len(files), "orfs": rows}, open(out / "inputs.json", "w"), indent=1)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(cnt)
        print(R.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
