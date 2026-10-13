#!/usr/bin/env python3
"""Per-biome and per-tranche separation, length-controlled (Phase B).

Sums analysis/v1/hist/*.npz (10_feature_histograms.py), whose strata include
length bin x biome x batch jointly, and for every contrast of complete ORFs
(as in 11_summarise_histograms.py) reports per biome and per batch:

  * pooled ROC AUC / AP over all lengths (reproduces separation_by_biome_batch.tsv);
  * ROC AUC / AP within each ORF-length bin;
  * the length-stratified AUC, sum_b P_b N_b AUC_b / sum_b P_b N_b, which
    removes the tautological ORF-length effect (Prodigal's own length rule)
    to bin resolution. Not defined for ORF length itself.

Writes biome_batch_separation.tsv (long format; length_bin = "all",
"stratified" or a bin label) and figS1_biome_batch_separation.{png,pdf}:
  A  within-bin AUC (100-149 aa) by biome, B  by tranche at 100-149 and
  150-199 aa (replication),
  C  3Di-entropy AUC within each length bin, by biome.

  15b_biome_batch_figure.py --out qc/figures_final --table qc/analysis_final
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load("h", "10_feature_histograms.py")
S = _load("s", "11_summarise_histograms.py")
F = _load("f", "15_analysis_figures.py")

FEAT_LABEL = {"three_di_entropy": "3Di entropy", "twelve_state_entropy": "12-state entropy",
              "mi_3di_12st": "3Di–12st MI", "aa_entropy": "AA entropy"}
BIOME_LABEL = {b: b.replace("_", " ") for b in H.BIOMES}


def summed_histograms() -> tuple[dict, int]:
    files = sorted(glob.glob(str(H.out_dir() / "*.npz")))
    tot = None
    for f in files:
        z = np.load(f)
        if tot is None:
            tot = {k: z[k].astype(np.int64) for k in H.FEATURES}
        else:
            for k in H.FEATURES:
                tot[k] += z[k]
    shape = H.SHAPE + (-1,)
    return {k: v.reshape(shape) for k, v in tot.items()}, len(files)


def table(T: dict) -> pd.DataFrame:
    L = {name: i for i, name in enumerate(H.LABELS)}
    rows = []
    for cname, negs in S.CONTRASTS.items():
        for f in H.FEATURES:
            A = T[f]
            pos_all = A[L["prodigal_complete"], 0]                 # len, biome, batch, bin
            neg_all = sum(A[L[n], 0] for n in negs)
            for by, names, ax in (("biome", H.BIOMES + ["unassigned"], 1), ("batch", H.BATCHES, 2)):
                for i, name in enumerate(names):
                    p = np.take(pos_all, i, axis=ax).sum(axis=1)  # len, bin
                    n = np.take(neg_all, i, axis=ax).sum(axis=1)
                    if p.sum() == 0:
                        continue
                    a, ap, P, N = S.auc_ap(p.sum(axis=0), n.sum(axis=0))
                    base = {"contrast": cname, "feature": f, "by": by, "stratum": name}
                    rows.append({**base, "length_bin": "all", "positives": P, "negatives": N,
                                 "roc_auc": a, "avg_precision": ap})
                    num = den = 0.0
                    for b, lab in enumerate(S.LEN_LABELS):
                        a_b, ap_b, P_b, N_b = S.auc_ap(p[b], n[b])
                        rows.append({**base, "length_bin": lab, "positives": P_b, "negatives": N_b,
                                     "roc_auc": a_b, "avg_precision": ap_b})
                        if P_b and N_b:
                            num += float(P_b) * N_b * a_b
                            den += float(P_b) * N_b
                    if f != "aa_len":
                        rows.append({**base, "length_bin": "stratified", "positives": P, "negatives": N,
                                     "roc_auc": num / den if den else np.nan, "avg_precision": np.nan})
    return pd.DataFrame(rows)


def figure(R: pd.DataFrame, out: Path, n_shards: int):
    d = R[(R.contrast == "vs all Prodigal-")]
    pooled = d[(d.length_bin == "all")]
    feats = ["aa_len", "aa_entropy", "twelve_state_entropy", "three_di_entropy", "mi_3di_12st"]
    lab = {"aa_len": "ORF length (within bin)", **FEAT_LABEL}
    col = {"aa_len": F.SLOTS[6], "aa_entropy": F.SLOTS[3], "twelve_state_entropy": F.SLOTS[2],
           "three_di_entropy": F.SLOTS[0], "mi_3di_12st": F.SLOTS[1]}
    w = 0.16
    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 1.05], hspace=0.42, wspace=0.15)

    # A: within-bin AUC at 100-149 aa by biome, ordered by 3Di
    axA = fig.add_subplot(gs[0, :])
    sb = d[(d.by == "biome") & (d.length_bin == "100-149")]
    order = sb[sb.feature == "three_di_entropy"].sort_values("roc_auc").stratum.tolist()
    x = np.arange(len(order))
    for j, f in enumerate(feats):
        v = sb[sb.feature == f].set_index("stratum").reindex(order).roc_auc
        axA.bar(x + (j - 2) * w, v, width=w, color=col[f], label=lab[f])
    npos = sb[sb.feature == "aa_len"].set_index("stratum").positives
    axA.set_xticks(x, [f"{BIOME_LABEL[b]}\n{npos[b] / 1e6:.1f} M" for b in order], fontsize=9.5)
    axA.set_ylim(0.5, 1.0)
    axA.axhline(0.5, color=F.INK2, lw=1, ls=":")
    axA.set_ylabel("ROC AUC within 100–149 aa")
    axA.set_title("A  By biome, ORFs of 100–149 aa: complete Prodigal+ vs all complete Prodigal− "
                  "(n = Prodigal+ ORFs in the bin)", loc="left", fontsize=11.5)
    axA.legend(ncol=5, fontsize=10, loc="upper left")

    # B: by tranche, two length bins
    axB = fig.add_subplot(gs[1, 0])
    bins = ["100-149", "150-199"]
    groups = [(bt, lb) for lb in bins for bt in H.BATCHES]
    xb = np.arange(len(groups))
    sbt = d[d.by == "batch"].set_index(["stratum", "length_bin", "feature"])
    for j, f in enumerate(feats):
        v = [sbt.loc[(bt, lb, f), "roc_auc"] for bt, lb in groups]
        axB.bar(xb + (j - 2) * w, v, width=w, color=col[f])
        if f in ("aa_entropy", "three_di_entropy"):
            for xi, vi in zip(xb, v):
                axB.text(xi + (j - 2) * w, vi + 0.006, f"{vi:.3f}", ha="center", va="bottom",
                         fontsize=7, color=F.INK2, rotation=90)
    nb = pooled[(pooled.by == "batch") & (pooled.feature == "aa_len")].set_index("stratum").positives
    axB.set_xticks(xb, [f"{bt}\n{lb.replace('-', '–')} aa" for bt, lb in groups], fontsize=9.5)
    axB.set_ylim(0.5, 1.05)
    axB.axhline(0.5, color=F.INK2, lw=1, ls=":")
    axB.set_ylabel("ROC AUC within bin")
    axB.set_title("B  By tranche (complete Prodigal+: " + ", ".join(
        f"{b} {nb[b] / 1e6:,.1f} M" for b in H.BATCHES) + ")", loc="left", fontsize=11.5)

    # C: 3Di within-bin AUC heatmap, biome x length bin
    axC = fig.add_subplot(gs[1, 1])
    hb = d[(d.by == "biome") & (d.feature == "three_di_entropy")
           & ~d.length_bin.isin(["all", "stratified"])]
    M = hb.pivot(index="stratum", columns="length_bin", values="roc_auc").reindex(
        index=order[::-1], columns=S.LEN_LABELS)
    sns.heatmap(M, ax=axC, cmap="viridis", vmin=0.5, vmax=1.0, annot=True, fmt=".2f",
                annot_kws={"fontsize": 6.5}, cbar_kws={"label": "ROC AUC within bin", "shrink": 0.8})
    axC.set_yticklabels([BIOME_LABEL[b] for b in order[::-1]], rotation=0, fontsize=9)
    axC.set_xlabel("ORF length bin (aa)")
    axC.set_ylabel("")
    axC.set_title("C  3Di entropy within each length bin, by biome", loc="left", fontsize=11.5)

    fig.suptitle(f"Length-controlled separation by biome and tranche ({n_shards} shards, all complete ORFs, "
                 "exact AUC from histograms)", x=0.01, ha="left", fontsize=13)
    F.save(fig, out, "figS1_biome_batch_separation")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="figure directory")
    ap.add_argument("--table", required=True, help="directory for biome_batch_separation.tsv")
    args = ap.parse_args()
    out, tdir = Path(args.out), Path(args.table)
    out.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    T, n = summed_histograms()
    R = table(T)
    R.to_csv(tdir / "biome_batch_separation.tsv", sep="\t", index=False)
    F.style()
    figure(R, out, n)
    s = R[(R.contrast == "vs all Prodigal-") & (R.length_bin == "stratified")]
    with pd.option_context("display.width", 200):
        print(s.pivot(index="stratum", columns="feature", values="roc_auc").round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
