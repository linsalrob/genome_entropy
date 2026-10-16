#!/usr/bin/env python3
"""Figures for the ~50 k-metagenome analysis (models, length bins, fragments, distributions).

Inputs:
  --models DIR   13_models.py output (models_overall / models_by_length / models_truncated)
  histograms     analysis/v1/hist/*.npz (10_feature_histograms.py), summed here
Outputs PNG + PDF to --out.

  15_analysis_figures.py --models qc/analysis_models --out qc/figures_analysis
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
_spec = importlib.util.spec_from_file_location("h", HERE / "10_feature_histograms.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SET_LABEL = {"length": "length", "length+aa": "+ AA entropy", "length+3di": "+ 3Di entropy",
             "length+12st": "+ 12-state entropy", "length+mi": "+ 3Di–12st MI",
             "length+3di+12st+mi": "+ 3Di, 12st, MI", "length+aa+3di+12st+mi": "+ all four"}
SET_ORDER = list(SET_LABEL)
CONTRASTS = ["vs all Prodigal-", "vs antisense shadows", "vs frameshift shadows", "vs intergenic"]
CLASS_LABEL = {"prodigal_complete": "Prodigal+ (complete gene)", "antisense_overlap": "antisense shadow",
               "frameshift_overlap": "frameshift shadow", "intergenic": "intergenic"}


def style():
    sns.set_theme(context="paper", style="ticks", font_scale=1.15, rc={
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": "white",
        "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2})


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_models(m: Path, out: Path):
    r = pd.read_csv(m / "models_overall.tsv", sep="\t")
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharey=True)
    for c, contrast in enumerate(CONTRASTS):
        d = r[r.contrast == contrast].set_index("features").loc[SET_ORDER]
        y = np.arange(len(SET_ORDER))
        for row, (metric, sd, lab) in enumerate((("avg_precision", "ap_fold_sd", "average precision"),
                                                 ("roc_auc", "auc_fold_sd", "ROC AUC"))):
            ax = axes[row, c]
            ax.barh(y, d[metric], xerr=d[sd], color=[SLOTS[0]] + [SLOTS[1]] * 6, height=0.65,
                    error_kw={"ecolor": INK2, "elinewidth": 1})
            base = d.loc["length", metric]
            ax.axvline(base, color=INK2, lw=1, ls="--")
            lo = max(0.0, min(d[metric]) - 0.08) if metric == "avg_precision" else max(0.9, min(d[metric]) - 0.01)
            ax.set_xlim(lo, 1.0)
            for yi, v in zip(y, d[metric]):
                ax.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
            ax.set_xlabel(lab)
            if row == 0:
                ax.set_title(f"Prodigal+ {contrast}", loc="left", fontsize=11)
            ax.set_yticks(y, [SET_LABEL[s] for s in SET_ORDER])
            ax.invert_yaxis()
    fig.suptitle("Complete ORFs, gradient-boosted models, 5-fold CV grouped by accession "
                 "(population-weighted; bars ± fold SD; dashed = length alone)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig5_models_beyond_length")


def sampled_counts() -> pd.DataFrame:
    """Sampled positives / negatives per contrast and length bin (complete ORFs)."""
    import pyarrow.parquet as pq
    sys.path.insert(0, str(HERE))
    import logan_common as lc
    files = sorted(glob.glob(str(lc.root() / "analysis" / lc.CACHE_VERSION / "sample" / "*.parquet")))
    d = pd.concat([pq.read_table(f, columns=["label", "trunc", "aa_len"]).to_pandas() for f in files])
    d = d[d.trunc == "complete"]
    edges = [30, 50, 75, 100, 150, 200, 300, 500, 10**9]
    d["length_bin"] = pd.cut(d.aa_len, edges, right=False).astype(str)
    neg = {"vs all Prodigal-": ["antisense_overlap", "frameshift_overlap", "intergenic", "marginal_overlap",
                                "same_frame_unmatched"], "vs intergenic": ["intergenic"]}
    rows = []
    for c, ns in neg.items():
        g = d.groupby("length_bin")
        for b, x in g:
            rows.append({"contrast": c, "length_bin": b, "pos": int((x.label == "prodigal_complete").sum()),
                         "neg": int(x.label.isin(ns).sum())})
    return pd.DataFrame(rows)


def fig_bylen(m: Path, out: Path, min_n: int = 200):
    r = pd.read_csv(m / "models_by_length.tsv", sep="\t")
    # Drop bins too thin to estimate: e.g. >= 300 aa intergenic ORFs number
    # only ~2,400 in the whole population, a handful in the sample.
    r = r.merge(sampled_counts(), on=["contrast", "length_bin"], how="left")
    r = r[(r.pos >= min_n) & (r.neg >= min_n)]
    sets = ["length", "length+aa", "length+3di", "length+mi", "length+aa+3di+12st+mi"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for ax, contrast in zip(axes, ["vs all Prodigal-", "vs intergenic"]):
        d = r[r.contrast == contrast]
        order = sorted(d.length_bin.unique(), key=lambda s: float(s.strip("[(").split(",")[0]))
        labels = [f"{int(float(s.strip('[(').split(',')[0]))}–" for s in order]
        for i, s in enumerate(sets):
            x = d[d.features == s].set_index("length_bin").reindex(order)
            ax.plot(range(len(order)), x.roc_auc, marker="o", ms=5, color=SLOTS[i], label=SET_LABEL[s])
        ax.set_xticks(range(len(order)), labels)
        ax.set_xlabel("ORF length bin (aa)")
        ax.set_ylabel("ROC AUC within bin")
        ax.set_title(f"Prodigal+ {contrast}", loc="left", fontsize=11)
        ax.axhline(0.5, color=INK2, lw=1, ls=":")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), fontsize=10)
    fig.suptitle(f"Within-length-bin AUC of out-of-fold model scores (complete ORFs; bins with "
                 f"< {min_n} sampled positives or negatives omitted)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig6_models_within_length")


def fig_trunc(m: Path, out: Path):
    r = pd.read_csv(m / "models_truncated.tsv", sep="\t")
    r["features"] = r.features.map({"length": "length alone", "length+aa+3di+12st+mi": "length + all four"})
    r["truncation"] = r.truncation.map({"open5": "open at 5′ end", "open3": "open at 3′ end"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, lab in zip(axes, ["roc_auc", "avg_precision"], ["ROC AUC", "average precision"]):
        sns.barplot(data=r, x="truncation", y=metric, hue="features", palette=[SLOTS[0], SLOTS[1]], ax=ax)
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.3f", fontsize=9, color=INK2)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel(lab)
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Contig-end fragments: Prodigal+ partial genes vs truncated Prodigal− "
                 "(model trained on complete ORFs)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig7_fragments")


def fig_distributions(out: Path):
    files = sorted(glob.glob(str(H.out_dir() / "*.npz")))
    tot = None
    for f in files:
        z = np.load(f)
        if tot is None:
            tot = {k: z[k].astype(np.int64) for k in H.FEATURES[1:]}
        else:
            for k in tot:
                tot[k] += z[k]
    shape = H.SHAPE + (-1,)
    bins = (np.arange(H.ENT_BINS) + 0.5) / 100
    lens = [(3, "100–149 aa"), (5, "200–299 aa")]
    feats = [("aa_entropy", "AA entropy"), ("three_di_entropy", "3Di entropy"),
             ("twelve_state_entropy", "12-state entropy"), ("mi_3di_12st", "3Di–12st MI")]
    fig, axes = plt.subplots(len(lens), len(feats), figsize=(19, 8.2))
    for r, (lb, ltxt) in enumerate(lens):
        for c, (f, ftxt) in enumerate(feats):
            ax = axes[r, c]
            A = tot[f].reshape(shape)
            for i, cl in enumerate(CLASS_LABEL):
                h = A[H.LABELS.index(cl), 0, lb].sum(axis=(0, 1)).astype(float)
                if h.sum() == 0:
                    continue
                dens = h / h.sum() / 0.01
                ax.plot(bins, dens, color=SLOTS[i], lw=1.8, label=f"{CLASS_LABEL[cl]} (n={int(h.sum()):,})")
            ax.set_xlim(0, 4.4 if f != "mi_3di_12st" else 2.0)
            ax.set_xlabel(f"{ftxt} (bits)")
            ax.set_ylabel("density" if c == 0 else "")
            if c == 0:
                ax.set_title(f"Complete ORFs, {ltxt}", loc="left", fontsize=11)
            if c == len(feats) - 1:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Feature distributions at fixed ORF length ({len(files)} shards, all ORFs, from histograms)",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig8_distributions_fixed_length")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    m = Path(args.models)
    fig_models(m, out)
    fig_bylen(m, out)
    fig_trunc(m, out)
    fig_distributions(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
