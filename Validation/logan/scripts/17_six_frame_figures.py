#!/usr/bin/env python3
"""Figures for the six-frame experiment (16_six_frame_summary.py output).

  17_six_frame_figures.py --summary qc/six_frame --out qc/figures_six_frame
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FEAT = {"aa_len": "ORF length", "aa_entropy": "AA entropy", "three_di_entropy": "3Di entropy",
        "twelve_state_entropy": "12-state entropy", "mi_3di_12st": "3Di–12st MI"}
FRAME = {"s1": "same strand +1", "s2": "same strand +2", "a0": "opposite a0", "a1": "opposite a1",
         "a2": "opposite a2"}


def style():
    sns.set_theme(context="paper", style="ticks", font_scale=1.15, rc={
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": "white", "grid.color": GRID,
        "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False})


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s, out = Path(args.summary), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    # Fig 9: inheritance heatmap (Spearman, selected vs alternative frame).
    I = pd.read_csv(s / "inheritance.tsv", sep="\t")
    M = I.pivot(index="feature", columns="frame", values="spearman").loc[list(FEAT), list(FRAME)]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    sns.heatmap(M.rename(index=FEAT, columns=FRAME), annot=True, fmt=".2f", vmin=-0.5, vmax=0.5,
                cmap=sns.diverging_palette(250, 25, s=75, l=55, center="light", as_cmap=True),
                linewidths=2, linecolor="white", cbar_kws={"label": "Spearman ρ"}, ax=ax)
    ax.set_xlabel("alternative frame (relative to Prodigal's selected frame)")
    ax.set_ylabel("")
    ax.set_title("Cross-frame inheritance: correlation between the selected frame and each alternative",
                 loc="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig9_six_frame_inheritance")

    # Fig 10: discrimination, genes with all five alternatives present (chance 1/6, 1/3, 1/2).
    D = pd.read_csv(s / "discrimination.tsv", sep="\t")
    d = D[D.scope == "genes with all 5 alternatives"].set_index("feature").loc[list(FEAT)]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(FEAT))
    w = 0.26
    for i, (top, lab) in enumerate(((1, "rank 1"), (2, "top 2"), (3, "top 3"))):
        b = ax.bar(x + (i - 1) * w, d[f"top{top}_rate"], width=w, color=SLOTS[i], label=lab)
        ax.bar_label(b, fmt="%.3f", fontsize=8.5, color=INK2, padding=1)
        ax.hlines(d[f"top{top}_chance"].iloc[0], x[0] - 1.5 * w, x[-1] + 1.5 * w, colors=SLOTS[i],
                  linestyles=":", lw=1.4)
    ax.set_xticks(x, [FEAT[f] for f in FEAT])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of genes")
    ax.legend(loc="lower right", ncol=3, fontsize=9)
    n = int(d.genes.iloc[0])
    ax.set_title(f"Six-frame discrimination: how often each feature ranks Prodigal's frame among the "
                 f"top 1/2/3 ({n:,} genes with all five alternatives; dotted = chance 1/6, 1/3, 1/2)",
                 loc="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, out, "fig10_six_frame_discrimination")

    # Fig 11: delta_X = X_selected - median(X_alternatives).
    dl = pd.read_parquet(s / "delta_sample.parquet")
    dl = dl[dl.k == 5]
    feats = [f for f in FEAT if f != "aa_len"]
    fig, axes = plt.subplots(1, len(feats) + 1, figsize=(18, 4.2))
    for ax, f, col in zip(axes, ["aa_len"] + feats, SLOTS):
        v = dl[f"delta_{f}"].dropna()
        lo, hi = np.quantile(v, [0.005, 0.995])
        ax.hist(v[(v >= lo) & (v <= hi)], bins=120, color=col, alpha=0.9)   # drop tails; clipping piled them into edge bins
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.set_xlabel(f"Δ {FEAT[f]}" + (" (aa)" if f == "aa_len" else " (bits)"))
        ax.set_yticks([])
        ax.set_title(f"{(v > 0).mean():.1%} > 0", loc="left", fontsize=10)
    fig.suptitle("Δ = selected frame − median of the five alternatives (genes with all five; central 99% shown, fraction > 0 over all)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig11_six_frame_delta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
