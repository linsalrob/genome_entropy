#!/usr/bin/env python3
"""Density views of protein vs 3Di entropy, split by GenBank CDS support.

Writes two figures:

  protein_vs_3di_hexbin.png  counts per hexagonal bin, log-scaled
  protein_vs_3di_kde.png     smoothed density, per class and overlaid

Both exist because the scatter version cannot answer the question it
raises. With ~175k unmatched ORFs drawn over ~23k matched ones, whichever
class is drawn last wins, and the interior of the dense mass is a solid
block of colour whatever the order. Binning and smoothing encode how many
points are somewhere instead of stacking them, so neither depends on draw
order at all.

Colour does different work in the two figures. In the hexbin it is
sequential -- one hue per panel, light to dark, encoding count -- and the
hue also names the class, so each panel identifies itself. In the KDE it
is categorical again: two hues, one per class, distinguishing identity
while contour height carries density.

Counts are log-scaled because bin occupancy spans several orders of
magnitude; on a linear scale the dense ridge saturates and everything else
reads as empty.
"""
import argparse
import os
import textwrap
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Working directory for intermediate samples. Session-scratch on the
# machine this was run on; set SCRATCH in the environment, or edit, to
# point somewhere writable on yours.
SCRATCH = os.environ.get("GE_SCRATCH", "./work")

SAMPLE = os.path.join(SCRATCH, "plotdata/sample_clean.tsv")
OUTDIR = "/g/data/ob80/re3494/gtdb_entropy/figures"

TRUE_C, FALSE_C = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

# Single-hue sequential ramps, light to dark, starting at the chart surface
# so empty and near-empty bins recede rather than forming a coloured floor.
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "ge_blue", [SURFACE, "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
ORANGE_RAMP = LinearSegmentedColormap.from_list(
    "ge_orange", [SURFACE, "#fde3d5", "#f9bfa2", "#f28f63", "#eb6834", "#c04a1d", "#8c3313"])
GREY_RAMP = LinearSegmentedColormap.from_list(
    "ge_grey", [SURFACE, "#e1e0d9", "#c3c2b7", "#898781", "#52514e", "#0b0b0b"])

XLAB, YLAB = "Protein entropy (bits)", "3Di entropy (bits)"

# Only 0.036% of ORFs sit below 2.4 bits of protein entropy, so the full
# range spent most of the axis on empty space and compressed the structure
# into a narrow strip. Clipping there costs nothing visible and roughly
# triples the usable width.
XLIM, YLIM = (2.35, 4.35), (-0.15, 4.25)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def load(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["in_genbank", "protein_entropy", "three_di_entropy"],
                     dtype={"in_genbank": str})
    df = df[df.in_genbank.isin(["True", "False"])].copy()
    for c in ("protein_entropy", "three_di_entropy"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


def hexbin_figure(df, out_name, note=""):
    t = df[df.in_genbank == "True"]
    f = df[df.in_genbank == "False"]
    xlim, ylim = XLIM, YLIM

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    panels = [
        (axes[0], t,  BLUE_RAMP,   "A  in_genbank = True", len(t)),
        (axes[1], f,  ORANGE_RAMP, "B  in_genbank = False", len(f)),
        (axes[2], df, GREY_RAMP,   "C  all ORFs", len(df)),
    ]
    for ax, data, cmap, title, n in panels:
        style(ax)
        hb = ax.hexbin(data.protein_entropy, data.three_di_entropy,
                       gridsize=70, cmap=cmap, bins="log",
                       mincnt=1, linewidths=0, extent=(*xlim, *ylim))
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
        ax.text(0.03, 0.97, f"n = {n:,}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color=MUTED)
        ax.set_xlabel(XLAB, fontsize=11, color=INK)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("ORFs per bin (log scale)", fontsize=9, color=MUTED)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_visible(False)
    axes[0].set_ylabel(YLAB, fontsize=11, color=INK)

    fig.suptitle("Protein vs 3Di entropy: binned density of bacterial ORFs",
                 fontsize=14, color=INK, x=0.01, ha="left", y=0.99)
    sub = (f"Hexagonal binning of {len(df):,} sampled ORFs (GTDB r232 bacterial "
           "representatives). Colour encodes count, not identity, so the result "
           "does not depend on draw order. " + (note + " " if note else "")
           + "Each panel is scaled to its own counts; compare shapes between "
             "panels, not colours.")
    fig.text(0.01, 0.945, textwrap.fill(sub, 175), fontsize=9.5, color=MUTED,
             ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.855])
    out = os.path.join(OUTDIR, out_name)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def kde_figure(df, out_name, note="", seed=0):
    t = df[df.in_genbank == "True"]
    f = df[df.in_genbank == "False"]
    # KDE cost scales with sample size and the estimate is smooth, so cap
    # each class. Equal n is deliberate here: each density is normalised
    # within its own class, so this compares distribution SHAPE and is not
    # a statement about relative abundance (True is 12.03% of the data).
    n = min(len(t), len(f), 20000)
    ts = t.sample(n, random_state=seed)
    fs = f.sample(n, random_state=seed)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, data, cmap, title in (
            (axes[0], ts, BLUE_RAMP,   "A  in_genbank = True"),
            (axes[1], fs, ORANGE_RAMP, "B  in_genbank = False")):
        style(ax)
        sns.kdeplot(data=data, x="protein_entropy", y="three_di_entropy",
                    fill=True, levels=12, cmap=cmap, thresh=0.02, ax=ax)
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
        ax.text(0.03, 0.97, f"n = {n:,}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color=MUTED)
        ax.set_xlabel(XLAB, fontsize=11, color=INK)
        ax.set_ylabel("")
        ax.set_xlim(XLIM); ax.set_ylim(YLIM)

    # Overlaid contour lines rather than fills: outlines let both classes be
    # read where they overlap, which is exactly what the scatter could not do.
    ax = axes[2]
    style(ax)
    for data, colour in ((fs, FALSE_C), (ts, TRUE_C)):
        sns.kdeplot(data=data, x="protein_entropy", y="three_di_entropy",
                    levels=6, color=colour, linewidths=1.6, thresh=0.05, ax=ax)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_title("C  both, as contours", fontsize=12, color=INK, loc="left", pad=10)
    ax.set_xlabel(XLAB, fontsize=11, color=INK)
    ax.set_ylabel("")
    handles = [plt.Line2D([], [], color=c, linewidth=2, label=l)
               for c, l in ((TRUE_C, "True"), (FALSE_C, "False"))]
    leg = ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.94))
    for txt in leg.get_texts():
        txt.set_color(INK)

    axes[0].set_ylabel(YLAB, fontsize=11, color=INK)

    fig.suptitle("Protein vs 3Di entropy: smoothed density of bacterial ORFs",
                 fontsize=14, color=INK, x=0.01, ha="left", y=0.99)
    sub = (f"Gaussian KDE on {n:,} ORFs per class, each density normalised within "
           "its class. Panels compare distribution shape, not abundance. "
           + (note + " " if note else "")
           + "Contours in C outline both classes at once, so neither hides the "
             "other regardless of draw order.")
    fig.text(0.01, 0.945, textwrap.fill(sub, 175), fontsize=9.5, color=MUTED,
             ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.855])
    out = os.path.join(OUTDIR, out_name)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--hexbin-out", default="protein_vs_3di_hexbin.png")
    ap.add_argument("--kde-out", default="protein_vs_3di_kde.png")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    df = load(args.sample)
    print(f"{len(df):,} ORFs "
          f"({(df.in_genbank == 'True').sum():,} True, "
          f"{(df.in_genbank == 'False').sum():,} False)")
    sns.set_theme(style="ticks")
    hexbin_figure(df, args.hexbin_out, args.note)
    kde_figure(df, args.kde_out, args.note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
