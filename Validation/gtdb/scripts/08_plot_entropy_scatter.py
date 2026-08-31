#!/usr/bin/env python3
"""Protein vs 3Di entropy, split by whether the ORF matches a GenBank CDS.

Four panels, deliberately:
  A  in_genbank=True only
  B  in_genbank=False only
  C  both, True drawn first  -> False sits on top
  D  both, False drawn first -> True sits on top

C and D contain identical data and differ only in draw order. Whatever
changes between them is an artefact of overplotting, not of biology, which
is the point of showing them side by side.

The full table is ~1.35 billion ORFs, so this plots a systematic sample:
every 300th row from 20 chunks spread across the completed range, which
spans many taxa rather than a few related genomes. The sample preserves the
population's True fraction (11.8% sampled vs 12.03% actual), so the density
difference between the classes in C and D is real and not a sampling choice.

Colours are the categorical blue/orange pair, checked for colour-vision
separation before use: OKLab dE 33.6 normal, 46.7 deuteranopia, 24.7
protanopia, 33.5 tritanopia -- all above the >=15 / >=8 thresholds.
"""
import argparse
import os
import textwrap
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Working directory for intermediate samples. Session-scratch on the
# machine this was run on; set SCRATCH in the environment, or edit, to
# point somewhere writable on yours.
SCRATCH = os.environ.get("GE_SCRATCH", "./work")

SAMPLE = os.path.join(SCRATCH, "plotdata/sample_clean.tsv")
OUTDIR = "/g/data/ob80/re3494/gtdb_entropy/figures"

TRUE_C  = "#2a78d6"   # categorical slot 1
FALSE_C = "#eb6834"   # categorical slot 2
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

# Identical across every panel: the panels are only comparable if the mark
# spec is.
S, ALPHA = 3, 0.18

# Matches the density figures. Only 0.036% of ORFs lie below 2.4 bits, so
# the wider range mostly showed blank axis.
XLIM, YLIM = (2.35, 4.35), (-0.15, 4.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--out", default="protein_vs_3di_entropy.png")
    ap.add_argument("--note", default="", help="extra line under the title")
    args = ap.parse_args()

    df = pd.read_csv(args.sample, sep="\t", header=None,
                     names=["in_genbank", "protein_entropy", "three_di_entropy"],
                     dtype={"in_genbank": str})
    n_raw = len(df)
    # The sampler was interrupted mid-write, leaving one truncated line.
    df = df[df["in_genbank"].isin(["True", "False"])]
    df = df.dropna(subset=["protein_entropy", "three_di_entropy"])
    for c in ("protein_entropy", "three_di_entropy"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    print(f"rows: {n_raw} read, {len(df)} usable ({n_raw - len(df)} malformed)")

    t = df[df.in_genbank == "True"]
    f = df[df.in_genbank == "False"]
    frac = len(t) / len(df) * 100
    print(f"  in_genbank True : {len(t):,} ({frac:.1f}%)")
    print(f"  in_genbank False: {len(f):,}")

    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    # (axis, title, [(data, colour, label), ...] in draw order)
    panels = [
        (axes[0, 0], "A  in_genbank = True only",
         [(t, TRUE_C, "True")]),
        (axes[0, 1], "B  in_genbank = False only",
         [(f, FALSE_C, "False")]),
        (axes[1, 0], "C  both — True drawn first, False on top",
         [(t, TRUE_C, "True"), (f, FALSE_C, "False")]),
        (axes[1, 1], "D  both — False drawn first, True on top",
         [(f, FALSE_C, "False"), (t, TRUE_C, "True")]),
    ]

    for ax, title, layers in panels:
        ax.set_facecolor(SURFACE)
        for data, colour, label in layers:
            ax.scatter(data.protein_entropy, data.three_di_entropy,
                       s=S, c=colour, alpha=ALPHA, linewidths=0,
                       label=label, rasterized=True)
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
        n = sum(len(d) for d, _, _ in layers)
        ax.text(0.03, 0.97, f"n = {n:,}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlim(XLIM); ax.set_ylim(YLIM)

    for ax in axes[1, :]:
        ax.set_xlabel("Protein entropy (bits)", fontsize=11, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("3Di entropy (bits)", fontsize=11, color=INK)

    # Full-opacity proxy handles: the marks themselves are alpha 0.18 and
    # would be invisible at legend size.
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                          color=c, label=l)
               for c, l in ((TRUE_C, "in_genbank = True (matched to a GenBank CDS)"),
                            (FALSE_C, "in_genbank = False (ORF call only)"))]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005), labelcolor=INK)

    fig.suptitle("Protein vs 3Di entropy of bacterial ORFs, by GenBank CDS support",
                 fontsize=14, color=INK, x=0.02, ha="left", y=0.985)
    sub = (f"Systematic sample of {len(df):,} ORFs from 20 chunks of GTDB r232 "
           f"bacterial representatives ({frac:.1f}% True). "
           + (args.note + " " if args.note else "")
           + "Panels C and D contain identical data and differ only in draw "
             "order, so any difference between them is an overplotting artefact.")
    # Wrapped rather than one long line: an unwrapped subtitle runs off the
    # right edge of the canvas and is silently clipped in the saved PNG.
    fig.text(0.02, 0.958, textwrap.fill(sub, 145), fontsize=9.5, color=MUTED,
             ha="left", va="top", linespacing=1.5)

    fig.tight_layout(rect=[0, 0.035, 1, 0.905])
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, args.out)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
