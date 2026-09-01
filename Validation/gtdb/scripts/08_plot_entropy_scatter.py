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

The full table is 2.57 billion bacterial ORFs, so this plots a systematic
sample: every 300th row of every chunk in the domain, written by
08b_sample_for_figures.pbs. Earlier versions of this figure sampled 20
chunks because that was all that existed while the run was going. Scatter
panels are then capped with --max-points, since a mark per point stops
carrying information long before eight million of them; the density figures
in 09 use the whole sample.

Dotted horizontal lines mark log2(k) for k distinct 3Di states. The hard
boundary at 1.585 is log2(3) -- a mechanical ceiling on ORFs that encode to
three states, not a biological threshold. Section 5 of the report.

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

import figstyle

OUTDIR = "/g/data/ob80/re3494/gtdb_entropy/figures"
DOMAIN_NAME = {"bac": "bacterial", "arc": "archaeal"}

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
    ap.add_argument("--domain", default="bac", choices=("bac", "arc"))
    ap.add_argument("--sample", default=None,
                    help="default: the sample for --domain")
    ap.add_argument("--out", default=None,
                    help="default: protein_vs_3di_entropy_<domain>.png")
    ap.add_argument("--max-points", type=int, default=250_000,
                    help="cap on marks drawn; 0 plots every sampled row")
    ap.add_argument("--include-unannotated", action="store_true",
                    help="diagnostic only: keep genomes with no CDS "
                         "annotation, in which every ORF is False by "
                         "construction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="", help="extra line under the title")
    args = ap.parse_args()

    sample = args.sample or figstyle.default_sample(args.domain)
    out_name = args.out or f"protein_vs_3di_entropy_{args.domain}.png"
    full = figstyle.load_sample(sample,
                                annotated_only=not args.include_unannotated)
    n_sampled = len(full)
    df = figstyle.subsample(full, args.max_points, args.seed)

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
        # Labelled on the right-hand panels only: on the left they collide
        # with the y-axis ticks.
        figstyle.log2_ceilings(ax, label=ax in (axes[0, 1], axes[1, 1]))

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
    handles.append(plt.Line2D([], [], color="#52514e", linewidth=1,
                              linestyle=(0, (1, 2.5)),
                              label="log$_2$(k) ceiling for k distinct 3Di states"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005), labelcolor=INK)

    domain_word = DOMAIN_NAME.get(args.domain, args.domain)
    fig.suptitle(f"Protein vs 3Di entropy of {domain_word} ORFs, "
                 "by GenBank CDS support",
                 fontsize=14, color=INK, x=0.02, ha="left", y=0.985)
    drawn = (f"{len(df):,} of them drawn" if len(df) < n_sampled
             else "all of them drawn")
    scope = ("ALL genomes, including unannotated — diagnostic view"
             if args.include_unannotated
             else "genomes with at least one annotated CDS only")
    sub = (f"Every 300th ORF of every GTDB r232 {domain_word} representative "
           f"chunk, {scope}: {n_sampled:,} sampled, {drawn} here "
           f"({frac:.1f}% True). "
           + (args.note + " " if args.note else "")
           + "Dotted lines are log$_2$(k) for k distinct 3Di states; the edge at "
             "1.585 is log$_2$(3), a ceiling on three-state ORFs rather than a "
             "biological threshold. "
           + "Panels C and D contain identical data and differ only in draw "
             "order, so any difference between them is an overplotting artefact.")
    # Wrapped rather than one long line: an unwrapped subtitle runs off the
    # right edge of the canvas and is silently clipped in the saved PNG.
    fig.text(0.02, 0.958, textwrap.fill(sub, 145), fontsize=9.5, color=MUTED,
             ha="left", va="top", linespacing=1.5)

    fig.tight_layout(rect=[0, 0.035, 1, 0.905])
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, out_name)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
