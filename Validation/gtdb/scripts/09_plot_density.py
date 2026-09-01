#!/usr/bin/env python3
"""Density views of protein vs 3Di entropy, split by GenBank CDS support.

Writes four figures:

  protein_vs_3di_hexbin.png  counts per hexagonal bin, log-scaled
  protein_vs_3di_kde.png     smoothed density, per class and overlaid
  protein_vs_3di_joint.png   joint density with marginal distributions
  protein_vs_3di_domains.png bacteria against archaea

The joint figure is the one that names the effect. A marginal histogram of
3Di entropy shows the bimodality and the hard edge at log2(3) = 1.585
directly, where the joint density alone shows only that something happens
there; the protein-entropy marginal shows by comparison how little that axis
separates the two classes. Every panel with a 3Di axis carries dotted
log2(k) reference lines, so the boundary reads as the three-state ceiling it
is (section 5 of the report) rather than as an unexplained feature.

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

import figstyle

OUTDIR = "/g/data/ob80/re3494/gtdb_entropy/figures"
DOMAIN_NAME = {"bac": "bacterial", "arc": "archaeal"}

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


def hexbin_figure(df, out_name, domain="bac", note=""):
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
        figstyle.log2_ceilings(ax, label=(ax is axes[2]))
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
        ax.text(0.03, 0.97, f"n = {n:,}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color=MUTED)
        ax.set_xlabel(XLAB, fontsize=11, color=INK)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("ORFs per bin (log scale)", fontsize=9, color=MUTED)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_visible(False)
    axes[0].set_ylabel(YLAB, fontsize=11, color=INK)

    word = DOMAIN_NAME.get(domain, domain)
    fig.suptitle(f"Protein vs 3Di entropy: binned density of {word} ORFs",
                 fontsize=14, color=INK, x=0.01, ha="left", y=0.99)
    sub = (f"Hexagonal binning of {len(df):,} sampled ORFs (GTDB r232 {word} "
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


def kde_figure(df, out_name, domain="bac", note="", seed=0):
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
        figstyle.log2_ceilings(ax, label=False)

    # Overlaid contour lines rather than fills: outlines let both classes be
    # read where they overlap, which is exactly what the scatter could not do.
    ax = axes[2]
    style(ax)
    for data, colour in ((fs, FALSE_C), (ts, TRUE_C)):
        sns.kdeplot(data=data, x="protein_entropy", y="three_di_entropy",
                    levels=6, color=colour, linewidths=1.6, thresh=0.05, ax=ax)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    figstyle.log2_ceilings(ax)
    ax.set_title("C  both, as contours", fontsize=12, color=INK, loc="left", pad=10)
    ax.set_xlabel(XLAB, fontsize=11, color=INK)
    ax.set_ylabel("")
    handles = [plt.Line2D([], [], color=c, linewidth=2, label=l)
               for c, l in ((TRUE_C, "True"), (FALSE_C, "False"))]
    leg = ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.94))
    for txt in leg.get_texts():
        txt.set_color(INK)

    axes[0].set_ylabel(YLAB, fontsize=11, color=INK)

    word = DOMAIN_NAME.get(domain, domain)
    fig.suptitle(f"Protein vs 3Di entropy: smoothed density of {word} ORFs",
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


def joint_figure(df, out_name, domain="bac", note=""):
    """Joint density with marginal distributions on both axes.

    Report section 11, item 2. The central panel is a hexbin of everything;
    the marginals are per class and normalised within class, so they compare
    shape and not abundance -- in_genbank=True is 12% of the data and would
    otherwise be a flat line beside False.

    The 3Di marginal is the point of the figure: it puts the bimodality and
    the edge at log2(3) on an axis where they can be read off directly.
    """
    t = df[df.in_genbank == "True"]
    f = df[df.in_genbank == "False"]

    fig = plt.figure(figsize=(11, 10))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=(4.2, 1.25),
                          height_ratios=(1.25, 4.2),
                          wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    # The top-right cell is otherwise dead space, and it is the only place
    # the colourbar does not sit on top of either the data or the margins.
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_cb.axis("off")

    style(ax)
    hb = ax.hexbin(df.protein_entropy, df.three_di_entropy, gridsize=80,
                   cmap=GREY_RAMP, bins="log", mincnt=1, linewidths=0,
                   extent=(*XLIM, *YLIM))
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    figstyle.log2_ceilings(ax, label=False)
    ax.set_xlabel(XLAB, fontsize=11, color=INK)
    ax.set_ylabel(YLAB, fontsize=11, color=INK)

    # Protein entropy, top margin. Densities rather than counts: see above.
    style(ax_top)
    for data, colour in ((f, FALSE_C), (t, TRUE_C)):
        sns.kdeplot(x=data.protein_entropy, ax=ax_top, color=colour,
                    fill=True, alpha=0.22, linewidth=1.5, cut=0)
    ax_top.set_ylabel("density", fontsize=9, color=MUTED)
    ax_top.set_xlabel("")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_yticks([])

    # 3Di entropy, right margin. Histogram, not KDE: smoothing would round
    # off the very edge this figure exists to show.
    style(ax_right)
    bins = np.linspace(YLIM[0], YLIM[1], 220)
    for data, colour in ((f, FALSE_C), (t, TRUE_C)):
        ax_right.hist(data.three_di_entropy, bins=bins, orientation="horizontal",
                      color=colour, alpha=0.5, density=True, linewidth=0)
    figstyle.log2_ceilings(ax_right, label=True, label_x=0.97,
                           label_ha="right")
    ax_right.set_xlabel("density", fontsize=9, color=MUTED)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xticks([])

    cax = ax_cb.inset_axes([0.08, 0.30, 0.84, 0.11])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_label("ORFs per bin (log)", fontsize=8, color=MUTED, labelpad=2)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_visible(False)

    # Inside ax_top, on the left: the density curves occupy the right half of
    # that panel, so this is empty canvas. Anchoring above the axes put the
    # legend straight through the subtitle.
    handles = [plt.Line2D([], [], color=c, linewidth=6, alpha=0.6, label=l)
               for c, l in ((TRUE_C, "in_genbank = True"),
                            (FALSE_C, "in_genbank = False"))]
    handles.append(plt.Line2D([], [], color="#52514e", linewidth=1,
                              linestyle=(0, (1, 2.5)), label="log$_2$(k) ceilings"))
    leg = ax_top.legend(handles=handles, frameon=False, fontsize=9,
                        loc="upper left", labelcolor=INK,
                        handlelength=1.6, borderaxespad=0.4)
    for txt in leg.get_texts():
        txt.set_color(INK)

    word = DOMAIN_NAME.get(domain, domain)
    fig.suptitle(f"Joint and marginal distributions of {word} ORF entropy",
                 fontsize=14, color=INK, x=0.02, ha="left", y=0.985)
    sub = (f"Centre: all {len(df):,} sampled ORFs, binned. Margins: per class, "
           "each normalised within its own class, so they show shape and not "
           "abundance. " + (note + " " if note else "")
           + "The 3Di margin is why this figure exists -- the mass below "
             "log$_2$(3) = 1.585 is ORFs encoding to three or fewer states, whose "
             "entropy is capped there mechanically.")
    fig.text(0.02, 0.948, textwrap.fill(sub, 128), fontsize=9.5, color=MUTED,
             ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.97, top=0.855, bottom=0.07)
    out = os.path.join(OUTDIR, out_name)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def domain_figure(bac, arc, out_name, note=""):
    """Bacteria against archaea, report section 11 item 4.

    The two samples were taken at different strides (every 300th bacterial
    ORF, every 30th archaeal) because the domains differ in size by two
    orders of magnitude. Panel counts are therefore not comparable to each
    other; the marginals are normalised, and shape is what this figure is
    for.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor(SURFACE)

    for ax, data, cmap, title in (
            (axes[0, 0], bac, BLUE_RAMP,   f"A  bacteria ({len(bac):,} sampled ORFs)"),
            (axes[0, 1], arc, ORANGE_RAMP, f"B  archaea ({len(arc):,} sampled ORFs)")):
        style(ax)
        hb = ax.hexbin(data.protein_entropy, data.three_di_entropy, gridsize=70,
                       cmap=cmap, bins="log", mincnt=1, linewidths=0,
                       extent=(*XLIM, *YLIM))
        ax.set_xlim(XLIM); ax.set_ylim(YLIM)
        figstyle.log2_ceilings(ax, label=False)
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
        ax.set_xlabel(XLAB, fontsize=11, color=INK)
        ax.set_ylabel(YLAB, fontsize=11, color=INK)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("ORFs per bin (log)", fontsize=9, color=MUTED)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_visible(False)

    # Marginals, both domains overlaid, split by class so the comparison is
    # like for like: an unannotated archaeal genome and an unannotated
    # bacterial one are different populations from annotated ones.
    style(axes[1, 0])
    bins3 = np.linspace(YLIM[0], YLIM[1], 220)
    for data, colour, label in ((bac, BLUE_RAMP(0.75), "bacteria"),
                                (arc, ORANGE_RAMP(0.75), "archaea")):
        axes[1, 0].hist(data.three_di_entropy, bins=bins3, color=colour,
                        alpha=0.5, density=True, linewidth=0, label=label)
    # The ceilings are labelled here rather than on the hexbins: this panel
    # has the room, and it is where the reader is looking at the 3Di axis.
    for value, text in figstyle.CEILINGS:
        axes[1, 0].axvline(value, color="#52514e", linewidth=0.8,
                           linestyle=(0, (1, 2.5)), alpha=0.75, zorder=0.5)
        axes[1, 0].text(value, 0.985, text, transform=axes[1, 0].get_xaxis_transform(),
                        rotation=90, ha="right", va="top", fontsize=7.5,
                        color="#52514e", zorder=3,
                        bbox=dict(facecolor="#fcfcfb", alpha=0.72,
                                  edgecolor="none", boxstyle="round,pad=0.15"))
    axes[1, 0].set_xlim(YLIM)
    axes[1, 0].set_title("C  3Di entropy, both domains", fontsize=12, color=INK,
                         loc="left", pad=10)
    axes[1, 0].set_xlabel(YLAB, fontsize=11, color=INK)
    axes[1, 0].set_ylabel("density", fontsize=11, color=INK)
    # Upper left is where the log2(1)=0 spike is; the right side is empty.
    leg = axes[1, 0].legend(frameon=False, fontsize=10, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(INK)

    style(axes[1, 1])
    binsp = np.linspace(*XLIM, 200)
    for data, colour, label in ((bac, BLUE_RAMP(0.75), "bacteria"),
                                (arc, ORANGE_RAMP(0.75), "archaea")):
        axes[1, 1].hist(data.protein_entropy, bins=binsp, color=colour,
                        alpha=0.5, density=True, linewidth=0, label=label)
    axes[1, 1].set_xlim(XLIM)
    axes[1, 1].set_title("D  protein entropy, both domains", fontsize=12,
                         color=INK, loc="left", pad=10)
    axes[1, 1].set_xlabel(XLAB, fontsize=11, color=INK)
    axes[1, 1].set_ylabel("density", fontsize=11, color=INK)

    fig.suptitle("Entropy of bacterial and archaeal ORFs compared",
                 fontsize=14, color=INK, x=0.01, ha="left", y=0.99)
    sub = ("Sampled at different rates by design -- every 300th bacterial ORF, "
           "every 30th archaeal -- because the domains differ in size by two "
           "orders of magnitude. Panel counts are therefore not comparable "
           "between A and B; C and D are normalised densities, and shape is "
           "the comparison. " + (note + " " if note else "")
           + "The log$_2$(3) ceiling sits in the same place in both domains, as "
             "it must: it is a property of the encoding, not of the organism.")
    fig.text(0.01, 0.955, textwrap.fill(sub, 165), fontsize=9.5, color=MUTED,
             ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(OUTDIR, out_name)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bac", choices=("bac", "arc"))
    ap.add_argument("--sample", default=None,
                    help="default: the sample for --domain")
    ap.add_argument("--compare-sample", default=None,
                    help="second domain's sample; with it, also write the "
                         "bacteria-vs-archaea figure")
    ap.add_argument("--suffix", default=None,
                    help="output filename suffix; default is _<domain>")
    ap.add_argument("--include-unannotated", action="store_true",
                    help="diagnostic only: keep genomes with no CDS "
                         "annotation, in which every ORF is False by "
                         "construction")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    annotated_only = not args.include_unannotated
    suffix = args.suffix if args.suffix is not None else f"_{args.domain}"
    if args.include_unannotated:
        args.note = ("ALL genomes including unannotated: diagnostic view, not "
                     "a result." + (" " + args.note if args.note else ""))
    else:
        args.note = ("Restricted to genomes with at least one annotated CDS."
                     + (" " + args.note if args.note else ""))
    df = figstyle.load_sample(args.sample or figstyle.default_sample(args.domain),
                              annotated_only=annotated_only)
    print(f"{len(df):,} ORFs "
          f"({(df.in_genbank == 'True').sum():,} True, "
          f"{(df.in_genbank == 'False').sum():,} False)")
    sns.set_theme(style="ticks")
    hexbin_figure(df, f"protein_vs_3di_hexbin{suffix}.png", args.domain, args.note)
    kde_figure(df, f"protein_vs_3di_kde{suffix}.png", args.domain, args.note)
    joint_figure(df, f"protein_vs_3di_joint{suffix}.png", args.domain, args.note)

    if args.compare_sample:
        other = figstyle.load_sample(args.compare_sample,
                                     annotated_only=annotated_only)
        bac, arc = ((df, other) if args.domain == "bac" else (other, df))
        domain_figure(bac, arc, "protein_vs_3di_domains.png", args.note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
