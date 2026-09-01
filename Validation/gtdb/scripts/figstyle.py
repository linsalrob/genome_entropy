#!/usr/bin/env python3
"""Shared pieces for the entropy figures: sample loading and the log2(k) lines.

Imported by 08_plot_entropy_scatter.py and 09_plot_density.py. Only what
both need lives here; each script keeps its own palette and layout.
"""
import math
import sys

import pandas as pd

# 3Di entropy cannot exceed log2(k) for an ORF encoded into k distinct
# states, and a large population of ORFs uses only two or three -- mostly D,
# the coil state. The sharp horizontal boundary in every density view of
# this data is log2(3), not a biological threshold, which is section 5 of
# the report. Drawing the constants makes that readable without prose.
CEILINGS = [
    (0.0, "log$_2$(1) = 0"),
    (1.0, "log$_2$(2) = 1"),
    (math.log2(3), "log$_2$(3) = 1.585"),
    (2.0, "log$_2$(4) = 2"),
]

SAMPLE_DIR = "/g/data/ob80/re3494/gtdb_entropy/figure_samples"
SAMPLE_BAC = f"{SAMPLE_DIR}/sample_bac.tsv.gz"
SAMPLE_ARC = f"{SAMPLE_DIR}/sample_arc.tsv.gz"

NUMERIC = ("protein_entropy", "three_di_entropy")


def default_sample(domain):
    return SAMPLE_ARC if domain == "arc" else SAMPLE_BAC


def load_sample(path, annotated_only=True, quiet=False):
    """Read a sample written by 08b_sample_for_figures.pbs.

    By default this keeps only ORFs from genomes in which some ORF matched a
    CDS. Roughly half of GTDB representatives have no CDS annotation at all,
    so every ORF in them is in_genbank=False whatever it is; including them
    inflates the unmatched class with rows that carry no information about
    whether an ORF is a gene, and makes the two classes look better
    separated than the evidence supports. Section 6 of the report removes
    the same confounder before it will interpret anything.

    The flag this filters on is the matcher proxy, so it drops a genome
    whose real CDS features all failed the strict match along with genomes
    that were never annotated: the surviving set is slightly conservative.
    12_genome_cds_counts.pbs answers annotation presence from the GenBank
    records and is what 10_missed_genes.py uses; the figures have not been
    rebuilt on it.

    annotated_only=False reproduces the unfiltered view. It is a diagnostic
    -- useful for showing what the confounder does -- and not something to
    publish.

    Also accepts the older headerless three-column form
    (in_genbank, protein_entropy, three_di_entropy) so a figure can still be
    regenerated from an archived sample; those samples predate the
    annotation flag and cannot be filtered. Malformed rows are counted and
    reported rather than coerced: a truncated final line from an interrupted
    sampler used to arrive as a row of NaNs.
    """
    head = pd.read_csv(path, sep="\t", nrows=0)
    if "three_di_entropy" in head.columns:
        df = pd.read_csv(path, sep="\t", dtype={"in_genbank": str,
                                                "domain": str})
    else:
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["in_genbank", *NUMERIC], dtype={"in_genbank": str})
        df["domain"] = "bac"
    n_raw = len(df)

    df = df[df.in_genbank.isin(["True", "False"])].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(NUMERIC))
    n_bad = n_raw - len(df)

    n_unfiltered = len(df)
    if annotated_only:
        # 11_genome_annotation_status.pbs renamed this column from
        # "annotated" to matcher_matched_a_cds, because that is what the
        # flag records: an ORF passed the strict CDS match. Samples written
        # before the rename carry genome_annotated and are still readable.
        flag = next((c for c in ("matcher_matched_a_cds", "genome_annotated")
                     if c in df.columns), None)
        if flag is None:
            sys.exit(f"ERROR: {path} carries no matcher_matched_a_cds column, "
                     "so genomes without annotation cannot be removed. Re-run "
                     "08b_sample_for_figures.pbs, or pass annotated_only=False "
                     "to draw the unfiltered diagnostic deliberately.")
        df = df[df[flag].astype(str) == "True"].copy()

    if not quiet:
        frac = (df.in_genbank == "True").mean() * 100 if len(df) else 0.0
        dropped = n_unfiltered - len(df)
        print(f"{path}: {len(df):,} usable rows"
              + (f", {n_bad:,} malformed dropped" if n_bad else "")
              + f" ({frac:.2f}% in_genbank True)")
        if annotated_only:
            print(f"  restricted to annotated genomes: {dropped:,} of "
                  f"{n_unfiltered:,} rows removed "
                  f"({dropped/max(n_unfiltered,1)*100:.1f}%) — every ORF in "
                  "them is False by construction")
        else:
            print("  UNFILTERED: includes genomes with no CDS annotation, in "
                  "which every ORF is False by construction")
    if not len(df):
        sys.exit(f"ERROR: no usable rows in {path}")
    return df


def log2_ceilings(ax, colour="#52514e", label=True, fontsize=7.5,
                  label_x=0.015, label_ha="left", plate=True, only=None):
    """Dotted horizontal reference lines at the 3Di entropy ceilings.

    Labels sit inside the axes on a translucent plate. Placing them at the
    right edge instead put them under the colourbars of the hexbin panels,
    where they were silently clipped to fragments like "585" and ") = 1".

    `only` restricts which ceilings are drawn, for panels where the full set
    would crowd the axis. Lines are drawn under the data (zorder 0.5 sits
    below scatter and hexbin marks but above the grid).
    """
    for value, text in CEILINGS:
        if only is not None and value not in only:
            continue
        lo, hi = ax.get_ylim()
        if not (lo <= value <= hi):
            continue
        ax.axhline(value, color=colour, linewidth=0.8, linestyle=(0, (1, 2.5)),
                   alpha=0.75, zorder=0.5)
        if label:
            ax.text(label_x, value, text, transform=ax.get_yaxis_transform(),
                    ha=label_ha, va="bottom", fontsize=fontsize, color=colour,
                    alpha=0.95, zorder=3,
                    bbox=(dict(facecolor="#fcfcfb", alpha=0.72, edgecolor="none",
                               boxstyle="round,pad=0.15") if plate else None))


def subsample(df, max_points, seed=0, quiet=False):
    """Cap row count for mark-per-point figures, keeping class proportions."""
    if not max_points or len(df) <= max_points:
        return df
    out = df.sample(max_points, random_state=seed)
    if not quiet:
        print(f"  scatter subsample: {len(out):,} of {len(df):,} rows "
              f"({(out.in_genbank == 'True').mean()*100:.2f}% True, "
              "class proportions preserved by random draw)")
    return out
