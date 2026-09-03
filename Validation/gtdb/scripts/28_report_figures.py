#!/usr/bin/env python3
"""Figures for the scientific report's missed-gene sections.

Every panel reads a machine-readable artefact written by the stage that
computed it — the coverage TSV from 18_pilot_analysis.py, the ladder TSV from
22_rank_candidates.py, the per-ORF coincidence tables from
24_prodigal_overlap.py, the classified candidates from 25_functional_classes.py
and the neighbour table from 27_build_dossiers.py. Nothing here re-derives a
published number, because a plotting script that reimplements the analysis is a
second implementation that will drift from the first.

  28_report_figures.py --out-dir <figures dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Patch

GD = "/g/data/ob80/re3494/gtdb_entropy"
BAC = f"{GD}/missed_genes/full_bac"
ARC = f"{GD}/missed_genes/pilot_arc"

INK = "#2b2a28"
MUTED = "#52514e"
PLATE = "#fcfcfb"

ARM = {
    "candidate": "#1f6f8b",
    "shadow_hi": "#c46b4a",
    "clean_shadow": "#c46b4a",
    "annotated_cds": "#4b8b3b",
    "intergenic_lo": "#9a9793",
    "unannot_hi": "#7a5c9e",
}

CLASS_COLOUR = {
    "metabolism": "#4b8b3b",
    "mobile_element": "#c0392b",
    "transport": "#1f6f8b",
    "translation": "#7a5c9e",
    "regulation": "#d99a2b",
    "replication_repair": "#2f8f7f",
    "defence": "#8b5a2b",
    "cell_envelope": "#5c7fa3",
    # These three were originally greys, which made a candidate of that class
    # indistinguishable from the grey neighbour CDS around it -- two of the ten
    # exemplars were literally invisible in the first render. They are now
    # distinct hues; NEIGHBOUR_GREY below is reserved for deposited CDS.
    "uncharacterized": "#6b6763",
    "other_named": "#a8703f",
    "unclassified": "#8f8a84",
}

NEIGHBOUR_GREY = "#d8d5d0"
NEIGHBOUR_EDGE = "#b9b5af"


def style():
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 160,
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": PLATE, "savefig.facecolor": PLATE,
        "axes.grid": True, "grid.color": "#e6e3de", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def save(fig, out_dir, name):
    p = Path(out_dir) / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}  ({p.stat().st_size/1024:.0f} kB)")


# --------------------------------------------------------------------------
def fig_funnel(out_dir):
    """How 2.62 billion ORF calls become 545,793 candidates."""
    frames = {}
    for dom, d in (("bacteria", f"{GD}/missed_genes/bac"),
                   ("archaea", f"{GD}/missed_genes/arc")):
        f = pd.concat([pd.read_csv(p, sep="\t")
                       for p in sorted(Path(d).glob("*.stats.tsv"))],
                      ignore_index=True)
        frames[dom] = dict(
            orfs=f.orfs.sum(),
            in_annotated=f.orfs_annotated_genomes.sum(),
            unmatched=f.unmatched.sum(),
            hi=f.unmatched_hi.sum(),
            candidates=f.candidates.sum())
    steps = [("all ORF calls", "orfs"),
             ("in annotated genomes", "in_annotated"),
             ("unmatched by GenBank", "unmatched"),
             ("3Di entropy ≥ 2.5", "hi"),
             ("not a shadow — candidates", "candidates")]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4))
    for ax, (dom, v) in zip(axes, frames.items()):
        vals = [v[k] for _, k in steps]
        y = np.arange(len(steps))[::-1]
        cols = ["#c9c5bf"] * (len(steps) - 1) + [ARM["candidate"]]
        ax.barh(y, vals, color=cols, height=0.62)
        ax.set_xscale("log")
        ax.set_yticks(y, [lbl for lbl, _ in steps])
        ax.set_xlabel("ORFs (log scale)")
        ax.set_title(f"{dom}", loc="left", fontweight="bold")
        for yy, val in zip(y, vals):
            ax.text(val * 1.35, yy, f"{val:,}", va="center", fontsize=7.6,
                    color=INK)
        ax.set_xlim(right=max(vals) * 12)
    # Title states the whole chain, not a single ratio. An earlier version read
    # "two confounders remove 99.98% of the high-entropy pool", which conflated
    # the reduction from ALL ORF calls (99.98%) with the one the shadow test
    # actually performs on the high-3Di pool (94.5%).
    fig.suptitle("From 2.6 billion ORF calls to 545,793 candidates",
                 x=0.008, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, "candidate_funnel.png")


# --------------------------------------------------------------------------
def fig_coverage(out_dir):
    """Full-length mutual coverage per arm, and the implied real-gene share."""
    got = {}
    for dom, path in (("bacteria", f"{BAC}/analysis_v5_bac_coverage.tsv"),
                      ("archaea", f"{ARC}/analysis_v5_arc_coverage.tsv")):
        if Path(path).exists():
            got[dom] = pd.read_csv(path, sep="\t")
    if not got:
        print("  (no coverage TSVs; skipping candidate_coverage.png)")
        return

    fig, axes = plt.subplots(2, len(got), figsize=(5.3 * len(got), 6.0),
                             gridspec_kw={"height_ratios": [1.35, 1]})
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.T
    for j, (dom, d) in enumerate(got.items()):
        d = d.sort_values("database")
        x = np.arange(len(d))
        w = 0.26
        top = axes[0, j]
        for k, (col, lbl, c) in enumerate([
                ("candidate", "candidate", ARM["candidate"]),
                ("clean_shadow", "matched shadow (clean)", ARM["shadow_hi"]),
                ("annotated", "annotated CDS", ARM["annotated_cds"])]):
            top.bar(x + (k - 1) * w, d[col] * 100, width=w, label=lbl, color=c)
        top.set_xticks(x, d.database, rotation=12, ha="right")
        top.set_ylabel("% with qcov ≥ 0.8 and tcov ≥ 0.8")
        top.set_title(dom, loc="left", fontweight="bold")
        if j == 0:
            top.legend(loc="upper left", fontsize=7.8)

        bot = axes[1, j]
        est = d[d.database != "bfvd"]
        bot.bar(np.arange(len(est)), est.share * 100, width=0.5,
                color=ARM["candidate"],
                yerr=[(est.share - est.share_lo) * 100,
                      (est.share_hi - est.share) * 100],
                error_kw=dict(ecolor=MUTED, lw=1.0, capsize=3))
        bot.set_xticks(np.arange(len(est)), est.database, rotation=12,
                       ha="right")
        bot.set_ylabel("implied real-gene share (%)")
        bot.set_ylim(0, max(45, est.share_hi.max() * 100 * 1.25))
        for i, (sh, hi_) in enumerate(zip(est.share, est.share_hi)):
            bot.text(i, hi_ * 100 + 1.6, f"{sh*100:.1f}%", ha="center",
                     fontsize=7.8, color=INK)
        bot.text(0.99, 0.96, "BFVD excluded:\nannotated ceiling ~11%",
                 transform=bot.transAxes, ha="right", va="top", fontsize=7.2,
                 color=MUTED)
    # NOT "candidates resemble real genes": they do not. They sit BETWEEN their
    # matched shadows and annotated CDS, which is precisely what makes a
    # two-component mixture the right readout rather than a direct comparison.
    fig.suptitle("Candidates sit between their matched shadows and real genes",
                 x=0.008, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, out_dir, "candidate_coverage.png")


# --------------------------------------------------------------------------
def fig_ladder(out_dir):
    """Direct evidence, candidates against their matched clean shadows."""
    got = {}
    for dom, path in (("bacteria", f"{BAC}/ranked5_ladder.tsv"),
                      ("archaea", f"{ARC}/ranked5_ladder.tsv")):
        if Path(path).exists():
            d = pd.read_csv(path, sep="\t")
            d = d[d.comparator.str.startswith("CLEAN")]
            got[dom] = d[~d.criterion.str.startswith(("C1", "C2"))]
    if not got:
        print("  (no ladder TSVs; skipping evidence_ladder.png)")
        return

    fig, axes = plt.subplots(1, len(got), figsize=(5.6 * len(got), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (dom, d) in zip(axes, got.items()):
        # Grouped, NOT overlaid. Drawing the shadow bar on top of the candidate
        # bar makes it read as a subset of it; they are separate arms and the
        # point of the panel is the gap between them.
        y = np.arange(len(d))[::-1]
        scaled = d.shadows * d.n_candidate_arm / d.n_shadow_arm
        h = 0.36
        ax.barh(y + h / 2, d.candidates, height=h, color=ARM["candidate"],
                label="candidates")
        ax.barh(y - h / 2, scaled, height=h, color=ARM["shadow_hi"],
                label="matched shadows (scaled to the candidate arm)")
        ax.set_yticks(y, [c.split(" ", 1)[1].replace("C2 + ", "")
                          for c in d.criterion], fontsize=7.8)
        ax.set_xlabel("ORFs")
        ax.set_title(dom, loc="left", fontweight="bold")
        for yy, exc in zip(y, d.excess):
            ax.text(d.candidates.max() * 1.02, yy, f"excess {exc:,.0f}",
                    va="center", fontsize=7.6, color=INK)
        ax.set_xlim(right=d.candidates.max() * 1.42)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("The excess over matched shadows is the conservative floor",
                 x=0.008, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save(fig, out_dir, "evidence_ladder.png")


# --------------------------------------------------------------------------
def fig_prodigal(out_dir):
    """Agreement with an independent gene caller, by arm and by frame class."""
    got = {}
    for dom, path in (("bacteria", f"{BAC}/prodigal_bac4.coincidence.tsv.gz"),
                      ("archaea", f"{ARC}/prodigal_arc4.coincidence.tsv.gz")):
        if Path(path).exists():
            got[dom] = pd.read_csv(path, sep="\t", low_memory=False)
    if not got:
        print("  (no coincidence tables; skipping prodigal_validation.png)")
        return

    CLEAN = {"opposite strand", "same strand, frameshift"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7))

    ax = axes[0]
    arms = ["annotated_cds", "candidate", "shadow_hi", "intergenic_lo"]
    labels = ["annotated CDS\n(positive control)", "candidate",
              "matched shadow\n(clean)", "intergenic_lo\n(unoccupied space)"]
    w = 0.36
    for k, (dom, d) in enumerate(got.items()):
        vals = []
        for a in arms:
            sub = d[d.group == a]
            if a == "shadow_hi":
                sub = sub[sub.cds_frame_class.isin(CLEAN)]
            vals.append(sub.coincides.mean() * 100 if len(sub) else np.nan)
        ax.bar(np.arange(len(arms)) + (k - 0.5) * w, vals, width=w, label=dom,
               color=[ARM["candidate"], "#3f8fa8"][k])
        for i, v in enumerate(vals):
            ax.text(i + (k - 0.5) * w, v + 1.6, f"{v:.1f}", ha="center",
                    fontsize=7.4, color=INK)
    ax.set_xticks(np.arange(len(arms)), labels, fontsize=7.8)
    ax.set_ylabel("% coinciding with a Prodigal gene call")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8)
    ax.set_title("Independent agreement, by arm", loc="left",
                 fontweight="bold")
    # Placed over the low bars on the right, not over the candidate bars: at
    # (0.5, 0.52) the box hid the archaeal candidate's value label.
    ax.text(0.72, 0.82,
            "the two backgrounds agree to within\nhalf a point — the "
            "already-committed-call\nconfound is measurably absent",
            transform=ax.transAxes, ha="center", fontsize=7.4, color=MUTED,
            bbox=dict(facecolor=PLATE, edgecolor="#e0ddd8", boxstyle="round,pad=0.35"))

    ax = axes[1]
    order = ["same strand, same frame", "same strand, frameshift",
             "same strand, frame undefined", "opposite strand"]
    for k, (dom, d) in enumerate(got.items()):
        sh = d[d.group == "shadow_hi"]
        vals = [sh[sh.cds_frame_class == c].coincides.mean() * 100
                if (sh.cds_frame_class == c).any() else np.nan for c in order]
        ax.bar(np.arange(len(order)) + (k - 0.5) * w, vals, width=w, label=dom,
               color=[ARM["shadow_hi"], "#d68b6d"][k])
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i + (k - 0.5) * w, v + 1.6, f"{v:.1f}", ha="center",
                        fontsize=7.4, color=INK)
    ax.set_xticks(np.arange(len(order)),
                  ["same\nframe", "frameshift", "frame\nundefined",
                   "opposite\nstrand"], fontsize=7.8)
    ax.set_ylabel("% coinciding")
    ax.set_ylim(0, 108)
    ax.set_title("Why the frame class had to be fixed", loc="left",
                 fontweight="bold")
    ax.text(0.99, 0.74,
            "a same-frame shadow IS the annotated\nprotein, so Prodigal calls "
            "it — leaving\nthose in the comparator hides the signal",
            transform=ax.transAxes, ha="right", fontsize=7.4, color=MUTED,
            bbox=dict(facecolor=PLATE, edgecolor="#e0ddd8", boxstyle="round,pad=0.35"))
    fig.suptitle("A gene caller that consults neither GenBank nor any structure "
                 "database", x=0.008, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out_dir, "prodigal_validation.png")


# --------------------------------------------------------------------------
def fig_composition(out_dir):
    """What the strongly supported candidates actually are."""
    got = {}
    for dom, path in (("bacteria", f"{BAC}/func_bac_classified.tsv.gz"),
                      ("archaea", f"{ARC}/func_arc_classified.tsv.gz")):
        if Path(path).exists():
            d = pd.read_csv(path, sep="\t", low_memory=False,
                            usecols=["functional_class", "strict_rule"])
            got[dom] = d[d.strict_rule.astype(bool)].functional_class
    if not got:
        print("  (no classified tables; skipping functional_composition.png)")
        return

    order = ["metabolism", "other_named", "mobile_element", "transport",
             "uncharacterized", "translation", "regulation",
             "replication_repair", "defence", "cell_envelope", "unclassified"]
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    w = 0.38
    for k, (dom, s) in enumerate(got.items()):
        share = (s.value_counts(normalize=True) * 100).reindex(order).fillna(0)
        y = np.arange(len(order))[::-1] + (0.5 - k) * w
        ax.barh(y, share.values, height=w,
                color=[CLASS_COLOUR.get(c, "#bdbab5") for c in order],
                alpha=1.0 if k == 0 else 0.55,
                edgecolor=PLATE, linewidth=0.6)
        for yy, v in zip(y, share.values):
            if v >= 1.0:
                ax.text(v + 0.5, yy, f"{v:.1f}%", va="center", fontsize=7.2,
                        color=INK)
    ax.set_yticks(np.arange(len(order))[::-1],
                  [c.replace("_", " ") for c in order], fontsize=8.2)
    ax.set_xlabel("% of the strict-evidence set")
    # Describes what is plotted. The earlier title said mobile elements are the
    # most frequent FAMILIES, which is true of individual product names
    # (Resolvase YokA, n = 589) but is not what this chart shows -- it shows
    # classes, and metabolism is the largest.
    ax.set_title("Metabolic enzymes dominate the strongly supported set; "
                 "mobile elements are 12–14%", loc="left", fontweight="bold")
    ax.text(0.99, 0.06,
            "solid = bacteria (n = 23,540)\nfaded = archaea (n = 1,574)",
            transform=ax.transAxes, ha="right", fontsize=7.6, color=MUTED)
    fig.tight_layout()
    save(fig, out_dir, "functional_composition.png")


# --------------------------------------------------------------------------
def fig_exemplar_loci(out_dir):
    """Genomic context of the manuscript examples, coloured by class."""
    nb_path = Path(f"{GD}/missed_genes/dossiers/exemplar_neighbours.tsv")
    ex_path = Path(f"{GD}/missed_genes/manuscript_examples.tsv")
    if not (nb_path.exists() and ex_path.exists()):
        print("  (no exemplar tables; skipping exemplar_loci.png)")
        return
    nb = pd.read_csv(nb_path, sep="\t")
    ex = pd.read_csv(ex_path, sep="\t", low_memory=False)
    ex = ex.reset_index(drop=True)
    ex["example"] = ex.index + 1

    n = len(ex)
    fig, axes = plt.subplots(n, 1, figsize=(10.6, 1.42 * n), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, (_, row) in zip(axes, ex.iterrows()):
        g = nb[nb.example == row.example]
        lo = int(g.window_lo.iloc[0]) if len(g) else int(row.g_start) - 6000
        hi = int(g.window_hi.iloc[0]) if len(g) else int(row.g_end) + 6000
        cls = str(row.get("functional_class", "unclassified"))
        cc = CLASS_COLOUR.get(cls, "#bdbab5")

        for _, f in g.iterrows():
            s, e = max(f.nb_start, lo), min(f.nb_end, hi)
            if e <= s:
                continue
            y = 0.40 if f.nb_strand == "+" else -0.40
            ax.add_patch(FancyArrow(
                s if f.nb_strand == "+" else e, y,
                (e - s) * (1 if f.nb_strand == "+" else -1), 0,
                width=0.30, head_width=0.30,
                head_length=min((hi - lo) * 0.012, (e - s) * 0.55),
                length_includes_head=True, facecolor=NEIGHBOUR_GREY,
                edgecolor=NEIGHBOUR_EDGE, linewidth=0.5))

        # The candidate carries a heavy dark outline as well as its class
        # colour, so it stays identifiable even where the class hue is muted.
        cs, ce = int(row.g_start), int(row.g_end)
        y = 0.40 if row.strand == "+" else -0.40
        ax.add_patch(FancyArrow(
            cs if row.strand == "+" else ce, y,
            (ce - cs) * (1 if row.strand == "+" else -1), 0,
            width=0.46, head_width=0.46,
            head_length=min((hi - lo) * 0.014, (ce - cs) * 0.55),
            length_includes_head=True, facecolor=cc, edgecolor=INK,
            linewidth=1.3, zorder=3))

        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.95, 1.55)
        ax.set_yticks([])
        ax.grid(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([])
        prod = str(row.best_product)[:60]
        ax.text(0.002, 1.00,
                f"{row.genome} · {row.orf_id} · {int(row.aa_length)} aa · {prod}",
                transform=ax.transAxes, va="top", fontsize=8.0, color=INK,
                fontweight="bold")
        ax.text(0.002, 0.80, f"{row.slot}  ·  {cls.replace('_', ' ')}",
                transform=ax.transAxes, va="top", fontsize=7.3, color=MUTED)
        ax.text(0.998, 1.00, f"{(hi-lo)/1000:.0f} kb window",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
                color=MUTED)
        ax.axhline(0, color="#e8e5e0", linewidth=0.8, zorder=0)

    present = [c for c in CLASS_COLOUR
               if c in set(ex.get("functional_class", pd.Series(dtype=str)))]
    handles = [Patch(facecolor=CLASS_COLOUR[c], edgecolor=INK, linewidth=0.6,
                     label=c.replace("_", " ")) for c in present]
    handles.append(Patch(facecolor=NEIGHBOUR_GREY, edgecolor=NEIGHBOUR_EDGE,
                         label="deposited CDS"))
    axes[0].legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, 1.28), ncol=min(len(handles), 6),
                   fontsize=7.8)
    fig.suptitle("Manuscript examples in their genomic context — arrows show "
                 "strand; the coloured ORF is the candidate",
                 x=0.008, ha="left", fontsize=10.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, out_dir, "exemplar_loci.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=f"{GD}/figures")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    style()
    print(f"writing to {args.out_dir}")
    fig_funnel(args.out_dir)
    fig_coverage(args.out_dir)
    fig_ladder(args.out_dir)
    fig_prodigal(args.out_dir)
    fig_composition(args.out_dir)
    fig_exemplar_loci(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
