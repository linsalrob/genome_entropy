#!/usr/bin/env python3
"""Pilot figures: QC summary, Prodigal+/- entropy distributions, entropy vs length, ROC/PR.

Reads qc/<batch>/features.tsv.zst (05_pilot_qc.py) and each accession's
orf_prodigal / prodigal_genes tables; writes PNG + PDF figures and the small
tables behind them to --out.

WHAT "COMPLETE" MEANS HERE. LOGAN contigs are short (pilot median N50 260 bp)
and 94% of Prodigal+ ORFs match a *partial* Prodigal gene, so comparisons are
restricted to complete coding units, defined per class:

  Prodigal+  the matched Prodigal gene is partial=00 (start and stop on the
             contig) and the ORF ends in a stop. The stop-to-stop ORF may still
             be open upstream: Prodigal trims to the first start codon.
  Prodigal-  the ORF ends in a stop and has an in-frame stop upstream
             (five_open = 0), i.e. it is a whole stop-to-stop ORF.

Features are computed once in 05_pilot_qc.py from the cached AA / 3Di /
12-state strings with genome_entropy's own entropy functions.

  07_pilot_figures.py --batch pilot --out <dir> [--sample 4000000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

# Reference categorical palette (dataviz skill, light mode), fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CLASSES = ["prodigal_match", "antisense_overlap", "frameshift_overlap", "intergenic"]
CLASS_LABEL = {"prodigal_match": "Prodigal+ (matched gene)",
               "antisense_overlap": "Prodigal− antisense shadow",
               "frameshift_overlap": "Prodigal− same-strand frameshift",
               "intergenic": "Prodigal− intergenic"}
CLASS_COLOR = dict(zip(CLASSES, SLOTS))
FEATURES = [("aa_len", "ORF length (aa)"), ("aa_entropy", "AA entropy (bits)"),
            ("three_di_entropy", "3Di entropy (bits)"),
            ("twelve_state_entropy", "12-state entropy (bits)"),
            ("mi_3di_12st", "3Di–12st MI (bits)")]
FEATURE_COLOR = dict(zip([f for f, _ in FEATURES], SLOTS))
CEIL = {"three_di_entropy": (np.log2(20), "log₂20"), "twelve_state_entropy": (np.log2(12), "log₂12"),
        "aa_entropy": (np.log2(20), "log₂20")}


def style():
    sns.set_theme(context="paper", style="ticks", font_scale=1.15, rc={
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
        "figure.facecolor": "white", "grid.color": GRID, "axes.grid": True,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2})


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_sample(batch: str, n: int, seed: int) -> pd.DataFrame:
    path = lc.root() / "qc" / batch / "features.tsv.zst"
    total = 0
    with pd.read_csv(path, sep="\t", usecols=["accession"], chunksize=5_000_000,
                     compression={"method": "zstd"}) as it:
        for ch in it:
            total += len(ch)
    frac = min(1.0, n / total)
    rng = np.random.default_rng(seed)
    parts = []
    with pd.read_csv(path, sep="\t", chunksize=5_000_000, keep_default_na=False,
                     compression={"method": "zstd"}) as it:
        for ch in it:
            parts.append(ch[rng.random(len(ch)) < frac])
    s = pd.concat(parts, ignore_index=True)
    print(f"sampled {len(s):,} of {total:,} ORFs (fraction {frac:.4f})")
    return s, total


def gene_partial(accs) -> pd.DataFrame:
    """orf_id -> partial flag of its matched Prodigal gene, and per-accession partial stats."""
    rows, stats = [], []
    for a in accs:
        o = pd.read_csv(lc.work_dir(a) / "orf_prodigal.tsv.zst", sep="\t", keep_default_na=False,
                        usecols=["orf_id", "match_gene", "prodigal"])
        g = pd.read_csv(lc.work_dir(a) / "prodigal_genes.tsv.zst", sep="\t", keep_default_na=False,
                        dtype={"partial": str}, usecols=["gene_id", "partial"])
        m = o[o.prodigal == "Prodigal+"].merge(g, left_on="match_gene", right_on="gene_id")
        m["accession"] = a
        rows.append(m[["accession", "orf_id", "partial"]])
        stats.append({"accession": a, "genes": len(g),
                      "genes_complete": int((g.partial == "00").sum()),
                      "plus_complete_gene": int((m.partial == "00").sum()),
                      "plus": len(m)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(stats)


def fig_qc(pa: pd.DataFrame, pstats: pd.DataFrame, out: Path):
    d = pa.merge(pstats, on="accession")
    d["orfs_per_mbp"] = d.orfs / d.contig_bp * 1e6
    d["prodigal_plus_frac"] = d.prodigal_plus
    d["genes_complete_frac"] = d.genes_complete / d.genes
    order = d.groupby("biome").n50.median().sort_values().index.tolist()
    panels = [("n50", "Contig N50 (bp)", True), ("orfs_per_mbp", "ORFs ≥30 aa per Mbp", False),
              ("prodigal_plus_frac", "Fraction of ORFs Prodigal+", False),
              ("genes_complete_frac", "Fraction of Prodigal genes complete (partial=00)", False),
              ("truncated", "Fraction of ORFs open at a contig end", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(19, 5.2), sharey=True)
    for ax, (col, lab, logx) in zip(axes, panels):
        sns.stripplot(data=d, x=col, y="biome", order=order, ax=ax, color=SLOTS[0],
                      size=5, alpha=0.75, jitter=0.15)
        med = d.groupby("biome")[col].median().reindex(order)
        ax.scatter(med.values, range(len(order)), marker="|", s=260, color=INK, linewidths=2,
                   zorder=5, label="biome median")
        ax.set_xlabel(lab)
        ax.set_ylabel("")
        if logx:
            ax.set_xscale("log")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("LOGAN pilot: 91 metagenomes, one dot per accession (biomes ordered by median N50)",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig1_pilot_qc_by_biome")
    d.to_csv(out / "fig1_per_accession.tsv", sep="\t", index=False)


def fig_distributions(s: pd.DataFrame, out: Path):
    feats = FEATURES[1:]
    rows = [("All complete ORFs", s),
            ("Complete ORFs 100–300 aa (length partly controlled)", s[s.aa_len.between(100, 300)])]
    fig, axes = plt.subplots(len(rows), len(feats), figsize=(18, 8.4), sharey="col")
    for r, (title, d) in enumerate(rows):
        d = d[d.prodigal_class.isin(CLASSES)]
        for c, (f, lab) in enumerate(feats):
            ax = axes[r, c]
            sns.violinplot(data=d, x="prodigal_class", y=f, order=CLASSES, hue="prodigal_class",
                           hue_order=CLASSES, palette=CLASS_COLOR, cut=0, inner="quartile",
                           linewidth=0.8, density_norm="width", ax=ax, legend=False)
            if f in CEIL:
                ax.axhline(CEIL[f][0], color=INK2, lw=1, ls="--")
                ax.text(3.45, CEIL[f][0], CEIL[f][1], va="bottom", ha="right", fontsize=9, color=INK2)
            ax.set_xlabel("")
            ax.set_ylabel(lab)
            ax.set_xticks(range(4), ["P+", "anti", "frame", "inter"])
            if c == 0:
                ax.set_title(title, loc="left", fontsize=12, x=0, pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[k]) for k in CLASSES]
    fig.legend(handles, [CLASS_LABEL[k] for k in CLASSES],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save(fig, out, "fig2_entropy_by_prodigal_class")
    summ = (s[s.prodigal_class.isin(CLASSES)].groupby("prodigal_class")
            [[f for f, _ in FEATURES]].describe().T)
    summ.to_csv(out / "fig2_summary.tsv", sep="\t")


def fig_hexbin(s: pd.DataFrame, out: Path):
    groups = [("prodigal_match", "Prodigal+ (complete genes)"), ("intergenic", "Prodigal− intergenic"),
              ("antisense_overlap", "Prodigal− antisense shadows")]
    feats = [("three_di_entropy", "3Di entropy (bits)"), ("twelve_state_entropy", "12-state entropy (bits)")]
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(len(feats), len(groups), figsize=(16, 9.5), sharex=True, sharey="row")
    hbs = []
    for r, (f, flab) in enumerate(feats):
        for c, (cls, clab) in enumerate(groups):
            ax = axes[r, c]
            d = s[s.prodigal_class == cls]
            hb = ax.hexbin(d.aa_len, d[f], xscale="log", gridsize=70, mincnt=1,
                           cmap=sns.light_palette(SLOTS[0], as_cmap=True), linewidths=0)
            hbs.append(hb)
            ax.axhline(CEIL[f][0], color=INK2, lw=1, ls="--")
            ax.set_ylim(0, CEIL[f][0] + 0.25)
            ax.set_xlim(30, 3000)
            if r == 0:
                ax.set_title(f"{clab}  (n = {len(d):,})", loc="left", fontsize=11)
            ax.set_xlabel("ORF length (aa, log scale)" if r == len(feats) - 1 else "")
            ax.set_ylabel(flab if c == 0 else "")
            if r == 0 and c == len(groups) - 1:
                ax.text(2900, CEIL[f][0] + 0.03, CEIL[f][1] + " alphabet ceiling", ha="right",
                        va="bottom", fontsize=9, color=INK2)
    # One colour scale for every panel, so densities compare across classes.
    norm = LogNorm(1, max(h.get_array().max() for h in hbs))
    for h in hbs:
        h.set_norm(norm)
    cb = fig.colorbar(hbs[0], ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("ORFs per hexagon (log)")
    fig.suptitle("Structural-alphabet entropy against ORF length, complete ORFs (sampled)",
                 x=0.01, ha="left", fontsize=13)
    save(fig, out, "fig3_entropy_vs_length")


def fig_roc(s: pd.DataFrame, out: Path):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    contrasts = [("vs all Prodigal−", s),
                 ("vs Prodigal− intergenic only", s[s.prodigal_class.isin(["prodigal_match", "intergenic"])])]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5))
    rows = []
    for c, (cname, d) in enumerate(contrasts):
        y = (d.prodigal == "Prodigal+").to_numpy().astype(int)
        for f, lab in FEATURES:
            x = d[f].to_numpy(float)
            fpr, tpr, _ = roc_curve(y, x)
            prec, rec, _ = precision_recall_curve(y, x)
            auc, ap = roc_auc_score(y, x), average_precision_score(y, x)
            rows.append({"contrast": cname, "feature": f, "n": len(d), "positives": int(y.sum()),
                         "roc_auc": round(auc, 4), "avg_precision": round(ap, 4)})
            col = FEATURE_COLOR[f]
            axes[0, c].plot(fpr, tpr, color=col, label=f"{lab}  AUC {auc:.3f}")
            axes[1, c].plot(rec, prec, color=col, label=f"{lab}  AP {ap:.3f}")
        axes[0, c].plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
        base = y.mean()
        axes[1, c].axhline(base, color=INK2, lw=1, ls="--")
        axes[1, c].text(0.99, base, f"prevalence {base:.3f}", ha="right", va="bottom", fontsize=9, color=INK2)
        axes[0, c].set(title=f"ROC — Prodigal+ {cname}", xlabel="False positive rate",
                       ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
        axes[1, c].set(title=f"Precision–recall — Prodigal+ {cname}", xlabel="Recall",
                       ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
        for r in range(2):
            axes[r, c].title.set_fontsize(11)
            if r == 0:
                axes[r, c].legend(loc="lower right", fontsize=9)
            else:   # clear of the prevalence line, which sits at 0.2 in the intergenic panel
                axes[r, c].legend(loc="lower left", fontsize=9, bbox_to_anchor=(0, max(0.0, base) + 0.03))
    fig.suptitle("Single-feature separation of Prodigal+ ORFs, complete ORFs (sampled); "
                 "higher score = predicted Prodigal+", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig4_roc_pr")
    pd.DataFrame(rows).to_csv(out / "fig4_roc_pr_summary.tsv", sep="\t", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", default="pilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=102)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    s, total = load_sample(args.batch, args.sample, args.seed)
    accs = sorted(s.accession.unique())
    part, pstats = gene_partial(accs)
    s = s.merge(part, on=["accession", "orf_id"], how="left")
    plus = s.prodigal == "Prodigal+"
    s["complete"] = np.where(plus, (s.partial == "00") & (s.stop.astype(int) == 1),
                             (s.stop.astype(int) == 1) & (s.five_open.astype(int) == 0))
    c = s[s.complete].copy()
    counts = s.groupby(["prodigal_class", "complete"]).size().unstack(fill_value=0)
    counts.to_csv(out / "sample_class_counts.tsv", sep="\t")
    print(counts)

    pa = pd.read_csv(lc.root() / "qc" / args.batch / "per_accession.tsv", sep="\t")
    fig_qc(pa, pstats, out)
    fig_distributions(c, out)
    fig_hexbin(c, out)
    fig_roc(c, out)
    json.dump({"batch": args.batch, "orfs_total": int(total), "sampled": int(len(s)),
               "complete_in_sample": int(len(c)), "seed": args.seed,
               "complete_definition": "Prodigal+: matched gene partial=00 and ORF stop; "
                                      "Prodigal-: ORF stop and five_open=0"},
              open(out / "figure_inputs.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
