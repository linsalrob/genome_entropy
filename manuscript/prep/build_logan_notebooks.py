#!/usr/bin/env python3
"""Write the LOGAN figure notebooks (manuscript/notebooks/figL*.ipynb) from the cell sources below.

The notebooks are the deliverable; this script only keeps their shared style
and structure in one place. Each notebook reads plotting-ready data from
../data/ (made by logan_prepare_data.py) and writes ../images/<name>.{pdf,png}.
Outputs are not stored in the notebooks.

  python manuscript/prep/build_logan_notebooks.py          # writes manuscript/notebooks/figL*.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
PROV = json.load(open(Path(__file__).resolve().parent.parent / "data" / "logan_provenance.json"))

SETUP = r'''
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA, IMAGES = Path("../data"), Path("../images")
IMAGES.mkdir(exist_ok=True)
plt.rcParams["svg.hashsalt"] = "genome_entropy"   # deterministic SVG ids if SVG is ever written

# Colour slots (fixed order, colour-blind-checked categorical palette used across the LOGAN figures)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
sns.set_theme(context="paper", style="ticks", font_scale=1.15, rc={
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": "white", "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "font.family": "DejaVu Sans"})


def panel(ax, letter, x=-0.08, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    """Vector PDF (fixed metadata, so reruns are identical) and a 300-dpi PNG preview."""
    fig.savefig(IMAGES / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(IMAGES / f"{name}.png", dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print("wrote", IMAGES / f"{name}.pdf", IMAGES / f"{name}.png")
'''

FIGS: dict[str, dict] = {}

# ------------------------------------------------------------------ L01
FIGS["figL01_feature_distributions"] = dict(
    title="LOGAN: feature distributions at fixed ORF length",
    caption=(
        "Distributions of amino-acid (AA) entropy, predicted 3Di entropy, 12-state entropy and 3Di–12-state "
        "mutual information (MI) for complete ORFs of 100–149 aa (a–d) and 200–299 aa (e–h) in 50,009 LOGAN "
        "metagenomes. Classes: complete Prodigal genes (Prodigal+), and Prodigal− ORFs that are antisense "
        "shadows, same-strand frameshift shadows or intergenic. Densities from exact histograms over all "
        "89.99 billion ORFs (0.01-bit bins; no sampling); n per class in the legends."),
    data=["figL01_feature_distributions.tsv.gz"],
    code=r'''
d = pd.read_csv(DATA / "figL01_feature_distributions.tsv.gz", sep="\t")
CLASS = {"prodigal_complete": "Prodigal+ (complete gene)", "antisense_overlap": "antisense shadow",
         "frameshift_overlap": "frameshift shadow", "intergenic": "intergenic"}
FEATS = [("aa_entropy", "AA entropy"), ("three_di_entropy", "3Di entropy"),
         ("twelve_state_entropy", "12-state entropy"), ("mi_3di_12st", "3Di–12st MI")]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
letters = iter("abcdefgh")
for r, lb in enumerate(["100-149", "200-299"]):
    for c, (f, ftxt) in enumerate(FEATS):
        ax = axes[r, c]
        for i, (cl, ctxt) in enumerate(CLASS.items()):
            x = d[(d.length_bin_aa == lb) & (d.feature == f) & (d.orf_class == cl)].sort_values("bin_centre_bits")
            if x.empty:
                continue
            n = x["count"].sum()
            ax.plot(x.bin_centre_bits, x["count"] / n / 0.01, color=SLOTS[i], lw=1.8, label=f"{ctxt} (n={n:,})")
        ax.set_xlim(0, 2.0 if f == "mi_3di_12st" else 4.4)
        ax.set_xlabel(f"{ftxt} (bits)")
        ax.set_ylabel("density (per bit)" if c == 0 else "")
        if c == 0:
            ax.set_title(f"complete ORFs, {lb.replace('-', '–')} aa", loc="left", fontsize=11)
        if c == 3:
            ax.legend(fontsize=8, loc="upper right")
        panel(ax, next(letters))
fig.tight_layout()
save(fig, "figL01_feature_distributions")
''')

# ------------------------------------------------------------------ L02
FIGS["figL02_models_beyond_length"] = dict(
    title="LOGAN: what the structural features add beyond ORF length",
    caption=(
        "Average precision (a–d) and ROC AUC (e–h) of gradient-boosted classifiers separating complete "
        "Prodigal+ ORFs from complete Prodigal− ORFs, for four negative sets: all Prodigal−, antisense shadows, "
        "frameshift shadows and intergenic ORFs. Feature sets are nested on ORF length (dashed line: length "
        "alone). Five-fold cross-validation grouped by accession; metrics are population-weighted "
        "(sample weight = 1/sampling rate) over 9.5 million sampled complete ORFs from all 90 shards; error "
        "bars, ± SD across folds."),
    data=["figL02_models_overall.tsv.gz"],
    code=r'''
r = pd.read_csv(DATA / "figL02_models_overall.tsv.gz", sep="\t")
SET_LABEL = {"length": "length", "length+aa": "+ AA entropy", "length+3di": "+ 3Di entropy",
             "length+12st": "+ 12-state entropy", "length+mi": "+ 3Di–12st MI",
             "length+3di+12st+mi": "+ 3Di, 12st, MI", "length+aa+3di+12st+mi": "+ all four"}
ORDER = list(SET_LABEL)
CONTRASTS = [("vs all Prodigal-", "all Prodigal−"), ("vs antisense shadows", "antisense shadows"),
             ("vs frameshift shadows", "frameshift shadows"), ("vs intergenic", "intergenic")]
fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharey=True)
letters = iter("abcdefgh")
for row, (metric, sd, lab) in enumerate((("avg_precision", "ap_fold_sd", "average precision"),
                                         ("roc_auc", "auc_fold_sd", "ROC AUC"))):
    for c, (contrast, ctxt) in enumerate(CONTRASTS):
        ax = axes[row, c]
        d = r[r.contrast == contrast].set_index("features").loc[ORDER]
        y = np.arange(len(ORDER))
        ax.barh(y, d[metric], xerr=d[sd], color=[SLOTS[0]] + [SLOTS[1]] * 6, height=0.65,
                error_kw={"ecolor": INK2, "elinewidth": 1})
        ax.axvline(d.loc["length", metric], color=INK2, lw=1, ls="--")
        lo = max(0.0, d[metric].min() - 0.08) if metric == "avg_precision" else max(0.9, d[metric].min() - 0.01)
        ax.set_xlim(lo, 1.0)
        for yi, v in zip(y, d[metric]):
            ax.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
        ax.set_xlabel(lab)
        if row == 0:
            ax.set_title(f"Prodigal+ vs {ctxt}", loc="left", fontsize=11)
        ax.set_yticks(y, [SET_LABEL[s] for s in ORDER])
        ax.invert_yaxis()
        panel(ax, next(letters), x=-0.02 if c else -0.32)
fig.tight_layout()
save(fig, "figL02_models_beyond_length")
''')

# ------------------------------------------------------------------ L03
FIGS["figL03_models_within_length"] = dict(
    title="LOGAN: discrimination within ORF-length bins",
    caption=(
        "ROC AUC of out-of-fold model scores evaluated within ORF-length bins, for complete Prodigal+ ORFs "
        "against all complete Prodigal− ORFs (a) and against intergenic ORFs (b). Within a bin, length "
        "carries little information, so the length-only model falls towards 0.5 (dotted) from 75 aa upward "
        "while models including 3Di entropy remain high. Bins with fewer than 200 sampled positives or "
        "negatives are omitted. Same models and cross-validation as the beyond-length figure."),
    data=["figL03_models_by_length.tsv.gz"],
    code=r'''
r = pd.read_csv(DATA / "figL03_models_by_length.tsv.gz", sep="\t")
MIN_N = 200
r = r[(r.sampled_pos >= MIN_N) & (r.sampled_neg >= MIN_N)]
SETS = {"length": "length", "length+aa": "+ AA entropy", "length+3di": "+ 3Di entropy",
        "length+mi": "+ 3Di–12st MI", "length+aa+3di+12st+mi": "+ all four"}
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
for ax, (contrast, ctxt), letter in zip(axes, [("vs all Prodigal-", "all Prodigal−"),
                                                ("vs intergenic", "intergenic")], "ab"):
    d = r[r.contrast == contrast]
    order = sorted(d.length_bin.unique(), key=lambda s: float(s.strip("[(").split(",")[0]))
    labels = [f"{int(float(s.strip('[(').split(',')[0]))}–" for s in order]
    for i, (s, lab) in enumerate(SETS.items()):
        x = d[d.features == s].set_index("length_bin").reindex(order)
        ax.plot(range(len(order)), x.roc_auc, marker="o", ms=8, color=SLOTS[i], label=lab)
    ax.set_xticks(range(len(order)), labels)
    ax.set_xlabel("ORF length bin (aa)")
    ax.set_ylabel("ROC AUC within bin")
    ax.set_title(f"Prodigal+ vs {ctxt}", loc="left", fontsize=11)
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    panel(ax, letter)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=10)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "figL03_models_within_length")
''')

# ------------------------------------------------------------------ L04
FIGS["figL04_fragments"] = dict(
    title="LOGAN: contig-end fragments",
    caption=(
        "The complete-ORF model applied unchanged to contig-end fragments: Prodigal partial genes against "
        "truncated Prodigal− ORFs, open at the 5′ or 3′ end. ROC AUC (a) and average precision (b), length "
        "alone versus length plus all four entropy features; population-weighted, out-of-fold."),
    data=["figL04_fragments.tsv.gz"],
    code=r'''
r = pd.read_csv(DATA / "figL04_fragments.tsv.gz", sep="\t")
r["features"] = r.features.map({"length": "length alone", "length+aa+3di+12st+mi": "length + all four"})
r["truncation"] = r.truncation.map({"open5": "open at 5′ end", "open3": "open at 3′ end"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, metric, lab, letter in zip(axes, ["roc_auc", "avg_precision"], ["ROC AUC", "average precision"], "ab"):
    sns.barplot(data=r, x="truncation", y=metric, hue="features", palette=[SLOTS[0], SLOTS[1]], ax=ax)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=9, color=INK2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel(lab)
    ax.legend(fontsize=9, loc="upper left")
    panel(ax, letter)
fig.tight_layout()
save(fig, "figL04_fragments")
''')

# ------------------------------------------------------------------ L05
FIGS["figL05_six_frame"] = dict(
    title="LOGAN: six-frame inheritance, discrimination and Δ",
    caption=(
        "Six-frame experiment on 8.82 million complete Prodigal genes (1% deterministic sample, all 90 "
        "shards). (a) Spearman correlation between the value in Prodigal's selected frame and in each of the "
        "five alternative frames (same strand +1/+2, opposite strand a0–a2): length and AA entropy are partly "
        "shared between frames; 3Di entropy and MI are not. (b) Fraction of genes for which each feature ranks "
        "the selected frame first, in the top two or top three of six (5,722,417 genes with all five "
        "alternatives; ties counted fractionally; dotted lines, chance 1/6, 1/3, 1/2). (c–g) Distribution of "
        "Δ = selected-frame value − median of the five alternatives (central 99% shown); the fraction of genes "
        "with Δ > 0 is computed over all genes in the sample."),
    data=["figL05_six_frame_inheritance.tsv.gz", "figL05_six_frame_discrimination.tsv.gz",
          "figL05_six_frame_delta.tsv.gz"],
    code=r'''
FEAT = {"aa_len": "ORF length", "aa_entropy": "AA entropy", "three_di_entropy": "3Di entropy",
        "twelve_state_entropy": "12-state entropy", "mi_3di_12st": "3Di–12st MI"}
FRAME = {"s1": "same strand +1", "s2": "same strand +2", "a0": "opposite a0", "a1": "opposite a1",
         "a2": "opposite a2"}
I = pd.read_csv(DATA / "figL05_six_frame_inheritance.tsv.gz", sep="\t")
D = pd.read_csv(DATA / "figL05_six_frame_discrimination.tsv.gz", sep="\t")
H = pd.read_csv(DATA / "figL05_six_frame_delta.tsv.gz", sep="\t")

fig = plt.figure(figsize=(19, 10))
gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1], hspace=0.45, wspace=0.35)
ax = fig.add_subplot(gs[0, :2])
M = I.pivot(index="feature", columns="frame", values="spearman").loc[list(FEAT), list(FRAME)]
sns.heatmap(M.rename(index=FEAT, columns=FRAME), annot=True, fmt=".2f", vmin=-0.5, vmax=0.5,
            cmap=sns.diverging_palette(250, 25, s=75, l=55, center="light", as_cmap=True),
            linewidths=2, linecolor="white", cbar_kws={"label": "Spearman ρ"}, ax=ax)
ax.set_xlabel("alternative frame")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=30)
panel(ax, "a", x=-0.25)

ax = fig.add_subplot(gs[0, 2:])
d = D[D.scope == "genes with all 5 alternatives"].set_index("feature").loc[list(FEAT)]
x = np.arange(len(FEAT))
w = 0.26
for i, (top, lab) in enumerate(((1, "rank 1"), (2, "top 2"), (3, "top 3"))):
    b = ax.bar(x + (i - 1) * w, d[f"top{top}_rate"], width=w, color=SLOTS[i], label=lab)
    ax.bar_label(b, fmt="%.3f", fontsize=8.5, color=INK2, padding=1)
    ax.hlines(d[f"top{top}_chance"].iloc[0], x[0] - 1.5 * w, x[-1] + 1.5 * w, colors=SLOTS[i],
              linestyles=":", lw=1.4)
ax.set_xticks(x, [FEAT[f] for f in FEAT])
ax.set_ylim(0, 1.08)
ax.set_ylabel(f"fraction of genes (n = {int(d.genes.iloc[0]):,})")
ax.legend(loc="lower right", ncol=3, fontsize=9)
panel(ax, "b")

for j, (f, col) in enumerate(zip(FEAT, SLOTS)):
    ax = fig.add_subplot(gs[1, j])
    h = H[H.feature == f]
    edges = np.r_[h.bin_left.to_numpy(), h.bin_right.to_numpy()[-1]]
    ax.stairs(h["count"].to_numpy(), edges, fill=True, color=col, alpha=0.9)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_xlabel(f"Δ {FEAT[f]}" + (" (aa)" if f == "aa_len" else " (bits)"))
    ax.set_yticks([])
    ax.set_title(f"{h.fraction_positive.iloc[0]:.1%} > 0", loc="left", fontsize=10)
    panel(ax, "cdefg"[j], x=-0.05)
save(fig, "figL05_six_frame")
''')

# ------------------------------------------------------------------ L06
FIGS["figL06_missed_gene_evidence"] = dict(
    title="LOGAN: Foldseek evidence for missed-gene candidates and matched controls",
    caption=(
        "Structural (3Di+AA, solid) and sequence-only (AA, hatched) Foldseek evidence for 20,000 high-scoring "
        "intergenic ORFs ≥100 aa that Prodigal did not call (candidates, score ≥ the 0.90-precision threshold), "
        "and arms matched to them by nearest length and truncation: complete Prodigal genes (positive "
        "control), low-scoring intergenic ORFs, antisense and frameshift shadows, and the candidates with "
        "their 3Di strings shuffled (amino acids unchanged). (a) Hits with full-length mutual coverage "
        "(query and target coverage ≥0.8), the primary readout; (b) any hit at E ≤ 10⁻³. Databases: PDB100, "
        "AFDB/Swiss-Prot, CATH50, BFVD and AFDB50. The shuffled null has one full-length hit in 20,000."),
    data=["figL06_evidence_rates.tsv.gz"],
    code=r'''
R = pd.read_csv(DATA / "figL06_evidence_rates.tsv.gz", sep="\t")
R = R[R.db == "any"]
ARMS = ["prodigal", "candidate", "candidate_shuffled_3di", "intergenic_lo", "antisense", "frameshift"]
LABEL = {"prodigal": "complete Prodigal genes", "candidate": "high-score intergenic (candidates)",
         "candidate_shuffled_3di": "candidates, shuffled 3Di", "intergenic_lo": "low-score intergenic",
         "antisense": "antisense shadows", "frameshift": "frameshift shadows"}
COLOR = dict(zip(ARMS, [SLOTS[0], SLOTS[1], SLOTS[6], SLOTS[2], SLOTS[3], SLOTS[4]]))


def pct(v):
    return f"{100 * v:.3f}%" if v < 0.001 else f"{100 * v:.2f}%" if v < 0.01 else f"{100 * v:.1f}%"


fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
for ax, (metric, title), letter in zip(axes, (("fullcov_rate", "full-length hit (qcov and tcov ≥ 0.8)"),
                                              ("hit_rate", "any hit (E ≤ 10⁻³)")), "ab"):
    y = np.arange(len(ARMS))
    for off, mode, hatch in ((-0.19, "struct", None), (0.19, "seq", "///")):
        sel = R[R["mode"] == mode].set_index("arm")
        v = [sel.loc[a, metric] if a in sel.index else np.nan for a in ARMS]
        ax.barh(y + off, v, height=0.36, color=[COLOR[a] for a in ARMS], hatch=hatch, edgecolor="white", lw=0.5)
        for yi, vi in zip(y, v):
            if not np.isnan(vi):
                ax.text(vi + 0.008, yi + off, pct(vi), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [LABEL[a] for a in ARMS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("fraction of queries")
    ax.set_title(title, loc="left", fontsize=11)
    panel(ax, letter, x=-0.02 if letter == "b" else -0.52)
axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=INK2), plt.Rectangle((0, 0), 1, 1, fc=INK2, hatch="///", ec="white")],
               labels=["3Di+AA (structure)", "AA only (sequence)"], loc="upper right")
fig.tight_layout()
save(fig, "figL06_missed_gene_evidence")
''')

# ------------------------------------------------------------------ L07
FIGS["figL07_candidate_evidence_detail"] = dict(
    title="LOGAN: candidate evidence by length, evidence class and biome",
    caption=(
        "(a) Full-length structural hit rate by ORF length for candidates, complete Prodigal genes and the "
        "low-score intergenic and antisense controls (the controls lie together near zero). (b) Full-length "
        "evidence classes over all five databases: sequence+structure, structure only, sequence only, "
        "neither. (c) Full-length structural hit rate by biome for candidates and complete Prodigal genes. "
        "(d) Superkingdom of the best full-length target for candidates with a hit, by biome; shares reflect "
        "the composition of the target databases and are compared between arms rather than read as absolute."),
    data=["figL07_evidence_by_length.tsv.gz", "figL07_evidence_classes.tsv.gz", "figL07_hit_biome.tsv.gz"],
    code=r'''
L = pd.read_csv(DATA / "figL07_evidence_by_length.tsv.gz", sep="\t")
E = pd.read_csv(DATA / "figL07_evidence_classes.tsv.gz", sep="\t", index_col=0)
B = pd.read_csv(DATA / "figL07_hit_biome.tsv.gz", sep="\t")
ARMS = ["prodigal", "candidate", "intergenic_lo", "antisense"]
LABEL = {"prodigal": "complete Prodigal genes", "candidate": "candidates", "intergenic_lo": "low-score intergenic",
         "antisense": "antisense shadows"}
COLOR = dict(zip(ARMS, [SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[3]]))
EV = ["sequence+structure", "structure-only", "sequence-only", "neither"]
EVC = dict(zip(EV, [SLOTS[0], SLOTS[1], SLOTS[3], GRID]))
SK = ["Bacteria", "Archaea", "Eukaryota", "Viruses"]
SKC = dict(zip(SK, [SLOTS[0], SLOTS[6], SLOTS[2], SLOTS[1]]))

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], hspace=0.38, wspace=0.42)
ax = fig.add_subplot(gs[0, 0])
L["lo"] = L.length_bin.str.extract(r"\[(\d+)").astype(float)
for a in ARMS:
    g = L[L.arm == a].sort_values("lo")
    lab = g.length_bin.str.replace(r"\[(\d+)\.0, (\d+)\.0\)", r"\1–\2", regex=True)
    ax.plot(lab, g.fullcov_struct_rate, marker="o", ms=8, color=COLOR[a], label=LABEL[a])
ax.set_ylim(0, 1)
ax.set_xlabel("ORF length (aa)")
ax.set_ylabel("full-length structural hit rate")
ax.legend(loc="upper left", fontsize=9)
panel(ax, "a")

ax = fig.add_subplot(gs[0, 1])
Ee = E.reindex(ARMS)[EV]
left = np.zeros(len(ARMS))
for ev in EV:
    ax.barh(range(len(ARMS)), Ee[ev], left=left, color=EVC[ev], label=ev, height=0.6, edgecolor="white", lw=2)
    left += Ee[ev].to_numpy()
ax.set_yticks(range(len(ARMS)), [LABEL[a] for a in ARMS])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("fraction of queries (full-length evidence, five databases)")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=9)
panel(ax, "b", x=-0.3)

Bc = B[B.arm == "candidate"].set_index("biome").sort_values("rate")
Bp = B[B.arm == "prodigal"].set_index("biome").reindex(Bc.index)
y = np.arange(len(Bc))
ax = fig.add_subplot(gs[1, 0])
ax.barh(y - 0.19, Bp.rate, height=0.36, color=SLOTS[0], label="complete Prodigal genes")
ax.barh(y + 0.19, Bc.rate, height=0.36, color=SLOTS[1], label="candidates")
ax.set_yticks(y, [b.replace("_", " ") for b in Bc.index])
ax.set_xlim(0, 1)
ax.set_xlabel("full-length structural hit rate")
ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
panel(ax, "c", x=-0.25)
ax = fig.add_subplot(gs[1, 1])
left = np.zeros(len(Bc))
for k in SK:
    v = Bc[f"{k}_of_hits"].fillna(0).to_numpy()
    ax.barh(y, v, left=left, color=SKC[k], label=k, height=0.6, edgecolor="white", lw=2)
    left += v
ax.set_yticks(y, [b.replace("_", " ") for b in Bc.index])
ax.set_xlim(0, 1)
ax.set_xlabel("candidates with a hit: superkingdom of best target")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
panel(ax, "d", x=-0.25)
save(fig, "figL07_candidate_evidence_detail")
''')

# ------------------------------------------------------------------ L08
FIGS["figL08_contig_taxonomy"] = dict(
    title="LOGAN: contig taxonomy and the eukaryotic test",
    caption=(
        "Kraken2 (2.17.2; PlusPFP 2023-06-05) classification of the 79,774 contigs carrying the Foldseek "
        "queries, re-streamed from LOGAN. (a) Contig class for each arm. (b) For candidates and complete "
        "Prodigal genes whose best full-length target is eukaryotic, the class of their contig: 69.8% of such "
        "candidates (n = 2,525) lie on eukaryotic contigs. PlusPFP contains a single animal genome (human), "
        "so animal contigs are reported as one group; 15.6% of contigs are unclassified."),
    data=["figL08_contig_group_by_arm.tsv.gz", "figL08_hit_vs_contig.tsv.gz"],
    code=r'''
C = pd.read_csv(DATA / "figL08_contig_group_by_arm.tsv.gz", sep="\t", index_col=0)
X = pd.read_csv(DATA / "figL08_hit_vs_contig.tsv.gz", sep="\t")
GROUPS = ["Bacteria", "Archaea", "Viruses", "plants", "animals", "fungi", "other eukaryotes", "unclassified"]
GC = dict(zip(GROUPS, [SLOTS[0], SLOTS[6], SLOTS[1], SLOTS[5], SLOTS[4], SLOTS[3], SLOTS[2], GRID]))
EUK = {"plants", "animals", "fungi", "other eukaryotes"}
ARMS = ["prodigal", "candidate", "intergenic_lo", "antisense"]
LABEL = {"prodigal": "complete Prodigal genes", "candidate": "candidates", "intergenic_lo": "low-score intergenic",
         "antisense": "antisense shadows"}

fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
Cc = C.reindex(ARMS)[GROUPS]
left = np.zeros(len(ARMS))
for g in GROUPS:
    ax.barh(range(len(ARMS)), Cc[g], left=left, color=GC[g], label=g, height=0.6, edgecolor="white", lw=2)
    left += Cc[g].to_numpy()
ax.set_yticks(range(len(ARMS)), [LABEL[a] for a in ARMS])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("share of queries, by Kraken2 class of the contig")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=9)
panel(ax, "a", x=-0.32)
ax = axes[1]
arms = ["prodigal", "candidate"]
e = X[X.arm.isin(arms) & X.hit_group.isin(EUK)]
S = e.pivot_table(index="arm", columns="contig_group", values="queries", aggfunc="sum", fill_value=0)
n = S.sum(axis=1)
S = S.div(n, axis=0).reindex(index=arms, columns=GROUPS, fill_value=0)
left = np.zeros(len(arms))
for g in GROUPS:
    ax.barh(range(len(arms)), S[g], left=left, color=GC[g], height=0.6, edgecolor="white", lw=2)
    left += S[g].to_numpy()
ax.set_yticks(range(len(arms)), [f"{LABEL[a]}\n(n = {int(n[a]):,})" for a in arms])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("queries with a eukaryotic best hit: Kraken2 class of the contig")
panel(ax, "b", x=-0.3)
fig.tight_layout()
save(fig, "figL08_contig_taxonomy")
''')

# ------------------------------------------------------------------ L09
FIGS["figL09_prodigal_skip"] = dict(
    title="LOGAN: why Prodigal did not call the candidates",
    caption=(
        "Prodigal 2.6.3 (-p meta) rerun on the query contigs with -s, which lists every potential gene and its "
        "scores; the rerun reproduces all original calls. For each query, the best-scoring potential gene "
        "ending at its stop. (a) Coding score; (b) total score (coding + start + RBS + upstream); (c) ORF GC "
        "minus the GC of the Prodigal model chosen for the contig; values outside each axis range are clipped "
        "to its edges. (d) Why the ORF was not called: no potential gene at that stop, negative coding score, "
        "positive coding score but start penalties make the total negative, or positive total but not "
        "chosen. Bacterial-hit candidates: best full-length target bacterial."),
    data=["figL09_prodigal_skip.tsv.gz"],
    code=r'''
F = pd.read_csv(DATA / "figL09_prodigal_skip.tsv.gz", sep="\t")
GROUPS = ["Prodigal gene, bacterial hit", "candidate, bacterial hit", "candidate, other hit",
          "candidate, no hit", "low-score intergenic"]
GCOL = dict(zip(GROUPS, [SLOTS[0], SLOTS[1], SLOTS[4], SLOTS[3], SLOTS[2]]))
REASONS = ["positive, not called", "start penalties", "negative coding", "no potential gene"]
RCOL = dict(zip(REASONS, [SLOTS[0], SLOTS[3], SLOTS[1], GRID]))
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, (col, lab, lim), letter in zip(axes.flat[:3], (("CodPot", "Prodigal coding score (best start)", (-30, 60)),
                                                       ("Total", "Prodigal total score (best start)", (-40, 60)),
                                                       ("orf_minus_model_gc", "ORF GC − Prodigal model GC", (-.3, .3))),
                                       "abc"):
    for g in GROUPS:
        v = F.loc[F.group == g, col].dropna().clip(*lim)
        if len(v) > 20:
            sns.kdeplot(v, ax=ax, color=GCOL[g], label=f"{g} (n={len(v):,})", lw=2, cut=0)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_xlabel(lab)
    ax.set_xlim(*lim)
    panel(ax, letter)
axes[0, 0].legend(fontsize=8.5, loc="upper right")
ax = axes[1, 1]
cg = GROUPS[1:]
R = pd.crosstab(F.group, F.reason, normalize="index").reindex(index=cg, columns=REASONS, fill_value=0)
left = np.zeros(len(cg))
for rs in REASONS:
    ax.barh(range(len(cg)), R[rs], left=left, color=RCOL[rs], label=rs, height=0.6, edgecolor="white", lw=2)
    left += R[rs].to_numpy()
ax.set_yticks(range(len(cg)), cg)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("why Prodigal did not call the ORF")
ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=9)
panel(ax, "d", x=-0.4)
fig.tight_layout()
save(fig, "figL09_prodigal_skip")
''')

# ------------------------------------------------------------------ L10
FIGS["figL10_prodigal_lowscore"] = dict(
    title="LOGAN: Prodigal genes the model scores low",
    caption=(
        "The second discordant class: complete Prodigal genes scoring below the 0.80-precision threshold. "
        "(a) Share below the threshold by gene length, from the scan's 1-in-1,000 sample of complete Prodigal "
        "genes (880,405). (b) Full-length structural hit rate for 10,269 low-scoring genes and 8,769 "
        "high-scoring genes (≥ the 0.90 threshold) matched to them by length (from 50 aa; high-scoring genes "
        "below 50 aa barely exist); bins with fewer than 100 genes are not drawn. (c) Full-length structural hit "
        "rate of every length-matched arm (five databases), including the frameshift-shadow control."),
    data=["figL10_lowscore_population.tsv.gz", "figL10_lowscore_evidence_by_length.tsv.gz",
          "figL10_all_arms_evidence.tsv.gz"],
    code=r'''
P = pd.read_csv(DATA / "figL10_lowscore_population.tsv.gz", sep="\t", index_col=0).drop(index="all")
L = pd.read_csv(DATA / "figL10_lowscore_evidence_by_length.tsv.gz", sep="\t", index_col=0)
A = pd.read_csv(DATA / "figL10_all_arms_evidence.tsv.gz", sep="\t", index_col=0)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1, 1.1, 1]})
ax = axes[0]
ax.bar(range(len(P)), P.below_t80, color=SLOTS[1], width=0.7)
ax.set_xticks(range(len(P)), P.index, rotation=30)
ax.set_ylim(0, 1.08)
ax.set_xlabel("gene length (aa)")
ax.set_ylabel("complete Prodigal genes below threshold")
for i, v in enumerate(P.below_t80):
    ax.text(i, v + 0.015, f"{100 * v:.1f}%" if v < 0.1 else f"{100 * v:.0f}%", ha="center", fontsize=8.5, color=INK2)
panel(ax, "a")
ax = axes[1]
x = np.arange(len(L))
for arm, c, lab in (("prodigal_hi", SLOTS[0], "score ≥ 0.90 threshold"), ("prodigal_lo", SLOTS[1], "score < 0.80 threshold")):
    ok = L[f"queries_{arm}"].fillna(0).to_numpy() >= 100
    ax.plot(x[ok], L[f"full_struct_{arm}"].to_numpy()[ok], marker="o", ms=8, color=c, label=f"Prodigal genes, {lab}")
ax.set_xticks(x, L.index, rotation=30)
ax.set_ylim(0, 1)
ax.set_xlabel("gene length (aa)")
ax.set_ylabel("full-length structural hit rate")
ax.legend(loc="center right", fontsize=9)
panel(ax, "b")
ax = axes[2]
arms = ["prodigal", "prodigal_hi", "candidate", "prodigal_lo", "frameshift", "antisense", "intergenic_lo",
        "candidate_shuffled_3di"]
names = {"prodigal": "complete Prodigal genes", "prodigal_hi": "Prodigal genes, high score",
         "candidate": "candidates", "prodigal_lo": "Prodigal genes, low score", "frameshift": "frameshift shadows",
         "antisense": "antisense shadows", "intergenic_lo": "low-score intergenic",
         "candidate_shuffled_3di": "candidates, shuffled 3Di"}
cols = {"prodigal": SLOTS[0], "prodigal_hi": SLOTS[0], "candidate": SLOTS[1], "prodigal_lo": SLOTS[1],
        "frameshift": SLOTS[4], "antisense": SLOTS[3], "intergenic_lo": SLOTS[2], "candidate_shuffled_3di": SLOTS[6]}
v = A.loc[arms, "fullcov_rate_struct"]
ax.barh(range(len(arms)), v, color=[cols[a] for a in arms], height=0.6)
for i, xv in enumerate(v):
    ax.text(xv + 0.01, i, f"{100 * xv:.3f}%" if xv < 0.001 else f"{100 * xv:.2f}%" if xv < 0.01 else f"{100 * xv:.1f}%",
            va="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(arms)), [names[a] for a in arms])
ax.invert_yaxis()
ax.set_xlim(0, 1.05)
ax.set_xlabel("full-length structural hit rate")
panel(ax, "c", x=-0.55)
fig.tight_layout()
save(fig, "figL10_prodigal_lowscore")
''')

# ------------------------------------------------------------------ LS1
FIGS["figLS1_biome_batch_separation"] = dict(
    title="LOGAN: length-controlled separation by biome and tranche",
    caption=(
        "Exact within-length-bin ROC AUC (from all-ORF histograms) for complete Prodigal+ against all complete "
        "Prodigal− ORFs. (a) By biome at 100–149 aa, ordered by 3Di entropy (n = Prodigal+ ORFs in the bin). "
        "(b) By tranche (pilot, phaseb1, phaseb2) at 100–149 and 150–199 aa. (c) 3Di entropy within each length "
        "bin, by biome. ORF length within a bin is shown as a reference."),
    data=["figLS1_biome_batch_separation.tsv.gz"],
    code=r'''
R = pd.read_csv(DATA / "figLS1_biome_batch_separation.tsv.gz", sep="\t")
LEN_LABELS = ["30-49", "50-74", "75-99", "100-149", "150-199", "200-299", "300-499", "500-999", ">=1000"]
FEATS = ["aa_len", "aa_entropy", "twelve_state_entropy", "three_di_entropy", "mi_3di_12st"]
LAB = {"aa_len": "ORF length (within bin)", "aa_entropy": "AA entropy", "twelve_state_entropy": "12-state entropy",
       "three_di_entropy": "3Di entropy", "mi_3di_12st": "3Di–12st MI"}
COL = {"aa_len": SLOTS[6], "aa_entropy": SLOTS[3], "twelve_state_entropy": SLOTS[2], "three_di_entropy": SLOTS[0],
       "mi_3di_12st": SLOTS[1]}
BATCHES = ["pilot", "phaseb1", "phaseb2"]
w = 0.16
fig = plt.figure(figsize=(19, 10.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 1.05], hspace=0.42, wspace=0.15)
axA = fig.add_subplot(gs[0, :])
sb = R[(R.by == "biome") & (R.length_bin == "100-149")]
order = sb[sb.feature == "three_di_entropy"].sort_values("roc_auc").stratum.tolist()
x = np.arange(len(order))
for j, f in enumerate(FEATS):
    v = sb[sb.feature == f].set_index("stratum").reindex(order).roc_auc
    axA.bar(x + (j - 2) * w, v, width=w, color=COL[f], label=LAB[f])
npos = sb[sb.feature == "aa_len"].set_index("stratum").positives
axA.set_xticks(x, [f"{b.replace('_', ' ')}\n{npos[b] / 1e6:.1f} M" for b in order], fontsize=9.5)
axA.set_ylim(0.5, 1.0)
axA.set_ylabel("ROC AUC within 100–149 aa")
axA.legend(ncol=5, fontsize=10, loc="upper left")
panel(axA, "a", x=-0.03)
axB = fig.add_subplot(gs[1, 0])
groups = [(bt, lb) for lb in ["100-149", "150-199"] for bt in BATCHES]
xb = np.arange(len(groups))
sbt = R[R.by == "batch"].set_index(["stratum", "length_bin", "feature"])
for j, f in enumerate(FEATS):
    v = [sbt.loc[(bt, lb, f), "roc_auc"] for bt, lb in groups]
    axB.bar(xb + (j - 2) * w, v, width=w, color=COL[f])
axB.set_xticks(xb, [f"{bt}\n{lb.replace('-', '–')} aa" for bt, lb in groups], fontsize=9.5)
axB.set_ylim(0.5, 1.0)
axB.set_ylabel("ROC AUC within bin")
panel(axB, "b", x=-0.06)
axC = fig.add_subplot(gs[1, 1])
hb = R[(R.by == "biome") & (R.feature == "three_di_entropy") & ~R.length_bin.isin(["all", "stratified"])]
M = hb.pivot(index="stratum", columns="length_bin", values="roc_auc").reindex(
    index=order[::-1], columns=[c for c in LEN_LABELS if c in set(hb.length_bin)])
sns.heatmap(M, ax=axC, cmap="viridis", vmin=0.5, vmax=1.0, annot=True, fmt=".2f", annot_kws={"fontsize": 6.5},
            cbar_kws={"label": "ROC AUC within bin", "shrink": 0.8})
axC.set_yticklabels([b.replace("_", " ") for b in order[::-1]], rotation=0, fontsize=9)
axC.set_xlabel("ORF length bin (aa)")
axC.set_ylabel("")
panel(axC, "c", x=-0.25)
save(fig, "figLS1_biome_batch_separation")
''')


def provenance_md(data_files: list[str]) -> str:
    lines = []
    for f in data_files:
        o = PROV["outputs"][f]
        srcs = "; ".join(f"`{Path(s).relative_to(PROV['logan_root']) if s.startswith(PROV['logan_root']) else s}`"
                         for s in o["sources"])
        lines.append(f"- `../data/{f}` ({o['rows']:,} rows; sha256 `{o['sha256'][:12]}…`): {o['note']}. "
                     f"Source on Setonix (`$LOGAN_ROOT`): {srcs or 'see logan_provenance.json'}.")
    return "\n".join(lines)


def main() -> int:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, f in FIGS.items():
        nb = nbf.v4.new_notebook()
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}
        md = (f"# {f['title']}\n\n"
              f"**Draft caption.** {f['caption']}\n\n"
              f"**Outputs:** `../images/{name}.pdf` (vector) and `../images/{name}.png` (300 dpi).\n\n"
              f"**Data** (plotting-ready, committed; produced by `../prep/logan_prepare_data.py` on Setonix):\n\n"
              f"{provenance_md(f['data'])}\n\n"
              f"**Upstream.** LOGAN v1.2 metagenomes (50,009), genome_entropy #102 (closed by PR #115); analysis "
              f"scripts `Validation/logan/scripts/` at commit `{PROV['scripts_commit'][:7]}`. Full run provenance: "
              f"`../data/logan_provenance.json`.\n\n"
              f"**Run:** `jupyter nbconvert --to notebook --execute {name}.ipynb` from `manuscript/notebooks/`. "
              f"Needs only pandas, numpy, matplotlib and seaborn; no HPC or remote paths.")
        nb.cells = [nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(SETUP.strip()),
                    nbf.v4.new_code_cell(f["code"].strip())]
        nbf.write(nb, NB_DIR / f"{name}.ipynb")
        print("wrote", NB_DIR / f"{name}.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
