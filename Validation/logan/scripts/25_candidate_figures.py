#!/usr/bin/env python3
"""Missed-gene candidates: target taxonomy and figures from the Foldseek evidence.

Reads qc/candidates (24_candidates_evidence.py outputs, search/fullcov/*.m8,
selection.parquet). The best full-coverage structural hit of each query
(qcov >= 0.8 and tcov >= 0.8, highest bits over all databases) is assigned
the superkingdom of its target's taxid, using an NCBI nodes.dmp. Writes:

  fullcov_struct_best.parquet   best full-coverage structural hit per query
  hit_superkingdom.tsv          superkingdom share of those hits, by arm
  hit_biome.tsv                 hit rate and superkingdom share by arm x biome
  figures: fig13_candidate_evidence, fig14_candidate_by_length,
           fig15_candidate_evidence_classes, fig16_candidate_biome_taxonomy

The superkingdom of a best hit partly reflects database composition (BFVD is
all viral), so compare arms rather than read the shares as absolute.

  25_candidate_figures.py --dir qc/candidates --nodes /scratch/references/kraken_dec2025/nodes.dmp \
      --out qc/figures_candidates
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FIELDS = ["query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits",
          "taxid", "taxname"]
ARMS = ["prodigal", "candidate", "candidate_shuffled_3di", "intergenic_lo", "antisense"]
ARM_LABEL = {"prodigal": "complete Prodigal genes", "candidate": "high-score intergenic (candidates)",
             "candidate_shuffled_3di": "candidates, shuffled 3Di", "intergenic_lo": "low-score intergenic",
             "antisense": "antisense shadows"}
ARM_COLOR = dict(zip(ARMS, [SLOTS[0], SLOTS[1], SLOTS[6], SLOTS[2], SLOTS[3]]))
SK = {2: "Bacteria", 2157: "Archaea", 2759: "Eukaryota", 10239: "Viruses"}
SK_ORDER = ["Bacteria", "Archaea", "Eukaryota", "Viruses", "unknown"]
SK_COLOR = dict(zip(SK_ORDER, [SLOTS[0], SLOTS[6], SLOTS[2], SLOTS[1], GRID]))
EV_ORDER = ["sequence+structure", "structure-only", "sequence-only", "neither"]
EV_COLOR = dict(zip(EV_ORDER, [SLOTS[0], SLOTS[1], SLOTS[3], GRID]))


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


def superkingdoms(taxids, nodes: str) -> list[str]:
    par = {}
    with open(nodes) as fh:
        for line in fh:
            f = line.split("\t|\t")
            par[int(f[0])] = int(f[1])
    cache: dict[int, str] = {}

    def one(t: int) -> str:
        if t not in cache:
            u, name = t, "unknown"
            for _ in range(100):
                if u in SK:
                    name = SK[u]
                    break
                if u not in par or par[u] == u:
                    break
                u = par[u]
            cache[t] = name
        return cache[t]
    return [one(int(t)) if pd.notna(t) else "unknown" for t in taxids]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    rows = []
    for f in sorted(glob.glob(str(D / "search" / "fullcov" / "qdb.*.struct.m8"))):
        d = pd.read_csv(f, sep="\t", names=FIELDS, dtype={"query": str, "target": str, "taxname": str})
        d["db"] = Path(f).name.split(".")[1]
        rows.append(d)
    h = pd.concat(rows, ignore_index=True)
    best = h.sort_values("bits", ascending=False).drop_duplicates("query").reset_index(drop=True)
    best["arm"] = best["query"].str.split("|").str[0]
    best["superkingdom"] = superkingdoms(best.taxid, args.nodes)
    best.to_parquet(D / "fullcov_struct_best.parquet", index=False)
    sk = pd.crosstab(best.arm, best.superkingdom).reindex(columns=SK_ORDER, fill_value=0)
    sk.div(sk.sum(axis=1), axis=0).to_csv(D / "hit_superkingdom.tsv", sep="\t")

    sel = pd.read_parquet(D / "selection.parquet", columns=["query_id", "arm", "biome"])
    s = sel.merge(best[["query", "superkingdom"]], left_on="query_id", right_on="query", how="left")
    s["hit"] = s["query"].notna()
    B = s.groupby(["arm", "biome"]).agg(queries=("hit", "size"), hits=("hit", "sum"), rate=("hit", "mean"))
    for k in SK_ORDER[:4]:
        B[f"{k}_of_hits"] = s.assign(x=s.superkingdom == k).groupby(["arm", "biome"]).x.sum() / B.hits
    B.to_csv(D / "hit_biome.tsv", sep="\t")

    # fig13: evidence rates by arm (structure vs sequence; any hit vs full length)
    R = pd.read_csv(D / "evidence_rates.tsv", sep="\t")
    R = R[R.db == "any"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (metric, title) in zip(axes, (("fullcov_rate", "full-length hit (qcov and tcov ≥ 0.8)"),
                                          ("hit_rate", "any hit (E ≤ 10⁻³)"))):
        y = np.arange(len(ARMS))
        for off, mode, hatch in ((-0.19, "struct", None), (0.19, "seq", "///")):
            v = [R[(R.arm == a) & (R["mode"] == mode)][metric].sum() if ((R.arm == a) & (R["mode"] == mode)).any()
                 else np.nan for a in ARMS]
            ax.barh(y + off, v, height=0.36, color=[ARM_COLOR[a] for a in ARMS], hatch=hatch,
                    edgecolor="white", linewidth=0.5)
            for yi, vi in zip(y, v):
                if not np.isnan(vi):
                    ax.text(vi + 0.008, yi + off, f"{100 * vi:.1f}%", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(y, [ARM_LABEL[a] for a in ARMS])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of queries")
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=INK2), plt.Rectangle((0, 0), 1, 1, fc=INK2, hatch="///",
                                                                                ec="white")],
                   labels=["3Di+AA (structure)", "AA only (sequence)"], loc="upper right")
    dbs = ", ".join(sorted(set(h.db)))
    fig.suptitle(f"Foldseek evidence by query arm, 20,000 length-matched queries each ({dbs})",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig13_candidate_evidence")

    # fig14: full-length structural hit rate by length bin
    L = pd.read_csv(D / "evidence_by_length.tsv", sep="\t")
    L["lo"] = L.length_bin.str.extract(r"\[(\d+)").astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for a in ["prodigal", "candidate", "intergenic_lo", "antisense"]:
        g = L[L.arm == a].sort_values("lo")
        lab = g.length_bin.str.replace(r"\[(\d+)\.0, (\d+)\.0\)", r"\1–\2", regex=True)
        ax.plot(lab, g.fullcov_struct_rate, marker="o", ms=8, color=ARM_COLOR[a], label=ARM_LABEL[a])
    ax.set_ylim(0, 1)
    ax.set_xlabel("ORF length (aa)")
    ax.set_ylabel("full-length structural hit rate")
    ax.legend(loc="upper left")
    save(fig, out, "fig14_candidate_by_length")

    # fig15: evidence classes
    E = pd.read_csv(D / "evidence_classes.tsv", sep="\t", index_col=0).reindex(columns=EV_ORDER, fill_value=0)
    arms = ["prodigal", "candidate", "intergenic_lo", "antisense"]
    E = E.reindex(arms)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    left = np.zeros(len(arms))
    for ev in EV_ORDER:
        ax.barh(range(len(arms)), E[ev], left=left, color=EV_COLOR[ev], label=ev, height=0.6,
                edgecolor="white", linewidth=2)
        left += E[ev].to_numpy()
    ax.set_yticks(range(len(arms)), [ARM_LABEL[a] for a in arms])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of queries (full-length evidence over all databases)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0))
    save(fig, out, "fig15_candidate_evidence_classes")

    # fig16: candidate hit rate and target superkingdom by biome
    Bc = B.loc["candidate"].sort_values("rate")
    Bp = B.loc["prodigal"].reindex(Bc.index)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True, gridspec_kw={"width_ratios": [1, 1.4]})
    y = np.arange(len(Bc))
    axes[0].barh(y - 0.19, Bp.rate, height=0.36, color=ARM_COLOR["prodigal"], label=ARM_LABEL["prodigal"])
    axes[0].barh(y + 0.19, Bc.rate, height=0.36, color=ARM_COLOR["candidate"], label=ARM_LABEL["candidate"])
    axes[0].set_yticks(y, [b.replace("_", " ") for b in Bc.index])
    axes[0].set_xlabel("full-length structural hit rate")
    axes[0].set_xlim(0, 0.8)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    left = np.zeros(len(Bc))
    for k in SK_ORDER[:4]:
        v = Bc[f"{k}_of_hits"].fillna(0).to_numpy()
        axes[1].barh(y, v, left=left, color=SK_COLOR[k], label=k, height=0.6, edgecolor="white", linewidth=2)
        left += v
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("candidates with a hit: superkingdom of best target")
    axes[1].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "fig16_candidate_biome_taxonomy")

    with pd.option_context("display.width", 200):
        print(sk.div(sk.sum(axis=1), axis=0).round(3).to_string())
        print(B.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
