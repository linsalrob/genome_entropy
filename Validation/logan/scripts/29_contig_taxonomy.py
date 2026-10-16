#!/usr/bin/env python3
"""Contig taxonomy of the query ORFs, and whether eukaryotic contigs explain the eukaryotic hits.

Inputs (qc/candidates): contigs/kraken.out (28_kraken_contigs.slurm),
selection.parquet, fullcov_struct_best.parquet (25_candidate_figures.py).
Kraken2 taxids are placed with the database's own ktaxonomy.tsv; Foldseek
target taxids with an NCBI nodes.dmp. Both are reduced to the same groups:
Bacteria, Archaea, Viruses, Eukaryota split into plants (Viridiplantae),
animals (Metazoa), fungi and other eukaryotes, plus unclassified.

Writes to --out:
  contig_taxonomy.tsv        one row per query contig: Kraken2 taxid, rank, name, group
  contig_group_by_arm.tsv    share of each arm's queries by contig group
  hit_vs_contig.tsv          candidates and Prodigal genes with a full-length
                             structural hit: best-target group x contig group
  euk_explained.tsv          for queries whose best target is eukaryotic: share
                             on eukaryotic / bacterial / unclassified contigs,
                             and agreement of the eukaryotic group
  euk_by_biome.tsv           the same by biome (candidates)
  figure fig17_contig_taxonomy

  29_contig_taxonomy.py --dir qc/candidates \
      --ktax /scratch/references/kraken2/pluspfp_20230605/ktaxonomy.tsv \
      --nodes /scratch/references/kraken_dec2025/nodes.dmp --out qc/candidates/taxonomy
"""

from __future__ import annotations

import argparse
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
# Most specific first: a plant is checked before Eukaryota.
GROUP_TAXA = [(33090, "plants"), (33208, "animals"), (4751, "fungi"), (2759, "other eukaryotes"),
              (2, "Bacteria"), (2157, "Archaea"), (10239, "Viruses")]
GROUPS = ["Bacteria", "Archaea", "Viruses", "plants", "animals", "fungi", "other eukaryotes", "unclassified"]
EUK = {"plants", "animals", "fungi", "other eukaryotes"}
GROUP_COLOR = dict(zip(GROUPS, [SLOTS[0], SLOTS[6], SLOTS[1], SLOTS[5], SLOTS[4], SLOTS[3], SLOTS[2], GRID]))
ARMS = ["prodigal", "candidate", "intergenic_lo", "antisense"]
ARM_LABEL = {"prodigal": "complete Prodigal genes", "candidate": "high-score intergenic (candidates)",
             "intergenic_lo": "low-score intergenic", "antisense": "antisense shadows"}


def style():
    sns.set_theme(context="paper", style="ticks", font_scale=1.15, rc={
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": "white",
        "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False})


def grouper(parent: dict[int, int]):
    cache: dict[int, str] = {}
    targets = dict(GROUP_TAXA)

    def one(t) -> str:
        if pd.isna(t) or int(t) <= 1:
            return "unclassified"
        t = int(t)
        if t not in cache:
            u, found = t, set()
            for _ in range(200):
                if u in targets:
                    found.add(u)
                if u not in parent or parent[u] == u:
                    break
                u = parent[u]
            g = next((name for tx, name in GROUP_TAXA if tx in found), "unclassified")
            cache[t] = g
        return cache[t]
    return one


def read_ktax(path: str) -> tuple[dict[int, int], dict[int, tuple[str, str]]]:
    par, info = {}, {}
    for line in open(path):
        f = [x.strip() for x in line.rstrip("\n").split("|")]
        par[int(f[0])] = int(f[1])
        info[int(f[0])] = (f[2], f[4])
    return par, info


def read_nodes(path: str) -> dict[int, int]:
    par = {}
    for line in open(path):
        f = line.split("\t|\t")
        par[int(f[0])] = int(f[1])
    return par


def share(df: pd.DataFrame, by: str, col: str) -> pd.DataFrame:
    return pd.crosstab(df[by], df[col], normalize="index").reindex(columns=GROUPS, fill_value=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ktax", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    kpar, kinfo = read_ktax(args.ktax)
    kg = grouper(kpar)
    k = pd.read_csv(D / "contigs" / "kraken.out", sep="\t", header=None, usecols=[0, 1, 2, 3],
                    names=["status", "contig", "taxid", "length"])
    k["contig_group"] = [kg(t) if s == "C" else "unclassified" for s, t in zip(k.status, k.taxid)]
    k["rank"] = [kinfo.get(int(t), ("", ""))[0] for t in k.taxid]
    k["name"] = [kinfo.get(int(t), ("", ""))[1] for t in k.taxid]
    k.to_csv(out / "contig_taxonomy.tsv", sep="\t", index=False)
    print(f"{len(k):,} contigs; classified {100 * (k.status == 'C').mean():.1f}%")

    sel = pd.read_parquet(D / "selection.parquet", columns=["query_id", "arm", "accession", "contig_n", "biome"])
    sel["contig"] = sel.accession + "_" + sel.contig_n.astype(str)
    q = sel.merge(k[["contig", "contig_group", "rank", "name"]], on="contig", how="left")
    missing = int(q.contig_group.isna().sum())
    q["contig_group"] = q.contig_group.fillna("not fetched")
    print(f"queries without a fetched contig: {missing}")
    C = share(q[q.contig_group != "not fetched"], "arm", "contig_group").reindex(ARMS)
    C.to_csv(out / "contig_group_by_arm.tsv", sep="\t")

    best = pd.read_parquet(D / "fullcov_struct_best.parquet", columns=["query", "taxid", "bits", "fident"])
    ng = grouper(read_nodes(args.nodes))
    best["hit_group"] = [ng(t) for t in best.taxid]
    h = q.merge(best[["query", "hit_group"]], left_on="query_id", right_on="query", how="inner")
    h = h[h.contig_group != "not fetched"]
    X = (h[h.arm.isin(["candidate", "prodigal"])].groupby(["arm", "hit_group", "contig_group"]).size()
         .rename("queries").reset_index())
    X.to_csv(out / "hit_vs_contig.tsv", sep="\t", index=False)

    rows = []
    for arm, g in h[h.arm.isin(["candidate", "prodigal"])].groupby("arm"):
        e = g[g.hit_group.isin(EUK)]
        rows.append({"arm": arm, "queries_with_hit": len(g), "euk_hits": len(e),
                     "euk_hit_share": len(e) / len(g),
                     "on_euk_contig": e.contig_group.isin(EUK).mean(),
                     "on_bacterial_contig": (e.contig_group == "Bacteria").mean(),
                     "on_unclassified_contig": (e.contig_group == "unclassified").mean(),
                     "same_euk_group": (e.hit_group == e.contig_group).mean(),
                     "euk_hit_rate_on_euk_contigs": g[g.contig_group.isin(EUK)].hit_group.isin(EUK).mean(),
                     "euk_hit_rate_on_bacterial_contigs": g[g.contig_group == "Bacteria"].hit_group.isin(EUK).mean(),
                     "euk_hit_rate_on_unclassified_contigs":
                         g[g.contig_group == "unclassified"].hit_group.isin(EUK).mean()})
    E = pd.DataFrame(rows)
    E.to_csv(out / "euk_explained.tsv", sep="\t", index=False)

    c = q[(q.arm == "candidate") & (q.contig_group != "not fetched")]
    hc = h[h.arm == "candidate"]
    B = pd.DataFrame({
        "candidates": c.groupby("biome").size(),
        "on_euk_contig": c.assign(x=c.contig_group.isin(EUK)).groupby("biome").x.mean(),
        "unclassified_contig": c.assign(x=c.contig_group == "unclassified").groupby("biome").x.mean(),
        "with_hit": hc.groupby("biome").size(),
        "euk_hit_share": hc.assign(x=hc.hit_group.isin(EUK)).groupby("biome").x.mean(),
        "euk_hits_on_euk_contig": hc[hc.hit_group.isin(EUK)].assign(x=lambda d: d.contig_group.isin(EUK))
        .groupby("biome").x.mean()}).sort_values("euk_hit_share")
    B.to_csv(out / "euk_by_biome.tsv", sep="\t")

    # fig17: (a) contig group by arm; (b) eukaryotic best hits by contig group
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    left = np.zeros(len(ARMS))
    for gname in GROUPS:
        v = C[gname].to_numpy()
        ax.barh(range(len(ARMS)), v, left=left, color=GROUP_COLOR[gname], label=gname, height=0.6,
                edgecolor="white", linewidth=2)
        left += v
    ax.set_yticks(range(len(ARMS)), [ARM_LABEL[a] for a in ARMS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of queries, by Kraken2 class of the contig")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=9)
    ax = axes[1]
    arms2 = ["prodigal", "candidate"]
    sub = h[h.arm.isin(arms2) & h.hit_group.isin(EUK)]
    S = share(sub, "arm", "contig_group").reindex(arms2)
    left = np.zeros(len(arms2))
    for gname in GROUPS:
        v = S[gname].to_numpy()
        ax.barh(range(len(arms2)), v, left=left, color=GROUP_COLOR[gname], height=0.6, edgecolor="white",
                linewidth=2)
        left += v
    ax.set_yticks(range(len(arms2)),
                  [f"{ARM_LABEL[a]}\n(n = {int((sub.arm == a).sum()):,})" for a in arms2])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("queries with a eukaryotic best hit: Kraken2 class of the contig")
    fig.tight_layout()
    fig.savefig(out / "fig17_contig_taxonomy.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig17_contig_taxonomy.pdf", bbox_inches="tight")
    plt.close(fig)

    with pd.option_context("display.width", 220):
        print(C.round(3).to_string())
        print(E.round(3).to_string(index=False))
        print(B.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
