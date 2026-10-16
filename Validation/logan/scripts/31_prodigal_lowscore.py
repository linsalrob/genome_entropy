#!/usr/bin/env python3
"""The "Prodigal+ / low genome_entropy" discordant class (#102 §5), and the frameshift control (§7.3).

Population (the scan's 1e-3 sample of complete Prodigal genes,
analysis/v1/candidates, role "control", label "prodigal_complete"): share of
genes below the 0.80 / above the 0.90 score thresholds by length bin and
biome, and how the features of low-scoring genes differ from high-scoring
genes of the same length.

Evidence (Foldseek, 24_candidates_evidence.py run on --extra): full-length
structural and sequence hit rates of low- and high-scoring Prodigal genes by
length bin (prodigal_lo vs its length-matched prodigal_hi), and of the
frameshift-shadow arm beside the main arms (--main). Within the main
Prodigal arm, the full-length rate is also given by score bin.

Writes to --out: lowscore_population.tsv, lowscore_features.tsv,
lowscore_biome.tsv, lowscore_evidence_by_length.tsv, prodigal_by_score.tsv,
all_arms_evidence.tsv and fig19_prodigal_lowscore.

  31_prodigal_lowscore.py --main qc/candidates --extra qc/candidates_extra \
      --thresholds qc/analysis_models_final/final_model_thresholds.json --out qc/candidates_extra/lowscore
"""

from __future__ import annotations

import argparse
import glob
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

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BINS = [30, 50, 75, 100, 150, 200, 300, 500, 10**9]
FEATS = ["aa_entropy", "three_di_entropy", "twelve_state_entropy", "mi_3di_12st"]
FIELDS = ["query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits",
          "taxid", "taxname"]
ARM_ORDER = ["prodigal", "candidate", "candidate_shuffled_3di", "intergenic_lo", "antisense", "frameshift",
             "prodigal_hi", "prodigal_lo"]


def style():
    sns.set_theme(context="paper", style="ticks", font_scale=1.15, rc={
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": "white",
        "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2})


def label_bins(s: pd.Series) -> pd.Series:
    return pd.cut(s, BINS, right=False).astype(str).str.replace(r"\[(\d+), (\d+)\)", r"\1–\2", regex=True) \
        .str.replace("–1000000000", "+", regex=False)


def fullcov_sets(d: Path) -> tuple[set, set]:
    def qs(mode):
        out = set()
        for f in glob.glob(str(d / "search" / "fullcov" / f"qdb.*.{mode}.m8")):
            out |= set(pd.read_csv(f, sep="\t", names=FIELDS, usecols=["query"]).iloc[:, 0])
        return out
    return qs("struct"), qs("seq")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--main", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    M, X, out = Path(args.main), Path(args.extra), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    thr = json.load(open(args.thresholds))["precision_thresholds"]
    t80, t90 = float(thr["0.8"]), float(thr["0.9"])

    files = sorted(glob.glob(str(lc.root() / "analysis" / lc.CACHE_VERSION / "candidates" / "*.parquet")))
    p = pd.concat([pd.read_parquet(f, filters=[("role", "=", "control"), ("label", "=", "prodigal_complete")])
                   for f in files], ignore_index=True)
    p["bin"] = label_bins(p.aa_len)
    p["low"], p["high"] = p.score < t80, p.score >= t90
    P = p.groupby("bin", sort=False).agg(genes=("score", "size"), below_t80=("low", "mean"),
                                         above_t90=("high", "mean"), median_score=("score", "median"))
    P = P.reindex([b for b in label_bins(pd.Series(BINS[:-1])).tolist() if b in P.index])
    P.loc["all"] = [len(p), p.low.mean(), p.high.mean(), p.score.median()]
    P.to_csv(out / "lowscore_population.tsv", sep="\t")
    Fz = p[p.low | p.high].assign(cls=lambda d: np.where(d.low, "low", "high")).groupby(["bin", "cls"])[FEATS + ["aa_len"]] \
        .median().unstack("cls")
    Fz.to_csv(out / "lowscore_features.tsv", sep="\t")
    B = p.groupby("biome").agg(genes=("score", "size"), below_t80=("low", "mean"),
                               below_t80_100aa_plus=("low", lambda s: s[p.loc[s.index, "aa_len"] >= 100].mean()))
    B.sort_values("below_t80").to_csv(out / "lowscore_biome.tsv", sep="\t")

    sx, qx = fullcov_sets(X)
    sel = pd.read_parquet(X / "selection.parquet", columns=["query_id", "arm", "aa_len", "score"])
    sel["full_struct"], sel["full_seq"] = sel.query_id.isin(sx), sel.query_id.isin(qx)
    sel["bin"] = label_bins(sel.aa_len)
    L = sel[sel.arm.isin(["prodigal_lo", "prodigal_hi"])].groupby(["bin", "arm"], sort=False) \
        .agg(queries=("full_struct", "size"), full_struct=("full_struct", "mean"), full_seq=("full_seq", "mean")) \
        .unstack("arm")
    L.to_csv(out / "lowscore_evidence_by_length.tsv", sep="\t")

    sm, qm = fullcov_sets(M)
    ms = pd.read_parquet(M / "selection.parquet", columns=["query_id", "arm", "score", "aa_len"])
    pr = ms[ms.arm == "prodigal"].assign(full_struct=lambda d: d.query_id.isin(sm))
    pr["score_bin"] = pd.cut(pr.score, [0, .2, .4, t80, t90, .95, 1.0001], right=False).astype(str)
    S = pr.groupby("score_bin").agg(genes=("full_struct", "size"), full_struct=("full_struct", "mean"),
                                    median_len=("aa_len", "median"))
    S.to_csv(out / "prodigal_by_score.tsv", sep="\t")

    rm = pd.read_csv(M / "evidence_rates.tsv", sep="\t")
    rx = pd.read_csv(X / "evidence_rates.tsv", sep="\t")
    A = pd.concat([rm, rx]).query("db == 'any'").pivot_table(index="arm", columns="mode",
                                                             values=["fullcov_rate", "hit_rate"])
    A.columns = [f"{a}_{b}" for a, b in A.columns]
    A = A.reindex([a for a in ARM_ORDER if a in A.index])
    A.to_csv(out / "all_arms_evidence.tsv", sep="\t")

    # fig19
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1, 1.1, 1]})
    ax = axes[0]
    Pp = P.drop(index="all")
    ax.bar(range(len(Pp)), Pp.below_t80, color=SLOTS[1], width=0.7)
    ax.set_xticks(range(len(Pp)), Pp.index, rotation=30)
    ax.set_ylim(0, 1)
    ax.set_xlabel("gene length (aa)")
    ax.set_ylabel("complete Prodigal genes below the 0.80 threshold")
    for i, v in enumerate(Pp.below_t80):
        ax.text(i, v + 0.015, f"{100 * v:.0f}%", ha="center", fontsize=8.5, color=INK2)
    ax = axes[1]
    order = [b for b in label_bins(pd.Series(BINS[:-1])).tolist() if b in L.index]
    L = L.reindex(order)
    xpos = np.arange(len(order))
    for arm, c, lab in (("prodigal_hi", SLOTS[0], "Prodigal genes, score ≥ 0.90 threshold"),
                        ("prodigal_lo", SLOTS[1], "Prodigal genes, score < 0.80 threshold")):
        v = L[("full_struct", arm)].to_numpy()
        ok = L[("queries", arm)].fillna(0).to_numpy() >= 100   # bins with too few genes are not drawn
        ax.plot(xpos[ok], v[ok], marker="o", ms=8, color=c, label=lab)
    ax.set_xticks(xpos, order)
    ax.set_ylim(0, 1)
    ax.set_xlabel("gene length (aa)")
    ax.set_ylabel("full-length structural hit rate")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[2]
    arms = [a for a in ["prodigal", "candidate", "frameshift", "antisense", "intergenic_lo",
                        "candidate_shuffled_3di"] if a in A.index]
    names = {"prodigal": "complete Prodigal genes", "candidate": "candidates", "frameshift": "frameshift shadows",
             "antisense": "antisense shadows", "intergenic_lo": "low-score intergenic",
             "candidate_shuffled_3di": "candidates, shuffled 3Di"}
    cols = {"prodigal": SLOTS[0], "candidate": SLOTS[1], "frameshift": SLOTS[4], "antisense": SLOTS[3],
            "intergenic_lo": SLOTS[2], "candidate_shuffled_3di": SLOTS[6]}
    v = A.loc[arms, "fullcov_rate_struct"]
    ax.barh(range(len(arms)), v, color=[cols[a] for a in arms], height=0.6)
    for i, x in enumerate(v):
        ax.text(x + 0.01, i, f"{100 * x:.3f}%" if x < 0.001 else f"{100 * x:.2f}%" if x < 0.01 else f"{100 * x:.1f}%",
                va="center", fontsize=9,
                color=INK2)
    ax.set_yticks(range(len(arms)), [names[a] for a in arms])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("full-length structural hit rate (length-matched arms)")
    fig.tight_layout()
    fig.savefig(out / "fig19_prodigal_lowscore.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig19_prodigal_lowscore.pdf", bbox_inches="tight")
    plt.close(fig)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for name, df in (("population", P), ("features", Fz), ("biome", B), ("evidence by length", L),
                         ("prodigal arm by score", S), ("all arms", A)):
            print(f"== {name}")
            print(df.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
