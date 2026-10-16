#!/usr/bin/env python3
"""Why did Prodigal not call the candidate genes? Prodigal's own scores and gene context.

Inputs (qc/candidates): contigs/part_*.{fa,gff,starts} (27_fetch_query_contigs.py:
Prodigal -p meta rerun on the query contigs, -s = every potential gene with its
scores), selection.parquet, fullcov_struct_best.parquet (25_candidate_figures.py)
and, if present, taxonomy/contig_taxonomy.tsv (29_contig_taxonomy.py).

First the rerun is checked: each Prodigal-arm query must again match a called
gene with the same stop, and no candidate may overlap a called gene.

For each query ORF the potential genes Prodigal listed with the same stop are
looked up (coordinates: Prodigal 1-based inclusive; ORFs 0-based half-open,
stop-to-stop, so the stop codon ends at `end` on + and starts at `start` + 1
on -). The best-scoring one gives Total, CodPot (coding), StrtSc (start),
RBS/upstream/type scores, start codon and RBS motif. Each candidate is put in
one class:
  no potential gene     nothing listed with that stop
  negative coding       best CodPot < 0 (codon statistics of the contig's
                        model say non-coding)
  start penalties       CodPot > 0 but best Total < 0 (start, RBS, upstream)
  positive, not called  best Total > 0 yet not in the final gene set
Context: contig length and Prodigal's metagenomic model (GC, genetic code,
SD use), ORF GC and GC3 against the contig, distance to contig ends and to the
nearest called gene, called-gene density.

Groups compared: bacterial candidates (best full-length structural target is
bacterial), other candidates with a hit, candidates without a hit, complete
Prodigal genes with a bacterial hit, low-score intergenic ORFs.

Writes skip_features.parquet, skip_reasons.tsv, skip_feature_summary.tsv,
skip_model_mismatch.tsv and fig18_prodigal_skip to --out.

  30_prodigal_skip.py --dir qc/candidates --out qc/candidates/skip
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUPS = ["Prodigal gene, bacterial hit", "candidate, bacterial hit", "candidate, other hit",
          "candidate, no hit", "low-score intergenic"]
GROUP_COLOR = dict(zip(GROUPS, [SLOTS[0], SLOTS[1], SLOTS[4], SLOTS[3], SLOTS[2]]))
REASONS = ["positive, not called", "start penalties", "negative coding", "no potential gene"]
REASON_COLOR = dict(zip(REASONS, [SLOTS[0], SLOTS[3], SLOTS[1], GRID]))
COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def style():
    sns.set_theme(context="paper", style="ticks", font_scale=1.15, rc={
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": "white",
        "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False})


def read_fasta(path: str) -> dict[str, str]:
    seqs, name, parts = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(parts)
            name, parts = line[1:].split(None, 1)[0], []
        else:
            parts.append(line.strip())
    if name:
        seqs[name] = "".join(parts)
    return seqs


def read_gff(path: str):
    """Model per contig and called genes (1-based inclusive)."""
    model, genes = {}, defaultdict(list)
    cur = None
    for line in open(path):
        if line.startswith("# Sequence Data:"):
            cur = re.search(r'seqhdr="([^"]+)"', line).group(1).split()[0]
        elif line.startswith("# Model Data:"):
            kv = dict(x.split("=", 1) for x in line.split(":", 1)[1].strip().rstrip(";").split(";"))
            model[cur] = {"model": kv["model"].strip('"'), "model_gc": float(kv["gc_cont"]),
                          "transl_table": int(kv["transl_table"]), "uses_sd": int(kv["uses_sd"])}
        elif line and not line.startswith("#"):
            f = line.rstrip("\n").split("\t")
            if len(f) == 9:
                genes[f[0]].append((int(f[3]), int(f[4]), f[6]))
    return model, genes


def read_starts(path: str, wanted: dict[str, set]) -> dict[tuple, dict]:
    """Best-Total potential gene per (contig, strand, stop position) that is wanted."""
    best: dict[tuple, dict] = {}
    cur, cols = None, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("# Sequence Data:"):
                cur = re.search(r'seqhdr="([^"]+)"', line).group(1).split()[0]
                continue
            if line.startswith("Beg\t"):
                cols = line.rstrip("\n").split("\t")
                continue
            if not line.strip() or line.startswith("#") or cur not in wanted:
                continue
            f = line.rstrip("\n").split("\t")
            beg, end, std = int(f[0]), int(f[1]), f[2]
            key = (cur, std, end if std == "+" else beg)
            if key not in wanted[cur]:
                continue
            row = dict(zip(cols, f))
            tot = float(row["Total"])
            b = best.get(key)
            if b is None or tot > b["Total"]:
                best[key] = {"Total": tot, "CodPot": float(row["CodPot"]), "StrtSc": float(row["StrtSc"]),
                             "RBSScr": float(row["RBSScr"]), "UpsScr": float(row["UpsScr"]),
                             "TypeScr": float(row["TypeScr"]), "Codon": row["Codon"], "RBSMot": row["RBSMot"],
                             "n_potential": (b["n_potential"] + 1) if b else 1}
            else:
                b["n_potential"] += 1
    return best


def gc(s: str) -> float:
    s = s.upper()
    n = sum(s.count(x) for x in "ACGT")
    return (s.count("G") + s.count("C")) / n if n else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    sel = pd.read_parquet(D / "selection.parquet",
                          columns=["query_id", "arm", "accession", "contig_n", "contig_len", "start", "end",
                                   "strand", "stop", "five_open", "trunc", "aa_len", "biome", "score"])
    sel = sel[sel.arm.isin(["candidate", "prodigal", "intergenic_lo"])].copy()
    sel["contig"] = sel.accession + "_" + sel.contig_n.astype(str)
    sel["stop_pos"] = np.where(sel.strand == "+", sel.end, sel.start + 1)
    wanted = defaultdict(set)
    for c, s, p in zip(sel.contig, sel.strand, sel.stop_pos):
        wanted[c].add((c, s, int(p)))

    seqs, model, genes, best = {}, {}, defaultdict(list), {}
    parts = sorted(glob.glob(str(D / "contigs" / "part_*.gff")))
    for gp in parts:
        base = gp[:-4]
        seqs.update(read_fasta(base + ".fa"))
        m, g = read_gff(gp)
        model.update(m)
        genes.update(g)
        best.update(read_starts(base + ".starts", wanted))
    print(f"{len(parts)} parts, {len(seqs):,} contigs, {sum(map(len, genes.values())):,} called genes")
    sel = sel[sel.contig.isin(seqs)].copy()

    rows = []
    for r in sel.itertuples(index=False):
        g = genes.get(r.contig, [])
        lo, hi = r.start + 1, r.end            # ORF span, 1-based inclusive
        called_same_stop = any((s == r.strand) and ((e if s == "+" else b) == r.stop_pos) for b, e, s in g)
        overlaps = any(b <= hi and e >= lo for b, e, s in g)
        dists = [max(b - hi, lo - e, 0) for b, e, s in g if not (b <= hi and e >= lo)]
        seq = seqs[r.contig][r.start:r.end]
        if r.strand == "-":
            seq = seq.translate(COMP)[::-1]
        cds = seq[: 3 * r.aa_len]
        x = {"query_id": r.query_id, "arm": r.arm, "contig": r.contig, "biome": r.biome, "trunc": r.trunc,
             "aa_len": r.aa_len, "contig_len": r.contig_len, "called_same_stop": called_same_stop,
             "overlaps_called_gene": overlaps,
             "nearest_called_gene_bp": min(dists) if dists else np.nan,
             "called_genes_on_contig": len(g), "called_genes_per_kb": 1000 * len(g) / r.contig_len,
             "dist_to_contig_end": min(r.start, r.contig_len - r.end),
             "orf_gc": gc(cds), "orf_gc3": gc(cds[2::3]), "contig_gc": gc(seqs[r.contig])}
        x.update(model.get(r.contig, {}))
        x.update(best.get((r.contig, r.strand, int(r.stop_pos)), {}))
        rows.append(x)
    F = pd.DataFrame(rows)
    F["orf_minus_model_gc"] = F.orf_gc - F.model_gc / 100

    # Rerun check
    chk = {"prodigal_arm_called_again": float(F[F.arm == "prodigal"].called_same_stop.mean()),
           "candidates_overlapping_a_called_gene": float(F[F.arm == "candidate"].overlaps_called_gene.mean()),
           "candidates_called_same_stop": float(F[F.arm == "candidate"].called_same_stop.mean())}
    print("rerun check:", {k: round(v, 4) for k, v in chk.items()})

    bh = pd.read_parquet(D / "fullcov_struct_best.parquet", columns=["query", "superkingdom"])
    F = F.merge(bh.rename(columns={"query": "query_id", "superkingdom": "hit_superkingdom"}), on="query_id",
                how="left")
    tax = D / "taxonomy" / "contig_taxonomy.tsv"
    if tax.exists():
        t = pd.read_csv(tax, sep="\t", usecols=["contig", "contig_group"])
        F = F.merge(t, on="contig", how="left")
    F["group"] = np.select(
        [(F.arm == "prodigal") & (F.hit_superkingdom == "Bacteria"),
         (F.arm == "candidate") & (F.hit_superkingdom == "Bacteria"),
         (F.arm == "candidate") & F.hit_superkingdom.notna(),
         (F.arm == "candidate"),
         (F.arm == "intergenic_lo")], GROUPS, default="other")
    F["reason"] = np.select(
        [F.Total.isna(), F.CodPot < 0, F.Total < 0], ["no potential gene", "negative coding", "start penalties"],
        default="positive, not called")
    F.loc[F.called_same_stop, "reason"] = "called"
    F.to_parquet(out / "skip_features.parquet", index=False)

    R = pd.crosstab(F.group, F.reason, normalize="index").reindex(GROUPS)
    R["queries"] = F.groupby("group").size()
    R.to_csv(out / "skip_reasons.tsv", sep="\t")
    feats = ["Total", "CodPot", "StrtSc", "RBSScr", "UpsScr", "orf_gc", "orf_gc3", "contig_gc",
             "orf_minus_model_gc", "contig_len", "dist_to_contig_end", "nearest_called_gene_bp",
             "called_genes_per_kb", "aa_len"]
    Sm = F[F.group.isin(GROUPS)].groupby("group")[feats].median().reindex(GROUPS).T
    frac = F[F.group.isin(GROUPS)].assign(
        start_ATG=F.Codon == "ATG", start_GTG=F.Codon == "GTG", start_TTG=F.Codon == "TTG",
        rbs_none=F.RBSMot.isin(["None", "none"]), uses_sd=F.uses_sd == 1, code_4=F.transl_table == 4,
        contig_lt_2kb=F.contig_len < 2000, no_called_gene_on_contig=F.called_genes_on_contig == 0,
    ).groupby("group")[["start_ATG", "start_GTG", "start_TTG", "rbs_none", "uses_sd", "code_4", "contig_lt_2kb",
                         "no_called_gene_on_contig"]].mean().reindex(GROUPS).T
    pd.concat([Sm, frac]).to_csv(out / "skip_feature_summary.tsv", sep="\t")
    # Does the contig's chosen model fit the candidate? GC distance of ORF from model.
    M = F[F.group.isin(GROUPS)].groupby(["group", pd.cut(F.orf_minus_model_gc.abs(), [0, .05, .1, .15, 1],
                                                             include_lowest=True)], observed=True).size() \
        .unstack(fill_value=0)
    M.div(M.sum(axis=1), axis=0).reindex(GROUPS).to_csv(out / "skip_model_mismatch.tsv", sep="\t")

    # fig18
    sub = F[F.group.isin(GROUPS)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, lab, lim) in zip(axes.flat[:3], (("CodPot", "Prodigal coding score (best start)", (-30, 60)),
                                                   ("Total", "Prodigal total score (best start)", (-40, 60)),
                                                   ("orf_minus_model_gc", "ORF GC − Prodigal model GC", (-.3, .3)))):
        for gname in GROUPS:
            v = sub.loc[sub.group == gname, col].dropna().clip(*lim)
            if len(v) > 20:
                sns.kdeplot(v, ax=ax, color=GROUP_COLOR[gname], label=f"{gname} (n={len(v):,})", lw=2,
                            common_norm=False, cut=0)
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.set_xlabel(lab)
        ax.set_xlim(*lim)
    axes[0, 0].legend(fontsize=9, loc="upper right")
    ax = axes[1, 1]
    cg = [g for g in GROUPS if g != "Prodigal gene, bacterial hit"]
    Rr = pd.crosstab(sub.group, sub.reason, normalize="index").reindex(index=cg, columns=REASONS, fill_value=0)
    left = np.zeros(len(cg))
    for rs in REASONS:
        ax.barh(range(len(cg)), Rr[rs], left=left, color=REASON_COLOR[rs], label=rs, height=0.6,
                edgecolor="white", linewidth=2)
        left += Rr[rs].to_numpy()
    ax.set_yticks(range(len(cg)), cg)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("why Prodigal did not call the ORF")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig18_prodigal_skip.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig18_prodigal_skip.pdf", bbox_inches="tight")
    plt.close(fig)

    with pd.option_context("display.width", 240, "display.max_columns", 20):
        print(R.round(3).to_string())
        print(pd.concat([Sm, frac]).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
