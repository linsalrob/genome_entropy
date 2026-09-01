#!/usr/bin/env python3
"""Is the candidate burden biology, or a property of the assembly?

96,701 of 96,875 annotated bacterial representatives (99.8%) carry at least
one candidate missed gene, median 26 per genome, maximum 5,763. That
prevalence is too universal to read as "annotation misses genes everywhere"
without first asking what else varies with it. If burden tracks assembly
fragmentation, contamination or coding density rather than taxonomy, then a
hit-rate measured on these candidates is partly measuring assembly quality,
and the Foldseek pilot would be interpreted wrongly (issue #92).

Joins the per-genome candidate counts against GTDB's own metadata --
genome size, GC, coding density, contig count, N50, CheckM2 completeness and
contamination, assembly level, taxonomy -- and reports Spearman correlations
plus a per-covariate figure. Spearman rather than Pearson because burden is
heavily right-skewed and several covariates are bounded percentages;
monotone association is the claim, linearity is not.

Correlation here is descriptive. These covariates are themselves correlated
(fragmented assemblies are also less complete), so a coefficient is a flag
for "look at this", not an effect size.

  16_candidate_burden.py --domain bac
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

GD = "/g/data/ob80/re3494/gtdb_entropy"
META = {
    "bac": "/g/data/ob80/re3494/Projects/genome_entropy/claude/gtdb_metadata/bac120_metadata.tsv.gz",
    "arc": "/g/data/ob80/re3494/Projects/genome_entropy/claude/gtdb_metadata/ar53_metadata.tsv.gz",
}
META_COLS = ["accession", "genome_size", "gc_percentage", "coding_density",
             "contig_count", "n50_contigs", "checkm2_completeness",
             "checkm2_contamination", "ncbi_assembly_level", "gtdb_taxonomy",
             "protein_count"]
COVARIATES = ["genome_size_mb", "gc_percentage", "coding_density",
              "contig_count", "n50_contigs", "checkm2_completeness",
              "checkm2_contamination", "n_orfs", "n_cds", "cds_per_mb",
              "orfs_per_mb", "frac_orfs_in_genbank"]


def load_metadata(path):
    """GTDB accessions are prefixed RS_ or GB_; ours are not."""
    have = pd.read_csv(path, sep="\t", nrows=0).columns
    use = [c for c in META_COLS if c in have]
    missing = set(META_COLS) - set(use)
    if missing:
        print(f"note: metadata lacks {sorted(missing)}", file=sys.stderr)
    md = pd.read_csv(path, sep="\t", usecols=use, low_memory=False)
    md["genome"] = md.accession.str.replace(r"^(RS_|GB_)", "", regex=True)
    md["genome_size_mb"] = md.genome_size / 1e6
    if "gtdb_taxonomy" in md.columns:
        md["phylum"] = md.gtdb_taxonomy.str.extract(r"p__([^;]*)")
    return md.drop(columns=["accession"])


def spearman(x, y):
    """Spearman rho on complete pairs, with the n it was computed on."""
    ok = x.notna() & y.notna()
    if ok.sum() < 50 or x[ok].nunique() < 3:
        return np.nan, int(ok.sum())
    return float(x[ok].rank().corr(y[ok].rank())), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", default="bac", choices=("bac", "arc"))
    ap.add_argument("--out-dir", default=f"{GD}/missed_genes")
    ap.add_argument("--figure", default=None,
                    help="default: figures/candidate_burden_<domain>.png")
    args = ap.parse_args()

    cand_path = f"{GD}/missed_genes/{args.domain}/candidates_{args.domain}.tsv.gz"
    ann_path = f"{GD}/genome_annotation_status_{args.domain}.tsv"
    for path in (cand_path, ann_path, META[args.domain]):
        if not os.path.exists(path):
            sys.exit(f"ERROR: missing {path}")

    # Candidate counts per genome. Genomes with zero candidates matter as
    # much as the rest -- they are the comparison -- so the annotation-status
    # table, not the candidate table, defines the row set.
    cand = pd.read_csv(cand_path, sep="\t",
                       usecols=["genome", "aa_length", "three_di_entropy"])
    per = cand.groupby("genome").agg(
        n_candidates=("genome", "size"),
        median_cand_aa=("aa_length", "median"),
        median_cand_3di=("three_di_entropy", "median")).reset_index()

    ann = pd.read_csv(ann_path, sep="\t")
    ann = ann[ann.annotated.astype(str) == "True"].copy()
    ann = ann.rename(columns={"n_orfs_in_genbank": "n_cds"})
    df = ann.merge(per, on="genome", how="left")
    df["n_candidates"] = df.n_candidates.fillna(0).astype(int)

    md = load_metadata(META[args.domain])
    before = len(df)
    df = df.merge(md, on="genome", how="left")
    n_nometa = int(df.genome_size.isna().sum())

    df["frac_orfs_in_genbank"] = df.n_cds / df.n_orfs
    df["cand_per_mb"] = df.n_candidates / df.genome_size_mb
    df["cds_per_mb"] = df.n_cds / df.genome_size_mb
    df["orfs_per_mb"] = df.n_orfs / df.genome_size_mb
    df["cand_per_cds"] = df.n_candidates / df.n_cds.replace(0, np.nan)

    print(f"annotated {args.domain} genomes : {before:,}")
    print(f"  matched to GTDB metadata  : {before - n_nometa:,}"
          + (f"  ({n_nometa:,} unmatched)" if n_nometa else ""))
    print(f"  carrying >=1 candidate    : {(df.n_candidates > 0).sum():,} "
          f"({(df.n_candidates > 0).mean()*100:.1f}%)")
    print(f"  candidates per genome     : median {df.n_candidates.median():.0f}, "
          f"mean {df.n_candidates.mean():.1f}, max {df.n_candidates.max():,}")
    print(f"  candidates per Mb         : median {df.cand_per_mb.median():.1f}")
    print(f"  candidates per annotated CDS: median "
          f"{df.cand_per_cds.median()*100:.2f}%\n")

    print("Spearman rho against candidate burden "
          "(rho > 0 means more candidates):")
    print(f"  {'covariate':<26}{'per genome':>12}{'per Mb':>10}{'n':>10}")
    rows = []
    for cov in COVARIATES:
        if cov not in df.columns:
            continue
        r_abs, n = spearman(df[cov], df.n_candidates)
        r_dens, _ = spearman(df[cov], df.cand_per_mb)
        rows.append((cov, r_abs, r_dens, n))
        print(f"  {cov:<26}{r_abs:>12.3f}{r_dens:>10.3f}{n:>10,}")

    if "ncbi_assembly_level" in df.columns:
        print("\nBy assembly level:")
        g = df.groupby("ncbi_assembly_level").agg(
            genomes=("genome", "size"),
            median_cand=("n_candidates", "median"),
            median_per_mb=("cand_per_mb", "median"),
            median_contigs=("contig_count", "median"))
        print(g.sort_values("genomes", ascending=False).to_string())

    if "phylum" in df.columns:
        print("\nTop 10 phyla by genome count:")
        g = df.groupby("phylum").agg(
            genomes=("genome", "size"),
            median_cand=("n_candidates", "median"),
            median_per_mb=("cand_per_mb", "median"))
        print(g.sort_values("genomes", ascending=False).head(10).to_string())

    print("\nThe 5 genomes with the most candidates:")
    cols = [c for c in ("genome", "n_candidates", "cand_per_mb", "n_orfs",
                        "n_cds", "genome_size_mb", "gc_percentage",
                        "coding_density", "contig_count", "n50_contigs",
                        "checkm2_completeness", "checkm2_contamination",
                        "ncbi_assembly_level", "phylum") if c in df.columns]
    print(df.nlargest(5, "n_candidates")[cols].to_string(index=False))

    out = f"{args.out_dir}/candidate_burden_{args.domain}.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"\nper-genome burden table -> {out}")

    make_figure(df, rows, args.domain,
                args.figure or f"{GD}/figures/candidate_burden_{args.domain}.png")
    return 0


def make_figure(df, rows, domain, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    POINT = "#2a78d6"
    panels = [c for c in ("contig_count", "checkm2_completeness",
                          "checkm2_contamination", "coding_density",
                          "gc_percentage", "genome_size_mb")
              if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(SURFACE)
    rho = {r[0]: r[2] for r in rows}

    for ax, cov in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        d = df[[cov, "cand_per_mb"]].dropna()
        ax.hexbin(d[cov], d.cand_per_mb, gridsize=60, bins="log",
                  cmap="Blues", mincnt=1, linewidths=0)
        if cov == "contig_count":
            ax.set_xscale("log")
        ax.set_xlabel(cov, fontsize=10, color=INK)
        ax.set_ylabel("candidates per Mb", fontsize=10, color=INK)
        r = rho.get(cov, float("nan"))
        ax.set_title(f"{cov}   Spearman rho = {r:.3f}", fontsize=11,
                     color=INK, loc="left", pad=8)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        # A few genomes have enormous burden and would otherwise set the
        # scale for everything; clip the view, not the data.
        ax.set_ylim(0, float(np.nanpercentile(d.cand_per_mb, 99.5)))

    word = {"bac": "bacterial", "arc": "archaeal"}.get(domain, domain)
    fig.suptitle(f"Candidate missed-gene burden against assembly and genome "
                 f"properties, {word} representatives",
                 fontsize=14, color=INK, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.945,
             "Burden is candidates per Mb, one hexbin per genome, log counts. "
             "y-axis clipped at the 99.5th percentile so a few extreme genomes "
             "do not set the scale. If burden tracked assembly quality rather "
             "than biology, these panels are where it would show.",
             fontsize=9.5, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"figure -> {out}")


if __name__ == "__main__":
    sys.exit(main())
