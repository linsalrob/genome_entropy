#!/usr/bin/env python3
"""Reduce the GTDB r232 analysis outputs to the plotting-ready tables in ../data/.

Runs on NCI Gadi, where the full-scale outputs live. Nothing here re-derives a
published number: every value is either copied from a committed summary table
(Validation/gtdb/matched_comparator/, written by 41_comparator_summary.py) or is
a plain sum/mean over the per-chunk or per-ORF tables the analysis stages wrote.
The notebooks in ../notebooks/ read only the files this script writes, so the
figures can be redrawn anywhere without the HPC filesystem.

  gtdb_prepare_data.py --repo <genome_entropy checkout at the recorded commit> \
                       --gtdb-dir /g/data/ob80/re3494/gtdb_entropy \
                       [--only G04 G05]

Inputs and their sha256 are printed, so the provenance cell of each notebook
can be checked against a rerun.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
# Deterministic gzip: no embedded timestamp, so a rerun is byte-identical.
GZ = {"method": "gzip", "mtime": 0}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def note(path):
    print(f"  in   {path}  sha256={sha256(path)}")


def write(df, name):
    p = OUT / name
    df.to_csv(p, sep="\t", index=False, compression=GZ)
    print(f"  out  {p.name}  rows={len(df)}  sha256={sha256(p)}")


# --------------------------------------------------------------------------
FUNNEL_STEPS = [("all ORF calls", "orfs"),
                ("in annotated genomes", "orfs_annotated_genomes"),
                ("unmatched by GenBank", "unmatched"),
                ("3Di entropy ≥ 2.5", "unmatched_hi"),
                ("not a shadow (candidates)", "candidates")]


def prep_g04(repo, gd):
    """Missed-gene funnel and the v7 matched-comparator summary."""
    rows = []
    for dom, sub in (("bacteria", "bac"), ("archaea", "arc")):
        stats = sorted((gd / "missed_genes" / sub).glob("*.stats.tsv"))
        if not stats:
            sys.exit(f"no *.stats.tsv under {gd}/missed_genes/{sub}")
        print(f"  in   {gd}/missed_genes/{sub}/*.stats.tsv  ({len(stats)} chunks)")
        f = pd.concat([pd.read_csv(p, sep="\t") for p in stats],
                      ignore_index=True)
        for i, (label, col) in enumerate(FUNNEL_STEPS):
            rows.append(dict(domain=dom, step_order=i, step=label,
                             column=col, n_orfs=int(f[col].sum()),
                             n_chunks=len(stats)))
    write(pd.DataFrame(rows), "figG04_funnel.tsv.gz")

    write(read_matched(repo), "figG04_matched_comparator.tsv.gz")


def read_matched(repo):
    frames = []
    for dom, tag in (("bacteria", "bac"), ("archaea", "arc")):
        p = repo / "Validation/gtdb/matched_comparator" / f"{tag}_v7_matched.tsv"
        note(p)
        d = pd.read_csv(p, sep="\t")
        d.insert(0, "domain_name", dom)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


FRAME_CLASSES = ["same strand, same frame", "same strand, frameshift",
                 "same strand, frame undefined", "opposite strand"]


def prep_g05(repo, gd):
    """Prodigal coincidence: per-arm (matched) and per frame class (shadows)."""
    m = read_matched(repo)
    write(m[m.metric.str.startswith("prodigal_") | m.metric.isin(
        ["n_candidate", "n_comparator_draws", "n_comparator_distinct",
         "n_annotated"])], "figG05_prodigal_by_arm.tsv.gz")

    rows = []
    for dom, p in (("bacteria", gd / "missed_genes/full_bac/prodigal_bac4.coincidence.tsv.gz"),
                   ("archaea", gd / "missed_genes/pilot_arc/prodigal_arc4.coincidence.tsv.gz")):
        note(p)
        d = pd.read_csv(p, sep="\t", low_memory=False,
                        usecols=["group", "cds_frame_class", "coincides"])
        sh = d[d.group == "shadow_hi"]
        for c in FRAME_CLASSES:
            s = sh[sh.cds_frame_class == c].coincides.astype(bool)
            rows.append(dict(domain=dom, arm="shadow_hi (v6, unmatched)",
                             cds_frame_class=c, n=int(s.size),
                             n_coincides=int(s.sum())))
    write(pd.DataFrame(rows), "figG05_prodigal_frame_class.tsv.gz")


DOMAINS = (("bacteria", "bac"), ("archaea", "arc"))
# 3Di bins are (lo, hi], so an ORF exactly at a log2(k) ceiling lands in the bin
# that ENDS on the ceiling and the edge stays sharp. Width 0.005 makes 1, log2(3)
# (= 316.99 bins) and 2 fall on or within 0.0001 of a bin edge.
TDI_W, TDI_MAX = 0.005, 4.4
PROT_W, PROT_LO, PROT_HI = 0.01, 2.35, 4.35
# Coarser 2-D grid for the landscape panels; 3Di edges offset so log2(3) is one.
GRID_P = np.round(np.arange(2.35, 4.35 + 1e-9, 0.02), 4)
GRID_T = np.round(np.arange(-0.015, 4.40 + 1e-9, 0.02), 4)


def right_closed_bin(x, width):
    """Index i such that x is in (i*width, (i+1)*width]; x == 0 goes to bin 0."""
    return np.maximum(np.ceil(np.round(x / width, 9)).astype(np.int64) - 1, 0)


def prep_g01(repo, gd):
    """Entropy landscape: binned figure sample, plus full-population band shares."""
    joint, marg = [], []
    for dom, tag in DOMAINS:
        p = gd / f"figure_samples/sample_{tag}.tsv.gz"
        note(p)
        # Same restriction as figstyle.load_sample(annotated_only=True): the
        # sampler now writes matcher_matched_a_cds; samples drawn before the
        # rename (including the committed figure samples) carry genome_annotated.
        cols = pd.read_csv(p, sep="\t", nrows=0).columns
        flag = next((c for c in ("matcher_matched_a_cds", "genome_annotated") if c in cols), None)
        if flag is None:
            sys.exit(f"{p}: no annotation flag column; re-run 08b_sample_for_figures.pbs")
        d = pd.read_csv(p, sep="\t", dtype={"in_genbank": str, flag: str},
                        usecols=["in_genbank", "protein_entropy", "three_di_entropy", flag])
        n_all = len(d)
        d = d[d[flag] == "True"]
        print(f"    {dom}: {n_all:,} sampled rows, {len(d):,} in CDS-bearing genomes")
        for cls, sub in (("matched", d[d.in_genbank == "True"]),
                         ("unmatched", d[d.in_genbank == "False"])):
            h, _, _ = np.histogram2d(sub.protein_entropy, sub.three_di_entropy,
                                     bins=[GRID_P, GRID_T])
            ii, jj = np.nonzero(h)
            joint.append(pd.DataFrame(dict(
                domain=dom, cls=cls, protein_lo=GRID_P[ii], three_di_lo=GRID_T[jj],
                n=h[ii, jj].astype(np.int64))))
            outside = int(len(sub) - h.sum())
            t = right_closed_bin(sub.three_di_entropy.to_numpy(), TDI_W)
            tc = np.bincount(t, minlength=int(round(TDI_MAX / TDI_W)))
            pr = sub.protein_entropy.to_numpy()
            pin = pr[(pr > PROT_LO) & (pr <= PROT_HI)]
            pc = np.bincount(right_closed_bin(pin - PROT_LO, PROT_W),
                             minlength=int(round((PROT_HI - PROT_LO) / PROT_W)))
            for alphabet, counts, lo0, w in (("three_di", tc, 0.0, TDI_W),
                                             ("protein", pc, PROT_LO, PROT_W)):
                marg.append(pd.DataFrame(dict(
                    domain=dom, cls=cls, alphabet=alphabet,
                    bin_lo=np.round(lo0 + np.arange(len(counts)) * w, 4),
                    bin_hi=np.round(lo0 + (np.arange(len(counts)) + 1) * w, 4),
                    n=counts, n_class=len(sub),
                    n_outside_joint_grid=outside)))
    write(pd.concat(joint, ignore_index=True), "figG01_joint_sample.tsv.gz")
    write(pd.concat(marg, ignore_index=True), "figG01_marginals_sample.tsv.gz")

    bands = []
    for dom, tag in DOMAINS:
        p = gd / f"population_summary/population_entropy_{tag}_bands.tsv"
        note(p)
        b = pd.read_csv(p, sep="\t")
        b = b[b.group.str.startswith("CDS-bearing genomes: ")
              & b.metric.isin(["three_di", "protein"])].copy()
        b.insert(0, "domain_name", dom)
        bands.append(b)
    write(pd.concat(bands, ignore_index=True), "figG01_population_bands.tsv.gz")


ALPHABETS = ("protein", "three_di", "twelve_state")
MATCHED_STRATUM = 3


def prep_g02(repo, gd):
    """Estimator bias vs biology (mean curves) and calibration of the references."""
    nd = gd / "entropy_null"
    note(nd / "entropy_null_quantiles.tsv")
    null = pd.read_csv(nd / "entropy_null_quantiles.tsv", sep="\t",
                       usecols=["domain", "alphabet", "aa_length", "mean"])
    curves = []
    for dom, tag in DOMAINS:
        p = nd / f"empirical/length_entropy_{tag}.tsv"
        note(p)
        e = pd.read_csv(p, sep="\t", usecols=["stratum", "alphabet", "length_lo",
                                              "length_hi", "n", "mean"])
        # As 38_bias_vs_biology.py: matched stratum, 1 aa bins only.
        e = e[(e.stratum == MATCHED_STRATUM) & (e.length_lo == e.length_hi)]
        for a in ALPHABETS:
            ea = e[e.alphabet == a].sort_values("length_lo")
            r = null[(null.domain == tag) & (null.alphabet == a)].sort_values("aa_length")
            L = ea.length_lo.to_numpy()
            curves.append(pd.DataFrame(dict(
                domain=dom, alphabet=a, aa_length=L.astype(int), n=ea.n.to_numpy(),
                observed_mean=ea["mean"].to_numpy(),
                bias_mean=np.interp(L, r.aa_length.to_numpy(), r["mean"].to_numpy()))))
    write(pd.concat(curves, ignore_index=True), "figG02_mean_curves.tsv.gz")

    zedges = np.linspace(-15, 8, 231)
    dedges = np.linspace(0, 1, 11)
    zh, dec, summ = [], [], []
    for dom, tag in DOMAINS:
        for a in ("protein", "three_di"):
            p = nd / f"calibration/scores_{tag}_{a}.tsv.gz"
            note(p)
            s = pd.read_csv(p, sep="\t", usecols=["z", "percentile", "emp_z",
                                                  "emp_percentile"])
            for ref, zc, pc in (("iid", "z", "percentile"),
                                ("empirical", "emp_z", "emp_percentile")):
                z = s[zc].to_numpy()
                h = np.histogram(z, bins=zedges)[0]
                zh.append(pd.DataFrame(dict(domain=dom, alphabet=a, reference=ref,
                                            z_lo=np.round(zedges[:-1], 4),
                                            z_hi=np.round(zedges[1:], 4), n=h)))
                o = np.histogram(s[pc].clip(0, 1), bins=dedges)[0]
                dec.append(pd.DataFrame(dict(domain=dom, alphabet=a, reference=ref,
                                             decile=np.arange(1, 11), n=o)))
                summ.append(dict(domain=dom, alphabet=a, reference=ref, n=len(z),
                                 z_mean=z.mean(), z_sd=z.std(ddof=1),
                                 n_off_axis=int(((z < zedges[0]) | (z > zedges[-1])).sum())))
    write(pd.concat(zh, ignore_index=True), "figGS02_z_hist.tsv.gz")
    write(pd.concat(dec, ignore_index=True), "figGS02_deciles.tsv.gz")
    write(pd.DataFrame(summ), "figGS02_summary.tsv.gz")


PREP = {"G01": prep_g01, "G02": prep_g02, "G04": prep_g04, "G05": prep_g05}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", type=Path, required=True,
                    help="genome_entropy checkout holding Validation/gtdb")
    ap.add_argument("--gtdb-dir", type=Path,
                    default=Path("/g/data/ob80/re3494/gtdb_entropy"))
    ap.add_argument("--only", nargs="+", choices=PREP, default=list(PREP))
    a = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(a.repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"genome_entropy commit {commit or 'UNKNOWN'}")
    OUT.mkdir(parents=True, exist_ok=True)
    for k in a.only:
        print(k)
        PREP[k](a.repo, a.gtdb_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
