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


PREP = {"G04": prep_g04, "G05": prep_g05}


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
