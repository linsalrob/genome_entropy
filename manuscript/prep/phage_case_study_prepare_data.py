#!/usr/bin/env python3
"""Reduce the phage case-study genome (NC_042013, Agrobacterium phage Atu_ph07) to the plotting data for figPS02.

Inputs are the case-study outputs of the all-genome Inphared ranking on
Setonix (Validation/Inphared/scripts/11_case_study.{py,slurm}; not in git):

  <case-dir>/NC_042013_cds.tsv       Pharokka + phold CDS
  <case-dir>/NC_042013_orfs.tsv      every ORF >= 90 aa, out-of-fold p (intrinsic model,
                                     genome-grouped CV), corrected label, context
  <case-dir>/NC_042013_foldseek.m8   Foldseek 3Di+AA search of the 28 candidates against
                                     PDB100, AFDB/Swiss-Prot, CATH50, BFVD and phold

Outputs (manuscript/data): figPS02_atu_ph07_{cds,orfs,hits}.tsv.gz and
phage_case_provenance.json. Nothing is re-estimated.

  phage_case_study_prepare_data.py --case-dir /scratch/pawsey1018/edwa0468/genome_entropy/work/phage_rank_all/case_study

Run from the repository root.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "manuscript/data"
G = "NC_042013"
FIELDS = ["db", "query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits",
          "taxid", "taxname", "theader"]
PROVENANCE = {}


def write(df, name, sources):
    df.to_csv(OUT / name, sep="\t", index=False, float_format="%.6g", compression={"method": "gzip", "mtime": 0})
    PROVENANCE[name] = {"rows": len(df),
                        "sources": {str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}}
    print(f"  {name}: {len(df)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case-dir", required=True, type=Path)
    a = ap.parse_args()
    d = a.case_dir
    cds_p, orf_p, hit_p = d / f"{G}_cds.tsv", d / f"{G}_orfs.tsv", d / f"{G}_foldseek.m8"

    cds = pd.read_csv(cds_p, sep="\t")
    write(cds[["cds_id", "g_start", "g_end", "strand", "aa_len", "product", "function", "record_length"]],
          "figPS02_atu_ph07_cds.tsv.gz", [cds_p])

    orf = pd.read_csv(orf_p, sep="\t")
    write(orf[["row", "orf_id", "g_start", "g_end", "strand", "protein_length", "three_di_entropy", "p",
               "in_genbank", "candidate", "context", "context_cds", "cds_overlap_frac"]],
          "figPS02_atu_ph07_orfs.tsv.gz", [orf_p])

    # best full-length (query and target coverage >= 0.8) structural hit per candidate and database
    h = pd.read_csv(hit_p, sep="\t", names=FIELDS)
    h = h[(h.qcov >= 0.8) & (h.tcov >= 0.8)].copy()
    h["row"] = h["query"].str.split("|").str[1].astype(int)
    h = h.sort_values(["bits", "target"], ascending=[False, True]).drop_duplicates(["row", "db"])
    h = h.merge(orf[["row", "orf_id"]], on="row")
    write(h[["orf_id", "row", "db", "target", "fident", "qcov", "tcov", "evalue", "bits", "taxname", "theader"]]
          .sort_values(["orf_id", "db"]), "figPS02_atu_ph07_hits.tsv.gz", [hit_p])
    (OUT / "phage_case_provenance.json").write_text(json.dumps(PROVENANCE, indent=1) + "\n")
    print("  phage_case_provenance.json")


if __name__ == "__main__":
    main()
