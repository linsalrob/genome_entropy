#!/usr/bin/env python3
"""Reduce the phage analyses to the plotting data for figP01 and figPS01.

Inputs are the merged result tables in this repository (genome_entropy #108,
#110) plus, for the OM869708 panel, the per-ORF tables of the corrected phage
ranking run on Setonix (not in git: 28 GB). Outputs are the small TSVs in
manuscript/data/ that the notebooks read. Nothing is re-estimated: every
value is copied from a table written by the analysis that computed it.

  phage_prepare_data.py --phage-rank /scratch/pawsey1018/edwa0468/genome_entropy/work/phage_rank \
                        [--blast <om869708_blastp_nr.tsv>]

Run from the repository root.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MI = REPO / "Validation/mutual_information/tables"
IR = REPO / "Validation/Inphared/tables"
OUT = REPO / "manuscript/data"

FEATURE_LABEL = {
    "dna_entropy": "DNA entropy",
    "protein_entropy": "protein entropy",
    "three_di_entropy": "3Di entropy",
    "twelve_state_entropy": "12-state entropy",
    "three_di_twelve_state_mutual_information": "3Di–12-state MI",
    "protein_length": "length",
}


def write(df, name, sources):
    path = OUT / name
    df.to_csv(path, sep="\t", index=False, float_format="%.6g")
    PROVENANCE[name] = {
        "rows": len(df),
        "sources": {str(s.relative_to(REPO)) if s.is_relative_to(REPO) else str(s):
                    hashlib.sha256(s.read_bytes()).hexdigest()
                    for s in sources},
    }
    print(f"  {name}: {len(df)} rows")


def mi_panel():
    src = MI / "mi_single_and_pairs.tsv"
    cond = MI / "conditional_mi_given_length.tsv"
    d = pd.read_csv(src, sep="\t")
    d = d[d.bins_per_axis == 20] if (d.bins_per_axis == 20).any() else d
    keep = d[(d.kind == "single") | (d.features == "protein_entropy+three_di_entropy")]
    out = keep[["features", "kind", "bins_per_axis", "mi_balanced_bits",
                "boot_lo_bits", "boot_hi_bits", "null_p99_bits",
                "mi_full_bits"]].copy()
    c = pd.read_csv(cond, sep="\t")[["variable", "I_X_label",
                                     "I_X_label_given_len20",
                                     "fraction_of_I_X_retained_given_len20"]]
    out = out.merge(c, left_on="features", right_on="variable", how="left").drop(columns="variable")
    out["label"] = out.features.map(lambda f: " + ".join(FEATURE_LABEL[x] for x in f.split("+")))
    write(out, "figP01_mutual_information.tsv.gz", [src, cond])


def model_panel():
    src = IR / "model_metrics_pooled.tsv"
    d = pd.read_csv(src, sep="\t")
    write(d, "figP01_ranking_models.tsv.gz", [src])


def foldseek_panel():
    src = IR / "evidence_rates.tsv"
    d = pd.read_csv(src, sep="\t")
    d = d[(d["mode"] == "struct") & d.db.isin(["any", "non_bfvd"])]
    write(d, "figP01_foldseek_rates.tsv.gz", [src])


def label_audit():
    src = MI / "label_legacy_vs_v2.tsv"
    imp = IR / "model_importance.tsv"
    write(pd.read_csv(src, sep="\t"), "figPS01_label_audit.tsv.gz", [src])
    d = pd.read_csv(imp, sep="\t")
    g = (d.groupby(["set", "labels", "feature"], as_index=False)["gain"]
         .agg(gain_mean="mean", gain_min="min", gain_max="max"))
    write(g, "figPS01_feature_importance.tsv.gz", [imp])


def om869708(phage_rank, blast):
    pr = Path(phage_rank)
    f = [("input_id", "==", "OM869708")]
    orfs = pd.read_parquet(pr / "tables/orfs.parquet",
                           columns=["orf_id", "g_start", "g_end", "strand", "frame", "record_length"],
                           filters=f)
    cols = ["orf_id", "protein_length", "protein_entropy", "three_di_entropy",
            "actual_label", "probability_in_genbank"]
    new = pd.read_parquet(pr / "rank_orfs_relabelled_intrinsic/all_orf_oof_predictions.parquet",
                          columns=cols, filters=f)
    old = pd.read_parquet(pr / "rank_orfs_orig/all_orf_oof_predictions.parquet",
                          columns=["orf_id", "probability_in_genbank"], filters=f)
    o = (orfs.merge(new, on="orf_id")
         .merge(old.rename(columns={"probability_in_genbank": "probability_legacy"}), on="orf_id")
         .rename(columns={"probability_in_genbank": "probability_corrected",
                          "actual_label": "matches_cds"})
         .sort_values("g_start"))
    cds = pd.read_parquet(pr / "tables/cds.parquet",
                          columns=["cds_id", "g_start", "g_end", "strand", "product"],
                          filters=f).sort_values("g_start")
    rep = pr / "rank_orfs_relabelled_intrinsic/ranked_overlapping_gene_replacements.tsv.gz"
    r = pd.read_csv(rep, sep="\t")
    r["rank"] = range(1, len(r) + 1)
    r = r[r.input_id == "OM869708"]
    o["replacement_rank"] = o.orf_id.map(dict(zip(r.orf_id, r["rank"])))
    o["replacement_pairs_total"] = len(pd.read_csv(rep, sep="\t", usecols=[0]))
    sources = [pr / "tables/orfs.parquet", pr / "tables/cds.parquet", rep]
    write(o, "figP01_om869708_orfs.tsv.gz", sources)
    write(cds, "figP01_om869708_cds.tsv.gz", [pr / "tables/cds.parquet"])
    if blast:
        names = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "qstart", "qend",
                 "sstart", "send", "evalue", "bitscore", "qcovs", "staxids", "sscinames", "stitle"]
        b = pd.read_csv(blast, sep="\t", names=names)
        write(b, "figP01_om869708_blastp.tsv.gz", [Path(blast)])


PROVENANCE = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phage-rank", required=True,
                    help="work/phage_rank directory of the corrected ranking run")
    ap.add_argument("--blast", help="BLASTP tabular output for OM869708 ORF3/ORF7")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mi_panel()
    model_panel()
    foldseek_panel()
    label_audit()
    om869708(args.phage_rank, args.blast)
    (OUT / "phage_provenance.json").write_text(json.dumps(PROVENANCE, indent=1) + "\n")
    print("  phage_provenance.json")


if __name__ == "__main__":
    main()
