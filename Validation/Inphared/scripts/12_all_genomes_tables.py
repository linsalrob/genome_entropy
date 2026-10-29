#!/usr/bin/env python3
"""Small result tables for the all-genome (36,025) Inphared ranking, for Validation/Inphared/all_genomes/tables.

Reads the work directory of the all-genome run and writes:
  model_metrics_pooled.tsv  pooled out-of-fold AUC / AP / Brier per feature set and CV grouping
                            (genome, ani95 = skani 95 % ANI cluster, genus)
  model_thresholds.tsv      recall and unannotated ORFs above p >= 0.8 per set and grouping
  model_metrics_by_length.tsv  genome-grouped, per length bin
  model_importance.tsv      genome-grouped gain importance
  candidates_by_context.tsv p >= 0.8 unannotated ORFs by context, split by whether the
                            genome holds IUPAC ambiguity codes (non-ACGT bases)
  candidates_per_genome.tsv searched-context candidates per genome, genome length, non-ACGT
                            count, ANI cluster and NCBI genus
  candidates_per_genome_summary.json
  groups_summary.json       copy of the clustering summary (10_genome_groups.py)

  12_all_genomes_tables.py --work $PHAGE_WORK --out Validation/Inphared/all_genomes/tables
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

SEARCHED = ["intergenic", "antisense", "frameshift"]
GROUPINGS = {"genome": "models_relabelled", "ani95": "models_relabelled_ani95", "genus": "models_relabelled_genus"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--fs", default="intrinsic")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    W, out = a.work, a.out
    out.mkdir(parents=True, exist_ok=True)

    met, thr = [], []
    for g, d in GROUPINGS.items():
        m = pd.read_csv(W / d / "metrics.tsv", sep="\t")
        met.append(m[m.fold == "pooled"].drop(columns="fold").assign(cv_grouping=g))
        t = pd.read_csv(W / d / "thresholds.tsv", sep="\t")
        thr.append(t.assign(cv_grouping=g))
    met = pd.concat(met)[["cv_grouping", "set", "n", "positives", "auc", "ap", "brier"]]
    met.to_csv(out / "model_metrics_pooled.tsv", sep="\t", index=False, float_format="%.5f")
    pd.concat(thr).to_csv(out / "model_thresholds.tsv", sep="\t", index=False, float_format="%.5f")
    shutil.copy(W / "models_relabelled" / "metrics_by_length.tsv", out / "model_metrics_by_length.tsv")
    imp = pd.read_csv(W / "models_relabelled" / "importance.tsv", sep="\t")
    imp.groupby(["set", "feature"]).gain.agg(["mean", "min", "max"]).reset_index() \
        .to_csv(out / "model_importance.tsv", sep="\t", index=False, float_format="%.5f")

    c = pd.read_csv(W / f"rank_orfs_relabelled_{a.fs}" / "ranked_putative_missing_orfs.tsv.gz", sep="\t",
                    usecols=["input_id", "orf_id", "probability_in_genbank", "protein_length"])
    o = pq.read_table(W / "tables" / "orfs.parquet", columns=["input_id", "orf_id", "context"]).to_pandas()
    c = c.merge(o, on=["input_id", "orf_id"], how="left", validate="one_to_one")
    amb = pd.read_csv(W / "groups" / "genome_nonACGT.tsv", sep="\t").set_index("input_id").nonACGT
    c["genome_has_nonACGT"] = c.input_id.map(amb).fillna(0) > 0
    ctx = pd.crosstab(c.context, c.genome_has_nonACGT, margins=True, margins_name="all")
    ctx.columns = [{False: "clean_genomes", True: "genomes_with_nonACGT"}.get(x, x) for x in ctx.columns]
    ctx.to_csv(out / "candidates_by_context.tsv", sep="\t")

    L = pq.read_table(W / "tables" / "cds.parquet", columns=["input_id", "record_length"]).to_pandas() \
        .drop_duplicates("input_id").set_index("input_id").record_length
    s = c[c.context.isin(SEARCHED)]
    n = s.groupby("input_id").size().reindex(L.index, fill_value=0)
    ani = pd.read_csv(W / "groups" / "groups_ani95.tsv", sep="\t").set_index("input_id").group
    gen = pd.read_csv(W / "meta" / "ncbi_genus.tsv", sep="\t", dtype=str).fillna("").set_index("input_id")
    per = pd.DataFrame({"candidates": n, "genome_bp": L, "nonACGT": amb.reindex(L.index).fillna(0).astype(int),
                        "ani95_cluster": ani.reindex(L.index), "genus": gen.genus.reindex(L.index),
                        "family": gen.family.reindex(L.index)})
    per["candidates_per_10kb"] = per.candidates / (per.genome_bp / 1e4)
    per.sort_values(["candidates", "genome_bp"], ascending=False).to_csv(
        out / "candidates_per_genome.tsv.gz", sep="\t", index_label="input_id", float_format="%.4f",
        compression={"method": "gzip", "mtime": 0})
    rho = spearmanr(per.genome_bp, per.candidates)
    summ = {"candidates_p08": int(len(c)), "searched_context_candidates": int(len(s)),
            "pieces_of_annotated_genes": int((~c.context.isin(SEARCHED)).sum()),
            "candidates_in_genomes_with_nonACGT": int(c.genome_has_nonACGT.sum()),
            "genomes_with_nonACGT": int((amb > 0).sum()), "genomes": int(len(L)),
            "genomes_with_candidates": int((n > 0).sum()), "genomes_without": int((n == 0).sum()),
            "median_per_genome": float(n.median()), "p90_per_genome": float(n.quantile(0.9)),
            "max_per_genome": int(n.max()), "max_genome": str(n.idxmax()),
            "median_per_genome_with_any": float(n[n > 0].median()),
            "median_per_10kb": float(per.candidates_per_10kb.median()),
            "p90_per_10kb": float(per.candidates_per_10kb.quantile(0.9)),
            "spearman_genome_length_vs_count": float(rho.statistic),
            "median_candidate_aa": float(s.protein_length.median())}
    json.dump(summ, open(out / "candidates_per_genome_summary.json", "w"), indent=1)
    shutil.copy(W / "groups" / "cluster_summary.json", out / "groups_summary.json")
    print(met.round(4).to_string(index=False))
    print(ctx.to_string())
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
