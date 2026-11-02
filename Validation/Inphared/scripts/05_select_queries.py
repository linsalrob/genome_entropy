#!/usr/bin/env python3
"""Draw the Foldseek query arms for the re-ranked Inphared candidates (after Validation/logan 21_candidates_select.py).

Inputs: orfs.parquet + seqs.parquet (01), labels.parquet (02), the re-ranked
run (04: all_orf_oof_predictions.parquet, ranked_overlapping_gene_replacements)
and the ORIGINAL run (rank_orfs_orig/: ranked_putative_missing_orfs,
ranked_overlapping_gene_replacements) for comparison arms.

Arms (all unannotated/annotated status from the recomputed labels; p is the
re-ranked out-of-fold probability):
  candidate        unannotated, p >= 0.8, context intergenic / antisense /
                   frameshift (same_stop and same_frame ORFs are pieces of an
                   annotated gene, counted but not searched); up to --n at random
  annotated        annotated ORFs, nearest-length matched (positive control)
  antisense_lo     unannotated antisense ORFs, p < 0.5, length matched
  frameshift_lo    unannotated same-strand other-frame ORFs, p < 0.5, length matched
  intergenic_lo    unannotated intergenic ORFs, p < 0.5, length matched (the
                   pool is small in phage genomes; matched as far as it goes)
  replacement / replaced
                   the re-ranked run's overlapping-replacement ORFs and the
                   annotated ORFs they would replace
  orig_candidate   random sample of the ORIGINAL 52,057 candidates
  orig_replacement / orig_replaced
                   the ORIGINAL 1,797 replacement pairs
A shuffled-3Di null of the candidate arm is built at query-DB time.

Writes selection.parquet (query_id, arm, row, ... ) and queries.{aa,3di}.fasta
plus cand.{aa,3di}.fasta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

LEN_EDGES = [90, 100, 150, 200, 300, 500, 1000, 10**9]


def match(pool: pd.DataFrame, targets: pd.Series, rng) -> pd.DataFrame:
    """Nearest-length match without replacement (LOGAN 21_candidates_select.match)."""
    if pool.empty:
        return pool
    p = pool.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    p = p.sort_values("protein_length", kind="stable").reset_index(drop=True)
    lens = p.protein_length.to_numpy()
    used = np.zeros(len(p), bool)
    free = len(p)
    for L in targets.to_numpy():
        if free == 0:
            break
        i = int(np.searchsorted(lens, L))
        lo, hi = i - 1, i
        while lo >= 0 or hi < len(p):
            cand = []
            if hi < len(p) and not used[hi]:
                cand.append(hi)
            if lo >= 0 and not used[lo]:
                cand.append(lo)
            if cand:
                j = min(cand, key=lambda k: abs(lens[k] - L))
                used[j] = True
                free -= 1
                break
            lo -= 1
            hi += 1
    return p[used]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", required=True, type=Path)
    ap.add_argument("--rank", required=True, type=Path, help="re-ranked rank_missing_orfs output dir")
    ap.add_argument("--orig", type=Path, help="original rank_orfs output dir (same ORF set only; omit otherwise)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-orig", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=102)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    o = pd.read_parquet(a.tables / "orfs.parquet", columns=[
        "row", "input_id", "orf_id", "strand", "g_start", "g_end", "protein_length", "three_di_entropy",
        "context", "context_cds", "cds_overlap_frac", "contig_edge"]).sort_values("row").reset_index(drop=True)
    lab = pd.read_parquet(a.tables / "labels.parquet", columns=["row", "in_genbank", "in_genbank_orig"]).sort_values("row")
    o["annotated"] = lab.in_genbank.to_numpy()
    o["annotated_json"] = lab.in_genbank_orig.to_numpy()
    pr = pd.read_parquet(a.rank / "all_orf_oof_predictions.parquet", columns=["input_id", "orf_id",
                                                                               "probability_in_genbank"])
    assert (pr.input_id.to_numpy() == o.input_id.to_numpy()).all() and (pr.orf_id.to_numpy() == o.orf_id.to_numpy()).all()
    o["p"] = pr.probability_in_genbank.to_numpy()
    o["length_bin"] = pd.cut(o.protein_length, LEN_EDGES, right=False).astype(str)
    key = o.set_index(["input_id", "orf_id"]).row

    un = o[~o.annotated]
    hi = un[un.p >= 0.8]
    summary = {"unannotated": int(len(un)), "candidates_p08": int(len(hi)),
               "candidates_by_context": hi.context.value_counts().to_dict()}
    cpool = hi[hi.context.isin(["intergenic", "antisense", "frameshift"])]
    cand = cpool.sample(min(a.n, len(cpool)), random_state=a.seed).assign(arm="candidate")
    arms = [cand,
            match(o[o.annotated], cand.protein_length, rng).assign(arm="annotated"),
            match(un[(un.context == "antisense") & (un.p < 0.5)], cand.protein_length, rng).assign(arm="antisense_lo"),
            match(un[(un.context == "frameshift") & (un.p < 0.5)], cand.protein_length, rng).assign(arm="frameshift_lo"),
            match(un[(un.context == "intergenic") & (un.p < 0.5)], cand.protein_length, rng).assign(arm="intergenic_lo")]

    rep = pd.read_csv(a.rank / "ranked_overlapping_gene_replacements.tsv.gz", sep="\t",
                      usecols=["input_id", "orf_id", "competing_orf_id"])
    arms.append(o.loc[key.loc[list(zip(rep.input_id, rep.orf_id))].to_numpy()].assign(arm="replacement"))
    arms.append(o.loc[key.loc[list(zip(rep.input_id, rep.competing_orf_id))].drop_duplicates().to_numpy()]
                .assign(arm="replaced"))
    if a.orig:
        orep = pd.read_csv(a.orig / "ranked_overlapping_gene_replacements.tsv.gz", sep="\t",
                           usecols=["input_id", "orf_id", "competing_orf_id"])
        arms.append(o.loc[key.loc[list(zip(orep.input_id, orep.orf_id))].to_numpy()].assign(arm="orig_replacement"))
        arms.append(o.loc[key.loc[list(zip(orep.input_id, orep.competing_orf_id))].drop_duplicates().to_numpy()]
                    .assign(arm="orig_replaced"))
        oc = pd.read_csv(a.orig / "ranked_putative_missing_orfs.tsv.gz", sep="\t", usecols=["input_id", "orf_id"])
        oc = oc.sample(min(a.n_orig, len(oc)), random_state=a.seed)
        arms.append(o.loc[key.loc[list(zip(oc.input_id, oc.orf_id))].to_numpy()].assign(arm="orig_candidate"))

    sel = pd.concat(arms, ignore_index=True)
    sel["query_id"] = sel.arm + "|" + sel.row.astype(str)
    sel.to_parquet(a.out / "selection.parquet", index=False)
    summary["arms"] = sel.arm.value_counts().to_dict()
    summary["arm_length_median"] = sel.groupby("arm").protein_length.median().to_dict()
    json.dump(summary, open(a.out / "selection.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

    seqs = pd.read_parquet(a.tables / "seqs.parquet")
    seqs = seqs.set_index("row").loc[sel.row.to_numpy()]
    assert (seqs.aa.str.len().to_numpy() == seqs.three_di.str.len().to_numpy()).all(), "AA / 3Di length mismatch"
    for kind, col in (("aa", "aa"), ("3di", "three_di")):
        with open(a.out / f"queries.{kind}.fasta", "w") as fa, open(a.out / f"cand.{kind}.fasta", "w") as fc:
            for q, arm, x in zip(sel.query_id, sel.arm, seqs[col].to_numpy()):
                fa.write(f">{q}\n{x}\n")
                if arm == "candidate":
                    fc.write(f">{q}\n{x}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
