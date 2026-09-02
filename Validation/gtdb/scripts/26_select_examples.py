#!/usr/bin/env python3
"""Pick the 5-10 manuscript examples, by biological class (issue #97, D3).

Rob's direction:

  "Select manuscript examples by biological class, not simply by score. Do not
   take the top 10 TonB-dependent receptors or the top mobile-element hits."

and a named wish-list: a bacterial TonB/outer-membrane transporter; an archaeal
housekeeping or metabolic protein; a replication/repair example (RecF if the
locus is clean); one mobile element / phage; one structure-only case with strong
coverage AND ORDINARY genome-level burden; ideally one independently called by
GTDB Prodigal but absent from GenBank.

So this walks a list of SLOTS and fills each from the best available candidate
that passes the quality gates, rather than sorting the whole table and taking a
head. Ranking by score put 900-1,080 aa Bacteroidota TonB receptors in all ten
top bacterial slots, which is one phenomenon reported ten times.

THE GATES, AND WHY EACH ONE IS THERE

  not truncated              #97 criterion 1. An ORF running off a contig edge
                             cannot make an unambiguous gene model however good
                             its hit is.
  CDS on BOTH sides          #97 criterion 2 -- "sits cleanly between two
                             annotated genes". Also what makes a locus figure
                             worth drawing.
  aa_length >= 100           criterion 9, an unambiguous gene model.
  completeness >= 95%,       criterion 10, and a POSITIVE reason as well: the
  contamination <= 5%        candidate/shadow separation exists only in
                             well-assembled genomes (archaeal N50 Q4 43.5% vs
                             32.6%, Q1 no gap at all), so a poorly assembled
                             host carries no evidential weight even if the
                             locus looks lovely.
  ORDINARY genome burden     Rob's explicit wording. GCA_000965745.1 carries 108
                             strict candidates, ~50x the archaeal median; that
                             genome is lightly annotated, so leading with it
                             invites "what is wrong with that genome" and the
                             honest answer is a story about annotation rather
                             than about the method. Capped at the 90th centile
                             of per-genome candidate count.

DEDUPLICATION. One example per genome and one per product name across the whole
shortlist, so "diversity" is real rather than nominal.

WHAT IS DELIBERATELY *NOT* A GATE. Prodigal agreement. It is reported for every
example -- and one slot requires it -- but it is not required of all of them,
because the examples illustrate the biology while the rate comes from the
full-population comparisons. Filtering every example on Prodigal would quietly
turn the illustrations into a biased subsample of the evidence.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# (slot name, domain or None, predicate) -- first match wins, in this order.
SLOTS = [
    ("bacterial outer-membrane transporter", "bac",
     lambda d: (d.functional_class == "transport")
     & d.best_product.str.contains(r"TonB|outer membrane|receptor", case=False, na=False)),
    ("archaeal metabolic / housekeeping", "arc",
     lambda d: d.functional_class.isin(["metabolism", "translation"])),
    ("replication / repair (RecF preferred)", None,
     lambda d: (d.functional_class == "replication_repair")),
    ("mobile element / phage / prophage", None,
     lambda d: d.mobile_element),
    ("structure-only, ordinary burden", None,
     lambda d: d.structure_only & (d.n_db_full_struct >= 3)),
    ("Prodigal-confirmed, absent from GenBank", None,
     lambda d: d.prodigal_coincides.fillna(False)),
    ("bacterial metabolic / housekeeping", "bac",
     lambda d: d.functional_class.isin(["metabolism", "translation"])),
    ("defence system", None, lambda d: d.functional_class == "defence"),
    ("regulation", None, lambda d: d.functional_class == "regulation"),
    ("archaeal uncharacterized, strong support", "arc",
     lambda d: d.functional_class.isin(["uncharacterized", "other_named"])),
]

SHOW = ["slot", "domain", "genome", "input_id", "orf_id", "strand", "aa_length",
        "g_start", "g_end", "functional_class", "best_product",
        "n_db_full_struct", "structure_only", "prodigal_coincides",
        "afdb_swissprot_struct_qcov", "afdb_swissprot_struct_tcov",
        "afdb_swissprot_struct_evalue", "three_di_entropy",
        "twelve_state_entropy", "three_di_twelve_state_mutual_information",
        "dist_up", "dist_down", "gap_len", "up_cds_id", "down_cds_id",
        "checkm2_completeness", "checkm2_contamination", "n50_contigs",
        "phylum", "genome_n_candidates", "bfvd_only", "strict_rule"]


def load(domain, classified, coincidence, burden):
    d = pd.read_csv(classified, sep="\t", low_memory=False)
    d["domain"] = domain
    co = pd.read_csv(coincidence, sep="\t", low_memory=False,
                     usecols=["genome", "input_id", "orf_id", "group",
                              "coincides", "offset3"])
    co = co[co.group == "candidate"].drop(columns=["group"])
    key = ["genome", "input_id", "orf_id"]
    d = d.merge(co.rename(columns={"coincides": "prodigal_coincides"}),
                on=key, how="left")
    b = pd.read_csv(burden, sep="\t", usecols=["genome", "n_candidates"])
    d = d.merge(b.rename(columns={"n_candidates": "genome_n_candidates"}),
                on="genome", how="left")
    return d


def gate(d, burden_cap):
    n0 = len(d)
    m = (~d.truncated_calc.astype(bool)
         & (d.dist_up >= 0) & (d.dist_down >= 0)
         & (d.aa_length >= 100)
         & (d.checkm2_completeness >= 95)
         & (d.checkm2_contamination <= 5)
         & (d.genome_n_candidates <= burden_cap)
         & d.strict_rule.astype(bool))
    out = d[m].copy()
    print(f"  {d.domain.iloc[0] if len(d) else '?'}: {n0:,} candidates -> "
          f"{len(out):,} pass the gates (burden cap {burden_cap:.0f})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gd", default="/g/data/ob80/re3494/gtdb_entropy")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--burden-centile", type=float, default=90.0)
    args = ap.parse_args()

    G = args.gd
    frames = []
    for dom, sub, tag in [("bac", "full_bac", "bac"), ("arc", "pilot_arc", "arc")]:
        d = load(dom,
                 f"{G}/missed_genes/{sub}/func_{tag}_classified.tsv.gz",
                 f"{G}/missed_genes/{sub}/prodigal_{tag}4.coincidence.tsv.gz",
                 f"{G}/missed_genes/candidate_burden_{tag}.tsv")
        cap = d.genome_n_candidates.quantile(args.burden_centile / 100.0)
        frames.append(gate(d, cap))
    pool = pd.concat(frames, ignore_index=True)
    print(f"\ncombined eligible pool: {len(pool):,} "
          f"({pool.genome.nunique():,} genomes)")
    print("  by class: " + ", ".join(f"{k} {v}" for k, v in
                                     pool.functional_class.value_counts().items()))

    # Order within a slot: multi-database support, then Prodigal agreement,
    # then the best structural E-value. Not raw bit score -- that just
    # reselects the longest proteins.
    pool["_rank_e"] = pool.best_struct_evalue.fillna(1.0)
    pool = pool.sort_values(
        ["n_db_full_struct", "prodigal_coincides", "_rank_e"],
        ascending=[False, False, True])

    used_genomes, used_products, picks = set(), set(), []
    for slot, dom, pred in SLOTS:
        sub = pool[pred(pool)]
        if dom:
            sub = sub[sub.domain == dom]
        sub = sub[~sub.genome.isin(used_genomes)
                  & ~sub.best_product.isin(used_products)]
        # RecF gets first refusal in the replication/repair slot: Rob named it.
        if "RecF" in slot:
            pref = sub[sub.best_product.str.contains("RecF", case=False, na=False)]
            if len(pref):
                sub = pref
        if not len(sub):
            print(f"  UNFILLED: {slot}")
            continue
        row = sub.iloc[0].copy()
        row["slot"] = slot
        picks.append(row)
        used_genomes.add(row.genome)
        used_products.add(row.best_product)

    if not picks:
        sys.exit("ERROR: no slot could be filled")
    ex = pd.DataFrame(picks)
    cols = [c for c in SHOW if c in ex.columns]
    out = f"{args.out_prefix}_examples.tsv"
    ex[cols].to_csv(out, sep="\t", index=False)
    print(f"\nselected {len(ex)} examples -> {out}")
    pd.set_option("display.width", 250, "display.max_colwidth", 46)
    print()
    print(ex[["slot", "domain", "genome", "orf_id", "aa_length", "best_product",
              "n_db_full_struct", "prodigal_coincides",
              "dist_up", "dist_down"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
