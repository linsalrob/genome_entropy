#!/usr/bin/env python3
"""Ranked candidate table and the direct-evidence ladder (issue #97).

Produces the two things #97 asks to be kept apart:

  1. a per-candidate table carrying every piece of evidence the project has
     generated, for choosing manuscript examples;
  2. counts of candidates meeting progressively stricter direct-evidence
     criteria -- the conservative number, as opposed to the mixture estimate.

EVERY TIER IS REPORTED AGAINST ITS MATCHED SHADOWS, NOT ALONE

A bare count of "candidates with a full-length structural hit" is not evidence
of anything: shadows of real genes get full-length hits too, at 5-9% depending
on the database, because they overlap a real protein. The number that means
something is the EXCESS over a length- and 3Di-matched shadow from the same
genome, which is why the shadow arm was searched 1:1 with the candidates.

So the ladder reports, at every rung, candidates / matched shadows / excess.
The excess is the conservative floor. Reporting the raw candidate count as the
"conservative direct-evidence number" would be a larger overclaim than the
mixture estimate it is supposed to be more careful than.

THE COMPARATOR IS REPORTED TWICE, AND THE SECOND ONE IS THE ANSWER

About a quarter of bacterial shadows and a third of archaeal ones are
same-strand-SAME-FRAME, which means they largely ARE the annotated protein.
They score like real genes because they are real genes, so leaving them in the
background measures the wrong thing and understates the candidate excess.

An earlier version of this docstring claimed they were "excluded where a frame
classification is supplied". There was no such option and no such code -- the
first published ladder ran against the contaminated background. The frame
class now arrives per ORF as `cds_frame_class` from 20_orf_context.py,
computed from normalised coordinates against the deposited CDS intervals, and
BOTH ladders are printed: pooled, for continuity with what was already posted,
and clean (antisense and frameshift only), which is the one to quote.

ON "genome_entropy coding probability"

#97 lists it as a column and as criterion 3. It does not exist -- there is no
classifier or calibrated probability in the installed package, only entropies
and mutual information. No column is emitted for it, deliberately, rather than
filling one with a proxy that a later reader would take at face value. The
entropy support that DOES exist -- protein, 3Di, 12-state, and 3Di-12st mutual
information -- is all present.

ON TARGET DESCRIPTIONS

Only Swiss-Prot and PDB100 carry free text (21_target_descriptions.py). CATH50
gives a superfamily code and BFVD nothing at all. A missing description for a
CATH or BFVD hit is a property of the reference database and is NOT scored as
weak evidence.

  22_rank_candidates.py --domain bac \
      --context <...>/full_bac/context \
      --search-dirs <...>/search_shard1/best,<...>/search_shard2/best,... \
      --out-prefix <...>/full_bac/ranked
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GD = "/g/data/ob80/re3494/gtdb_entropy"
DBS = ["afdb_swissprot", "pdb100", "cath50", "bfvd"]
# The three used for estimation. BFVD is excluded because its annotated-CDS
# ceiling is ~12%, so it is kept for interpretation (a BFVD-only hit suggests
# phage or prophage) but never for a denominator.
EST_DBS = ["afdb_swissprot", "pdb100", "cath50"]
M8_COLS = ["query", "target", "fident", "alnlen", "qlen", "tlen", "qcov",
           "tcov", "evalue", "bits", "taxid", "taxname"]
FULL = 0.8          # qcov and tcov threshold for a full-length hit


def load_context(context_dir, arms):
    files = sorted(glob.glob(f"{context_dir}/*.context.tsv.gz"))
    if not files:
        sys.exit(f"ERROR: no context tables in {context_dir}")
    frames = []
    for p in files:
        d = pd.read_csv(p, sep="\t", dtype={"genome": str, "input_id": str,
                                            "orf_id": str, "group": str})
        frames.append(d[d.group.isin(arms)])
    df = pd.concat(frames, ignore_index=True)
    df["qid"] = (df.genome + "|" + df.input_id + "|" + df.orf_id
                 + "|" + df.group)
    return df


def load_hits(search_dirs, dbs):
    """best[(db, mode)] -> one row per query, across all shards."""
    best = {}
    for d in search_dirs:
        for path in sorted(glob.glob(f"{d}/*.m8")):
            stem = os.path.basename(path)[:-3]
            qtag, db, mode = stem.split(".")
            if qtag == "null" or db not in dbs:
                continue
            if not os.path.getsize(path):
                continue
            df = pd.read_csv(path, sep="\t", names=M8_COLS, low_memory=False)
            best.setdefault((db, mode), []).append(df)
    out = {}
    for k, v in best.items():
        df = pd.concat(v, ignore_index=True)
        # Shards are disjoint by construction (contiguous chunk blocks), but a
        # resubmitted shard could overlap. Reduce again rather than trust it.
        df = df.loc[df.groupby("query")["bits"].idxmax()]
        out[k] = df
    return out


def load_descriptions(desc_dir, dbs):
    desc = {}
    for db in dbs:
        p = Path(desc_dir) / f"{db}.desc.tsv"
        if p.exists():
            desc[db] = pd.read_csv(p, sep="\t", dtype=str).fillna("")
    return desc


def widen(orfs, best, desc, dbs):
    """One column block per (database, mode), joined onto the ORF table."""
    out = orfs.copy()
    for db in dbs:
        for mode in ("struct", "seq"):
            h = best.get((db, mode))
            pre = f"{db}_{mode}"
            if h is None or not len(h):
                out[f"{pre}_hit"] = False
                out[f"{pre}_full"] = False
                continue
            keep = h[["query", "target", "qcov", "tcov", "evalue", "bits",
                      "taxname"]].rename(columns={
                          "query": "qid",
                          "target": f"{pre}_target",
                          "qcov": f"{pre}_qcov",
                          "tcov": f"{pre}_tcov",
                          "evalue": f"{pre}_evalue",
                          "bits": f"{pre}_bits",
                          "taxname": f"{pre}_taxname"})
            out = out.merge(keep, on="qid", how="left")
            out[f"{pre}_hit"] = out[f"{pre}_target"].notna()
            out[f"{pre}_full"] = ((out[f"{pre}_qcov"] >= FULL)
                                  & (out[f"{pre}_tcov"] >= FULL)).fillna(False)
        # Descriptions attach to the structural hit, which is the primary
        # readout; the sequence hit's target is reported but not described.
        if db in desc:
            d = desc[db].rename(columns={
                "target": f"{db}_struct_target",
                "description": f"{db}_product",
                "cath_superfamily": f"{db}_cath_superfamily"})
            cols = [f"{db}_struct_target", f"{db}_product"]
            if db == "cath50":
                cols.append(f"{db}_cath_superfamily")
            if f"{db}_struct_target" in out.columns:
                out = out.merge(d[cols], on=f"{db}_struct_target", how="left")

    # Evidence class per database, and overall. "structure-only" is the one
    # #97 singles out, because it is what demonstrates information beyond
    # conventional sequence similarity.
    for db in dbs:
        s, q = out[f"{db}_struct_full"], out[f"{db}_seq_full"]
        out[f"{db}_class"] = np.select(
            [s & q, s & ~q, ~s & q],
            ["sequence+structure", "structure-only", "sequence-only"],
            default="neither")

    out["n_db_full_struct"] = sum(out[f"{db}_struct_full"].astype(int)
                                  for db in dbs)
    out["n_est_db_full_struct"] = sum(out[f"{db}_struct_full"].astype(int)
                                      for db in EST_DBS)
    out["any_full_struct"] = out["n_db_full_struct"] > 0
    out["any_full_seq"] = sum(out[f"{db}_seq_full"].astype(int)
                              for db in dbs) > 0
    out["structure_only"] = out["any_full_struct"] & ~out["any_full_seq"]
    # An interpretable annotation is only available from Swiss-Prot and PDB.
    prod = pd.Series(False, index=out.index)
    for db in ("afdb_swissprot", "pdb100"):
        if f"{db}_product" in out.columns:
            prod |= (out[f"{db}_struct_full"]
                     & out[f"{db}_product"].fillna("").str.len().gt(0))
    out["interpretable_product"] = prod

    best_bits = pd.concat([out[f"{db}_struct_bits"] for db in dbs
                           if f"{db}_struct_bits" in out.columns], axis=1)
    out["best_struct_bits"] = best_bits.max(axis=1)
    best_e = pd.concat([out[f"{db}_struct_evalue"] for db in dbs
                        if f"{db}_struct_evalue" in out.columns], axis=1)
    out["best_struct_evalue"] = best_e.min(axis=1)
    return out


def tiers(df):
    """The #97 criteria, as boolean masks.

    These are a SET, not a nested ladder, and must not be presented as one.
    C6 (structure-only) is not stricter than C4 (>=2 databases) -- on archaea
    C6 holds for 4,439 candidates and C4 for 2,777, so a "progressively
    stricter" reading of the rows in order is simply false. Each row except C7
    is C2 plus one independent requirement; C7 is the conjunction.
    """
    t = {}
    t["C1 all candidates"] = pd.Series(True, index=df.index)
    t["C2 not contig-truncated"] = ~df.truncated_calc.astype(bool)
    t["C3 C2 + any full-length structural hit"] = t["C2 not contig-truncated"] & df.any_full_struct
    t["C4 C2 + full-length in >=2 databases"] = t["C2 not contig-truncated"] & (df.n_db_full_struct >= 2)
    t["C5 C2 + full-length SwissProt/PDB/CATH"] = t["C2 not contig-truncated"] & (df.n_est_db_full_struct >= 1)
    t["C6 C2 + structure-only, full-length"] = t["C2 not contig-truncated"] & df.structure_only & df.any_full_struct
    # The strict combined rule. Every clause is one of #97's criteria, and the
    # length floor is there because a 60 aa ORF cannot make an unambiguous gene
    # model however good its hit is (criterion 9).
    t["C7 strict combined rule"] = (
        t["C2 not contig-truncated"]
        & (df.n_est_db_full_struct >= 1)
        & (df.n_db_full_struct >= 2)
        & df.interpretable_product
        & (df.aa_length >= 100)
        & (df.best_struct_evalue < 1e-10))
    return t


def ladder(cand, shadow, out, label):
    w = 42
    lines = ["", "=" * 92,
             f"DIRECT EVIDENCE -- {label}",
             "=" * 92,
             "",
             "  Every row is a count of candidates AND of matched shadows meeting the",
             "  same criterion. Shadows overlap real genes, so they clear these bars too;",
             "  the excess is the conservative floor, not the candidate column.",
             "",
             "  These criteria are a SET, not a nested ladder. C3-C6 are each C2 plus one",
             "  independent requirement, so the counts are NOT monotone down the column",
             "  and reading them as increasing stringency is wrong. C7 is the conjunction.",
             ""]
    tc, ts = tiers(cand), tiers(shadow)
    lines.append(f"  {'criterion':<{w}}{'candidates':>12}{'shadows':>10}"
                 f"{'excess':>10}{'excess %':>10}")
    lines.append("  " + "-" * (w + 42))
    n_c, n_s = len(cand), len(shadow)
    for k in tc:
        c, s = int(tc[k].sum()), int(ts[k].sum())
        # Scale the shadow count if the arms differ in size, so the excess is
        # a like-for-like difference rather than an artefact of unequal n.
        s_scaled = s * n_c / n_s if n_s else 0.0
        exc = c - s_scaled
        lines.append(f"  {k:<{w}}{c:>12,}{s:>10,}{exc:>10,.0f}"
                     f"{100.0 * exc / n_c:>9.1f}%")
    lines.append("")
    lines.append(f"  candidate arm {n_c:,};  shadow arm {n_s:,}"
                 + ("  (shadow counts scaled to the candidate arm)"
                    if n_c != n_s else ""))
    text = "\n".join(lines)
    print(text)
    with open(out, "a") as fh:
        fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", default="bac")
    ap.add_argument("--context", required=True)
    ap.add_argument("--search-dirs", required=True,
                    help="comma-separated best/ directories")
    ap.add_argument("--descriptions", default=f"{GD}/foldseek_db/descriptions")
    ap.add_argument("--burden", default=None)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--top", type=int, default=2000,
                    help="rows written to the shortlist table")
    args = ap.parse_args()

    burden = args.burden or f"{GD}/missed_genes/candidate_burden_{args.domain}.tsv"
    dirs = [d.strip() for d in args.search_dirs.split(",") if d.strip()]

    print(f"context      : {args.context}")
    print(f"search dirs  : {len(dirs)}")
    orfs = load_context(args.context, {"candidate", "shadow_hi"})
    print(f"ORFs loaded  : {len(orfs):,} "
          f"({int((orfs.group == 'candidate').sum()):,} candidate, "
          f"{int((orfs.group == 'shadow_hi').sum()):,} shadow_hi)")

    best = load_hits(dirs, DBS)
    if not best:
        sys.exit(f"ERROR: no usable .m8 tables under {dirs}")
    print("hit tables   : " + ", ".join(f"{db}/{mode} {len(v):,}"
                                        for (db, mode), v in sorted(best.items())))

    desc = load_descriptions(args.descriptions, DBS)
    print(f"descriptions : {', '.join(sorted(desc)) or 'none'}")

    wide = widen(orfs, best, desc, DBS)

    if Path(burden).exists():
        b = pd.read_csv(burden, sep="\t",
                        usecols=["genome", "n50_contigs", "contig_count",
                                 "checkm2_completeness", "checkm2_contamination",
                                 "gtdb_taxonomy", "phylum"],
                        dtype={"genome": str})
        wide = wide.merge(b, on="genome", how="left")
        print(f"burden joined: {int(wide.n50_contigs.notna().sum()):,} of "
              f"{len(wide):,} rows have genome quality")
    else:
        print(f"WARNING: no {burden}; genome-quality columns absent",
              file=sys.stderr)

    cand = wide[wide.group == "candidate"].copy()
    shadow = wide[wide.group == "shadow_hi"].copy()

    report = f"{args.out_prefix}_ladder.txt"
    Path(report).write_text(
        f"domain: {args.domain}\ncandidates: {len(cand):,}\n"
        f"shadows: {len(shadow):,}\n")
    if "cds_frame_class" not in shadow.columns:
        sys.exit("ERROR: the context tables predate cds_frame_class. Without "
                 "it the shadow background is contaminated by same-frame "
                 "shadows, which ARE the annotated protein. Rerun "
                 "20_orf_context.pbs.")

    fc = shadow.cds_frame_class.fillna("(no overlap)")
    print("shadow frame classes:")
    for k, v in fc.value_counts().items():
        print(f"  {k:<28}{v:>10,}  {100.0 * v / len(shadow):5.1f}%")

    print("### comparator 1 of 2: ALL shadows (contaminated, for continuity)")
    ladder(cand, shadow, report, "ALL shadows -- contaminated by same-frame")

    CLEAN = {"opposite strand", "same strand, frameshift"}
    clean = shadow[fc.isin(CLEAN)]
    if len(clean):
        print("### comparator 2 of 2: CLEAN shadows -- QUOTE THIS ONE")
        ladder(cand, clean, report,
               "CLEAN shadows -- antisense and frameshift only")
    else:
        print("WARNING: no antisense or frameshift shadows; clean comparator "
              "not computed", file=sys.stderr)

    # Rank on HOW MANY criteria are met, not on the ordinal of the last one
    # matched. The criteria are unordered, so "highest-numbered criterion
    # satisfied" would rank a structure-only candidate above one supported by
    # four databases purely because C6 is printed below C4.
    tc = tiers(cand)
    scored = [k for k in tc if k not in ("C1 all candidates",)]
    cand["n_criteria_met"] = sum(tc[k].astype(int) for k in scored)
    cand["strict_rule"] = tc["C7 strict combined rule"]
    for k in scored:
        cand[k.split()[0].lower()] = tc[k]
    cand = cand.sort_values(
        ["strict_rule", "n_criteria_met", "n_db_full_struct",
         "best_struct_bits"],
        ascending=[False, False, False, False])

    full_out = f"{args.out_prefix}_candidates.tsv.gz"
    cand.drop(columns=["qid"]).to_csv(full_out, sep="\t", index=False,
                                      compression="gzip")
    short = f"{args.out_prefix}_shortlist.tsv"
    cand.drop(columns=["qid"]).head(args.top).to_csv(short, sep="\t", index=False)
    print(f"\nranked table -> {full_out}  ({len(cand):,} candidates)")
    print(f"shortlist    -> {short}  (top {min(args.top, len(cand)):,})")
    print(f"ladder       -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
