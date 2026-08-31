#!/usr/bin/env python3
"""Are high-3Di in_genbank=False ORFs real proteins the annotation missed?

The hypothesis: unmatched ORFs with 3Di entropy above ~2.5 look
structurally like real proteins, so perhaps the original annotation
software simply failed to call them.

Two things have to be separated out before that can be tested.

FIRST, and much the larger effect: 46% of GTDB bacterial representatives
carry no CDS annotation at all. Every ORF in those genomes is False by
construction, and nothing was "missed" because nothing was ever run. Only
genomes with at least one annotated CDS can speak to the hypothesis, so
everything below is restricted to those.

SECOND, get_orfs reads all six frames, so an unmatched ORF that overlaps an
annotated CDS is usually a shadow of that gene -- an alternative frame or
the opposite strand of real coding sequence. It inherits real sequence
structure and can score high 3Di entropy without being a distinct protein.
A genuine missed gene should sit in intergenic space, overlapping nothing
annotated.

That gives three groups among unmatched ORFs in annotated genomes:
  shadow      overlaps an annotated CDS -> explained, not a missed gene
  intergenic  overlaps nothing annotated -> the actual candidate pool
  and each splits on the 3Di >= 2.5 line.

Length is the independent check. Real bacterial proteins have a
characteristic length distribution; spurious ORF calls skew short. If the
high-3Di intergenic ORFs are missed genes, their lengths should resemble
the annotated CDS population rather than the short-ORF background.
"""
import glob
import os
import sys
import numpy as np
import pandas as pd

# Working directory for intermediate samples. Session-scratch on the
# machine this was run on; set SCRATCH in the environment, or edit, to
# point somewhere writable on yours.
SCRATCH = os.environ.get("GE_SCRATCH", "./work")

# Full ORF complement for whole chunks. Overlap testing needs every ORF of
# a genome, so these are complete chunks rather than a sampled subset.
# Produce them with, per chunk:
#   zcat <results>/bac_NNN.tsv.gz | awk -F'\t' 'NR>1 && $12!="" && $13!="" \
#     {print $3"\t"$4"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10"\t"$12"\t"$13}' \
#     > $GE_SCRATCH/missed/bac_NNN.tsv
FILES = sorted(glob.glob(os.path.join(SCRATCH, "missed", "*.tsv")))
COLS = ["genome", "contig", "start", "end", "strand", "aa_length",
        "in_genbank", "protein_entropy", "three_di_entropy"]
THRESH = 2.5


def overlaps_annotated(df):
    """Flag each unmatched ORF that overlaps an annotated CDS on its contig.

    Coordinates are one-based inclusive and, per the schema, negative-strand
    ORFs are reported on the reverse-complement sequence. Strand is
    therefore ignored: the question is whether this span coincides with
    coding sequence at all, in any frame or orientation.
    """
    flag = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby("contig", sort=False):
        t = g[g.in_genbank]
        f = g[~g.in_genbank]
        if len(t) == 0 or len(f) == 0:
            continue
        ts = t.start.to_numpy(); te = t.end.to_numpy()
        order = np.argsort(ts)
        ts, te = ts[order], te[order]
        # running max of end, so a binary search over starts is sufficient
        te_max = np.maximum.accumulate(te)
        fs = f.start.to_numpy(); fe = f.end.to_numpy()
        # last annotated CDS whose start <= this ORF's end
        idx = np.searchsorted(ts, fe, side="right") - 1
        ok = idx >= 0
        hit = np.zeros(len(f), dtype=bool)
        hit[ok] = te_max[idx[ok]] >= fs[ok]
        flag[df.index.get_indexer(f.index)] = hit
    return flag


def main():
    if not FILES:
        print(f"No chunk TSVs under {SCRATCH}/missed -- see the comment on "
              f"FILES for how to produce them.", file=sys.stderr)
        return 1
    df = pd.concat([pd.read_csv(f, sep="\t", header=None, names=COLS)
                    for f in FILES], ignore_index=True)
    # pandas infers a column of "True"/"False" as bool, so comparing it to
    # the string "True" silently yields all-False. Handle either dtype.
    if df.in_genbank.dtype != bool:
        df["in_genbank"] = df.in_genbank.astype(str) == "True"
    print(f"ORFs loaded: {len(df):,} from {df.genome.nunique()} genomes\n")

    # --- confounder 1: genomes with no annotation at all ---
    ann = df.groupby("genome").in_genbank.any()
    annotated = set(ann[ann].index)
    n_all, n_ann = len(ann), len(annotated)
    print("STEP 1 - remove genomes that were never annotated")
    print(f"  genomes with >=1 annotated CDS : {n_ann} of {n_all} ({n_ann/n_all*100:.0f}%)")

    hi_all = df[(~df.in_genbank) & (df.three_di_entropy >= THRESH)]
    hi_ann = hi_all[hi_all.genome.isin(annotated)]
    print(f"  unmatched ORFs with 3Di >= {THRESH}      : {len(hi_all):,}")
    print(f"    of which in annotated genomes  : {len(hi_ann):,} "
          f"({len(hi_ann)/len(hi_all)*100:.1f}%)")
    print(f"    of which in UNannotated genomes: {len(hi_all)-len(hi_ann):,} "
          f"({(1-len(hi_ann)/len(hi_all))*100:.1f}%)  <- not 'missed', never annotated\n")

    d = df[df.genome.isin(annotated)].reset_index(drop=True)

    # --- confounder 2: shadow ORFs over real CDS ---
    print("STEP 2 - within annotated genomes, separate shadows from intergenic")
    d["shadow"] = overlaps_annotated(d)
    unm = d[~d.in_genbank]
    hi = unm[unm.three_di_entropy >= THRESH]
    print(f"  unmatched ORFs                  : {len(unm):,}")
    print(f"    3Di >= {THRESH}                     : {len(hi):,}")
    print(f"      overlapping an annotated CDS : {hi.shadow.sum():,} "
          f"({hi.shadow.mean()*100:.1f}%)  <- shadow of a real gene")
    cand = hi[~hi.shadow]
    print(f"      intergenic                   : {len(cand):,} "
          f"({(~hi.shadow).mean()*100:.1f}%)  <- CANDIDATE missed genes\n")

    # --- length check ---
    print("STEP 3 - does the candidate pool look like real protein?")
    matched = d[d.in_genbank]
    lo = unm[(unm.three_di_entropy < THRESH) & (~unm.shadow)]
    groups = [
        ("annotated CDS (in_genbank=True)", matched),
        ("candidate: intergenic, 3Di >= 2.5", cand),
        ("intergenic, 3Di < 2.5", lo),
        ("shadow of a CDS, 3Di >= 2.5", hi[hi.shadow]),
    ]
    print(f"  {'group':<36}{'n':>10}{'med aa':>8}{'%>=100aa':>10}{'med 3Di':>9}")
    for name, g in groups:
        if len(g) == 0:
            continue
        print(f"  {name:<36}{len(g):>10,}{g.aa_length.median():>8.0f}"
              f"{(g.aa_length >= 100).mean()*100:>9.1f}%{g.three_di_entropy.median():>9.2f}")

    print("\n  per annotated genome:")
    per = cand.groupby("genome").size()
    ncds = matched.groupby("genome").size()
    j = pd.concat([per.rename("cand"), ncds.rename("cds")], axis=1).fillna(0)
    print(f"    annotated CDS per genome (median)      : {j.cds.median():.0f}")
    print(f"    candidate missed genes per genome (med): {j.cand.median():.0f}")
    print(f"    candidates as % of annotated CDS (med) : "
          f"{(j.cand/j.cds.replace(0, np.nan)).median()*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
