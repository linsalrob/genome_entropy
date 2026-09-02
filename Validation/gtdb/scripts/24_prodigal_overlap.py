#!/usr/bin/env python3
"""Do candidates coincide with GTDB Prodigal gene calls that GenBank lacks?

The independent test. Foldseek, the structural databases and the two-component
mixture all share one assumption chain; Prodigal shares none of it. It is a
different algorithm, run by a different group, on the same DNA, and it does not
consult GenBank. If our candidates are real genes, an independent gene caller
should be calling them.

WHY MERE OVERLAP IS THE WRONG STATISTIC

A matched shadow overlaps a deposited CDS by definition, and Prodigal calls
that CDS. So shadows overlap Prodigal genes essentially always, and a
candidate-vs-shadow contrast on plain overlap is the SAME measurement artefact
that inflates the neighbour-distance comparison: it is created by the arm
definitions, not by biology.

The discriminating statistic is FRAME-AND-STRAND-CONSISTENT COINCIDENCE.
Shadows are antisense or frameshifted relative to the gene they sit in, so a
Prodigal call in the shadow's OWN frame and strand would mean Prodigal agrees
with the shadow rather than with the annotated gene -- which should be rare. A
Prodigal call in a candidate's own frame and strand, on the other hand, is an
independent caller asserting exactly the gene we claim GenBank missed.

HOW COINCIDENCE IS MEASURED, AND WHY THE 3' END

Both Prodigal genes and our ORFs terminate at a stop codon, so two calls of the
same gene agree exactly at their 3' end whatever they do at the 5' end, where
start-site prediction genuinely differs between callers. Matching on the 3' end
is therefore both sharper and fairer than matching on span.

The two coordinate conventions are NOT assumed to agree. Prodigal writes
1-based inclusive forward-axis coordinates; our g_start/g_end are 0-based
half-open forward-axis; and whether each includes the stop codon has to be
established, not guessed -- a silent 3 bp convention difference would produce
zero matches and the false conclusion that Prodigal disagrees with us. So this
script MEASURES the offset distribution between each ORF's 3' end and the
nearest same-strand Prodigal 3' end, and reports the histogram. A spike at a
single offset identifies the convention; the spike's position is then the
match criterion rather than an assumption.

  24_prodigal_overlap.py --coords gtdb_prodigal_coords_arc.tsv.gz \
      --context <...>/context --out prodigal_overlap_arc.txt
"""
import argparse
import glob
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# All four biological arms, not just the two the contrast needs.
#
# WHY intergenic_lo AND annotated_cds MATTER HERE, added after the first
# bacterial run.
#
# The clean-shadow background has a confound that is easy to miss: an antisense
# shadow sits INSIDE a gene Prodigal already calls on the other strand, and
# gene callers deliberately avoid predicting overlapping genes. So Prodigal
# declining a shadow may reflect a commitment it has already made rather than
# the shadow looking non-coding -- which would inflate the candidate excess,
# because candidates sit in intergenic space where Prodigal has no competing
# call to protect.
#
# intergenic_lo is the control that separates those: length-matched ORFs in the
# SAME unoccupied space as the candidates, differing only in having low 3Di.
# If Prodigal calls those rarely, the candidate excess is real. If it calls
# them often, this whole test is measuring "is there room for a gene here"
# rather than "is this a gene".
#
# annotated_cds is the positive control and has to come out near the ceiling.
ARMS = ("candidate", "shadow_hi", "intergenic_lo", "annotated_cds")


def load_context(context_dir, arms):
    files = sorted(glob.glob(f"{context_dir}/*.context.tsv.gz"))
    if not files:
        sys.exit(f"ERROR: no context tables in {context_dir}")
    keep = ["genome", "input_id", "orf_id", "strand", "g_start", "g_end",
            "aa_length", "group", "truncated_calc", "overlaps_cds",
            "cds_frame_class"]
    frames = []
    for p in files:
        d = pd.read_csv(p, sep="\t", dtype={"genome": str, "input_id": str,
                                            "orf_id": str, "group": str})
        if "cds_frame_class" not in d.columns:
            sys.exit(f"ERROR: {p} predates cds_frame_class. Without it the "
                     "shadow background is dominated by same-frame shadows, "
                     "which ARE the annotated gene, and the excess is badly "
                     "understated. Rerun 20_orf_context.pbs.")
        frames.append(d.loc[d.group.isin(arms), keep])
    return pd.concat(frames, ignore_index=True)


def three_prime(strand, g_start, g_end):
    """Forward-axis coordinate of the stop end, on our 0-based half-open axis."""
    return np.where(np.asarray(strand) == "+", g_end, g_start)


def prodigal_three_prime(strand, start, end):
    """Same, for Prodigal's 1-based inclusive coordinates, before calibration.

    Plus strand: the gene ends at `end`, which as a half-open bound is `end`.
    Minus strand: the gene ends at `start`, which as a 0-based coordinate is
    `start - 1`.
    """
    return np.where(np.asarray(strand) == "+", end, start - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coords", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=int, default=30,
                    help="window for the offset histogram")
    args = ap.parse_args()

    orfs = load_context(args.context, ARMS)
    print(f"ORFs            : {len(orfs):,} "
          f"({int((orfs.group == 'candidate').sum()):,} candidate, "
          f"{int((orfs.group == 'shadow_hi').sum()):,} shadow_hi)")

    # FILTER WHILE READING, NOT AFTER.
    #
    # The full-set table is 618,638,921 rows over 32,366,333 contigs. Loading
    # it whole and then subsetting -- which is what this did while archaea's
    # 30 M rows made it look harmless -- needs tens of GB of object-dtype
    # pandas before the first row is discarded. Our ORFs sit on a small
    # fraction of those contigs, so the filter belongs inside the read loop.
    contigs = set(orfs.input_id)
    print(f"contigs of interest: {len(contigs):,}")
    kept, total, seen_contigs = [], 0, set()
    for chunk in pd.read_csv(args.coords, sep="\t", chunksize=5_000_000,
                             dtype={"contig": str, "partial": str,
                                    "gene_index": "int32", "start": "int64",
                                    "end": "int64", "strand": str}):
        total += len(chunk)
        seen_contigs.update(chunk.contig.unique())
        kept.append(chunk[chunk.contig.isin(contigs)])
    pro = pd.concat(kept, ignore_index=True)
    del kept
    print(f"Prodigal genes  : {total:,} on {len(seen_contigs):,} contigs")
    print(f"  on our contigs: {len(pro):,} on {pro.contig.nunique():,} contigs")
    if not len(pro):
        sys.exit("ERROR: no Prodigal genes on any contig carrying our ORFs -- "
                 "the coordinate table and the context tables disagree about "
                 "contig naming.")

    pro["p3"] = prodigal_three_prime(pro.strand, pro.start.to_numpy(),
                                     pro.end.to_numpy())
    orfs["o3"] = three_prime(orfs.strand, orfs.g_start.to_numpy(),
                             orfs.g_end.to_numpy())

    # Nearest same-(contig, strand) Prodigal 3' end for each ORF.
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    pro_sorted = pro.sort_values(["contig", "strand", "p3"], kind="stable")
    groups = {k: v.p3.to_numpy() for k, v in
              pro_sorted.groupby(["contig", "strand"], sort=False)}

    offsets = np.full(len(orfs), np.iinfo(np.int64).max, dtype=np.int64)
    o_by = orfs.groupby(["input_id", "strand"], sort=False).indices
    for (contig, strand), rows in o_by.items():
        arr = groups.get((contig, strand))
        if arr is None or not len(arr):
            continue
        want = orfs.o3.to_numpy()[rows]
        pos = np.searchsorted(arr, want)
        lo = np.clip(pos - 1, 0, len(arr) - 1)
        hi = np.clip(pos, 0, len(arr) - 1)
        d_lo = want - arr[lo]
        d_hi = want - arr[hi]
        pick = np.where(np.abs(d_lo) <= np.abs(d_hi), d_lo, d_hi)
        offsets[rows] = pick
    orfs["offset3"] = offsets

    have = orfs.offset3 != np.iinfo(np.int64).max
    emit()
    emit("=" * 78)
    emit("CALIBRATION: offset between our 3' end and the nearest Prodigal 3' end")
    emit("=" * 78)
    emit("  Same contig and same strand. A single dominant offset identifies the")
    emit("  coordinate convention; it is then used as the match criterion rather")
    emit("  than assumed to be zero.")
    emit()
    emit(f"  ORFs with any same-strand Prodigal gene on their contig: "
         f"{int(have.sum()):,} of {len(orfs):,}")
    emit()
    cnt = Counter(orfs.loc[have, "offset3"].to_numpy())
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:10]
    emit(f"  {'offset (bp)':>12}{'ORFs':>12}{'share':>9}")
    for off, n in top:
        emit(f"  {off:>12,}{n:>12,}{100.0 * n / int(have.sum()):>8.2f}%")
    best_off = top[0][0] if top else 0
    emit()
    emit(f"  dominant offset = {best_off} bp  -> used as exact-coincidence criterion")

    orfs["coincides"] = have & (orfs.offset3 == best_off)

    emit()
    emit("=" * 78)
    emit("FRAME-AND-STRAND-CONSISTENT COINCIDENCE WITH A PRODIGAL GENE")
    emit("=" * 78)
    emit("  A candidate that coincides is a gene an independent caller asserts and")
    emit("  GenBank omits. A shadow that coincides would mean Prodigal agrees with")
    emit("  the antisense/frameshifted ORF rather than the annotated gene, which")
    emit("  should be rare -- that difference is the whole test.")
    emit()
    emit(f"  {'arm':<16}{'n':>10}{'coincide':>11}{'rate':>9}"
         f"{'not truncated':>15}{'rate':>9}")
    emit("  (annotated_cds is the positive control and must sit near the ceiling;")
    emit("   intergenic_lo is unoccupied-space background -- see the ARMS note.)")
    rates = {}
    for arm in ARMS:
        sub = orfs[orfs.group == arm]
        nt = sub[~sub.truncated_calc.astype(bool)]
        r = sub.coincides.mean() if len(sub) else 0.0
        rnt = nt.coincides.mean() if len(nt) else 0.0
        rates[arm] = (len(sub), int(sub.coincides.sum()), r,
                      len(nt), int(nt.coincides.sum()), rnt)
        emit(f"  {arm:<16}{len(sub):>10,}{int(sub.coincides.sum()):>11,}"
             f"{r * 100:>8.2f}%{int(nt.coincides.sum()):>15,}{rnt * 100:>8.2f}%")

    # The breakdown that matters. A same-frame shadow IS the annotated protein,
    # so Prodigal calling it is not a false positive -- it is Prodigal being
    # right. Leaving those in the background measures Prodigal's agreement with
    # GenBank, not its agreement with a wrong ORF, and understates the excess.
    emit()
    emit("=" * 78)
    emit("SHADOW COINCIDENCE BY FRAME CLASS -- why the pooled background is wrong")
    emit("=" * 78)
    sh = orfs[orfs.group == "shadow_hi"]
    emit(f"  {'frame class':<28}{'n':>10}{'coincide':>11}{'rate':>9}")
    for cls, g in sh.groupby(sh.cds_frame_class.fillna("(no overlap)")):
        emit(f"  {cls:<28}{len(g):>10,}{int(g.coincides.sum()):>11,}"
             f"{g.coincides.mean() * 100:>8.2f}%")
    emit()
    emit("  A same-frame shadow coinciding with a Prodigal call is Prodigal")
    emit("  agreeing with GenBank about a gene that IS annotated. It says")
    emit("  nothing about whether Prodigal endorses a spurious ORF, so it does")
    emit("  not belong in the background for this test.")

    CLEAN = {"opposite strand", "same strand, frameshift"}
    clean = sh[sh.cds_frame_class.isin(CLEAN)]
    if len(clean):
        cand = orfs[orfs.group == "candidate"]
        p1, n1 = cand.coincides.mean(), len(cand)
        p2, n2 = clean.coincides.mean(), len(clean)
        diff = p1 - p2
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        emit()
        emit("=" * 78)
        emit("AGAINST THE CLEAN COMPARATOR (antisense and frameshift shadows only)")
        emit("=" * 78)
        emit(f"  clean shadows            : {n2:,} of {len(sh):,}")
        emit(f"  candidate coincidence    : {p1 * 100:.2f}%")
        emit(f"  clean shadow coincidence : {p2 * 100:.2f}%")
        emit(f"  excess                   : {diff * 100:.2f}% "
             f"(95% CI {100 * (diff - 1.96 * se):.2f} - "
             f"{100 * (diff + 1.96 * se):.2f})")
        emit(f"  implied candidates independently called: {diff * n1:,.0f} "
             f"of {n1:,}")
        if p2 > 0:
            emit(f"  enrichment over background: {p1 / p2:.1f}x")
        emit()
        emit("  HOW INDEPENDENT IS THIS, HONESTLY")
        emit("  Independent of GenBank annotation and of structural homology --")
        emit("  Prodigal consults neither. NOT independent of 'looks")
        emit("  compositionally like coding sequence': Prodigal scores codon")
        emit("  usage and GC, and our candidates are selected partly on")
        emit("  protein-like composition, so the two share that prior. The")
        emit("  clean-shadow comparator is what bounds it: those ORFs are also")
        emit("  compositionally protein-like and Prodigal still declines them.")

    if "intergenic_lo" in rates and rates["intergenic_lo"][0]:
        cand = orfs[orfs.group == "candidate"]
        ilo = orfs[orfs.group == "intergenic_lo"]
        p1, n1 = cand.coincides.mean(), len(cand)
        p2, n2 = ilo.coincides.mean(), len(ilo)
        diff = p1 - p2
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        emit()
        emit("=" * 78)
        emit("AGAINST intergenic_lo -- SAME unoccupied space, no competing call")
        emit("=" * 78)
        emit(f"  candidate coincidence     : {p1 * 100:.2f}%")
        emit(f"  intergenic_lo coincidence : {p2 * 100:.2f}%  (n = {n2:,})")
        emit(f"  excess                    : {diff * 100:.2f}% "
             f"(95% CI {100 * (diff - 1.96 * se):.2f} - "
             f"{100 * (diff + 1.96 * se):.2f})")
        emit(f"  implied candidates        : {diff * n1:,.0f} of {n1:,}")
        if p2 > 0:
            emit(f"  enrichment                : {p1 / p2:.1f}x")
        emit()
        emit("  This is the stricter of the two backgrounds where it is higher")
        emit("  than the clean-shadow rate, because it removes the")
        emit("  already-committed-call confound. Quote whichever is LOWER as")
        emit("  the excess -- i.e. take the larger background.")

    c, s = rates["candidate"], rates["shadow_hi"]
    if c[0] and s[0]:
        # Excess and a normal-approximation interval on the difference of two
        # proportions. The shadow rate is the background: Prodigal makes
        # mistakes too, and some of them will land in a shadow's frame.
        p1, n1 = c[2], c[0]
        p2, n2 = s[2], s[0]
        diff = p1 - p2
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        emit()
        emit(f"  excess over matched shadows: {diff * 100:.2f}% "
             f"(95% CI {100 * (diff - 1.96 * se):.2f} - "
             f"{100 * (diff + 1.96 * se):.2f})")
        emit(f"  implied candidates independently called by Prodigal: "
             f"{diff * n1:,.0f} of {n1:,}")
        if p2 > 0:
            emit(f"  enrichment over background: {p1 / p2:.1f}x")

    emit()
    emit("  (The pooled row above is retained for continuity but the clean")
    emit("   comparator is the one to quote.)")
    emit()
    emit("  Caveat: Prodigal over-calls relative to curated annotation, so its")
    emit("  agreement is corroboration and not proof. The shadow arm is what")
    emit("  makes it quantitative -- it measures how often Prodigal endorses an")
    emit("  ORF that we can independently say is NOT the real gene.")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nreport -> {args.out}")

    det = Path(args.out).with_suffix(".coincidence.tsv.gz")
    orfs.drop(columns=["o3"]).to_csv(det, sep="\t", index=False,
                                     compression="gzip")
    print(f"per-ORF  -> {det}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
