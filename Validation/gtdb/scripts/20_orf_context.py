#!/usr/bin/env python3
"""Genomic context and the remaining entropy axes, for every ORF in a query set.

Issue #97 asks for a ranked candidate table carrying "genome_entropy coding
probability if available" and "genomic context". The first does not exist --
there is no calibrated coding-probability model in the package. The second is
essentially free, and this script computes it, along with the two entropy axes
that never reached the candidate tables.

WHY THIS COVERS THE WHOLE WANTED LIST AND NOT JUST THE CANDIDATES

Because context is only evidence in contrast. "Candidates sit in operon-like
spacing" means nothing on its own -- every ORF in a dense bacterial genome has
a CDS nearby. The claim that can carry weight is "candidates sit in operon-like
spacing MORE OFTEN THAN their length- and 3Di-matched shadows do", and that
needs the identical measurement on both arms. Computing it for candidates alone
would produce a number no referee should accept.

That contrast is also a line of evidence completely independent of Foldseek,
structural databases and the mixture model, which is why it is worth the pass.

WHAT COMES FROM WHERE

  entropy_rows/<domain>/<chunk>.tsv.gz
      twelve_state_entropy and three_di_twelve_state_mutual_information --
      present per ORF but never carried into the candidate tables, which stop
      at protein_entropy and three_di_entropy. MI in particular is a different
      axis from either entropy alone.
      Also start/end/strand/contig_length, which is what lets this place the
      CONTROL arms on the forward genomic axis: the controls table carries raw
      coordinates and no contig_length, so g_start/g_end cannot be recovered
      from it. Getting this wrong is the coordinate defect that swapped two
      arms once already.

  cds_intervals/<domain>/<chunk>.tsv
      every deposited GenBank CDS, on the same forward axis.

A TRAP IN THE SPACING COMPARISON, FOUND ON THE FIRST TEST CHUNK

Do NOT compare dist_up/dist_down/gap_len between candidate and shadow_hi.
A shadow OVERLAPS a deposited CDS by definition, so its nearest
NON-overlapping neighbour is measured past the far end of the gene it sits
inside. Its distances are inflated by the arm definition, not by biology.

On bac_000 that artefact looks exactly like a result: candidates at a median
125/146 bp from their neighbours in a 720 bp gap, against shadows at 351/390
in a 1,319 bp gap -- and candidates land right on top of real annotated CDS
(113.5/95, 734). Reported naively that is a headline. It is measurement.

The comparisons that survive:
  * candidate vs intergenic_lo -- both non-overlapping, length-matched
    (confounded by 3Di, which is what unannot_hi was added to address, though
    unannot_hi genomes have NO CDS at all so they have no context to measure:
    dist_up/dist_down/gap_len are all -1 there, correctly).
  * gap occupancy -- 3 * aa_length against gap_len. Asks whether the ORF fills
    the space it sits in, which needs no second arm.
  * strand coherence -- up_strand, down_strand and the ORF's own strand, i.e.
    operon-like orientation. Also needs no second arm.

A SELF-CHECK THAT COMES FREE

Candidates are DEFINED as not overlapping any deposited CDS, and shadow_hi is
defined as overlapping one. This script recomputes overlap from the interval
tables independently of the classifier, so `overlaps_cds` must be False for
100% of the candidate arm and True for 100% of shadow_hi. Anything else means
the classification and this script disagree about where ORFs are, which is
exactly the failure that produced the withdrawn first pilot. Reported per chunk
rather than assumed.

  20_orf_context.py --wanted wanted.tsv --chunk bac_000 --out-dir context/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def add_genomic_coordinates(df):
    """Forward-axis g_start/g_end. Kept identical to 10_missed_genes.py.

    Duplicated rather than imported because the module name starts with a
    digit and is not importable without exec gymnastics; the arithmetic is
    four lines and the alternative is an import that fails silently under a
    different working directory.

        +   (start - 1, end)
        -   (L - end, L - start + 1)
    """
    start = df.start.to_numpy(dtype=np.int64)
    end = df.end.to_numpy(dtype=np.int64)
    strand = df.strand.to_numpy(dtype=object)
    length = df.contig_length.to_numpy(dtype="float64")
    if np.isnan(length).any():
        raise SystemExit(
            f"{int(np.isnan(length).sum()):,} rows have no contig_length; "
            "regenerate the chunk TSVs with a current extract_entropy_rows.py."
        )
    length = length.astype(np.int64)

    # end == contig_length + 1 is get_orfs' sentinel for an ORF running off the
    # end of a contig; clamp it, and keep the flag, exactly as the classifier
    # does. Unclamped, these place outside the contig and every neighbour
    # distance computed from them is nonsense.
    truncated = (end - length) == 1
    end = np.where(truncated, length, end)

    is_plus = strand == "+"
    g_start = np.where(is_plus, start - 1, length - end)
    g_end = np.where(is_plus, end, length - start + 1)

    out = df.copy()
    out["g_start"] = g_start.astype(np.int64)
    out["g_end"] = g_end.astype(np.int64)
    out["truncated_calc"] = truncated
    return out


def context_for_contig(orfs, cds):
    """Neighbour distances for the ORFs of one contig against its CDS.

    cds must be sorted by start. Everything is on the forward zero-based
    half-open axis, so an ORF [a,b) and a CDS [c,d) overlap iff a < d and c < b.
    """
    n = len(orfs)
    cols = {
        "n_cds_contig": np.full(n, len(cds), dtype=np.int64),
        "overlaps_cds": np.zeros(n, dtype=bool),
        "dist_up": np.full(n, -1, dtype=np.int64),
        "dist_down": np.full(n, -1, dtype=np.int64),
        "gap_len": np.full(n, -1, dtype=np.int64),
        "fits_in_gap": np.zeros(n, dtype=bool),
        "up_strand": np.full(n, "", dtype=object),
        "down_strand": np.full(n, "", dtype=object),
        "up_cds_id": np.full(n, "", dtype=object),
        "down_cds_id": np.full(n, "", dtype=object),
        # What an overlapping ORF is a shadow OF. Same strand and same reading
        # frame means the ORF largely IS the annotated protein, so it should
        # score like a real gene and is a useless comparator; antisense and
        # frameshifted overlaps are genuinely different amino acids. Computed
        # here rather than in the analysis so every arm carries it and the
        # ladder can drop same-frame shadows from its background.
        "cds_frame_class": np.full(n, "", dtype=object),
        "cds_overlap_id": np.full(n, "", dtype=object),
        # Whether that CDS part could serve as a reading-frame reference at
        # all. Emitted so a reader can see WHY a class came out undefined
        # rather than having to re-derive it.
        "cds_frame_reference": np.zeros(n, dtype=bool),
    }
    if len(cds) == 0:
        return cols

    c_start = cds.start.to_numpy(dtype=np.int64)
    c_end = cds.end.to_numpy(dtype=np.int64)
    c_strand = cds.strand.to_numpy(dtype=object)
    c_id = cds.cds_id.to_numpy(dtype=object)
    # Is this CDS part usable as a reading-frame reference at all? Only if it
    # is a whole single-part CDS whose length is a multiple of three. See
    # frame_class() for why.
    c_ref = cds.frame_reference.to_numpy(dtype=bool)
    # Running maximum of the CDS end. Sorting by start does NOT sort by end --
    # a long CDS can contain a short one -- so "does any earlier CDS reach past
    # this point" needs the cumulative max, not c_end[i-1]. Using the latter
    # would miss overlaps with nested genes.
    c_end_max = np.maximum.accumulate(c_end)
    # ...and the INDEX that supplied each running maximum, which is not
    # necessarily the index itself. Where a later CDS is nested inside an
    # earlier longer one, dist_up came from the earlier CDS while up_strand and
    # up_cds_id were read from the nested one -- two different genes reported as
    # one neighbour, which would corrupt the operon-orientation call. Rare
    # (measured: 10 of 2,080,688 archaeal CDS parts) but wrong.
    _idx = np.arange(len(c_end))
    c_end_max_arg = np.maximum.accumulate(
        np.where(c_end == c_end_max, _idx, -1))

    a = orfs.g_start.to_numpy(dtype=np.int64)
    b = orfs.g_end.to_numpy(dtype=np.int64)

    # First CDS starting at or after the ORF's end -> the downstream neighbour.
    j_down = np.searchsorted(c_start, b, side="left")
    # Last CDS ending at or before the ORF's start -> the upstream neighbour.
    # searchsorted on the running max gives the first index whose reach exceeds
    # a; everything before it ends at or before a.
    j_up = np.searchsorted(c_end_max, a, side="right") - 1

    has_down = j_down < len(cds)
    has_up = j_up >= 0

    cols["dist_down"] = np.where(has_down, c_start[np.clip(j_down, 0, len(cds) - 1)] - b, -1)
    j_up_c = np.clip(j_up, 0, len(cds) - 1)
    # The CDS that actually reaches furthest, not merely the one at j_up.
    j_up_src = np.clip(c_end_max_arg[j_up_c], 0, len(cds) - 1)
    cols["dist_up"] = np.where(has_up, a - c_end_max[j_up_c], -1)
    cols["down_strand"] = np.where(has_down, c_strand[np.clip(j_down, 0, len(cds) - 1)], "")
    cols["up_strand"] = np.where(has_up, c_strand[j_up_src], "")
    cols["down_cds_id"] = np.where(has_down, c_id[np.clip(j_down, 0, len(cds) - 1)], "")
    cols["up_cds_id"] = np.where(has_up, c_id[j_up_src], "")

    both = has_up & has_down
    gap = np.where(both,
                   c_start[np.clip(j_down, 0, len(cds) - 1)]
                   - c_end_max[j_up_c],
                   -1)
    cols["gap_len"] = gap
    cols["fits_in_gap"] = both & (gap > 0) & (cols["dist_up"] >= 0) & (cols["dist_down"] >= 0)

    # Overlap: any CDS with c_start < b and c_end > a. The upstream index
    # already encodes "everything at or before j_up ends by a", so an overlap
    # exists iff some CDS strictly after j_up starts before b.
    k = np.searchsorted(c_start, b, side="left")          # CDS starting before b
    cols["overlaps_cds"] = k > (j_up + 1)

    # Frame class, for the overlapping ORFs only. The convention matches
    # 18_pilot_analysis.py::shadow_frames exactly -- same strand plus
    # (g_start - cds_start) % 3 == 0 -- so the two do not drift apart.
    o_strand = orfs.strand.to_numpy(dtype=object)
    for i in np.flatnonzero(cols["overlaps_cds"]):
        hit = (c_start < b[i]) & (c_end > a[i])
        if not hit.any():
            continue
        idx = np.flatnonzero(hit)
        # The largest overlap is the CDS this ORF is a shadow of.
        width = (np.minimum(c_end[idx], b[i]) - np.maximum(c_start[idx], a[i]))
        j = idx[int(np.argmax(width))]
        cols["cds_overlap_id"][i] = c_id[j]
        cols["cds_frame_reference"][i] = bool(c_ref[j])
        cols["cds_frame_class"][i] = frame_class(
            int(a[i]), int(b[i]), o_strand[i],
            int(c_start[j]), int(c_end[j]), c_strand[j], bool(c_ref[j]))
    return cols


def frame_class(g_start, g_end, o_strand, c_start, c_end, c_strand, c_is_ref):
    """Reading-frame relationship between an ORF and the CDS it overlaps.

    THE ANCHOR DEPENDS ON THE STRAND, and getting that wrong was a real bug.

    A plus-strand feature is translated left to right from g_start, so its
    codon boundaries are g_start + 3k and its frame is g_start % 3. A
    minus-strand feature is translated right to left from its far end, so its
    boundaries are g_end - 3k and its frame is g_end % 3. Two same-strand
    features share a frame iff their anchors are congruent mod 3 -- g_start
    for plus, g_end for minus.

    The previous code used g_start for BOTH strands. Verified against
    hand-built cases: the 5'-anchored test is correct on plus strand and WRONG
    on minus, and the 3'-anchored test is exactly the reverse; each is wrong on
    2 of 11 ground-truth cases and only the strand-dependent anchor is right on
    all of them.

    WHY THAT BUG WAS MOSTLY INVISIBLE, AND WHERE IT WAS NOT.

    If both features have a length that is a multiple of three then
    g_end - g_start and c_end - c_start are both 0 mod 3, so the two anchors
    give identical verdicts and the anchor choice cannot matter. Measured:
    complete ORFs are 100% divisible by three and 95.6% of deposited CDS parts
    are, so the anchors agree on the overwhelming majority of pairs.

    They disagree exactly where the CDS is not a valid frame reference. Among
    same-strand shadow/CDS pairs only 63.4% of the overlapping CDS parts have a
    length divisible by three -- a 9x enrichment over the 3.9% seen
    genome-wide -- and 95.8% of those are SINGLE-part CDS, so compound
    `join(...)` features are not the cause. They are PARTIAL CDS: features
    annotated with fuzzy ends (`<1..500`) because the gene runs off a contig
    edge or is otherwise incomplete.

    A partial CDS has no true codon boundary at the truncated end, so its
    frame is not recoverable from its coordinates and NEITHER anchor is
    meaningful. Forcing such a pair into "same frame" or "frameshift" is a coin
    flip, and it was scattering genuinely-same-frame shadows into the
    frameshift class -- which is the clean comparator, i.e. contaminating the
    background with ORFs that ARE the annotated protein.

    So those pairs get their own class and the consumers, which all select
    "clean" by allow-list, drop them automatically. Fewer shadows, honestly
    classified, rather than more shadows with a fabricated verdict.
    """
    if c_strand != o_strand:
        return "opposite strand"
    if not c_is_ref:
        return "same strand, frame undefined"
    anchor_ok = ((g_start - c_start) % 3 == 0 if o_strand == "+"
                 else (g_end - c_end) % 3 == 0)
    return ("same strand, same frame" if anchor_ok
            else "same strand, frameshift")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wanted", required=True)
    ap.add_argument("--chunk", required=True, help="chunk tag, e.g. bac_000")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--entropy-rows",
                    default="/g/data/ob80/re3494/gtdb_entropy/entropy_rows")
    ap.add_argument("--cds-intervals",
                    default="/g/data/ob80/re3494/gtdb_entropy/cds_intervals")
    args = ap.parse_args()

    tag = args.chunk
    domain = tag.split("_")[0]
    out = Path(args.out_dir) / f"{tag}.context.tsv.gz"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    wanted = pd.read_csv(args.wanted, sep="\t", dtype={"chunk": str})
    wanted = wanted[(wanted.domain.astype(str) + "_" + wanted.chunk.astype(str)) == tag]
    if not len(wanted):
        print(f"{tag}: nothing wanted from this chunk")
        return 0
    # The key MUST include input_id. orf ids are unique per contig, not per
    # genome, so genome+orf_id alone pulls in same-named ORFs from other
    # contigs -- the bug that once inflated the shadow arm.
    key_cols = ["genome", "input_id", "orf_id"]

    def keyof(d):
        return (d.genome.astype(str) + "@" + d.input_id.astype(str)
                + "@" + d.orf_id.astype(str))

    want_keys = set(keyof(wanted))

    rows_path = Path(args.entropy_rows) / domain / f"{tag}.tsv.gz"
    if not rows_path.exists():
        raise SystemExit(f"ERROR: no {rows_path}")
    rows = pd.read_csv(
        rows_path, sep="\t",
        # protein_entropy and three_di_entropy were omitted here at first,
        # so they were absent from every downstream table -- including the
        # ranked candidate table, where #97 lists both explicitly. They live
        # in entropy_rows alongside the others; there was never a reason to
        # leave them out.
        usecols=["genome", "input_id", "orf_id", "start", "end", "strand",
                 "aa_length", "in_genbank", "dna_entropy",
                 "protein_entropy", "three_di_entropy",
                 "twelve_state_entropy",
                 "three_di_twelve_state_mutual_information", "contig_length"],
        dtype={"genome": str, "input_id": str, "orf_id": str})
    rows = rows[keyof(rows).isin(want_keys)]

    # Every wanted ORF must be found. A silent shortfall here would publish a
    # context table that downstream code joins as if it covered the arm.
    if len(rows) != len(wanted):
        print(f"ERROR: {tag}: matched {len(rows):,} of {len(wanted):,} wanted "
              f"ORFs in {rows_path.name}", file=sys.stderr)
        return 1

    rows = add_genomic_coordinates(rows)

    cds_path = Path(args.cds_intervals) / domain / f"{tag}.tsv"
    if not cds_path.exists():
        raise SystemExit(f"ERROR: no {cds_path}")
    cds = pd.read_csv(cds_path, sep="\t",
                      dtype={"genome": str, "contig": str, "cds_id": str})
    # A CDS part is a usable reading-frame reference only if it is the WHOLE
    # CDS (single part -- a piece of a join() carries the gene's frame only
    # with its cumulative phase, which is not in this table) and its length is
    # a multiple of three (otherwise an end is fuzzy/partial and is not a
    # codon boundary).
    n_parts = (cds.genome + "@" + cds.cds_id).value_counts()
    cds["n_parts"] = (cds.genome + "@" + cds.cds_id).map(n_parts).astype(int)
    cds["frame_reference"] = ((cds.n_parts == 1)
                              & (((cds.end - cds.start) % 3) == 0))
    cds = cds.sort_values(["genome", "contig", "start"], kind="stable")

    by_contig = {k: v for k, v in cds.groupby(["genome", "contig"], sort=False)}
    empty = cds.iloc[:0]

    parts = []
    for (genome, contig), g in rows.groupby(["genome", "input_id"], sort=False):
        g = g.reset_index(drop=True)
        c = by_contig.get((genome, contig), empty)
        ctx = context_for_contig(g, c)
        parts.append(pd.concat([g, pd.DataFrame(ctx, index=g.index)], axis=1))

    ctx = pd.concat(parts, ignore_index=True)
    ctx = ctx.merge(wanted[key_cols + ["group"]].astype({c: str for c in key_cols}),
                    on=key_cols, how="left")
    if ctx.group.isna().any():
        print(f"ERROR: {tag}: {int(ctx.group.isna().sum()):,} rows lost their "
              "arm label in the merge", file=sys.stderr)
        return 1

    ctx.to_csv(out, sep="\t", index=False, compression="gzip")

    # The free self-check: the classifier and this script must agree on which
    # ORFs touch a deposited CDS.
    bad = []
    for arm, expect in (("candidate", False), ("shadow_hi", True)):
        sub = ctx[ctx.group == arm]
        if not len(sub):
            continue
        wrong = int((sub.overlaps_cds != expect).sum())
        if wrong:
            bad.append(f"{arm}: {wrong:,} of {len(sub):,} have "
                       f"overlaps_cds != {expect}")
    sh = ctx[ctx.group == "shadow_hi"]
    frames = ""
    if len(sh):
        vc = sh.cds_frame_class.value_counts()
        frames = ("  shadow frames: "
                  + " ".join(f"{k.split(',')[-1].strip()[:9]}={v}"
                             for k, v in vc.items()))
    print(f"{tag}: {len(ctx):,} ORFs -> {out.name}"
          + ("  OVERLAP DISAGREEMENT: " + "; ".join(bad) if bad else "  overlap check OK")
          + frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
