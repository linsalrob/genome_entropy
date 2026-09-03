#!/usr/bin/env python3
"""Are high-3Di in_genbank=False ORFs real proteins the annotation missed?

The hypothesis: unmatched ORFs with 3Di entropy above ~2.5 look
structurally like real proteins, so perhaps the original annotation
software simply failed to call them.

Two things have to be separated out before that can be tested.

FIRST, and much the larger effect: a large share of GTDB bacterial
representatives carry no CDS annotation at all. Every ORF in those genomes
is False by construction, and nothing was "missed" because nothing was ever
run. Only genomes with at least one annotated CDS can speak to the
hypothesis, so everything below is restricted to those.

Which genomes those are must come from the GenBank records, not from
in_genbank. That flag is set only when a called ORF passes the coordinate,
frame, and translation match, so a genome whose real CDS features all fail
that strict match looks identical to one that was never annotated. Using
any(in_genbank) as the proxy can therefore only over-count unannotated
genomes, which inflates the share of high-3Di ORFs written off as "never
annotated". Run 12_genome_cds_counts.pbs and pass its table with
--annotation-status.

SECOND, get_orfs reads all six frames, so an unmatched ORF that overlaps an
annotated CDS is usually a shadow of that gene -- an alternative frame or
the opposite strand of real coding sequence. It inherits real sequence
structure and can score high 3Di entropy without being a distinct protein.
A genuine missed gene should sit in intergenic space, overlapping nothing
annotated.

"Overlapping nothing annotated" is tested against the deposited GenBank CDS
coordinates, from 13_cds_intervals.pbs, and not against the spans of ORFs
whose in_genbank flag is True. Those spans mislead in both directions: a CDS
the matcher rejected is absent from them, so an ORF sitting on a real gene
reads as intergenic, and a matched ORF runs stop to stop and can extend past
the deposited CDS, so its overhang manufactures shadows.

Both sides of that test are placed on the forward genomic axis first. ORF
coordinates arrive one-based inclusive and, on the negative strand, indexing
the reverse complement; comparing such a span directly against a plus-strand
one compares two different coordinate systems, inventing overlaps between
distant spans and missing the real cross-strand ones. Since the question is
whether an ORF coincides with coding sequence in ANY frame or orientation,
that conversion cannot be skipped.

That gives three groups among unmatched ORFs in annotated genomes:
  shadow      overlaps an annotated CDS -> explained, not a missed gene
  intergenic  overlaps nothing annotated -> the actual candidate pool
  and each splits on the 3Di >= 2.5 line.

A fourth group comes from the genomes the split above excludes. Genomes
carrying no CDS features at all are dropped from the shadow/intergenic
analysis, because with nothing annotated there is nothing for an ORF to
overlap. Their high-3Di ORFs are kept anyway, as `unannot_hi`: the pilot's
intergenic control is low-3Di by construction (3Di < 2.5), so its hit rate
is confounded with entropy and reports nothing the 3Di quintiles do not
already say (issue #92). `unannot_hi` is the entropy-matched intergenic arm
that control was meant to be. Note what it is and is not: an unannotated
genome's ORFs are a MIXTURE containing many real genes -- nobody ran a
pipeline, not "a pipeline looked and found nothing" -- so this arm is not a
non-coding floor. It asks whether the assay recovers genes in genomes whose
annotation is simply absent, which is the population the missed-gene
hypothesis is ultimately about.

Length is the independent check. Real bacterial proteins have a
characteristic length distribution; spurious ORF calls skew short. If the
high-3Di intergenic ORFs are missed genes, their lengths should resemble
the annotated CDS population rather than the short-ORF background.

Two modes:

  --chunk-tsv <file>   one chunk at a time, writing the candidate table and
                       a one-line stats file. This is what the PBS driver
                       (13_missed_gene_candidates.pbs) runs 760 times.
  --aggregate          read those per-chunk outputs back and print the
                       report over the whole domain.

A chunk can be processed alone because each genome lives in exactly one
chunk, so "does this genome have annotated CDS" and "does this ORF overlap
one" are both answerable within a chunk given that chunk's CDS intervals.

The candidate table carries genome, input_id and orf_id because that triple
is what identifies an ORF inside the per-chunk JSON archives -- the
downstream Foldseek work has to pull amino acids and 3Di back out of them
(issue #92). The entropy TSVs alone cannot serve that search: they keep the
numbers, not the sequences.

Controls are sampled here rather than in a second pass over all 2.57
billion rows: --controls-per-chunk rows of each control group, chosen with
a fixed seed, are written alongside the candidates. Sampled counts must not
be used as population counts -- the stats file keeps the exact totals.
"""
import argparse
import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THRESH = 2.5

# Columns of the per-chunk entropy TSV written by extract_entropy_rows.py.
# Only these are read; the other entropies are not part of this analysis.
#
# contig_length is required. Without it a negative-strand ORF cannot be
# placed on the genomic axis, so TSVs written before extract_entropy_rows.py
# appended that column cannot be used here -- regenerate them from the JSON
# archives rather than dropping the column.
USECOLS = ["domain", "chunk", "genome", "input_id", "orf_id", "start", "end",
           "strand", "aa_length", "in_genbank", "protein_entropy",
           "three_di_entropy", "contig_length"]
DTYPES = {
    "domain": "str", "chunk": "str", "genome": "str", "input_id": "str",
    "orf_id": "str", "strand": "str", "in_genbank": "str",
    "start": "int64", "end": "int64", "aa_length": "int32",
    "protein_entropy": "float32", "three_di_entropy": "float32",
    # float, not int: the column is empty when the encoder wrote no record
    # length, and that has to survive to the check in add_genomic_coordinates
    # rather than blowing up in the parser.
    "contig_length": "float64",
}
OUT_COLS = ["domain", "chunk", "genome", "input_id", "orf_id", "start", "end",
            "strand", "aa_length", "protein_entropy", "three_di_entropy",
            "truncated",
            # The normalised forward-axis interval, carried so that nothing
            # downstream has to redo this conversion. Getting it wrong is
            # what invalidated the first pilot, and 18_pilot_analysis.py's
            # shadow-frame diagnostic then got it wrong a second time,
            # independently. raw start/end are kept alongside because they
            # are the key back into the JSON archives.
            "g_start", "g_end",
            "group"]
STATS_COLS = ["chunk", "genomes", "genomes_annotated",
              "genomes_annotated_matcher_proxy", "orfs",
              "orfs_annotated_genomes", "unmatched", "unmatched_hi",
              "unmatched_hi_shadow", "candidates", "intergenic_lo",
              "annotated_cds", "cds_parts", "unparseable_in_genbank",
              "genomes_unannotated", "orfs_unannotated_genomes",
              "unannot_hi", "unannot_matched", "truncated_at_contig_end"]


def add_genomic_coordinates(df):
    """Place every ORF on the forward genomic axis, as g_start/g_end.

    Returns (frame, n_truncated). n_truncated counts ORFs that run off the
    end of their contig; see the boundary-sentinel note below.

    Vectorised equivalent of genome_entropy.io.genbank.normalise_orf_interval,
    which is the same helper the GenBank CDS matcher uses, so this analysis
    and the in_genbank flag it reads agree on where an ORF is. Row-at-a-time
    calls are correct but this runs over billions of rows, so the two cases
    are evaluated as whole arrays:

        +   (start - 1, end)
        -   (L - end, L - start + 1)

    Results are zero-based half-open, which also removes the off-by-one that
    inclusive coordinates introduce into overlap lengths. Every validation
    the scalar helper performs is kept, because a silently mis-placed ORF
    changes the shadow/intergenic split this script exists to produce.
    """
    if len(df) == 0:
        out = df.copy()
        out["g_start"] = np.zeros(0, dtype=np.int64)
        out["g_end"] = np.zeros(0, dtype=np.int64)
        out["truncated"] = np.zeros(0, dtype=bool)
        return out, 0

    start = df.start.to_numpy(dtype=np.int64)
    end = df.end.to_numpy(dtype=np.int64)
    strand = df.strand.to_numpy(dtype=object)
    length = df.contig_length.to_numpy(dtype="float64")

    missing = int(np.isnan(length).sum())
    if missing:
        raise SystemExit(
            f"{missing:,} rows have no contig_length, so their negative-strand "
            "coordinates cannot be placed on the genomic axis. Regenerate the "
            "chunk TSVs with a current extract_entropy_rows.py."
        )
    length = length.astype(np.int64)

    is_plus = strand == "+"
    is_minus = strand == "-"

    # An ORF that runs off the end of its contig is emitted by get_orfs with
    # end = contig_length + 1, and its dna.length follows the EXCLUSIVE
    # convention (end - start) where a complete ORF uses the inclusive one
    # (end - start + 1). Measured over bac_000: of 1,189,718 ORFs with
    # end <= length, 100% carry has_stop_codon=True; of 28,728 with
    # end > length, 100% carry has_stop_codon=False, and the overhang is
    # exactly 1 in every case on both strands. So the +1 is a boundary
    # sentinel meaning "truncated here", not a coordinate: the bases
    # actually encoded are start..contig_length, which is what clamping
    # gives. 1.6% of all ORFs are affected, and every chunk holds some, so
    # raising on them would abort the whole reclassification.
    #
    # Only an overhang of exactly 1 is the known convention. Anything
    # larger is unexplained input and still falls through to the checks
    # below rather than being quietly clamped into range.
    truncated = (end - length) == 1
    n_truncated = int(truncated.sum())
    end = np.where(truncated, length, end)

    bad_strand = ~(is_plus | is_minus)
    bad_coords = (start < 1) | (end < start)
    # Only the minus strand consults the record length, exactly as the
    # scalar helper does; a plus-strand ORF running past it is not this
    # function's error to raise.
    bad_length = is_minus & (end > length)

    bad = bad_strand | bad_coords | bad_length
    # Refuse to report a shadow/intergenic split computed from a partial
    # coordinate set: dropping ORFs would silently inflate the candidate
    # pool, which is the number this script exists to produce.
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise SystemExit(
            f"{int(bad.sum()):,} ORF(s) could not be placed on the genomic "
            f"axis, e.g. {start[i]}-{end[i]}{strand[i]} on a contig of "
            f"{length[i]}. Fix the inputs rather than analysing a subset."
        )

    g_start = np.where(is_plus, start - 1, length - end)
    g_end = np.where(is_plus, end, length - start + 1)

    out = df.copy()
    out["g_start"] = g_start.astype(np.int64)
    out["g_end"] = g_end.astype(np.int64)
    # Carried per ORF, not just counted, so the pilot and the report can
    # stratify on it: a truncated ORF has no stop codon by construction,
    # which is weaker evidence of a real gene, and that should be testable
    # rather than only noted.
    out["truncated"] = truncated
    return out, n_truncated


def load_annotation_status(path):
    """Return the table and the set of genomes carrying >=1 GenBank CDS.

    Read from 12_genome_cds_counts.pbs, which counts CDS features in the
    source GenBank records. Deliberately not derived from in_genbank: see
    the module docstring.
    """
    status = pd.read_csv(path, sep="\t", dtype={"genome": "str"})
    for column in ("genome", "has_annotation", "n_cds"):
        if column not in status.columns:
            raise SystemExit(
                f"{path} has no '{column}' column; expected the output of "
                "12_genome_cds_counts.pbs."
            )
    if status.has_annotation.dtype != bool:
        status["has_annotation"] = status.has_annotation.astype(str) == "True"
    return status, set(status.loc[status.has_annotation, "genome"])


def load_cds_intervals(directory, genomes, tag=None):
    """Return deposited CDS intervals for the genomes given.

    Read from 13_cds_intervals.pbs: the actual GenBank CDS coordinates,
    zero-based half-open on the forward strand, one row per contiguous part
    of a compound location. Its 'contig' is Biopython's record.id, which is
    the same value the entropy TSVs carry as input_id, so the two join
    directly.

    In per-chunk mode only that chunk's file is read; a chunk's genomes
    cannot appear in another chunk's intervals.
    """
    directory = Path(directory)
    if tag is not None and (directory / f"{tag}.tsv").exists():
        files = [directory / f"{tag}.tsv"]
    else:
        files = sorted(directory.glob("*.tsv"))
    if not files:
        raise SystemExit(
            f"No CDS interval TSVs under {directory}"
            + (f" for chunk {tag}" if tag else "")
            + ". Run 13_cds_intervals.pbs for the chunks being analysed."
        )
    frame = pd.concat(
        [pd.read_csv(f, sep="\t", dtype={"genome": "str", "contig": "str"})
         for f in files],
        ignore_index=True,
    )
    for column in ("genome", "contig", "start", "end"):
        if column not in frame.columns:
            raise SystemExit(
                f"CDS interval files in {directory} have no '{column}' "
                "column; expected the output of 13_cds_intervals.pbs."
            )
    frame = frame[frame.genome.isin(genomes)]

    # Checked per genome, not per contig: a genome whose chromosome is
    # annotated can legitimately have a plasmid or short contig carrying no
    # CDS at all, and that contig having no rows is the correct answer. A
    # whole genome missing means its chunk was never extracted, and treating
    # that as "no CDS anywhere" would report every ORF in it as a candidate.
    missing = genomes - set(frame.genome)
    if missing:
        raise SystemExit(
            f"{len(missing):,} genome(s) in the chunk TSVs have no CDS "
            f"interval rows, e.g. {sorted(missing)[0]}. Every ORF in them "
            "would look intergenic, so this cannot be assumed empty: run "
            "13_cds_intervals.pbs over the same chunks. Genomes genuinely "
            "without CDS features were already removed by STEP 1."
        )
    return frame


def overlaps_annotated(df, cds):
    """Flag each unmatched ORF that overlaps a deposited CDS on its contig.

    Both sides are zero-based half-open on the forward strand, so a
    plus-strand ORF and a minus-strand CDS are comparable. Strand is
    deliberately ignored after that: a shadow in the opposite orientation is
    still a shadow.

    The CDS side comes from the GenBank records, not from in_genbank=True ORF
    spans. Those spans are wrong in both directions -- a CDS the matcher
    rejected is missing from them, and a matched ORF runs stop to stop and
    can extend past the deposited CDS, so its overhang invents shadows.
    """
    flag = np.zeros(len(df), dtype=bool)
    if len(df) == 0 or len(cds) == 0:
        return flag
    cds_by_contig = dict(tuple(cds.groupby("contig", sort=False)))

    for contig, g in df.groupby("input_id", sort=False):
        f = g[~g.in_genbank]
        annotated = cds_by_contig.get(contig)
        if len(f) == 0 or annotated is None or len(annotated) == 0:
            continue
        ts = annotated.start.to_numpy(); te = annotated.end.to_numpy()
        order = np.argsort(ts)
        ts, te = ts[order], te[order]
        # running max of end, so a binary search over starts is sufficient
        te_max = np.maximum.accumulate(te)
        fs = f.g_start.to_numpy(); fe = f.g_end.to_numpy()
        # last CDS whose start < this ORF's end (half-open, so a CDS
        # starting exactly at fe does not overlap)
        idx = np.searchsorted(ts, fe, side="left") - 1
        ok = idx >= 0
        hit = np.zeros(len(f), dtype=bool)
        hit[ok] = te_max[idx[ok]] > fs[ok]
        flag[df.index.get_indexer(f.index)] = hit
    return flag


def load_chunk(path):
    """Read one chunk TSV, returning the frame and a malformed-row count.

    in_genbank is written as the strings True/False. Anything else is a
    parse failure worth reporting rather than silently reading as False,
    which would inflate the candidate pool.
    """
    try:
        df = pd.read_csv(path, sep="\t", usecols=USECOLS, dtype=DTYPES)
    except ValueError as exc:
        raise SystemExit(
            f"{path}: {exc}\n"
            "If contig_length is the missing column, this TSV predates it. "
            "Regenerate the chunk TSVs from the JSON archives with a current "
            "extract_entropy_rows.py: without the contig length a "
            "negative-strand ORF cannot be placed on the genomic axis."
        )
    mapped = df.in_genbank.map({"True": True, "False": False})
    n_bad = int(mapped.isna().sum())
    df = df.loc[mapped.notna()].copy()
    df["in_genbank"] = mapped.loc[mapped.notna()].to_numpy(dtype=bool)
    return df.reset_index(drop=True), n_bad


def classify(df, with_cds, cds_dir, tag, thresh=THRESH):
    """Split one chunk into the groups described in the module docstring.

    Returns (groups, stats). groups maps a label to a frame; stats holds the
    exact population counts, which the sampled control frames cannot give.
    """
    present = set(df.genome.unique())
    annotated = present & with_cds

    # The old in_genbank proxy, kept only to measure how far it was off.
    # Any genome in the difference has real CDS features that no called ORF
    # matched, so the proxy would have discarded its ORFs as never annotated.
    proxy = set(df.groupby("genome").in_genbank.any().pipe(lambda s: s[s].index))

    d = df[df.genome.isin(annotated)].reset_index(drop=True)
    d, n_trunc_ann = add_genomic_coordinates(d)

    if len(d):
        cds = load_cds_intervals(cds_dir, set(d.genome.unique()), tag=tag)
        d["shadow"] = overlaps_annotated(d, cds)
    else:
        cds = d.iloc[:0]
        d["shadow"] = np.zeros(0, dtype=bool)

    unm = d[~d.in_genbank]
    hi = unm[unm.three_di_entropy >= thresh]
    candidates = hi[~hi.shadow]

    # The entropy-matched intergenic arm, from the genomes the split above
    # discards. No CDS features means no annotation to overlap, so the shadow
    # test does not apply and every ORF here is intergenic by construction.
    u = df[~df.genome.isin(annotated)]
    u, n_trunc_un = add_genomic_coordinates(u.reset_index(drop=True))
    # in_genbank cannot be True in a genome with no CDS features to match it.
    # If it is, the matcher and the CDS-count table disagree about the same
    # genome, and the arm would quietly carry matched ORFs; count it so the
    # disagreement is visible in the stats rather than absorbed into the arm.
    unannot_matched = int(u.in_genbank.sum()) if len(u) else 0
    unannot_hi = u[(~u.in_genbank) & (u.three_di_entropy >= thresh)]

    groups = {
        "candidate": candidates,
        "shadow_hi": hi[hi.shadow],
        "intergenic_lo": unm[(unm.three_di_entropy < thresh) & (~unm.shadow)],
        "annotated_cds": d[d.in_genbank],
        "unannot_hi": unannot_hi,
    }
    stats = {
        "genomes": int(df.genome.nunique()),
        "genomes_annotated": len(annotated),
        "genomes_annotated_matcher_proxy": len(present & proxy),
        "orfs": len(df),
        "orfs_annotated_genomes": len(d),
        "unmatched": len(unm),
        "unmatched_hi": len(hi),
        "unmatched_hi_shadow": int(hi.shadow.sum()),
        "candidates": len(candidates),
        "intergenic_lo": len(groups["intergenic_lo"]),
        "annotated_cds": len(groups["annotated_cds"]),
        "cds_parts": len(cds),
        "genomes_unannotated": len(present - annotated),
        "orfs_unannotated_genomes": len(u),
        "unannot_hi": len(unannot_hi),
        "unannot_matched": unannot_matched,
        "truncated_at_contig_end": n_trunc_ann + n_trunc_un,
    }
    return groups, stats


def write_chunk(path, out_dir, annotation_status, cds_dir, thresh,
                n_controls, seed, quiet):
    tag = os.path.basename(path).split(".tsv")[0]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / f"{tag}.candidates.tsv.gz"
    ctrl_path = out_dir / f"{tag}.controls.tsv.gz"
    stats_path = out_dir / f"{tag}.stats.tsv"

    df, n_bad = load_chunk(path)

    status, with_cds = load_annotation_status(annotation_status)
    # Only a genome absent from the table altogether is an error; one that is
    # present and marked False is a legitimate "carries no annotation".
    unknown = set(df.genome.unique()) - set(status.genome)
    if unknown:
        raise SystemExit(
            f"{len(unknown):,} genome(s) in {tag} are absent from "
            f"{annotation_status}, e.g. {sorted(unknown)[0]}. Rerun "
            "12_genome_cds_counts.pbs over the same chunks rather than "
            "assuming their annotation status."
        )

    groups, stats = classify(df, with_cds, cds_dir, tag, thresh)
    stats["chunk"] = tag
    stats["unparseable_in_genbank"] = n_bad

    cand = groups["candidate"].assign(group="candidate")
    _write(cand_path, cand)

    # Controls are a fixed-seed sample: the full annotated-CDS population of
    # a chunk is ~1.5 M rows and is not worth writing 760 times over.
    rng = np.random.default_rng(seed)
    parts = []
    for label in ("annotated_cds", "intergenic_lo", "shadow_hi", "unannot_hi"):
        g = groups[label]
        if len(g) > n_controls:
            take = rng.choice(len(g), size=n_controls, replace=False)
            g = g.iloc[np.sort(take)]
        parts.append(g.assign(group=label))
    _write(ctrl_path, pd.concat(parts, ignore_index=True) if parts else None)

    pd.DataFrame([stats])[STATS_COLS].to_csv(stats_path, sep="\t", index=False)

    if not quiet:
        print(f"{tag}: {stats['candidates']:,} candidates from "
              f"{stats['orfs']:,} ORFs in {stats['genomes']} genomes"
              f"{f', {n_bad} unparseable in_genbank' if n_bad else ''}")
    return 0


def _write(path, frame):
    if frame is None or len(frame) == 0:
        # An empty table still has to exist, or the driver cannot tell a
        # finished chunk from an unstarted one.
        with gzip.open(path, "wt") as fh:
            fh.write("\t".join(OUT_COLS) + "\n")
        return
    frame[OUT_COLS].to_csv(path, sep="\t", index=False, compression="gzip")


def aggregate(out_dir, thresh):
    out_dir = Path(out_dir)
    stats_files = sorted(out_dir.glob("*.stats.tsv"))
    if not stats_files:
        print(f"ERROR: no per-chunk stats under {out_dir}", file=sys.stderr)
        return 1
    s = pd.concat([pd.read_csv(f, sep="\t", dtype={"chunk": "str"})
                   for f in stats_files], ignore_index=True)
    tot = s.drop(columns=["chunk"]).sum()

    print(f"chunks aggregated: {len(s)}")
    if int(tot.unparseable_in_genbank):
        print(f"  unparseable in_genbank rows: {int(tot.unparseable_in_genbank):,}")
    print(f"ORFs loaded: {int(tot.orfs):,} from {int(tot.genomes):,} genomes\n")

    print("STEP 1 - remove genomes that were never annotated")
    print(f"  genomes with >=1 GenBank CDS   : {int(tot.genomes_annotated):,} of "
          f"{int(tot.genomes):,} ({tot.genomes_annotated/tot.genomes*100:.0f}%)")
    if "genomes_annotated_matcher_proxy" in tot:
        delta = int(tot.genomes_annotated) - int(tot.genomes_annotated_matcher_proxy)
        if delta:
            print(f"  of which no ORF matched a CDS  : {delta:,} "
                  f"({delta/tot.genomes_annotated*100:.1f}% of annotated) "
                  f"<- counted as unannotated by the old in_genbank proxy")
    print(f"  ORFs in those genomes          : {int(tot.orfs_annotated_genomes):,}")
    print(f"  deposited CDS parts read       : {int(tot.cds_parts):,}")
    if "truncated_at_contig_end" in tot:
        n_t = int(tot.truncated_at_contig_end)
        print(f"  truncated at a contig end      : {n_t:,} "
              f"({n_t/max(int(tot.orfs), 1)*100:.2f}% of ORFs) "
              f"<- end clamped to the contig length")
    print()

    print("STEP 2 - within annotated genomes, separate shadows from intergenic")
    print(f"  unmatched ORFs                  : {int(tot.unmatched):,}")
    print(f"    3Di >= {thresh}                     : {int(tot.unmatched_hi):,}")
    print(f"      overlapping an annotated CDS : {int(tot.unmatched_hi_shadow):,} "
          f"({tot.unmatched_hi_shadow/tot.unmatched_hi*100:.1f}%)  <- shadow of a real gene")
    print(f"      intergenic                   : {int(tot.candidates):,} "
          f"({tot.candidates/tot.unmatched_hi*100:.1f}%)  <- CANDIDATE missed genes\n")

    if "unannot_hi" in tot:
        # 1b, not 3: the later "does the candidate pool look like real
        # protein" section is already STEP 3, and this belongs with STEP 1
        # anyway -- it is what STEP 1 discarded.
        print("STEP 1b - the genomes step 1 removed, kept as the "
              "entropy-matched arm")
        n_un = int(tot.genomes_unannotated)
        print(f"  genomes with no GenBank CDS     : {n_un:,} "
              f"({n_un/tot.genomes*100:.0f}%)")
        print(f"    ORFs in them                  : "
              f"{int(tot.orfs_unannotated_genomes):,}")
        print(f"    3Di >= {thresh}                     : "
              f"{int(tot.unannot_hi):,}  <- unannot_hi arm")
        if int(tot.unannot_matched):
            # Not fatal, but it means the two annotation sources disagree,
            # and silence here would let matched ORFs sit in an arm defined
            # by there being nothing for them to match.
            print(f"  WARNING: {int(tot.unannot_matched):,} ORF(s) in these "
                  f"genomes have in_genbank=True, which cannot happen if the "
                  f"genome truly has no CDS features. Excluded from the arm; "
                  f"reconcile 12_genome_cds_counts.pbs against the matcher.")
        print()

    cand = _read_all(out_dir, "*.candidates.tsv.gz")
    ctrl = _read_all(out_dir, "*.controls.tsv.gz")
    if cand is None:
        return 0

    merged = out_dir / f"candidates_{cand.domain.iloc[0]}.tsv.gz"
    cand.to_csv(merged, sep="\t", index=False, compression="gzip")
    print(f"  candidate table -> {merged}  ({len(cand):,} rows)\n")

    print("STEP 3 - does the candidate pool look like real protein?")
    print("  candidates are the full population; control rows are a per-chunk")
    print("  sample, so read their n as a sample size and not a count.")
    print(f"  {'group':<36}{'n':>12}{'med aa':>8}{'%>=100aa':>10}{'med 3Di':>9}")
    rows = [("candidate: intergenic, 3Di >= %.1f" % thresh, cand)]
    if ctrl is not None:
        for label, name in (("annotated_cds", "annotated CDS (sampled)"),
                            ("intergenic_lo", "intergenic, 3Di < %.1f (sampled)" % thresh),
                            ("shadow_hi", "shadow of a CDS, 3Di >= %.1f (sampled)" % thresh)):
            rows.append((name, ctrl[ctrl.group == label]))
    for name, g in rows:
        if len(g) == 0:
            continue
        print(f"  {name:<36}{len(g):>12,}{g.aa_length.median():>8.0f}"
              f"{(g.aa_length >= 100).mean()*100:>9.1f}%"
              f"{g.three_di_entropy.median():>9.2f}")

    per = cand.groupby("genome").size()
    print("\n  per annotated genome:")
    print(f"    genomes carrying >=1 candidate         : {len(per):,} of "
          f"{int(tot.genomes_annotated):,}")
    print(f"    candidates per such genome (median)    : {per.median():.0f}")
    print(f"    candidates per annotated genome (mean) : "
          f"{len(cand)/tot.genomes_annotated:.1f}")
    return 0


def _read_all(out_dir, pattern):
    files = sorted(Path(out_dir).glob(pattern))
    frames = [f for f in (pd.read_csv(p, sep="\t", dtype={"chunk": "str"})
                          for p in files) if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chunk-tsv", help="one per-chunk entropy TSV (.tsv.gz)")
    ap.add_argument("--aggregate", action="store_true",
                    help="report over the per-chunk outputs already written")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--annotation-status",
        help="genome_cds_counts_<domain>.tsv from 12_genome_cds_counts.pbs. "
             "Required with --chunk-tsv: annotation presence must come from "
             "the GenBank records, not from in_genbank.")
    ap.add_argument(
        "--cds-intervals",
        help="directory of per-chunk CDS interval TSVs from "
             "13_cds_intervals.pbs. Required with --chunk-tsv: the shadow "
             "test needs the deposited CDS coordinates, not the spans of "
             "ORFs that happened to match.")
    ap.add_argument("--threshold", type=float, default=THRESH)
    ap.add_argument("--controls-per-chunk", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        return aggregate(args.out_dir, args.threshold)
    if not args.chunk_tsv:
        ap.error("pass --chunk-tsv <file> or --aggregate")
    # Required for the chunk path only: --aggregate re-reads finished output.
    missing = [n for n, v in (("--annotation-status", args.annotation_status),
                              ("--cds-intervals", args.cds_intervals))
               if not v]
    if missing:
        ap.error(
            f"{' and '.join(missing)} required with --chunk-tsv. The "
            "annotated-genome set and the shadow test must come from the "
            "GenBank records; deriving either from in_genbank reproduces the "
            "defect this script was corrected for.")
    return write_chunk(args.chunk_tsv, args.out_dir, args.annotation_status,
                       args.cds_intervals, args.threshold,
                       args.controls_per_chunk, args.seed, args.quiet)


if __name__ == "__main__":
    sys.exit(main())
