#!/usr/bin/env python3
"""Read out the Foldseek pilot: does structure find genes that sequence misses?

Searches matched arms -- candidate, shadow_hi, annotated_cds, intergenic_lo
and, from the second run on, unannot_hi -- against four target databases twice
over, once with
3Di+amino acid (`foldseek search
--alignment-type 2`) and once amino acid only (`foldseek base:search`), plus
a shuffled-3Di null on the candidate arm. Same queries, same targets, one
difference: whether the structural alphabet was allowed to contribute.

That design makes three quantities readable directly:

  structure-only homology  a query with a struct hit and no seq hit, on the
                           same target database, is a hit the amino-acid
                           side could not find
  the positive control     annotated_cds sets the ceiling a real, annotated,
                           length-matched gene achieves
  the noise floor          intergenic_lo (biological) and the shuffled-3Di
                           null (technical) bracket what a non-gene scores.
                           intergenic_lo is low-3Di by construction, so it
                           is confounded with entropy and is kept for
                           continuity rather than as evidence.
  the recovery check       unannot_hi -- 3Di-matched ORFs from genomes with
                           no annotation at all. A mixture, not a floor: if
                           the assay cannot recover genes where annotation
                           is merely absent, a null on the candidate arm is
                           uninformative rather than negative.

The comparison the pilot exists to make is candidate against shadow_hi:
matched on length, 3Di entropy and -- for 98.5% of pairs -- genome, so N50,
completeness, contamination, GC and annotation pipeline are all held fixed
(see the burden analysis, 16_candidate_burden.py).

  18_pilot_analysis.py --search-dir .../pilot/search --wanted .../wanted4.tsv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

GD = "/g/data/ob80/re3494/gtdb_entropy"
# Canonical display order. main() narrows this to the arms the wanted list
# actually carries: unannot_hi was added after the first pilot ran, and the
# per-arm tables index sizes[arm] directly, so a missing arm would be a
# KeyError rather than an absent column.
ARMS = ["candidate", "shadow_hi", "annotated_cds", "intergenic_lo",
        "unannot_hi"]
M8_COLS = ["query", "target", "fident", "alnlen", "qlen", "tlen", "qcov",
           "tcov", "evalue", "bits", "taxid", "taxname"]


def load_m8(path):
    """Best hit per query: lowest E-value, ties broken by bit score.

    One row per query is the right unit here -- the question is "does this
    ORF have a homolog", not "how many". Keeping every hit would let a
    single query with 4,000 hits to one superfamily dominate any average.
    """
    if not os.path.getsize(path):
        return pd.DataFrame(columns=M8_COLS)
    df = pd.read_csv(path, sep="\t", names=M8_COLS, low_memory=False)
    df = df.sort_values(["evalue", "bits"], ascending=[True, False])
    return df.drop_duplicates("query", keep="first")


def parse_ids(s):
    """Query ids are genome|input_id|orf_id|group."""
    p = s.str.split("|", expand=True)
    p.columns = ["genome", "input_id", "orf_id", "group"][:p.shape[1]]
    return p


def pct(n, d):
    return f"{100.0 * n / d:5.1f}%" if d else "    --"


def hit_rate_table(best, sizes, dbs):
    print("\n" + "=" * 78)
    # n was hardcoded as "8,651 per arm" -- true only of the FIRST pilot, and
    # wrong for pilot_v2 (8,712), for archaea (22,447) and for the full
    # bacterial run, where the arms are deliberately unequal: every candidate
    # plus a 1:1 shadow, against 200-per-chunk control arms. Print what was
    # actually searched. The same class of defect as the figure that titled
    # archaeal panels "bacterial ORFs".
    print("HIT RATES -- queries with at least one hit at E < 1e-3")
    print("=" * 78)
    print("  arm sizes: " + ",  ".join(f"{a} {sizes[a]:,}" for a in ARMS
                                       if a in sizes))
    for db in dbs:
        print(f"\n  {db}")
        print(f"    {'arm':<16}{'struct':>10}{'seq':>10}{'struct-only':>14}"
              f"{'seq-only':>11}{'both':>9}")
        for arm in ARMS:
            st = best.get((db, "struct"), pd.DataFrame())
            sq = best.get((db, "seq"), pd.DataFrame())
            s_ids = set(st.loc[st.group == arm, "query"]) if len(st) else set()
            q_ids = set(sq.loc[sq.group == arm, "query"]) if len(sq) else set()
            n = sizes[arm]
            print(f"    {arm:<16}{pct(len(s_ids), n):>10}{pct(len(q_ids), n):>10}"
                  f"{pct(len(s_ids - q_ids), n):>14}"
                  f"{pct(len(q_ids - s_ids), n):>11}"
                  f"{pct(len(s_ids & q_ids), n):>9}")
        nl = best.get((db, "null"), pd.DataFrame())
        n_null = len(set(nl["query"])) if len(nl) else 0
        print(f"    {'[null, shuffled]':<16}{pct(n_null, sizes['candidate']):>10}"
              "        --            --         --        --")


def union_table(best, sizes, dbs):
    """Any database at all -- the number a downstream reader wants."""
    print("\n" + "=" * 78)
    print("UNION ACROSS ALL FOUR DATABASES")
    print("=" * 78)
    print(f"  {'arm':<16}{'struct':>10}{'seq':>10}{'struct-only':>14}"
          f"{'seq-only':>11}{'neither':>10}")
    out = {}
    for arm in ARMS:
        s_ids, q_ids = set(), set()
        for db in dbs:
            st = best.get((db, "struct"), pd.DataFrame())
            sq = best.get((db, "seq"), pd.DataFrame())
            if len(st):
                s_ids |= set(st.loc[st.group == arm, "query"])
            if len(sq):
                q_ids |= set(sq.loc[sq.group == arm, "query"])
        n = sizes[arm]
        out[arm] = (s_ids, q_ids)
        print(f"  {arm:<16}{pct(len(s_ids), n):>10}{pct(len(q_ids), n):>10}"
              f"{pct(len(s_ids - q_ids), n):>14}{pct(len(q_ids - s_ids), n):>11}"
              f"{pct(n - len(s_ids | q_ids), n):>10}")
    n_null = set()
    for db in dbs:
        nl = best.get((db, "null"), pd.DataFrame())
        if len(nl):
            n_null |= set(nl["query"])
    print(f"  {'[null, shuffled]':<16}{pct(len(n_null), sizes['candidate']):>10}")
    return out


def quality_table(best, dbs):
    """A hit rate says nothing about whether the hits are any good."""
    print("\n" + "=" * 78)
    print("BEST-HIT QUALITY (median over queries that hit), struct mode")
    print("=" * 78)
    print(f"  {'db':<16}{'arm':<16}{'fident':>8}{'qcov':>8}{'tcov':>8}"
          f"{'bits':>8}{'-log10 E':>10}")
    for db in dbs:
        st = best.get((db, "struct"), pd.DataFrame())
        if not len(st):
            continue
        for arm in ARMS:
            d = st[st.group == arm]
            if not len(d):
                continue
            e = -np.log10(d.evalue.clip(lower=1e-300))
            print(f"  {db:<16}{arm:<16}{d.fident.median():>8.3f}"
                  f"{d.qcov.median():>8.3f}{d.tcov.median():>8.3f}"
                  f"{d.bits.median():>8.0f}{e.median():>10.1f}")


def coverage_table(best, sizes, dbs):
    """Full-length mutual coverage: best hit covering >=80% of BOTH sides.

    This is the primary homology readout, not the loose hit rate. The hit
    rate saturates on encoder confidence -- a sequence-only 3Di encoder
    emits confident, ordered 3Di for anything protein-like, so Foldseek
    aligns antisense nonsense at nearly the rate it aligns real genes. A
    local high-scoring alignment is cheap; a mutually near-full-length one
    is not.

    The last column treats the candidate arm as a mixture of real genes
    behaving like annotated CDS and non-genes behaving like matched shadows:

        f = (candidate - shadow) / (annotated - shadow)

    which is the implied real-gene fraction. The interval is a normal
    approximation propagated through that ratio; it can run below zero, and
    an interval containing zero is consistent with no real-gene content at
    all. Arms are length-matched exactly, so this is not a length artefact.
    """
    print("\n" + "=" * 78)
    print("FULL-LENGTH MUTUAL COVERAGE (qcov >= 0.8 AND tcov >= 0.8), struct")
    print("=" * 78)
    print(f"  {'db':<16}" + "".join(f"{a:>16}" for a in ARMS)
          + f"{'real-gene share (95% CI)':>28}")

    rows = {}
    for db in dbs:
        st = best.get((db, "struct"), pd.DataFrame())
        if not len(st):
            continue
        full = st[(st.qcov >= 0.8) & (st.tcov >= 0.8)]
        frac, cells = {}, []
        for arm in ARMS:
            n = sizes.get(arm, 0)
            k = int((full.group == arm).sum())
            frac[arm] = (k / n) if n else float("nan")
            cells.append(f"{frac[arm]*100:>15.1f}%" if n else f"{'--':>16}")
        note = ""
        if all(a in frac for a in ("candidate", "shadow_hi", "annotated_cds")):
            c, s, a = frac["candidate"], frac["shadow_hi"], frac["annotated_cds"]
            n = sizes["candidate"]
            denom = a - s
            if denom > 1e-9:
                f = (c - s) / denom
                # Variance of each proportion, propagated through the ratio
                # by first-order error analysis.
                vc, vs, va = (p * (1 - p) / n for p in (c, s, a))
                var = (vc + vs * (1 - f) ** 2 + va * f ** 2) / denom ** 2
                se = var ** 0.5
                note = f"{f*100:>10.1f}% ({(f-1.96*se)*100:>5.1f} - {(f+1.96*se)*100:>5.1f})"
        rows[db] = frac
        print(f"  {db:<16}" + "".join(cells) + f"{note:>28}")

    if rows:
        print("\n  Read the last column as an upper bound rather than an "
              "estimate: it assumes\n  candidates are a two-component mixture "
              "of exactly these two behaviours.")
    return rows


def entropy_strata(best, wanted, dbs):
    """Does the 3Di entropy that defined a candidate predict a hit?

    If low-3Di ORFs are the disordered/compositionally-degenerate calls we
    suspect, they should hit less. If the relationship is flat, 3Di entropy
    is not carrying the information the candidate definition assumes.
    """
    print("\n" + "=" * 78)
    print("STRUCT HIT RATE BY 3Di ENTROPY QUINTILE (union over databases)")
    print("=" * 78)
    hit = set()
    for db in dbs:
        st = best.get((db, "struct"), pd.DataFrame())
        if len(st):
            hit |= set(st["query"])
    w = wanted.copy()
    w["qid"] = (w.genome + "|" + w.input_id + "|" + w.orf_id + "|" + w.group)
    w["hit"] = w.qid.isin(hit)
    try:
        w["bin"] = pd.qcut(w.three_di_entropy, 5,
                           labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    except ValueError:
        print("  (3Di entropy not divisible into five bins, skipped)")
        return
    tab = w.pivot_table(index="bin", columns="group", values="hit",
                        aggfunc="mean", observed=True) * 100
    rng = w.groupby("bin", observed=True).three_di_entropy.agg(["min", "max"])
    print(f"  {'quintile':<10}{'3Di range':>16}"
          + "".join(f"{a:>16}" for a in ARMS))
    for b in tab.index:
        lo, hi = rng.loc[b, "min"], rng.loc[b, "max"]
        row = "".join(f"{tab.loc[b, a]:>15.1f}%" if a in tab.columns else
                      f"{'--':>16}" for a in ARMS)
        print(f"  {str(b):<10}{f'{lo:.2f}-{hi:.2f}':>16}{row}")


def truncation_strata(best, wanted, dbs):
    """Split each arm by whether the ORF is truncated by a contig end.

    get_orfs emits end = contig_length + 1 for an ORF that runs off the end
    of its contig, and every such ORF lacks a stop codon by construction
    (verified: 28,728 of 28,728 in bac_000, against 0 of 1,189,718 complete
    ORFs). 1.6% of ORFs are affected. A stop-codon-free fragment is weaker
    evidence of a real gene than a complete ORF, and fragmentation is not
    evenly distributed across genomes, so it could inflate an arm's rate
    without any biology behind it -- particularly the candidate arm, which
    the burden analysis already showed tracks assembly fragmentation.

    Absent from wanted lists built before the column existed, in which case
    this says so rather than reporting a table over one class.
    """
    print("\n" + "=" * 78)
    print("STRUCT HIT RATE BY CONTIG-END TRUNCATION (union over databases)")
    print("=" * 78)
    if "truncated" not in wanted.columns:
        print("  (no `truncated` column in the wanted list -- built before "
              "the clamp landed, skipped)")
        return
    hit = set()
    for db in dbs:
        st = best.get((db, "struct"), pd.DataFrame())
        if len(st):
            hit |= set(st["query"])
    w = wanted.copy()
    w["qid"] = (w.genome + "|" + w.input_id + "|" + w.orf_id + "|" + w.group)
    w["hit"] = w.qid.isin(hit)
    w["trunc"] = w.truncated.astype(str).str.lower().isin(("true", "1"))
    print(f"  {'class':<14}{'n':>8}" + "".join(f"{a:>16}" for a in ARMS))
    for label, sub in (("complete", w[~w.trunc]), ("truncated", w[w.trunc])):
        if len(sub) == 0:
            print(f"  {label:<14}{0:>8}" + "".join(f"{'--':>16}" for a in ARMS))
            continue
        rate = sub.groupby("group", observed=True).hit.mean() * 100
        row = "".join(f"{rate[a]:>15.1f}%" if a in rate.index else
                      f"{'--':>16}" for a in ARMS)
        print(f"  {label:<14}{len(sub):>8,}{row}")
    n_t = int(w.trunc.sum())
    print(f"\n  truncated share of the query set: {n_t:,} of {len(w):,} "
          f"({n_t/max(len(w),1)*100:.2f}%)")
    per_arm = w.groupby("group", observed=True).trunc.mean() * 100
    print("  truncated share within each arm: "
          + ", ".join(f"{a} {per_arm[a]:.1f}%"
                      for a in ARMS if a in per_arm.index))


def assembly_strata(best, wanted, dbs, domain):
    """The burden analysis found candidates track assembly fragmentation.

    So the hit rate has to be read stratified by it, or a difference between
    arms could just be assembly quality. Same-genome shadow matching should
    make candidate and shadow move together across these strata; if they do,
    that is the confounder being visibly controlled rather than assumed away.
    """
    burden = f"{GD}/missed_genes/candidate_burden_{domain}.tsv"
    if not os.path.exists(burden):
        print(f"\n  (no {burden}, assembly stratification skipped)")
        return
    md = pd.read_csv(burden, sep="\t",
                     usecols=["genome", "n50_contigs", "checkm2_completeness"])
    hit = set()
    for db in dbs:
        st = best.get((db, "struct"), pd.DataFrame())
        if len(st):
            hit |= set(st["query"])
    w = wanted.copy()
    w["qid"] = (w.genome + "|" + w.input_id + "|" + w.orf_id + "|" + w.group)
    w["hit"] = w.qid.isin(hit)
    w = w.merge(md, on="genome", how="left")
    n_nom = int(w.n50_contigs.isna().sum())

    print("\n" + "=" * 78)
    print("STRUCT HIT RATE BY ASSEMBLY QUALITY (union over databases)")
    print("=" * 78)
    if n_nom:
        print(f"  {n_nom:,} of {len(w):,} query rows have no metadata "
              "(unannotated-genome controls), excluded from this table")
    for cov, label in (("n50_contigs", "contig N50"),
                       ("checkm2_completeness", "completeness")):
        d = w[w[cov].notna()].copy()
        try:
            d["bin"] = pd.qcut(d[cov], 4, labels=["Q1", "Q2", "Q3", "Q4"],
                               duplicates="drop")
        except ValueError:
            continue
        tab = d.pivot_table(index="bin", columns="group", values="hit",
                            aggfunc="mean", observed=True) * 100
        rng = d.groupby("bin", observed=True)[cov].agg(["min", "max"])
        print(f"\n  {label}")
        print(f"    {'quartile':<10}{'range':>22}"
              + "".join(f"{a:>16}" for a in ARMS))
        for b in tab.index:
            lo, hi = rng.loc[b, "min"], rng.loc[b, "max"]
            row = "".join(f"{tab.loc[b, a]:>15.1f}%" if a in tab.columns else
                          f"{'--':>16}" for a in ARMS)
            print(f"    {str(b):<10}{f'{lo:,.0f}-{hi:,.0f}':>22}{row}")


def shadow_frames_from_context(context_dir):
    """Frame classes read from 20_orf_context.py's tables, not recomputed.

    Preferred over shadow_frames() below for two reasons.

    First, it works at all. shadow_frames() needs g_start/g_end on the
    CONTROLS tables and they are not there -- only the candidates table
    carries them -- so on the full bacterial run it skips itself and the clean
    comparator silently disappears from the report. That is exactly how the
    first full-scale analysis came out without its headline number.

    Second, it removes a duplicate implementation. shadow_frames() computed the
    overlap and the frame independently, and an earlier version of it made both
    of the coordinate mistakes 10_missed_genes.py had been corrected for -- so
    it agreed with the bug it existed to detect. One definition, computed once
    per ORF in the context stage, cannot drift from itself.
    """
    files = sorted(glob.glob(f"{context_dir}/*.context.tsv.gz"))
    if not files:
        print(f"\n  (no context tables in {context_dir}, frame check skipped)")
        return None
    per_shadow = {}
    counts = {}
    for path in files:
        d = pd.read_csv(path, sep="\t", dtype={"genome": str, "input_id": str,
                                               "orf_id": str, "group": str})
        if "cds_frame_class" not in d.columns:
            print(f"\n  ({os.path.basename(path)} predates cds_frame_class; "
                  "rerun 20_orf_context.pbs)")
            return None
        d = d[d.group == "shadow_hi"]
        cls = d.cds_frame_class.fillna("no overlapping CDS found")
        qid = (d.genome + "|" + d.input_id + "|" + d.orf_id + "|shadow_hi")
        per_shadow.update(dict(zip(qid, cls)))
        for k, v in cls.value_counts().items():
            counts[k] = counts.get(k, 0) + int(v)
    print("\n" + "=" * 78)
    print("WHAT THE SHADOW CONTROL ACTUALLY IS (from the context tables)")
    print("=" * 78)
    tot = sum(counts.values())
    for k in sorted(counts, key=lambda x: -counts[x]):
        print(f"  {k:<32}{counts[k]:>10,}  {pct(counts[k], tot)}")
    print("\n  Same strand + same frame means the shadow largely IS the "
          "annotated protein;\n  a hit there is expected and carries no "
          "information about missed genes.")
    return per_shadow


def shadow_frames(wanted, pilot_dir, cds_dir):
    """What IS a shadow, structurally speaking?

    A shadow is an unannotated six-frame ORF call overlapping an annotated
    CDS. That covers two very different things. Same strand and same reading
    frame means the shadow is largely the real protein -- a longer or shorter
    call of the same gene -- and it SHOULD hit, which makes it a poor null.
    Opposite strand, or a frameshift, means a genuinely different amino-acid
    sequence, and a hit there is the interesting kind.

    Any candidate-vs-shadow comparison is only as good as this breakdown, so
    it is computed rather than assumed.
    """
    ctrl_files = sorted(glob.glob(f"{pilot_dir}/*.controls.tsv.gz"))
    if not ctrl_files:
        print("\n  (no controls tables found, shadow frame check skipped)")
        return None

    # The DEPOSITED CDS coordinates, from 13_cds_intervals.pbs. An earlier
    # version of this function used the spans of the annotated_cds arm as a
    # stand-in, which is the same proxy 10_missed_genes.py was corrected for:
    # a matched ORF runs stop to stop and can extend past the real CDS, and a
    # CDS the matcher rejected is absent entirely.
    tags = sorted({os.path.basename(f).split(".controls")[0]
                   for f in ctrl_files})
    cds_frames = []
    for tag in tags:
        p = f"{cds_dir}/{tag}.tsv"
        if not os.path.exists(p):
            continue
        cds_frames.append(pd.read_csv(
            p, sep="\t", usecols=["genome", "contig", "start", "end", "strand"],
            dtype={"genome": str, "contig": str}))
    if not cds_frames:
        print(f"\n  (no CDS intervals under {cds_dir}, shadow frame check "
              "skipped)")
        return None
    cds = pd.concat(cds_frames, ignore_index=True)
    cds = cds.rename(columns={"contig": "input_id"})

    shad = []
    for f in ctrl_files:
        d = pd.read_csv(f, sep="\t")
        if "g_start" not in d.columns:
            print("\n  (controls tables predate g_start/g_end; the shadow "
                  "frame check needs normalised coordinates and is skipped. "
                  "Rerun 10_missed_genes.py.)")
            return None
        shad.append(d.loc[d.group == "shadow_hi",
                          ["genome", "input_id", "orf_id", "start", "end",
                           "strand", "g_start", "g_end"]])
    shad = pd.concat(shad, ignore_index=True)
    # The key must include input_id: orf ids are unique per contig, not per
    # genome, so genome+orf_id alone pulls in same-named ORFs from other
    # contigs and inflates the count above the number actually selected.
    def key(d):
        return d.genome + "@" + d.input_id + "@" + d.orf_id
    w = wanted[wanted.group == "shadow_hi"]
    shad = shad[key(shad).isin(set(key(w)))]
    if len(shad) != len(w):
        print(f"  WARNING: matched {len(shad):,} control rows to {len(w):,} "
              "selected shadows", file=sys.stderr)

    counts = {"same strand, same frame": 0, "same strand, frameshift": 0,
              "opposite strand": 0, "no overlapping CDS found": 0}
    # Keyed by the m8 query id so the coverage tables can filter on it.
    per_shadow = {}
    by_contig = {k: v for k, v in cds.groupby(["genome", "input_id"], sort=False)}
    for r in shad.itertuples(index=False):
        qid = f"{r.genome}|{r.input_id}|{r.orf_id}|shadow_hi"
        g = by_contig.get((r.genome, r.input_id))
        if g is None:
            counts["no overlapping CDS found"] += 1
            per_shadow[qid] = "no overlapping CDS found"
            continue
        # Both sides are now zero-based half-open on the forward genomic
        # axis -- g_start/g_end from normalise_orf_interval, the CDS from
        # cds_intervals.py -- so this is a same-coordinate-system test.
        # Comparing raw ORF start/end against CDS coordinates is what made
        # the earlier version of this diagnostic agree with the bug it was
        # supposed to detect.
        ov = g[(g.start < r.g_end) & (g.end > r.g_start)]
        if not len(ov):
            counts["no overlapping CDS found"] += 1
            per_shadow[qid] = "no overlapping CDS found"
            continue
        # The largest overlap is the CDS the shadow is a shadow OF.
        width = (np.minimum(ov.end.values, r.g_end)
                 - np.maximum(ov.start.values, r.g_start))
        best = ov.iloc[int(np.argmax(width))]
        if best.strand != r.strand:
            counts["opposite strand"] += 1
            per_shadow[qid] = "opposite strand"
        elif (int(r.g_start) - int(best.start)) % 3 == 0:
            counts["same strand, same frame"] += 1
            per_shadow[qid] = "same strand, same frame"
        else:
            counts["same strand, frameshift"] += 1
            per_shadow[qid] = "same strand, frameshift"

    print("\n" + "=" * 78)
    print("WHAT THE SHADOW CONTROL ACTUALLY IS")
    print("=" * 78)
    tot = sum(counts.values())
    for k, v in counts.items():
        print(f"  {k:<32}{v:>8,}  {pct(v, tot)}")
    print("\n  Same strand + same frame means the shadow largely IS the "
          "annotated protein;\n  a hit there is expected and carries no "
          "information about missed genes.")
    return per_shadow


def shadow_class_coverage(best, sizes, dbs, shadow_class):
    """Re-run the coverage comparison against antisense shadows only.

    A same-strand same-frame shadow largely IS the annotated protein, so it
    should score like a real gene and it does. Leaving those in the
    comparator inflates the shadow arm and makes the candidate excess an
    underestimate. This drops them and reports what is left, which is the
    comparison the pilot was actually trying to make: an intergenic ORF
    against a protein-like sequence that cannot itself be a protein.

    n changes between rows, so the shadow column is smaller than the arm it
    was drawn from and the mixture share is recomputed against the reduced arm.
    """
    if not shadow_class:
        return
    keep = {"opposite strand", "same strand, frameshift"}
    clean = {q for q, c in shadow_class.items() if c in keep}
    n_clean = len(clean)
    if not n_clean:
        return
    print("\n" + "=" * 78)
    print("COVERAGE vs ANTISENSE/FRAMESHIFT SHADOWS ONLY "
          f"(n = {n_clean:,} of {sizes.get('shadow_hi', 0):,})")
    print("=" * 78)
    print("  Same-frame shadows are removed: they are the annotated protein "
          "and belong\n  in the positive control, not the comparator.")
    print(f"\n  {'db':<16}{'candidate':>12}{'shadow (clean)':>16}"
          f"{'annotated':>12}{'real-gene share (95% CI)':>28}")
    for db in dbs:
        st = best.get((db, "struct"), pd.DataFrame())
        if not len(st):
            continue
        full = st[(st.qcov >= 0.8) & (st.tcov >= 0.8)]
        n_c = sizes.get("candidate", 0)
        n_a = sizes.get("annotated_cds", 0)
        if not (n_c and n_a):
            continue
        c = int((full.group == "candidate").sum()) / n_c
        a = int((full.group == "annotated_cds").sum()) / n_a
        s = len(set(full.loc[full.group == "shadow_hi", "query"]) & clean) / n_clean
        note = ""
        denom = a - s
        if denom > 1e-9:
            f = (c - s) / denom
            vc, va = (p * (1 - p) / n_c for p in (c,)), a * (1 - a) / n_a
            vc = c * (1 - c) / n_c
            vs = s * (1 - s) / n_clean
            var = (vc + vs * (1 - f) ** 2 + va * f ** 2) / denom ** 2
            se = var ** 0.5
            note = (f"{f*100:>10.1f}% ({(f-1.96*se)*100:>5.1f} - "
                    f"{(f+1.96*se)*100:>5.1f})")
        print(f"  {db:<16}{c*100:>11.1f}%{s*100:>15.1f}%{a*100:>11.1f}%"
              f"{note:>28}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--search-dir", default=f"{GD}/missed_genes/pilot/search")
    ap.add_argument("--wanted", default=f"{GD}/missed_genes/pilot/wanted4.tsv")
    ap.add_argument("--pilot-dir", default=f"{GD}/missed_genes/pilot")
    ap.add_argument("--domain", default="bac")
    ap.add_argument("--context", default=None,
                    help="directory of 20_orf_context.py tables. When given, "
                         "shadow frame classes are READ from there rather "
                         "than recomputed -- the controls tables lack "
                         "g_start/g_end, so the fallback path silently skips "
                         "the clean comparator at full scale.")
    ap.add_argument("--cds-intervals",
                    default=None,
                    help="directory of deposited CDS intervals from "
                         "13_cds_intervals.pbs; defaults to "
                         "<GD>/cds_intervals/<domain>")
    args = ap.parse_args()

    if args.cds_intervals is None:
        args.cds_intervals = f"{GD}/cds_intervals/{args.domain}"
    wanted = pd.read_csv(args.wanted, sep="\t", dtype={"chunk": str})
    sizes = wanted.group.value_counts().to_dict()
    missing = [a for a in ARMS if not sizes.get(a)]
    ARMS[:] = [a for a in ARMS if sizes.get(a)]
    if not ARMS:
        print(f"ERROR: {args.wanted} carries none of the known arms",
              file=sys.stderr)
        return 1
    print(f"pilot arms: " + ", ".join(f"{a} {sizes[a]:,}" for a in ARMS))
    if missing:
        # Named rather than skipped silently: a reader comparing this output
        # against an earlier run needs to know which arm is absent, not
        # infer it from a table that is one column narrower.
        print(f"  absent from this query set: {', '.join(missing)}")

    best, dbs = {}, []
    for path in sorted(glob.glob(f"{args.search_dir}/*.m8")):
        stem = os.path.basename(path)[:-3]
        qtag, db, mode = stem.split(".")
        df = load_m8(path)
        if len(df):
            df = pd.concat([df.reset_index(drop=True),
                            parse_ids(df["query"].reset_index(drop=True))], axis=1)
        else:
            df["group"] = []
        best[(db, "null" if qtag == "null" else mode)] = df
        if db not in dbs:
            dbs.append(db)
    if not dbs:
        sys.exit(f"ERROR: no .m8 files in {args.search_dir}")
    print(f"target databases: {', '.join(dbs)}")

    hit_rate_table(best, sizes, dbs)
    union_table(best, sizes, dbs)
    quality_table(best, dbs)
    coverage_table(best, sizes, dbs)
    entropy_strata(best, wanted, dbs)
    truncation_strata(best, wanted, dbs)
    assembly_strata(best, wanted, dbs, args.domain)
    # Prefer the context tables. The fallback recomputes from the controls
    # tables, which lack g_start/g_end at full scale and therefore skip
    # themselves -- taking the clean comparator, i.e. the headline number,
    # out of the report without failing.
    shadow_class = None
    if args.context:
        shadow_class = shadow_frames_from_context(args.context)
    if shadow_class is None:
        shadow_class = shadow_frames(wanted, args.pilot_dir,
                                     args.cds_intervals)
    shadow_class_coverage(best, sizes, dbs, shadow_class)
    return 0


if __name__ == "__main__":
    sys.exit(main())
