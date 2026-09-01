#!/usr/bin/env python3
"""Read out the Foldseek pilot: does structure find genes that sequence misses?

The pilot (issue #92, step 5) searched four arms of 8,651 ORFs each --
candidate, shadow_hi, annotated_cds, intergenic_lo -- against four target
databases twice over, once with 3Di+amino acid (`foldseek search
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
                           null (technical) bracket what a non-gene scores

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
ARMS = ["candidate", "shadow_hi", "annotated_cds", "intergenic_lo"]
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
    print("HIT RATES -- queries with at least one hit at E < 1e-3, "
          "n = 8,651 per arm")
    print("=" * 78)
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


def shadow_frames(wanted, pilot_dir):
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
        return
    cds = []
    for f in ctrl_files:
        d = pd.read_csv(f, sep="\t", usecols=["genome", "input_id", "start",
                                              "end", "strand", "group"])
        cds.append(d[d.group == "annotated_cds"].drop(columns="group"))
    cds = pd.concat(cds, ignore_index=True)

    shad = []
    for f in ctrl_files:
        d = pd.read_csv(f, sep="\t", usecols=["genome", "input_id", "orf_id",
                                              "start", "end", "strand", "group"])
        shad.append(d[d.group == "shadow_hi"])
    shad = pd.concat(shad, ignore_index=True)
    # The key must include input_id: orf ids are unique per contig, not per
    # genome, so genome+orf_id alone pulls in same-named ORFs from other
    # contigs and inflates the count above the 8,651 actually selected.
    def key(d):
        return d.genome + "@" + d.input_id + "@" + d.orf_id
    w = wanted[wanted.group == "shadow_hi"]
    shad = shad[key(shad).isin(set(key(w)))]
    if len(shad) != len(w):
        print(f"  WARNING: matched {len(shad):,} control rows to {len(w):,} "
              "selected shadows", file=sys.stderr)

    counts = {"same strand, same frame": 0, "same strand, frameshift": 0,
              "opposite strand": 0, "no overlapping CDS found": 0}
    by_contig = {k: v for k, v in cds.groupby(["genome", "input_id"], sort=False)}
    for r in shad.itertuples(index=False):
        g = by_contig.get((r.genome, r.input_id))
        if g is None:
            counts["no overlapping CDS found"] += 1
            continue
        ov = g[(g.start <= r.end) & (g.end >= r.start)]
        if not len(ov):
            counts["no overlapping CDS found"] += 1
            continue
        # The largest overlap is the CDS the shadow is a shadow OF.
        width = np.minimum(ov.end.values, r.end) - np.maximum(ov.start.values, r.start)
        best = ov.iloc[int(np.argmax(width))]
        if best.strand != r.strand:
            counts["opposite strand"] += 1
        elif (int(r.start) - int(best.start)) % 3 == 0:
            counts["same strand, same frame"] += 1
        else:
            counts["same strand, frameshift"] += 1

    print("\n" + "=" * 78)
    print("WHAT THE SHADOW CONTROL ACTUALLY IS")
    print("=" * 78)
    tot = sum(counts.values())
    for k, v in counts.items():
        print(f"  {k:<32}{v:>8,}  {pct(v, tot)}")
    print("\n  Same strand + same frame means the shadow largely IS the "
          "annotated protein;\n  a hit there is expected and carries no "
          "information about missed genes.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--search-dir", default=f"{GD}/missed_genes/pilot/search")
    ap.add_argument("--wanted", default=f"{GD}/missed_genes/pilot/wanted4.tsv")
    ap.add_argument("--pilot-dir", default=f"{GD}/missed_genes/pilot")
    ap.add_argument("--domain", default="bac")
    args = ap.parse_args()

    wanted = pd.read_csv(args.wanted, sep="\t", dtype={"chunk": str})
    sizes = wanted.group.value_counts().to_dict()
    print(f"pilot arms: " + ", ".join(f"{a} {sizes.get(a, 0):,}" for a in ARMS))

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
    entropy_strata(best, wanted, dbs)
    assembly_strata(best, wanted, dbs, args.domain)
    shadow_frames(wanted, args.pilot_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
