#!/usr/bin/env python3
"""Choose the pilot query set for the Foldseek search (issue #92, step 2).

Four biological arms plus a technical null, per the design agreed on
issue #92:

  candidate      intergenic, unmatched, 3Di >= 2.5 in an annotated genome
                 -- the hypothesis group
  shadow_hi      unmatched, 3Di >= 2.5, overlapping an annotated CDS
                 -- the PRINCIPAL confounding control
  annotated_cds  in_genbank=True -- positive control
  intergenic_lo  intergenic, 3Di < 2.5 -- negative structural-complexity
                 control
  (null)         not selected here: it is the candidates' own 3Di strings
                 shuffled, generated at database-build time so that the
                 amino acids and lengths are identical by construction

The shadow arm is the point of the whole exercise. At full scale, candidates
and high-3Di shadows are indistinguishable on every axis available without a
search -- median 175 vs 180 aa, 89.5% vs 89.1% over 100 aa, 3Di 2.94 vs
2.88, protein entropy 4.00 vs 3.98. The experiment is therefore not "do
candidates get hits" but "do candidates get hits more often than shadows
that look exactly like them".

Matching differs by arm, deliberately:

  shadow_hi      matched on length AND 3Di entropy, preferring a shadow from
                 the SAME GENOME as the candidate. Same genome means same
                 lineage, GC, annotation pipeline and assembly quality, so
                 it removes those as explanations without needing to model
                 them. Matching on 3Di as well makes the test "at equal
                 apparent structure, does an intergenic ORF have more
                 structural homology than a shadow of a real gene".
  annotated_cds  length only. Matching real CDS on 3Di entropy would defeat
                 the purpose: their high 3Di is the signal, not a nuisance.
  intergenic_lo  length only, for the same reason in reverse.

Match quality is measured and reported rather than assumed -- how many
matched within genome, and the achieved differences in each dimension.

  14_pilot_queryset.py --group-dir <dir with per-chunk candidates/controls>
                       --chunks bac_000,bac_051 --out wanted.tsv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LENGTH_ONLY_ARMS = ("annotated_cds", "intergenic_lo")
OUT_COLS = ["domain", "chunk", "genome", "input_id", "orf_id", "group",
            "aa_length", "three_di_entropy", "protein_entropy"]


def load(group_dir, chunks, pattern):
    frames = []
    for tag in chunks:
        path = Path(group_dir) / f"{tag}.{pattern}.tsv.gz"
        if not path.exists():
            raise SystemExit(f"ERROR: missing {path}")
        frames.append(pd.read_csv(path, sep="\t", dtype={"chunk": "str"}))
    return pd.concat(frames, ignore_index=True)


def match_2d(cand, pool, prefer_same_genome=True):
    """Match each candidate to a pool row on length and 3Di, no replacement.

    Distance is standardised by the candidates' own spread in each
    dimension, so neither axis dominates just because it has larger units.
    Same-genome partners are taken first and the count is reported: a
    within-genome match controls lineage, GC, annotation pipeline and
    assembly quality all at once, which is worth more than a marginally
    closer match from an unrelated organism.
    """
    pool = pool.reset_index(drop=True)
    p_len = pool.aa_length.to_numpy(dtype=float)
    p_ent = pool.three_di_entropy.to_numpy(dtype=float)
    used = np.zeros(len(pool), dtype=bool)

    sd_len = max(float(cand.aa_length.std()), 1.0)
    sd_ent = max(float(cand.three_di_entropy.std()), 1e-3)

    by_genome = {g: np.asarray(idx, dtype=int)
                 for g, idx in pool.groupby("genome").indices.items()}

    picks, d_len, d_ent, same_genome = [], [], [], 0
    for genome, want_len, want_ent in zip(cand.genome.to_numpy(),
                                          cand.aa_length.to_numpy(dtype=float),
                                          cand.three_di_entropy.to_numpy(dtype=float)):
        chosen = -1
        if prefer_same_genome and genome in by_genome:
            idx = by_genome[genome]
            idx = idx[~used[idx]]
            if len(idx):
                d = (np.abs(p_len[idx] - want_len) / sd_len
                     + np.abs(p_ent[idx] - want_ent) / sd_ent)
                chosen = int(idx[int(np.argmin(d))])
                same_genome += 1
        if chosen < 0:
            free = np.flatnonzero(~used)
            if not len(free):
                break
            d = (np.abs(p_len[free] - want_len) / sd_len
                 + np.abs(p_ent[free] - want_ent) / sd_ent)
            chosen = int(free[int(np.argmin(d))])
        used[chosen] = True
        picks.append(chosen)
        d_len.append(abs(p_len[chosen] - want_len))
        d_ent.append(abs(p_ent[chosen] - want_ent))
    return (pool.iloc[picks].copy(), np.array(d_len), np.array(d_ent),
            same_genome)


def match_on_length(target_lengths, pool, quiet=False):
    """Pick one pool row per target length, nearest length, no replacement.

    Walks outward from the insertion point until an unused row is found, so
    a pool that runs out of a given length degrades to the nearest spare
    rather than failing. Returns the chosen rows plus the achieved
    difference, which the caller reports -- a match this loose would matter
    to interpretation, so it does not stay hidden.
    """
    pool = pool.sort_values("aa_length", kind="stable").reset_index(drop=True)
    lengths = pool.aa_length.to_numpy()
    used = np.zeros(len(pool), dtype=bool)
    picks, deltas = [], []
    for want in target_lengths:
        i = int(np.searchsorted(lengths, want))
        lo, hi = i - 1, i
        best = -1
        while lo >= 0 or hi < len(pool):
            cand_lo = abs(lengths[lo] - want) if lo >= 0 else None
            cand_hi = abs(lengths[hi] - want) if hi < len(pool) else None
            if cand_lo is not None and (cand_hi is None or cand_lo <= cand_hi):
                if not used[lo]:
                    best = lo
                    break
                lo -= 1
            else:
                if not used[hi]:
                    best = hi
                    break
                hi += 1
        if best < 0:
            break                      # pool exhausted
        used[best] = True
        picks.append(best)
        deltas.append(abs(int(lengths[best]) - int(want)))
    chosen = pool.iloc[picks].copy()
    return chosen, np.array(deltas)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group-dir", required=True,
                    help="directory of per-chunk candidates/controls tables "
                         "written by 10_missed_genes.py; the controls must be "
                         "unsampled (--controls-per-chunk large) or the "
                         "matched arms will be drawn from a sample")
    ap.add_argument("--chunks", required=True,
                    help="comma-separated chunk tags, e.g. bac_000,bac_051")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="0 = all candidates in those chunks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunks = [c.strip() for c in args.chunks.split(",") if c.strip()]
    cand = load(args.group_dir, chunks, "candidates")
    ctrl = load(args.group_dir, chunks, "controls")

    if args.max_candidates and len(cand) > args.max_candidates:
        cand = cand.sample(args.max_candidates, random_state=args.seed)
    cand = cand.sort_values("aa_length", kind="stable").reset_index(drop=True)
    print(f"candidates          : {len(cand):,} from {', '.join(chunks)}")

    parts = [cand.assign(group="candidate")]

    # The confounding control: length AND 3Di, same genome where possible.
    shadows = ctrl[ctrl.group == "shadow_hi"]
    if len(shadows) == 0:
        print("ERROR: no shadow_hi rows -- the principal control is missing",
              file=sys.stderr)
        return 1
    chosen, d_len, d_ent, same_genome = match_2d(cand, shadows)
    parts.append(chosen.assign(group="shadow_hi"))
    print(f"{'shadow_hi':<20}: {len(chosen):,} matched from a pool of "
          f"{len(shadows):,}")
    print(f"{'':<20}  same genome as its candidate: {same_genome:,} "
          f"({same_genome/max(len(chosen),1)*100:.1f}%)")
    print(f"{'':<20}  |dlen| median {np.median(d_len):.0f} aa, "
          f"|d3Di| median {np.median(d_ent):.3f} bits")

    for arm in LENGTH_ONLY_ARMS:
        pool = ctrl[ctrl.group == arm]
        if len(pool) == 0:
            print(f"WARNING: no {arm} rows in the control tables", file=sys.stderr)
            continue
        chosen, deltas = match_on_length(cand.aa_length.to_numpy(), pool)
        parts.append(chosen.assign(group=arm))
        within = (deltas <= np.maximum(1, 0.1 * cand.aa_length.to_numpy()[:len(deltas)])).mean()
        print(f"{arm:<20}: {len(chosen):,} matched from a pool of {len(pool):,}"
              f"  |dlen| median {np.median(deltas):.0f}"
              f"  within 10%: {within*100:.1f}%")

    out = pd.concat(parts, ignore_index=True)[OUT_COLS]
    dup = out.duplicated(subset=["genome", "input_id", "orf_id"]).sum()
    if dup:
        # The same ORF cannot serve in two arms; that would put one sequence
        # under one id in the query database and silently drop an arm member.
        print(f"ERROR: {dup} ORFs selected into more than one arm", file=sys.stderr)
        return 1
    out.to_csv(args.out, sep="\t", index=False)

    print(f"\n{'arm':<20}{'n':>10}{'med aa':>8}{'med 3Di':>9}")
    for arm, g in out.groupby("group", sort=False):
        print(f"{arm:<20}{len(g):>10,}{g.aa_length.median():>8.0f}"
              f"{g.three_di_entropy.median():>9.2f}")
    print(f"\nwanted list -> {args.out}  ({len(out):,} ORFs, "
          f"{out.genome.nunique():,} genomes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
