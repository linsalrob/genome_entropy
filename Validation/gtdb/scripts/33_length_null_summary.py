#!/usr/bin/env python3
"""Combine length_entropy_agg partials into length-conditioned empirical tables.

The random null (stage 31) answers "is this composition more skewed than an
i.i.d. draw of this length". Measured against real ORFs, the answer is
essentially always yes -- 97.8% of archaeal ORFs sit below the 0.1st centile
of the protein null -- because real proteins each have their own idiosyncratic
composition while a pooled background averages over all of them. That makes
the i.i.d. percentile useless as a significance test, though its z-score
remains the right length correction.

This builds the reference that IS calibrated: the distribution of entropy
among REAL ORFs of the same length. A percentile against it means what a
reader expects -- "this ORF is in the bottom 1% of 300 aa ORFs for 3Di
entropy" -- and needs no assumption about how sequence is generated.

Stratification matches population_agg.c exactly, so the marginals must
reproduce stage 29's population figures; that agreement is the cross-check.
"""
import argparse
import os
import sys

import numpy as np

NENT, ENTW, NLEN, NMET, NSTRAT = 4400, 1e-3, 1042, 3, 6
METRICS = ("protein", "three_di", "twelve_state")
STRATUM_NAME = {
    0: "no_deposited_cds,unmatched",
    1: "no_deposited_cds,matched",
    2: "has_deposited_cds,unmatched",
    3: "has_deposited_cds,matched",
    4: "genome_absent,unmatched",
    5: "genome_absent,matched",
}
QUANTILES = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
             0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999)


def lenbin_edges(index):
    """Inverse of lenbin() in length_entropy_agg.c. Returns (lo, hi) inclusive."""
    if index <= 909:
        return 90 + index, 90 + index
    if index <= 1009:
        lo = 1000 + (index - 910) * 10
        return lo, lo + 9
    if index <= 1039:
        lo = 2000 + (index - 1010) * 100
        return lo, lo + 99
    if index == 1040:
        return 5000, -1        # -1 = unbounded
    return 0, 89               # below the ORF floor


def quantiles_from_hist(counts, levels):
    """Quantiles from a 1e-3-wide histogram, as bin midpoints.

    This returns the ceil-rank order statistic, binned -- accurate to +/-5e-4
    bits, verified against the raw rows. It will NOT equal numpy.quantile on a
    small sample, because numpy interpolates between neighbouring order
    statistics: at n=490 the 5th centile sat between two values 0.047 apart,
    so the two definitions differed by 0.021 bits while the binning error was
    0.0005. That is the quantile definition, not a binning error, and it
    vanishes as n grows -- the published cells hold 1e3 to 1e6 rows.
    """
    total = counts.sum()
    if total == 0:
        return np.full(len(levels), np.nan)
    cumulative = np.cumsum(counts)
    targets = np.asarray(levels) * total
    idx = np.searchsorted(cumulative, targets, side="left")
    idx = np.clip(idx, 0, NENT - 1)
    return (idx + 0.5) * ENTW


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("domain")
    ap.add_argument("output_tsv")
    ap.add_argument("output_npz")
    ap.add_argument("partials", nargs="+",
                    help="worker .bin files; each must have a .txt sidecar beside it")
    ap.add_argument("--min-n", type=int, default=100,
                    help="omit a (stratum, alphabet, length) cell with fewer rows")
    ap.add_argument("--allow-malformed", type=int, default=0,
                    help="tolerate at most this many rows the aggregator could "
                         "not parse (default 0: any is fatal)")
    args = ap.parse_args()

    shape = (NSTRAT, NMET, NLEN, NENT)
    ncell = NSTRAT * NMET * NLEN * NENT
    hist = np.zeros(ncell, dtype=np.uint64)

    cell_n = np.zeros((NSTRAT, NMET, NLEN), dtype=np.uint64)
    cell_s1 = np.zeros((NSTRAT, NMET, NLEN), dtype=np.float64)
    cell_s2 = np.zeros((NSTRAT, NMET, NLEN), dtype=np.float64)
    n_rows = np.zeros(NSTRAT, dtype=np.uint64)
    n_short = np.zeros(NSTRAT, dtype=np.uint64)
    nan_ct = np.zeros((NSTRAT, NMET), dtype=np.uint64)
    oob_ct = np.zeros((NSTRAT, NMET), dtype=np.uint64)
    total_lines = malformed = 0

    metric_index = {m: i for i, m in enumerate(METRICS)}

    for path in args.partials:
        expected = ncell * 4
        actual = os.path.getsize(path)
        if actual != expected:
            sys.exit(f"{path}: {actual} bytes, expected {expected} -- refusing to "
                     f"combine a truncated partial")
        hist += np.fromfile(path, dtype=np.uint32).astype(np.uint64)

        sidecar = os.path.splitext(path)[0] + ".txt"
        with open(sidecar) as handle:
            for line in handle:
                if line.startswith("#lines="):
                    parts = line[1:].strip().split()
                    total_lines += int(parts[0].split("=")[1])
                    malformed += int(parts[1].split("=")[1])
                    continue
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if f[0] == "N":
                    n_rows[int(f[1])] += int(f[2])
                    n_short[int(f[1])] += int(f[3])
                elif f[0] == "M":
                    nan_ct[int(f[1]), metric_index[f[2]]] += int(f[3])
                    oob_ct[int(f[1]), metric_index[f[2]]] += int(f[4])
                elif f[0] == "C":
                    s, m, l = int(f[1]), metric_index[f[2]], int(f[3])
                    cell_n[s, m, l] += int(f[4])
                    cell_s1[s, m, l] += float(f[5])
                    cell_s2[s, m, l] += float(f[6])

    hist = hist.reshape(shape)

    # The histogram and the exact counters are accumulated independently, so
    # they agree only if both are right. Cheap, and it has caught a real bug.
    if not np.array_equal(hist.sum(axis=3), cell_n):
        sys.exit("histogram totals disagree with the exact per-cell counts -- "
                 "partials are inconsistent, refusing to publish")

    print(f"{args.domain}: {total_lines:,} rows, {malformed:,} malformed")
    # A malformed row is silently absent from BOTH the histogram and the exact
    # counters, so the consistency check below cannot see it. Schema drift or a
    # damaged record would therefore produce a quietly incomplete reference.
    # The partials are kept, so re-summarising after investigating costs
    # nothing -- there is no reason to let this through by default.
    if malformed > args.allow_malformed:
        sys.exit(f"{malformed:,} malformed rows exceed --allow-malformed "
                 f"{args.allow_malformed}: the reference would silently omit "
                 f"them. Investigate the chunk TSVs, then re-run this "
                 f"summariser over the kept partials.")
    for s in range(NSTRAT):
        if n_rows[s]:
            print(f"  stratum {s} {STRATUM_NAME[s]:<30} {int(n_rows[s]):>15,}"
                  + (f"   ({int(n_short[s]):,} below the 90 aa floor)" if n_short[s] else ""))
    if nan_ct.any() or oob_ct.any():
        print(f"  missing values {int(nan_ct.sum()):,}, out of histogram range "
              f"{int(oob_ct.sum()):,}")

    rows = []
    for s in range(NSTRAT):
        for m, metric in enumerate(METRICS):
            for l in range(NLEN):
                n = int(cell_n[s, m, l])
                if n < args.min_n:
                    continue
                lo, hi = lenbin_edges(l)
                mean = cell_s1[s, m, l] / n
                var = max(cell_s2[s, m, l] / n - mean * mean, 0.0)
                q = quantiles_from_hist(hist[s, m, l].astype(np.float64), QUANTILES)
                rows.append((s, metric, l, lo, hi, n, mean, var ** 0.5, q))

    with open(args.output_tsv, "w") as fh:
        fh.write("domain\tstratum\tstratum_name\talphabet\tlength_bin\tlength_lo\t"
                 "length_hi\tn\tmean\tsd\t"
                 + "\t".join(f"q{q:g}".replace("0.", "p") for q in QUANTILES) + "\n")
        for s, metric, l, lo, hi, n, mean, sd, q in rows:
            fh.write(f"{args.domain}\t{s}\t{STRATUM_NAME[s]}\t{metric}\t{l}\t{lo}\t{hi}\t"
                     f"{n}\t{mean:.8f}\t{sd:.8f}\t"
                     + "\t".join(f"{v:.6f}" for v in q) + "\n")

    npz = {"quantile_levels": np.array(QUANTILES)}
    for s in range(NSTRAT):
        for metric in METRICS:
            subset = [r for r in rows if r[0] == s and r[1] == metric]
            if not subset:
                continue
            key = f"{args.domain}|{s}|{metric}"
            # Midpoints for display and for anything that wants a curve, but
            # the true edges too: these are BINS, not samples of a continuous
            # function, and above 999 aa they are 10 or 100 aa wide. A consumer
            # must be able to select the bin that contains a length rather than
            # interpolate between midpoints.
            npz[f"{key}|lengths"] = np.array([(r[3] + r[4]) / 2 if r[4] > 0 else r[3]
                                              for r in subset], dtype=float)
            npz[f"{key}|length_lo"] = np.array([r[3] for r in subset], dtype=float)
            npz[f"{key}|length_hi"] = np.array([r[4] if r[4] > 0 else np.inf
                                                for r in subset], dtype=float)
            npz[f"{key}|quantiles"] = np.vstack([r[8] for r in subset])
            npz[f"{key}|mean"] = np.array([r[6] for r in subset])
            npz[f"{key}|sd"] = np.array([r[7] for r in subset])
            npz[f"{key}|n"] = np.array([r[5] for r in subset])
    np.savez_compressed(args.output_npz, **npz)

    print(f"  wrote {args.output_tsv} ({len(rows):,} cells) and {args.output_npz}")


if __name__ == "__main__":
    main()
