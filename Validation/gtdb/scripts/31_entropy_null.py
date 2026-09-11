#!/usr/bin/env python3
"""Length-conditioned null distribution of Shannon entropy.

WHY THIS EXISTS

Plug-in Shannon entropy is a biased estimator, and the bias depends on how
many residues it was computed from. A 90 aa ORF and a 900 aa ORF drawn from
exactly the same residue distribution do not have the same expected entropy:
the short one is lower, by roughly (k-1)/(2 L ln2) bits, purely because it
cannot sample every symbol. Comparing raw entropies across lengths therefore
measures length as much as it measures composition. This is the same effect
already recorded in the report as "length matching lowers annotated-CDS median
3Di from 3.29 to 3.13".

This builds the reference that removes it: for each alphabet and each length,
the distribution of entropy for a random sequence of that length drawn from
the population's own residue composition. An observed entropy can then be
reported as a percentile, a z-score, or a one-sided p against sequences of its
own length instead of as a raw number.

WHAT THE NULL IS, AND WHAT IT IS NOT

Residues are drawn i.i.d. from the pooled background, so the count vector is
Multinomial(L, p) and the null asks: **is this sequence's composition more
skewed than a random draw of that length from the population pool?**

It is not a test of sequence order. Shannon entropy of a residue distribution
is permutation-invariant, so shuffling a sequence leaves its entropy exactly
unchanged -- a shuffle null here has zero variance and is degenerate. Anything
about motifs, repeats or periodicity is invisible to this statistic and needs
a different one.

WHY MONTE CARLO RATHER THAN THE TEXTBOOK FORMULA

The first-order asymptotic variance of plug-in entropy is
(1/L)[sum p log2(p)^2 - H^2]. That bracket vanishes for an exactly uniform p,
where the true variance is O(1/L^2) instead, so the formula degenerates to a
zero-width interval. Amino acid composition is not uniform, so it does not
degenerate here -- but it is the closest of the three to uniform, and it is
where the approximation is worst. Measured against 400,000 draws:

  alphabet        L=90    L=150   L=300   L=1000     (MC sd / asymptotic sd)
  protein         1.26     1.18    1.09     1.02
  three_di        1.02     1.02    1.01     1.00
  twelve_state    1.05     1.03    1.01     1.00

So an analytic interval for protein entropy is **26% too narrow at 90 aa** and
still 9% too narrow at 300 aa -- across the length range where most ORFs sit
(the 90 aa floor to ~300 aa). That is a real error in the alphabet a reader is
most likely to reach for, not a rounding difference. Monte Carlo has no such
failure mode, costs 2.16 SU for the whole table, and needs no assumption about
which regime p is in. Miller-Madow means and asymptotic sds are emitted
alongside so the discrepancy stays visible rather than hidden; note the
Miller-Madow mean is also a first-order correction and is ~0.013 bits high for
protein at L=90, about 15% of a standard deviation.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

LOG2 = np.log(2.0)
QUANTILES = (0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
             0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995)
BATCH = 100_000


def entropy_from_counts(counts, length):
    """Plug-in Shannon entropy in bits, matching shannon_entropy()."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(counts > 0, counts * np.log(counts), 0.0)
    return np.log2(length) - terms.sum(axis=1) / (length * LOG2)


def analytic(probs):
    """Miller-Madow mean and first-order sd coefficient, both in bits."""
    probs = probs[probs > 0]
    entropy = -(probs * np.log2(probs)).sum()
    support = probs.size
    variance_coefficient = (probs * np.log2(probs) ** 2).sum() - entropy ** 2
    return entropy, support, max(variance_coefficient, 0.0)


def simulate(task):
    domain, alphabet, length, probs, n_reps, seed = task
    probs = np.asarray(probs, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_reps, dtype=np.float64)
    filled = 0
    while filled < n_reps:
        size = min(BATCH, n_reps - filled)
        counts = rng.multinomial(length, probs, size=size)
        draws[filled:filled + size] = entropy_from_counts(counts, length)
        filled += size
    draws.sort()
    entropy, support, variance_coefficient = analytic(probs)
    return {
        "domain": domain,
        "alphabet": alphabet,
        "aa_length": length,
        "n_reps": n_reps,
        "mean": draws.mean(),
        "sd": draws.std(ddof=1),
        "skew": float(((draws - draws.mean()) ** 3).mean() / draws.std() ** 3),
        "h_background": entropy,
        "k_support": support,
        "mean_miller_madow": entropy - (support - 1) / (2 * length * LOG2),
        "sd_asymptotic": np.sqrt(variance_coefficient / length),
        "quantiles": np.quantile(draws, QUANTILES),
    }


def load_composition(path, min_frequency):
    """domain -> alphabet -> (symbols, probabilities)."""
    table = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            domain = fields[index["domain"]]
            alphabet = fields[index["alphabet"]]
            table.setdefault(domain, {}).setdefault(alphabet, {})[
                fields[index["symbol"]]] = float(fields[index["frequency"]])
    out = {}
    for domain, alphabets in table.items():
        for alphabet, frequencies in alphabets.items():
            symbols = sorted(frequencies)
            probs = np.array([frequencies[s] for s in symbols], dtype=float)
            keep = probs >= min_frequency
            if not keep.all():
                dropped = [s for s, k in zip(symbols, keep) if not k]
                print(f"  {domain}/{alphabet}: dropping {dropped} below "
                      f"{min_frequency}", file=sys.stderr)
            symbols = [s for s, k in zip(symbols, keep) if k]
            probs = probs[keep]
            out.setdefault(domain, {})[alphabet] = (symbols, probs / probs.sum())
    return out


def length_grid(spec):
    """"90:600:1,600:1000:5" -> sorted unique lengths."""
    values = []
    for part in spec.split(","):
        start, stop, step = (int(x) for x in part.split(":"))
        values.extend(range(start, stop + 1, step))
    return sorted(set(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--composition", required=True,
                    help="composition_pooled.tsv from 30_residue_composition.py")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--grid", default="90:600:1,600:1000:5,1000:2000:20,2000:5000:100")
    ap.add_argument("--reps", type=int, default=400_000)
    ap.add_argument("--min-frequency", type=float, default=0.0,
                    help="drop background symbols rarer than this")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NPROC", 8)))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    composition = load_composition(args.composition, args.min_frequency)
    lengths = length_grid(args.grid)

    tasks = []
    counter = 0
    for domain in sorted(composition):
        for alphabet in sorted(composition[domain]):
            symbols, probs = composition[domain][alphabet]
            print(f"{domain}/{alphabet}: {len(symbols)} symbols "
                  f"({''.join(symbols)}), H(p) = {analytic(probs)[0]:.4f} bits")
            for length in lengths:
                counter += 1
                tasks.append((domain, alphabet, length, probs, args.reps,
                              args.seed + counter))

    print(f"\n{len(lengths)} lengths x {len(tasks) // len(lengths)} "
          f"(domain, alphabet) = {len(tasks)} tasks, {args.reps:,} reps each, "
          f"{args.workers} workers", flush=True)

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(simulate, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)

    results.sort(key=lambda r: (r["domain"], r["alphabet"], r["aa_length"]))

    out_tsv = os.path.join(args.output_dir, "entropy_null_quantiles.tsv")
    with open(out_tsv, "w") as fh:
        fh.write("domain\talphabet\taa_length\tn_reps\th_background\tk_support\t"
                 "mean\tsd\tskew\tmean_miller_madow\tsd_asymptotic\t"
                 + "\t".join(f"q{q:g}".replace("0.", "p") for q in QUANTILES) + "\n")
        for row in results:
            fh.write(
                f"{row['domain']}\t{row['alphabet']}\t{row['aa_length']}\t"
                f"{row['n_reps']}\t{row['h_background']:.8f}\t{row['k_support']}\t"
                f"{row['mean']:.8f}\t{row['sd']:.8f}\t{row['skew']:.6f}\t"
                f"{row['mean_miller_madow']:.8f}\t{row['sd_asymptotic']:.8f}\t"
                + "\t".join(f"{v:.8f}" for v in row["quantiles"]) + "\n")

    # Compact form for the scorer: one (length x quantile) grid per key.
    npz = {}
    for domain in sorted(composition):
        for alphabet in sorted(composition[domain]):
            subset = [r for r in results
                      if r["domain"] == domain and r["alphabet"] == alphabet]
            key = f"{domain}|{alphabet}"
            npz[f"{key}|lengths"] = np.array([r["aa_length"] for r in subset])
            npz[f"{key}|quantiles"] = np.vstack([r["quantiles"] for r in subset])
            npz[f"{key}|mean"] = np.array([r["mean"] for r in subset])
            npz[f"{key}|sd"] = np.array([r["sd"] for r in subset])
    npz["quantile_levels"] = np.array(QUANTILES)
    np.savez_compressed(os.path.join(args.output_dir, "entropy_null.npz"), **npz)

    print(f"\nwrote {out_tsv} ({len(results)} rows)")
    print("\nsanity -- Monte Carlo against the analytic approximations:")
    print(f"{'domain':>4} {'alphabet':>13} {'L':>5} {'MC mean':>9} {'MM':>9} "
          f"{'MC sd':>8} {'asympt sd':>9}")
    for row in results:
        if row["aa_length"] in (90, 150, 300, 1000):
            print(f"{row['domain']:>4} {row['alphabet']:>13} {row['aa_length']:>5} "
                  f"{row['mean']:>9.4f} {row['mean_miller_madow']:>9.4f} "
                  f"{row['sd']:>8.4f} {row['sd_asymptotic']:>9.4f}")


if __name__ == "__main__":
    main()
