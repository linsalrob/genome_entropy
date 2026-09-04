#!/usr/bin/env python3
"""Combine agg partials into the population summary requested in issue #99.

Reads one or more partial files written by agg (counts, sums, sums of squares
and 1e-4 histograms, stratified by (genome has >=1 deposited CDS, in_genbank))
and emits a TSV of n, mean, sd, median and quartiles for each requested group.

Quantiles are read from the histograms, so they are bounded by the bin width
(1e-4) rather than exact; every mean and count is exact.
"""
import sys, math, collections

NBIN, BINW = 44000, 1e-4
METS = ["three_di", "protein", "twelve_state", "mi", "dna"]

def read(paths):
    n = collections.Counter()
    aan = collections.Counter(); aas = collections.Counter()
    s1 = collections.Counter(); s2 = collections.Counter()
    nan = collections.Counter(); oob = collections.Counter()
    mx = {}
    hist = collections.defaultdict(lambda: [0] * NBIN)
    nline = malformed = 0
    for p in paths:
        with open(p) as fh:
            for ln in fh:
                if ln[0] == '#':
                    if ln.startswith('#lines='):
                        a, b = ln[1:].split()
                        nline += int(a.split('=')[1]); malformed += int(b.split('=')[1])
                    continue
                f = ln.rstrip('\n').split('\t')
                if f[0] == 'N':
                    st = int(f[1]); n[st] += int(f[2]); aan[st] += int(f[3]); aas[st] += float(f[4])
                elif f[0] == 'S':
                    st, m = int(f[1]), f[2]
                    s1[(st, m)] += float(f[3]); s2[(st, m)] += float(f[4])
                    nan[(st, m)] += int(f[5]); oob[(st, m)] += int(f[6])
                    hi = float(f[8])
                    if hi == hi:
                        mx[(st, m)] = max(mx.get((st, m), -1e300), hi)
                elif f[0] == 'H':
                    hist[(int(f[1]), f[2])][int(f[3])] += int(f[4])
    return dict(n=n, aan=aan, aas=aas, s1=s1, s2=s2, nan=nan, oob=oob,
                mx=mx, hist=hist, nline=nline, malformed=malformed)

def band_fractions(h, cuts):
    """Share of a group's values below each cut, read from the histogram.

    The cuts are the log2(k) ceilings of section 5 and the section 6 candidate
    threshold, so these are the shares the interpretation actually turns on.
    """
    tot = sum(h)
    if not tot:
        return [float('nan')] * len(cuts)
    out = []
    for c in cuts:
        b = int(round(c / BINW))
        out.append(sum(h[:b]) / tot)
    return out


def quantiles(h, qs):
    tot = sum(h)
    if not tot:
        return [float('nan')] * len(qs)
    out, cum, b = [], 0, 0
    for q in qs:
        target = q * tot
        while b < NBIN and cum + h[b] < target:
            cum += h[b]; b += 1
        out.append((b + 0.5) * BINW)
    return out

GROUPS = [
    ("all ORFs",                       (0, 1, 2, 3)),
    ("in_genbank=True",                (1, 3)),
    ("in_genbank=False",               (0, 2)),
    ("CDS-bearing genomes: all",       (2, 3)),
    ("CDS-bearing genomes: matched",   (3,)),
    ("CDS-bearing genomes: unmatched", (2,)),
    ("no-CDS genomes: all",            (0, 1)),
]

def main():
    domain, out = sys.argv[1], sys.argv[2]
    d = read(sys.argv[3:])
    unk = d['n'][4] + d['n'][5]
    print(f"[{domain}] rows read = {d['nline']:,}  malformed = {d['malformed']:,}  "
          f"rows whose genome is absent from the CDS-count table = {unk:,}", file=sys.stderr)
    print(f"[{domain}] stratum n: no-CDS/unmatched={d['n'][0]:,} no-CDS/matched={d['n'][1]:,} "
          f"CDS/unmatched={d['n'][2]:,} CDS/matched={d['n'][3]:,}", file=sys.stderr)

    CUTS = [1.0, 1.5849625007211562, 2.0, 2.5, 3.0, 3.5]
    fr = open(out.replace('.tsv', '_bands.tsv'), 'w')
    fr.write("domain\tgroup\tmetric\tn_valid\t"
             + "\t".join(f"frac_lt_{c:.4f}" for c in CUTS) + "\n")

    with open(out, 'w') as fh:
        fh.write("domain\tgroup\tmetric\tn\tn_valid\tmean\tsd\tmin\tq1\tmedian\tq3\tmax\tiqr\n")
        for gname, sts in GROUPS:
            gn = sum(d['n'][s] for s in sts)
            mean_aa = (sum(d['aas'][s] for s in sts) / sum(d['aan'][s] for s in sts)
                       if sum(d['aan'][s] for s in sts) else float('nan'))
            fh.write(f"{domain}\t{gname}\taa_length\t{gn}\t{sum(d['aan'][s] for s in sts)}"
                     f"\t{mean_aa:.4f}\t\t\t\t\t\t\n")
            for m in METS:
                h = [0] * NBIN
                any_h = False
                for s in sts:
                    if (s, m) in d['hist']:
                        any_h = True
                        for i, v in enumerate(d['hist'][(s, m)]):
                            if v:
                                h[i] += v
                nv = sum(h)
                if not nv:
                    continue
                S1 = sum(d['s1'][(s, m)] for s in sts)
                S2 = sum(d['s2'][(s, m)] for s in sts)
                mean = S1 / nv
                var = max(S2 / nv - mean * mean, 0.0)
                lo, q1, med, q3, hi = quantiles(h, [0.0, 0.25, 0.5, 0.75, 1.0])
                mxv = max((d['mx'][(s, m)] for s in sts if (s, m) in d['mx']), default=float('nan'))
                first = next(i for i, v in enumerate(h) if v)
                fh.write(f"{domain}\t{gname}\t{m}\t{gn}\t{nv}\t{mean:.6f}\t{math.sqrt(var):.6f}\t"
                         f"{(first + 0.5) * BINW:.4f}\t{q1:.4f}\t{med:.4f}\t{q3:.4f}\t{mxv:.4f}\t"
                         f"{q3 - q1:.4f}\n")
                fr.write(f"{domain}\t{gname}\t{m}\t{nv}\t"
                         + "\t".join(f"{v:.6f}" for v in band_fractions(h, CUTS)) + "\n")
    fr.close()
    print(f"[{domain}] wrote {out}", file=sys.stderr)

main()
