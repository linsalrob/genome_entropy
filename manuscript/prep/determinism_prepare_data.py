#!/usr/bin/env python3
"""Parse the Inphared encoding-determinism report into a supplementary table.

Source: linsalrob/agent-handoffs, projects/genome_entropy_logan/inphared_gadi/
tables/determinism_50genomes.txt at commit 4b81cfb, written by
scripts/50_determinism_compare.py after PBS job 181181761 (scripts/50_determinism.pbs).
The test re-encoded the first 50 genomes of Inphared chunk phg_000 (6,284 ORFs)
on one V100 (genome_entropy 0.3.0, gbouras13/modernprost-50M) at encoding sizes
10,000 and 20,000, and compared both with the Gadi production run (10,000) and
with the Setonix run (genome_entropy 0.2.0, ROCm fp16, 20,000).

  determinism_prepare_data.py <determinism_50genomes.txt>
"""
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
GZ = {"method": "gzip", "mtime": 0}

WHAT = {
    "det es10k vs production": ("repeat run (nothing changed)", "Gadi V100, 0.3.0, 10,000", "Gadi V100, 0.3.0, 10,000"),
    "det es20k vs production": ("encoding size", "Gadi V100, 0.3.0, 20,000", "Gadi V100, 0.3.0, 10,000"),
    "det es10k vs det es20k": ("encoding size", "Gadi V100, 0.3.0, 10,000", "Gadi V100, 0.3.0, 20,000"),
    "det es10k vs Setonix": ("hardware + precision + version + encoding size",
                             "Gadi V100, 0.3.0, 10,000", "Setonix ROCm fp16, 0.2.0, 20,000"),
    "det es20k vs Setonix": ("hardware + precision + version",
                             "Gadi V100, 0.3.0, 20,000", "Setonix ROCm fp16, 0.2.0, 20,000"),
}
GADI = re.compile(r"^(det es\S+ vs (?:production|det es\S+))\s+(\S+)\s+identical\s+([\d.]+)%"
                  r"\s+n_diff\s+(\d+)\s+p99\s+([\d.]+)\s+max\s+([\d.]+)")
SET = re.compile(r"^(det es\S+ vs Setonix)\s+(\S+)\s+identical\s+([\d.]+)%\s+<0\.01\s+([\d.]+)%"
                 r"\s+median\s+([\d.]+)\s+p99\s+([\d.]+)\s+max\s+([\d.]+)\s+r\s+([\d.]+)")


def main():
    src = Path(sys.argv[1])
    print(f"  in   {src}  sha256={hashlib.sha256(src.read_bytes()).hexdigest()}")
    rows, n = [], None
    for line in src.read_text().splitlines():
        if line.startswith("n: "):
            n = int(line.split()[1])
        if m := GADI.match(line):
            k, feat, ident, ndiff, p99, mx = m.groups()
            rows.append(dict(comparison=k, metric=feat, n_orfs=n, pct_identical=float(ident),
                             n_differ=int(ndiff), pct_abs_diff_lt_0_01=None, median_abs_diff=None,
                             p99_abs_diff=float(p99), max_abs_diff=float(mx), pearson_r=None))
        elif m := SET.match(line):
            k, feat, ident, lt, med, p99, mx, r = m.groups()
            rows.append(dict(comparison=k, metric=feat, n_orfs=n, pct_identical=float(ident),
                             n_differ=None, pct_abs_diff_lt_0_01=float(lt), median_abs_diff=float(med),
                             p99_abs_diff=float(p99), max_abs_diff=float(mx), pearson_r=float(r)))
    d = pd.DataFrame(rows)
    if len(d) != 15 or d.comparison.nunique() != 5:
        sys.exit(f"parsed {len(d)} rows / {d.comparison.nunique()} comparisons, expected 15 / 5")
    d.insert(1, "what_differs", d.comparison.map(lambda k: WHAT[k][0]))
    d.insert(2, "run_a", d.comparison.map(lambda k: WHAT[k][1]))
    d.insert(3, "run_b", d.comparison.map(lambda k: WHAT[k][2]))
    p = OUT / "tabG01_encoding_determinism.tsv.gz"
    d.to_csv(p, sep="\t", index=False, compression=GZ)
    print(f"  out  {p.name}  rows={len(d)}  sha256={hashlib.sha256(p.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
