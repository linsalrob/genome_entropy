#!/usr/bin/env python3
"""Fetch the contigs that carry the Foldseek query ORFs, and rerun Prodigal on them.

The full LOGAN contig files are no longer on /scratch and the Prodigal
bundles are only on OneDrive, so the contigs are streamed again from LOGAN
(curl | zstd -dc | awk) and only those carrying a query ORF (all arms of
qc/candidates/selection.parquet) are kept. Accessions are dealt round-robin
to parts (--part i of --parts k); each part runs --workers streams at once.

Every wanted contig must be found exactly once with the length recorded in
the selection; accessions where this fails are reported in the part's JSON,
not silently dropped.

Prodigal -p meta scores each sequence on its own (best of its 50
metagenomic models), so running it on the extracted contigs reproduces the
original calls on them (checked downstream against the selection's
Prodigal+ ORFs). -s also writes every potential gene with its scores, which
is what tells us why an ORF was not called.

Outputs in <out>/: part_<i>.fa, part_<i>.gff, part_<i>.starts, part_<i>.json

  27_fetch_query_contigs.py --selection qc/candidates/selection.parquet \
      --out qc/candidates/contigs --part 1 --parts 24 --workers 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

URL = "https://s3.amazonaws.com/logan-pub/c/{acc}/{acc}.contigs.fa.zst"
AWK = (r'BEGIN { while ((getline l < want) > 0) w[l] = 1 } '
       r'/^>/ { n = substr($1, 2); keep = (n in w) } keep')


def fetch(acc: str, want: dict[str, int], tmpdir: str) -> tuple[str, dict, str]:
    """Stream one accession; return (acc, {contig: seq}, error)."""
    wf = Path(tmpdir) / f"{acc}.want"
    wf.write_text("\n".join(want) + "\n")
    cmd = (f"set -o pipefail; curl -sSf --retry 5 --retry-delay 10 --retry-all-errors {URL.format(acc=acc)} "
           f"| zstd -dcq | awk -v want={wf} '{AWK}'")
    for attempt in range(3):
        p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        if p.returncode == 0:
            break
        time.sleep(30)
    wf.unlink()
    if p.returncode != 0:
        return acc, {}, f"exit {p.returncode}: {p.stderr.strip()[:200]}"
    seqs: dict[str, list[str]] = {}
    name = None
    for line in p.stdout.splitlines():
        if line.startswith(">"):
            name = line[1:].split(None, 1)[0]
            if name in seqs:
                return acc, {}, f"{name} found twice"
            seqs[name] = []
        elif name is not None:
            seqs[name].append(line.strip())
    out = {k: "".join(v) for k, v in seqs.items()}
    missing = [c for c in want if c not in out]
    if missing:
        return acc, {}, f"{len(missing)} contigs missing, e.g. {missing[0]}"
    bad = [c for c, L in want.items() if len(out[c]) != L]
    if bad:
        return acc, {}, f"{len(bad)} contigs differ in length, e.g. {bad[0]}"
    return acc, out, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--parts", type=int, required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(args.selection, columns=["accession", "contig_n", "contig_len"])
    s["contig"] = s.accession + "_" + s.contig_n.astype(str)
    s = s.drop_duplicates("contig")
    accs = sorted(s.accession.unique())[args.part - 1::args.parts]
    want = {a: dict(zip(g.contig, g.contig_len.astype(int))) for a, g in s[s.accession.isin(accs)].groupby("accession")}
    print(f"part {args.part}/{args.parts}: {len(accs)} accessions, {sum(map(len, want.values()))} contigs", flush=True)

    fa = out / f"part_{args.part}.fa"
    errors, n_contigs, n_bp = {}, 0, 0
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td, open(f"{fa}.partial", "w") as fo, \
            ThreadPoolExecutor(args.workers) as ex:
        for i, (acc, seqs, err) in enumerate(ex.map(lambda a: fetch(a, want[a], td), accs), 1):
            if err:
                errors[acc] = err
                continue
            for c in sorted(seqs):
                fo.write(f">{c}\n{seqs[c]}\n")
                n_contigs += 1
                n_bp += len(seqs[c])
            if i % 200 == 0:
                print(f"  {i}/{len(accs)} accessions, {time.time() - t0:.0f} s", flush=True)
    Path(f"{fa}.partial").rename(fa)

    gff, starts = out / f"part_{args.part}.gff", out / f"part_{args.part}.starts"
    t1 = time.time()
    subprocess.run(["prodigal", "-q", "-p", "meta", "-i", str(fa), "-f", "gff", "-o", str(gff), "-s", str(starts)],
                   check=True)
    info = {"part": args.part, "parts": args.parts, "accessions": len(accs),
            "accessions_failed": len(errors), "errors": errors, "contigs": n_contigs, "bp": n_bp,
            "fetch_seconds": round(t1 - t0), "prodigal_seconds": round(time.time() - t1),
            "prodigal_command": "prodigal -q -p meta -i part.fa -f gff -o part.gff -s part.starts"}
    json.dump(info, open(out / f"part_{args.part}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "errors"}))
    if errors:
        print(f"WARNING: {len(errors)} accessions failed; first: {next(iter(errors.items()))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
