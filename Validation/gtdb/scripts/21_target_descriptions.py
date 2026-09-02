#!/usr/bin/env python3
"""Target id -> functional description, from the Foldseek header databases.

Issue #97 asks the ranked table for "target annotation/product" and makes
"interpretable target annotation, especially Swiss-Prot, PDB or CATH" a
high-confidence criterion. The m8 output does not carry it: 17_pilot_search.pbs
requests query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits,taxid,taxname
and there is no description among them.

WHY NOT JUST ADD theader TO --format-output

Because the archaeal searches are already complete and the bacterial ones are
already running, and re-running either to collect a string costs hours of node
time. `foldseek convertalis` cannot be re-run after the fact either: the search
driver deletes its result database as soon as it has converted it.

The headers are already on disk. Each <db>_h is a null-separated record per
target whose FIRST TOKEN is exactly the id that appears in the m8 target
column -- verified against real m8 output for all four databases -- and whose
remainder is the description. So the mapping is recoverable offline, for both
domains, uniformly, at no compute cost.

WHAT EACH DATABASE ACTUALLY GIVES

  afdb_swissprot  full product name.  "AF-Q9WY52-F1-model_v6 ATP-dependent
                  6-phosphofructokinase"  -- the most interpretable of the four.
  pdb100          the structure title. "101m-assembly1_A SPERM WHALE MYOGLOBIN
                  F46V N-BUTYL ISOCYANIDE AT PH 9.0"  -- a title, not a
                  curated function, so it can be verbose or refer to a
                  mutant/complex rather than the protein's role.
  cath50          NO free text. "af_A0A023GPK8_21_130_2.60.40.10" encodes the
                  CATH superfamily as the trailing four-part code, which is
                  interpretable only against the CATH hierarchy. Reported as
                  `cath_superfamily`; naming it needs CathNames.txt, which is
                  not held locally.
  bfvd            NO description at all, just a UniProt accession.

That asymmetry matters for how the table is read: a candidate can only get an
"interpretable target annotation" from Swiss-Prot or PDB. Absence of a
description for a CATH or BFVD hit is a property of the reference database,
not weak evidence, and must not be scored as such.

  21_target_descriptions.py --db-root <foldseek_db> --out-dir <dir>
"""
import argparse
import re
import sys
from pathlib import Path

CATH_SF = re.compile(r"_(\d+\.\d+\.\d+\.\d+)$")


def read_headers(path, chunk_bytes=1 << 22):
    """Yield (id, description) from a null-separated Foldseek header db."""
    tail = b""
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk_bytes)
            if not buf:
                break
            parts = (tail + buf).split(b"\0")
            tail = parts.pop()
            for rec in parts:
                rec = rec.decode("utf-8", "replace").strip()
                if not rec:
                    continue
                tid, _, desc = rec.partition(" ")
                yield tid, desc.strip()
    rec = tail.decode("utf-8", "replace").strip()
    if rec:
        tid, _, desc = rec.partition(" ")
        yield tid, desc.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db-root",
                    default="/g/data/ob80/re3494/gtdb_entropy/foldseek_db")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dbs", default="afdb_swissprot,pdb100,cath50,bfvd")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0

    for db in [d.strip() for d in args.dbs.split(",") if d.strip()]:
        src = Path(args.db_root) / f"{db}_h"
        if not src.exists():
            print(f"WARNING: no {src}, skipped", file=sys.stderr)
            rc = 1
            continue
        out = out_dir / f"{db}.desc.tsv"
        n, n_desc = 0, 0
        with open(f"{out}.partial", "w") as fh:
            fh.write("target\tdescription\tcath_superfamily\n")
            for tid, desc in read_headers(src):
                sf = ""
                if db == "cath50":
                    m = CATH_SF.search(tid)
                    sf = m.group(1) if m else ""
                n += 1
                if desc:
                    n_desc += 1
                fh.write(f"{tid}\t{desc}\t{sf}\n")
        Path(f"{out}.partial").rename(out)
        pct = 100.0 * n_desc / n if n else 0.0
        print(f"{db:<16} {n:>10,} targets, {n_desc:>10,} with a description "
              f"({pct:5.1f}%)  -> {out.name}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
