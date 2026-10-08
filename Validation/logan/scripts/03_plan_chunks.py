#!/usr/bin/env python3
"""Pack accessions whose ORFs are called into ~1 M-ORF cache chunks.

A chunk is a list of (accession, first_row, n_rows) slices of each
accession's orfs.tsv.zst, so a large metagenome can span several chunks and a
small one shares a chunk with others. Chunk ids are global and never reused:
the next id is one past the highest manifest on disk, and an accession slice
already assigned to a chunk is never planned again. Re-running is therefore a
no-op until more accessions reach ``orfs_called``.

The last, partly filled chunk is held back unless --final is given, so
Phase B can keep appending accessions to the same sequence of chunk sizes.

  03_plan_chunks.py --accessions manifests/pilot_accessions.tsv --final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

PREFIX = "logan_metagenome"


def chunk_dir() -> Path:
    return lc.root() / "manifests" / "chunks"


def existing_chunks():
    d = chunk_dir()
    d.mkdir(parents=True, exist_ok=True)
    return sorted(d.glob(f"{PREFIX}_*.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--accessions", required=True, help="TSV with an accession column")
    ap.add_argument("--target-orfs", type=int, default=1_000_000)
    ap.add_argument("--final", action="store_true",
                    help="also emit the last, partly filled chunk")
    args = ap.parse_args()

    import json
    accs = pd.read_csv(args.accessions, sep="\t").accession.tolist()
    assigned = {}
    for p in existing_chunks():
        for m in json.load(open(p))["members"]:
            assigned[m["accession"]] = assigned.get(m["accession"], 0) + m["n_rows"]

    pending = []
    for acc in accs:
        st = lc.load_state(acc)
        if not lc.reached(st, "orfs_called"):
            continue
        n = int(st["info"]["orfs"])
        done = assigned.get(acc, 0)
        if done < n:
            pending.append((acc, done, n - done))
    if not pending:
        print("nothing to plan")
        return 0

    next_id = len(existing_chunks()) + 1
    chunks, cur, cur_n = [], [], 0
    for acc, start, n in pending:
        while n > 0:
            take = min(n, args.target_orfs - cur_n)
            cur.append({"accession": acc, "first_row": start, "n_rows": take})
            cur_n += take
            start += take
            n -= take
            if cur_n >= args.target_orfs:
                chunks.append(cur)
                cur, cur_n = [], 0
    if cur and args.final:
        chunks.append(cur)
    elif cur:
        print(f"holding back a partial chunk of {cur_n:,} ORFs (use --final)")

    new_chunks: dict = {}
    for members in chunks:
        cid = f"{PREFIX}_{next_id:06d}"
        next_id += 1
        man = {
            "chunk_id": cid,
            "state": "planned",
            "planned": lc.now(),
            "logan_release": lc.LOGAN_RELEASE,
            "model_name": lc.MODEL_NAME,
            "cache_version": lc.CACHE_VERSION,
            "members": members,
            "accession_count": len({m["accession"] for m in members}),
            "orf_count_expected": sum(m["n_rows"] for m in members),
        }
        lc.atomic_write_json(chunk_dir() / f"{cid}.json", man)
        for m in members:
            new_chunks.setdefault(m["accession"], []).append(cid)
        if len(chunks) <= 20:
            print(f"{cid}: {man['accession_count']} accessions, "
                  f"{man['orf_count_expected']:,} ORFs")
    # Record each accession's chunks in its state, so later stages never
    # have to scan every manifest to find them (tens of thousands at scale).
    for acc, ids in new_chunks.items():
        prev = lc.load_state(acc).get("info", {}).get("chunks", [])
        lc.update_info(acc, chunks=list(dict.fromkeys(prev + ids)))
    print(f"planned {len(chunks):,} chunks for {len(new_chunks):,} accessions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
