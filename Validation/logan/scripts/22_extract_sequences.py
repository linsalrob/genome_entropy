#!/usr/bin/env python3
"""Fetch AA and 3Di strings for the Foldseek query set from the encoded cache.

Reads qc/candidates/selection.parquet (21_candidates_select.py), and for the
cache chunks of one shard (chunk // 1000 == shard) pulls the rows whose
orf_id is selected. Writes <out>/seqs/<shard>.tsv with query_id, aa, three_di.
Run as an array over shards; 23_foldseek_search.slurm concatenates them.

Every selected ORF must be found exactly once and its AA length must equal
the selection's aa_len; otherwise the shard fails rather than silently
dropping queries.

  22_extract_sequences.py --shard 041 --selection qc/candidates/selection.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", required=True)
    ap.add_argument("--selection", required=True)
    args = ap.parse_args()
    sel = pd.read_parquet(args.selection, columns=["query_id", "orf_id", "chunk", "aa_len"])
    sel = sel[sel.chunk // 1000 == int(args.shard)]
    out_dir = Path(args.selection).parent / "seqs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.shard}.tsv"
    if sel.empty:
        out.write_text("query_id\taa\tthree_di\n")
        print(f"shard {args.shard}: nothing selected")
        return 0
    found = []
    for chunk, g in sel.groupby("chunk"):
        cid = f"logan_metagenome_{int(chunk):06d}"
        m = json.load(open(lc.chunk_manifest_path(cid)))
        f = lc.root() / "cache" / "modernprost-50M" / lc.CACHE_VERSION / m["cache_file"]
        want = set(g.orf_id)
        df = pd.read_csv(f, sep="\t", usecols=["orf_id", "aa_sequence", "three_di"], dtype=str,
                         keep_default_na=False, compression={"method": "zstd"})
        hit = df[df.orf_id.isin(want)]
        found.append(g.merge(hit, on="orf_id", how="left"))
    res = pd.concat(found, ignore_index=True)
    if res.aa_sequence.isna().any():
        raise RuntimeError(f"{int(res.aa_sequence.isna().sum())} selected ORFs not found in the cache")
    if res.orf_id.duplicated().any() and res.query_id.duplicated().any():
        raise RuntimeError("selected ORF found more than once")
    if not (res.aa_sequence.str.len() == res.aa_len).all():
        raise RuntimeError("cached AA length differs from the selection")
    res.rename(columns={"aa_sequence": "aa"})[["query_id", "aa", "three_di"]].to_csv(out, sep="\t", index=False)
    print(f"shard {args.shard}: {len(res):,} sequences from {sel.chunk.nunique():,} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
