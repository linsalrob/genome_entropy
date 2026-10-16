#!/usr/bin/env python3
"""Six-frame experiment, map step: one record per sampled Prodigal gene with all six frames.

For each complete Prodigal+ ORF (the frame Prodigal selected), collect the
Prodigal- ORFs whose largest overlap is with the same Prodigal gene
(best_gene) and assign each to a frame relative to the selected one:

  s1, s2   same strand, frame offset +1 / +2. Plus strand anchors on start,
           minus strand on end (strand-dependent anchoring, as in #92's
           frame_class); end = start + 3 * (aa_len + stop).
  a0..a2   opposite strand, offset of the alternative's 3' anchor from the
           selected ORF's 5' anchor, modulo 3.

Where a frame has several ORFs overlapping the gene, the one with the
largest overlap is kept. A frame with no ORF >= 30 aa overlapping the gene
is missing (NaN), which is itself informative: the gene's region is
stop-dense in that frame.

Genes are sampled deterministically by hashing (accession, gene id) at
--rate (default 1%). Accessions whose ORFs span two shards lose the
alternatives that fall in the other shard (an edge effect on ~1 accession
per 1,000 chunks).

  14_six_frame.py --shard 000 [--rate 0.01]
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

FEATS = ["aa_len", "aa_entropy", "three_di_entropy", "twelve_state_entropy", "mi_3di_12st"]
FRAMES = ["sel", "s1", "s2", "a0", "a1", "a2"]
COLS = ["accession", "contig_n", "start", "strand", "aa_len", "stop", "five_open", "prodigal_class",
        "match_gene", "match_gene_partial", "best_gene", "best_overlap_bp", "biome", "batch",
        "aa_entropy_q4", "three_di_entropy_q4", "twelve_state_entropy_q4", "mi_3di_12st_q4"]


def keep_gene(acc: pd.Series, gene: pd.Series, rate: float) -> np.ndarray:
    h = np.fromiter((zlib.crc32(f"{a}|{g}".encode()) for a, g in zip(acc, gene)), np.uint32, len(acc))
    return h < rate * 2**32


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", required=True)
    ap.add_argument("--rate", type=float, default=0.01)
    args = ap.parse_args()
    src = lc.root() / "analysis" / lc.CACHE_VERSION / "orfs" / f"{args.shard}.parquet"
    out = lc.root() / "analysis" / lc.CACHE_VERSION / "sixframe" / f"{args.shard}.parquet"
    if out.exists():
        print(f"{out} exists")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)

    # Two streaming passes: a shard holds ~1 B rows, far too many to load.
    pf = pq.ParquetFile(src)

    def prep(t):
        t = t.to_pandas()
        for c in ("accession", "strand", "prodigal_class", "match_gene_partial", "biome", "batch"):
            t[c] = t[c].astype(str)
        for f in FEATS[1:]:
            t[f] = t.pop(f + "_q4") / 1e4
        t["end"] = t.start + 3 * (t.aa_len + t.stop)
        return t

    sels = []
    for g in range(pf.num_row_groups):          # pass 1: sampled selected ORFs
        t = prep(pf.read_row_group(g, columns=COLS))
        s_ = t[(t.prodigal_class == "prodigal_match") & (t.match_gene_partial == "00") & (t.stop == 1)]
        sels.append(s_[keep_gene(s_.accession, s_.match_gene, args.rate)])
    sel = pd.concat(sels, ignore_index=True)
    sel["gene"] = sel.match_gene
    wanted = set(sel.accession + "|" + sel.gene)
    alts = []
    for g in range(pf.num_row_groups):          # pass 2: their alternative-frame ORFs
        t = prep(pf.read_row_group(g, columns=COLS))
        c = t[t.prodigal_class.isin(["antisense_overlap", "frameshift_overlap"]) & (t.best_gene != "")]
        alts.append(c[(c.accession + "|" + c.best_gene).isin(wanted)])
    alt = pd.concat(alts, ignore_index=True).rename(columns={"best_gene": "gene"})
    key = sel[["accession", "gene", "contig_n", "start", "end", "strand"]].rename(
        columns={"contig_n": "s_contig", "start": "s_start", "end": "s_end", "strand": "s_strand"})
    a = alt.merge(key, on=["accession", "gene"], how="inner")
    a = a[a.contig_n == a.s_contig]
    same = a.strand == a.s_strand
    plus = a.s_strand == "+"
    off_same = np.where(plus, (a.start - a.s_start) % 3, (a.end - a.s_end) % 3)
    off_opp = np.where(plus, (a.end - a.s_start) % 3, (a.start - a.s_end) % 3)
    a["frame"] = np.where(same, np.where(off_same == 1, "s1", np.where(off_same == 2, "s2", "s0")),
                          np.char.add("a", off_opp.astype(str)))
    a = a[a.frame != "s0"]
    a = a.sort_values("best_overlap_bp", ascending=False).drop_duplicates(["accession", "gene", "frame"])

    rec = sel[["accession", "gene", "biome", "batch", "contig_n", "start", "strand"] + FEATS].copy()
    rec.columns = ["accession", "gene", "biome", "batch", "contig_n", "start", "strand"] + [f"sel_{f}" for f in FEATS]
    for fr in FRAMES[1:]:
        x = a[a.frame == fr][["accession", "gene", "best_overlap_bp", "five_open"] + FEATS]
        x.columns = ["accession", "gene", f"{fr}_overlap_bp", f"{fr}_five_open"] + [f"{fr}_{f}" for f in FEATS]
        rec = rec.merge(x, on=["accession", "gene"], how="left")
    tmp = out.with_name(out.name + ".partial")
    rec.to_parquet(tmp, compression="zstd", index=False)
    tmp.rename(out)
    present = {fr: int(rec[f"{fr}_aa_len"].notna().sum()) for fr in FRAMES[1:]}
    print(f"shard {args.shard}: {len(rec):,} genes; frames present {present}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
