#!/usr/bin/env python3
"""Map step: stratified feature histograms for one analysis shard.

Every ORF in analysis/v1/orfs/<shard>.parquet is assigned to a stratum

  label     prodigal_complete | prodigal_partial | antisense_overlap |
            frameshift_overlap | intergenic | marginal_overlap |
            same_frame_unmatched | frame_undefined_overlap
  trunc     complete | open5 | open3
  lenbin    ORF length bins (aa): 30,50,75,100,150,200,300,500,1000,inf
  biome     13 biomes + other
  batch     pilot | phaseb1 | phaseb2

and each of five features is histogrammed in fixed bins:

  aa_len (log-spaced, 400 bins 30..40,000 aa) and AA / 3Di / 12-state
  entropy and 3Di-12st MI (0.01-bit bins, 0..4.4 bits).

"Complete" follows the definition agreed on #102: a Prodigal+ ORF is
complete when its matched Prodigal gene is partial=00 and the ORF ends in a
stop (the stop-to-stop ORF may still be open upstream, since Prodigal trims
to the first start codon); a Prodigal- ORF is complete when it ends in a
stop and has an in-frame stop upstream (five_open = 0).

The reduce step (11_summarise_histograms.py) sums the arrays over shards;
exact AUC / average precision for any contrast and stratum follow from the
per-class histograms (ties at 0.01 bits are counted as half).

  10_feature_histograms.py --shard 000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

LABELS = ["prodigal_complete", "prodigal_partial", "antisense_overlap", "frameshift_overlap",
          "intergenic", "marginal_overlap", "same_frame_unmatched", "frame_undefined_overlap"]
TRUNC = ["complete", "open5", "open3"]
LEN_EDGES = np.array([30, 50, 75, 100, 150, 200, 300, 500, 1000, 10**9])
BIOMES = ["animal_host", "built_environment", "extreme", "food", "freshwater", "human_gut",
          "human_other", "marine", "other", "plant", "sediment", "soil", "wastewater"]
BATCHES = ["pilot", "phaseb1", "phaseb2"]
ENT_BINS = 440                                     # 0.01-bit bins over 0..4.4 bits
LEN_BINS = np.unique(np.round(np.logspace(np.log10(30), np.log10(40000), 400)).astype(int))
FEATURES = ["aa_len", "aa_entropy", "three_di_entropy", "twelve_state_entropy", "mi_3di_12st"]
SHAPE = (len(LABELS), len(TRUNC), len(LEN_EDGES) - 1, len(BIOMES) + 1, len(BATCHES))


def out_dir() -> Path:
    return lc.root() / "analysis" / lc.CACHE_VERSION / "hist"


def _coded(col, values, default=-1):
    """Map a dictionary-encoded string column to indices into `values` via its codes.

    Converting to Python strings cost ~2 s per million rows; mapping the
    dictionary and gathering by code is a few array operations."""
    import pyarrow as pa
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    if not pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_encode()
    lut = np.array([values.index(v) if v in values else default for v in arr.dictionary.to_pylist()],
                   np.int64)
    return lut[arr.indices.to_numpy(zero_copy_only=False)]


def strata(t):
    """Integer stratum indices for a row group (pyarrow table -> numpy)."""
    PC = ["prodigal_match"] + LABELS[2:]
    pci = _coded(t.column("prodigal_class"), PC)
    pc = np.array(PC + ["?"])[pci]
    part = np.array(["00", "other"])[np.where(_coded(t.column("match_gene_partial"), ["00"]) == 0, 0, 1)]
    stop = t.column("stop").to_numpy().astype(bool)
    open5 = t.column("five_open").to_numpy().astype(bool)
    aa = t.column("aa_len").to_numpy()
    lab = np.full(len(pc), -1, np.int64)
    for i, name in enumerate(LABELS[2:], start=2):
        lab[pc == name] = i
    pmatch = pc == "prodigal_match"
    pcomplete = pmatch & (part == "00") & stop
    lab[pmatch] = 1
    lab[pcomplete] = 0
    if (lab < 0).any():
        raise RuntimeError(f"unknown prodigal_class values: {set(pc[lab < 0])}")
    trunc = np.where(~stop, 2, np.where(pcomplete | ~open5, 0, 1))
    lenb = np.clip(np.searchsorted(LEN_EDGES, aa, side="right") - 1, 0, len(LEN_EDGES) - 2)
    bidx = _coded(t.column("biome"), BIOMES, default=len(BIOMES))
    tidx = _coded(t.column("batch"), BATCHES, default=0)
    return np.ravel_multi_index((lab, trunc, lenb, bidx, tidx), SHAPE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", required=True)
    args = ap.parse_args()
    src = lc.root() / "analysis" / lc.CACHE_VERSION / "orfs" / f"{args.shard}.parquet"
    out = out_dir() / f"{args.shard}.npz"
    if out.exists():
        print(f"{out} exists")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    n_strata = int(np.prod(SHAPE))
    hist = {f: np.zeros((n_strata, (len(LEN_BINS) if f == "aa_len" else ENT_BINS)), np.int64)
            for f in FEATURES}
    cols = ["prodigal_class", "match_gene_partial", "stop", "five_open", "aa_len", "biome", "batch",
            "aa_entropy_q4", "three_di_entropy_q4", "twelve_state_entropy_q4", "mi_3di_12st_q4"]
    pf = pq.ParquetFile(src)
    rows = 0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=cols)
        s = strata(t)
        aa = t.column("aa_len").to_numpy()
        lb = np.clip(np.searchsorted(LEN_BINS, aa, side="right") - 1, 0, len(LEN_BINS) - 1)
        nb = hist["aa_len"].shape[1]
        hist["aa_len"] += np.bincount(s * nb + lb, minlength=n_strata * nb).reshape(n_strata, nb)
        for f in FEATURES[1:]:
            q = t.column(f + "_q4").to_numpy().astype(np.int64)
            b = np.minimum(q // 100, ENT_BINS - 1)        # q4 is 1e-4 bits; bins are 0.01 bits
            hist[f] += np.bincount(s * ENT_BINS + b, minlength=n_strata * ENT_BINS).reshape(n_strata, ENT_BINS)
        rows += len(s)
    if rows != pf.metadata.num_rows:
        raise RuntimeError("row count mismatch")
    tmp = out.with_name(out.name + ".partial.npz")
    np.savez_compressed(tmp, rows=rows, **hist)
    tmp.rename(out)
    print(f"shard {args.shard}: {rows:,} rows histogrammed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
