#!/usr/bin/env python3
"""Missed-gene scan, map step: score intergenic Prodigal- ORFs >= 100 aa, plus controls.

Every intergenic Prodigal- ORF (no overlap with any Prodigal gene) of at
least --min-aa amino acids in the shard is kept, in every truncation state,
and scored with the final complete-ORF model from 13_models.py (length + AA /
3Di / 12-state entropy + 3Di-12st MI; Prodigal+ complete vs all Prodigal-).
Deterministic samples of three control classes are scored the same way, so
candidates can always be read against a background:

  prodigal_complete     complete Prodigal+ genes                  (rate 1e-3)
  antisense_overlap     antisense shadows >= min-aa               (rate 1e-3)
  frameshift_overlap    same-strand frameshift shadows >= min-aa  (rate 1e-3)

Each row keeps what is needed to fetch its sequences from the cache later:
accession, contig_n, start, strand, aa_len, stop, chunk.

  18_score_candidates.py --shard 000 --model qc/analysis_models_final/final_model.joblib
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logan_common as lc  # noqa: E402

_spec = importlib.util.spec_from_file_location("h", HERE / "10_feature_histograms.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

COLS = ["accession", "contig_n", "start", "strand", "contig_len", "stop", "five_open", "aa_len",
        "prodigal_class", "match_gene_partial", "biome", "batch", "chunk",
        "aa_entropy_q4", "three_di_entropy_q4", "twelve_state_entropy_q4", "mi_3di_12st_q4"]
CONTROL_RATE = {"prodigal_complete": 1e-3, "antisense_overlap": 1e-3, "frameshift_overlap": 1e-3}


def main() -> int:
    import joblib
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--min-aa", type=int, default=100)
    args = ap.parse_args()
    src = lc.root() / "analysis" / lc.CACHE_VERSION / "orfs" / f"{args.shard}.parquet"
    out = lc.root() / "analysis" / lc.CACHE_VERSION / "candidates" / f"{args.shard}.parquet"
    if out.exists():
        print(f"{out} exists")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    M = joblib.load(args.model)
    pf = pq.ParquetFile(src)
    parts, n_cand = [], 0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=COLS)
        s = H.strata(t)
        lab, tr = np.unravel_index(s, H.SHAPE)[:2]
        label = np.array(H.LABELS)[lab]
        aa = t.column("aa_len").to_numpy()
        cand = (label == "intergenic") & (aa >= args.min_aa)
        rng = np.random.default_rng([int(args.shard), g, 18])
        u = rng.random(len(label))
        ctrl = np.zeros(len(label), bool)
        for name, r in CONTROL_RATE.items():
            ok = (label == name) & (u < r)
            if name != "prodigal_complete":
                ok &= aa >= args.min_aa
            ctrl |= ok
        keep = cand | ctrl
        if not keep.any():
            continue
        d = t.filter(keep).to_pandas()
        for c in ("accession", "strand", "prodigal_class", "match_gene_partial", "biome", "batch"):
            d[c] = d[c].astype(str)
        d["label"] = label[keep]
        d["trunc"] = np.array(H.TRUNC)[tr[keep]]
        d["role"] = np.where(cand[keep], "candidate", "control")
        d["log_len"] = np.log(d.aa_len.astype(float))
        for f in ("aa_entropy", "three_di_entropy", "twelve_state_entropy", "mi_3di_12st"):
            d[f] = d.pop(f + "_q4") / 1e4
        d["score"] = M["model"].predict_proba(d[M["features"]].to_numpy())[:, 1]
        d["score_length_only"] = M["length_model"].predict_proba(d[["log_len"]].to_numpy())[:, 1]
        parts.append(d.drop(columns=["log_len"]))
        n_cand += int(cand.sum())
    res = pd.concat(parts, ignore_index=True)
    tmp = out.with_name(out.name + ".partial")
    res.to_parquet(tmp, compression="zstd", index=False)
    tmp.rename(out)
    print(f"shard {args.shard}: {n_cand:,} candidates, {int((res.role == 'control').sum()):,} controls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
