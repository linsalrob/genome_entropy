#!/usr/bin/env python3
"""Extra Foldseek arms: frameshift-shadow controls and low- vs high-scoring Prodigal genes.

Completes #102 §5 and §7 alongside 21_candidates_select.py, written to a
separate directory (default qc/candidates_extra) so the main query set and
its results are untouched:

  frameshift   frameshift shadows (same-strand, other-frame overlaps of
               Prodigal genes), nearest-length matched to the candidate arm
               of qc/candidates/selection.parquet within truncation state;
               the matched control #102 §7.3 asks for beside antisense
  prodigal_lo  complete Prodigal genes scoring below the 0.80 threshold
               (the "Prodigal+ / low genome_entropy" discordant class of
               §5), up to --per-bin from each length bin, 30 aa upward
  prodigal_hi  complete Prodigal genes scoring at or above the 0.90
               threshold, nearest-length matched to prodigal_lo from 50 aa
               up (high-scoring genes below 50 aa barely exist: 6 of 13,259
               in the scan's 1e-3 sample)

Pools are the scan's sampled controls (analysis/v1/candidates, role
"control"). Writes <out>/selection.parquet with the same columns as the main
selection, plus selection.json.

  21b_extra_arms.py --thresholds qc/analysis_models_final/final_model_thresholds.json \
      --main qc/candidates/selection.parquet --out qc/candidates_extra
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logan_common as lc  # noqa: E402

_spec = importlib.util.spec_from_file_location("sel", HERE / "21_candidates_select.py")
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)

BINS = [30, 50, 75, 100, 150, 200, 300, 500, 10**9]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--main", required=True, help="main selection.parquet (candidate arm to match)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bin", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1022)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thr = json.load(open(args.thresholds))["precision_thresholds"]
    t80, t90 = float(thr["0.8"]), float(thr["0.9"])
    rng = np.random.default_rng(args.seed)

    files = sorted(glob.glob(str(lc.root() / "analysis" / lc.CACHE_VERSION / "candidates" / "*.parquet")))
    d = pd.concat([pd.read_parquet(f, filters=[("role", "=", "control")]) for f in files], ignore_index=True)
    cand = pd.read_parquet(args.main)
    cand = cand[cand.arm == "candidate"]
    keep = [c for c in cand.columns if c in d.columns]

    fs = SEL.match(d[d.label == "frameshift_overlap"], cand, rng).assign(arm="frameshift")

    p = d[d.label == "prodigal_complete"].copy()
    p["bin"] = pd.cut(p.aa_len, BINS, right=False)
    lo = (p[p.score < t80].groupby("bin", observed=True, group_keys=False)
          .apply(lambda g: g.sample(min(len(g), args.per_bin), random_state=rng.integers(1 << 31))))
    lo = lo.assign(arm="prodigal_lo")
    hi = SEL.match(p[p.score >= t90], lo[lo.aa_len >= 50], rng).assign(arm="prodigal_hi")

    sel = pd.concat([fs, lo, hi], ignore_index=True)[keep + ["arm"]]
    sel["length_bin"] = pd.cut(sel.aa_len, SEL.LEN_EDGES, right=False).astype(str)
    sel["end"] = sel.start + 3 * (sel.aa_len + sel.stop)
    sel["orf_id"] = (sel.accession + "_" + sel.contig_n.astype(str) + ":" + sel.start.astype(str) + "-"
                     + sel.end.astype(str) + ":" + sel.strand)
    sel["query_id"] = sel.arm + "|" + sel.orf_id
    assert not sel.query_id.duplicated().any()
    sel.to_parquet(out / "selection.parquet", index=False)
    json.dump({"t80": t80, "t90": t90, "per_bin": args.per_bin, "seed": args.seed,
               "arms": sel.arm.value_counts().to_dict()}, open(out / "selection.json", "w"), indent=1)
    print(sel.groupby("arm").aa_len.describe().round(1).to_string())
    print(sel.assign(b=pd.cut(sel.aa_len, BINS, right=False)).groupby(["arm", "b"], observed=True).size()
          .unstack(0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
