#!/usr/bin/env python3
"""Missed-gene candidates: summarise the scan and draw the Foldseek query set with matched controls.

Reads analysis/v1/candidates/*.parquet (18_score_candidates.py) and the score
thresholds from 13_models.py (final_model_thresholds.json: the out-of-fold
scores at which population-weighted precision of Prodigal+ among complete
ORFs reaches 0.80 / 0.90 / 0.95 / 0.99).

Summary (to --out):
  candidate_counts.tsv     intergenic ORFs >= 100 aa above each threshold, by
                           truncation state and length bin
  candidate_biomes.tsv     the same at the 0.90 threshold, by biome, with the
                           biome's share of all scanned intergenic ORFs
  score_by_role.tsv        score quantiles for candidates and each control

Foldseek query set (selection.parquet, arm column), following #92's design:
  candidate       high-score intergenic ORFs (score >= t0.90), complete or
                  5'-open, sampled stratified by length bin
  intergenic_lo   intergenic ORFs with score < t0.80, nearest-length matched
                  without replacement, same truncation state
  antisense       antisense shadows, nearest-length matched (structured-
                  looking 3Di that should NOT find full-length homologues)
  prodigal        complete Prodigal+ genes, nearest-length matched (positive
                  control: should)
A shuffled-3Di null of the candidate arm is generated at search time.

  21_candidates_select.py --thresholds qc/analysis_models_final/final_model_thresholds.json \
      --out qc/candidates --n 20000
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

LEN_EDGES = [100, 150, 200, 300, 500, 1000, 10**9]


def match(pool: pd.DataFrame, targets: pd.DataFrame, rng) -> pd.DataFrame:
    """Nearest-length match without replacement, within truncation state."""
    out = []
    for tr, tg in targets.groupby("trunc"):
        p = pool[pool.trunc == tr].sample(frac=1, random_state=rng.integers(1 << 31))
        if p.empty:
            continue
        p = p.sort_values("aa_len", kind="stable").reset_index(drop=True)
        lens = p.aa_len.to_numpy()
        used = np.zeros(len(p), bool)
        for L in tg.aa_len.to_numpy():
            i = int(np.searchsorted(lens, L))
            # walk outwards to the nearest unused
            lo, hi = i - 1, i
            while lo >= 0 or hi < len(p):
                cand = []
                if hi < len(p) and not used[hi]:
                    cand.append(hi)
                if lo >= 0 and not used[lo]:
                    cand.append(lo)
                if cand:
                    j = min(cand, key=lambda k: abs(lens[k] - L))
                    used[j] = True
                    break
                lo -= 1
                hi += 1
        out.append(p[used])
    return pd.concat(out, ignore_index=True) if out else pool.iloc[0:0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000, help="candidate arm size")
    ap.add_argument("--seed", type=int, default=102)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thr = {float(k): v for k, v in json.load(open(args.thresholds))["precision_thresholds"].items()
           if v is not None}
    files = sorted(glob.glob(str(lc.root() / "analysis" / lc.CACHE_VERSION / "candidates" / "*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["length_bin"] = pd.cut(d.aa_len, LEN_EDGES, right=False).astype(str)
    cand = d[d.role == "candidate"]
    print(f"{len(files)} shards; {len(cand):,} intergenic ORFs >= 100 aa scanned; "
          f"{int((d.role == 'control').sum()):,} controls")

    rows = []
    for p, t in sorted(thr.items()):
        hit = cand[cand.score >= t]
        for (tr, lb), g in hit.groupby(["trunc", "length_bin"]):
            rows.append({"precision_target": p, "score_threshold": t, "trunc": tr, "length_bin": lb, "orfs": len(g),
                         "accessions": g.accession.nunique()})
        rows.append({"precision_target": p, "score_threshold": t, "trunc": "all", "length_bin": "all",
                     "orfs": len(hit), "accessions": hit.accession.nunique()})
    C = pd.DataFrame(rows)
    C.to_csv(out / "candidate_counts.tsv", sep="\t", index=False)
    t90 = thr[0.9]
    t80 = thr.get(0.8, t90)
    bio = cand.groupby("biome").size().rename("scanned").to_frame()
    bio["above_t90"] = cand[cand.score >= t90].groupby("biome").size()
    bio = bio.fillna(0)
    bio["rate_per_million_scanned"] = 1e6 * bio.above_t90 / bio.scanned
    bio.to_csv(out / "candidate_biomes.tsv", sep="\t")
    q = d.assign(group=np.where(d.role == "candidate", "intergenic>=100aa", d.label)).groupby("group").score.describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    q.to_csv(out / "score_by_role.tsv", sep="\t")
    print(C[C.trunc == "all"].to_string(index=False))
    print(q.round(4).to_string())

    rng = np.random.default_rng(args.seed)
    hi = cand[(cand.score >= t90) & cand.trunc.isin(["complete", "open5"])]
    # Stratify by length bin: up to n / bins from each, rest filled from the remainder.
    per = max(1, args.n // hi.length_bin.nunique()) if len(hi) else 0
    arm = (hi.groupby("length_bin", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), per), random_state=rng.integers(1 << 31))))
    if len(arm) < min(args.n, len(hi)):
        rest = hi.drop(arm.index)
        arm = pd.concat([arm, rest.sample(min(len(rest), args.n - len(arm)), random_state=args.seed)])
    arm = arm.assign(arm="candidate")
    lo = match(cand[(cand.score < t80) & cand.trunc.isin(["complete", "open5"])], arm, rng).assign(arm="intergenic_lo")
    anti = match(d[(d.role == "control") & (d.label == "antisense_overlap")], arm, rng).assign(arm="antisense")
    prod_pool = d[(d.role == "control") & (d.label == "prodigal_complete")].assign(trunc="complete")
    prod = match(prod_pool, arm.assign(trunc="complete"), rng).assign(arm="prodigal")
    sel = pd.concat([arm, lo, anti, prod], ignore_index=True)
    sel["end"] = sel.start + 3 * (sel.aa_len + sel.stop)
    sel["orf_id"] = (sel.accession + "_" + sel.contig_n.astype(str) + ":" + sel.start.astype(str) + "-"
                     + sel.end.astype(str) + ":" + sel.strand)
    sel["query_id"] = sel.arm + "|" + sel.orf_id
    sel.to_parquet(out / "selection.parquet", index=False)
    print(sel.groupby("arm").aa_len.describe().round(1).to_string())
    json.dump({"shards": len(files), "scanned": int(len(cand)), "t80": t80, "t90": t90,
               "arms": sel.arm.value_counts().to_dict()}, open(out / "selection.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
