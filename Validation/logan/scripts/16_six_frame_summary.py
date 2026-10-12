#!/usr/bin/env python3
"""Six-frame experiment, reduce step: cross-frame inheritance and six-frame discrimination.

Reads analysis/v1/sixframe/*.parquet (14_six_frame.py): one row per sampled
complete Prodigal+ gene, with features of the selected frame (sel) and of
the best-overlapping ORF in each alternative frame (s1, s2 same strand;
a0, a1, a2 opposite strand), NaN where no ORF >= 30 aa overlaps the gene in
that frame.

A. Inheritance: for each feature and alternative frame, Pearson and
   Spearman correlation between the selected and the alternative value
   across genes, and the mean difference.

B. Discrimination: for each feature, the rank of the selected frame among
   the frames present (higher = more coding-like for every feature here).
   Reported: rank-1, top-2 and top-3 rates against chance (mean of
   1/(k+1), 2/(k+1), 3/(k+1) over genes with k alternatives), overall and
   for genes with all five alternatives present (chance exactly 1/6), and
   delta_X = X_sel - median(X_alternatives).

Ties (equal values, e.g. integer lengths or the 1e-4-bit quantisation) are
broken at random in expectation: with g frames strictly better and t tied
including the selected one, P(top-k) = clip((k - g) / t, 0, 1). A feature
with no information therefore scores exactly chance.

  16_six_frame_summary.py --out qc/six_frame
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

FEATS = ["aa_len", "aa_entropy", "three_di_entropy", "twelve_state_entropy", "mi_3di_12st"]
ALT = ["s1", "s2", "a0", "a1", "a2"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(lc.root() / "analysis" / lc.CACHE_VERSION / "sixframe" / "*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    present = np.column_stack([d[f"{a}_aa_len"].notna().to_numpy() for a in ALT])
    k = present.sum(axis=1)
    print(f"{len(files)} shards, {len(d):,} genes; alternatives present per gene: "
          f"{pd.Series(k).value_counts().sort_index().to_dict()}")

    inh = []
    for f in FEATS:
        for a in ALT:
            x, y = d[f"sel_{f}"], d[f"{a}_{f}"]
            m = y.notna()
            inh.append({"feature": f, "frame": a, "genes": int(m.sum()),
                        "pearson": x[m].corr(y[m]), "spearman": x[m].corr(y[m], method="spearman"),
                        "mean_sel": x[m].mean(), "mean_alt": y[m].mean(),
                        "mean_delta": (x[m] - y[m]).mean()})
    I = pd.DataFrame(inh)
    I.to_csv(out / "inheritance.tsv", sep="\t", index=False)

    disc, deltas = [], {}
    for f in FEATS:
        sel = d[f"sel_{f}"].to_numpy()
        alts = np.column_stack([d[f"{a}_{f}"].to_numpy() for a in ALT])
        greater = np.where(present, alts > sel[:, None], False).sum(axis=1)
        ties = np.where(present, alts == sel[:, None], False).sum(axis=1)
        # P(selected in top-k) with ties broken at random: g frames strictly
        # better and t frames (including selected) tied -> clip((k - g) / t).
        topk = lambda kk: np.clip((kk - greater) / (1 + ties), 0, 1)  # noqa: E731
        med = np.where(k > 0, np.nanmedian(np.where(present, alts, np.nan), axis=1), np.nan)
        deltas[f] = sel - med
        for scope, msk in (("genes with >= 1 alternative", k >= 1), ("genes with all 5 alternatives", k == 5)):
            kk = k[msk]
            row = {"feature": f, "scope": scope, "genes": int(msk.sum())}
            for top in (1, 2, 3):
                row[f"top{top}_rate"] = float(np.mean(topk(top)[msk]))
                row[f"top{top}_chance"] = float(np.mean(np.minimum(top, kk + 1) / (kk + 1)))
            row["median_delta"] = float(np.nanmedian(deltas[f][msk]))
            row["frac_delta_positive"] = float(np.nanmean(deltas[f][msk] > 0))
            disc.append(row)
    D = pd.DataFrame(disc)
    D.to_csv(out / "discrimination.tsv", sep="\t", index=False)
    pd.DataFrame({f"delta_{f}": v for f, v in deltas.items()}).assign(k=k).sample(
        n=min(500_000, len(d)), random_state=102).to_parquet(out / "delta_sample.parquet", index=False)
    json.dump({"shards": len(files), "genes": int(len(d)),
               "alternatives_present": {int(a): int(b) for a, b in pd.Series(k).value_counts().items()}},
              open(out / "inputs.json", "w"), indent=1)
    with pd.option_context("display.width", 220):
        print(I.pivot(index="feature", columns="frame", values="spearman").round(3).to_string())
        print(D.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
