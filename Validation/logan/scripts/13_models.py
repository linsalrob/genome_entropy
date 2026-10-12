#!/usr/bin/env python3
"""Do 3Di / 12-state / MI add information beyond ORF length? Grouped-CV models on the weighted sample.

Reads analysis/v1/sample/*.parquet (12_sample_orfs.py). For complete ORFs
and each contrast (Prodigal+ complete gene vs all Prodigal-, vs intergenic,
vs antisense shadows, vs frameshift shadows) it fits a gradient-boosted
classifier (HistGradientBoostingClassifier) on nested feature sets:

  length                         log(aa_len) only
  length+aa                      + AA entropy
  length+3di                     + 3Di entropy
  length+12st                    + 12-state entropy
  length+mi                      + 3Di-12st MI
  length+3di+12st+mi             + all structural
  length+aa+3di+12st+mi          + everything

with 5-fold cross-validation GROUPED BY ACCESSION (no metagenome in both
train and test). Models are trained unweighted on the class-stratified
sample; out-of-fold scores are evaluated with the population weights
(1 / sampling rate), so AUC and AP describe the real class mix. Fold-to-fold
SD is reported. Out-of-fold scores are also scored within length bins.

Finally the complete-ORF model with all features is applied to contig-end
truncated ORFs (Prodigal+ partial vs truncated Prodigal-) to see how far the
signal carries to fragments.

  13_models.py --out qc/analysis_models
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

FEATS = {"aa": "aa_entropy", "3di": "three_di_entropy", "12st": "twelve_state_entropy",
         "mi": "mi_3di_12st"}
SETS = ["length", "length+aa", "length+3di", "length+12st", "length+mi",
        "length+3di+12st+mi", "length+aa+3di+12st+mi"]
CONTRASTS = {
    "vs all Prodigal-": ["antisense_overlap", "frameshift_overlap", "intergenic", "marginal_overlap",
                         "same_frame_unmatched"],
    "vs intergenic": ["intergenic"],
    "vs antisense shadows": ["antisense_overlap"],
    "vs frameshift shadows": ["frameshift_overlap"],
}
LEN_EDGES = [30, 50, 75, 100, 150, 200, 300, 500, 10**9]


def load() -> pd.DataFrame:
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(lc.root() / "analysis" / lc.CACHE_VERSION / "sample" / "*.parquet")))
    d = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    d["log_len"] = np.log(d.aa_len.astype(float))
    for k, v in FEATS.items():
        d[v] = d[v + "_q4"] / 1e4
    print(f"{len(files)} shard samples, {len(d):,} rows")
    return d, len(files)


def cols(s: str):
    return ["log_len"] + [FEATS[p] for p in s.split("+")[1:]]


def fit_eval(d: pd.DataFrame, y: np.ndarray, w: np.ndarray, groups: np.ndarray, threads: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    rows, oof_all = [], {}
    folds = list(GroupKFold(n_splits=5).split(d, y, groups))
    for s in SETS:
        X = d[cols(s)].to_numpy()
        oof = np.zeros(len(y))
        per = []
        for tr, te in folds:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                               early_stopping=True, random_state=0)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
            per.append((roc_auc_score(y[te], oof[te], sample_weight=w[te]),
                        average_precision_score(y[te], oof[te], sample_weight=w[te])))
        per = np.array(per)
        rows.append({"features": s, "roc_auc": roc_auc_score(y, oof, sample_weight=w),
                     "auc_fold_sd": per[:, 0].std(ddof=1),
                     "avg_precision": average_precision_score(y, oof, sample_weight=w),
                     "ap_fold_sd": per[:, 1].std(ddof=1)})
        oof_all[s] = oof
        print(f"  {s:24s} AUC {rows[-1]['roc_auc']:.4f}±{rows[-1]['auc_fold_sd']:.4f}  "
              f"AP {rows[-1]['avg_precision']:.4f}±{rows[-1]['ap_fold_sd']:.4f}", flush=True)
    return pd.DataFrame(rows), oof_all


def fit_eval_one(d: pd.DataFrame, y: np.ndarray, c: list):
    """Out-of-fold scores (grouped 5-fold) for one feature set."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    X = d[c].to_numpy()
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, d.accession.to_numpy()):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                           early_stopping=True, random_state=0)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return None, oof


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d, n_shards = load()
    comp = d[d.trunc == "complete"]

    overall, bylen = [], []
    for cname, negs in CONTRASTS.items():
        sub = comp[comp.label.isin(["prodigal_complete"] + negs)].reset_index(drop=True)
        y = (sub.label == "prodigal_complete").to_numpy().astype(int)
        w = sub.weight.to_numpy()
        print(f"{cname}: {y.sum():,} positives / {len(y) - y.sum():,} negatives sampled "
              f"(population {w[y == 1].sum():,.0f} / {w[y == 0].sum():,.0f})", flush=True)
        r, oof = fit_eval(sub, y, w, sub.accession.to_numpy(), args.threads)
        r.insert(0, "contrast", cname)
        r["positives_sampled"], r["negatives_sampled"] = int(y.sum()), int(len(y) - y.sum())
        overall.append(r)
        lb = pd.cut(sub.aa_len, LEN_EDGES, right=False)
        for s, sc in oof.items():
            for b, idx in sub.groupby(lb, observed=True).groups.items():
                idx = np.asarray(idx)
                if len(set(y[idx])) < 2:
                    continue
                bylen.append({"contrast": cname, "features": s, "length_bin": str(b),
                              "n": len(idx), "roc_auc": roc_auc_score(y[idx], sc[idx], sample_weight=w[idx]),
                              "avg_precision": average_precision_score(y[idx], sc[idx], sample_weight=w[idx])})
    R = pd.concat(overall, ignore_index=True)
    base = R[R.features == "length"].set_index("contrast")
    R["delta_auc_vs_length"] = R.roc_auc - R.contrast.map(base.roc_auc)
    R["delta_ap_vs_length"] = R.avg_precision - R.contrast.map(base.avg_precision)
    R.to_csv(out / "models_overall.tsv", sep="\t", index=False)
    pd.DataFrame(bylen).to_csv(out / "models_by_length.tsv", sep="\t", index=False)

    # Truncated ORFs: train on complete (all features), score fragments.
    from sklearn.ensemble import HistGradientBoostingClassifier
    negs = CONTRASTS["vs all Prodigal-"]
    trc = comp[comp.label.isin(["prodigal_complete"] + negs)]
    m = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63, random_state=0)
    m.fit(trc[cols("length+aa+3di+12st+mi")].to_numpy(), (trc.label == "prodigal_complete").astype(int))
    mlen = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63, random_state=0)
    mlen.fit(trc[["log_len"]].to_numpy(), (trc.label == "prodigal_complete").astype(int))
    trunc_rows = []
    for tr in ("open5", "open3"):
        t = d[(d.trunc == tr) & d.label.isin(["prodigal_partial"] + negs)]
        yt = (t.label == "prodigal_partial").astype(int).to_numpy()
        for name, model, c in (("length", mlen, ["log_len"]),
                               ("length+aa+3di+12st+mi", m, cols("length+aa+3di+12st+mi"))):
            sc = model.predict_proba(t[c].to_numpy())[:, 1]
            trunc_rows.append({"truncation": tr, "features": name, "positives": int(yt.sum()),
                               "negatives": int(len(yt) - yt.sum()),
                               "roc_auc": roc_auc_score(yt, sc, sample_weight=t.weight),
                               "avg_precision": average_precision_score(yt, sc, sample_weight=t.weight)})
    pd.DataFrame(trunc_rows).to_csv(out / "models_truncated.tsv", sep="\t", index=False)

    # The final scorer for the missed-gene scan (18_score_candidates.py):
    # trained on ALL complete-ORF sample rows, Prodigal+ complete vs all
    # Prodigal-. Score thresholds are set by population-weighted precision on
    # out-of-fold scores of the same contrast, saved alongside.
    import joblib
    joblib.dump({"model": m, "features": cols("length+aa+3di+12st+mi"), "length_model": mlen},
                out / "final_model.joblib")
    sub = comp[comp.label.isin(["prodigal_complete"] + negs)].reset_index(drop=True)
    y = (sub.label == "prodigal_complete").to_numpy().astype(int)
    _, oof = fit_eval_one(sub, y, cols("length+aa+3di+12st+mi"))
    w = sub.weight.to_numpy()
    o = np.argsort(-oof)
    tp, fp = np.cumsum(w[o] * y[o]), np.cumsum(w[o] * (1 - y[o]))
    prec = tp / (tp + fp)
    thr = {}
    for target in (0.80, 0.90, 0.95, 0.99):
        ok = np.nonzero(prec >= target)[0]
        thr[str(target)] = float(oof[o][ok.max()]) if len(ok) else None
    json.dump({"precision_thresholds": thr,
               "note": "score thresholds at which population-weighted precision of Prodigal+ "
                       "among complete ORFs reaches the target (out-of-fold, grouped CV)"},
              open(out / "final_model_thresholds.json", "w"), indent=1)
    json.dump({"sample_shards": n_shards, "sample_rows": len(d)}, open(out / "inputs.json", "w"), indent=1)
    with pd.option_context("display.width", 220):
        print(R.round(4).to_string(index=False))
        print(pd.DataFrame(trunc_rows).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
