#!/usr/bin/env python3
"""Grouped-CV coding classifiers on the Inphared ORFs: how much of the signal is coordinates, and how much is beyond length.

Reads orfs.parquet (01_build_orf_table.py) and, for comparison, the original
run's all_orf_oof_predictions.parquet (rank_missing_orfs.py, job 46481896,
features incl. absolute start/end). Fits the SAME model as
scripts/rank_missing_orfs.py (XGBRFClassifier: 500 trees, depth 8,
subsample/colsample_bynode 0.8, seed 42 + fold) with the SAME split
(GroupKFold(5) on input_id = one genome) for each feature set:

  orig14            the original 14 features (reproduces the August run)
  no_coord          orig14 minus start, end
  intrinsic         entropies (DNA, AA, 3Di, 12-state) + lengths only
  length_only       protein_length
  length_3di        protein_length + 3Di entropy
  length_entropies  protein_length + the four entropies
  entropy_only      the four entropies, no length

Outputs (in --out):
  oof.parquet               row, input_id, fold, label, p_<set> for every set
  metrics.tsv               AUC, AP, Brier per set, per fold and pooled
  metrics_by_length.tsv     pooled AUC / AP within protein-length bins
  calibration.tsv           20 equal-width probability bins per set
  importance.tsv            XGBoost gain importance per set and fold
  thresholds.tsv            OOF score at precision 0.80/0.90/0.95 per set
  fold_check.json           agreement of folds / probabilities with the original

--groups TSV (columns input_id, group) replaces the one-genome-per-group split
with a coarser one (a genus, or an ANI cluster): every genome of a group is
held out together, so near-duplicate genomes never sit on both sides of a fold.
Genomes absent from the TSV keep their own group.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRFClassifier

ORIG14 = ["dna_entropy", "protein_entropy", "three_di_entropy", "dna_length", "protein_length",
          "three_di_length", "start", "end", "strand_plus", "frame", "has_start_codon", "has_stop_codon",
          "twelve_state_entropy", "twelve_state_length"]
ENT = ["dna_entropy", "protein_entropy", "three_di_entropy", "twelve_state_entropy"]
LEN = ["dna_length", "protein_length", "three_di_length", "twelve_state_length"]
SETS = {
    "orig14": ORIG14,
    "no_coord": [f for f in ORIG14 if f not in ("start", "end")],
    "intrinsic": ENT + LEN,
    "length_only": ["protein_length"],
    "length_3di": ["protein_length", "three_di_entropy"],
    "length_entropies": ["protein_length"] + ENT,
    "entropy_only": ENT,
}
LEN_EDGES = [90, 100, 150, 200, 300, 500, 1000, 10**9]


def model(seed: int, threads: int) -> XGBRFClassifier:
    return XGBRFClassifier(n_estimators=500, max_depth=8, subsample=0.80, colsample_bynode=0.80,
                           objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                           device="cpu", random_state=seed, n_jobs=threads)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orfs", required=True, type=Path)
    ap.add_argument("--original", type=Path, help="original all_orf_oof_predictions.parquet")
    ap.add_argument("--labels", type=Path,
                    help="labels.parquet from 02_relabel.py: train and score on its recomputed in_genbank")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--groups", type=Path, help="TSV input_id<TAB>group: hold out whole groups, not genomes")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cols = sorted(set(sum(SETS.values(), [])) - {"strand_plus"}) + ["row", "input_id", "orf_id", "strand",
                                                                     "in_genbank"]
    d = pd.read_parquet(a.orfs, columns=cols).sort_values("row").reset_index(drop=True)
    d["in_genbank_json"] = d.in_genbank
    if a.labels:
        lab = pd.read_parquet(a.labels, columns=["row", "orf_id", "in_genbank"]).sort_values("row")
        assert (lab.row.to_numpy() == d.row.to_numpy()).all() and (lab.orf_id.to_numpy() == d.orf_id.to_numpy()).all()
        d["in_genbank"] = lab.in_genbank.to_numpy()
        print(f"labels from {a.labels}: {int(d.in_genbank.sum()):,} annotated "
              f"(JSON: {int(d.in_genbank_json.sum()):,})", flush=True)
    d["strand_plus"] = (d.strand == "+").astype(float)
    for c in ("has_start_codon", "has_stop_codon"):
        d[c] = d[c].astype(float)
    y = d.in_genbank.to_numpy().astype(int)
    groups = d.input_id.to_numpy()
    print(f"{len(d):,} ORFs, {y.sum():,} annotated, {len(np.unique(groups)):,} genomes", flush=True)
    if a.groups:
        gmap = pd.read_csv(a.groups, sep="\t", dtype=str).set_index("input_id").group
        g = d.input_id.map(gmap)
        print(f"groups from {a.groups}: {g.notna().mean():.4f} of ORFs mapped; "
              f"{g.nunique():,} groups (+{d.input_id[g.isna()].nunique():,} unmapped genomes kept alone)", flush=True)
        groups = g.fillna("genome:" + d.input_id).to_numpy()

    folds = np.full(len(d), -1)
    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(d)), y, groups))
    for k, (_, te) in enumerate(splits, start=1):
        folds[te] = k

    check = {}
    orig = None
    if a.original and a.original.exists():
        orig = pd.read_parquet(a.original, columns=["input_id", "orf_id", "fold", "probability_in_genbank",
                                                    "in_genbank"])
        same_order = bool((orig.input_id.to_numpy() == d.input_id.to_numpy()).all()
                          and (orig.orf_id.to_numpy() == d.orf_id.to_numpy()).all())
        check["same_row_order"] = same_order
        check["same_labels"] = bool((orig.in_genbank.to_numpy() == d.in_genbank.to_numpy()).all())
        check["labels_differ"] = int((orig.in_genbank.to_numpy() != d.in_genbank.to_numpy()).sum())
        check["same_folds"] = bool((orig.fold.to_numpy() == folds).all())
        print(check, flush=True)

    out = pd.DataFrame({"row": d.row, "input_id": d.input_id, "fold": folds, "label": y,
                        "label_json": d.in_genbank_json.to_numpy().astype(int)})
    if orig is not None and check.get("same_row_order"):
        out["p_orig14_stored"] = orig.probability_in_genbank.to_numpy()
    imp_rows = []
    for name in a.sets.split(","):
        feats = SETS[name]
        X = d[feats].to_numpy(dtype=np.float32)
        p = np.full(len(d), np.nan)
        t0 = time.time()
        for k, (tr, te) in enumerate(splits, start=1):
            m = model(a.seed + k, a.threads)
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
            gain = m.get_booster().get_score(importance_type="gain")
            tot = sum(gain.values()) or 1.0
            for i, f in enumerate(feats):
                imp_rows.append({"set": name, "fold": k, "feature": f, "gain": gain.get(f"f{i}", 0.0) / tot})
        out[f"p_{name}"] = p
        print(f"{name}: AUC {roc_auc_score(y, p):.4f} AP {average_precision_score(y, p):.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    out.to_parquet(a.out / "oof.parquet", index=False)
    pd.DataFrame(imp_rows).to_csv(a.out / "importance.tsv", sep="\t", index=False)

    pcols = [c for c in out.columns if c.startswith("p_")]
    met, bylen, cal, thr = [], [], [], []
    lb = pd.cut(d.protein_length, LEN_EDGES, right=False).astype(str)
    for c in pcols:
        name = c[2:]
        p = out[c].to_numpy()
        for k in ["pooled", 1, 2, 3, 4, 5]:
            msk = np.ones(len(p), bool) if k == "pooled" else folds == k
            met.append({"set": name, "fold": k, "n": int(msk.sum()), "positives": int(y[msk].sum()),
                        "auc": roc_auc_score(y[msk], p[msk]), "ap": average_precision_score(y[msk], p[msk]),
                        "brier": brier_score_loss(y[msk], p[msk])})
        for b, idx in pd.Series(np.arange(len(p))).groupby(lb.to_numpy()):
            yy, pp = y[idx.to_numpy()], p[idx.to_numpy()]
            ok = 0 < yy.sum() < len(yy)
            bylen.append({"set": name, "length_bin": b, "n": len(yy), "prevalence": yy.mean(),
                          "auc": roc_auc_score(yy, pp) if ok else np.nan,
                          "ap": average_precision_score(yy, pp) if ok else np.nan})
        bins = np.clip((p * 20).astype(int), 0, 19)
        g = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"),
                                                                           frac_pos=("y", "mean")).reset_index()
        g.insert(0, "set", name)
        cal.append(g)
        prec, rec, t = precision_recall_curve(y, p)
        for target in (0.8, 0.9, 0.95):
            i = np.where(prec[:-1] >= target)[0]
            thr.append({"set": name, "precision_target": target,
                        "threshold": float(t[i[0]]) if len(i) else np.nan,
                        "recall": float(rec[i[0]]) if len(i) else np.nan,
                        "unannotated_above": int(((p >= t[i[0]]) & (y == 0)).sum()) if len(i) else 0})
        thr.append({"set": name, "precision_target": "p>=0.8", "threshold": 0.8,
                    "recall": float(((p >= 0.8) & (y == 1)).sum() / y.sum()),
                    "unannotated_above": int(((p >= 0.8) & (y == 0)).sum())})
    pd.DataFrame(met).to_csv(a.out / "metrics.tsv", sep="\t", index=False)
    pd.DataFrame(bylen).to_csv(a.out / "metrics_by_length.tsv", sep="\t", index=False)
    pd.concat(cal).to_csv(a.out / "calibration.tsv", sep="\t", index=False)
    pd.DataFrame(thr).to_csv(a.out / "thresholds.tsv", sep="\t", index=False)
    if "p_orig14_stored" in out and "p_orig14" in out:
        diff = np.abs(out.p_orig14_stored - out.p_orig14)
        check["orig14_refit_vs_stored_max_abs_diff"] = float(diff.max())
        check["orig14_refit_vs_stored_mean_abs_diff"] = float(diff.mean())
        check["orig14_refit_vs_stored_candidates_jaccard"] = float(
            (((out.p_orig14 >= .8) & (out.p_orig14_stored >= .8)) & (y == 0)).sum()
            / (((out.p_orig14 >= .8) | (out.p_orig14_stored >= .8)) & (y == 0)).sum())
    json.dump(check, open(a.out / "fold_check.json", "w"), indent=1)
    print(pd.DataFrame(met).query("fold == 'pooled'").round(4).to_string(index=False))
    print(pd.DataFrame(thr).to_string(index=False))
    print(check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
