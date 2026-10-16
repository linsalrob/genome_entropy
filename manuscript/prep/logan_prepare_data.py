#!/usr/bin/env python3
"""Reduce the final LOGAN analysis outputs to the plotting-ready data committed in manuscript/data/.

Runs on Setonix against the LOGAN run directory ($LOGAN_ROOT, default
/scratch/pawsey1018/edwa0468/Logan) and the LOGAN scripts merged in
genome_entropy #115 (Validation/logan/scripts on main). Not needed to redraw
the figures: the notebooks in manuscript/notebooks/ read only the files this
writes. Every output is small (tens of kB to ~1 MB, gzip TSV).

Sources (all final; genome_entropy #102, closed by #115):
  qc/analysis_final/             11_summarise_histograms.py   exact AUC/AP from all-ORF histograms
  qc/analysis_models_final/      13_models.py                 grouped-CV models, final thresholds
  qc/six_frame_final/            16_six_frame_summary.py      six-frame inheritance / discrimination / Δ
  qc/candidates/                 21-25, 29, 30                missed-gene arms, Foldseek evidence, taxonomy, skip
  qc/candidates_extra/           21b, 24, 31                  frameshift and low/high-score Prodigal arms
  analysis/v1/hist/*.npz         10_feature_histograms.py     feature histograms (summed here)
  analysis/v1/sample/*.parquet   12_sample_orfs.py            sampled-row counts per length bin

Writes manuscript/data/{figL*,tabL*}.tsv.gz and manuscript/data/logan_provenance.json
(source path, sha256 of each source and output).

  python manuscript/prep/logan_prepare_data.py --logan-root /scratch/pawsey1018/edwa0468/Logan \
      --scripts ~/GitHubs/genome_entropy/Validation/logan/scripts --out manuscript/data
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROV: dict = {"outputs": {}}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write(df: pd.DataFrame, out: Path, name: str, sources: list[Path], note: str) -> None:
    p = out / f"{name}.tsv.gz"
    # mtime=0: gzip output depends only on content, so reruns are byte-identical
    with open(p, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write(df.to_csv(sep="\t", index=False, lineterminator="\n").encode())
    PROV["outputs"][p.name] = {"rows": int(len(df)), "sha256": sha256(p), "note": note,
                               "sources": {str(s): sha256(s) for s in sources if s.is_file()}}
    print(f"{p.name}: {len(df):,} rows")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logan-root", default="/scratch/pawsey1018/edwa0468/Logan")
    ap.add_argument("--scripts", required=True, help="Validation/logan/scripts on genome_entropy main")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R, S, out = Path(args.logan_root), Path(args.scripts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(S))
    import os
    os.environ["LOGAN_ROOT"] = str(R)
    H = load_module(S / "10_feature_histograms.py", "h10")
    PROV["scripts_commit"] = subprocess.run(["git", "-C", str(S), "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip()
    PROV["logan_root"] = str(R)

    AF, MF, SF = R / "qc/analysis_final", R / "qc/analysis_models_final", R / "qc/six_frame_final"
    C, X = R / "qc/candidates", R / "qc/candidates_extra"

    # ---- L01: feature densities at fixed length, from the summed all-ORF histograms
    files = sorted(glob.glob(str(H.out_dir() / "*.npz")))
    tot = None
    for f in files:
        z = np.load(f)
        if tot is None:
            tot = {k: z[k].astype(np.int64) for k in H.FEATURES[1:]}
        else:
            for k in tot:
                tot[k] += z[k]
    shape = H.SHAPE + (-1,)
    centres = (np.arange(H.ENT_BINS) + 0.5) / 100
    rows = []
    classes = ["prodigal_complete", "antisense_overlap", "frameshift_overlap", "intergenic"]
    for lb, ltxt in ((3, "100-149"), (5, "200-299")):
        for feat in tot:
            A = tot[feat].reshape(shape)
            for cl in classes:
                h = A[H.LABELS.index(cl), 0, lb].sum(axis=(0, 1))
                for c, n in zip(centres, h):
                    if n:
                        rows.append((ltxt, feat, cl, round(float(c), 3), int(n)))
    write(pd.DataFrame(rows, columns=["length_bin_aa", "feature", "orf_class", "bin_centre_bits", "count"]),
          out, "figL01_feature_distributions", [Path(files[0]).parent / "."],
          f"complete ORFs; counts in 0.01-bit bins, summed over {len(files)} shard histograms (all ORFs)")
    PROV["outputs"]["figL01_feature_distributions.tsv.gz"]["histogram_files"] = len(files)

    # ---- L02-L04: models
    write(pd.read_csv(MF / "models_overall.tsv", sep="\t"), out, "figL02_models_overall",
          [MF / "models_overall.tsv"], "AUC / AP ± fold SD, nested feature sets, 4 contrasts")
    import pyarrow.parquet as pq
    sfiles = sorted(glob.glob(str(R / "analysis/v1/sample/*.parquet")))
    d = pd.concat([pq.read_table(f, columns=["label", "trunc", "aa_len"]).to_pandas() for f in sfiles])
    d = d[d.trunc == "complete"]
    d["length_bin"] = pd.cut(d.aa_len, [30, 50, 75, 100, 150, 200, 300, 500, 10**9], right=False).astype(str)
    neg = {"vs all Prodigal-": ["antisense_overlap", "frameshift_overlap", "intergenic", "marginal_overlap",
                                "same_frame_unmatched", "frame_undefined_overlap"],
           "vs intergenic": ["intergenic"]}
    cnt = pd.DataFrame([{"contrast": c, "length_bin": b, "sampled_pos": int((x.label == "prodigal_complete").sum()),
                         "sampled_neg": int(x.label.isin(ns).sum())}
                        for c, ns in neg.items() for b, x in d.groupby("length_bin")])
    bl = pd.read_csv(MF / "models_by_length.tsv", sep="\t").merge(cnt, on=["contrast", "length_bin"], how="left")
    write(bl, out, "figL03_models_by_length", [MF / "models_by_length.tsv"],
          f"within-bin AUC of out-of-fold scores; sampled_pos/neg counted from {len(sfiles)} sample shards")
    write(pd.read_csv(MF / "models_truncated.tsv", sep="\t"), out, "figL04_fragments",
          [MF / "models_truncated.tsv"], "complete-ORF model applied to contig-end fragments")

    # ---- L05: six-frame
    write(pd.read_csv(SF / "inheritance.tsv", sep="\t"), out, "figL05_six_frame_inheritance",
          [SF / "inheritance.tsv"], "Spearman rho, selected vs each alternative frame")
    write(pd.read_csv(SF / "discrimination.tsv", sep="\t"), out, "figL05_six_frame_discrimination",
          [SF / "discrimination.tsv"], "tie-aware top-k rates and chance")
    dl = pd.read_parquet(SF / "delta_sample.parquet")
    dl = dl[dl.k == 5]
    rows = []
    for f in ["aa_len", "aa_entropy", "three_di_entropy", "twelve_state_entropy", "mi_3di_12st"]:
        v = dl[f"delta_{f}"].dropna().to_numpy()
        lo, hi = np.quantile(v, [0.005, 0.995])
        n, edges = np.histogram(v[(v >= lo) & (v <= hi)], bins=120, range=(lo, hi))
        for a, b, k in zip(edges[:-1], edges[1:], n):
            rows.append((f, float(a), float(b), int(k), int(len(v)), float((v > 0).mean())))
    write(pd.DataFrame(rows, columns=["feature", "bin_left", "bin_right", "count", "genes", "fraction_positive"]),
          out, "figL05_six_frame_delta", [SF / "delta_sample.parquet"],
          "histogram of Δ (selected − median of 5 alternatives), central 99%, 120 bins; genes with all 5")

    # ---- L06-L07: missed-gene evidence (main + extra arms)
    ev = pd.concat([pd.read_csv(C / "evidence_rates.tsv", sep="\t"), pd.read_csv(X / "evidence_rates.tsv", sep="\t")])
    write(ev, out, "figL06_evidence_rates", [C / "evidence_rates.tsv", X / "evidence_rates.tsv"],
          "hit and full-length (qcov,tcov>=0.8) rates by arm, mode and database (five DBs)")
    write(pd.read_csv(C / "evidence_by_length.tsv", sep="\t"), out, "figL07_evidence_by_length",
          [C / "evidence_by_length.tsv"], "full-length structural rate by arm and length bin")
    write(pd.read_csv(C / "evidence_classes.tsv", sep="\t"), out, "figL07_evidence_classes",
          [C / "evidence_classes.tsv"], "share of queries in each full-length evidence class")
    write(pd.read_csv(C / "hit_biome.tsv", sep="\t"), out, "figL07_hit_biome", [C / "hit_biome.tsv"],
          "full-length structural hit rate and best-target superkingdom by arm and biome")

    # ---- L08: contig taxonomy
    write(pd.read_csv(C / "taxonomy/contig_group_by_arm.tsv", sep="\t"), out, "figL08_contig_group_by_arm",
          [C / "taxonomy/contig_group_by_arm.tsv"], "Kraken2 (PlusPFP) class of each query's contig, share by arm")
    write(pd.read_csv(C / "taxonomy/hit_vs_contig.tsv", sep="\t"), out, "figL08_hit_vs_contig",
          [C / "taxonomy/hit_vs_contig.tsv"], "queries with a full-length hit: best-target group x contig group")

    # ---- L09: why Prodigal skipped candidates (per query; groups as in 30_prodigal_skip.py)
    sk = pd.read_parquet(C / "skip/skip_features.parquet",
                         columns=["group", "reason", "CodPot", "Total", "orf_minus_model_gc"])
    sk = sk[sk.group != "other"].round(4)
    write(sk, out, "figL09_prodigal_skip", [C / "skip/skip_features.parquet"],
          "Prodigal -s best potential gene at each query stop; one row per query")

    # ---- L10: Prodigal+ / low score
    write(pd.read_csv(X / "lowscore/lowscore_population.tsv", sep="\t"), out, "figL10_lowscore_population",
          [X / "lowscore/lowscore_population.tsv"], "share of complete Prodigal genes below t0.80, by length")
    lbl = pd.read_csv(X / "lowscore/lowscore_evidence_by_length.tsv", sep="\t", header=[0, 1], index_col=0)
    lbl.columns = [f"{a}_{b}" for a, b in lbl.columns]
    write(lbl.reset_index().rename(columns={"index": "length_bin"}), out, "figL10_lowscore_evidence_by_length",
          [X / "lowscore/lowscore_evidence_by_length.tsv"], "low- vs high-score Prodigal genes, length-matched")
    write(pd.read_csv(X / "lowscore/all_arms_evidence.tsv", sep="\t"), out, "figL10_all_arms_evidence",
          [X / "lowscore/all_arms_evidence.tsv"], "hit / full-length rates of every arm over all five databases")

    # ---- LS1: length-controlled separation by biome and tranche
    bb = pd.read_csv(AF / "biome_batch_separation.tsv", sep="\t")
    write(bb[bb.contrast == "vs all Prodigal-"], out, "figLS1_biome_batch_separation",
          [AF / "biome_batch_separation.tsv"], "exact AUC by biome / tranche and length bin (15b_biome_batch_figure.py)")

    # ---- Tables
    for name, src, note in (
            ("tabL01_orf_classes_by_truncation", AF / "class_by_truncation.tsv", "ORF classes x truncation, all 89.99 B ORFs"),
            ("tabL02_single_feature_overall", AF / "separation_overall.tsv", "exact single-feature AUC/AP"),
            ("tabL03_single_feature_by_length", AF / "separation_by_length.tsv", "within length bins"),
            ("tabL04_final_model_thresholds", None, "out-of-fold score thresholds by target precision"),
            ("tabL05_candidate_counts", C / "candidate_counts.tsv", "intergenic ORFs >=100 aa above each threshold"),
            ("tabL06_evidence_by_truncation", X / "evidence_by_truncation.tsv", "full-length evidence by arm and truncation"),
            ("tabL07_lowscore_features", X / "lowscore/lowscore_features.tsv", "median features, low vs high score"),
            ("tabL08_skip_reasons", C / "skip/skip_reasons.tsv", "why Prodigal did not call each group"),
            ("tabL09_euk_explained", C / "taxonomy/euk_explained.tsv", "eukaryotic best hits by contig class")):
        if src is None:
            t = json.load(open(MF / "final_model_thresholds.json"))["precision_thresholds"]
            write(pd.DataFrame({"precision_target": list(t), "score_threshold": list(t.values())}), out, name,
                  [MF / "final_model_thresholds.json"], note)
        else:
            hdr = [0, 1] if name == "tabL07_lowscore_features" else 0
            df = pd.read_csv(src, sep="\t", header=hdr)
            if name == "tabL07_lowscore_features":
                df.columns = ["length_bin"] + [f"{a}_{b}" for a, b in df.columns[1:]]
            write(df, out, name, [src], note)

    json.dump(PROV, open(out / "logan_provenance.json", "w"), indent=1)
    print("provenance:", out / "logan_provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
