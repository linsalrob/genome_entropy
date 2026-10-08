#!/usr/bin/env python3
"""Pilot acceptance checks (#102 §9), first Prodigal+/- comparison, and cost projections.

Reads only what the pipeline wrote: accession states, chunk manifests, the
cache chunks, and each accession's orf_prodigal.tsv.zst. Writes, under
$LOGAN_ROOT/qc/<batch>/:

  acceptance.tsv         one row per §9 check: pass/fail and the evidence
  per_accession.tsv      sizes, ORF density, Prodigal+ fraction, classes,
                         truncation, code-4 contigs
  features.tsv.zst       per-ORF cheap features recomputed from the cache
                         (AA / 3Di / 12st entropy, 3Di-12st MI, length) joined
                         to the Prodigal label: the input to later analysis
  separation.tsv         ROC AUC and average precision of each feature for
                         Prodigal+ vs Prodigal-, overall and by biome
  inspect_<class>.tsv    a small random sample per class for manual review
  projection.tsv         compute, download, storage and cache growth for
                         5k, 10k and every candidate metagenome, at both the
                         30 aa and 90 aa ORF floors

  05_pilot_qc.py --batch pilot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

from genome_entropy.entropy.shannon import calculate_sequence_entropy, mutual_information  # noqa: E402

FEATURES = ("aa_len", "aa_entropy", "three_di_entropy", "twelve_state_entropy", "mi_3di_12st")


def load_states(batch: str) -> pd.DataFrame:
    rows = []
    for p in sorted((lc.root() / "state").glob("*.json")):
        st = json.load(open(p))
        if st.get("info", {}).get("batch") != batch:
            continue
        rows.append({"accession": st["accession"], "state": st["state"], **{
            k: v for k, v in st["info"].items() if not isinstance(v, (dict, list))}})
    return pd.DataFrame(rows)


def load_chunks() -> list:
    return [json.load(open(p)) for p in sorted((lc.root() / "manifests" / "chunks").glob("*.json"))]


def _chunk_features(args) -> pd.DataFrame:
    path, accs = args
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     compression={"method": "zstd"})
    df = df[df.accession.isin(accs)]
    return pd.DataFrame({
        "accession": df.accession.to_numpy(), "orf_id": df.orf_id.to_numpy(),
        "aa_len": df.aa_sequence.str.len().to_numpy(),
        "len_ok": ((df.aa_sequence.str.len() == df.three_di.str.len())
                   & (df.aa_sequence.str.len() == df.twelve_state.str.len())).to_numpy(),
        "aa_entropy": [calculate_sequence_entropy(x) for x in df.aa_sequence],
        "three_di_entropy": [calculate_sequence_entropy(x) for x in df.three_di],
        "twelve_state_entropy": [calculate_sequence_entropy(x) for x in df.twelve_state],
        "mi_3di_12st": [mutual_information(a, b) for a, b in zip(df.three_di, df.twelve_state)],
    })


def features_from_cache(chunks: list, accs: set, procs: int) -> pd.DataFrame:
    """Per-ORF features recomputed from the cache, one chunk per process."""
    import multiprocessing as mp
    jobs = [(lc.root() / "cache" / "modernprost-50M" / lc.CACHE_VERSION / m["cache_file"], accs)
            for m in chunks if set(m.get("accessions", [])) & accs]
    with mp.get_context("spawn").Pool(procs) as pool:
        frames = pool.map(_chunk_features, jobs, chunksize=1)
    return pd.concat(frames, ignore_index=True)


def auc_ap(y: np.ndarray, x: np.ndarray):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ok = np.isfinite(x)
    if ok.sum() < 10 or len(set(y[ok])) < 2:
        return np.nan, np.nan
    return roc_auc_score(y[ok], x[ok]), average_precision_score(y[ok], x[ok])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", default="pilot")
    ap.add_argument("--inspect-n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    out = lc.root() / "qc" / args.batch
    out.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(lc.root() / "manifests" / f"{args.batch}_accessions.tsv", sep="\t")
    st = load_states(args.batch)
    chunks = load_chunks()
    acc_done = set(st.loc[st.state.isin(["cache_validated", "uploaded", "remote_verified",
                                         "complete"]), "accession"])
    checks = []

    def check(name, ok, evidence):
        checks.append({"check": name, "pass": bool(ok), "evidence": evidence})

    check("all selected accessions are METAGENOMIC WGS random-shotgun",
          (sel.librarysource.eq("METAGENOMIC") & sel.assay_type.eq("WGS")).all(),
          f"{len(sel)} selected; libraryselection {sel.libraryselection.value_counts().to_dict()}")
    check("all selected accessions reached cache_validated or later",
          len(acc_done) == len(sel), f"{len(acc_done)}/{len(sel)}; states {st.state.value_counts().to_dict()}")
    if "contig_bp_rel_diff" in st:
        worst = st[["contig_count_rel_diff", "contig_bp_rel_diff"]].abs().max()
        check("LOGAN contig counts/bp equal seqstats (<=0.1%)",
              (worst.fillna(0) <= 1e-3).all(), worst.to_dict())
    if "prodigal_reproducible" in st:
        check("Prodigal byte-reproducible on every accession",
              st.prodigal_reproducible.fillna(False).all(),
              f"{int(st.prodigal_reproducible.fillna(False).sum())}/{len(st)}")
    check("no get_orfs/translate mismatches",
          (st.get("translation_mismatches", pd.Series([0])).fillna(0) == 0).all(), "")

    # Cache row counts vs manifests vs accession ORF counts.
    bad = [m["chunk_id"] for m in chunks if m.get("orf_count") is not None
           and m["orf_count"] != m["orf_count_expected"]]
    check("chunk row counts equal manifest counts", not bad, f"mismatched: {bad}")
    per_acc = {}
    for m in chunks:
        for x in m["members"]:
            per_acc[x["accession"]] = per_acc.get(x["accession"], 0) + x["n_rows"]
    bad = [a for a in acc_done if per_acc.get(a) != int(st.set_index("accession").at[a, "orfs"])]
    check("every accession's ORFs appear in chunks exactly once", not bad, f"mismatched: {bad[:10]}")
    mine = [m for m in chunks if {x["accession"] for x in m["members"]} & acc_done]
    check("every chunk holding this batch is remote_verified",
          all(m["state"] == "remote_verified" for m in mine),
          {m["chunk_id"]: m["state"] for m in mine if m["state"] != "remote_verified"})

    # Features and labels.
    feats = features_from_cache(chunks, acc_done, args.procs)
    check("no AA/3Di/12st length mismatches", bool(feats.len_ok.all()),
          f"{int((~feats.len_ok).sum())} bad of {len(feats):,}")
    labels = []
    for a in sorted(acc_done):
        t = pd.read_csv(lc.work_dir(a) / "orf_prodigal.tsv.zst", sep="\t", keep_default_na=False,
                        usecols=["accession", "orf_id", "contig", "start", "end", "strand",
                                 "contig_len", "stop", "five_open", "prodigal", "prodigal_class",
                                 "best_overlap_bp", "best_gene_transl_table",
                                 "initiator_substituted"])
        labels.append(t)
    lab = pd.concat(labels, ignore_index=True)
    df = lab.merge(feats.drop(columns="len_ok"), on=["accession", "orf_id"], how="inner",
                   validate="one_to_one")
    check("every labelled ORF has a cache row (and vice versa)",
          len(df) == len(lab) == len(feats), f"labels {len(lab):,} cache {len(feats):,} joined {len(df):,}")
    biome = sel.set_index("accession").biome
    df["biome"] = df.accession.map(biome)
    df["truncated"] = (df.stop.astype(int) == 0) | (df.five_open.astype(int) == 1)
    df.to_csv(out / "features.tsv.zst", sep="\t", index=False, compression={"method": "zstd"})

    # Separation: complete (non-truncated) ORFs, Prodigal+ vs everything else,
    # and the sharper Prodigal+ vs intergenic contrast.
    sep = []
    full = df[~df.truncated]
    for scope, d in [("all", full)] + [(b, g) for b, g in full.groupby("biome")]:
        y = (d.prodigal == "Prodigal+").to_numpy().astype(int)
        for f in FEATURES:
            auc, apv = auc_ap(y, d[f].to_numpy(float))
            sep.append({"scope": scope, "contrast": "Prodigal+ vs Prodigal-", "feature": f,
                        "n": len(d), "pos": int(y.sum()), "roc_auc": auc, "avg_precision": apv})
        dd = d[d.prodigal_class.isin(["prodigal_match", "intergenic"])]
        y2 = (dd.prodigal == "Prodigal+").to_numpy().astype(int)
        for f in FEATURES:
            auc, apv = auc_ap(y2, dd[f].to_numpy(float))
            sep.append({"scope": scope, "contrast": "Prodigal+ vs intergenic", "feature": f,
                        "n": len(dd), "pos": int(y2.sum()), "roc_auc": auc, "avg_precision": apv})
    pd.DataFrame(sep).to_csv(out / "separation.tsv", sep="\t", index=False)

    rng = np.random.default_rng(args.seed)
    for cls, g in df.groupby("prodigal_class"):
        g.sample(n=min(args.inspect_n, len(g)), random_state=rng.integers(1 << 31)).to_csv(
            out / f"inspect_{cls}.tsv", sep="\t", index=False)

    # Per-accession summary.
    pa = df.groupby("accession").agg(
        orfs=("orf_id", "size"), aa=("aa_len", "sum"),
        prodigal_plus=("prodigal", lambda s: (s == "Prodigal+").mean()),
        truncated=("truncated", "mean"),
        code4_rows=("best_gene_transl_table", lambda s: (s.astype(int) == 4).mean()),
        orfs_ge90=("aa_len", lambda s: (s >= 90).sum()),
        aa_ge90=("aa_len", lambda s: s[s >= 90].sum()))
    pa = pa.join(st.set_index("accession")[[c for c in ("contig_bp", "bp_ge120", "contig_count",
                                                       "prodigal_genes", "download_bytes",
                                                       "prodigal_seconds", "orf_and_match_seconds")
                                            if c in st]])
    pa = pa.join(sel.set_index("accession")[["biome", "n50"]])
    for c, d in df.groupby(["accession", "prodigal_class"]).size().unstack(fill_value=0).items():
        pa[f"class_{c}"] = d
    pa.to_csv(out / "per_accession.tsv", sep="\t")

    # Cost model, all linear in contig bp: fitted on the pilot as ratios of
    # sums (robust to the size spread), then applied to the candidate pool.
    tot_bp = pa.contig_bp.sum()
    aa_per_bp = pa.aa.sum() / tot_bp
    aa90_per_bp = pa.aa_ge90.sum() / tot_bp
    orf_per_bp = pa.orfs.sum() / tot_bp
    orf90_per_bp = pa.orfs_ge90.sum() / tot_bp
    dl_per_bp = pa.download_bytes.sum() / tot_bp if "download_bytes" in pa else np.nan
    enc = [m for m in chunks if m.get("aa_per_second")]
    aa_per_gcd_s = (sum(m["aa_count"] for m in enc) / sum(m["encode_seconds"] for m in enc)) if enc else np.nan
    cache_bytes_per_aa = (sum(m["bytes"] for m in enc) / sum(m["aa_count"] for m in enc)) if enc else np.nan
    cpu_s_per_bp = (pa.prodigal_seconds.sum() + pa.orf_and_match_seconds.sum()) / tot_bp \
        if "prodigal_seconds" in pa else np.nan

    cand = pd.read_parquet(lc.root() / "meta" / "logan_metagenomic_candidates.parquet",
                           columns=["accession", "assay_type", "libraryselection", "platform",
                                    "organism", "contig_bp", "n50", "biome"])
    pool = cand[(cand.assay_type == "WGS") & cand.contig_bp.notna() & (cand.contig_bp > 0)]
    scen = []
    med = pool.contig_bp.median()
    for name, bp in [("5k (median-size draw)", 5000 * med), ("10k (median-size draw)", 10000 * med),
                     ("5k (pilot-size draw)", 5000 * tot_bp / len(pa)),
                     ("10k (pilot-size draw)", 10000 * tot_bp / len(pa)),
                     (f"all WGS metagenomes ({len(pool):,})", pool.contig_bp.sum())]:
        for floor, apb, opb in (("30 aa", aa_per_bp, orf_per_bp), ("90 aa", aa90_per_bp, orf90_per_bp)):
            aa = bp * apb
            scen.append({"scenario": name, "orf_floor": floor, "contig_Gbp": bp / 1e9,
                         "orfs_billion": bp * opb / 1e9, "aa_billion": aa / 1e9,
                         "gcd_hours": aa / aa_per_gcd_s / 3600 if aa_per_gcd_s else np.nan,
                         "cpu_core_hours": bp * cpu_s_per_bp / 3600,
                         "download_TB": bp * dl_per_bp / 1e12,
                         "cache_TB": aa * cache_bytes_per_aa / 1e12,
                         "chunks_1M": math.ceil(bp * opb / 1e6)})
    proj = pd.DataFrame(scen)
    proj.to_csv(out / "projection.tsv", sep="\t", index=False)
    rates = {"aa_per_bp_30": aa_per_bp, "aa_per_bp_90": aa90_per_bp,
             "orfs_per_Mbp_30": orf_per_bp * 1e6, "orfs_per_Mbp_90": orf90_per_bp * 1e6,
             "aa_per_gcd_second": aa_per_gcd_s, "cache_bytes_per_aa": cache_bytes_per_aa,
             "download_bytes_per_bp": dl_per_bp, "cpu_seconds_per_Mbp": cpu_s_per_bp * 1e6}
    json.dump(rates, open(out / "rates.json", "w"), indent=1)

    acc = pd.DataFrame(checks)
    acc.to_csv(out / "acceptance.tsv", sep="\t", index=False)
    print(acc.to_string(index=False, max_colwidth=90))
    print(json.dumps(rates, indent=1))
    print(proj.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(pd.DataFrame(sep).query("scope == 'all'").to_string(index=False))
    return 0 if acc["pass"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
