#!/usr/bin/env python3
"""Measured size and cost of a finished tranche, and projections for the next one.

Everything measured comes from what the pipeline recorded, not from pilot
extrapolation:

  * accession state files: contig bp, ORFs, aa, Prodigal genes, download
    bytes, per-accession CPU seconds, unavailable accessions;
  * chunk manifests: aa encoded, encode seconds, cache bytes;
  * Slurm accounting (`sacct`): billed SU per job, as billing x elapsed,
    split into CPU stage, GPU encoding, upload and orchestration. The job ids
    come from the orchestrator record plus any --extra-jobs (e.g. manual
    leftover reruns).

Projections apply the measured per-bp rates to the *remaining* candidate pool
(accessions not yet selected by any batch), under the pilot's filters and
several per-BioProject caps, because the remaining pool's size distribution
differs from what was drawn: the stratified draw takes small assemblies
first, so what is left is larger on average.

  08_tranche_costs.py --batch phaseb1 --extra-jobs 50589946 --out qc/phaseb1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

NON_NATURAL = r"synthetic|mock|mixed culture|bioreactor metagenome$|laboratory"


def billed_su(job_ids):
    """Sum of billing x elapsed over every job/array element, in SU."""
    if not job_ids:
        return 0.0, 0.0
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
                          "--format=JobID,ElapsedRaw,AllocTRES,State"],
                         capture_output=True, text=True, check=True).stdout
    su = hours = 0.0
    for line in out.splitlines():
        jid, el, tres, state = line.split("|")
        b = dict(kv.split("=", 1) for kv in tres.split(",") if "=" in kv).get("billing")
        if b and el:
            su += float(b) * int(el) / 3600
            hours += int(el) / 3600
    return su, hours


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", required=True)
    ap.add_argument("--extra-jobs", nargs="*", default=[],
                    help="CPU job ids not recorded by the orchestrator")
    ap.add_argument("--orch-job", nargs="*", default=[], help="orchestrator job id(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", nargs="*", type=int, default=[25000, 62000, 100000, 225000])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = lc.root()

    accs = pd.read_csv(root / "manifests" / f"{args.batch}_accessions.tsv", sep="\t").accession
    rows, chunk_ids = [], set()
    for a in accs:
        st = lc.load_state(a)
        i = st.get("info", {})
        chunk_ids.update(i.get("chunks", []))
        rows.append({"accession": a, "state": st.get("state"), "unavailable": bool(i.get("unavailable")),
                     **{k: i.get(k) for k in ("contig_bp", "bp_ge120", "contig_count", "orfs", "orf_aa",
                                              "prodigal_genes", "prodigal_plus", "download_bytes",
                                              "prodigal_seconds", "orf_and_match_seconds",
                                              "expected_contig_bp")}})
    st = pd.DataFrame(rows)
    done = st[st.orfs.notna()]
    mans = [json.load(open(lc.chunk_manifest_path(c))) for c in sorted(chunk_ids)]
    enc = [m for m in mans if m.get("aa_count")]

    orch = json.load(open(root / "manifests" / f"{args.batch}_orchestrator.json"))
    cpu_su, cpu_h = billed_su(orch["cpu_jobs"] + args.extra_jobs)
    gpu_su, gpu_h = billed_su(orch["gpu_jobs"])
    up_su, up_h = billed_su(orch["upload_jobs"])
    orch_su, _ = billed_su(args.orch_job)

    bp = float(done.contig_bp.sum())
    aa = float(done.orf_aa.sum())
    m = {
        "batch": args.batch,
        "accessions_selected": int(len(st)),
        "accessions_unavailable_in_logan": int(st.unavailable.sum()),
        "accessions_with_orfs": int(len(done)),
        "accessions_complete_remote": int((st.state == "complete").sum()),
        "contig_Tbp": bp / 1e12,
        "contig_bp_ge120_frac": float(done.bp_ge120.sum() / bp),
        "orfs_billion": float(done.orfs.sum() / 1e9),
        "aa_trillion": aa / 1e12,
        "prodigal_genes_million": float(done.prodigal_genes.sum() / 1e6),
        "prodigal_plus_frac": float(done.prodigal_plus.sum() / done.orfs.sum()),
        "download_TB": float(done.download_bytes.sum() / 1e12),
        "chunks": len(mans),
        "chunks_encoded": len(enc),
        "chunks_remote_verified": sum(1 for x in mans if x["state"] == "remote_verified"),
        "cache_TB": sum(x["bytes"] for x in enc) / 1e12,
        "cache_bytes_per_aa": sum(x["bytes"] for x in enc) / sum(x["aa_count"] for x in enc),
        "encode_aa_per_gcd_s": sum(x["aa_count"] for x in enc) / sum(x["encode_seconds"] for x in enc),
        "su_cpu_stage": cpu_su, "su_gpu_encoding": gpu_su, "su_upload": up_su, "su_orchestrator": orch_su,
        "gpu_node_hours": gpu_h, "gcd_hours_billed": gpu_su / 128,
        "gcd_hours_encoding_only": sum(x["encode_seconds"] for x in enc) / 3600,
        "cpu_core_seconds_per_Mbp_in_stage": float((done.prodigal_seconds.sum()
                                                    + done.orf_and_match_seconds.sum()) / bp * 1e6),
    }
    m["gpu_efficiency"] = m["gcd_hours_encoding_only"] / m["gcd_hours_billed"] if gpu_su else None
    verified = [x for x in mans if x["state"] == "remote_verified"]
    m["upload_TB_verified"] = sum(x["bytes"] for x in verified) / 1e12
    m["upload_TB_per_hour_measured"] = m["upload_TB_verified"] / up_h if up_h else None
    rate = {"su_gpu_per_Tbp": gpu_su / (bp / 1e12), "su_cpu_per_Tbp": cpu_su / (bp / 1e12),
            "cache_TB_per_Tbp": m["cache_TB"] / (bp / 1e12) * len(mans) / max(1, len(enc)),
            "download_TB_per_Tbp": m["download_TB"] / (bp / 1e12),
            "orfs_billion_per_Tbp": m["orfs_billion"] / (bp / 1e12),
            "unavailable_frac": m["accessions_unavailable_in_logan"] / m["accessions_selected"]}
    m["rates"] = rate
    json.dump(m, open(out / "tranche_measured.json", "w"), indent=1)

    # Remaining pool under the pilot filters.
    d = pd.read_parquet(root / "meta" / "logan_metagenomic_candidates.parquet",
                        columns=["accession", "assay_type", "libraryselection", "platform",
                                 "organism", "contig_bp", "n50", "biome", "bioproject"])
    keep = ((d.assay_type == "WGS") & d.libraryselection.isin(["RANDOM", "RANDOM PCR", "unspecified"])
            & d.platform.isin(["ILLUMINA", "BGISEQ", "DNBSEQ", "ION_TORRENT", "ELEMENT"])
            & d.organism.fillna("").str.contains("metagenome$", case=False)
            & ~d.organism.fillna("").str.contains(NON_NATURAL, case=False)
            & d.contig_bp.between(5e6, 3e8) & (d.n50 >= 150))
    taken = {p.stem for p in (root / "state").glob("*.json")}
    pool = d[keep & ~d.accession.isin(taken)].sample(frac=1, random_state=102)
    used = d[d.accession.isin(taken)].groupby("bioproject").size()
    pool["rank"] = (pool.groupby(pool.bioproject.fillna(pool.accession)).cumcount()
                    + pool.bioproject.map(used).fillna(0).astype(int))

    bal = subprocess.run(["pawseyAccountBalance", "-p", f"{os.environ.get('PAWSEY_PROJECT', 'pawsey1018')}-gpu"],
                         capture_output=True, text=True).stdout
    gpu_left = None
    for line in bal.splitlines():
        f = line.split()
        if f and f[0].endswith("-gpu") and len(f) >= 3:
            gpu_left = float(f[1]) - float(f[2])
    rc = subprocess.run([os.environ.get("RCLONE", "rclone"), "about", "genome_entropy:", "--json"],
                        capture_output=True, text=True)
    remote_free = json.loads(rc.stdout)["free"] / 1e12 if rc.returncode == 0 else None
    pending_upload = m["cache_TB"] - m["upload_TB_verified"]

    sc = []
    for n in args.scenarios:
        for cap in (25, 50, 100, 250, 1000):
            q = pool[pool["rank"] < cap]
            if len(q) < n:
                continue
            tb = q.contig_bp.mean() * n * (1 - rate["unavailable_frac"]) / 1e12
            gsu = tb * rate["su_gpu_per_Tbp"]
            cache = tb * rate["cache_TB_per_Tbp"]
            sc.append({"accessions": n, "max_per_bioproject": cap, "pool_available": len(q),
                       "mean_Mbp": q.contig_bp.mean() / 1e6, "contig_Tbp": tb,
                       "orfs_billion": tb * rate["orfs_billion_per_Tbp"],
                       "gpu_SU_million": gsu / 1e6, "cpu_SU_thousand": tb * rate["su_cpu_per_Tbp"] / 1e3,
                       "cache_TB": cache, "download_TB": tb * rate["download_TB_per_Tbp"],
                       "upload_days": (cache / m["upload_TB_per_hour_measured"] / 24)
                       if m["upload_TB_per_hour_measured"] else None,
                       "fits_gpu_SU": gpu_left is not None and gsu <= gpu_left,
                       "fits_remote": remote_free is not None and cache <= remote_free - pending_upload})
            break   # the tightest cap that can supply n runs
    proj = pd.DataFrame(sc)
    proj.to_csv(out / "next_tranche_projection.tsv", sep="\t", index=False)
    ctx = {"gpu_SU_remaining": gpu_left, "remote_free_TB": remote_free,
           "phaseb1_cache_not_yet_uploaded_TB": pending_upload,
           "remaining_pool_runs": int(len(pool)), "remaining_pool_Tbp": float(pool.contig_bp.sum() / 1e12)}
    json.dump(ctx, open(out / "budget_context.json", "w"), indent=1)
    print(json.dumps(m, indent=1))
    print(json.dumps(ctx, indent=1))
    print(proj.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
