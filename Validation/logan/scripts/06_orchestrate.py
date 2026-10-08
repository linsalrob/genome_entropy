#!/usr/bin/env python3
"""Drive one batch from CPU-stage results to verified remote cache, unattended.

A Phase B batch takes days, and the stages overlap: GPU nodes can start on
the chunks of the accessions that finished their CPU stage first. This loop,
run as a single-core job, does every hand-off the pilot did by hand:

  every cycle (default 20 min):
    1. plan new chunks from accessions that reached orfs_called
       (03_plan_chunks.py; the last partial chunk only once the CPU stage
       has finished);
    2. submit an 03_encode_node.slurm array for planned chunks not yet
       submitted, once there are enough to fill a node, or at the end;
    3. resubmit chunks whose GPU job ended without validating them (once);
    4. submit 04_upload_verify.slurm --cleanup when none is running and
       enough chunks are waiting, or at the end;
    5. after the CPU array has finished, resubmit accessions short of
       orfs_called (once, PER_JOB per element);
    6. write manifests/<batch>_progress.json and stop when nothing is left
       to do.

Everything it decides is recorded in manifests/<batch>_orchestrator.json, so
a resubmitted orchestrator (after the 4-day `long` limit, say) carries on
where the last one stopped. It never cancels jobs and never deletes data
other than through 04's --cleanup.

  06_orchestrate.py --batch phaseb1 --cpu-job 51234567
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

S = Path(__file__).resolve().parent
DONE = ("cache_validated", "uploaded", "remote_verified", "complete")


def log(msg: str) -> None:
    print(f"{lc.now()} {msg}", flush=True)


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def active(job_ids) -> bool:
    ids = [str(j) for j in job_ids if j]
    if not ids:
        return False
    out = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i"],
                         capture_output=True, text=True).stdout
    return bool(out.strip())


def sbatch(*args: str) -> str:
    # Setonix sets SBATCH_EXPORT=NONE, so a job submitted from here starts
    # with a clean environment and env.sh falls back to the production root
    # and remote. Pass them explicitly: a test batch under another
    # LOGAN_ROOT must never act on production state.
    export = "NONE," + ",".join(f"{k}={os.environ[k]}" for k in
                                ("LOGAN_ROOT", "LOGAN_REMOTE", "LOGAN_SCRIPTS") if k in os.environ)
    jid = sh("sbatch", "--parsable", f"--export={export}", *args).split(";")[0]
    log(f"submitted {jid}: {' '.join(args[-4:])}")
    return jid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", required=True)
    ap.add_argument("--cpu-job", required=True, help="job id of the 02_cpu_batch array")
    ap.add_argument("--per-node", type=int, default=400, help="chunks per GPU node element")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--upload-min", type=int, default=300,
                    help="chunks waiting before an upload job is worth starting")
    ap.add_argument("--cpu-per-job", type=int, default=100)
    ap.add_argument("--cycle", type=int, default=1200, help="seconds between cycles")
    ap.add_argument("--gpu-sbatch", default="", help="extra sbatch options for GPU arrays")
    ap.add_argument("--cpu-sbatch", default="", help="extra sbatch options for CPU retries")
    args = ap.parse_args()

    root = lc.root()
    os.chdir(root)
    accs_file = root / "manifests" / f"{args.batch}_accessions.tsv"
    accs = pd.read_csv(accs_file, sep="\t").accession.tolist()
    orch_path = root / "manifests" / f"{args.batch}_orchestrator.json"
    o = json.load(open(orch_path)) if orch_path.exists() else {
        "batch": args.batch, "cpu_jobs": [args.cpu_job], "cpu_retry_done": False,
        "gpu_jobs": [], "submitted": {}, "upload_jobs": []}
    if args.cpu_job not in o["cpu_jobs"]:
        o["cpu_jobs"].append(args.cpu_job)

    def save():
        lc.atomic_write_json(orch_path, o)

    while True:
        cpu_running = active(o["cpu_jobs"])

        # 5. one retry of accessions that did not reach orfs_called.
        if not cpu_running and not o["cpu_retry_done"]:
            short = [a for a in accs if not lc.reached(lc.load_state(a), "orfs_called")]
            o["cpu_retry_done"] = True
            if short:
                lst = root / "manifests" / f"{args.batch}_cpu_retry.tsv"
                pd.DataFrame({"accession": short}).to_csv(lst, sep="\t", index=False)
                n = math.ceil(len(short) / args.cpu_per_job)
                jid = sbatch(*args.cpu_sbatch.split(), f"--array=1-{n}", str(S / "02_cpu_batch.slurm"), str(lst),
                             str(args.cpu_per_job), "12")
                o["cpu_jobs"].append(jid)
                cpu_running = True
            save()

        # 1. plan.
        plan = [sys.executable, str(S / "03_plan_chunks.py"), "--accessions", str(accs_file)]
        if not cpu_running:
            plan.append("--final")
        subprocess.run(plan, check=True, stdout=subprocess.DEVNULL)

        mans = {}
        for a in accs:
            for c in lc.load_state(a).get("info", {}).get("chunks", []):
                if c not in mans:
                    mans[c] = json.load(open(lc.chunk_manifest_path(c)))["state"]
        gpu_running = active(o["gpu_jobs"])

        # 3. one resubmission of chunks whose GPU job has ended.
        lost = [c for c, s in mans.items() if s == "planned" and c in o["submitted"]
                and not active([o["submitted"][c]["job"]]) and o["submitted"][c]["tries"] < 2]
        # 2. new chunks.
        new = sorted(c for c, s in mans.items() if s == "planned" and c not in o["submitted"])
        todo = lost + new
        if todo and (len(todo) >= args.per_node or not cpu_running):
            k = len(o["gpu_jobs"]) + 1
            lst = root / "manifests" / f"{args.batch}_gpu_{k:03d}.txt"
            lst.write_text("\n".join(todo) + "\n")
            n = math.ceil(len(todo) / args.per_node)
            jid = sbatch(*args.gpu_sbatch.split(), f"--array=1-{n}", str(S / "03_encode_node.slurm"), str(lst),
                         str(args.per_node), str(args.workers))
            o["gpu_jobs"].append(jid)
            for c in todo:
                tries = o["submitted"].get(c, {}).get("tries", 0) + 1
                o["submitted"][c] = {"job": jid, "tries": tries}
            gpu_running = True
            save()

        # 4. upload.
        waiting = sum(1 for s in mans.values() if s in ("cache_validated", "uploaded"))
        upload_running = active(o["upload_jobs"])
        encoding_over = not cpu_running and not gpu_running
        if waiting and not upload_running and (waiting >= args.upload_min or encoding_over):
            jid = sbatch(str(S / "04_upload_verify.slurm"), "--batch", args.batch, "--cleanup")
            o["upload_jobs"].append(jid)
            upload_running = True
            save()

        # 6. progress, and the stopping rule.
        st = Counter(lc.load_state(a).get("state") for a in accs)
        cs = Counter(mans.values())
        prog = {"time": lc.now(), "accessions": dict(st), "chunks": dict(cs),
                "cpu_running": cpu_running, "gpu_running": gpu_running,
                "upload_running": upload_running}
        lc.atomic_write_json(root / "manifests" / f"{args.batch}_progress.json", prog)
        log(json.dumps(prog))
        unfinished_chunks = cs.get("planned", 0) + cs.get("cache_validated", 0) + cs.get("uploaded", 0)
        stuck = [c for c, s in mans.items() if s == "planned" and o["submitted"].get(c, {}).get("tries", 0) >= 2]
        if (not cpu_running and not gpu_running and not upload_running
                and o["cpu_retry_done"] and unfinished_chunks - len(stuck) == 0):
            log(f"finished; {len(stuck)} chunk(s) failed twice: {stuck[:20]}")
            return 0 if not stuck else 1
        time.sleep(args.cycle)


if __name__ == "__main__":
    sys.exit(main())
