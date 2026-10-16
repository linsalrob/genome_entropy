#!/usr/bin/env python3
"""Encode cache chunks with ModernProst and publish each as a validated .tsv.zst.

  planned -> cache_validated

For each chunk manifest from 03_plan_chunks.py, pull each member's rows out
of work/<acc>/orfs.tsv.zst (after checking that file's sha256 against the
accession state, so a chunk can never be built from a regenerated ORF table
that differs from the one that was planned), encode the amino-acid sequences
with genome_entropy's ModernProstThreeDiEncoder at the package's default
encoding size, and write one row per ORF:

  accession contig orf_id start end strand contig_len stop
  aa_sequence three_di twelve_state

AA, 3Di and 12-state are written together on one row, so they cannot drift
apart. Before publication every row must have len(aa) == len(3Di) ==
len(12st) and only valid alphabet letters; after compression the file is
zstd-tested and re-read in full, and the row count and per-column checksums
must equal what was written. Only then is the .partial renamed and the
manifest moved to cache_validated.

STREAMS. One process handles a list of chunks on one GCD. The --workers
encoder processes are started once and kept for the whole list (the pilot
reloaded the model for every chunk), and writing/validating chunk k runs in
a background thread while chunk k+1 encodes, so zstd -19 costs no GPU time.
A chunk that fails is logged and skipped; the others continue, and the exit
status is non-zero if any failed. Rerunning skips cache_validated chunks.

WORKERS. GTDB (#92) found ~4 processes per V100; the pilot measured
1/2/4/8 workers on an MI250X GCD at 42k/106k/113k/106k aa/s, so 4.

  03_encode_chunk.py logan_metagenome_000001 [more ids] --workers 4
  03_encode_chunk.py --chunks-file stream_3.txt --workers 4
  03_encode_chunk.py logan_metagenome_000001 --workers 2 --limit 100000   # calibration
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logan_common as lc  # noqa: E402

logging.getLogger("genome_entropy").setLevel(logging.WARNING)

THREEDI = set("ACDEFGHIKLMNPQRSTVWY")
TWELVE = set("ABCDEFGHIJKL")
DONE = ("cache_validated", "uploaded", "remote_verified", "complete")

_ENCODER = None


def _init_worker(model: str, device: str) -> None:
    global _ENCODER
    logging.getLogger("genome_entropy").setLevel(logging.WARNING)
    from genome_entropy.encode3di.modernprost import ModernProstThreeDiEncoder
    _ENCODER = ModernProstThreeDiEncoder(model, device=device)
    _ENCODER._load_model()


def _encode_shard(args: Tuple[List[str], int]) -> List[Tuple[str, str]]:
    seqs, encoding_size = args
    out = _ENCODER.encode(seqs, encoding_size=encoding_size)
    return [(e.three_di, e.twelve_state) for e in out]


class Encoder:
    """N encoder processes on one device, started once and reused."""

    def __init__(self, workers: int, device: str, encoding_size: int):
        self.workers, self.encoding_size = workers, encoding_size
        if workers <= 1:
            _init_worker(lc.MODEL_NAME, device)
            self.pool = None
        else:
            import multiprocessing as mp
            self.pool = mp.get_context("spawn").Pool(
                workers, initializer=_init_worker, initargs=(lc.MODEL_NAME, device))

    def encode(self, seqs: List[str]) -> List[Tuple[str, str]]:
        if self.pool is None:
            return _encode_shard((seqs, self.encoding_size))
        # Length-balanced shards: longest first, dealt round-robin.
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        shards = [order[w::self.workers] for w in range(self.workers)]
        parts = self.pool.map(_encode_shard,
                              [([seqs[i] for i in s], self.encoding_size) for s in shards])
        out: List[Tuple[str, str]] = [None] * len(seqs)  # type: ignore
        for s, part in zip(shards, parts):
            for i, enc in zip(s, part):
                out[i] = enc
        return out

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def model_revision() -> str:
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    ref = hf / "hub" / ("models--" + lc.MODEL_NAME.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else "unknown"


def load_members(man: dict) -> pd.DataFrame:
    frames = []
    for m in man["members"]:
        acc = m["accession"]
        st = lc.load_state(acc)
        path = lc.work_dir(acc) / "orfs.tsv.zst"
        if lc.sha256(path) != st["info"]["orfs_sha256"]:
            raise RuntimeError(f"{acc}: orfs.tsv.zst does not match the planned table")
        df = pd.read_csv(path, sep="\t", dtype={"contig": str, "strand": str},
                         keep_default_na=False)
        if len(df) != st["info"]["orfs"]:
            raise RuntimeError(f"{acc}: {len(df)} rows, state says {st['info']['orfs']}")
        sl = df.iloc[m["first_row"]: m["first_row"] + m["n_rows"]].copy()
        sl.insert(0, "accession", acc)
        frames.append(sl)
    return pd.concat(frames, ignore_index=True)


def column_digest(values) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode())
        h.update(b"\n")
    return h.hexdigest()


def check_rows(df: pd.DataFrame) -> None:
    bad = [i for i, (a, t, w) in enumerate(zip(df.aa_sequence, df.three_di, df.twelve_state))
           if not (len(a) == len(t) == len(w)) or not set(t) <= THREEDI or not set(w) <= TWELVE]
    if bad:
        raise RuntimeError(f"{len(bad)} rows fail length/alphabet checks, first {bad[:5]}")


def publish(chunk_id: str, man: dict, df: pd.DataFrame, timing: dict, args) -> str:
    """Write, verify and publish one encoded chunk; update its accessions."""
    out = lc.cache_path(chunk_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    cols = list(lc.CACHE_COLUMNS)
    digests = {c: column_digest(df[c]) for c in ("orf_id", "aa_sequence", "three_di", "twelve_state")}

    t0 = time.time()
    proc = subprocess.Popen(["zstd", "-q", f"-{args.zstd_level}", f"-T{args.zstd_threads}",
                             "-f", "-o", str(tmp)], stdin=subprocess.PIPE)
    with io.TextIOWrapper(proc.stdin, encoding="ascii", newline="") as fh:
        df[cols].to_csv(fh, sep="\t", index=False, lineterminator="\n")
    if proc.wait() != 0:
        raise RuntimeError("zstd failed")
    subprocess.run(["zstd", "-tq", str(tmp)], check=True)
    back = pd.read_csv(tmp, sep="\t", dtype=str, keep_default_na=False,
                       compression={"method": "zstd"})
    if list(back.columns) != cols or len(back) != len(df):
        raise RuntimeError(f"re-read: {len(back)} rows / {list(back.columns)}")
    for c, d in digests.items():
        if column_digest(back[c]) != d:
            raise RuntimeError(f"re-read: column {c} differs from what was written")
    del back
    t_write = time.time() - t0
    os.replace(tmp, out)

    accs = list(dict.fromkeys(m["accession"] for m in man["members"]))
    n_aa = int(df.aa_sequence.str.len().sum())
    man.update({
        "state": "cache_validated",
        "cache_file": f"{lc.chunk_shard(chunk_id)}/{out.name}",
        "accessions": accs,
        "orf_count": int(len(df)),
        "aa_count": n_aa,
        "bytes": out.stat().st_size,
        "sha256": lc.sha256(out),
        "column_sha256": digests,
        "model_revision": model_revision(),
        "encoding_size": args.encoding_size_resolved,
        "workers": args.workers,
        "zstd_level": args.zstd_level,
        "load_seconds": round(timing["load"], 1),
        "encode_seconds": round(timing["encode"], 1),
        "write_validate_seconds": round(t_write, 1),
        "aa_per_second": round(n_aa / timing["encode"]) if timing["encode"] else 0,
        "encoded": lc.now(),
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
        "node": os.uname().nodename,
        "gpu": os.environ.get("ROCR_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES", "")),
        "columns": cols,
        "remote_verified": False,
        **lc.tool_versions(),
    })
    lc.atomic_write_json(lc.chunk_manifest_path(chunk_id), man)

    for acc in accs:
        if all_slices_validated(acc):
            if not lc.reached(lc.load_state(acc), "genome_entropy_done"):
                lc.advance(acc, "genome_entropy_done")
            if not lc.reached(lc.load_state(acc), "cache_validated"):
                lc.advance(acc, "cache_validated")
    return f"{chunk_id}: cache_validated -> {out} ({out.stat().st_size / 1e6:.1f} MB, write {t_write:.0f} s)"


def planned_chunks(acc: str) -> List[str]:
    """The chunk ids holding an accession's ORFs, recorded by the planner.

    Pilot accessions predate that record, so fall back to scanning manifests.
    """
    st = lc.load_state(acc)
    ids = st.get("info", {}).get("chunks")
    if ids:
        return list(ids)
    out = []
    for p in sorted((lc.root() / "manifests" / "chunks").glob("*.json")):
        m = json.load(open(p))
        if any(x["accession"] == acc for x in m["members"]):
            out.append(m["chunk_id"])
    return out


def all_slices_validated(acc: str) -> bool:
    total, ok = int(lc.load_state(acc)["info"]["orfs"]), 0
    for cid in planned_chunks(acc):
        m = json.load(open(lc.chunk_manifest_path(cid)))
        if m["state"] not in DONE:
            return False
        ok += sum(x["n_rows"] for x in m["members"] if x["accession"] == acc)
    return ok == total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("chunk_ids", nargs="*")
    ap.add_argument("--chunks-file", help="file of chunk ids, one per line")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encoding-size", type=int, default=None,
                    help="default: genome_entropy DEFAULT_ENCODING_SIZE")
    ap.add_argument("--zstd-level", type=int, default=19)
    ap.add_argument("--zstd-threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="encode only the first N rows of the first chunk and do not publish (calibration)")
    args = ap.parse_args()

    from genome_entropy.config import DEFAULT_ENCODING_SIZE
    args.encoding_size_resolved = args.encoding_size or DEFAULT_ENCODING_SIZE

    ids = list(args.chunk_ids)
    if args.chunks_file:
        ids += [l.strip() for l in open(args.chunks_file) if l.strip()]
    todo = []
    for cid in ids:
        man = json.load(open(lc.chunk_manifest_path(cid)))
        if man["state"] in DONE and not args.limit:
            print(f"{cid}: already {man['state']}; skipping", flush=True)
            continue
        todo.append(cid)
    if not todo:
        return 0

    enc = Encoder(args.workers, args.device, args.encoding_size_resolved)
    writer = ThreadPoolExecutor(1)
    pending: Optional[Tuple[str, object]] = None
    failed: List[str] = []

    def collect(p):
        cid, fut = p
        try:
            print(fut.result(), flush=True)
        except Exception:
            failed.append(cid)
            print(f"{cid}: FAILED while publishing\n{traceback.format_exc()}", flush=True)

    for cid in todo:
        try:
            man = json.load(open(lc.chunk_manifest_path(cid)))
            t0 = time.time()
            df = load_members(man)
            if args.limit:
                df = df.iloc[: args.limit].copy()
            elif len(df) != man["orf_count_expected"]:
                raise RuntimeError(f"{len(df)} rows loaded, manifest expects {man['orf_count_expected']}")
            t_load = time.time() - t0
            t1 = time.time()
            res = enc.encode(df.aa_sequence.tolist())
            t_enc = time.time() - t1
            df["three_di"] = [e[0] for e in res]
            df["twelve_state"] = [e[1] for e in res]
            del res
            check_rows(df)
            n_aa = int(df.aa_sequence.str.len().sum())
            print(f"{cid}: encoded {len(df):,} ORFs / {n_aa:,} aa in {t_enc:.0f} s "
                  f"({n_aa / t_enc:,.0f} aa/s) workers={args.workers}", flush=True)
            if args.limit:
                calib = {"chunk_id": cid, "rows": len(df), "aa": n_aa, "workers": args.workers,
                         "encoding_size": args.encoding_size_resolved,
                         "encode_seconds": round(t_enc, 1), "aa_per_s": round(n_aa / t_enc),
                         "time": lc.now(), "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
                with open(lc.root() / "manifests" / "calibration.jsonl", "a") as fh:
                    fh.write(json.dumps(calib) + "\n")
                print(json.dumps(calib))
                break
        except Exception:
            failed.append(cid)
            print(f"{cid}: FAILED\n{traceback.format_exc()}", flush=True)
            continue
        # One publish in flight: wait for the previous chunk's before queueing
        # this one, so at most two encoded chunks are held in memory.
        if pending is not None:
            collect(pending)
        pending = (cid, writer.submit(publish, cid, man, df, {"load": t_load, "encode": t_enc}, args))
        del df
    if pending is not None:
        collect(pending)
    writer.shutdown()
    enc.close()
    if failed:
        print(f"{len(failed)} chunk(s) failed: {' '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
