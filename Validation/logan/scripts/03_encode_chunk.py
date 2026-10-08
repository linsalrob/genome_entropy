#!/usr/bin/env python3
"""Encode one cache chunk with ModernProst and publish it as a validated .tsv.zst.

  planned -> encoded -> cache_validated

Reads the chunk manifest from 03_plan_chunks.py, pulls each member's rows out
of work/<acc>/orfs.tsv.zst (after checking that file's sha256 against the
accession state, so a chunk can never be built from a regenerated ORF table
that differs from the one that was planned), encodes the amino-acid sequences
with genome_entropy's ModernProstThreeDiEncoder at the package's default
encoding size, and writes one row per ORF:

  accession contig orf_id start end strand contig_len stop
  aa_sequence three_di twelve_state

AA, 3Di and 12-state are written together on one row, so they cannot drift
apart. Before publication every row must have len(aa) == len(3Di) ==
len(12st) and only valid alphabet letters, and after compression the file is
zstd-tested and re-read in full: the row count and the per-column checksums
must equal what was written. Only then is the .partial renamed and the
manifest moved to cache_validated.

WORKERS. GTDB (#92) found that one encoding process leaves a V100 mostly idle
and that ~4 processes per GPU roughly doubled throughput. --workers N starts N
processes on the same device, each loading the model once and encoding a
length-balanced share of the chunk. The right N for an MI250X GCD is measured
in the pilot, not assumed.

  03_encode_chunk.py logan_metagenome_000001 --workers 4
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
from pathlib import Path
from typing import List, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logan_common as lc  # noqa: E402

logging.getLogger("genome_entropy").setLevel(logging.WARNING)

THREEDI = set("ACDEFGHIKLMNPQRSTVWY")
TWELVE = set("ABCDEFGHIJKL")

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


def encode_all(seqs: List[str], workers: int, device: str, encoding_size: int):
    if workers <= 1:
        _init_worker(lc.MODEL_NAME, device)
        return _encode_shard((seqs, encoding_size))
    import multiprocessing as mp
    # Length-balanced shards: longest first, dealt round-robin.
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    shards = [order[w::workers] for w in range(workers)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(lc.MODEL_NAME, device)) as pool:
        parts = pool.map(_encode_shard,
                         [([seqs[i] for i in s], encoding_size) for s in shards])
    out: List[Tuple[str, str]] = [None] * len(seqs)  # type: ignore
    for s, part in zip(shards, parts):
        for i, enc in zip(s, part):
            out[i] = enc
    return out


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
        want = st["info"]["orfs_sha256"]
        if lc.sha256(path) != want:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("chunk_id")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encoding-size", type=int, default=None,
                    help="default: genome_entropy DEFAULT_ENCODING_SIZE")
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--limit", type=int, default=0,
                    help="encode only the first N rows and do not publish (calibration)")
    args = ap.parse_args()

    from genome_entropy.config import DEFAULT_ENCODING_SIZE
    enc_size = args.encoding_size or DEFAULT_ENCODING_SIZE

    man_path = lc.root() / "manifests" / "chunks" / f"{args.chunk_id}.json"
    man = json.load(open(man_path))
    if man["state"] in ("cache_validated", "uploaded", "remote_verified", "complete") \
            and not args.limit:
        print(f"{args.chunk_id}: already {man['state']}; nothing to do")
        return 0

    t0 = time.time()
    df = load_members(man)
    if args.limit:
        df = df.iloc[: args.limit].copy()
    elif len(df) != man["orf_count_expected"]:
        raise RuntimeError(f"{len(df)} rows loaded, manifest expects {man['orf_count_expected']}")
    t_load = time.time() - t0

    t1 = time.time()
    enc = encode_all(df.aa_sequence.tolist(), args.workers, args.device, enc_size)
    t_enc = time.time() - t1
    df["three_di"] = [e[0] for e in enc]
    df["twelve_state"] = [e[1] for e in enc]
    n_aa = int(df.aa_sequence.str.len().sum())
    rate = n_aa / t_enc if t_enc else 0.0
    print(f"encoded {len(df):,} ORFs / {n_aa:,} aa in {t_enc:.0f} s "
          f"({rate:,.0f} aa/s, {len(df) / t_enc:,.0f} ORF/s) workers={args.workers}")

    bad = [i for i, (a, t, w) in enumerate(zip(df.aa_sequence, df.three_di, df.twelve_state))
           if not (len(a) == len(t) == len(w)) or not set(t) <= THREEDI or not set(w) <= TWELVE]
    if bad:
        raise RuntimeError(f"{len(bad)} rows fail length/alphabet checks, first {bad[:5]}")

    if args.limit:
        calib = {"chunk_id": args.chunk_id, "rows": len(df), "aa": n_aa,
                 "workers": args.workers, "encoding_size": enc_size,
                 "encode_seconds": round(t_enc, 1), "aa_per_s": round(rate),
                 "time": lc.now(), "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
        print(json.dumps(calib))
        with open(lc.root() / "manifests" / "calibration.jsonl", "a") as fh:
            fh.write(json.dumps(calib) + "\n")
        return 0

    out_dir = lc.root() / "cache" / "modernprost-50M" / lc.CACHE_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.chunk_id}.tsv.zst"
    tmp = out.with_name(out.name + ".partial")
    cols = list(lc.CACHE_COLUMNS)
    digests = {c: column_digest(df[c]) for c in ("orf_id", "aa_sequence", "three_di", "twelve_state")}

    t2 = time.time()
    proc = subprocess.Popen(["zstd", "-q", f"-{args.zstd_level}", "-T8", "-f", "-o", str(tmp)],
                            stdin=subprocess.PIPE)
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
    t_write = time.time() - t2
    os.replace(tmp, out)

    accs = []
    for m in man["members"]:
        if m["accession"] not in accs:
            accs.append(m["accession"])
    man.update({
        "state": "cache_validated",
        "cache_file": out.name,
        "accessions": accs,
        "orf_count": int(len(df)),
        "aa_count": n_aa,
        "bytes": out.stat().st_size,
        "sha256": lc.sha256(out),
        "column_sha256": digests,
        "model_revision": model_revision(),
        "encoding_size": enc_size,
        "workers": args.workers,
        "load_seconds": round(t_load, 1),
        "encode_seconds": round(t_enc, 1),
        "write_validate_seconds": round(t_write, 1),
        "aa_per_second": round(rate),
        "encoded": lc.now(),
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
        "node": os.uname().nodename,
        "columns": cols,
        "remote_verified": False,
        **lc.tool_versions(),
    })
    lc.atomic_write_json(man_path, man)

    # An accession is cache_validated once every slice of it is.
    for acc in accs:
        if all_slices_validated(acc):
            st = lc.load_state(acc)
            if not lc.reached(st, "genome_entropy_done"):
                lc.advance(acc, "genome_entropy_done")
            if not lc.reached(lc.load_state(acc), "cache_validated"):
                lc.advance(acc, "cache_validated", chunks=chunks_of(acc))
    print(f"{args.chunk_id}: cache_validated -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def chunk_manifests():
    for p in sorted((lc.root() / "manifests" / "chunks").glob("*.json")):
        yield json.load(open(p))


def chunks_of(acc: str) -> List[str]:
    return [m["chunk_id"] for m in chunk_manifests()
            if any(x["accession"] == acc for x in m["members"])]


def all_slices_validated(acc: str) -> bool:
    st = lc.load_state(acc)
    total, ok = int(st["info"]["orfs"]), 0
    for m in chunk_manifests():
        for x in m["members"]:
            if x["accession"] == acc:
                if m["state"] not in ("cache_validated", "uploaded", "remote_verified", "complete"):
                    return False
                ok += x["n_rows"]
    return ok == total


if __name__ == "__main__":
    sys.exit(main())
