#!/usr/bin/env python3
"""Upload validated chunks and per-accession tables to the Teams remote, verify, then clean up.

  chunk:     cache_validated -> uploaded -> remote_verified
  accession: cache_validated -> uploaded -> remote_verified -> complete

The remote is OneDrive (Teams document library), which stores QuickXorHash.
A copy is accepted only when BOTH

  * `rclone check --one-way` reports the object as matching (size + hash), and
  * an explicit `rclone lsjson --hash` of the remote object gives the same
    size and QuickXorHash as `rclone hashsum quickxor` of the local file,

and that remote hash is written into the manifest. The exit status of
`rclone copy` alone is never taken as evidence the object is there.

Large local intermediates (contigs.fa.zst, orfs.tsv.zst) are deleted only
with --cleanup, and only for accessions that have reached `complete`, i.e.
after every chunk holding their ORFs and every one of their tables is
remote-verified. A rerun skips everything already verified.

Remote layout under genome_entropy:General/LOGAN/:

  modernprost-50M/v1/<chunk>.tsv.zst      the encoded cache
  manifests/chunks/<chunk>.json           one manifest per chunk
  manifests/<batch>_accessions.tsv        selection manifest(s)
  manifests/accession_status.tsv          one row per accession, all states
  prodigal/v1/<acc>/{orf_prodigal,prodigal_genes}.tsv.zst,
                    prodigal.{gff,faa}.zst

  04_upload_verify.py [--cleanup] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

CACHE_REMOTE = f"{lc.REMOTE_ROOT}/modernprost-50M/{lc.CACHE_VERSION}"
TABLES = ("orf_prodigal.tsv.zst", "prodigal_genes.tsv.zst", "prodigal.gff.zst",
          "prodigal.faa.zst")
DRY = False


RCLONE = os.environ.get("RCLONE", "rclone")


def rclone(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = [RCLONE, *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=capture, text=True)


def local_quickxor(path: Path) -> str:
    r = rclone("hashsum", "quickxor", str(path))
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.split()[0].lower()


def upload_and_verify(path: Path, remote_dir: str) -> dict:
    """Copy one file and prove the remote object matches. Returns evidence."""
    size = path.stat().st_size
    qx = local_quickxor(path)
    if DRY:
        return {"dry_run": True, "bytes": size, "quickxor": qx}
    r = rclone("copyto", str(path), f"{remote_dir}/{path.name}", "--retries", "5")
    if r.returncode:
        raise RuntimeError(f"copy failed for {path}: {r.stderr[-2000:]}")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(path.name + "\n")
        flist = fh.name
    chk = rclone("check", str(path.parent), remote_dir, "--one-way",
                 "--files-from-raw", flist)
    Path(flist).unlink()
    if chk.returncode:
        raise RuntimeError(f"rclone check failed for {path}: {chk.stderr[-2000:]}")
    lj = rclone("lsjson", "--hash", f"{remote_dir}/{path.name}")
    if lj.returncode:
        raise RuntimeError(f"lsjson failed for {path}: {lj.stderr}")
    entries = json.loads(lj.stdout)
    if len(entries) != 1:
        raise RuntimeError(f"lsjson returned {len(entries)} entries for {path.name}")
    e = entries[0]
    rqx = (e.get("Hashes") or {}).get("quickxor", "").lower()
    if e["Size"] != size or rqx != qx:
        raise RuntimeError(f"remote mismatch for {path.name}: size {e['Size']} vs {size}, "
                           f"quickxor {rqx} vs {qx}")
    return {"remote": f"{remote_dir}/{path.name}", "bytes": size, "quickxor": qx,
            "rclone_check": "pass", "verified": lc.now()}


def do_chunks() -> None:
    for p in sorted((lc.root() / "manifests" / "chunks").glob("*.json")):
        man = json.load(open(p))
        if man["state"] not in ("cache_validated", "uploaded"):
            continue
        f = lc.root() / "cache" / "modernprost-50M" / lc.CACHE_VERSION / man["cache_file"]
        if lc.sha256(f) != man["sha256"]:
            raise RuntimeError(f"{f} no longer matches its manifest sha256")
        if man["state"] == "cache_validated" and not DRY:
            man["state"] = "uploaded"
            lc.atomic_write_json(p, man)
        ev = upload_and_verify(f, CACHE_REMOTE)
        if DRY:
            print(f"[dry-run] {man['chunk_id']}: would upload {ev['bytes']:,} bytes")
            continue
        man.update(state="remote_verified", remote_verified=True, remote=ev)
        lc.atomic_write_json(p, man)
        upload_and_verify(p, f"{lc.REMOTE_ROOT}/manifests/chunks")
        print(f"{man['chunk_id']}: remote_verified ({ev['bytes'] / 1e6:.1f} MB)")


def chunk_states_for(acc: str):
    out = []
    for p in sorted((lc.root() / "manifests" / "chunks").glob("*.json")):
        man = json.load(open(p))
        if any(m["accession"] == acc for m in man["members"]):
            out.append(man["state"])
    return out


def do_accessions(cleanup: bool) -> None:
    for sp in sorted((lc.root() / "state").glob("*.json")):
        acc = sp.stem
        st = lc.load_state(acc)
        if not lc.reached(st, "cache_validated"):
            continue
        wd = lc.work_dir(acc)
        if not lc.reached(st, "remote_verified"):
            states = chunk_states_for(acc)
            if not states or any(s != "remote_verified" for s in states):
                continue
            if not lc.reached(st, "uploaded") and not DRY:
                lc.advance(acc, "uploaded")
            evidence = {}
            for t in TABLES:
                evidence[t] = upload_and_verify(wd / t, f"{lc.REMOTE_ROOT}/prodigal/{lc.CACHE_VERSION}/{acc}")
            if DRY:
                continue
            lc.advance(acc, "remote_verified", remote_tables=evidence)
            lc.advance(acc, "complete")
            print(f"{acc}: complete")
        if cleanup and lc.reached(lc.load_state(acc), "complete"):
            freed = 0
            for name in ("contigs.fa.zst", "orfs.tsv.zst", "contigs.ge120.fa"):
                f = wd / name
                if f.exists():
                    freed += f.stat().st_size
                    if not DRY:
                        f.unlink()
            if freed:
                print(f"{acc}: removed {freed / 1e6:.1f} MB of intermediates")


def status_table() -> Path:
    rows = []
    for sp in sorted((lc.root() / "state").glob("*.json")):
        st = json.load(open(sp))
        info = st.get("info", {})
        rows.append({"accession": st["accession"], "state": st.get("state"),
                     "batch": info.get("batch"), "biome": info.get("biome"),
                     "contig_count": info.get("contig_count"),
                     "contig_bp": info.get("contig_bp"), "bp_ge120": info.get("bp_ge120"),
                     "prodigal_genes": info.get("prodigal_genes"), "orfs": info.get("orfs"),
                     "orf_aa": info.get("orf_aa"), "prodigal_plus": info.get("prodigal_plus"),
                     "chunks": ",".join(info.get("chunks", [])),
                     "last_update": st["history"][-1]["time"] if st.get("history") else ""})
    out = lc.root() / "manifests" / "accession_status.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    return out


def main() -> int:
    global DRY
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    DRY = args.dry_run
    do_chunks()
    do_accessions(args.cleanup)
    out = status_table()
    if not DRY:
        upload_and_verify(out, f"{lc.REMOTE_ROOT}/manifests")
        for sel in sorted((lc.root() / "manifests").glob("*_accessions.tsv")):
            upload_and_verify(sel, f"{lc.REMOTE_ROOT}/manifests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
