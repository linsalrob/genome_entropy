#!/usr/bin/env python3
"""Bundle, upload and verify encoded chunks on the Teams remote, then clean up.

  chunk:     cache_validated -> uploaded -> remote_verified
  accession: cache_validated -> uploaded -> remote_verified -> complete

The remote is OneDrive (Teams document library), which stores QuickXorHash.
An object counts as verified only when BOTH

  * `rclone check --one-way` reports it as matching (size + hash), and
  * the remote listing (`rclone lsjson --hash`) gives the same size and
    QuickXorHash as `rclone hashsum quickxor` of the local file.

The remote hash is then written into the manifest. The exit status of
`rclone copy` is never taken on its own.

SCALE. The pilot verified one object at a time with four rclone calls each,
which took 1 h 38 min for 545 objects. Here every remote folder is handled
with one copy, one check, one listing and one local hashsum over the whole
batch (rclone's own --transfers / --checkers parallelism). Folders are
sharded 1,000 chunks deep, so no SharePoint folder grows past that.

PRODIGAL TABLES. Four small files per accession would be ~100,000 remote
objects at 25k accessions. Instead each chunk carries a bundle: an
uncompressed tar of the tables (already zstd) of every accession whose first
row is in that chunk, so each accession is in exactly one bundle. A chunk is
remote_verified only when its cache file AND its bundle are.

An accession is complete when every chunk holding its ORFs is
remote_verified. --cleanup then removes its large intermediates
(contigs.fa.zst, orfs.tsv.zst, any contigs.ge120.fa) and the local bundle;
the small tables stay in work/<acc>/ for analysis, and the local cache is
kept unless --cleanup-cache.

Remote layout under genome_entropy:General/LOGAN/:

  modernprost-50M/v1/<shard>/<chunk>.tsv.zst     encoded cache
  prodigal/v1/bundles/<shard>/<chunk>.tar        Prodigal + match tables
  manifests/chunks/<shard>/<chunk>.json          per-chunk manifest
  manifests/<batch>_accessions.tsv               selection manifests
  manifests/accession_status.tsv                 one row per accession
  prodigal/v1/<acc>/...                          pilot only (pre-bundling layout)

  04_upload_verify.py [--batch phaseb1] [--cleanup] [--cleanup-cache] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

TABLES = ("orf_prodigal.tsv.zst", "prodigal_genes.tsv.zst", "prodigal.gff.zst",
          "prodigal.faa.zst")
RCLONE = os.environ.get("RCLONE", "rclone")
DRY = False


def rclone(*args: str) -> subprocess.CompletedProcess:
    cmd = [RCLONE, *args]
    print("+", " ".join(cmd[:6]), "..." if len(cmd) > 6 else "", flush=True)
    return subprocess.run(cmd, capture_output=True, text=True)


def verify_folder(local_dir: Path, remote_dir: str, names: List[str]) -> Dict[str, dict]:
    """Upload `names` from local_dir to remote_dir; return evidence per verified name."""
    if not names:
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False) as fh:
        fh.write("\n".join(names) + "\n")
        flist = fh.name
    try:
        h = rclone("hashsum", "quickxor", str(local_dir), "--files-from-raw", flist,
                   "--checkers", "16")
        if h.returncode:
            raise RuntimeError(h.stderr[-2000:])
        local = {l.split(None, 1)[1].strip(): l.split()[0].lower()
                 for l in h.stdout.splitlines() if l.strip()}
        sizes = {n: (local_dir / n).stat().st_size for n in names}
        if DRY:
            print(f"[dry-run] {len(names)} file(s), {sum(sizes.values()) / 1e9:.2f} GB -> {remote_dir}")
            return {}
        c = rclone("copy", str(local_dir), remote_dir, "--files-from-raw", flist,
                   "--transfers", "16", "--checkers", "16", "--retries", "5")
        if c.returncode:
            print(f"WARNING: copy to {remote_dir} exited {c.returncode}: {c.stderr[-1500:]}")
        with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as rep:
            report = rep.name
        rclone("check", str(local_dir), remote_dir, "--one-way", "--files-from-raw", flist,
               "--checkers", "16", "--combined", report)
        status = {}
        for line in open(report):
            if len(line) > 2:
                status[line[2:].rstrip("\n")] = line[0]
        os.unlink(report)
        lj = rclone("lsjson", "--hash", "--files-only", remote_dir, "--files-from-raw", flist)
        remote = {}
        if lj.returncode == 0:
            for e in json.loads(lj.stdout or "[]"):
                remote[e["Path"]] = (e["Size"], (e.get("Hashes") or {}).get("quickxor", "").lower())
        ev = {}
        for n in names:
            r = remote.get(n)
            if status.get(n) == "=" and r and r[0] == sizes[n] and r[1] == local.get(n):
                ev[n] = {"remote": f"{remote_dir}/{n}", "bytes": sizes[n], "quickxor": r[1],
                         "rclone_check": "pass", "verified": lc.now()}
            else:
                print(f"NOT VERIFIED {remote_dir}/{n}: check={status.get(n)} remote={r} "
                      f"local=({sizes[n]}, {local.get(n)})")
        return ev
    finally:
        os.unlink(flist)


def build_bundle(man: dict) -> Tuple[Path, List[str]]:
    """Tar the tables of accessions whose first row lies in this chunk."""
    accs = [m["accession"] for m in man["members"] if m["first_row"] == 0]
    out = lc.bundle_path(man["chunk_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and man.get("bundle", {}).get("sha256") == lc.sha256(out):
        return out, accs
    tmp = out.with_name(out.name + ".partial")
    with tarfile.open(tmp, "w") as tar:
        for a in accs:
            for t in TABLES:
                tar.add(lc.work_dir(a) / t, arcname=f"{a}/{t}")
    with tarfile.open(tmp) as tar:
        got = sorted(tar.getnames())
    want = sorted(f"{a}/{t}" for a in accs for t in TABLES)
    if got != want:
        raise RuntimeError(f"{man['chunk_id']}: bundle members differ from the expected list")
    os.replace(tmp, out)
    return out, accs


def do_chunks(chunk_ids: List[str]) -> List[str]:
    mans = {c: json.load(open(lc.chunk_manifest_path(c))) for c in chunk_ids}
    todo = [c for c, m in mans.items() if m["state"] in ("cache_validated", "uploaded")]
    print(f"{len(todo):,} chunk(s) to upload")
    by_dir: Dict[Tuple[Path, str], List[str]] = defaultdict(list)
    for c in todo:
        m = mans[c]
        f = lc.cache_path(c)
        if lc.sha256(f) != m["sha256"]:
            raise RuntimeError(f"{f} no longer matches its manifest sha256")
        by_dir[(f.parent, lc.cache_remote_dir(c))].append(f.name)
        # A chunk in which no accession starts has no bundle to carry.
        if any(x["first_row"] == 0 for x in m["members"]):
            b, baccs = build_bundle(m)
            m["bundle"] = {"file": f"{lc.chunk_shard(c)}/{b.name}", "accessions": baccs,
                           "bytes": b.stat().st_size, "sha256": lc.sha256(b)}
            by_dir[(b.parent, lc.bundle_remote_dir(c))].append(b.name)
        else:
            m["bundle"] = None
        if m["state"] == "cache_validated" and not DRY:
            m["state"] = "uploaded"
            lc.atomic_write_json(lc.chunk_manifest_path(c), m)
    evidence: Dict[str, dict] = {}
    for (ldir, rdir), names in sorted(by_dir.items()):
        for n, e in verify_folder(ldir, rdir, names).items():
            evidence[f"{rdir}/{n}"] = e
    done = []
    for c in todo:
        m = mans[c]
        ce = evidence.get(f"{lc.cache_remote_dir(c)}/{c}.tsv.zst")
        be = evidence.get(f"{lc.bundle_remote_dir(c)}/{c}.tar") if m["bundle"] else True
        if DRY or not (ce and be):
            continue
        m["remote"] = ce
        if m["bundle"]:
            m["bundle"]["remote"] = be
        m["state"], m["remote_verified"] = "remote_verified", True
        lc.atomic_write_json(lc.chunk_manifest_path(c), m)
        done.append(c)
    # Manifests last, so the remote copy carries its own verification record.
    by_shard: Dict[str, List[str]] = defaultdict(list)
    for c in done:
        by_shard[lc.chunk_shard(c)].append(c)
    mdir = lc.root() / "manifests" / "chunks"
    for shard, cs in by_shard.items():
        verify_folder(mdir, f"{lc.REMOTE_ROOT}/manifests/chunks/{shard}", [f"{c}.json" for c in cs])
    print(f"{len(done):,} chunk(s) remote_verified")
    return done


def do_accessions(accs: List[str], cleanup: bool, cleanup_cache: bool) -> None:
    n_complete = 0
    for acc in accs:
        st = lc.load_state(acc)
        if not lc.reached(st, "cache_validated"):
            continue
        if not lc.reached(st, "complete"):
            ids = st["info"].get("chunks", [])
            states = [json.load(open(lc.chunk_manifest_path(c)))["state"] for c in ids]
            if not ids or any(s != "remote_verified" for s in states) or DRY:
                continue
            if not lc.reached(st, "uploaded"):
                lc.advance(acc, "uploaded")
            lc.advance(acc, "remote_verified")
            lc.advance(acc, "complete")
            n_complete += 1
        if cleanup and lc.reached(lc.load_state(acc), "complete"):
            for name in ("contigs.fa.zst", "orfs.tsv.zst", "contigs.ge120.fa"):
                f = lc.work_dir(acc) / name
                if f.exists() and not DRY:
                    f.unlink()
    print(f"{n_complete:,} accession(s) newly complete")
    if (cleanup or cleanup_cache) and not DRY:
        for p in sorted((lc.root() / "manifests" / "chunks").glob("*.json")):
            m = json.load(open(p))
            if m["state"] != "remote_verified":
                continue
            if cleanup and m.get("bundle"):
                lc.bundle_path(m["chunk_id"]).unlink(missing_ok=True)
            if cleanup_cache:
                lc.cache_path(m["chunk_id"]).unlink(missing_ok=True)


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
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")   # concurrent upload jobs
    pd.DataFrame(rows).to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, out)
    return out


def main() -> int:
    global DRY
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", help="limit to accessions of this batch (default: all)")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--cleanup-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--part", help="i/k: only chunks whose number %% k == i")
    args = ap.parse_args()
    DRY = args.dry_run

    if args.batch:
        accs = pd.read_csv(lc.root() / "manifests" / f"{args.batch}_accessions.tsv",
                           sep="\t").accession.tolist()
    else:
        accs = [p.stem for p in sorted((lc.root() / "state").glob("*.json"))]
    chunk_ids = sorted({c for a in accs for c in lc.load_state(a).get("info", {}).get("chunks", [])})
    if args.part:
        # Disjoint subsets for concurrent upload jobs: phaseb1 measured ~47 GB/h
        # for one job, limited by verification and listing, not bandwidth.
        i, k = map(int, args.part.split("/"))
        chunk_ids = [c for c in chunk_ids if lc.chunk_number(c) % k == i]
    do_chunks(chunk_ids)
    do_accessions(accs, args.cleanup, args.cleanup_cache)
    out = status_table()
    if not DRY:
        mdir = lc.root() / "manifests"
        verify_folder(mdir, f"{lc.REMOTE_ROOT}/manifests",
                      [out.name] + sorted(p.name for p in mdir.glob("*_accessions.tsv")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
