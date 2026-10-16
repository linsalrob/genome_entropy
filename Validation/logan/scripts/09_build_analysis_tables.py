#!/usr/bin/env python3
"""One analysis table per shard: Prodigal labels + coordinates + cheap features, no sequences.

For every cache chunk in a shard (<= 1,000 chunks), read the encoded rows,
take the matching rows of each member accession's orf_prodigal.tsv.zst
(sliced by first_row / n_rows, the same order the chunk was built in), check
orf_id row for row, compute

  aa_len, aa_entropy, three_di_entropy, twelve_state_entropy, mi_3di_12st
  (stored quantised: *_q4 = round(value * 1e4) as uint16)

and stream the result into analysis/v1/orfs/<shard>.parquet, one row group
per chunk. Columns are compact: contig as its integer suffix (contig_n), no
orf_id (= "<accession>_<contig_n>:<start>-<end>:<strand>"), no end
(= start + 3 * (aa_len + stop)), features as uint16 in units of 1e-4 bits
(columns *_q4; divide by 1e4), low-cardinality strings as dictionaries. The Prodigal genes of every accession whose first row is in the
shard go to analysis/v1/genes/<shard>.parquet, and each ORF row carries its
matched gene's `partial` flag, so "complete" can be defined without a join.

ENTROPIES are computed vectorised (bincount over row x symbol) and must equal
genome_entropy.entropy.shannon.calculate_sequence_entropy / mutual_information:
--selftest checks this against the library on real rows before anything runs.

A shard is built only when all its chunks are remote_verified. Output is
written via .partial and renamed after the row count is checked against the
chunk manifests; a manifest analysis/v1/<shard>.json records counts and
sha256. Rebuilding a shard that has a manifest is skipped.

--cleanup deletes work/<acc>/ for accessions that are `complete` and whose
every chunk is recorded in its built shard's manifest, and only when none of
those shards can still change. A shard is closed when its manifest holds
every allocated chunk id in its range and no further id can fall into it:
ids are allocated in order from 1 and never reused (03_plan_chunks.py), so
that is when the highest allocated id has reached the shard's last id
(999 for shard 000, which has no id 0; 1999 for 001; ...), or when --final
says no further tranche will allocate chunks. A shard that can still change
keeps its accessions' work directories, because a rebuild re-reads their
prodigal_genes.tsv.zst. The same tables also travel,
verified, in the remote per-chunk bundles (and prodigal/v1/<acc>/ for the
pilot).

  09_build_analysis_tables.py --selftest
  09_build_analysis_tables.py --shards 000 001 --procs 16
  09_build_analysis_tables.py --all --procs 16 [--cleanup [--final]]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

OUT = lambda: lc.root() / "analysis" / lc.CACHE_VERSION  # noqa: E731
LABEL_COLS = ["orf_id", "five_open", "aa_len", "prodigal", "prodigal_class", "match_gene",
              "initiator_substituted", "best_gene", "best_relation", "best_overlap_bp",
              "same_strand_overlap_bp", "opp_strand_overlap_bp", "best_gene_transl_table"]
THREEDI = "ACDEFGHIKLMNPQRSTVWY"
TWELVE = "ABCDEFGHIJKL"


# --------------------------------------------------------------------------
# vectorised entropies
# --------------------------------------------------------------------------

def _codes(strings: pd.Series):
    """Concatenate strings into one uint8 array plus row ids."""
    s = strings.str.upper()
    lens = s.str.len().to_numpy(np.int64)
    buf = np.frombuffer("".join(s.tolist()).encode("ascii"), dtype=np.uint8)
    rows = np.repeat(np.arange(len(lens), dtype=np.int64), lens)
    return buf, rows, lens


def shannon_vec(strings: pd.Series, batch: int = 50_000) -> np.ndarray:
    out = np.zeros(len(strings))
    for b0 in range(0, len(strings), batch):
        buf, rows, lens = _codes(strings.iloc[b0:b0 + batch])
        if not len(buf):
            continue
        cnt = np.bincount(rows * 128 + buf, minlength=len(lens) * 128).reshape(len(lens), 128)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = cnt / lens[:, None]
            h = -np.where(cnt > 0, p * np.log2(np.where(p > 0, p, 1)), 0.0).sum(axis=1)
        h[lens == 0] = 0.0
        out[b0:b0 + batch] = h
    return out


def mi_vec(a: pd.Series, b: pd.Series, batch: int = 25_000) -> np.ndarray:
    """Empirical MI (bits) between aligned 3Di and 12-state strings."""
    ia = np.full(128, -1, np.int64)
    ia[np.frombuffer(THREEDI.encode(), np.uint8)] = np.arange(20)
    ib = np.full(128, -1, np.int64)
    ib[np.frombuffer(TWELVE.encode(), np.uint8)] = np.arange(12)
    out = np.zeros(len(a))
    for b0 in range(0, len(a), batch):
        ba, rows, lens = _codes(a.iloc[b0:b0 + batch])
        bb, rows_b, lens_b = _codes(b.iloc[b0:b0 + batch])
        if not np.array_equal(lens, lens_b):
            raise ValueError("3Di and 12-state lengths differ")
        if not len(ba):
            continue
        xa, xb = ia[ba], ib[bb]
        if (xa < 0).any() or (xb < 0).any():
            raise ValueError("symbol outside the 3Di / 12-state alphabet")
        n = len(lens)
        joint = np.bincount(rows * 240 + xa * 12 + xb, minlength=n * 240).reshape(n, 20, 12)
        pa = joint.sum(axis=2) / lens[:, None]
        pb = joint.sum(axis=1) / lens[:, None]
        pj = joint / lens[:, None, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(joint > 0, pj * np.log2(pj / (pa[:, :, None] * pb[:, None, :])), 0.0)
        mi = term.sum(axis=(1, 2))
        mi[(mi < 0) & (mi > -1e-12)] = 0.0
        mi[lens == 0] = 0.0
        out[b0:b0 + batch] = mi
    return out


def selftest() -> int:
    from genome_entropy.entropy.shannon import calculate_sequence_entropy, mutual_information
    m = json.load(open(lc.chunk_manifest_path("logan_metagenome_000001")))
    df = pd.read_csv(cache_file(m), sep="\t", nrows=20000, dtype=str, keep_default_na=False,
                     compression={"method": "zstd"})
    worst = 0.0
    for col in ("aa_sequence", "three_di", "twelve_state"):
        v = shannon_vec(df[col])
        ref = np.array([calculate_sequence_entropy(x) for x in df[col]])
        worst = max(worst, np.abs(v - ref).max())
    v = mi_vec(df.three_di, df.twelve_state)
    ref = np.array([mutual_information(x, y) for x, y in zip(df.three_di, df.twelve_state)])
    worst = max(worst, np.abs(v - ref).max())
    print(f"selftest on {len(df):,} rows: max |vectorised - genome_entropy| = {worst:.3e}")
    return 0 if worst < 1e-9 else 1


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

def cache_file(m: dict) -> Path:
    return lc.root() / "cache" / "modernprost-50M" / lc.CACHE_VERSION / m["cache_file"]


def acc_batch_biome(acc: str):
    i = lc.load_state(acc).get("info", {})
    return i.get("batch", ""), i.get("biome", "")


def label_slice(acc: str, first: int, n: int) -> pd.DataFrame:
    """Rows [first, first+n) of an accession's orf_prodigal, with matched-gene partial flags.

    Read in 500k-row pieces and kept only for the slice: holding whole
    tables (5 M rows of strings for a large accession) in every worker
    exhausted 115 GB twice."""
    wd = lc.work_dir(acc)
    parts, seen = [], 0
    with pd.read_csv(wd / "orf_prodigal.tsv.zst", sep="\t", keep_default_na=False, usecols=LABEL_COLS,
                     dtype={"match_gene": str, "best_gene": str, "best_relation": str},
                     chunksize=500_000) as it:
        for piece in it:
            lo, hi = max(first - seen, 0), min(first + n - seen, len(piece))
            if lo < hi:
                parts.append(piece.iloc[lo:hi])
            seen += len(piece)
            if seen >= first + n:
                break
    t = pd.concat(parts, ignore_index=True)
    want = set(t.match_gene) - {""}
    part = {}
    if want:
        g = pd.read_csv(wd / "prodigal_genes.tsv.zst", sep="\t", keep_default_na=False,
                        dtype={"partial": str, "gene_id": str}, usecols=["gene_id", "partial"])
        g = g[g.gene_id.isin(want)]
        part = dict(zip(g.gene_id, g.partial))
    t["match_gene_partial"] = t.match_gene.map(part).fillna("")
    return t


def chunk_table(m: dict) -> pd.DataFrame:
    df = pd.read_csv(cache_file(m), sep="\t", dtype={"contig": str, "strand": str}, keep_default_na=False,
                     compression={"method": "zstd"})
    if len(df) != m["orf_count"]:
        raise RuntimeError(f"{m['chunk_id']}: {len(df)} rows, manifest {m['orf_count']}")
    parts = []
    for x in m["members"]:
        parts.append(label_slice(x["accession"], x["first_row"], x["n_rows"]))
    lab = pd.concat(parts, ignore_index=True)
    if not np.array_equal(lab.orf_id.to_numpy(), df.orf_id.to_numpy()):
        raise RuntimeError(f"{m['chunk_id']}: orf_id order differs between cache and labels")
    out = df.drop(columns=["aa_sequence", "three_di", "twelve_state"])
    for c in LABEL_COLS[1:] + ["match_gene_partial"]:
        out[c] = lab[c].to_numpy()
    if not np.array_equal(out.aa_len.to_numpy(), df.aa_sequence.str.len().to_numpy()):
        raise RuntimeError(f"{m['chunk_id']}: aa_len differs from cached sequence length")
    out["aa_entropy"] = shannon_vec(df.aa_sequence)
    out["three_di_entropy"] = shannon_vec(df.three_di)
    out["twelve_state_entropy"] = shannon_vec(df.twelve_state)
    out["mi_3di_12st"] = mi_vec(df.three_di, df.twelve_state)
    bb = {a: acc_batch_biome(a) for a in out.accession.unique()}
    out["batch"] = out.accession.map(lambda a: bb[a][0])
    out["biome"] = out.accession.map(lambda a: bb[a][1])
    out["chunk"] = lc.chunk_number(m["chunk_id"])
    # Compact schema. orf_id is exactly "<contig>:<start>-<end>:<strand>" and
    # contig is "<accession>_<n>", so both are stored as the integer n: the
    # string columns made a 1,000-chunk shard ~55 GB, as large as the cache.
    pref = out.accession + "_"
    if not (out.contig.str.startswith(pref.iloc[0]) if out.accession.nunique() == 1
            else pd.Series([c.startswith(p) for c, p in zip(out.contig, pref)])).all():
        raise RuntimeError(f"{m['chunk_id']}: contig names do not follow <accession>_<n>")
    out["contig_n"] = out.contig.str.rsplit("_", n=1).str[1].astype("int64")
    rebuilt = (out.contig + ":" + out.start.astype(str) + "-" + out.end.astype(str) + ":" + out.strand)
    if not np.array_equal(rebuilt.to_numpy(), df.orf_id.to_numpy()):
        raise RuntimeError(f"{m['chunk_id']}: orf_id is not reconstructible from contig/start/end/strand")
    # end is derivable: a codon-aligned ORF spans 3 * (aa_len + stop) nt.
    if not (out.end - out.start == 3 * (out.aa_len + out.stop)).all():
        raise RuntimeError(f"{m['chunk_id']}: end != start + 3 * (aa_len + stop)")
    out = out.drop(columns=["contig", "orf_id", "end"])
    # Features quantised to 1e-4 bits (max error 5e-5) as uint16: float64
    # features were 28 of 33 MB per million rows.
    for f in ("aa_entropy", "three_di_entropy", "twelve_state_entropy", "mi_3di_12st"):
        out[f + "_q4"] = np.round(out.pop(f).to_numpy() * 1e4).astype("uint16")
    for c, d in (("start", "int32"), ("contig_len", "int32"), ("aa_len", "int32"),
                 ("best_overlap_bp", "int32"), ("same_strand_overlap_bp", "int32"),
                 ("opp_strand_overlap_bp", "int32"), ("stop", "int8"), ("five_open", "int8"),
                 ("initiator_substituted", "int8"), ("best_gene_transl_table", "int8"),
                 ("chunk", "int32")):
        out[c] = out[c].astype(d)
    return out


def shards_from_manifests():
    by = defaultdict(list)
    for p in sorted((lc.root() / "manifests" / "chunks").glob("*.json")):
        m = json.load(open(p))
        by[lc.chunk_shard(m["chunk_id"])].append(m)
    return by


def build_shard(shard: str, mans: list, procs: int) -> bool:
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_dir = OUT()
    man_path = out_dir / f"{shard}.json"
    if man_path.exists():
        # Built already, but only for the chunks it recorded: a later tranche
        # may add chunks to the same 1,000-id shard, and the shard must then be
        # rebuilt rather than permanently omit them.
        built = set(json.load(open(man_path)).get("chunks", []))
        if built == {m["chunk_id"] for m in mans}:
            print(f"shard {shard}: built already")
            return True
        print(f"shard {shard}: chunk set changed ({len(built)} -> {len(mans)}); rebuilding")
    not_ready = [m["chunk_id"] for m in mans if m["state"] != "remote_verified"]
    if not_ready:
        print(f"shard {shard}: {len(not_ready)} chunk(s) not remote_verified yet; skipped")
        return False
    (out_dir / "orfs").mkdir(parents=True, exist_ok=True)
    (out_dir / "genes").mkdir(parents=True, exist_ok=True)
    orf_path = out_dir / "orfs" / f"{shard}.parquet"
    gene_path = out_dir / "genes" / f"{shard}.parquet"
    tmp_o, tmp_g = orf_path.with_suffix(".parquet.partial"), gene_path.with_suffix(".parquet.partial")

    mans = sorted(mans, key=lambda m: m["chunk_id"])
    writer, rows, accs = None, 0, set()

    def bounded(fn, items, procs):
        """Ordered results with at most 2*procs chunks in flight.

        Executor.map submits everything at once, and finished tables (~200 MB
        each) pile up faster than the single writer drains them: the first
        shard-000 build ran out of memory in 2 minutes that way."""
        if procs <= 1:
            yield from map(fn, items)
            return
        from collections import deque
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(procs) as ex:
            q, it = deque(), iter(items)
            for x in it:
                q.append(ex.submit(fn, x))
                if len(q) >= procs + 2:
                    break
            while q:
                yield q.popleft().result()
                for x in it:
                    q.append(ex.submit(fn, x))
                    break

    for m, t in zip(mans, bounded(chunk_table, mans, procs)):
        for c in ("accession", "strand", "prodigal", "prodigal_class", "best_relation",
                  "match_gene_partial", "batch", "biome"):
            t[c] = t[c].astype("category")
        tbl = pa.Table.from_pandas(t, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_o, tbl.schema, compression="zstd", compression_level=6)
        else:
            tbl = tbl.cast(writer.schema)
        writer.write_table(tbl, row_group_size=len(t))
        rows += len(t)
        accs.update(t.accession.unique())
    writer.close()
    want = sum(m["orf_count"] for m in mans)
    got = pq.ParquetFile(tmp_o).metadata.num_rows
    if got != want or rows != want:
        raise RuntimeError(f"shard {shard}: wrote {got} rows, manifests say {want}")

    # Genes of accessions whose first row is in this shard.
    starters = sorted({x["accession"] for m in mans for x in m["members"] if x["first_row"] == 0})
    gw, n_genes = None, 0
    for a in starters:
        g = pd.read_csv(lc.work_dir(a) / "prodigal_genes.tsv.zst", sep="\t", keep_default_na=False,
                        dtype={"partial": str, "contig": str, "gene_id": str, "start_type": str,
                               "rbs_motif": str, "model": str, "matched_orf": str})
        if len(g) != int(lc.load_state(a)["info"]["prodigal_genes"]):
            raise RuntimeError(f"{a}: prodigal_genes rows differ from state")
        g["accession"] = a
        tbl = pa.Table.from_pandas(g, preserve_index=False)
        if gw is None:
            gw = pq.ParquetWriter(tmp_g, tbl.schema, compression="zstd")
        elif tbl.schema != gw.schema:
            tbl = tbl.cast(gw.schema)
        gw.write_table(tbl)
        n_genes += len(g)
    if gw is not None:
        gw.close()
        os.replace(tmp_g, gene_path)
    os.replace(tmp_o, orf_path)
    lc.atomic_write_json(man_path, {
        "shard": shard, "built": lc.now(), "chunks": [m["chunk_id"] for m in mans],
        "orf_rows": got, "accessions": len(accs), "gene_accessions": len(starters),
        "gene_rows": n_genes, "orfs_sha256": lc.sha256(orf_path),
        "genes_sha256": lc.sha256(gene_path) if gw is not None else None,
        "feature_definition": "genome_entropy.entropy.shannon.calculate_sequence_entropy / "
                              "mutual_information, vectorised; checked by --selftest"})
    print(f"shard {shard}: {got:,} ORFs, {len(accs):,} accessions, {n_genes:,} genes")
    return True


def shard_closed(shard: str, built: dict, allocated: dict, max_id: int, final: bool) -> bool:
    """Whether a shard's table can no longer change (see the module docstring)."""
    if built.get(shard, set()) != allocated.get(shard, set()):
        return False                    # allocated chunks not yet in the table
    return final or max_id >= (int(shard) + 1) * 1000 - 1


def cleanup(built: dict, allocated: dict, final: bool) -> None:
    """built / allocated: shard -> chunk ids in its analysis manifest / on disk."""
    max_id = max((lc.chunk_number(c) for ids in allocated.values() for c in ids), default=0)
    freed = 0
    for sp in sorted((lc.root() / "state").glob("*.json")):
        st = json.load(open(sp))
        if st.get("state") != "complete":
            continue
        ids = st.get("info", {}).get("chunks", [])
        if not ids or any(c not in built.get(lc.chunk_shard(c), ()) for c in ids):
            continue
        if not all(shard_closed(lc.chunk_shard(c), built, allocated, max_id, final) for c in ids):
            continue
        wd = lc.work_dir(st["accession"])
        if wd.exists():
            shutil.rmtree(wd)
            freed += 1
            lc.update_info(st["accession"], work_dir_removed=lc.now())
    print(f"cleanup: removed {freed:,} work directories")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shards", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="no further tranche will allocate chunks: let --cleanup "
                         "also clear accessions in the last, partly filled shard")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    by = shards_from_manifests()
    shards = sorted(by) if args.all else args.shards
    for s in shards:
        build_shard(s, by[s], args.procs)
    if args.cleanup:
        built = {p.stem: set(json.load(open(p)).get("chunks", []))
                 for p in OUT().glob("*.json")}
        allocated = {s: {m["chunk_id"] for m in ms} for s, ms in by.items()}
        cleanup(built, allocated, args.final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
