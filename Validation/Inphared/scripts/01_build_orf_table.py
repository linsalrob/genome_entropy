#!/usr/bin/env python3
"""Flatten the 312 Inphared genome_entropy JSONs and their GenBank files into three tables.

Inputs (one pair per part, matched by name):
  <json-dir>/phold_NNN.json      genome_entropy output (schema 2.1.0)
  <gbk-dir>/phold_NNN.gbk.gz     the Pharokka/phold GenBank the JSON was built from

Outputs (in --out):
  orfs.parquet   one row per ORF, in the SAME order rank_missing_orfs.py reads
                 them (sorted file name, record order, feature order), so its
                 `row` column aligns with the original all_orf_oof_predictions.
                 Columns: model features, genomic interval (0-based half-open,
                 via the package's normalise_orf_interval), genomic phase, the
                 contig-edge flag and the ORF's context relative to the
                 annotated CDS (see classify()).
  seqs.parquet   row, aa, three_di (ModernProst-50M argmax 3Di), for Foldseek.
  cds.parquet    every CDS feature in the GenBank: interval, phase, product,
                 function, PHROG, partial flag and caller (`inference`).

Context classes for an ORF (relative to annotated CDS on the genomic axis):
  same_stop    shares its stop codon with a CDS on the same strand: this ORF IS
               that gene's stop-to-stop ORF (an unannotated one is a label miss)
  same_frame   overlaps a same-strand CDS in the same codon phase, other stop
  frameshift   overlaps a same-strand CDS in a different phase (frame shadow)
  antisense    overlaps a CDS only on the opposite strand
  intergenic   overlaps no CDS
same_stop wins over everything; otherwise the CDS with the largest nucleotide
overlap decides. Phase uses the strand-dependent anchor (start for +, end for
-), the lesson from Validation/gtdb #92.

  01_build_orf_table.py --json-dir genbank_split_json --gbk-dir genbank_split \
      --out tables --threads 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from genome_entropy.io.genbank import normalise_orf_interval  # noqa: E402


def parse_gbk(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt") as fh:
        for rec in SeqIO.parse(fh, "genbank"):
            L = len(rec.seq)
            for f in rec.features:
                if f.type != "CDS":
                    continue
                q = f.qualifiers
                loc = f.location
                compound = len(getattr(loc, "parts", [loc])) > 1
                s, e = int(loc.start), int(loc.end)
                strand = "+" if (loc.strand or 1) > 0 else "-"
                tr = q.get("translation", [""])[0]
                rows.append({
                    "input_id": rec.id,
                    "record_length": L,
                    "cds_id": q.get("locus_tag", q.get("ID", [""]))[0],
                    "g_start": s, "g_end": e, "strand": strand,
                    "compound": compound,
                    "partial": q.get("partial", [""])[0],
                    "product": q.get("product", [""])[0],
                    "function": q.get("function", [""])[0],
                    "phrog": q.get("phrog", [""])[0],
                    "inference": q.get("inference", [""])[0],
                    "transl_table": int(q.get("transl_table", ["11"])[0]),
                    "aa_len": len(tr.rstrip("*")),
                })
    return pd.DataFrame(rows)


def phase(gs, ge, strand):
    return np.where(strand == "+", gs % 3, ge % 3)


CLASS_ORDER = ["same_stop", "same_frame", "frameshift", "antisense", "intergenic"]


def classify(orf: pd.DataFrame, cds: pd.DataFrame) -> pd.DataFrame:
    """Context class, max single-CDS overlap and the CDS that decided it."""
    n = len(orf)
    cls = np.full(n, "intergenic", dtype=object)
    ovl = np.zeros(n)
    ovl_cds = np.full(n, "", dtype=object)
    if len(cds):
        os_, oe = orf.g_start.to_numpy()[:, None], orf.g_end.to_numpy()[:, None]
        cs, ce = cds.g_start.to_numpy()[None, :], cds.g_end.to_numpy()[None, :]
        nt = np.clip(np.minimum(oe, ce) - np.maximum(os_, cs), 0, None)
        same_strand = orf.strand.to_numpy()[:, None] == cds.strand.to_numpy()[None, :]
        o_plus = (orf.strand.to_numpy() == "+")[:, None]
        same_stop = same_strand & np.where(o_plus, oe == ce, os_ == cs)
        same_phase = same_strand & (orf.phase.to_numpy()[:, None] == cds.phase.to_numpy()[None, :])
        j = nt.argmax(axis=1)
        best = nt[np.arange(n), j]
        olen = (orf.g_end - orf.g_start).to_numpy()
        ovl = np.where(olen > 0, best / np.maximum(olen, 1), 0.0)
        ids = cds.cds_id.to_numpy()
        ovl_cds = np.where(best > 0, ids[j], "")
        r = np.arange(n)
        c = np.where(best == 0, "intergenic",
             np.where(same_phase[r, j], "same_frame",
              np.where(same_strand[r, j], "frameshift", "antisense")))
        has_stop = same_stop.any(axis=1)
        k = same_stop.argmax(axis=1)
        cls = np.where(has_stop, "same_stop", c)
        ovl_cds = np.where(has_stop, ids[k], ovl_cds)
        ovl = np.where(has_stop, nt[r, k] / np.maximum(olen, 1), ovl)
    return pd.DataFrame({"context": cls, "cds_overlap_frac": ovl, "context_cds": ovl_cds})


def part_name(path: Path) -> str:
    """phold_NNN.json or phold_NNNN.json.gz -> phold_NNN(N)."""
    return path.name.removesuffix(".gz").removesuffix(".json")


def load_json(path: Path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as fh:
        return json.load(fh)


def process(args):
    json_path, gbk_path = args
    part = part_name(json_path)
    cds = parse_gbk(gbk_path)
    cds["part"] = part
    cds["phase"] = phase(cds.g_start.to_numpy(), cds.g_end.to_numpy(), cds.strand.to_numpy())
    data = load_json(json_path)
    orf_rows, seq_rows = [], []
    for rec in data:
        feats = rec.get("features")
        if not isinstance(feats, dict):
            continue
        iid = str(rec.get("input_id", ""))
        L = rec.get("input_dna_length")
        for oid, f in feats.items():
            loc, ent, md = f["location"], f["entropy"], f["metadata"]
            ts = f.get("twelve_state") or {}
            s, e, st = int(loc["start"]), int(loc["end"]), str(loc["strand"])
            edge = e > L  # get_orfs end = L+1 sentinel: ran off the contig, no stop
            try:
                iv = normalise_orf_interval(s, min(e, L), st, L)
                gs, ge = iv.start, iv.end
            except ValueError:
                gs = ge = -1
            orf_rows.append({
                "part": part, "input_id": iid, "orf_id": oid, "record_length": L,
                "start": s, "end": e, "strand": st, "frame": int(loc["frame"]),
                "g_start": gs, "g_end": ge, "contig_edge": edge,
                "dna_entropy": ent.get("dna_entropy"), "protein_entropy": ent.get("protein_entropy"),
                "three_di_entropy": ent.get("three_di_entropy"),
                "twelve_state_entropy": ent.get("twelve_state_entropy"),
                "dna_length": f["dna"]["length"], "protein_length": f["protein"]["length"],
                "three_di_length": f["three_di"]["length"], "twelve_state_length": ts.get("length"),
                "has_start_codon": bool(md["has_start_codon"]), "has_stop_codon": bool(md["has_stop_codon"]),
                "in_genbank": bool(md["in_genbank"]),
                "model_name": f["three_di"].get("model_name"),
            })
            seq_rows.append((f["protein"]["aa_sequence"], f["three_di"]["encoding"]))
    orf = pd.DataFrame(orf_rows)
    orf["phase"] = phase(orf.g_start.to_numpy(), orf.g_end.to_numpy(), orf.strand.to_numpy())
    ctx = []
    cds_by = dict(tuple(cds.groupby("input_id"))) if len(cds) else {}
    for iid, g in orf.groupby("input_id", sort=False):
        c = classify(g, cds_by.get(iid, cds.iloc[0:0]))
        c.index = g.index
        ctx.append(c)
    orf = orf.join(pd.concat(ctx))
    seqs = pd.DataFrame(seq_rows, columns=["aa", "three_di"])
    missing_gbk = sorted(set(orf.input_id) - set(cds.input_id))
    return part, orf, seqs, cds, missing_gbk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--gbk-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    # Same ordering as rank_missing_orfs.discover_json_files (sorted paths).
    jsons = sorted([*a.json_dir.glob("*.json"), *a.json_dir.glob("*.json.gz")])
    jobs = [(j, a.gbk_dir / f"{part_name(j)}.gbk.gz") for j in jsons]
    missing = [str(g) for _, g in jobs if not g.exists()]
    if missing:
        raise SystemExit(f"missing GenBank files: {missing[:5]} ...")
    writers = {}
    row0 = 0
    n_missing_gbk = 0
    with Pool(a.threads) as pool:
        for part, orf, seqs, cds, miss in pool.imap(process, jobs):  # imap keeps order
            orf.insert(0, "row", np.arange(row0, row0 + len(orf)))
            seqs.insert(0, "row", orf["row"].to_numpy())
            row0 += len(orf)
            n_missing_gbk += len(miss)
            for name, df in (("orfs", orf), ("seqs", seqs), ("cds", cds)):
                t = pa.Table.from_pandas(df, preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(a.out / f"{name}.parquet", t.schema, compression="zstd")
                writers[name].write_table(t.cast(writers[name].schema))
            print(f"{part}: {len(orf):,} ORFs, {len(cds):,} CDS; total {row0:,}", flush=True)
    for w in writers.values():
        w.close()
    print(f"done: {row0:,} ORFs; {n_missing_gbk} ORF records without a GenBank CDS record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
