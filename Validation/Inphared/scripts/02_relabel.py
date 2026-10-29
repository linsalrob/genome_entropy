#!/usr/bin/env python3
"""Recompute in_genbank for the Inphared ORFs with the current CDS matcher, and write relabelled JSONs.

The 312 JSONs were made on 2026-07-28 with genome_entropy 0.1.15 or earlier.
That release's matcher compared "biological stop coordinates" without
mapping minus-strand get_orfs coordinates (which index the reverse
complement) back to the genomic axis, so essentially no minus-strand ORF was
ever labelled in_genbank. The matcher was replaced on 2026-07-31 (2cc0e1a,
released in 0.2.0) by the strand/phase/overlap/identity matcher described in
docs/source/data_formats.rst.

This script re-labels every ORF with that matcher, called exactly as the
pipeline calls it (genome_entropy.io.genbank.extract_cds_features +
match_orf_to_genbank_cds from this checkout), using the protein string stored
in the JSON. Everything else in the JSON is left untouched.

Outputs:
  <out-json>/phold_NNN.json   the input JSON with metadata.in_genbank replaced
  <labels>                    parquet: part, input_id, orf_id, strand,
                              in_genbank_orig, in_genbank (recomputed)

  02_relabel.py --json-dir genbank_split_json --gbk-dir genbank_split \
      --out-json genbank_split_json_relabelled --labels tables/labels.parquet --threads 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from genome_entropy.io.genbank import extract_cds_features, match_orf_to_genbank_cds  # noqa: E402
from genome_entropy.orf.types import OrfRecord  # noqa: E402


def process(job):
    json_path, gbk_path, out_json = job
    by_parent = defaultdict(list)
    for cds in extract_cds_features(gbk_path):
        by_parent[cds.parent_id].append(cds)
    with (gzip.open(json_path, "rt") if json_path.suffix == ".gz" else open(json_path)) as fh:
        data = json.load(fh)
    part = json_path.name.removesuffix(".gz").removesuffix(".json")
    rows = []
    for rec in data:
        feats = rec.get("features")
        if not isinstance(feats, dict):
            continue
        iid = str(rec.get("input_id", ""))
        cds_list = by_parent.get(iid, [])
        for oid, f in feats.items():
            loc, md = f["location"], f["metadata"]
            orf = OrfRecord(parent_id=iid, orf_id=oid, start=int(loc["start"]), end=int(loc["end"]),
                            strand=loc["strand"], frame=int(loc["frame"]), nt_sequence="",
                            aa_sequence=f["protein"]["aa_sequence"], table_id=int(md.get("table_id", 11)),
                            has_start_codon=bool(md["has_start_codon"]),
                            has_stop_codon=bool(md["has_stop_codon"]))
            same = [c for c in cds_list if c.strand == orf.strand]
            new = bool(match_orf_to_genbank_cds(orf, same)) if same else False
            rows.append((part, iid, oid, orf.strand, bool(md["in_genbank"]), new))
            md["in_genbank"] = new
    with open(out_json, "w") as fh:
        json.dump(data, fh)
    n_cds = sum(len(v) for v in by_parent.values())
    return pd.DataFrame(rows, columns=["part", "input_id", "orf_id", "strand", "in_genbank_orig", "in_genbank"]), n_cds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--gbk-dir", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    a.out_json.mkdir(parents=True, exist_ok=True)
    jobs = []
    for j in sorted([*a.json_dir.glob("*.json"), *a.json_dir.glob("*.json.gz")]):
        part = j.name.removesuffix(".gz").removesuffix(".json")
        jobs.append((j, a.gbk_dir / f"{part}.gbk.gz", a.out_json / f"{part}.json"))
    parts, n_cds = [], 0
    with Pool(a.threads) as pool:
        for df, nc in pool.imap(process, jobs):
            parts.append(df)
            n_cds += nc
            print(f"{df.part.iloc[0]}: {len(df):,} ORFs, orig {df.in_genbank_orig.sum():,} -> "
                  f"new {df.in_genbank.sum():,}", flush=True)
    lab = pd.concat(parts, ignore_index=True)
    lab.insert(0, "row", range(len(lab)))
    lab.to_parquet(a.labels, index=False)
    print(f"{len(lab):,} ORFs; {n_cds:,} CDS")
    print(pd.crosstab([lab.strand, lab.in_genbank_orig], lab.in_genbank))
    return 0


if __name__ == "__main__":
    sys.exit(main())
