#!/usr/bin/env python3
"""Missed-gene candidates: structural and sequence evidence by arm (after #92's 18_pilot_analysis).

Reads qc/candidates/search/{best,fullcov}/*.m8 (23_foldseek_search.slurm) and
selection.parquet. For each arm (candidate, intergenic_lo, antisense,
prodigal) and the shuffled-3Di null of the candidates:

  hit_rate          any hit at e <= 1e-3 (loose; NOT the primary readout:
                    #92 showed shadows with highly structured 3Di also hit)
  fullcov_rate      a hit with qcov >= 0.8 AND tcov >= 0.8 (primary)

per target database and over all databases, for structure (3Di+AA) and
sequence-only searches. Every query is then classed, over all databases, by
full-length evidence as sequence+structure / structure-only /
sequence-only / neither. Rates are also given by length bin.

  24_candidates_evidence.py --dir qc/candidates
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits",
          "taxid", "taxname"]
LEN_EDGES = [100, 150, 200, 300, 500, 1000, 10**9]


def load(dirpath: Path, kind: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(dirpath / kind / "*.m8"))):
        tag, db, mode = Path(f).name[:-3].split(".")
        d = pd.read_csv(f, sep="\t", names=FIELDS, usecols=["query", "target", "qcov", "tcov", "evalue", "bits"],
                        dtype={"query": str, "target": str})
        d["qtag"], d["db"], d["mode"] = tag, db, mode
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=FIELDS + ["qtag", "db", "mode"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    D = Path(args.dir)
    S = D / "search"
    sel = pd.read_parquet(D / "selection.parquet")
    sel["length_bin"] = pd.cut(sel.aa_len, LEN_EDGES, right=False).astype(str)
    best, full = load(S, "best"), load(S, "fullcov")
    dbs = sorted(set(best.db))
    print(f"databases: {dbs}; {len(sel):,} queries")

    # Universe of queries per arm, plus the null (candidate arm, shuffled 3Di).
    univ = sel[["query_id", "arm", "length_bin", "aa_len", "trunc", "score"]].copy()
    null = univ[univ.arm == "candidate"].assign(arm="candidate_shuffled_3di")
    univ = pd.concat([univ, null], ignore_index=True)

    def hits(df, qtag, mode, db=None):
        x = df[(df.qtag == qtag) & (df["mode"] == mode)]
        if db is not None:
            x = x[x.db == db]
        return set(x["query"])

    rows = []
    for arm, g in univ.groupby("arm"):
        qtag = "qnull" if arm == "candidate_shuffled_3di" else "qdb"
        q = set(g.query_id)
        for mode in (["struct"] if qtag == "qnull" else ["struct", "seq"]):
            for db in dbs + ["any"]:
                h = hits(best, qtag, mode, None if db == "any" else db) & q
                f = hits(full, qtag, mode, None if db == "any" else db) & q
                rows.append({"arm": arm, "mode": mode, "db": db, "queries": len(q),
                             "hit_rate": len(h) / len(q), "fullcov_rate": len(f) / len(q)})
    R = pd.DataFrame(rows)
    R.to_csv(D / "evidence_rates.tsv", sep="\t", index=False)

    s_full = hits(full, "qdb", "struct")
    q_full = hits(full, "qdb", "seq")
    u = univ[univ.arm != "candidate_shuffled_3di"].copy()
    u["evidence"] = np.select(
        [u.query_id.isin(s_full) & u.query_id.isin(q_full), u.query_id.isin(s_full), u.query_id.isin(q_full)],
        ["sequence+structure", "structure-only", "sequence-only"], default="neither")
    E = (u.groupby(["arm", "evidence"]).size().unstack(fill_value=0)
          .pipe(lambda x: x.div(x.sum(axis=1), axis=0)))
    E.to_csv(D / "evidence_classes.tsv", sep="\t")
    L = (u.assign(full_struct=u.query_id.isin(s_full)).groupby(["arm", "length_bin"]).full_struct
          .agg(["size", "mean"]).rename(columns={"size": "queries", "mean": "fullcov_struct_rate"}))
    L.to_csv(D / "evidence_by_length.tsv", sep="\t")
    u[["query_id", "arm", "evidence"]].to_parquet(D / "evidence_per_query.parquet", index=False)
    json.dump({"databases": dbs, "queries": int(len(sel))}, open(D / "evidence_inputs.json", "w"), indent=1)
    with pd.option_context("display.width", 220):
        print(R[R.db == "any"].round(4).to_string(index=False))
        print(E.round(4).to_string())
        print(L.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
