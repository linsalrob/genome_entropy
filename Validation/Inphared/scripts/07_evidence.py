#!/usr/bin/env python3
"""Structural and sequence evidence for the Foldseek arms (after Validation/logan 24_candidates_evidence.py).

Reads <dir>/search/{best,fullcov}/*.m8 (06_foldseek_search.slurm), <dir>/selection.parquet
(05), cds.parquet (01), and the replacement tables of the re-ranked and original runs.

  hit_rate      any hit at e <= 1e-3 (loose)
  fullcov_rate  a hit with qcov >= 0.8 AND tcov >= 0.8 (primary readout, #92)

Writes to --out (small TSVs, committed) and --work (per-query detail):
  evidence_rates.tsv           arm x mode x db, with Wilson 95% CI on fullcov_rate
  evidence_excess.tsv          candidate minus each control and the mixture share
                               (cand - ctrl) / (annotated - ctrl), struct, per db
  evidence_classes.tsv         sequence+structure / structure-only / sequence-only / neither
  evidence_by_length.tsv       struct full-length rate by arm x length bin
  evidence_by_context.tsv      candidate arm by context (intergenic/antisense/frameshift)
  hit_superkingdom.tsv         superkingdom of the best full-length structural hit
  replacement_pairs.tsv        replacement ORF vs the annotated ORF it would replace
  exemplars.tsv                top candidates with full-length hits in >= 2 databases
  <work>/evidence_per_query.parquet
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
          "taxid", "taxname", "theader"]
LEN_EDGES = [90, 100, 150, 200, 300, 500, 1000, 10**9]
SK = {2: "Bacteria", 2157: "Archaea", 2759: "Eukaryota", 10239: "Viruses"}
CONTROLS = ["annotated", "antisense_lo", "frameshift_lo", "intergenic_lo", "candidate_shuffled_3di"]
# Viral structure databases: hits there mostly report that a related genome annotated the ORF.
VIRAL = {"bfvd", "phold"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(dirpath: Path, kind: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(dirpath / kind / "*.m8"))):
        tag, db, mode = Path(f).name[:-3].split(".")
        d = pd.read_csv(f, sep="\t", names=FIELDS, quoting=3, dtype={"query": str, "target": str, "theader": str})
        d["qtag"], d["db"], d["mode"] = tag, db, mode
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def superkingdoms(taxids, nodes: str) -> list[str]:
    par = {}
    with open(nodes) as fh:
        for line in fh:
            f = line.split("\t|\t")
            par[int(f[0])] = int(f[1])
    cache: dict[int, str] = {}

    def one(t: int) -> str:
        if t not in cache:
            u, name = t, "unknown"
            for _ in range(100):
                if u in SK:
                    name = SK[u]
                    break
                if u not in par or par[u] == u:
                    break
                u = par[u]
            cache[t] = name
        return cache[t]
    return [one(int(t)) if pd.notna(t) and str(t).isdigit() else "unknown" for t in taxids]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--tables", required=True, type=Path, help="work tables/ (cds.parquet, orfs.parquet)")
    ap.add_argument("--rank", required=True, type=Path)
    ap.add_argument("--orig", type=Path, help="original run (same ORF set only)")
    ap.add_argument("--nodes", default="/scratch/references/kraken_dec2025/nodes.dmp")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    S = a.dir / "search"
    sel = pd.read_parquet(a.dir / "selection.parquet")
    sel["length_bin"] = pd.cut(sel.protein_length, LEN_EDGES, right=False).astype(str)
    best, full = load(S, "best"), load(S, "fullcov")
    dbs = sorted(set(best.db))
    print(f"databases: {dbs}; {len(sel):,} queries", flush=True)

    univ = sel[["query_id", "arm", "row", "length_bin", "protein_length", "context", "p"]].copy()
    null = univ[univ.arm == "candidate"].assign(arm="candidate_shuffled_3di")
    univ = pd.concat([univ, null], ignore_index=True)

    def hits(df, qtag, mode, db=None):
        x = df[(df.qtag == qtag) & (df["mode"] == mode)]
        if db == "non_bfvd":
            x = x[x.db != "bfvd"]
        elif db == "non_viral":
            x = x[~x.db.isin(VIRAL)]
        elif db is not None:
            x = x[x.db == db]
        return set(x["query"])

    rows = []
    for arm, g in univ.groupby("arm"):
        qtag = "qnull" if arm == "candidate_shuffled_3di" else "qdb"
        q = set(g.query_id)
        for mode in (["struct"] if qtag == "qnull" else ["struct", "seq"]):
            for db in dbs + ["non_bfvd", "non_viral", "any"]:
                h = hits(best, qtag, mode, None if db == "any" else db) & q
                f = hits(full, qtag, mode, None if db == "any" else db) & q
                lo, hi = wilson(len(f), len(q))
                rows.append({"arm": arm, "mode": mode, "db": db, "queries": len(q), "hits": len(h),
                             "fullcov_hits": len(f), "hit_rate": len(h) / len(q), "fullcov_rate": len(f) / len(q),
                             "fullcov_ci_lo": lo, "fullcov_ci_hi": hi})
    R = pd.DataFrame(rows)
    R.to_csv(a.out / "evidence_rates.tsv", sep="\t", index=False, float_format="%.5f")

    # Excess over controls and the mixture share (GTDB §6.1): fraction of candidates
    # behaving like annotated genes, given control and annotated full-length rates.
    ex = []
    for mode in ("struct", "seq"):
        for db in dbs + ["non_bfvd", "non_viral", "any"]:
            r = R[(R["mode"] == mode) & (R.db == db)].set_index("arm").fullcov_rate
            if "candidate" not in r:
                continue
            for ctrl in CONTROLS:
                if ctrl not in r or ctrl == "annotated":
                    continue
                share = (r["candidate"] - r[ctrl]) / (r["annotated"] - r[ctrl]) if r["annotated"] > r[ctrl] else np.nan
                ex.append({"mode": mode, "db": db, "control": ctrl, "candidate": r["candidate"], "control_rate": r[ctrl],
                           "annotated": r["annotated"], "excess": r["candidate"] - r[ctrl], "mixture_share": share})
    pd.DataFrame(ex).to_csv(a.out / "evidence_excess.tsv", sep="\t", index=False, float_format="%.5f")

    # Identity of full-length hits: BFVD holds UniProt translations of INSDC phage
    # annotations, so a near-identical hit mostly reports that another
    # annotation (often of the same or a sibling genome) called the ORF.
    fq = full[full.qtag == "qdb"].copy()
    fq["arm"] = fq["query"].str.split("|").str[0]
    idt = (fq.groupby(["arm", "mode", "db"]).fident
             .agg(fullcov_hits="size", median_fident="median",
                  frac_fident_ge_0_9=lambda x: (x >= 0.9).mean(), frac_fident_lt_0_3=lambda x: (x < 0.3).mean()))
    idt.to_csv(a.out / "evidence_identity.tsv", sep="\t", float_format="%.4f")

    s_full, q_full = hits(full, "qdb", "struct"), hits(full, "qdb", "seq")
    s_full_nb = hits(full, "qdb", "struct", "non_bfvd")
    s_full_nv = hits(full, "qdb", "struct", "non_viral")
    s_full_bfvd_lowid = set(fq[(fq["mode"] == "struct") & (fq.db == "bfvd") & (fq.fident < 0.5)]["query"])
    s_any = hits(best, "qdb", "struct")
    u = univ[univ.arm != "candidate_shuffled_3di"].copy()
    u["full_struct"] = u.query_id.isin(s_full)
    u["full_seq"] = u.query_id.isin(q_full)
    u["any_struct"] = u.query_id.isin(s_any)
    u["full_struct_non_bfvd"] = u.query_id.isin(s_full_nb)
    u["full_struct_non_viral"] = u.query_id.isin(s_full_nv)
    u["full_struct_bfvd_fident_lt_0_5"] = u.query_id.isin(s_full_bfvd_lowid)
    u["evidence"] = np.select([u.full_struct & u.full_seq, u.full_struct, u.full_seq],
                              ["sequence+structure", "structure-only", "sequence-only"], default="neither")
    E = u.groupby(["arm", "evidence"]).size().unstack(fill_value=0)
    E = E.div(E.sum(axis=1), axis=0)
    E.to_csv(a.out / "evidence_classes.tsv", sep="\t", float_format="%.5f")
    L = (u.groupby(["arm", "length_bin"]).agg(queries=("full_struct", "size"), fullcov_struct_rate=("full_struct", "mean"),
                                             fullcov_struct_non_bfvd_rate=("full_struct_non_bfvd", "mean"),
                                             fullcov_struct_non_viral_rate=("full_struct_non_viral", "mean"),
                                             any_struct_rate=("any_struct", "mean")))
    L.to_csv(a.out / "evidence_by_length.tsv", sep="\t", float_format="%.5f")
    C = (u[u.arm == "candidate"].groupby("context").agg(queries=("full_struct", "size"),
                                                         fullcov_struct_rate=("full_struct", "mean"),
                                                         fullcov_seq_rate=("full_seq", "mean"),
                                                         any_struct_rate=("any_struct", "mean"),
                                                         fullcov_struct_non_bfvd_rate=("full_struct_non_bfvd", "mean"),
                                                         fullcov_struct_non_viral_rate=("full_struct_non_viral", "mean"),
                                                         median_length=("protein_length", "median")))
    C.to_csv(a.out / "evidence_by_context.tsv", sep="\t", float_format="%.5f")
    # full-length structural hits per database count
    nfull = (full[(full.qtag == "qdb") & (full["mode"] == "struct")].groupby("query").db.nunique())
    u["n_db_fullcov"] = u.query_id.map(nfull).fillna(0).astype(int)

    # best full-length structural hit across databases (by bits)
    fb = full[(full.qtag == "qdb") & (full["mode"] == "struct")].sort_values("bits", ascending=False)
    fb = fb.drop_duplicates("query")
    fb["superkingdom"] = superkingdoms(fb.taxid, a.nodes)
    u = u.merge(fb[["query", "db", "target", "theader", "taxname", "superkingdom", "evalue", "bits", "fident",
                    "qcov", "tcov"]].rename(columns={"query": "query_id", "db": "best_db"}), on="query_id", how="left")
    sk = pd.crosstab(u[u.full_struct].arm, u[u.full_struct].superkingdom)
    sk.div(sk.sum(axis=1), axis=0).to_csv(a.out / "hit_superkingdom.tsv", sep="\t", float_format="%.4f")
    # BFVD-only share among full-length structural hits
    per_db = full[(full.qtag == "qdb") & (full["mode"] == "struct")].groupby("query").db.agg(lambda x: set(x))
    u["bfvd_only"] = u.query_id.map(per_db).apply(lambda s: isinstance(s, set) and s == {"bfvd"})
    u["viral_only"] = u.query_id.map(per_db).apply(lambda s: isinstance(s, set) and bool(s) and s <= VIRAL)
    u.to_parquet(a.dir / "evidence_per_query.parquet", index=False)

    # Replacement pairs: does the replacement ORF earn better structural evidence than the gene it replaces?
    bb = best[(best.qtag == "qdb") & (best["mode"] == "struct")].groupby("query").bits.max()
    o = pd.read_parquet(a.tables / "orfs.parquet", columns=["row", "input_id", "orf_id"])
    key = o.set_index(["input_id", "orf_id"]).row
    prow = []
    for label, d, ra, rd in (("reranked", a.rank, "replacement", "replaced"),
                             ("original", a.orig, "orig_replacement", "orig_replaced")):
        if d is None:
            continue
        rep = pd.read_csv(d / "ranked_overlapping_gene_replacements.tsv.gz", sep="\t",
                          usecols=["input_id", "orf_id", "competing_orf_id"])
        if rep.empty:
            continue
        r_row = key.loc[list(zip(rep.input_id, rep.orf_id))].to_numpy()
        c_row = key.loc[list(zip(rep.input_id, rep.competing_orf_id))].to_numpy()
        qa = [f"{ra}|{x}" for x in r_row]
        qc = [f"{rd}|{x}" for x in c_row]
        df = pd.DataFrame({"run": label, "input_id": rep.input_id, "replacement_row": r_row, "replaced_row": c_row,
                           "replacement_full": [q in s_full for q in qa], "replaced_full": [q in s_full for q in qc],
                           "replacement_bits": [bb.get(q, 0.0) for q in qa], "replaced_bits": [bb.get(q, 0.0) for q in qc]})
        prow.append(df)
    P = pd.concat(prow, ignore_index=True) if prow else pd.DataFrame()
    if len(P):
        P["outcome"] = np.select([P.replacement_full & ~P.replaced_full, ~P.replacement_full & P.replaced_full,
                                  P.replacement_full & P.replaced_full],
                                 ["replacement only", "replaced only", "both"], default="neither")
        P.to_csv(a.dir / "replacement_pairs.tsv", sep="\t", index=False)
        summ = P.groupby(["run", "outcome"]).size().unstack(fill_value=0)
        summ["pairs"] = summ.sum(axis=1)
        summ["replacement_higher_bits"] = P.groupby("run").apply(lambda x: (x.replacement_bits > x.replaced_bits).mean())
        summ.to_csv(a.out / "replacement_pairs_summary.tsv", sep="\t", float_format="%.4f")
        print(summ.to_string())

    # Exemplars: candidate arm, full-length structural hits in >= 2 databases, highest p.
    cds = pd.read_parquet(a.tables / "cds.parquet", columns=["input_id", "cds_id", "g_start", "g_end", "strand",
                                                              "product", "function"])
    orfs = pd.read_parquet(a.tables / "orfs.parquet", columns=["row", "input_id", "orf_id", "g_start", "g_end",
                                                                "strand", "record_length", "three_di_entropy",
                                                                "protein_entropy"]).set_index("row")
    # Prefer independent evidence: full-length structural hits in >= 2 databases,
    # at least one outside the viral databases (BFVD, phold).
    ex = u[(u.arm == "candidate") & (u.n_db_fullcov >= 2) & u.full_struct_non_viral].sort_values(
        ["p", "bits"], ascending=False)
    # one exemplar per best target and per genome interval signature: Inphared holds
    # near-duplicate genomes (e.g. RefSeq NC_ copies of GenBank records)
    ex = ex.drop_duplicates("target").drop_duplicates(["protein_length", "theader"])
    ex_rows = []
    by_db = full[(full.qtag == "qdb") & (full["mode"] == "struct")].sort_values("bits", ascending=False).drop_duplicates(
        ["query", "db"]).set_index(["query", "db"])
    for ctx in ("intergenic", "antisense", "frameshift"):
        for _, r in ex[ex.context == ctx].head(5 if ctx == "intergenic" else 3).iterrows():
            orf = orfs.loc[r.row]
            g = cds[cds.input_id == orf.input_id].sort_values("g_start")
            left = g[g.g_end <= orf.g_start].tail(1)
            right = g[g.g_start >= orf.g_end].head(1)
            ov = g[(g.g_start < orf.g_end) & (g.g_end > orf.g_start)]
            rec = {"context": ctx, "input_id": orf.input_id, "orf_id": orf.orf_id,
                   "interval": f"{orf.g_start + 1}-{orf.g_end}({orf.strand})", "genome_bp": int(orf.record_length),
                   "aa": int(r.protein_length), "p": round(r.p, 3), "three_di_entropy": round(orf.three_di_entropy, 2),
                   "full_seq_hit": bool(r.full_seq), "n_db_fullcov": int(r.n_db_fullcov),
                   "left_cds": "; ".join(f"{x.product} ({x.strand})" for x in left.itertuples()),
                   "overlapping_cds": "; ".join(f"{x.product} ({x.strand}, {x.g_start + 1}-{x.g_end})" for x in ov.itertuples()),
                   "right_cds": "; ".join(f"{x.product} ({x.strand})" for x in right.itertuples())}
            for db in dbs:
                if (r.query_id, db) in by_db.index:
                    h = by_db.loc[(r.query_id, db)]
                    rec[f"{db}_hit"] = f"{str(h.theader)[:90]} (E={h.evalue:.1e}, qcov {h.qcov:.2f}, tcov {h.tcov:.2f})"
                else:
                    rec[f"{db}_hit"] = ""
            ex_rows.append(rec)
    pd.DataFrame(ex_rows).to_csv(a.out / "exemplars.tsv", sep="\t", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(R[R.db == "any"].round(4).to_string(index=False))
        print(E.round(4).to_string())
        print(C.round(4).to_string())
        print(sk.to_string())
    json.dump({"databases": dbs, "queries": int(len(sel))}, open(a.out / "evidence_inputs.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
