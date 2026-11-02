#!/usr/bin/env python3
"""Per-genome missed-gene candidates and the Table 4 sequence identities (round-2 review).

agent-handoffs#38, items 2 and 3. Runs on NCI Gadi against the v7 outputs.

  tabG03_per_genome_candidates.tsv.gz  one row per annotated genome: C3/C7
      candidates, their matched-comparator counts and the excess, genome quality,
      deposited CDS against GTDB's Prodigal count.
  tabG03_distribution.tsv              per-genome distribution and histogram.
  tabG04_candidate_tail.tsv.gz         genomes with >= 10 C3 or >= 5 C7, with
      clustering, contigs that carry no annotated CDS, and the top products.
  tabG05_table4_seqid.tsv              fident of the Swiss-Prot hit that names
      each Table 4 product, and of the sequence-mode hit.

C3/C7 are 22_rank_candidates.py's tiers, imported unchanged. Each comparator
draw belongs to one candidate (comparator.tsv.gz, cand_genome), so the ladder of
41_comparator_summary.py splits exactly by candidate genome; the per-genome
excess sums back to the published totals, which is checked.

  gtdb_tail_prepare_data.py --repo <genome_entropy checkout> \
                            --gtdb-dir /g/data/ob80/re3494/gtdb_entropy
"""
import argparse
import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
GZ = {"method": "gzip", "mtime": 0}
LINK = 10_000          # candidates closer than this on one contig form a cluster
EXPECT = {"bac": (25_620, 4_974), "arc": (1_731, 407)}   # Table 3 C3/C7 excess
C3K, C7K = "C3 C2 + any full-length structural hit", "C7 strict combined rule"
PHAGE = re.compile(r"\b(?:phage|prophage|terminase|capsid|portal|baseplate|holin|"
                   r"integrase|excisionase|virion)\b", re.I)
# Table 4 order (manuscript/genome_entropy.md).
TABLE4 = ["GCA_900230115.1", "GCA_035656275.1", "GCA_006738865.1",
          "GCA_902825215.1", "GCA_000961845.1", "GCA_900635495.1",
          "GCA_002898535.1", "GCA_049515795.1", "GCA_000156115.1",
          "GCA_001552015.1"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(df, name):
    p = OUT / name
    if name.endswith(".gz"):
        df.to_csv(p, sep="\t", index=False, compression=GZ)
    else:
        df.to_csv(p, sep="\t", index=False)
    print(f"  out  {p.name}  rows={len(df)}  sha256={sha256(p)}")


def load_m22(repo):
    spec = importlib.util.spec_from_file_location(
        "rank", repo / "Validation/gtdb/scripts/22_rank_candidates.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ranked(m22, gd, dom):
    """Candidates and their matched comparator draws, tiered, for one domain."""
    mc = gd / "missed_genes/matched_clean" / dom
    print(f"  in   {mc}/context_v7, best_v7, comparator.tsv.gz")
    ctx = m22.load_context(str(mc / "context_v7"), {"candidate", "shadow_hi"})
    best = m22.load_hits([str(mc / "best_v7")], m22.DBS)
    desc = m22.load_descriptions(str(gd / "foldseek_db/descriptions"), m22.DBS)
    wide = m22.widen(ctx, best, desc, m22.DBS)
    cand = wide[wide.group == "candidate"].copy()
    shad = wide[wide.group == "shadow_hi"].drop_duplicates("qid").set_index("qid")
    tc, ts = m22.tiers(cand), m22.tiers(shad)
    cand["c3"], cand["c7"] = tc[C3K].to_numpy(bool), tc[C7K].to_numpy(bool)

    draws = pd.read_csv(mc / "comparator.tsv.gz", sep="\t", dtype=str,
                        usecols=["genome", "input_id", "orf_id", "cand_genome"])
    q = draws.genome + "|" + draws.input_id + "|" + draws.orf_id + "|shadow_hi"
    if not set(q) <= set(shad.index):
        sys.exit(f"{dom}: comparator ORFs without context rows")
    scale = len(cand) / len(draws)
    comp = pd.DataFrame({"genome": draws.cand_genome.to_numpy(),
                         "comparator_c3": ts[C3K].reindex(q).to_numpy(float) * scale,
                         "comparator_c7": ts[C7K].reindex(q).to_numpy(float) * scale}
                        ).groupby("genome").sum()
    return cand, comp


def clusters(t):
    """(largest run, candidates in runs of >= 3) on one contig, gaps <= LINK bp."""
    big, n3 = 0, 0
    for _, s in t.sort_values("g_start").groupby("input_id"):
        st, en = s.g_start.to_numpy(), s.g_end.to_numpy()
        runs, run, end = [], 1, en[0]
        for a, b in zip(st[1:], en[1:]):
            if a - end <= LINK:
                run += 1
            else:
                runs.append(run)
                run = 1
            end = max(end, b)
        runs.append(run)
        big = max(big, max(runs))
        n3 += sum(r for r in runs if r >= 3)
    return big, n3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--gtdb-dir", type=Path,
                    default=Path("/g/data/ob80/re3494/gtdb_entropy"))
    a = ap.parse_args()
    gd = a.gtdb_dir
    commit = subprocess.run(["git", "-C", str(a.repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"repo {a.repo} @ {commit}")
    m22 = load_m22(a.repo)

    burden = pd.concat([pd.read_csv(gd / f"missed_genes/candidate_burden_{d}.tsv",
                                    sep="\t") for d in ("bac", "arc")])
    burden = burden[burden.annotated.astype(str).str.lower().eq("true")]
    # candidate_burden's n_cds counts ORFs matching a CDS; the deposited count
    # is in genome_cds_counts_<domain>.tsv.
    dep = pd.concat([pd.read_csv(gd / f"genome_cds_counts_{d}.tsv", sep="\t")
                     for d in ("bac", "arc")])[["genome", "n_cds"]]

    cands, comps = [], []
    for dom in ("bac", "arc"):
        c, s = ranked(m22, gd, dom)
        cands.append(c.assign(domain=dom))
        comps.append(s)
        e3 = c.c3.sum() - s.comparator_c3.sum()
        e7 = c.c7.sum() - s.comparator_c7.sum()
        print(f"  {dom}: C3 excess {e3:.1f}, C7 excess {e7:.1f}")
        if (round(e3), round(e7)) != EXPECT[dom]:
            sys.exit(f"{dom}: excess does not reproduce Table 3 {EXPECT[dom]}")
    cand = pd.concat(cands, ignore_index=True)
    comp = pd.concat(comps)

    g = cand.groupby("genome").agg(n_c3=("c3", "sum"), n_c7=("c7", "sum"))
    per = burden[["genome", "domain", "gtdb_taxonomy", "checkm2_completeness",
                  "checkm2_contamination", "contig_count", "n50_contigs",
                  "genome_size", "ncbi_assembly_level", "protein_count",
                  "n_candidates"]].merge(dep.rename(columns={"n_cds": "deposited_cds"}),
                                         on="genome", how="left")
    per = per.rename(columns={"protein_count": "prodigal_genes"})
    per = per.merge(g, on="genome", how="left").merge(comp, on="genome", how="left")
    per = per.fillna({"n_c3": 0, "n_c7": 0, "comparator_c3": 0.0, "comparator_c7": 0.0})
    per[["n_c3", "n_c7"]] = per[["n_c3", "n_c7"]].astype(int)
    if per.n_c7.sum() != cand.c7.sum() or per.n_c3.sum() != cand.c3.sum():
        sys.exit("candidate genomes missing from the annotated-genome table")
    per["excess_c3"] = per.n_c3 - per.comparator_c3
    per["excess_c7"] = per.n_c7 - per.comparator_c7
    per["deposited_over_prodigal"] = per.deposited_cds / per.prodigal_genes
    per = per.round({"comparator_c3": 2, "comparator_c7": 2, "excess_c3": 2,
                     "excess_c7": 2, "deposited_over_prodigal": 4})
    per = per.sort_values(["domain", "n_c7", "n_c3", "genome"],
                          ascending=[False, False, False, True])
    write(per, "tabG03_per_genome_candidates.tsv.gz")

    rows = []
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, np.inf]
    labels = ["0", "1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", ">=100"]
    for crit in ("c3", "c7"):
        for dom in ("bac", "arc"):
            d = per[per.domain == dom]
            x = d[f"n_{crit}"]
            ex = d[f"excess_{crit}"]
            sparse = d.deposited_over_prodigal < 0.5
            base = dict(criterion=crit.upper(), domain=dom)
            stats = {"genomes": len(x), "with_any": int((x > 0).sum()),
                     "candidates": int(x.sum()), "excess": round(float(ex.sum()), 1),
                     "median": float(x.median()), "p99": float(x.quantile(.99)),
                     "max": int(x.max()),
                     "genomes_ge10": int((x >= 10).sum()),
                     "candidates_in_ge10": int(x[x >= 10].sum()),
                     "excess_in_ge10": round(float(ex[x >= 10].sum()), 1),
                     "genomes_deposited_lt_half_prodigal": int(sparse.sum()),
                     "candidates_in_those": int(x[sparse].sum()),
                     "excess_in_those": round(float(ex[sparse].sum()), 1)}
            for k, v in stats.items():
                rows.append({**base, "kind": "summary", "key": k, "value": v})
            h = pd.cut(x, bins, right=False, labels=labels).value_counts().reindex(labels)
            for b, v in h.items():
                rows.append({**base, "kind": "histogram", "key": b, "value": int(v)})
    write(pd.DataFrame(rows), "tabG03_distribution.tsv")

    tail = per[(per.n_c3 >= 10) | (per.n_c7 >= 5)].copy()
    cand["product"] = cand.afdb_swissprot_product.fillna(cand.pdb100_product).fillna("")
    ctx = []
    for gid, s in cand[cand.genome.isin(tail.genome)].groupby("genome"):
        rec = dict(genome=gid)
        for crit in ("c3", "c7"):
            t = s[s[crit]]
            big, n3 = clusters(t) if len(t) else (0, 0)
            rec.update({f"{crit}_contigs": t.input_id.nunique(),
                        f"{crit}_largest_cluster": big,
                        f"{crit}_in_clusters_ge3": n3,
                        f"{crit}_on_contigs_without_cds": int((t.n_cds_contig == 0).sum()),
                        f"{crit}_phage_keyword": int(t["product"].str.contains(PHAGE).sum())})
        t = s[s.c7] if s.c7.any() else s[s.c3]
        top = t["product"][t["product"] != ""].value_counts().head(3)
        rec["top_products"] = "; ".join(f"{k} ({v})" for k, v in top.items())
        ctx.append(rec)
    tail = tail.merge(pd.DataFrame(ctx), on="genome", how="left")
    write(tail, "tabG04_candidate_tail.tsv.gz")

    m8 = {("bac"): gd / "missed_genes/full_bac/search_pooled/best",
          ("arc"): gd / "missed_genes/pilot_arc/search/best"}
    ex = pd.read_csv(gd / "missed_genes/manuscript_examples.tsv", sep="\t", dtype=str)
    ex["qid"] = ex.genome + "|" + ex.input_id + "|" + ex.orf_id + "|candidate"
    hits = {}
    for mode in ("struct", "seq"):
        fr = []
        for dom, d in m8.items():
            p = d / f"query.afdb_swissprot.{mode}.m8"
            print(f"  in   {p}")
            h = pd.read_csv(p, sep="\t", names=m22.M8_COLS, low_memory=False)
            fr.append(h[h["query"].isin(ex.qid)])
        hits[mode] = pd.concat(fr).set_index("query")
    rows = []
    for i, gid in enumerate(TABLE4, 1):
        e = ex[ex.genome == gid].iloc[0]
        st = hits["struct"].loc[e.qid]
        sq = hits["seq"].loc[e.qid] if e.qid in hits["seq"].index else None
        full = lambda h: h is not None and h.qcov >= m22.FULL and h.tcov >= m22.FULL
        if not full(st):
            sys.exit(f"{gid}: product-naming Swiss-Prot hit is not full length")
        rows.append(dict(
            table4_row=i, genome=gid, input_id=e.input_id, orf_id=e.orf_id,
            best_product=e.best_product, swissprot_target=st.target,
            struct_fident=st.fident, struct_qcov=st.qcov, struct_tcov=st.tcov,
            struct_evalue=st.evalue,
            seq_fident=sq.fident if full(sq) else np.nan,
            seq_full_length=full(sq), structure_only=e.structure_only))
    write(pd.DataFrame(rows), "tabG05_table4_seqid.tsv")


if __name__ == "__main__":
    main()
