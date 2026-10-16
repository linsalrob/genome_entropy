#!/usr/bin/env python3
"""v7 matched-comparator summary stratified by assembly quality (report §11 item 12, D15).

Every number is computed exactly as 41_comparator_summary.py computes it --
comparator weighted by draw multiplicity, first-order CIs from its share(), the
#97 ladder from 22_rank_candidates.tiers(), Prodigal shares against the matched
comparator and against intergenic_lo, and the report's rule of taking the
larger background -- but within strata of contig N50 and CheckM2
completeness. A
stratum "all" is computed too and must reproduce the committed
matched_comparator/<dom>_v7_matched.tsv; the script exits if it does not, so
the stratified rows cannot drift from the published pooled ones.

Strata use the quartile edges supplementary_analyses.md §7 used (pooled query
rows of the v6 query set, 34_supplementary_arms.py t_assembly), so the pre- and
post-matching tables share bins. Two assignments (decision D15):

  candidate_genome  (primary)     a comparator draw is placed in its
                                  candidate's genome quartile, so matched pairs
                                  stay together and the length x 3Di x
                                  truncation matching holds within the stratum
  own_genome        (sensitivity) every ORF, comparator draws included, is
                                  placed by its own genome

Annotated CDS and intergenic_lo ORFs have no partner and are placed by their
own genome in both.

  43_assembly_strata_v7.py --domain bac --mc <matched_clean/bac> \
      --v6-wanted <full_bac/wanted.tsv> --prodigal <prodigal_v7_bac.coincidence.tsv.gz> \
      --burden <candidate_burden_bac.tsv> --published <matched_comparator/bac_v7_matched.tsv> \
      --out <assembly_strata_v7_bac.tsv>
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, HERE / fn)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m22 = _load("rank", "22_rank_candidates.py")
m41 = _load("cmpsum", "41_comparator_summary.py")
KEY, STR, qid, share = m41.KEY, m41.STR, m41.qid, m41.share
COVS = ("n50_contigs", "checkm2_completeness")
NEW_METRICS = {"n_comparator_with_replacement_draws"}  # not in 41's output
LABELS = ["Q1", "Q2", "Q3", "Q4"]


def edges_from_v6(v6_wanted, md, cov):
    """Quartile edges over pooled v6 query rows, as 34 t_assembly 'pooled_query_rows'."""
    w = pd.read_csv(v6_wanted, sep="\t", dtype=STR, usecols=["genome"])
    v = w.merge(md[["genome", cov]], on="genome", how="left")[cov].dropna()
    _, e = pd.qcut(v, 4, labels=LABELS, retbins=True)
    e = e.astype(float)
    e[0], e[-1] = -np.inf, np.inf
    return e


def summarise(cand_ids, ann_ids, inter_ids, w, n_rep, full_sets, tc, ts, prod):
    """41_comparator_summary.py's numbers for one subset. Returns {metric: (v, lo, hi)}."""
    out = {}
    n_c, n_a = len(cand_ids), len(ann_ids)
    n_draw = float(w.sum())
    n_eff = float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else np.nan
    out["n_candidate"] = (n_c, np.nan, np.nan)
    out["n_comparator_draws"] = (n_draw, np.nan, np.nan)
    out["n_comparator_distinct"] = (len(w), np.nan, np.nan)
    out["n_comparator_effective"] = (n_eff, np.nan, np.nan)
    out["n_comparator_with_replacement_draws"] = (n_rep, np.nan, np.nan)
    out["n_annotated"] = (n_a, np.nan, np.nan)
    if not (n_c and n_a and n_draw):
        return out
    shares = {}
    for db, full in full_sets.items():
        c = len(cand_ids & full) / n_c
        a = len(ann_ids & full) / n_a
        s = float(w[w.index.isin(full)].sum()) / n_draw
        f, lo, hi = share(c, n_c, s, n_eff, a, n_a)
        shares[db] = f
        out[f"fulllen_{db}_candidate"] = (c, np.nan, np.nan)
        out[f"fulllen_{db}_comparator"] = (s, np.nan, np.nan)
        out[f"fulllen_{db}_annotated"] = (a, np.nan, np.nan)
        out[f"share_{db}"] = (f, lo, hi)
    est = [shares[d] for d in m22.EST_DBS]
    out["structural_estimate"] = (np.mean(est) * n_c, min(est) * n_c, max(est) * n_c)

    ci = sorted(cand_ids)
    sw = w.reindex(ts.index).fillna(0).to_numpy(float)
    for k in tc.columns:
        cc = int(tc[k].reindex(ci).fillna(False).astype(bool).sum())
        ss = float((ts[k].to_numpy(bool) * sw).sum()) * n_c / n_draw
        tag = k.split()[0]
        out[f"ladder_{tag}_candidates"] = (cc, np.nan, np.nan)
        out[f"ladder_{tag}_comparator_scaled"] = (ss, np.nan, np.nan)
        out[f"ladder_{tag}_excess"] = (cc - ss, np.nan, np.nan)

    pc, pa, pi, ps = prod
    rc = float(pc.reindex(ci).mean())
    ra = float(pa.reindex(sorted(ann_ids)).mean())
    s = float((ps.reindex(w.index).to_numpy(float) * w.to_numpy(float)).sum()) / n_draw
    out["prodigal_candidate"] = (rc, np.nan, np.nan)
    out["prodigal_annotated_cds"] = (ra, np.nan, np.nan)
    out["prodigal_comparator"] = (s, np.nan, np.nan)
    f, lo, hi = share(rc, n_c, s, n_eff, ra, n_a)
    out["prodigal_share_clean_bg"] = (f, lo, hi)
    # As 41: a second background from unoccupied space, and the report's rule
    # of taking the larger one, computed within the stratum.
    ri = float(pi.reindex(sorted(inter_ids)).mean()) if inter_ids else np.nan
    out["prodigal_intergenic_lo"] = (ri, np.nan, np.nan)
    if inter_ids:
        f2, lo2, hi2 = share(rc, n_c, ri, len(inter_ids), ra, n_a)
        out["prodigal_share_intergenic_lo_bg"] = (f2, lo2, hi2)
        out["prodigal_share_larger_bg"] = ((f if s >= ri else f2), np.nan, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", required=True)
    ap.add_argument("--mc", required=True, help="matched_clean/<dom> directory")
    ap.add_argument("--v6-wanted", required=True)
    ap.add_argument("--prodigal", required=True)
    ap.add_argument("--burden", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--descriptions", default=f"{m22.GD}/foldseek_db/descriptions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mc = Path(a.mc)

    wanted = pd.read_csv(mc / "wanted_v7.tsv", sep="\t", dtype=STR)
    ctx = m22.load_context(str(mc / "context_v7"), {"candidate", "shadow_hi"})
    draws = pd.read_csv(mc / "comparator.tsv.gz", sep="\t", dtype=STR,
                        usecols=KEY + ["match_level", "cand_genome"])
    draws["qid"] = qid(draws, "shadow_hi")
    w_all = draws.qid.value_counts()

    cand = wanted[wanted.group == "candidate"].assign(qid=lambda d: qid(d, "candidate"))
    # intergenic_lo is unchanged by the v7 re-matching; it comes from the
    # Prodigal table, keyed by qid (genome is the qid's first field).
    ann = wanted[wanted.group == "annotated_cds"].assign(qid=lambda d: qid(d, "annotated_cds"))

    best = m22.load_hits([str(mc / "best_v7")], m22.DBS)
    full_sets = {}
    for db in m22.DBS:
        st = best.get((db, "struct"))
        full_sets[db] = set(st.loc[(st.qcov >= m22.FULL) & (st.tcov >= m22.FULL), "query"])
    full_sets["any_sp_pdb_cath"] = set().union(*(full_sets[d] for d in m22.EST_DBS))

    desc = m22.load_descriptions(a.descriptions, m22.DBS)
    wide = m22.widen(ctx[ctx.group.eq("candidate") | ctx.qid.isin(w_all.index)],
                     best, desc, m22.DBS)
    wc = wide[wide.group == "candidate"].drop_duplicates("qid").set_index("qid")
    wsh = wide[wide.group == "shadow_hi"].drop_duplicates("qid").set_index("qid")
    if set(w_all.index) - set(wsh.index):
        sys.exit("ERROR: comparator ORFs without context rows")
    tc = pd.DataFrame(m22.tiers(wc))
    ts = pd.DataFrame(m22.tiers(wsh))

    p = pd.read_csv(a.prodigal, sep="\t", dtype=STR, usecols=KEY + ["group", "coincides"])
    p["coincides"] = p.coincides.astype(str).map({"True": 1.0, "False": 0.0})
    p["qid"] = qid(p, p.group)
    prod = tuple(p[p.group == g].drop_duplicates("qid").set_index("qid").coincides
                 for g in ("candidate", "annotated_cds", "intergenic_lo", "shadow_hi"))

    md = pd.read_csv(a.burden, sep="\t", dtype={"genome": str},
                     usecols=["genome"] + list(COVS))
    rows = []

    def emit(scheme, cov, stratum, lo, hi, res):
        for k, (v, l, h) in res.items():
            rows.append(dict(domain=a.domain, scheme=scheme, covariate=cov,
                             stratum=stratum, cov_lo=lo, cov_hi=hi, metric=k,
                             value=v, lo=l, hi=h))

    rep_all = int((draws.match_level == "with replacement").sum())
    inter = pd.DataFrame({"qid": prod[2].index})
    inter["genome"] = inter.qid.str.split("|").str[0]
    allres = summarise(set(cand.qid), set(ann.qid), set(inter.qid), w_all, rep_all,
                       full_sets, tc, ts, prod)
    emit("all", "all", "all", np.nan, np.nan, allres)

    # Reproduction check against the committed 41 output.
    pub = pd.read_csv(a.published, sep="\t")
    pub = {r.metric: (r.value, r.lo, r.hi) for r in pub.itertuples()}
    # Every metric this script produces must exist in the published table,
    # except the replacement count, which 41 does not report.
    expected = set(allres) - NEW_METRICS
    missing = sorted(expected - set(pub))
    if missing:
        sys.exit(f"ERROR: {len(missing)} metrics absent from {a.published}: {missing}")
    bad = []
    for k in sorted(expected):
        for x, y in zip(allres[k], pub[k]):
            if not (pd.isna(x) and pd.isna(y)) and not np.isclose(x, y, rtol=1e-9, atol=1e-9):
                bad.append((k, x, y))
    checked = len(expected)
    if bad:
        for b in bad[:20]:
            print("MISMATCH", b, file=sys.stderr)
        sys.exit(f"ERROR: stratum 'all' does not reproduce {a.published} ({len(bad)} mismatches)")
    print(f"reproduction check: {checked} metrics equal to {a.published}")

    gq = {}
    for cov in COVS:
        e = edges_from_v6(a.v6_wanted, md, cov)
        gq[cov] = (pd.Series(pd.cut(md[cov], e, labels=LABELS).astype(object).values,
                             index=md.genome), e)
    for cov in COVS:
        q, e = gq[cov]
        c_q = cand.genome.map(q)
        a_q = ann.genome.map(q)
        i_q = inter.genome.map(q)
        for scheme in ("candidate_genome", "own_genome"):
            d_q = (draws.cand_genome if scheme == "candidate_genome" else draws.genome).map(q)
            for i, s in enumerate(LABELS):
                dsel = draws[d_q.values == s]
                res = summarise(set(cand.qid[c_q.values == s]), set(ann.qid[a_q.values == s]),
                                set(inter.qid[i_q.values == s]),
                                dsel.qid.value_counts(),
                                int((dsel.match_level == "with replacement").sum()),
                                full_sets, tc, ts, prod)
                emit(scheme, cov, s, e[i], e[i + 1], res)
        n_un = int(c_q.isna().sum())
        if n_un:
            print(f"WARNING: {n_un} candidates without {cov}", file=sys.stderr)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, sep="\t", index=False)
    show = out[out.metric.isin(["n_candidate", "n_comparator_with_replacement_draws",
                                "share_afdb_swissprot", "share_any_sp_pdb_cath",
                                "ladder_C3_excess", "ladder_C7_excess",
                                "prodigal_share_clean_bg", "prodigal_share_larger_bg"])]
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(show.pivot_table(index=["covariate", "scheme", "stratum"], columns="metric",
                               values="value", sort=False).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
