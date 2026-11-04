#!/usr/bin/env python3
"""Does length explain why annotated phage ORFs sit lower in 3Di entropy than bacterial ones? (agent-handoffs#38)

Reduces per-ORF phage data to the plotting and table data of figPS03. Sources:

  phage 3Di   Gadi Inphared run (genome_entropy 0.3.0, encoding size 10,000, V100), the run behind figP04:
              genome_entropy:General/Validation/Inphared/gadi/ge90aa/phg_*.tsv.gz (SHA256SUMS checked),
              local copy --gadi-dir
  annotation  Setonix all-genome tables (genome_entropy #156): orfs / labels / cds / seqs parquet, --tables
              (contig_edge, Setonix 3Di entropy at encoding size 20,000 for the configuration check,
              3Di strings for the number of states used, CDS coordinates for coverage)
  clusters    skani 95 % ANI clusters, groups_ani95.tsv, --groups (bootstrap unit)
  bacteria    per-length matched-ORF reference, n / mean / q25 / median / q75 by exact aa length:
              data/figP04_gtdb_length_3di.tsv.gz (GTDB r232, all matched ORFs); --bacteria adds more
              references (e.g. Gadi's coverage-restricted table) as name=path

Positives are the Gadi rows with in_genbank True (the figP04 definition). A positive's coverage is the
length of the same-strand CDS that ends at its stop divided by the ORF length (as label_restrict_gap.py).
Phage classes: all, cov80 (coverage >= 0.8), complete (not at a contig edge), cov80_complete.

Gaps are bacteria minus phage, in bits, over the lengths in a bin where the phage class has >= MIN_N ORFs:
  raw        n-weighted bin means, as published (all lengths in the bin)
  phage_w    exact-length gap averaged with the phage length distribution (bacteria at phage lengths)
  bac_w      exact-length gap averaged with the bacterial length distribution
  overlap_w  weights n_p * n_b / (n_p + n_b)
Uncertainty: Poisson bootstrap over 95 % ANI clusters (B = 1000, seed 20261102); the bacterial means
(10^4-10^6 ORFs per length) are treated as fixed.

Outputs (data/): figPS03_phage_length_3di.tsv.gz, figPS03_length_dist.tsv.gz, figPS03_gap_by_bin.tsv.gz,
figPS03_percentiles.tsv.gz, figPS03_provenance.json.

  phage_length_gap_prepare_data.py --gadi-dir work/phage_length_gap/gadi_phg \\
      --tables work/phage_rank_all/tables --groups work/phage_rank_all/groups/groups_ani95.tsv
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

OUT = Path(__file__).resolve().parent.parent / "data"
GZ = {"method": "gzip", "mtime": 0}
LMIN, LMAX = 90, 999
BINS = [(90, 149), (150, 299), (300, 599), (600, 999), (90, 999)]
MIN_N = 10
B, SEED = 1000, 20261102
PROV = {"inputs": {}, "outputs": {}, "parameters": dict(min_n=MIN_N, bootstrap=B, seed=SEED,
                                                       bootstrap_unit="skani 95 % ANI cluster")}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def note(p):
    PROV["inputs"][str(p)] = sha256(p)


def write(df, name):
    p = OUT / name
    df.to_csv(p, sep="\t", index=False, compression=GZ, float_format="%.6g")
    PROV["outputs"][name] = dict(rows=len(df), sha256=sha256(p))
    print(f"  out  {name}  rows={len(df)}")


def load(a):
    files = sorted(Path(a.gadi_dir).glob("phg_*.tsv.gz"))
    assert len(files) == 10, files
    g = []
    for p in files:
        note(p)
        g.append(pd.read_csv(p, sep="\t", usecols=["input_id", "orf_id", "start", "end", "strand", "aa_length",
                                                   "in_genbank", "three_di_entropy"], dtype={"in_genbank": str}))
    g = pd.concat(g, ignore_index=True)
    T = Path(a.tables)
    for f in ("orfs", "labels", "cds", "seqs"):
        note(T / f"{f}.parquet")
    o = pd.read_parquet(T / "orfs.parquet", columns=["row", "input_id", "orf_id", "start", "end", "strand",
                                                    "g_start", "g_end", "protein_length", "contig_edge",
                                                    "three_di_entropy"])
    o = o.rename(columns={"three_di_entropy": "te_setonix"})
    lab = pd.read_parquet(T / "labels.parquet", columns=["row", "in_genbank"]).rename(columns={"in_genbank": "lab_setonix"})
    o = o.merge(lab, on="row", validate="one_to_one")
    d = g.merge(o, on=["input_id", "orf_id", "start", "end", "strand"], how="left", validate="one_to_one")
    PROV["join"] = dict(gadi_rows=len(g), joined=int(d.row.notna().sum()))
    assert d.row.notna().all(), "Gadi rows without a Setonix ORF"
    assert (d.aa_length == d.protein_length).all()
    d["pos"] = d.in_genbank == "True"
    PROV["label_agreement"] = dict(both=int((d.pos & d.lab_setonix).sum()), gadi_only=int((d.pos & ~d.lab_setonix).sum()),
                                   setonix_only=int((~d.pos & d.lab_setonix).sum()))
    PROV["positives"] = int(d.pos.sum())

    # coverage of each positive by the same-strand CDS ending at its stop (0-based half-open)
    c = pd.read_parquet(T / "cds.parquet", columns=["input_id", "g_start", "g_end", "strand", "aa_len"])
    c["cl"] = c.g_end - c.g_start
    c["stop"] = np.where(c.strand == "+", c.g_end, c.g_start)
    p = d[d.pos].copy()
    p["stop"] = np.where(p.strand == "+", p.g_end, p.g_start)
    m = (p[["row", "input_id", "strand", "stop"]].merge(c[["input_id", "strand", "stop", "cl", "aa_len"]],
                                                       on=["input_id", "strand", "stop"])
         .sort_values("cl").drop_duplicates("row", keep="last"))
    p = p.merge(m[["row", "cl", "aa_len"]], on="row", how="left")
    p["cov"] = p.cl / (p.g_end - p.g_start)
    PROV["positives_without_cds_at_stop"] = int(p.cl.isna().sum())

    # number of distinct 3Di states (Setonix strings) for the finite-sample bias bound
    s = pd.read_parquet(T / "seqs.parquet", columns=["row", "three_di"], filters=[("row", "in", p.row.tolist())])
    s["k"] = s.three_di.map(lambda x: len(set(x)))
    p = p.merge(s[["row", "k"]], on="row", how="left")

    grp = pd.read_csv(a.groups, sep="\t").set_index("input_id").group
    note(Path(a.groups))
    p["cluster"] = p.input_id.map(grp)
    assert p.cluster.notna().all()
    return d, p, c


def bacteria_refs(a):
    refs = {}
    f = OUT / "figP04_gtdb_length_3di.tsv.gz"
    note(f)
    t = pd.read_csv(f, sep="\t")
    refs["gtdb_all"] = t[(t.dataset == "bacteria") & (t.cls == "matched")].set_index("aa_length")
    for kv in a.bacteria or []:
        name, path = kv.split("=", 1)
        note(Path(path))
        refs[name] = pd.read_csv(path, sep="\t").set_index("aa_length")
    return {k: v.loc[LMIN:LMAX] for k, v in refs.items()}


def per_length(p, classes):
    rows = []
    for cname, x in classes.items():
        q = x.groupby("aa_length")
        t = pd.DataFrame({"n": q.size(), "mean": q.three_di_entropy.mean(), "sd": q.three_di_entropy.std(),
                          "q25": q.three_di_entropy.quantile(0.25), "median": q.three_di_entropy.median(),
                          "q75": q.three_di_entropy.quantile(0.75), "mean_setonix": q.te_setonix.mean(),
                          "mean_k": q.k.mean(),
                          # Miller-Madow: the plug-in estimate is low by about (K - 1) / (2 N ln 2) bits
                          "mm_correction": q.mm.mean()}).reset_index()
        t.insert(0, "cls", cname)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def bootstrap(x, ref, rng_seed):
    """Per-cluster sums by length; returns gap estimates (point + B replicates) per bin and scheme."""
    L = np.arange(LMIN, LMAX + 1)
    cl, ci = np.unique(x.cluster, return_inverse=True)
    li = x.aa_length.to_numpy() - LMIN
    shp = (len(cl), len(L))
    S = sp.csr_array((x.three_di_entropy.to_numpy(), (ci, li)), shape=shp)
    N = sp.csr_array((np.ones(len(x)), (ci, li)), shape=shp)
    bq = {}
    for q in ("q25", "median", "q75"):
        thr = ref[q].reindex(L).to_numpy()
        below = (x.three_di_entropy.to_numpy() < thr[li]).astype(float)
        bq[q] = sp.csr_array((below, (ci, li)), shape=shp)
    rng = np.random.default_rng(rng_seed)
    W = np.vstack([np.ones(len(cl)), rng.poisson(1.0, size=(B, len(cl)))])   # row 0 = point estimate
    s, n = np.asarray((S.T @ W.T).T), np.asarray((N.T @ W.T).T)
    nb = ref["n"].reindex(L).to_numpy(float)
    mb = ref["mean"].reindex(L).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = s / n
        below = {q: np.asarray((M.T @ W.T).T) / n for q, M in bq.items()}
    n0 = n[0]
    out, pct = [], []
    for lo, hi in BINS:
        sel = (L >= lo) & (L <= hi)
        sup = sel & (n0 >= MIN_N) & np.isfinite(mb)
        est = {}
        # published: n-weighted means over every length in the bin
        est["raw"] = (np.nansum(nb[sel] * mb[sel]) / np.nansum(nb[sel])) - s[:, sel].sum(1) / n[:, sel].sum(1)
        ok = n[:, sup] > 0   # a replicate can drop every cluster at a sparse length
        d = np.where(ok, mb[sup] - np.where(ok, mp[:, sup], 0), 0)
        wp = n[:, sup]
        est["phage_w"] = (d * wp).sum(1) / wp.sum(1)
        est["bac_w"] = (d * nb[sup]).sum(1) / (ok * nb[sup]).sum(1)
        wo = n[:, sup] * nb[sup] / (n[:, sup] + nb[sup])
        est["overlap_w"] = (d * wo).sum(1) / wo.sum(1)
        for k, v in est.items():
            out.append(dict(bin=f"{lo}-{hi}", scheme=k, estimate=v[0], ci_lo=np.nanpercentile(v[1:], 2.5),
                            ci_hi=np.nanpercentile(v[1:], 97.5), n_phage=int(n0[sel].sum()),
                            n_phage_support=int(n0[sup].sum()), n_bacteria=int(np.nansum(nb[sel])),
                            lengths_support=int(sup.sum())))
        for q, v in below.items():
            f = np.nan_to_num(v[:, sup] * n[:, sup]).sum(1) / n[:, sup].sum(1)
            pct.append(dict(bin=f"{lo}-{hi}", bacterial_quantile=q, phage_share_below=f[0],
                            ci_lo=np.nanpercentile(f[1:], 2.5), ci_hi=np.nanpercentile(f[1:], 97.5),
                            n_phage_support=int(n0[sup].sum())))
    return pd.DataFrame(out), pd.DataFrame(pct)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gadi-dir", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--bacteria", nargs="*", help="extra bacterial references, name=path")
    a = ap.parse_args()

    d, p, c = load(a)
    p = p[p.aa_length.between(LMIN, LMAX)].copy()
    p["mm"] = (p.k - 1) / (2 * p.aa_length * np.log(2))
    cov80 = p["cov"] >= 0.8
    classes = {"all": p, "cov80": p[cov80], "complete": p[~p.contig_edge], "cov80_complete": p[cov80 & ~p.contig_edge]}
    PROV["class_n"] = {k: len(v) for k, v in classes.items()}
    write(per_length(p, classes), "figPS03_phage_length_3di.tsv.gz")

    # length distributions: matched ORF length (3Di measured on it) and the CDS length behind each positive,
    # plus every annotated CDS; bacterial ORF lengths come from the reference n column. The index spans every
    # observed length, so CDS shorter than 90 aa behind a 90-999-aa ORF (and CDS > 999 aa) keep their mass
    L = pd.Index(np.arange(1, int(max(LMAX, p.aa_len.max(), c.aa_len.max())) + 1), name="aa_length")
    PROV["cds_length"] = dict(matched_cds=int(p.aa_len.notna().sum()), matched_cds_below_90=int((p.aa_len < LMIN).sum()),
                              matched_cds_min=int(p.aa_len.min()))
    dist = pd.DataFrame({
        "phage_matched_orf": p.aa_length.value_counts().reindex(L, fill_value=0),
        "phage_matched_cds": p.aa_len.value_counts().reindex(L, fill_value=0),
        "phage_cds_all": c.aa_len.value_counts().reindex(L, fill_value=0),
        "phage_matched_orf_cov80": p[cov80].aa_length.value_counts().reindex(L, fill_value=0),
    }).reset_index()
    assert dist.phage_matched_cds.sum() == p.aa_len.notna().sum() and dist.phage_cds_all.sum() == len(c)
    write(dist, "figPS03_length_dist.tsv.gz")

    refs = bacteria_refs(a)
    gaps, pcts = [], []
    for rname, ref in refs.items():
        for i, (cname, x) in enumerate(classes.items()):
            g, q = bootstrap(x, ref, SEED + i)
            gaps.append(g.assign(phage_class=cname, bacteria_ref=rname))
            pcts.append(q.assign(phage_class=cname, bacteria_ref=rname))
        # configuration check: the same comparison with the Setonix encoding (size 20,000, ROCm)
        xs = p.assign(three_di_entropy=p.te_setonix)
        g, _ = bootstrap(xs, ref, SEED)
        gaps.append(g.assign(phage_class="all_setonix_encoding", bacteria_ref=rname))
    cols = ["bacteria_ref", "phage_class", "bin", "scheme"]
    write(pd.concat(gaps)[cols + [c for c in gaps[0].columns if c not in cols]], "figPS03_gap_by_bin.tsv.gz")
    cols = ["bacteria_ref", "phage_class", "bin", "bacterial_quantile"]
    write(pd.concat(pcts)[cols + [c for c in pcts[0].columns if c not in cols]], "figPS03_percentiles.tsv.gz")
    (OUT / "figPS03_provenance.json").write_text(json.dumps(PROV, indent=1) + "\n")
    print(json.dumps({k: v for k, v in PROV.items() if k not in ("inputs", "outputs")}, indent=1))


if __name__ == "__main__":
    main()
