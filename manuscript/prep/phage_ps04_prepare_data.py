#!/usr/bin/env python3
"""Length-matched comparison of 3Di entropy between annotated phage and bacterial ORFs.

Answers agent-handoffs#38 ("could phage protein length explain lower 3Di
entropy?") for Results, "Phage genes sit lower in 3Di entropy than cellular
genes". Sources (all on NCI Gadi):

  phage rows    /g/data/ob80/re3494/inphared_entropy/entropy_3di_results/phg_*.tsv.gz
                (6,754,361 ORFs >= 90 aa, 36,025 genomes; genome_entropy 0.3.0)
  phage JSON    .../entropy_3di_results/phg_*.tar.zst (3Di strings)
  phage CDS     /g/data/ob80/re3494/inphared_entropy/genbank/phg_*.tar.zst
                (the Pyrodigal-gv/phold GenBank records that were labelled)
  GTDB rows     /g/data/ob80/re3494/gtdb_entropy/entropy_rows/bac/bac_*.tsv.gz (760 chunks)
  GTDB CDS      /g/data/ob80/re3494/gtdb_entropy/cds_intervals/bac/bac_*.tsv
                (deposited CDS, 0-based half-open)
  GTDB JSON     /g/data/ob80/re3494/gtdb_entropy/entropy_3di_results/bac/bac_*.tar.zst

Definitions, identical for both populations:
  positive   in_genbank True (the coordinate-anchored matcher; no stop check).
  ORF length aa_length, the stop-to-stop ORF whose entropy was measured.
  ORF interval  get_orfs coordinates normalised to the genomic axis exactly as
             genome_entropy.io.genbank.normalise_orf_interval (minus-strand
             coordinates index the reverse complement).
  CDS cover  length of the longest same-strand CDS on the same record that ends
             at the ORF's stop, divided by the ORF length (both in nt, stop
             included), as Validation/Inphared/scripts/label_restrict_gap.py.
             Class 0 = no CDS ends at the stop; 1 = < 0.5; 2 = 0.5-0.8; 3 = >= 0.8.
  truncated  the ORF reaches within 3 nt of either record end, so it has no
             stop codon or no upstream stop and its length is censored.

Stages (each writes under --work; combine writes manuscript/data):
  bac     --task I --ntasks N   PBS array: strided GTDB chunks -> per-chunk sums
                                and 0.01-bit histograms by (aa length, cover, truncated)
  phage                         per-positive phage table + CDS lengths
  states  --task I --ntasks N   3Di strings: distinct states, Miller-Madow entropy
                                and state counts for phage positives and a
                                seeded sample of GTDB chunks
  combine                       compact TSVs for figPS04 / Table PS04

  phage_ps04_prepare_data.py bac --task 0 --ntasks 10
  phage_ps04_prepare_data.py phage
  phage_ps04_prepare_data.py states --task 0 --ntasks 8
  phage_ps04_prepare_data.py combine
"""
import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

G = Path("/g/data/ob80/re3494")
IPH = G / "inphared_entropy"
GD = G / "gtdb_entropy"
OUT = Path(__file__).resolve().parent.parent / "data"
WORK = Path("/scratch/ob80/re3494/phage_ps03")
GZ = {"method": "gzip", "mtime": 0}

LMIN, LMAX = 90, 999          # exact-length strata; longer ORFs pooled as LTOP
LTOP = LMAX + 1
NL = LTOP - LMIN + 1
HB = 0.01                     # histogram bin width (bits)
NH = int(np.ceil(np.log2(20) / HB)) + 1   # 0 .. log2(20)
NCOV, NTR = 4, 2
CDS_LMAX = 5000               # CDS length histogram cap (aa)
SEED = 0
N_PHAGE_GROUPS = 200          # jackknife groups of phage genomes
N_STATE_CHUNKS = 10           # GTDB chunks sampled for 3Di strings
BINS = [(90, 149), (150, 299), (300, 599), (600, 999)]
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(df, name):
    p = OUT / name
    if name.endswith(".gz"):
        df.to_csv(p, sep="\t", index=False, compression=GZ, float_format="%.6g")
    else:
        df.to_csv(p, sep="\t", index=False, float_format="%.6g")
    print(f"  out  {p.name}  rows={len(df)}  sha256={sha256(p)}", flush=True)


# ------------------------------------------------------------------ shared
def normalise(d):
    """get_orfs 1-based inclusive -> genomic 0-based half-open (normalise_orf_interval)."""
    plus = (d.strand == "+").to_numpy()
    s, e, L = d.start.to_numpy(), d.end.to_numpy(), d.contig_length.to_numpy()
    d["g_start"] = np.where(plus, s - 1, L - e)
    d["g_end"] = np.where(plus, e, L - s + 1)
    d["stop"] = np.where(plus, d.g_end, d.g_start)
    d["trunc"] = ((d.g_start < 3) | (d.g_end > L - 3)).astype(np.int8)
    return d


def prep_cds(cds):
    """GTDB CDS parts -> one row per feature: input_id, strand, stop, cl (nt, summed over parts).

    cds_intervals.py writes one row per contiguous part, consecutively and in Biopython's
    biological order, so the last part of a feature holds its stop codon. A feature is a run
    of consecutive rows with the same (contig, cds_id): IDs need not be unique, and min/max
    over the parts would put the stop of an origin-crossing CDS at the record end.
    """
    new = (cds.input_id != cds.input_id.shift()) | (cds.cds_id != cds.cds_id.shift())
    c = cds.assign(feat=new.cumsum(), cl=cds.end - cds.start)
    f = c.groupby("feat", sort=False).agg(input_id=("input_id", "last"), strand=("strand", "last"),
                                         start=("start", "last"), end=("end", "last"), cl=("cl", "sum"))
    f["stop"] = np.where(f.strand == "+", f.end, f.start)
    return f[["input_id", "strand", "stop", "cl"]].reset_index(drop=True)


def cover(pos, cds):
    """pos: input_id, strand, stop, g_start, g_end; cds: input_id, strand, stop, cl."""
    c = cds.groupby(["input_id", "strand", "stop"], as_index=False).cl.max()
    m = pos.merge(c, on=["input_id", "strand", "stop"], how="left")
    cov = m.cl.to_numpy() / (m.g_end - m.g_start).to_numpy()
    covc = np.select([np.isnan(cov), cov < 0.5, cov < 0.8], [0, 1, 2], 3).astype(np.int8)
    m["cov"] = cov
    m["covc"] = covc
    return m


def lidx(aa):
    return np.clip(aa, LMIN, LTOP) - LMIN


def hidx(h):
    return np.clip((h / HB).astype(np.int64), 0, NH - 1)


# ------------------------------------------------------------------ bac
def bac_chunks():
    return sorted((GD / "entropy_rows" / "bac").glob("bac_*.tsv.gz"))


def stage_bac(a):
    files = bac_chunks()
    if len(files) != 760:
        sys.exit(f"expected 760 bacterial chunks, found {len(files)}")
    mine = files[a.task::a.ntasks]
    hist = np.zeros((NL, NCOV, NTR, NH), np.int64)
    cds_all = np.zeros(CDS_LMAX + 1, np.int64)      # every deposited CDS, aa (stop excluded)
    cds_pos = np.zeros(CDS_LMAX + 1, np.int64)      # stop-sharing CDS of positives
    cds_all_sum = 0
    cds_pos_sum = 0
    sums = []
    for f in mine:
        ch = f.name[:-7]
        d = pd.read_csv(f, sep="\t", usecols=["genome", "input_id", "start", "end", "strand",
                                             "aa_length", "in_genbank", "three_di_entropy",
                                             "contig_length"],
                        dtype={"in_genbank": str, "genome": str, "input_id": str})
        n_rows = len(d)
        d = normalise(d[d.in_genbank == "True"].copy())
        c = pd.read_csv(GD / "cds_intervals" / "bac" / f"{ch}.tsv", sep="\t",
                        usecols=["contig", "start", "end", "strand", "cds_id"],
                        dtype={"contig": str})
        c = prep_cds(c.rename(columns={"contig": "input_id"}))
        cl_aa = np.maximum(c.cl.to_numpy().astype(np.int64) // 3 - 1, 0)
        np.add.at(cds_all, np.clip(cl_aa, 0, CDS_LMAX), 1)
        cds_all_sum += int(cl_aa.sum())
        m = cover(d[["input_id", "strand", "stop", "g_start", "g_end", "aa_length",
                     "three_di_entropy", "trunc"]], c[["input_id", "strand", "stop", "cl"]])
        ok = ~np.isnan(m["cov"].to_numpy())
        cl_pos = np.maximum(m.cl.to_numpy()[ok].astype(np.int64) // 3 - 1, 0)
        np.add.at(cds_pos, np.clip(cl_pos, 0, CDS_LMAX), 1)
        cds_pos_sum += int(cl_pos.sum())
        li, h = lidx(m.aa_length.to_numpy()), m.three_di_entropy.to_numpy()
        cv, tr = m.covc.to_numpy(), m.trunc.to_numpy()
        np.add.at(hist, (li, cv, tr, hidx(h)), 1)
        k = pd.DataFrame({"li": li, "covc": cv, "trunc": tr, "h": h, "h2": h * h,
                          "aa": m.aa_length.to_numpy()})
        s = k.groupby(["li", "covc", "trunc"], as_index=False).agg(n=("h", "size"), s1=("h", "sum"),
                                                                    s2=("h2", "sum"),
                                                                    s_aa=("aa", "sum"))
        s.insert(0, "chunk", ch)
        sums.append(s)
        print(f"  {ch}: rows {n_rows:,}, positives {len(m):,}, stop-sharing CDS {ok.mean():.4f}",
              flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    pd.concat(sums).to_parquet(WORK / f"bac_sums_{a.task:02d}.parquet", index=False)
    np.savez_compressed(WORK / f"bac_hist_{a.task:02d}.npz", hist=hist, cds_all=cds_all,
                        cds_pos=cds_pos, cds_all_sum=np.int64(cds_all_sum),
                        cds_pos_sum=np.int64(cds_pos_sum))
    print(f"bac task {a.task}: {len(mine)} chunks done", flush=True)


# ------------------------------------------------------------------ phage
CDS_RE = re.compile(r"^     CDS             (.+)$")
PART_RE = re.compile(r"<?(\d+)\.\.>?(\d+)")


def phage_cds():
    rows = []
    for t in sorted((IPH / "genbank").glob("phg_*.tar.zst")):
        txt = subprocess.run(["tar", "-I", "zstd", "-xOf", str(t)], capture_output=True,
                             text=True, check=True).stdout
        loc = None
        for line in txt.splitlines():
            if line.startswith("LOCUS"):
                loc = line.split()[1]
                continue
            m = CDS_RE.match(line)
            if m:
                s = m.group(1)
                # parts in biological order (complement(join(a,b)) runs b then a, as in Biopython);
                # the last part holds the stop, and min..max would make an origin-crossing CDS
                # span the whole record
                parts = [(int(a) - 1, int(b)) for a, b in PART_RE.findall(s)]
                if s.startswith("complement("):
                    parts = parts[::-1]
                strand = "-" if "complement" in s else "+"
                rows.append((loc, strand, parts[-1][1] if strand == "+" else parts[-1][0],
                             sum(b - a for a, b in parts), len(parts) > 1, "<" in s or ">" in s))
    return pd.DataFrame(rows, columns=["input_id", "strand", "stop", "cl", "joined", "partial"])


def stage_phage(a):
    files = sorted((IPH / "entropy_3di_results").glob("phg_*.tsv.gz"))
    if len(files) != 10:
        sys.exit(f"expected 10 phage chunks, found {len(files)}")
    d = pd.concat([pd.read_csv(f, sep="\t", usecols=["genome", "input_id", "orf_id", "start", "end",
                                                     "strand", "aa_length", "in_genbank",
                                                     "three_di_entropy", "contig_length"],
                               dtype={"in_genbank": str, "genome": str, "input_id": str})
                   for f in files], ignore_index=True)
    print(f"  phage rows {len(d):,}", flush=True)
    c = phage_cds()
    print(f"  phage CDS {len(c):,} (joined {int(c.joined.sum())}, partial {int(c.partial.sum())})",
          flush=True)
    d = normalise(d[d.in_genbank == "True"].copy())
    m = cover(d[["genome", "input_id", "orf_id", "strand", "stop", "g_start", "g_end",
                 "aa_length", "three_di_entropy", "trunc"]], c[["input_id", "strand", "stop", "cl"]])
    genomes = np.sort(m.genome.unique())
    perm = np.random.default_rng(SEED).permutation(len(genomes))
    grp = pd.Series(perm % N_PHAGE_GROUPS, index=genomes)
    m["grp"] = grp.loc[m.genome].to_numpy().astype(np.int16)
    WORK.mkdir(parents=True, exist_ok=True)
    m.drop(columns=["stop"]).to_parquet(WORK / "phage_pos.parquet", index=False)
    c.assign(aa=c.cl // 3 - 1).to_parquet(WORK / "phage_cds.parquet", index=False)
    print(f"  phage positives {len(m):,}; stop-sharing CDS {(m.covc > 0).mean():.4f}; "
          f"genomes {len(genomes):,}", flush=True)


# ------------------------------------------------------------------ states
def json_members(tar_path):
    p = subprocess.Popen(["zstd", "-dc", str(tar_path)], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=p.stdout, mode="r|") as t:
        for mem in t:
            if mem.isfile() and mem.name.endswith(".json"):
                yield json.load(t.extractfile(mem))
    p.wait()


def state_rows(tar_path, keep):
    """keep: set of (input_id, orf_id) to retain."""
    out = []
    for rec in json_members(tar_path):
        recs = rec if isinstance(rec, list) else [rec]
        for r in recs:
            iid = r["input_id"]
            for oid, f in r["features"].items():
                if (iid, oid) not in keep:
                    continue
                s = f["three_di"]["encoding"]
                cnt = np.array([s.count(ch) for ch in ALPHABET], np.int32)
                n = cnt.sum()
                pr = cnt[cnt > 0] / n
                hp = float(-(pr * np.log2(pr)).sum())
                k = int((cnt > 0).sum())
                out.append((iid, oid, n, k, hp, hp + (k - 1) / (2 * n * np.log(2)), *cnt))
    return out


def state_tasks():
    ph = [("phage", p) for p in sorted((IPH / "entropy_3di_results").glob("phg_*.tar.zst"))]
    files = [f.name[:-7] for f in bac_chunks()]
    pick = sorted(np.random.default_rng(SEED).choice(len(files), N_STATE_CHUNKS, replace=False))
    bc = [("bacteria", GD / "entropy_3di_results" / "bac" / f"{files[i]}.tar.zst") for i in pick]
    return ph + bc


def stage_states(a):
    tasks = state_tasks()[a.task::a.ntasks]
    cols = ["input_id", "orf_id", "n_states", "k_distinct", "h_plugin", "h_mm"] + \
        [f"c_{ch}" for ch in ALPHABET]
    ph = pd.read_parquet(WORK / "phage_pos.parquet", columns=["input_id", "orf_id"])
    for ds, tar in tasks:
        name = tar.name[:-8]
        if ds == "phage":
            keep = set(zip(ph.input_id, ph.orf_id))
            meta = None
        else:
            d = pd.read_csv(GD / "entropy_rows" / "bac" / f"{name}.tsv.gz", sep="\t",
                            usecols=["genome", "input_id", "orf_id", "start", "end", "strand",
                                     "aa_length", "in_genbank", "three_di_entropy", "contig_length"],
                            dtype={"in_genbank": str, "genome": str, "input_id": str})
            d = normalise(d[d.in_genbank == "True"].copy())
            c = pd.read_csv(GD / "cds_intervals" / "bac" / f"{name}.tsv", sep="\t",
                            usecols=["contig", "start", "end", "strand", "cds_id"],
                            dtype={"contig": str})
            c = prep_cds(c.rename(columns={"contig": "input_id"}))
            meta = cover(d[["genome", "input_id", "orf_id", "strand", "stop", "g_start", "g_end",
                            "aa_length", "three_di_entropy", "trunc"]],
                         c[["input_id", "strand", "stop", "cl"]])
            keep = set(zip(meta.input_id, meta.orf_id))
        s = pd.DataFrame(state_rows(tar, keep), columns=cols)
        if meta is not None:
            s = meta.drop(columns=["stop"]).merge(s, on=["input_id", "orf_id"], how="inner")
        s.insert(0, "dataset", ds)
        s.insert(1, "source", name)
        s.to_parquet(WORK / f"states_{ds}_{name}.parquet", index=False)
        print(f"  states {ds} {name}: {len(s):,} of {len(keep):,} positives", flush=True)


# ------------------------------------------------------------------ combine
SETS = {
    # name: (phage covc allowed, phage trunc allowed, bacteria covc allowed, bacteria trunc allowed)
    "all": ((0, 1, 2, 3), (0, 1), (0, 1, 2, 3), (0, 1)),
    "phage_cov80": ((3,), (0, 1), (0, 1, 2, 3), (0, 1)),
    "both_cov80": ((3,), (0, 1), (3,), (0, 1)),
    "both_cov80_untrunc": ((3,), (0,), (3,), (0,)),
}


def load_bac():
    sums = pd.concat([pd.read_parquet(p) for p in sorted(WORK.glob("bac_sums_*.parquet"))])
    hz = [np.load(p) for p in sorted(WORK.glob("bac_hist_*.npz"))]
    if sums.chunk.nunique() != 760:
        sys.exit(f"bac sums cover {sums.chunk.nunique()} chunks, expected 760")
    if "s_aa" not in sums.columns or any("cds_all_sum" not in z.files for z in hz):
        sys.exit("bac work files predate the uncapped length sums; rerun the bac stage")
    hist = sum(z["hist"] for z in hz)
    cds_all = sum(z["cds_all"] for z in hz)
    cds_pos = sum(z["cds_pos"] for z in hz)
    cds_all_sum = sum(int(z["cds_all_sum"]) for z in hz)
    cds_pos_sum = sum(int(z["cds_pos_sum"]) for z in hz)
    return sums, hist, cds_all, cds_pos, cds_all_sum, cds_pos_sum


def bac_by_length(sums, cov, tr, groups=False):
    s = sums[sums.covc.isin(cov) & sums.trunc.isin(tr)]
    keys = ["chunk", "li"] if groups else ["li"]
    g = s.groupby(keys, as_index=False)[["n", "s1", "s2"]].sum()
    return g


def ph_by_length(ph, cov, tr, groups=False):
    p = ph[ph.covc.isin(cov) & ph.trunc.isin(tr)]
    keys = ["grp", "li"] if groups else ["li"]
    return p.groupby(keys, as_index=False).agg(n=("three_di_entropy", "size"),
                                              s1=("three_di_entropy", "sum"))


def gap_estimates(nb, sb, np_, sp, lo, hi, min_n=5):
    """Arrays over li (length index). Returns raw (n-weighted within bin) and
    length-standardised gaps (phage weights; bacterial weights)."""
    L = np.arange(NL) + LMIN
    inb = (L >= lo) & (L <= hi)
    raw = sb[inb].sum() / nb[inb].sum() - sp[inb].sum() / np_[inb].sum()
    sup = inb & (nb >= min_n) & (np_ >= min_n)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sb / nb - sp / np_
    wp = np.where(sup, np_, 0.0)
    wb = np.where(sup, nb, 0.0)
    adj_p = (wp * np.nan_to_num(d)).sum() / wp.sum()
    adj_b = (wb * np.nan_to_num(d)).sum() / wb.sum()
    return raw, adj_p, adj_b, np_[sup].sum() / np_[inb].sum()


def jackknife(fn, ngroups, getter):
    """fn(arrays) -> tuple of estimates; getter(k) returns arrays with group k deleted."""
    reps = np.array([fn(*getter(k)) for k in range(ngroups)])
    m = reps.mean(axis=0)
    return np.sqrt((ngroups - 1) / ngroups * ((reps - m) ** 2).sum(axis=0))


def dense(df, col):
    a = np.zeros(NL)
    a[df.li.to_numpy()] = df[col].to_numpy()
    return a


def stage_combine(a):
    OUT.mkdir(parents=True, exist_ok=True)
    sums, hist, cds_all_b, cds_pos_b, cds_all_sum_b, cds_pos_sum_b = load_bac()
    ph = pd.read_parquet(WORK / "phage_pos.parquet")
    ph["li"] = lidx(ph.aa_length.to_numpy())
    pcds = pd.read_parquet(WORK / "phage_cds.parquet")
    print(f"  bacteria positives {int(sums.n.sum()):,}; phage positives {len(ph):,}", flush=True)

    # ---- check against the published figP04 matched curves
    pub = pd.read_csv(OUT / "figP04_gtdb_length_3di.tsv.gz", sep="\t")
    pub = pub[(pub.dataset == "bacteria") & (pub.cls == "matched")].set_index("aa_length")
    allb = bac_by_length(sums, *SETS["all"][2:])
    allb = allb[allb.li <= LMAX - LMIN].assign(L=lambda x: x.li + LMIN).set_index("L")
    j = allb.join(pub[["n", "mean"]].rename(columns={"n": "n_pub"}), how="inner")
    print(f"  check vs figP04 bacteria matched: {len(j)} lengths, n equal at "
          f"{(j.n == j.n_pub).mean():.4f}, max |mean diff| "
          f"{np.abs(j.s1 / j.n - j['mean']).max():.2e}", flush=True)

    # ---- 1. length distributions
    hist_rows = []
    bl = sums.groupby("li").n.sum()
    for name, cnt in (("bacteria", bl), ("phage", ph.groupby("li").size())):
        for li, n in cnt.items():
            hist_rows.append(("orf_positive", name, int(li) + LMIN, int(n)))
    pc = np.bincount(np.clip(pcds.aa.to_numpy(), 0, CDS_LMAX), minlength=CDS_LMAX + 1)
    ok = ph.covc > 0
    pp = np.bincount(np.clip((ph.cl[ok].to_numpy() // 3 - 1).astype(int), 0, CDS_LMAX),
                     minlength=CDS_LMAX + 1)
    for name, kind, arr in (("bacteria", "cds_all", cds_all_b), ("phage", "cds_all", pc),
                            ("bacteria", "cds_of_positive", cds_pos_b), ("phage", "cds_of_positive", pp)):
        for L in np.nonzero(arr)[0]:
            hist_rows.append((kind, name, int(L), int(arr[L])))
    H = pd.DataFrame(hist_rows, columns=["kind", "dataset", "aa_length", "n"])
    H.loc[(H.kind == "orf_positive") & (H.aa_length == LTOP), "aa_length"] = LTOP  # >= 1000 pooled
    write(H.sort_values(["kind", "dataset", "aa_length"]), "figPS04_length_hist.tsv.gz")

    def qstats(sub, mean_override=None):
        x = sub.aa_length.to_numpy(); w = sub.n.to_numpy().astype(float)
        o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum()
        q = lambda p: int(x[np.searchsorted(c, p)])
        mu = (w * x).sum() / w.sum() if mean_override is None else mean_override
        return w.sum(), q(0.25), q(0.5), q(0.75), mu

    srows = []
    ok = ph.covc > 0
    mean_all = {
        ("orf_positive", "bacteria"): float(sums.s_aa.sum() / sums.n.sum()),
        ("orf_positive", "phage"): float(ph.aa_length.mean()),
        ("cds_all", "bacteria"): float(cds_all_sum_b / cds_all_b.sum()),
        ("cds_all", "phage"): float(pcds.aa.mean()),
        ("cds_of_positive", "bacteria"): float(cds_pos_sum_b / cds_pos_b.sum()),
        ("cds_of_positive", "phage"): float(
            np.maximum(ph.cl.to_numpy()[ok].astype(np.int64) // 3 - 1, 0).mean()
        ),
    }
    for kind in ("orf_positive", "cds_all", "cds_of_positive"):
        for ds in ("bacteria", "phage"):
            sub = H[(H.kind == kind) & (H.dataset == ds)]
            n, q1, q2, q3, mu = qstats(sub, mean_override=mean_all[(kind, ds)])
            srows.append(dict(kind=kind, dataset=ds, bin="all", n=int(n), q25=q1, median=q2,
                              q75=q3, mean=mu))
            if kind == "orf_positive":
                for lo, hi in BINS:
                    s2 = sub[sub.aa_length.between(lo, hi)]
                    n, q1, q2, q3, mu = qstats(s2)
                    srows.append(dict(kind=kind, dataset=ds, bin=f"{lo}-{hi}", n=int(n), q25=q1,
                                      median=q2, q75=q3, mean=mu))
    write(pd.DataFrame(srows), "tabPS04_length_summary.tsv")

    # ---- 2/3. raw and exact-length-standardised gaps, grouped jackknife
    grows, curve = [], []
    for sname, (pcov, ptr, bcov, btr) in SETS.items():
        b = bac_by_length(sums, bcov, btr)
        nb, sb = dense(b, "n"), dense(b, "s1")
        bg = bac_by_length(sums, bcov, btr, groups=True)
        chunks = np.sort(bg.chunk.unique())
        bgn = {c: (dense(x, "n"), dense(x, "s1")) for c, x in bg.groupby("chunk")}
        p = ph_by_length(ph, pcov, ptr)
        np_, sp = dense(p, "n"), dense(p, "s1")
        pg = ph_by_length(ph, pcov, ptr, groups=True)
        pgn = {g: (dense(x, "n"), dense(x, "s1")) for g, x in pg.groupby("grp")}
        for lo, hi in BINS + [(LMIN, LMAX)]:
            est = gap_estimates(nb, sb, np_, sp, lo, hi)
            f = lambda nb_, sb_, np2, sp2: gap_estimates(nb_, sb_, np2, sp2, lo, hi)[:3]
            zb = (np.zeros(NL), np.zeros(NL))
            se_b = jackknife(f, len(chunks), lambda k: (nb - bgn[chunks[k]][0], sb - bgn[chunks[k]][1],
                                                        np_, sp))
            se_p = jackknife(f, N_PHAGE_GROUPS, lambda k: (nb, sb, np_ - pgn.get(k, zb)[0],
                                                           sp - pgn.get(k, zb)[1]))
            se = np.sqrt(se_b ** 2 + se_p ** 2)
            L = np.arange(NL) + LMIN
            inb = (L >= lo) & (L <= hi)
            grows.append(dict(set=sname, bin=f"{lo}-{hi}", n_bacteria=int(nb[inb].sum()),
                              n_phage=int(np_[inb].sum()),
                              phage_share_in_support=est[3],
                              mean_bacteria=sb[inb].sum() / nb[inb].sum(),
                              mean_phage=sp[inb].sum() / np_[inb].sum(),
                              gap_raw=est[0], se_raw=se[0],
                              gap_matched_phage_w=est[1], se_matched_phage_w=se[1],
                              gap_matched_bacteria_w=est[2], se_matched_bacteria_w=se[2],
                              explained_by_length=1 - est[1] / est[0]))
        with np.errstate(invalid="ignore", divide="ignore"):
            for i in range(NL - 1):
                curve.append((sname, i + LMIN, int(nb[i]), sb[i] / nb[i] if nb[i] else np.nan,
                              int(np_[i]), sp[i] / np_[i] if np_[i] else np.nan))
    write(pd.DataFrame(grows), "tabPS04_length_matched_gap.tsv")
    write(pd.DataFrame(curve, columns=["set", "aa_length", "n_bacteria", "mean_bacteria",
                                       "n_phage", "mean_phage"]), "figPS04_gap_by_length.tsv.gz")

    # ---- 4a. length-conditioned percentile of each phage positive in the bacterial distribution
    prow = []
    for sname, (pcov, ptr, bcov, btr) in SETS.items():
        hb = hist[:, list(bcov)][:, :, list(btr)].sum(axis=(1, 2)).astype(float)   # (NL, NH)
        tot = hb.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cdf_mid = (np.cumsum(hb, axis=1) - hb / 2) / tot      # mid-rank percentile per bin
            mu = (hb * (np.arange(NH) + 0.5) * HB).sum(axis=1) / tot[:, 0]
            sd = np.sqrt((hb * ((np.arange(NH) + 0.5) * HB - mu[:, None]) ** 2).sum(axis=1) / tot[:, 0])
        p = ph[ph.covc.isin(pcov) & ph.trunc.isin(ptr)]
        li, hi_ = p.li.to_numpy(), hidx(p.three_di_entropy.to_numpy())
        pct = cdf_mid[li, hi_]
        z = (p.three_di_entropy.to_numpy() - mu[li]) / sd[li]
        for lo, hi in BINS + [(LMIN, LMAX)]:
            sel = (p.aa_length.to_numpy() >= lo) & (p.aa_length.to_numpy() <= hi)
            prow.append(dict(set=sname, bin=f"{lo}-{hi}", n_phage=int(sel.sum()),
                             mean_percentile=np.nanmean(pct[sel]),
                             median_percentile=np.nanmedian(pct[sel]),
                             share_below_bacterial_median=np.nanmean(pct[sel] < 0.5),
                             mean_z=np.nanmean(z[sel])))
    write(pd.DataFrame(prow), "tabPS04_percentile.tsv")

    # ---- 4b. 3Di strings: distinct states, Miller-Madow, state usage (length-matched)
    st = pd.concat([pd.read_parquet(p) for p in sorted(WORK.glob("states_*.parquet"))],
                   ignore_index=True)
    phs = ph[["genome", "input_id", "orf_id", "aa_length", "covc", "trunc"]]
    sp_ = st[st.dataset == "phage"][["dataset", "source", "input_id", "orf_id", "n_states",
                                     "k_distinct", "h_plugin", "h_mm"] +
                                    [f"c_{c}" for c in ALPHABET]].merge(phs, on=["input_id", "orf_id"])
    sb_ = st[st.dataset == "bacteria"]
    st = pd.concat([sp_, sb_], ignore_index=True)
    st["li"] = lidx(st.aa_length.to_numpy())
    srow, urow = [], []
    for sname, (pcov, ptr, bcov, btr) in SETS.items():
        P = st[(st.dataset == "phage") & st.covc.isin(pcov) & st.trunc.isin(ptr)]
        B = st[(st.dataset == "bacteria") & st.covc.isin(bcov) & st.trunc.isin(btr)]
        for lo, hi in BINS + [(LMIN, LMAX)]:
            Pb, Bb = P[P.aa_length.between(lo, hi)], B[B.aa_length.between(lo, hi)]
            r = dict(set=sname, bin=f"{lo}-{hi}", n_phage=len(Pb), n_bacteria=len(Bb),
                     n_bacterial_genomes=Bb.genome.nunique() if "genome" in Bb else 0)
            for col in ("h_plugin", "h_mm", "k_distinct"):
                gp = Pb.groupby("li")[col].agg(["size", "mean"])
                gb = Bb.groupby("li")[col].agg(["size", "mean"])
                j = gp.join(gb, lsuffix="_p", rsuffix="_b", how="inner")
                j = j[(j.size_p >= 5) & (j.size_b >= 5)]
                w = j.size_p / j.size_p.sum()
                r[f"{col}_gap_raw"] = Bb[col].mean() - Pb[col].mean()
                r[f"{col}_gap_matched"] = float((w * (j.mean_b - j.mean_p)).sum())
            r["bias_term_gap_matched"] = r["h_mm_gap_matched"] - r["h_plugin_gap_matched"]
            srow.append(r)
            if sname == "both_cov80_untrunc":
                cp = Pb[[f"c_{c}" for c in ALPHABET]].sum()
                cb = Bb[[f"c_{c}" for c in ALPHABET]].sum()
                for ch in ALPHABET:
                    urow.append(dict(bin=f"{lo}-{hi}", state=ch,
                                     freq_phage=cp[f"c_{ch}"] / cp.sum(),
                                     freq_bacteria=cb[f"c_{ch}"] / cb.sum()))
    write(pd.DataFrame(srow), "tabPS04_states.tsv")
    write(pd.DataFrame(urow), "figPS04_state_usage.tsv")
    prov = {"seed": SEED, "phage_jackknife_groups": N_PHAGE_GROUPS,
            "bacterial_jackknife_groups": int(sums.chunk.nunique()),
            "state_chunks": [t[1].name for t in state_tasks() if t[0] == "bacteria"],
            "histogram_bin_bits": HB, "exact_length_range": [LMIN, LMAX], "support_min_n": 5,
            "sets": {k: dict(zip(["phage_covc", "phage_trunc", "bacteria_covc", "bacteria_trunc"],
                                 [list(x) for x in v])) for k, v in SETS.items()}}
    (OUT / "figPS04_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print("  out  figPS04_provenance.json", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["bac", "phage", "states", "combine"])
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    a = ap.parse_args()
    {"bac": stage_bac, "phage": stage_phage, "states": stage_states,
     "combine": stage_combine}[a.stage](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
