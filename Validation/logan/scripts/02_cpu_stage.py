#!/usr/bin/env python3
"""CPU stage for one LOGAN accession: contigs -> Prodigal -> ORFs -> matching.

  selected      -> downloaded     curl to .partial, byte count == S3
                                  Content-Length, zstd -t, rename
  downloaded    -> validated      decompress, count every contig and base
                                  against the LOGAN v1.2 seqstats parquet,
                                  write the >= 120 nt contigs
  validated     -> prodigal_done  prodigal -q -p meta (LOGAN's command), plus
                                  -f gff so partial flags, start type, score
                                  and the per-contig transl_table are kept
  prodigal_done -> orfs_called    genome_entropy's own find_orfs and
                                  translate_orfs over the same contigs, and
                                  every ORF classified against Prodigal

Each transition is skipped if the state file already records it, so a rerun
resumes where the last one stopped.

OUTPUTS in work/<acc>/:

  contigs.fa.zst            the LOGAN object as downloaded
  prodigal.gff.zst          Prodigal GFF
  prodigal.faa.zst          Prodigal proteins (what LOGAN publishes)
  prodigal_genes.tsv.zst    one row per Prodigal gene, normalised coordinates,
                            and the ORF it was matched to, if any
  orfs.tsv.zst              one row per ORF with its amino-acid sequence;
                            the input to the GPU stage
  orf_prodigal.tsv.zst      one row per ORF, no sequence: the Prodigal+/-
                            label and the coordinate relationship behind it

COORDINATES. Every table here uses 0-based half-open forward-axis start/end.
ORFs are converted once, by genome_entropy.io.genbank.normalise_orf_interval;
Prodigal's 1-based inclusive start/end become (start - 1, end). No consumer
should convert again.

MATCHING. An ORF is Prodigal+ when a Prodigal gene on the same contig and
strand ends at the same stop (identical 3' coordinate) AND the package's
coordinate-anchored matcher accepts the pair (same codon phase, >= 90% overlap
of the shorter, >= 98% shared amino-acid identity). Both genome_entropy ORFs
(stop to stop) and Prodigal genes end at a stop codon, or at the same
contig edge when open, so the 3' end is the exact join key and the 5' end is
free to differ, which is where callers legitimately disagree.

Every other ORF is classified by its largest overlap with any Prodigal gene,
using 20_orf_context.py::frame_class from the GTDB analysis (#92), which is
the single authority for strand-dependent frame anchoring.

  02_cpu_stage.py SRR12770874 [--repro-check]
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logan_common as lc  # noqa: E402

from genome_entropy.io.genbank import (  # noqa: E402
    GenBankCDS,
    evaluate_orf_genbank_cds_match,
    normalise_orf_interval,
)
from genome_entropy.orf.finder import find_orfs  # noqa: E402
from genome_entropy.orf.types import OrfRecord  # noqa: E402
from genome_entropy.translate.translator import translate_orfs  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "orf_context", HERE.parent.parent / "gtdb" / "scripts" / "20_orf_context.py")
_orf_context = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orf_context)
frame_class = _orf_context.frame_class

# find_orfs logs every contig batch at INFO; keep the job logs to what matters.
logging.getLogger("genome_entropy").setLevel(logging.WARNING)

ORF_BATCH_BP = 20_000_000    # contigs per get_orfs call; find_orfs has a 300 s timeout
MIN_OVERLAP_BP = 30          # below this an overlap is adjacency, not a shadow


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def run(cmd: List[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def write_tsv_zst(df: pd.DataFrame, path: Path, level: int = 6) -> None:
    """Write a DataFrame as zstd TSV via a .partial name, then zstd -t."""
    tmp = path.with_name(path.name + ".partial")
    proc = subprocess.Popen(["zstd", "-q", f"-{level}", "-T4", "-f", "-o", str(tmp)],
                            stdin=subprocess.PIPE)
    with io.TextIOWrapper(proc.stdin, encoding="ascii", newline="") as fh:
        df.to_csv(fh, sep="\t", index=False, lineterminator="\n")
    if proc.wait() != 0:
        raise RuntimeError(f"zstd failed writing {path}")
    run(["zstd", "-tq", str(tmp)])
    os.replace(tmp, path)


def read_fasta_stream(fh) -> Iterator[Tuple[str, str]]:
    name, parts = None, []
    for line in fh:
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(parts)
            name = line[1:].split(None, 1)[0]
            parts = []
        else:
            parts.append(line.strip())
    if name is not None:
        yield name, "".join(parts)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_download(acc: str, wd: Path) -> None:
    url = lc.LOGAN_CONTIG_URL.format(acc=acc)
    head = subprocess.run(["curl", "-sSfI", url], capture_output=True, text=True,
                          check=True).stdout
    hdr = {k.strip().lower(): v.strip() for k, v in
           (l.split(":", 1) for l in head.splitlines() if ":" in l)}
    want = int(hdr["content-length"])
    dest = wd / "contigs.fa.zst"
    tmp = wd / "contigs.fa.zst.partial"
    run(["curl", "-sSf", "--retry", "5", "--retry-delay", "10", "-o", str(tmp), url])
    got = tmp.stat().st_size
    if got != want:
        raise RuntimeError(f"{acc}: downloaded {got} bytes, S3 says {want}")
    run(["zstd", "-tq", str(tmp)])
    os.replace(tmp, dest)
    lc.advance(acc, "downloaded", download_bytes=got, s3_etag=hdr.get("etag", ""),
               s3_last_modified=hdr.get("last-modified", ""), url=url)


def stage_validate(acc: str, wd: Path) -> None:
    st = lc.load_state(acc)["info"]
    n_all = bp_all = n_keep = bp_keep = 0
    n_amb = 0
    out = wd / "contigs.ge120.fa"
    proc = subprocess.Popen(["zstd", "-dcq", str(wd / "contigs.fa.zst")],
                            stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    with open(out, "w") as fo:
        for name, seq in read_fasta_stream(proc.stdout):
            n_all += 1
            bp_all += len(seq)
            if len(seq) >= lc.MIN_CONTIG_NT:
                n_keep += 1
                bp_keep += len(seq)
                if re.search(r"[^ACGTacgt]", seq):
                    n_amb += 1
                fo.write(f">{name}\n{seq}\n")
    if proc.wait() != 0:
        raise RuntimeError(f"{acc}: zstd decompression failed")
    exp_n = st.get("expected_contig_count")
    exp_bp = st.get("expected_contig_bp")
    # LOGAN seqstats are computed from the same objects, so they should agree
    # exactly. Anything beyond 0.1% means we are not looking at the release
    # the metadata describes and the accession is stopped here.
    def rel(a, b):
        return None if not b else abs(a - b) / b
    dn, dbp = rel(n_all, exp_n), rel(bp_all, exp_bp)
    ok = (dn is None or dn <= 1e-3) and (dbp is None or dbp <= 1e-3)
    info = dict(contig_count=n_all, contig_bp=bp_all,
                contigs_ge120=n_keep, bp_ge120=bp_keep,
                contigs_with_non_acgt=n_amb,
                contig_count_rel_diff=dn, contig_bp_rel_diff=dbp)
    if not ok:
        lc.advance(acc, "downloaded", validation_failed=info)
        raise RuntimeError(f"{acc}: contig stats disagree with seqstats: {info}")
    if n_keep == 0:
        lc.advance(acc, "downloaded", validation_failed=info)
        raise RuntimeError(f"{acc}: no contigs >= {lc.MIN_CONTIG_NT} nt")
    lc.advance(acc, "validated", **info)


def run_prodigal(fasta: Path, faa: Path, gff: Path) -> None:
    run(["prodigal", *lc.PRODIGAL_ARGS, "-i", str(fasta), "-a", str(faa),
         "-f", "gff", "-o", str(gff)])


def stage_prodigal(acc: str, wd: Path, repro_check: bool) -> None:
    fasta = wd / "contigs.ge120.fa"
    faa, gff = wd / "prodigal.faa", wd / "prodigal.gff"
    t0 = time.time()
    run_prodigal(fasta, faa, gff)
    secs = time.time() - t0
    info = {"prodigal_seconds": round(secs, 1),
            "prodigal_command": "prodigal " + " ".join(lc.PRODIGAL_ARGS)
            + " -i contigs.ge120.fa -a prodigal.faa -f gff -o prodigal.gff",
            "prodigal_faa_sha256": lc.sha256(faa),
            "prodigal_gff_sha256": lc.sha256(gff)}
    if repro_check:
        faa2, gff2 = wd / "prodigal.rep.faa", wd / "prodigal.rep.gff"
        run_prodigal(fasta, faa2, gff2)
        same = (lc.sha256(faa2) == info["prodigal_faa_sha256"]
                and lc.sha256(gff2) == info["prodigal_gff_sha256"])
        info["prodigal_reproducible"] = same
        faa2.unlink()
        gff2.unlink()
        if not same:
            raise RuntimeError(f"{acc}: two Prodigal runs differ")
    genes = parse_prodigal(gff, faa)
    info["prodigal_genes"] = len(genes)
    info["prodigal_transl_tables"] = {str(k): int(v) for k, v in
                                      genes.groupby("transl_table").contig.nunique().items()}
    info["prodigal_partial"] = {str(k): int(v) for k, v in genes.partial.value_counts().items()}
    for p in (faa, gff):
        run(["zstd", "-q", "-6", "-f", "--rm", str(p)])
    lc.advance(acc, "prodigal_done", **info)


_ATTR = re.compile(r"([^=;]+)=([^;]*)")


def open_text(path: Path):
    """Plain or .zst text, so the ORF stage can re-parse compressed Prodigal output."""
    if str(path).endswith(".zst"):
        import zstandard
        return zstandard.open(path, "rt")
    return open(path)


def parse_prodigal(gff: Path, faa: Path) -> pd.DataFrame:
    """Prodigal GFF + FAA -> one row per gene, normalised coordinates."""
    rows = []
    table = None
    model = ""
    with open_text(gff) as fh:
        for line in fh:
            if line.startswith("# Model Data:"):
                m = re.search(r"transl_table=(\d+)", line)
                table = int(m.group(1)) if m else None
                mm = re.search(r'model="([^"]*)"', line)
                model = mm.group(1) if mm else ""
                continue
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            a = dict(_ATTR.findall(f[8]))
            rows.append((f[0], a["ID"], int(f[3]) - 1, int(f[4]), f[6],
                         a.get("partial", ""), a.get("start_type", ""),
                         a.get("rbs_motif", ""), float(a.get("conf", "nan")),
                         float(a.get("score", "nan")), table, model))
    genes = pd.DataFrame(rows, columns=["contig", "gene_id", "start", "end", "strand",
                                        "partial", "start_type", "rbs_motif",
                                        "conf", "score", "transl_table", "model"])
    prot: Dict[str, str] = {}
    with open_text(faa) as fh:
        gid, parts = None, []
        for line in fh:
            if line.startswith(">"):
                if gid is not None:
                    prot[gid] = "".join(parts)
                gid = re.search(r"ID=([^;]+)", line).group(1)
                parts = []
            else:
                parts.append(line.strip())
        if gid is not None:
            prot[gid] = "".join(parts)
    genes["protein"] = genes.gene_id.map(prot)
    if genes.protein.isna().any():
        raise RuntimeError(f"{int(genes.protein.isna().sum())} GFF genes have no FAA protein")
    return genes


def iter_contig_batches(fasta: Path, batch_bp: int) -> Iterator[Dict[str, str]]:
    batch: Dict[str, str] = {}
    size = 0
    with open(fasta) as fh:
        for name, seq in read_fasta_stream(fh):
            batch[name] = seq
            size += len(seq)
            if size >= batch_bp:
                yield batch
                batch, size = {}, 0
    if batch:
        yield batch


def call_orfs(acc: str, wd: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """genome_entropy's find_orfs + translate_orfs over the filtered contigs.

    TWO get_orfs BEHAVIOURS ARE CORRECTED HERE, both verified on LOGAN contigs.

    1. MINIMUM LENGTH (issue #103). get_orfs -l is in amino acids and keeps an
       ORF when strlen(aa, including any '*') > l. find_orfs passes its
       nucleotide argument straight through, and `genome_entropy run` passes
       min_aa * 3, so the package default is an effective ~90 aa floor. We
       want the documented 30 aa: pass 29 so both stop-terminated (29 aa + '*')
       and contig-open (30 aa) ORFs survive, then keep aa_len >= 30 exactly.

    2. OPEN 3' ENDS. For an ORF that runs off the contig without a stop,
       get_orfs reports end = contig_len + 1 whatever the frame, so the span
       is neither codon-aligned nor inside the contig, and
       normalise_orf_interval rightly refuses it. The amino-acid string is
       correct. The true span is start .. start + 3 * len(aa incl '*') - 1,
       which is asserted to equal the reported end for every stop-terminated
       ORF before it is applied to the open ones.
    """
    frames = []
    n_mismatch = n_end_fixed = 0
    for batch in iter_contig_batches(wd / "contigs.ge120.fa", ORF_BATCH_BP):
        lens = {k: len(v) for k, v in batch.items()}
        orfs = find_orfs(batch, table_id=lc.GENETIC_CODE,
                         min_nt_length=lc.MIN_ORF_AA - 1)   # read as aa, see (1)
        for o in orfs:
            codon_end = o.start + 3 * len(o.aa_sequence) - 1
            if o.has_stop_codon:
                if codon_end != o.end:
                    raise RuntimeError(f"{acc}: stop-terminated ORF {o.parent_id} "
                                       f"{o.start}-{o.end} is not codon-aligned")
            elif codon_end != o.end:
                o.end = codon_end
                o.nt_sequence = o.nt_sequence[: 3 * len(o.aa_sequence)]
                n_end_fixed += 1
        orfs = [o for o in orfs if len(o.aa_sequence.rstrip("*")) >= lc.MIN_ORF_AA]
        prots = translate_orfs(orfs, table_id=lc.GENETIC_CODE)
        rec = []
        for o, p in zip(orfs, prots):
            L = lens[o.parent_id]
            iv = normalise_orf_interval(o.start, o.end, o.strand, L)
            if o.aa_sequence.rstrip("*") != p.aa_sequence:
                n_mismatch += 1
            five_open = iv.start < 3 if o.strand == "+" else L - iv.end < 3
            rec.append((o.parent_id, f"{o.parent_id}:{iv.start}-{iv.end}:{o.strand}",
                        iv.start, iv.end, o.strand, o.frame, o.start, o.end, L,
                        int(o.has_stop_codon), int(five_open), p.aa_length,
                        p.aa_sequence))
        frames.append(pd.DataFrame(rec, columns=[
            "contig", "orf_id", "start", "end", "strand", "frame", "raw_start",
            "raw_end", "contig_len", "stop", "five_open", "aa_len", "aa_sequence"]))
        del orfs, prots, rec
    df = pd.concat(frames, ignore_index=True)
    if df.orf_id.duplicated().any():
        raise RuntimeError(f"{acc}: duplicate ORF ids")
    return df, {"translation_mismatches": n_mismatch, "open_end_corrected": n_end_fixed}


def overlap_scan(orfs: pd.DataFrame, genes: pd.DataFrame, offsets: Dict[str, int]):
    """Largest-overlap Prodigal gene per ORF, and the largest overlap on each strand.

    Contigs are laid end to end on one axis (with a gap) so every interval
    query is a single sorted search; nothing can overlap across a contig
    boundary because of the gap.
    """
    n = len(orfs)
    best_ov = np.zeros(n, np.int64)
    best_idx = np.full(n, -1, np.int64)
    same_ov = np.zeros(n, np.int64)
    opp_ov = np.zeros(n, np.int64)
    if genes.empty:
        return best_ov, best_idx, same_ov, opp_ov
    goff = genes.contig.map(offsets).to_numpy(np.int64)
    gs = goff + genes.start.to_numpy(np.int64)
    ge = goff + genes.end.to_numpy(np.int64)
    order = np.argsort(gs, kind="stable")
    gs, ge = gs[order], ge[order]
    gstrand = genes.strand.to_numpy()[order]
    cm = np.maximum.accumulate(ge)
    ooff = orfs.contig.map(offsets).to_numpy(np.int64)
    os_ = ooff + orfs.start.to_numpy(np.int64)
    oe = ooff + orfs.end.to_numpy(np.int64)
    ostrand = orfs.strand.to_numpy()
    k = np.searchsorted(gs, oe, side="left") - 1
    active = np.nonzero(k >= 0)[0]
    while active.size:
        idx = k[active]
        ov = np.minimum(oe[active], ge[idx]) - np.maximum(os_[active], gs[idx])
        ov = np.maximum(ov, 0)
        same = gstrand[idx] == ostrand[active]
        better = ov > best_ov[active]
        best_ov[active[better]] = ov[better]
        best_idx[active[better]] = order[idx[better]]
        s_upd = same & (ov > same_ov[active])
        same_ov[active[s_upd]] = ov[s_upd]
        o_upd = (~same) & (ov > opp_ov[active])
        opp_ov[active[o_upd]] = ov[o_upd]
        k[active] -= 1
        nk = k[active]
        cont = nk >= 0
        cont[cont] = cm[nk[cont]] > os_[active[cont]]
        active = active[cont]
    return best_ov, best_idx, same_ov, opp_ov


def classify(acc: str, orfs: pd.DataFrame, genes: pd.DataFrame):
    genes = genes.copy()
    genes["three_prime"] = np.where(genes.strand == "+", genes.end, genes.start)
    o3 = np.where(orfs.strand == "+", orfs.end, orfs.start)
    key_o = pd.DataFrame({"i": np.arange(len(orfs)), "contig": orfs.contig.to_numpy(),
                          "strand": orfs.strand.to_numpy(), "three_prime": o3})
    pairs = key_o.merge(genes.reset_index()[["index", "contig", "strand", "three_prime"]],
                        on=["contig", "strand", "three_prime"], how="inner")
    if pairs.i.duplicated().any() or pairs["index"].duplicated().any():
        raise RuntimeError(f"{acc}: a 3' end is shared by more than one ORF or gene")

    match_gene = np.full(len(orfs), -1, np.int64)
    identity = np.full(len(orfs), np.nan)
    init_sub = np.zeros(len(orfs), np.int8)
    reason = np.full(len(orfs), "", dtype=object)
    for i, gi in zip(pairs.i.to_numpy(), pairs["index"].to_numpy()):
        o = orfs.iloc[i]
        g = genes.iloc[gi]
        prot = g.protein.rstrip("*")
        # Prodigal writes M for the initiator codon even when it is GTG or
        # TTG. The ORF translates that codon as V or L. Biologically the
        # initiator is Met, so the first residue is aligned to the ORF's
        # before the identity test; without this one mismatch fails the 98%
        # threshold for every gene shorter than 50 aa with a non-ATG start.
        off = o.aa_len - len(prot)
        if (g.partial[:1] if g.strand == "+" else g.partial[1:2]) == "0" \
                and prot[:1] == "M" and 0 <= off < o.aa_len \
                and o.aa_sequence[off] != "M":
            prot = o.aa_sequence[off] + prot[1:]
            init_sub[i] = 1
        orf_rec = OrfRecord(parent_id=o.contig, orf_id=o.orf_id, start=int(o.raw_start),
                            end=int(o.raw_end), strand=o.strand, frame=int(o.frame),
                            nt_sequence="", aa_sequence=o.aa_sequence,
                            table_id=lc.GENETIC_CODE, has_start_codon=False,
                            has_stop_codon=bool(o.stop))
        cds = GenBankCDS(parent_id=g.contig, start=int(g.start), end=int(g.end),
                         strand=g.strand, protein_sequence=prot,
                         record_length=int(o.contig_len), feature_id=g.gene_id,
                         translation_table=int(g.transl_table or 11),
                         partial=g.partial != "00")
        r = evaluate_orf_genbank_cds_match(orf_rec, cds)
        identity[i] = r.identity
        reason[i] = r.reason
        if r.matched:
            match_gene[i] = gi

    offsets, pos = {}, 0
    for c, L in orfs.groupby("contig", sort=False).contig_len.first().items():
        offsets[c] = pos
        pos += int(L) + 1000
    for c in genes.contig.unique():
        if c not in offsets:
            offsets[c] = pos
            pos += 10**9
    best_ov, best_idx, same_ov, opp_ov = overlap_scan(orfs, genes, offsets)

    relation = np.full(len(orfs), "intergenic", dtype=object)
    gs = genes.start.to_numpy()
    ge = genes.end.to_numpy()
    gst = genes.strand.to_numpy()
    for i in np.nonzero(best_idx >= 0)[0]:
        gi = best_idx[i]
        relation[i] = frame_class(int(orfs.start.iat[i]), int(orfs.end.iat[i]),
                                  orfs.strand.iat[i], int(gs[gi]), int(ge[gi]),
                                  gst[gi], (ge[gi] - gs[gi]) % 3 == 0)
    # A best overlap shorter than MIN_OVERLAP_BP is adjacency (genes commonly
    # share a few bases at their ends), not a shadow; those ORFs are labelled
    # marginal_overlap rather than given a frame class. The raw overlap
    # columns are kept so any other threshold can be applied downstream.
    real = best_ov >= MIN_OVERLAP_BP
    cls = np.select(
        [match_gene >= 0,
         (best_ov > 0) & ~real,
         relation == "same strand, same frame",
         relation == "same strand, frameshift",
         relation == "opposite strand",
         relation == "same strand, frame undefined"],
        ["prodigal_match", "marginal_overlap", "same_frame_unmatched", "frameshift_overlap",
         "antisense_overlap", "frame_undefined_overlap"],
        default="intergenic")

    gid = genes.gene_id.to_numpy()
    gtt = genes.transl_table.fillna(0).astype(int).to_numpy()
    table = pd.DataFrame({
        "accession": acc,
        "contig": orfs.contig, "orf_id": orfs.orf_id,
        "start": orfs.start, "end": orfs.end, "strand": orfs.strand,
        "contig_len": orfs.contig_len, "stop": orfs.stop,
        "five_open": orfs.five_open, "aa_len": orfs.aa_len,
        "prodigal": np.where(match_gene >= 0, "Prodigal+", "Prodigal-"),
        "prodigal_class": cls,
        "match_gene": np.where(match_gene >= 0, gid[np.maximum(match_gene, 0)], ""),
        "same_stop_identity": identity,
        "same_stop_reason": reason,
        "initiator_substituted": init_sub,
        "best_gene": np.where(best_idx >= 0, gid[np.maximum(best_idx, 0)], ""),
        "best_relation": np.where(best_idx >= 0, relation, ""),
        "best_overlap_bp": best_ov,
        # Prodigal -p meta picks one of 50 models per contig and some use
        # genetic code 4 (TGA = Trp). Their genes read through TGA, so they can
        # never share a stop with a table-11 ORF: same_frame_unmatched is
        # almost entirely these.
        "best_gene_transl_table": np.where(best_idx >= 0, gtt[np.maximum(best_idx, 0)], 0),
        "same_strand_overlap_bp": same_ov,
        "opp_strand_overlap_bp": opp_ov,
    })
    genes_out = genes.drop(columns=["protein", "three_prime"]).copy()
    genes_out.insert(0, "accession", acc)
    genes_out["aa_len"] = genes.protein.str.rstrip("*").str.len()
    matched = pd.Series(orfs.orf_id.to_numpy()[match_gene >= 0],
                        index=match_gene[match_gene >= 0])
    genes_out["matched_orf"] = genes_out.index.map(matched).fillna("")
    same_stop = pd.Series(1, index=pairs["index"].to_numpy())
    genes_out["orf_same_stop"] = genes_out.index.map(same_stop).fillna(0).astype(int)
    return table, genes_out


def stage_orfs(acc: str, wd: Path) -> None:
    t0 = time.time()
    orfs, extra = call_orfs(acc, wd)
    t_orf = time.time() - t0
    if extra["translation_mismatches"]:
        raise RuntimeError(f"{acc}: {extra['translation_mismatches']} ORFs translate "
                           "differently from get_orfs")
    genes = parse_prodigal(wd / "prodigal.gff.zst", wd / "prodigal.faa.zst")
    table, genes_out = classify(acc, orfs, genes)
    t_all = time.time() - t0

    write_tsv_zst(orfs[["contig", "orf_id", "start", "end", "strand", "contig_len",
                        "stop", "aa_len", "aa_sequence"]], wd / "orfs.tsv.zst")
    write_tsv_zst(table, wd / "orf_prodigal.tsv.zst")
    write_tsv_zst(genes_out, wd / "prodigal_genes.tsv.zst")

    counts = table.prodigal_class.value_counts().to_dict()
    info = dict(orfs=int(len(orfs)), orf_aa=int(orfs.aa_len.sum()),
                orf_min_aa=int(orfs.aa_len.min()), orf_max_aa=int(orfs.aa_len.max()),
                orfs_open_5p=int(orfs.five_open.sum()),
                orfs_no_stop=int((orfs.stop == 0).sum()),
                prodigal_plus=int((table.prodigal == "Prodigal+").sum()),
                prodigal_class_counts={k: int(v) for k, v in counts.items()},
                same_stop_pairs=int(genes_out.orf_same_stop.sum()),
                same_stop_rejected=int(((table.same_stop_reason != "")
                                        & (table.prodigal == "Prodigal-")).sum()),
                genes_without_orf=int((genes_out.matched_orf == "").sum()),
                initiator_substituted=int(table.initiator_substituted.sum()),
                orf_seconds=round(t_orf, 1), orf_and_match_seconds=round(t_all, 1),
                orfs_sha256=lc.sha256(wd / "orfs.tsv.zst"),
                **extra)
    lc.advance(acc, "orfs_called", **info)
    print({k: v for k, v in info.items() if k != "orfs_sha256"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("accession")
    ap.add_argument("--repro-check", action="store_true",
                    help="run Prodigal twice and require identical output")
    ap.add_argument("--keep-fasta", action="store_true",
                    help="keep contigs.ge120.fa after the ORF stage")
    args = ap.parse_args()
    acc = args.accession
    wd = lc.work_dir(acc)
    wd.mkdir(parents=True, exist_ok=True)
    st = lc.load_state(acc)
    if st.get("state") is None:
        sys.exit(f"{acc}: not selected (no state file)")
    if not lc.reached(st, "downloaded"):
        stage_download(acc, wd)
    st = lc.load_state(acc)
    # The filtered FASTA is an intermediate the later stages need; if a rerun
    # finds it gone, regenerate it by re-validating rather than failing.
    if not lc.reached(st, "validated") or (
            not lc.reached(st, "orfs_called") and not (wd / "contigs.ge120.fa").exists()):
        if lc.reached(st, "validated"):
            lc.reset(acc, "downloaded", "filtered FASTA missing; re-validating")
        stage_validate(acc, wd)
    if not lc.reached(lc.load_state(acc), "prodigal_done"):
        stage_prodigal(acc, wd, args.repro_check)
    if not lc.reached(lc.load_state(acc), "orfs_called"):
        stage_orfs(acc, wd)
    if not args.keep_fasta and (wd / "contigs.ge120.fa").exists():
        (wd / "contigs.ge120.fa").unlink()
    print(f"{acc}: {lc.load_state(acc)['state']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
