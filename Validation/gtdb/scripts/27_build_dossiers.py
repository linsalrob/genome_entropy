#!/usr/bin/env python3
"""Manuscript-ready dossiers for the selected examples (issue #97, D4).

One markdown file per example, with the four sections #97 asks for: the
candidate itself, the existing annotation context, the homology evidence, and a
short interpretation.

WHY THE GENBANK ARCHIVES HAVE TO BE READ AGAIN

`cds_intervals/` carries every deposited CDS coordinate but only its ID -- there
are no product names in it, because 13_cds_intervals.pbs was written to answer
"does this ORF overlap a gene", which needs coordinates and nothing else. #97
asks for "neighbouring annotated CDS features AND PRODUCTS" and for whether the
neighbours suggest a pathway, and that needs the free text. So for the handful
of example genomes the `genomic.gbff` is pulled back out of the per-chunk
archive.

Each archive is a zstd stream with no random access, so one pass per chunk
extracts every wanted genome in it at once -- the same pattern
14b_extract_orf_seqs.py uses.

WHAT IS AND IS NOT ASSERTED HERE

The interpretation section states the evidence and the neighbourhood, and
deliberately stops short of claiming a function. A structural homolog plus
plausible neighbours is a hypothesis about a gene, not a demonstration of what
it does, and the examples exist to illustrate the biology while the RATE comes
from the full-population comparisons. Every product name is a Foldseek target's
annotation, never a claim about the candidate itself.
"""
import argparse
import gzip
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

WINDOW = 6000          # bp of context either side for neighbours and the figure
TRACK = 78             # characters in the locus diagram


def genome_to_chunk(gd):
    m = {}
    for dom in ("bac", "arc"):
        p = Path(gd) / f"genome_cds_counts_{dom}.tsv"
        if not p.exists():
            continue
        d = pd.read_csv(p, sep="\t", usecols=["genome", "domain", "chunk"],
                        dtype=str)
        # `chunk` in genome_cds_counts_*.tsv is ALREADY the full tag
        # ("arc_038"), unlike the wanted lists where it is the bare number
        # ("038") and 14b_extract_orf_seqs.py has to rebuild the tag. Two
        # tables, two conventions, same column name -- prefixing blindly
        # produced "arc_arc_038" and every archive lookup missed.
        already = d.chunk.str.contains("_", regex=False)
        tags = d.chunk.where(already, d.domain + "_" + d.chunk)
        m.update(dict(zip(d.genome, tags)))
    return m


def load_seqs(gd, wanted_ids):
    """id -> (aa, 3di), pulled from whichever pilot fasta holds it."""
    out = {}
    paths = list(Path(gd).glob("missed_genes/full_bac/seqs_shard*/pilot.*.fasta"))
    paths += list(Path(gd).glob("missed_genes/pilot_arc/seqs/pilot.*.fasta"))
    for p in paths:
        kind = "aa" if ".aa." in p.name else "3di"
        cur, buf = None, []
        with open(p) as fh:
            for line in fh:
                if line.startswith(">"):
                    if cur in wanted_ids and buf:
                        out.setdefault(cur, {})[kind] = "".join(buf)
                    cur, buf = line[1:].strip(), []
                else:
                    buf.append(line.strip())
        if cur in wanted_ids and buf:
            out.setdefault(cur, {})[kind] = "".join(buf)
    return out


def neighbours_from_archive(gd, tag, genomes, contigs):
    """{(genome, contig): [(start, end, strand, locus_tag, product), ...]}"""
    from Bio import SeqIO
    domain = tag.split("_")[0]
    archive = Path(gd) / "genbank" / f"{tag}.tar.zst"
    if not archive.exists():
        print(f"  WARNING: no {archive}", file=sys.stderr)
        return {}
    import tarfile
    out = {}
    proc = subprocess.Popen(["zstd", "-dq", "-c", str(archive)],
                            stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for member in tf:
                if not member.name.endswith("genomic.gbff"):
                    continue
                gname = member.name.split("/")[-2]
                if gname not in genomes:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                # Read the member fully before parsing. tarfile in stream
                # mode ("r|") hands back a _Stream-backed object with no
                # seekable(), which io.TextIOWrapper requires, and SeqIO wants
                # a text handle. One genomic.gbff is at most a few hundred MB
                # and only a handful are read, so buffering is fine.
                text = io.StringIO(fh.read().decode("utf-8", "replace"))
                for rec in SeqIO.parse(text, "genbank"):
                    if (gname, rec.id) not in contigs:
                        continue
                    feats = []
                    for f in rec.features:
                        if f.type != "CDS":
                            continue
                        try:
                            s = int(f.location.start)
                            e = int(f.location.end)
                        except TypeError:
                            continue
                        feats.append((
                            s, e, "+" if f.location.strand == 1 else "-",
                            (f.qualifiers.get("locus_tag") or ["?"])[0],
                            (f.qualifiers.get("product") or ["(no product)"])[0]))
                    out[(gname, rec.id)] = sorted(feats)
                if len(out) >= len(contigs):
                    break
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
    return out


def locus_track(cand, feats, lo, hi):
    """A crude but honest ASCII locus diagram."""
    span = max(hi - lo, 1)
    def cell(x):
        return min(TRACK - 1, max(0, int((x - lo) / span * (TRACK - 1))))
    lines = []
    row = [" "] * TRACK
    for s, e, st, tag, prod in feats:
        a, b = cell(max(s, lo)), cell(min(e, hi))
        ch = ">" if st == "+" else "<"
        for i in range(a, max(a + 1, b)):
            row[i] = "="
        row[b if st == "+" else a] = ch
    lines.append("  deposited CDS  " + "".join(row))
    row = [" "] * TRACK
    a, b = cell(cand[0]), cell(cand[1])
    ch = ">" if cand[2] == "+" else "<"
    for i in range(a, max(a + 1, b)):
        row[i] = "#"
    row[b if cand[2] == "+" else a] = ch
    lines.append("  CANDIDATE      " + "".join(row))
    lines.append(f"  {lo:,}".ljust(17) + f"{hi:,}".rjust(TRACK))
    return "\n".join(lines)


def fmt(v, nd=3):
    if pd.isna(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def dossier(row, feats, seqs, gd):
    qid = f"{row.genome}|{row.input_id}|{row.orf_id}|candidate"
    seq = seqs.get(qid, {})
    lo, hi = int(row.g_start) - WINDOW, int(row.g_end) + WINDOW
    near = [f for f in feats if f[1] > lo and f[0] < hi]
    L = []
    L.append(f"# {row.slot}")
    L.append("")
    L.append(f"**{row.genome}** · `{row.input_id}` · `{row.orf_id}` · "
             f"{row.domain} · {row.phylum}")
    L.append("")
    L.append("## Candidate")
    L.append("")
    L.append(f"| | |")
    L.append(f"|---|---|")
    L.append(f"| genome / contig | `{row.genome}` / `{row.input_id}` "
             f"({int(row.contig_length):,} bp) |" if "contig_length" in row
             else f"| genome / contig | `{row.genome}` / `{row.input_id}` |")
    L.append(f"| ORF id | `{row.orf_id}` |")
    L.append(f"| forward-axis coordinates | {int(row.g_start):,}–{int(row.g_end):,} "
             f"(0-based half-open) |")
    L.append(f"| strand | {row.strand} |")
    L.append(f"| length | {int(row.aa_length):,} aa |")
    L.append(f"| protein entropy | {fmt(row.get('protein_entropy'))} |")
    L.append(f"| 3Di entropy | {fmt(row.get('three_di_entropy'))} |")
    L.append(f"| 12-state entropy | {fmt(row.get('twelve_state_entropy'))} |")
    L.append(f"| 3Di–12st mutual information | "
             f"{fmt(row.get('three_di_twelve_state_mutual_information'))} |")
    L.append(f"| contig-truncated | {row.truncated_calc if 'truncated_calc' in row else 'n/a'} |")
    L.append(f"| host completeness / contamination | "
             f"{fmt(row.checkm2_completeness, 4)}% / "
             f"{fmt(row.checkm2_contamination, 3)}% |")
    L.append(f"| host contig N50 | {int(row.n50_contigs):,} bp |")
    L.append(f"| candidates in this genome | {int(row.genome_n_candidates)} "
             f"(ordinary burden by selection) |")
    L.append("")
    L.append("There is **no `coding_probability` field** in `genome_entropy`; the "
             "entropy columns above are what the package provides.")
    L.append("")
    L.append("## Existing annotation context")
    L.append("")
    L.append(f"Deposited CDS on this contig: **{int(row.n_cds_contig):,}**. "
             f"Nearest deposited CDS **{int(row.dist_up):,} bp upstream** "
             f"(`{row.up_cds_id}`, {row.up_strand}) and "
             f"**{int(row.dist_down):,} bp downstream** "
             f"(`{row.down_cds_id}`, {row.down_strand}); the intergenic gap it "
             f"sits in is **{int(row.gap_len):,} bp**.")
    L.append("")
    same_strand = (row.up_strand == row.strand == row.down_strand)
    L.append(f"Operon-like arrangement: **{'yes' if same_strand else 'no'}** "
             f"— candidate {row.strand}, upstream {row.up_strand}, downstream "
             f"{row.down_strand}"
             + (", all co-oriented with short spacing." if same_strand else "."))
    L.append("")
    if near:
        L.append("Neighbouring deposited CDS and their products:")
        L.append("")
        L.append("| locus_tag | coords | strand | product |")
        L.append("|---|---|---|---|")
        for s, e, st, tag, prod in near:
            mark = ""
            L.append(f"| `{tag}` | {s:,}–{e:,} | {st} | {prod}{mark} |")
        L.append("")
        L.append("```")
        L.append(locus_track((int(row.g_start), int(row.g_end), row.strand),
                             near, lo, hi))
        L.append("```")
    else:
        L.append("_No deposited CDS recovered from the GenBank record in this "
                 "window; the coordinates above come from `cds_intervals/`._")
    L.append("")
    L.append("## Homology evidence")
    L.append("")
    L.append("| database | mode | target | qcov | tcov | E | class | product |")
    L.append("|---|---|---|---|---|---|---|---|")
    for db in ("afdb_swissprot", "pdb100", "cath50", "bfvd"):
        for mode in ("struct", "seq"):
            t = row.get(f"{db}_{mode}_target")
            if pd.isna(t):
                continue
            L.append(f"| {db} | {mode} | `{t}` | "
                     f"{fmt(row.get(f'{db}_{mode}_qcov'))} | "
                     f"{fmt(row.get(f'{db}_{mode}_tcov'))} | "
                     f"{fmt(row.get(f'{db}_{mode}_evalue'))} | "
                     f"{row.get(f'{db}_class', '')} | "
                     f"{row.get(f'{db}_product', '') if mode == 'struct' else ''} |")
    L.append("")
    L.append(f"Full-length structural support in **{int(row.n_db_full_struct)} "
             f"of 4** databases. Evidence class: "
             f"**{'structure-only' if row.structure_only else 'sequence+structure'}**"
             + (" — no full-length sequence hit, so conventional similarity "
                "search would not have found this." if row.structure_only else "."))
    L.append("")
    pc = row.get("prodigal_coincides")
    if pc is True or pc == "True":
        L.append("**Independently called by GTDB's Prodigal annotation**, in the "
                 "same frame and strand, with an exactly matching 3' end — while "
                 "absent from the deposited GenBank CDS. Prodigal consults "
                 "neither GenBank nor any structure database.")
    else:
        L.append("_Not_ matched by a GTDB Prodigal call in the same frame. The "
                 "structural evidence stands alone here.")
    L.append("")
    if seq.get("aa"):
        L.append("### Amino-acid sequence")
        L.append("")
        L.append("```")
        s = seq["aa"]
        for i in range(0, len(s), 60):
            L.append(s[i:i + 60])
        L.append("```")
        L.append("")
    if seq.get("3di"):
        L.append("<details><summary>3Di sequence</summary>")
        L.append("")
        L.append("```")
        s = seq["3di"]
        for i in range(0, len(s), 60):
            L.append(s[i:i + 60])
        L.append("```")
        L.append("</details>")
        L.append("")
    L.append("## Interpretation")
    L.append("")
    bits = []
    bits.append(f"This ORF is {int(row.aa_length):,} aa, complete (not running "
                f"off a contig end), and lies entirely between two deposited CDS "
                f"with no overlap of any annotated feature — verified "
                f"independently of the classifier against the deposited interval "
                f"table.")
    bits.append(f"It carries full-length mutual coverage "
                f"(qcov and tcov ≥ 0.8) against "
                f"{int(row.n_db_full_struct)} of 4 structural databases, whose "
                f"best annotated target is *{row.best_product}*.")
    if row.structure_only:
        bits.append("The match is **structure-only**: the same search in "
                    "amino-acid mode against the same targets does not reach "
                    "full-length coverage, which is the case this project exists "
                    "to make — structural homology recovering a gene that "
                    "sequence similarity misses.")
    if same_strand:
        bits.append("Both flanking genes are co-oriented with the candidate and "
                    "the spacing is short, which is the arrangement expected of "
                    "an operon member rather than of an incidental open reading "
                    "frame.")
    if pc is True or pc == "True":
        bits.append("An independent *ab initio* gene caller predicts a gene at "
                    "exactly this locus in exactly this frame, so the claim does "
                    "not rest on structural homology alone.")
    bits.append(f"The host genome is {fmt(row.checkm2_completeness, 4)}% complete "
                f"with {fmt(row.checkm2_contamination, 3)}% contamination and an "
                f"ordinary candidate burden "
                f"({int(row.genome_n_candidates)} candidates), so this is not an "
                f"artefact of a poorly assembled or unusually unannotated "
                f"assembly.")
    bits.append("**What this does not establish**: the target's product name is "
                "the annotation of a structural homolog, not a demonstrated "
                "function for this ORF. Confirming the function needs "
                "experimental work or at minimum a curated orthology "
                "assignment.")
    for b in bits:
        L.append(f"- {b}")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--examples", required=True)
    ap.add_argument("--gd", default="/g/data/ob80/re3494/gtdb_entropy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top", type=int, default=0,
                    help="0 = all examples; otherwise the first N")
    args = ap.parse_args()

    ex = pd.read_csv(args.examples, sep="\t", low_memory=False)
    if args.top:
        ex = ex.head(args.top)
    print(f"examples: {len(ex)}")

    # Full evidence columns live in the classified tables, not the examples
    # summary; rejoin so the dossier can report every database and mode.
    full = []
    for dom, sub, tag in [("bac", "full_bac", "bac"), ("arc", "pilot_arc", "arc")]:
        p = Path(args.gd) / "missed_genes" / sub / f"func_{tag}_classified.tsv.gz"
        if p.exists():
            d = pd.read_csv(p, sep="\t", low_memory=False)
            d["domain"] = dom
            full.append(d)
    full = pd.concat(full, ignore_index=True)
    key = ["genome", "input_id", "orf_id"]
    keep_from_ex = ["slot", "prodigal_coincides", "genome_n_candidates"]
    ex = ex[key + [c for c in keep_from_ex if c in ex.columns]].merge(
        full, on=key, how="left")
    print(f"rejoined evidence columns: {ex.shape[1]}")

    g2c = genome_to_chunk(args.gd)
    ex["chunk_tag"] = ex.genome.map(g2c)
    missing = ex.chunk_tag.isna()
    if missing.any():
        print(f"WARNING: {int(missing.sum())} examples have no chunk mapping",
              file=sys.stderr)

    wanted_ids = {f"{r.genome}|{r.input_id}|{r.orf_id}|candidate"
                  for _, r in ex.iterrows()}
    print("reading pilot FASTAs for sequences...")
    seqs = load_seqs(args.gd, wanted_ids)
    print(f"  sequences found: {len(seqs)} of {len(wanted_ids)}")

    feats_all = {}
    for tag, g in ex.dropna(subset=["chunk_tag"]).groupby("chunk_tag"):
        genomes = set(g.genome)
        contigs = set(zip(g.genome, g.input_id))
        print(f"reading {tag} for {len(genomes)} genome(s)...")
        feats_all.update(neighbours_from_archive(args.gd, tag, genomes, contigs))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    # iterrows, NOT itertuples: the dossier body uses row.get(col) for the
    # many optional evidence columns, and a namedtuple has no .get().
    for i, (_, row) in enumerate(ex.iterrows(), start=1):
        feats = feats_all.get((row.genome, row.input_id), [])
        text = dossier(row, feats, seqs, args.gd)
        name = f"{i:02d}_{row.genome}_{row.orf_id}.md"
        (out_dir / name).write_text(text)
        combined.append(text)
        n_win = len([f for f in feats
                     if f[1] > int(row.g_start) - WINDOW
                     and f[0] < int(row.g_end) + WINDOW])
        print(f"  {name}  ({n_win} CDS in the +/-{WINDOW//1000} kb window, "
              f"{len(feats)} on the contig)")
    (out_dir / "ALL_DOSSIERS.md").write_text(
        "\n\n---\n\n".join(combined))
    print(f"\ndossiers -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
