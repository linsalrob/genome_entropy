#!/usr/bin/env python3
"""Select metagenomic LOGAN accessions from metadata, before downloading anything.

Two subcommands:

  candidates   stream the 11.8 GB SRA metadata CSV once with DuckDB, keep
               librarysource = METAGENOMIC, join LOGAN v1.2 contig statistics,
               derive a broad biome, and write
               meta/logan_metagenomic_candidates.parquet plus a count table of
               every exclusion so the filter is auditable.

  pilot        draw the Phase A pilot from the candidates: random-shotgun WGS
               on short-read platforms, stratified by biome and by assembly
               size, at most one run per BioProject, fixed seed. Writes
               manifests/pilot_accessions.tsv and seeds each accession's state
               file at ``selected``.

WHY librarysource IS THE PRIMARY FILTER, AND WHY HOST IS NOT A FILTER.
Issue #102 asks to remove human and isolate WGS, not human-associated
microbiomes. librarysource = METAGENOMIC already excludes GENOMIC (isolates,
human WGS) and TRANSCRIPTOMIC runs. A human gut sample is METAGENOMIC with
organism "human gut metagenome" and is kept. What METAGENOMIC does not exclude
is amplicon sequencing (16S/ITS), which is METAGENOMIC with assay_type AMPLICON
or libraryselection PCR; those are excluded explicitly. For the pilot we also
require an NCBI "... metagenome" organism name, because it is both a strong
signal that the run is a community sample and the most reliable biome label.

  01_select_accessions.py candidates
  01_select_accessions.py pilot --n 100 --seed 102
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

META = lc.root() / "meta"
CSV = Path(os.environ.get("LOGAN_SRA_CSV", META / "logan_accessions_v1.2_SRA2025.csv.zst"))
SEQSTATS = META / "logan-seqstats-contigs-v1.2.parquet"
# WHY TWO SEQSTATS FILES. v1.1 (Jan 2025) recomputed every pre-2024 contig
# set to fix two Minia bugs, but the v1.2 parquet still carries the v1.0
# numbers for those accessions (processing_date_epoch in March 2024). The
# pilot found 41/91 objects disagreeing with v1.2 by 0.07-30% and matching
# v1.1 exactly. So v1.1 statistics win wherever they exist.
SEQSTATS_V11 = META / "logan-seqstats-contigs-v1.1.parquet"
CANDIDATES = META / "logan_metagenomic_candidates.parquet"

# jattr keys worth carrying. *_sam values are JSON lists; take the first.
JATTR_KEYS = (
    "host_sam", "isolation_source_sam", "env_broad_scale_sam",
    "env_local_scale_sam", "env_medium_sam", "lat_lon_sam_s_dpl34",
    "geo_loc_name_sam",
)

# Broad biome from the NCBI metagenome organism name first, then from host /
# isolation source. Order matters: the first matching rule wins, so specific
# rules (human gut) precede general ones (gut).
BIOME_RULES = (
    ("human_gut", r"\bhuman (gut|feces|faeces|fecal|intestinal|stool)"),
    ("human_other", r"\bhuman (oral|skin|lung|vaginal|nasopharyngeal|"
                    r"respiratory|saliva|milk|reproductive|urinary|blood|"
                    r"eye|nasal|sputum|tracheal|bile|semen)|^human metagenome"),
    ("animal_host", r"\b(gut|feces|faecal|fecal|rumen|intestin|stool|caecum|"
                    r"cecum|mouse|rat|pig|bovine|chicken|insect|fish|sponge|"
                    r"coral|invertebrate|oral|skin|lung|vaginal|"
                    r"respiratory|milk|blood|honey bee|termite|mosquito)"),
    ("wastewater", r"wastewater|activated sludge|sewage|sludge|bioreactor|"
                   r"anaerobic digester|biogas|landfill"),
    ("sediment", r"sediment|mud\b|silt"),
    ("plant", r"rhizosphere|phyllosphere|plant|root|leaf|endophyte|seed|"
              r"epiphyte|rhizoplane"),
    ("extreme", r"hot spring|hydrothermal|acid mine|hypersaline|saltern|"
                r"permafrost|volcan|thermal|alkali|halophil|glacier|"
                r"subsurface|deep sea|cryoconite|ice"),
    ("marine", r"marine|seawater|sea water|ocean|estuar|coastal|salt marsh|"
               r"mangrove|seagrass"),
    ("freshwater", r"freshwater|lake|river|pond|stream|groundwater|"
                   r"aquifer|reservoir|drinking water|wetland|bog"),
    ("soil", r"soil|compost|peat|agricultur|terrestrial|desert|grassland|"
             r"forest"),
    ("built_environment", r"built environment|indoor|air\b|dust|hospital|"
                          r"surface|subway|building|wastewater pipe|"
                          r"cleanroom|biofilm"),
    ("food", r"food|ferment|cheese|kefir|kombucha|wine|beer|yogurt|"
             r"sourdough|dairy"),
)
_BIOME_RX = [(b, re.compile(rx, re.I)) for b, rx in BIOME_RULES]


_HUMAN_HOST = re.compile(r"homo sapiens|\bhuman\b", re.I)
_GUT = re.compile(r"\b(gut|feces|faeces|fecal|faecal|stool|intestin)", re.I)
_BODY = re.compile(r"\b(oral|skin|lung|vagina|nasopharyn|respiratory|saliva|"
                   r"milk|urin|blood|sputum|airway|nasal|mouth|tongue|dental|plaque)", re.I)
# Not natural communities: excluded from pilot draws.
NON_NATURAL = r"synthetic|mock|mixed culture|bioreactor metagenome$|laboratory"


def biome_of(organism: str, host: str, source: str) -> str:
    # Generic NCBI names (gut / feces / skin metagenome) are human when the
    # BioSample host says so; without this they would all fall to animal_host.
    if _HUMAN_HOST.search(host or ""):
        if _GUT.search(organism or "") or _GUT.search(source or ""):
            return "human_gut"
        if _BODY.search(organism or "") or _BODY.search(source or ""):
            return "human_other"
    for text in (organism, f"{host} {source}"):
        text = text or ""
        for biome, rx in _BIOME_RX:
            if rx.search(text):
                return biome
    return "other"


def cmd_candidates(args: argparse.Namespace) -> int:
    import duckdb

    for p in (CSV, SEQSTATS, SEQSTATS_V11):
        if not p.exists():
            sys.exit(f"ERROR: {p} missing; run 01a_fetch_metadata.slurm first")

    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute(f"SET temp_directory = '{META / 'duckdb_tmp'}'")

    # (expression, name) for every column carried for METAGENOMIC runs.
    cols = [
        ("m.acc", "accession"), ("m.libraryselection", "libraryselection"),
        ("m.librarylayout", "librarylayout"), ("m.platform", "platform"),
        ("m.instrument", "instrument"), ("m.organism", "organism"),
        ("m.bioproject", "bioproject"), ("m.biosample", "biosample"),
        ("m.sra_study", "sra_study"), ("m.center_name", "center_name"),
        ("m.releasedate", "releasedate"),
        ("TRY_CAST(m.mbases AS BIGINT)", "mbases"),
        ("TRY_CAST(m.avgspotlen AS INTEGER)", "avgspotlen"),
        ("m.geo_loc_name_country_calc", "country"),
        ("m.biosamplemodel_sam", "biosamplemodel_sam"),
        ("m.collection_date_sam", "collection_date"),
    ] + [(f"CASE WHEN json_valid(m.jattr) THEN coalesce("
          f"json_extract_string(m.jattr, '$.{k}[0]'), "
          f"json_extract_string(m.jattr, '$.{k}')) END", k.replace("_s_dpl34", ""))
         for k in JATTR_KEYS]     # *_sam values are usually lists, sometimes strings

    # DuckDB's parallel CSV reader refuses this file (quoted JSON in
    # `jattr`), and a zstd stream is sequential anyway: one pass.
    print("single pass over the SRA CSV", flush=True)
    sel = ",\n               ".join(f"{e} AS {n}" for e, n in cols)
    # PREFILTER OUTSIDE DUCKDB. DuckDB ran out of memory on this file twice
    # (90 GB; CREATE TABLE and then a streaming COPY), apparently in the
    # zstd CSV reader itself. So the stream is decompressed once by zstd, the
    # census of librarysource/assay_type is NOT taken over every row, and
    # an awk index(",METAGENOMIC,") test keeps a superset of the METAGENOMIC rows (the
    # string can also occur inside a quoted field). DuckDB then parses only
    # that subset and the librarysource test below makes it exact. Total
    # records are counted in the same pass.
    sub = META / "metagenomic_rows.csv.zst"
    if not sub.exists():
        tmp = sub.with_name(sub.name + ".partial")
        # Records end in CRLF; quoted fields can hold bare LFs, so split on
        # CRLF (gawk multi-character RS) to keep records whole.
        cmd = (f"set -o pipefail; zstd -dcq '{CSV}' "
               f"| awk 'BEGIN {{ RS = \"\\r\\n\"; ORS = \"\\r\\n\" }} "
               f"NR == 1 || index($0, \",METAGENOMIC,\") {{ print }} "
               f"END {{ print NR - 1 > \"{META}/total_records.txt\" }}' "
               f"| zstd -q -T8 -3 -o '{tmp}' -f")
        subprocess.run(["bash", "-c", cmd], check=True)
        os.replace(tmp, sub)
    lines = int((META / "total_records.txt").read_text().split()[0])
    print(f"  CSV records: {lines:,}", flush=True)
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"""
        CREATE TABLE scan AS
        SELECT m.librarysource, m.assay_type,
               {sel}
        FROM read_csv('{sub}', compression='zstd', header=true,
                      all_varchar=true, parallel=false, strict_mode=false) m
        WHERE m.librarysource = 'METAGENOMIC'
    """)
    census = con.execute("""
        SELECT librarysource, assay_type, count(*) AS runs
        FROM scan GROUP BY ALL ORDER BY runs DESC
    """).df()
    census.to_csv(META / "census_librarysource_assay.tsv", sep="\t", index=False)
    meta_n = int(census.runs.sum())
    print(f"  METAGENOMIC runs: {meta_n:,} of {lines:,} runs", flush=True)

    con.execute(f"""
        CREATE TABLE cand AS
        SELECT c.*,
               coalesce(TRY_CAST(o.seqstats_contigs_nbseq AS BIGINT), s.seqstats_contigs_nbseq) AS contig_count,
               coalesce(TRY_CAST(o.seqstats_contigs_sumlen AS BIGINT), s.seqstats_contigs_sumlen) AS contig_bp,
               coalesce(TRY_CAST(o.seqstats_contigs_n50 AS BIGINT), s.seqstats_contigs_n50) AS n50,
               coalesce(TRY_CAST(o.seqstats_contigs_maxlen AS BIGINT), s.seqstats_contigs_maxlen) AS max_contig,
               CASE WHEN o.accession IS NOT NULL THEN 'v1.1'
                    WHEN s.accession IS NOT NULL THEN 'v1.2' END AS seqstats_source
        FROM scan c
        LEFT JOIN read_parquet('{SEQSTATS}') s ON s.accession = c.accession
        LEFT JOIN read_parquet('{SEQSTATS_V11}') o ON o.accession = c.accession
        WHERE c.librarysource = 'METAGENOMIC'
    """)
    df = con.execute("SELECT * FROM cand").df()
    df["biome"] = [biome_of(o, h, s) for o, h, s in
                   zip(df.organism.fillna(""), df.host_sam.fillna(""),
                       df.isolation_source_sam.fillna(""))]
    df.to_parquet(CANDIDATES, index=False)

    # The exclusion ledger the pilot applies, counted over all candidates.
    led = []
    led.append(("METAGENOMIC runs", len(df)))
    has = df.contig_bp.notna() & (df.contig_bp > 0)
    led.append(("  with LOGAN v1.2 contigs", int(has.sum())))
    d = df[has]
    wgs = d.assay_type == "WGS"
    led.append(("  assay_type = WGS", int(wgs.sum())))
    led.append(("  assay_type = AMPLICON", int((d.assay_type == "AMPLICON").sum())))
    rnd = d.libraryselection.isin(RANDOM_SELECTIONS)
    led.append(("  WGS and random-shotgun selection", int((wgs & rnd).sum())))
    mg = d.organism.fillna("").str.contains(r"metagenome$", case=False)
    led.append(("  ... and organism '* metagenome'", int((wgs & rnd & mg).sum())))
    pd.DataFrame(led, columns=["filter", "runs"]).to_csv(
        META / "candidate_ledger.tsv", sep="\t", index=False)
    for f, n in led:
        print(f"  {f:<45}{n:>14,}")
    print(df.biome.value_counts().to_string())
    print(f"wrote {CANDIDATES}")
    return 0


RANDOM_SELECTIONS = ("RANDOM", "RANDOM PCR", "unspecified")
SHORT_READ_PLATFORMS = ("ILLUMINA", "BGISEQ", "DNBSEQ", "ION_TORRENT", "ELEMENT")


def cmd_pilot(args: argparse.Namespace) -> int:
    df = pd.read_parquet(CANDIDATES)
    keep = (
        (df.assay_type == "WGS")
        & df.libraryselection.isin(RANDOM_SELECTIONS)
        & df.platform.isin(SHORT_READ_PLATFORMS)
        & df.organism.fillna("").str.contains(r"metagenome$", case=False)
        & ~df.organism.fillna("").str.contains(NON_NATURAL, case=False)
        & df.contig_bp.between(args.min_bp, args.max_bp)
        & (df.n50 >= args.min_n50)
    )
    pool = df[keep].copy()
    print(f"pilot pool: {len(pool):,} runs after filters")
    rng = np.random.default_rng(args.seed)

    # Size strata: log-spaced contig-bp bins so the pilot spans small to
    # large assemblies rather than clustering at the median.
    edges = np.logspace(np.log10(args.min_bp), np.log10(args.max_bp), args.size_bins + 1)
    pool["size_bin"] = np.clip(np.digitize(pool.contig_bp, edges) - 1, 0, args.size_bins - 1)
    biomes = [b for b, _ in BIOME_RULES] + ["other"]
    per_biome = max(1, args.n // len([b for b in biomes if (pool.biome == b).any()]))

    picks, used_projects = [], set()
    for biome in biomes:
        sub = pool[pool.biome == biome]
        if sub.empty:
            continue
        got = 0
        # Round-robin over size bins, one run per BioProject across the pilot.
        order = {b: sub[sub.size_bin == b].sample(frac=1, random_state=rng.integers(1 << 31))
                 for b in range(args.size_bins)}
        cursors = {b: 0 for b in order}
        while got < per_biome and any(cursors[b] < len(order[b]) for b in order):
            for b in range(args.size_bins):
                if got >= per_biome:
                    break
                o = order[b]
                while cursors[b] < len(o):
                    row = o.iloc[cursors[b]]
                    cursors[b] += 1
                    if row.bioproject in used_projects:
                        continue
                    used_projects.add(row.bioproject)
                    picks.append(row)
                    got += 1
                    break
    pilot = pd.DataFrame(picks)
    if len(pilot) > args.n:
        pilot = pilot.sample(n=args.n, random_state=args.seed)
    pilot = pilot.sort_values(["biome", "contig_bp"]).reset_index(drop=True)
    cols = ["accession", "biome", "organism", "bioproject", "biosample",
            "sra_study", "assay_type", "librarysource", "libraryselection",
            "librarylayout", "platform", "instrument", "mbases",
            "contig_count", "contig_bp", "n50", "max_contig", "seqstats_source", "size_bin",
            "host_sam", "isolation_source_sam", "env_broad_scale_sam",
            "env_local_scale_sam", "env_medium_sam", "country",
            "collection_date"]
    out = lc.root() / "manifests" / f"{args.name}_accessions.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pilot[cols].to_csv(out, sep="\t", index=False)
    print(pilot.groupby("biome").agg(runs=("accession", "size"),
                                     median_mbp=("contig_bp", lambda x: x.median() / 1e6),
                                     total_mbp=("contig_bp", lambda x: x.sum() / 1e6)).to_string())
    print(f"total contig bp: {pilot.contig_bp.sum() / 1e9:.2f} Gbp in {len(pilot)} runs")
    print(f"wrote {out}")
    if not args.no_state:
        for _, r in pilot.iterrows():
            st = lc.load_state(r.accession)
            if st.get("state") is None:
                lc.advance(r.accession, "selected", batch=args.name,
                           biome=r.biome, expected_contig_count=int(r.contig_count),
                           expected_contig_bp=int(r.contig_bp), expected_n50=int(r.n50))
    return 0


MANIFEST_COLS = ["accession", "biome", "organism", "bioproject", "biosample",
                 "sra_study", "assay_type", "librarysource", "libraryselection",
                 "librarylayout", "platform", "instrument", "mbases",
                 "contig_count", "contig_bp", "n50", "max_contig", "seqstats_source",
                 "size_bin", "host_sam", "isolation_source_sam", "env_broad_scale_sam",
                 "env_local_scale_sam", "env_medium_sam", "country", "collection_date"]


def cmd_draw(args: argparse.Namespace) -> int:
    """Phase B draw: the pilot's filters and strata at scale.

    Differences from `pilot`, which stays as it was so the pilot can be
    reproduced from the code that drew it:

      * accessions already selected by an earlier batch (any state file)
        are excluded, so tranches never overlap;
      * at most --max-per-project runs per BioProject, counting runs it
        already has in earlier batches (one per project is impossible at
        25k: the whole pool spans ~9,000 projects);
      * quotas are water-filled over biome x log-size strata, so a stratum
        too small for its equal share gives its remainder to the others
        instead of leaving the draw short.
    """
    df = pd.read_parquet(CANDIDATES)
    keep = (
        (df.assay_type == "WGS")
        & df.libraryselection.isin(RANDOM_SELECTIONS)
        & df.platform.isin(SHORT_READ_PLATFORMS)
        & df.organism.fillna("").str.contains(r"metagenome$", case=False)
        & ~df.organism.fillna("").str.contains(NON_NATURAL, case=False)
        & df.contig_bp.between(args.min_bp, args.max_bp)
        & (df.n50 >= args.min_n50)
    )
    taken = {p.stem for p in (lc.root() / "state").glob("*.json")}
    pool = df[keep & ~df.accession.isin(taken)].copy()
    print(f"pool: {len(pool):,} runs after filters, {len(taken):,} already selected excluded")

    pool = pool.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    # The cap is cumulative across batches: runs a BioProject already has in
    # earlier batches count against it.
    used = df[df.accession.isin(taken)].groupby("bioproject").size()
    pool["project_rank"] = (pool.groupby(pool.bioproject.fillna(pool.accession)).cumcount()
                            + pool.bioproject.map(used).fillna(0).astype(int))
    pool = pool[pool.project_rank < args.max_per_project]
    edges = np.logspace(np.log10(args.min_bp), np.log10(args.max_bp), args.size_bins + 1)
    pool["size_bin"] = np.clip(np.digitize(pool.contig_bp, edges) - 1, 0, args.size_bins - 1)
    cap = pool.groupby(["biome", "size_bin"]).size()
    print(f"after <= {args.max_per_project} per BioProject: {len(pool):,} runs in {len(cap)} strata")
    if cap.sum() < args.n:
        sys.exit(f"ERROR: only {cap.sum():,} runs available for n={args.n:,}")

    # Water-fill: raise a common per-stratum quota until the total reaches n.
    lo, hi = 0, int(cap.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cap.clip(upper=mid).sum() <= args.n:
            lo = mid
        else:
            hi = mid - 1
    quota = cap.clip(upper=lo)
    short = args.n - int(quota.sum())
    for k in cap[cap > lo].index[:short]:   # the remainder, one each
        quota[k] += 1
    pool["stratum_rank"] = pool.groupby(["biome", "size_bin"]).cumcount()
    q = pool.set_index(["biome", "size_bin"]).index.map(quota)
    draw = pool[pool.stratum_rank.to_numpy() < np.asarray(q)]
    assert len(draw) == args.n, len(draw)
    draw = draw.sort_values(["biome", "contig_bp"]).reset_index(drop=True)

    out = lc.root() / "manifests" / f"{args.name}_accessions.tsv"
    if out.exists():
        sys.exit(f"ERROR: {out} exists; a batch name is never reused")
    draw[MANIFEST_COLS].to_csv(out, sep="\t", index=False)
    summ = draw.groupby("biome").agg(runs=("accession", "size"),
                                     median_mbp=("contig_bp", lambda x: x.median() / 1e6),
                                     total_gbp=("contig_bp", lambda x: x.sum() / 1e9))
    print(summ.to_string())
    print(f"total contig bp: {draw.contig_bp.sum() / 1e12:.3f} Tbp in {len(draw):,} runs; "
          f"{draw.bioproject.nunique():,} BioProjects; per-stratum quota {lo}")
    print(f"wrote {out}")
    if not args.no_state:
        for r in draw.itertuples():
            lc.advance(r.accession, "selected", batch=args.name, biome=r.biome,
                       expected_contig_count=int(r.contig_count),
                       expected_contig_bp=int(r.contig_bp), expected_n50=int(r.n50),
                       seqstats_source=r.seqstats_source)
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    """Re-read a selection manifest's assembly statistics from the candidates
    table and update each accession's expected values in its state file.
    Used once, after the seqstats-version fix above, so the pilot validates
    against the right numbers without being redrawn."""
    path = lc.root() / "manifests" / f"{args.name}_accessions.tsv"
    sel = pd.read_csv(path, sep="\t")
    cand = pd.read_parquet(CANDIDATES, columns=["accession", "contig_count", "contig_bp",
                                                "n50", "max_contig", "seqstats_source"])
    cand = cand.set_index("accession").loc[sel.accession]
    changed = int((sel.set_index("accession").contig_bp != cand.contig_bp).sum())
    for c in ("contig_count", "contig_bp", "n50", "max_contig"):
        sel[c] = cand[c].to_numpy()
    sel["seqstats_source"] = cand.seqstats_source.to_numpy()
    sel.to_csv(path, sep="\t", index=False)
    for _, r in sel.iterrows():
        st = lc.load_state(r.accession)
        st.setdefault("info", {}).update(expected_contig_count=int(r.contig_count),
                                         expected_contig_bp=int(r.contig_bp),
                                         expected_n50=int(r.n50),
                                         seqstats_source=r.seqstats_source)
        lc.atomic_write_json(lc.state_path(r.accession), st)
    print(f"{changed} of {len(sel)} accessions had different contig_bp; "
          f"sources {sel.seqstats_source.value_counts().to_dict()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--threads", type=int, default=16)
    c.add_argument("--memory", default="48GB")
    p = sub.add_parser("pilot")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=102)
    p.add_argument("--name", default="pilot")
    p.add_argument("--min-bp", type=float, default=5e6)
    p.add_argument("--max-bp", type=float, default=5e8)
    p.add_argument("--min-n50", type=int, default=300)
    p.add_argument("--size-bins", type=int, default=4)
    p.add_argument("--no-state", action="store_true")
    d = sub.add_parser("draw")
    d.add_argument("--n", type=int, required=True)
    d.add_argument("--name", required=True)
    d.add_argument("--seed", type=int, default=102)
    d.add_argument("--min-bp", type=float, default=5e6)
    d.add_argument("--max-bp", type=float, default=3e8)
    d.add_argument("--min-n50", type=int, default=150)
    d.add_argument("--size-bins", type=int, default=4)
    d.add_argument("--max-per-project", type=int, default=25)
    d.add_argument("--no-state", action="store_true")
    r = sub.add_parser("refresh")
    r.add_argument("--name", default="pilot")
    args = ap.parse_args()
    return {"candidates": cmd_candidates, "pilot": cmd_pilot, "draw": cmd_draw,
            "refresh": cmd_refresh}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
