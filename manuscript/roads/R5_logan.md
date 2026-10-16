# R5. LOGAN metagenome validation: ROADS deposition candidates

**Owner:** the LOGAN agent (agent-handoffs#24, checkpoint E). This file completes the R5 entry in `manuscript/roads_deposition_candidates.md` (#126). **No DOI is claimed.** Rob arranges any deposit.

**Analysis:** genome_entropy #102, merged in #115.
- **Code (immutable):**
  - the LOGAN scripts that produced the reported results: `Validation/logan/scripts/` at genome_entropy commit [`b41d24d3517d930bd0b5986f1434e467a17ffd84`](https://github.com/linsalrob/genome_entropy/tree/b41d24d3517d930bd0b5986f1434e467a17ffd84/Validation/logan/scripts), the #115 merge on `main`;
  - the current version, with #130's robustness fixes (reported run unaffected): commit [`3b7e6b1c05061315df67127448c8382192ae92b0`](https://github.com/linsalrob/genome_entropy/tree/3b7e6b1c05061315df67127448c8382192ae92b0/Validation/logan/scripts).

  Script names below refer to that directory. On the `manuscript` branch they arrive with #127.
- **Data:** 50,009 LOGAN v1.2 metagenomes, 89.99 B ORFs ≥30 aa, 4.90 Tbp.
- **Encoding:** every one of the 89,994 cache chunks was encoded with ModernProst `gbouras13/modernprost-50M`, revision `2d68d506d4ce8d59025c985fd00243a8860363c6`, at encoding size 10,000. The encoder was `genome_entropy` 0.3.0 at commit `750e2b92283d9c008a9f4711d6c2a396112c8b50`, with a clean source tree. Each chunk manifest records these values.
- **Hardware:** Setonix MI250X GCDs.
- **Gene calling:** Prodigal 2.6.3 `-p meta` on contigs ≥120 nt; ORFs from `get_orfs` (translation table 11, ≥30 aa).
- **Setonix root:** all locations below are relative to `/scratch/pawsey1018/edwa0468/Logan`.
- **Teams copies:** under `genome_entropy:General/LOGAN/`, at the same relative paths, checked with `rclone check`:
  - `analysis/v1/`: uploaded 2026-10-14;
  - `qc/` as of 2026-10-15: uploaded;
  - `manifests/chunks/`: uploaded with the cache;
  - not yet on Teams: the 2026-10-16 `qc/` additions (`candidates_extra`, AFDB50), `roads/` and `meta/`. These are scheduled for the night of 2026-10-16 (Slurm 50981821, 50992389, 50992390). I will confirm on #24 when they are checked.
- **Checksums:** per-file sha256 come from `prep/logan_roads_manifest.sh`, rerun on 2026-10-16 after the fixes from Codex review. Every requested input must exist, and the archive is deterministic.

| ID | package | size | files | recommendation |
|---|---|---:|---:|---|
| R5a | LOGAN accession selection and run provenance | 0.54 GB | 10 | deposit |
| R5b | LOGAN analysis summaries: all inputs to the reported statistics | 2.41 GB | 466 | deposit |
| R5c | LOGAN missed-gene evidence: query sets, Foldseek best hits, contigs, taxonomy | 1.07 GB | 321 | deposit |

The file lists, sizes and sha256 are in `R5{a,b,c}_*.{files,sizes,SHA256SUMS,total}` next to this file.

## R5a. Accession selection and run provenance (0.54 GB)

**Purpose:** which metagenomes were analysed and why, and the provenance of every cache chunk.

**Contents:**
- **Candidate pool:** `meta/logan_metagenomic_candidates.parquet` (490 MB). This is the metadata-filtered candidate pool the draws came from, built with `01_select_accessions.py candidates` from the public LOGAN/SRA metadata.
- **Filter accounting:** `candidate_ledger.tsv`, `census_librarysource_assay.tsv` and `total_records.txt`. `meta/SHA256SUMS` holds the checksums of the public inputs.
- **The analysed accessions:** `manifests/{pilot,phaseb1,phaseb2}_accessions.tsv`, plus `accession_status.tsv`, which gives each accession's final state or recorded exclusion (81 missing from LOGAN, 1 failed validation).
- **Chunk manifests:** `roads/R5a_chunk_manifests.tar.zst` (26.5 MB, sha256 `b963f9ea…`). It packs all 89,994 chunk manifests into one deterministic archive: GNU tar format, sorted names, fixed mtime, owner and modes. The archive is rebuilt on every run; two consecutive runs gave identical bytes.
  - Each manifest gives the chunk's member accessions and row ranges, ORF and amino-acid counts.
  - It records the software versions and the model revision.
  - It gives the sha256 of the cache file and of each column, and the remote QuickXorHash verification.

**Not included:** the public inputs, which should be cited instead. These are the LOGAN v1.2 SRA metadata (`logan_accessions_v1.2_SRA2025.csv.zst`, 11.8 GB) and the v1.1/v1.2 contig seqstats.

## R5b. Analysis summaries (2.41 GB)

**Purpose:** everything needed to recompute every LOGAN statistic and figure in the paper, without the 7.6 TB of per-ORF data. **Full reproduction needs R5b and R5c together.** R5b covers the dataset, single-feature, model, fragment, six-frame and biome results (figL01–L05, figLS1; tabL01–L04). The missed-gene figures and tables (figL06–L10, tabL05–L09) also read the `qc/candidates*` inputs in R5c. With both packages, `manuscript/prep/logan_prepare_data.py` and the notebooks in `manuscript/notebooks/figL*.ipynb` (#117) regenerate every LOGAN number and figure.

**Contents:**

| path | count | what it is | produced by |
|---|---|---|---|
| `analysis/v1/hist/*.npz` | 90 | Exact all-ORF histograms (class × truncation × length × biome × tranche × feature); the source of every single-feature AUC/AP. | `10_feature_histograms.py` |
| `analysis/v1/sample/*.parquet` | 90 | The class-stratified model sample, with weights. | `12_sample_orfs.py` |
| `analysis/v1/sixframe/*.parquet` | 90 | The six-frame gene sample. | `14_six_frame.py` |
| `analysis/v1/candidates/*.parquet` | 90 | Every intergenic Prodigal− ORF ≥100 aa scored by the final model (49.95 M), plus sampled controls (4.58 M). | `18_score_candidates.py` |
| `analysis/v1/<shard>.json` | 90 | Shard manifests: chunk lists, row counts and table sha256. | — |
| `qc/analysis_final/` | — | Exact separation tables, including by biome and tranche. | — |
| `qc/analysis_models_final/` | — | Grouped-CV model tables, `final_model.joblib` (scikit-learn `HistGradientBoostingClassifier`) and `final_model_thresholds.json`. | — |
| `qc/six_frame_final/` | — | Inheritance, discrimination and the Δ sample. | — |

## R5c. Missed-gene evidence (1.07 GB)

**Purpose:** everything needed to recompute the missed-gene results without re-running Foldseek, plus the inputs needed to re-run it. Those results are:
- candidate and control full-length hit rates over five databases;
- the evidence classes;
- target superkingdoms;
- the contig taxonomy;
- the Prodigal skip reasons;
- the Prodigal+ / low-score class.

**Contents:**
- **Query sets:**
  - `qc/candidates/selection.parquet`: 80,000 queries. Candidates and matched controls: complete Prodigal genes, low-score intergenic ORFs, antisense shadows.
  - `qc/candidates_extra/selection.parquet`: 39,038 queries. Frameshift shadows, plus low- and high-score Prodigal genes.
- **Query sequences:** `seqs/*.tsv` in both directories, holding the amino-acid and 3Di strings of every query. Together with the cited databases, these are what a Foldseek re-run needs.
- **Foldseek reductions:** for every database and mode, the best hit per query in `search/best/*.m8` and the best full-length hit (qcov and tcov ≥0.8) in `search/fullcov/*.m8`. The evidence tables are `evidence_*.tsv` and `evidence_per_query.parquet`.
- **Contigs:**
  - `qc/candidates/contigs/all_contigs.fa`: the 79,774 query contigs, 675 Mbp, re-streamed from LOGAN;
  - `part_*.gff`: the Prodigal `-p meta` rerun, which reproduces 100% of the original calls;
  - `part_*.json`: fetch provenance, 24,824 accessions with 0 failures;
  - `kraken.out` and `kraken.report`: Kraken2 2.17.2 against PlusPFP 2023-06-05.
- **Analyses:** `taxonomy/` (the eukaryotic test), `skip/` (why Prodigal did not call each query) and `candidates_extra/lowscore/` (the Prodigal+ / low-score class).

**Excluded**, because each is regenerable from deposited inputs:
- **Raw Foldseek outputs,** `search/*.m8`, about 5.7 GB. They regenerate from `seqs/` plus the cited databases with `23a_build_queries.slurm` and `23_foldseek_search.slurm`, at the commits above.
- **Prodigal `-s` start files,** `contigs/part_*.starts`, about 2 GB. They regenerate from `all_contigs.fa` with `prodigal -q -p meta -s`.
- **The per-part FASTA,** `contigs/part_*.fa`, which duplicates `all_contigs.fa`.

**Provenance:**
- Foldseek 10.941cd33, `--alignment-type 2` for structure and `base:search` for sequence, `-e 0.001`;
- databases: PDB100, AFDB/Swiss-Prot, CATH50, BFVD, and AFDB50 v6 2025_03;
- scripts `Validation/logan/scripts/18`–`31`, at the commits above.

## Not recommended for deposit

| data | size | where | why |
|---|---:|---|---|
| ModernProst cache (AA, 3Di and 12-state strings per ORF) | 5.95 TB | Teams `LOGAN/modernprost-50M/v1/` | Far above any likely per-item limit. Regenerable from LOGAN contigs with the recorded model revision and encoding size, at about 1 M GPU SU. Cross-platform encodings agree closely but not bit-for-bit (see tabG01, #125). Keep on Teams; ask the library about institutional bulk storage if a permanent copy is wanted. |
| Per-ORF analysis tables (`analysis/v1/orfs`, `genes`) | 1.64 TB | Teams `LOGAN/analysis/v1/` | Derived from the cache by `09_build_analysis_tables.py`. R5b holds every summary the paper uses. |
| Prodigal bundles | 3.32 TB | Teams `LOGAN/prodigal/v1/` | Regenerable exactly: Prodigal output was byte-identical on repeat runs. |
| Pilot feature table `qc/pilot/features.tsv.zst` | 4.3 GB | Teams `LOGAN/qc/pilot/` | Covers 91 pilot accessions only and is superseded by the analysis tables. |
| Superseded previews (`figures_candidates_4db`, `lowscore_4db`, the 61- and 77-shard analyses) | small | Teams `LOGAN/qc/` | Superseded; do not cite. |
| LOGAN contigs, SRA metadata, Foldseek databases | — | public | Cite instead. |

## Licence and metadata

- **Licence:** derived data from public SRA/LOGAN inputs, with no embargo issue. CC BY 4.0 is suggested, as for R1–R3; Rob to confirm.
- **Metadata to supply:** the LOGAN v1.2 release, the accession list (R5a), column dictionaries (in the README of each package when built), and the software and model versions above.
- **Package boundaries:** one item each. R5a could go to GitHub instead, apart from the 490 MB candidate pool.
