# ROADS / Flinders figshare deposition candidates

This document is checkpoint E of agent-handoffs#24 (Gadi). It lists the data packages larger than 500 MB that should accompany the paper but cannot go in GitHub.

> **No DOI has been minted or claimed. Rob arranges ROADS deposition.** These are proposals. Sizes and checksums were measured on 2026-10-16 on NCI Gadi.

## Measurement

- **Sizes** are exact byte counts in GB (10⁹ bytes), from `stat`. The integrator's Teams inventory quotes some of the same sets in GiB, so 143 GiB there is 153.5 GB here.
- **Inventories and checksums** are at Gadi `/g/data/ob80/re3494/deposit_manifests/`, with four files per package:
  - `<package>.files` lists the files;
  - `.sizes` gives bytes per file;
  - `.total` gives the totals;
  - `.SHA256SUMS` gives a sha256 per file.

  They were produced by PBS job 181368857 (0.64 SU, exit 0). R2 was extended with the query sequences by PBS 181374800.

  Copies of the `.SHA256SUMS`, `.sizes` and `.total` files are committed in [`roads/`](roads/). Paths in them are relative to `/g/data/ob80/re3494/`.

## Summary

| ID | proposed title | size | files | owner | recommendation |
|---|---|---:|---:|---|---|
| R1 | Per-ORF sequence and structural entropy for six-frame ORFs of GTDB r232 representatives | 152.9 GB | 807 | Gadi | deposit; splitting is likely needed (see notes) |
| R2 | GTDB r232 missed-gene candidates: structural search, matched comparator (v7), Prodigal coincidence and ranking, with the query sequences | 3.25 GB | 5,695 | Gadi | deposit (priority: supports the headline estimate) |
| R3 | Length-conditioned reference distributions for plug-in entropy of ORFs (GTDB r232) | 0.59 GB | 111 | Gadi | deposit (reusable by `32_entropy_percentile.py`) |
| R4 | Sequence and structural entropy of 43.7 M GMSC benchmark smORFs | 6.08 GB | 69 | Gadi | deposit **after** confirming GMSC terms |
| R5a | LOGAN accession selection and run provenance (all 89,994 chunk manifests) | 0.54 GB | 10 | LOGAN agent | deposit; see [`roads/R5_logan.md`](roads/R5_logan.md) (#129) |
| R5b | LOGAN analysis summaries: every input to the reported LOGAN statistics | 2.41 GB | 466 | LOGAN agent | deposit; see [`roads/R5_logan.md`](roads/R5_logan.md) (#129) |
| R5c | LOGAN missed-gene evidence, including query sequences sufficient to re-run Foldseek | 1.07 GB | 321 | LOGAN agent | deposit; see [`roads/R5_logan.md`](roads/R5_logan.md) (#129) |

**Not recommended for deposit:**
- Public inputs, which should be cited instead:
  - GenBank flatfiles, 385 GB; cite the accessions;
  - GTDB r232 Prodigal gene coordinates, 5.7 GB, which are GTDB's own public gene calls (`23_gtdb_prodigal_coords.pbs`); cite GTDB r232;
  - LOGAN contigs (`s3://logan-pub`);
  - the GMSC catalogue;
  - Foldseek databases, 414 GiB on Gadi; cite the versions.
- Data too large to deposit and regenerable from code and inputs:
  - GTDB per-genome raw JSON, 1.65 TB;
  - LOGAN raw outputs, 5.95 + 3.32 TB.
- Superseded data:
  - the v6-era `missed_genes/` on Teams;
  - the old `rank_orfs/` and pre-correction mutual-information tables (see the integrator's `methods_provenance.md`).

---

## R1. GTDB r232 per-ORF entropy

- **Purpose:** the primary result table. It has one row for each six-frame ORF of at least 90 aa: 2,568,244,984 bacterial and 54,858,398 archaeal rows. Every GTDB number in the paper derives from it (§4 statistics, figure samples, the missed-gene population).
- **Location:**
  - Gadi `/g/data/ob80/re3494/gtdb_entropy/entropy_rows/{bac,arc}/*.tsv.gz` (801 chunks);
  - mirrored on Teams at `genome_entropy:General/entropy_rows/`.
- **Columns:** `domain, chunk, genome, input_id, orf_id, start, end, strand, aa_length, in_genbank, dna_entropy, protein_entropy, three_di_entropy, twelve_state_entropy, three_di_twelve_state_mutual_information, contig_length`.
- **Companion tables:** `summary_per_genome_{bac,arc}.tsv`, `genome_annotation_status_{bac,arc}.tsv` and `genome_cds_counts_{bac,arc}.tsv`.
- **Size and integrity:** 152.9 GB in 807 files; sha256 in `R1_gtdb_per_orf_entropy.SHA256SUMS`. The files are already gzip-compressed, so recompressing gains little.
- **Provenance:**
  - `genome_entropy` 0.2.0, whose exact build is not recorded (gap 2 in `methods_provenance.md`);
  - `gbouras13/modernprost-50M`, revision not recorded;
  - encoding size 10,000;
  - V100 on `gpuvolta`;
  - `get_orfs` with a 90 aa floor and translation table 11;
  - scripts `Validation/gtdb/scripts/04_run_entropy.pbs`, `04b_regenerate_entropy_rows.pbs`, `05_aggregate_results.py`.
- **Metadata to supply:** the GTDB r232 release, the accession list (the `genome` column), and the column dictionary above.
- **Licence:** derived data. The GTDB metadata is CC BY-SA 4.0 and GenBank has no restrictions; this should be confirmed by Rob or the library. There is no embargo issue.
- **Package boundaries:** one item per domain, R1a bacteria (760 chunks, 149.6 GB) and R1b archaea (41 chunks, 3.2 GB), with the companion genome tables (0.08 GB) in R1b or in a small separate item. If the ROADS per-item limit is below the bacterial size, R1a can be split into chunk ranges such as `bac_000–199` without re-processing. **Rob needs to check the item size limit with the library.**

## R2. GTDB missed-gene analysis, final v7

- **Purpose:** everything needed to recompute the mixture shares, the evidence ladder (C3–C7), the Prodigal share and the structural estimate of about 51–59k (the v7 matched clean-shadow comparator, genome_entropy #109/#111/#113/#114) without re-running Foldseek.
- **Location:** Gadi `/g/data/ob80/re3494/gtdb_entropy/missed_genes/`. This set is **not on Teams**: the Teams `missed_genes/` copy is v6-era and superseded.
- **Contents** (the full list is in `R2_gtdb_missed_genes_v7.files`):
  - **query sets:** `matched_clean/{bac,arc}/wanted_v7.tsv` with arm, length, 3Di and truncation;
  - **matching:** `comparator.tsv.gz` (candidate → comparator draws) and `balance.tsv`;
  - **best hits:** best Foldseek hit per query per database in `best_v7/`, in struct and seq modes;
  - **ORF context:** `context_v7/`;
  - **summaries:** the `summary/*_v7_matched.tsv` tables, which are the exact source of the manuscript numbers;
  - **ranking, classes and Prodigal:** `ranked6_*`, `func_*_classified.tsv.gz` and `prodigal_v7_*.coincidence.tsv.gz`;
  - **per-chunk funnel counts:** `{bac,arc}/*.stats.tsv`;
  - **examples:** `manuscript_examples.tsv`, the `dossiers/` and `candidate_burden_*.tsv`;
  - **query sequences:** the amino-acid and 3Di FASTA for every query, which is sufficient to re-run Foldseek. These are `full_bac/seqs_shard[1-4]/` and `pilot_arc/seqs/`, the v6 queries (candidates, annotated CDS, intergenic ORFs and the reused shadows), plus `matched_clean/{bac,arc}/seqs*/` and `to_search*.tsv`, the newly searched v7 comparators.
    - Every query in `wanted_v7.tsv` has a sequence in these files: 1,501,656 bacterial and 111,263 archaeal queries, 0 missing (PBS 181371728).
    - They are uncompressed FASTA (about 2.1 GB), and compression is recommended at packaging.
- **Size and integrity:** 3.25 GB in 5,695 files; sha256 in `R2_gtdb_missed_genes_v7.SHA256SUMS`.
- **Excluded:**
  - raw Foldseek outputs, `matched_clean/*/search*` (about 11 GB). They can be regenerated from the deposited query sequences plus the cited databases (Foldseek 10.941cd33, parameters below), and `best_v7/` already holds the per-query best hits;
  - the full shadow `pool/`, about 0.3 GB. Optional, since it is regenerable with `39_shadow_pool.py`.
- **Provenance:**
  - Foldseek 10.941cd33 with `-e 0.001`;
  - databases AFDB/Swiss-Prot, PDB100, CATH50 and BFVD;
  - scripts 39–42 plus `18_pilot_analysis.py`, `22_rank_candidates.py` and `24_prodigal_overlap.py`;
  - matching seed 0;
  - details in `Validation/gtdb/matched_comparator.md`.
- **Licence:** derived. Target identifiers from AFDB, PDB, CATH and BFVD are public IDs.
- **Package boundaries:** a single item, which could be split by domain if wanted.

## R3. Length-conditioned entropy references (#101)

- **Purpose:** the empirical and i.i.d. reference distributions of plug-in entropy by ORF length, for three alphabets and both domains. They let anyone score an ORF's entropy as a length-conditioned percentile or z-score with `32_entropy_percentile.py`. Figures G02 and GS02 rest on them.
- **Location:** Gadi `/g/data/ob80/re3494/gtdb_entropy/entropy_null/`. **Not on Teams.**
- **Contents:**
  - `entropy_null.npz` and `entropy_null_quantiles.tsv` (the i.i.d. reference, 400,000 draws per length);
  - `empirical/length_entropy_{bac,arc}.{tsv,npz}`;
  - `composition/`;
  - `calibration/scores_*`;
  - `bias_decomposition.tsv` and `calibration_summary.tsv`.
- **Size and integrity:** 0.59 GB in 111 files; sha256 in `R3_gtdb_length_references.SHA256SUMS`.
- **Provenance:** scripts 30–33 and 37–38 (genome_entropy #101).
- **Licence:** derived. There is no restriction.
- **Package boundaries:** a single item. The calibration scores (about 120 MB) could be left out, since they only verify calibration and are not needed to use the references.

## R4. GMSC smORF entropy (#105)

- **Purpose:** per-smORF nucleotide, amino-acid, 3Di and 12-state entropies for 33,647,794 positive and 10,079,115 negative GMSC benchmark smORFs, plus the 3Di and 12-state strings and the MMseqs2 clusters behind the homology-aware evaluation. Figures S01, SS01 and SS02 rest on them.
- **Location:** Gadi `/g/data/ob80/re3494/Projects/smORFs/results/`, mirrored on Teams at `genome_entropy:General/smORFs/results/`.
- **Contents:**
  - `smorf_entropy.tsv.gz` (anonymised `SP10.POS/NEG.*` IDs, lengths and four entropies);
  - `states/` (per-chunk 3Di and 12-state strings);
  - `clusters/` (MMseqs2 50% and 30% identity cluster TSVs and model tables);
  - `stripes/`;
  - `manuscript/` tables;
  - the density count array.
- **Size and integrity:** 6.08 GB in 69 files; sha256 in `R4_gmsc_smorf_entropy.SHA256SUMS`.
- **Provenance:** `genome_entropy` 0.3.0, ModernProst revision `2d68d506`, encoding size 50,000, V100, MMseqs2 14.7e284. See `Validation/smorfs/README.md`.
- **Licence and permission: UNCONFIRMED.** The positive and negative sets were supplied by the GMSC authors as benchmark data, and their terms of reuse are not recorded anywhere in the project. The 3Di and 12-state strings are derived per sequence from those sets. Before deposit, Rob should confirm with the GMSC authors whether derived per-sequence data can be redistributed. A fallback that needs no permission is to deposit only aggregates. Those are added to GitHub by linsalrob/genome_entropy#124, which **must be merged before this fallback is relied on**. The files are:
  - `manuscript/data/figS01_length_hist.tsv.gz`, `figS01_entropy_hist.tsv.gz`, `figS01_single_feature_auc.tsv.gz` and `figS01_models_by_split.tsv.gz`;
  - `figSS01_density_aa_3di.tsv.gz` and `figSS02_stripe_bands.tsv.gz`;
  - `tabS01_length_bins.tsv.gz` to `tabS04_dataset_summary.tsv.gz`.

  These contain only per-class counts, histograms and model metrics, with no per-sequence values.
- **Package boundaries:**
  - R4a: entropy table plus clusters, stripes and tables, 1.56 GB;
  - R4b: the 3Di and 12-state strings (`states/`), 4.52 GB, which are the most sensitive to the GMSC terms.

## R5. LOGAN (owner: LOGAN agent)

Specified in [`roads/R5_logan.md`](roads/R5_logan.md), added by linsalrob/genome_entropy#129, with per-file sha256, sizes and totals in the same format as R1–R4.

| package | size | files |
|---|---:|---:|
| R5a: accession selection and run provenance | 0.54 GB | 10 |
| R5b: analysis summaries | 2.41 GB | 466 |
| R5c: missed-gene evidence | 1.07 GB | 321 |

That file gives the reasons for **not** depositing:
- the 5.95 TB cache;
- the 1.64 TB per-ORF tables;
- the 3.32 TB of Prodigal bundles;
- the 4.3 GB pilot `qc/pilot/features.tsv.zst`, which is superseded by the analysis tables.

## Open items for Rob

1. **Item size limit.** Confirm the ROADS/figshare per-item and per-file size limits. These decide how R1 is split.
2. **GMSC terms.** Confirm the GMSC terms for R4. Without permission, deposit R4 as aggregates only.
3. **Licence.** Choose one (for example CC BY 4.0) for the derived packages R1–R3.
4. **Order of deposit.** The suggested order is R2 (headline estimate), then R3, R1 and R4.
