# Methods provenance

This file records, for each dataset in the paper, where the inputs and outputs live, which code produced them, and with what configuration. It feeds the Methods and Data-availability sections and checkpoint E (deposition). Inventory date: 2026-10-16 ([agent-handoffs#24](https://github.com/linsalrob/agent-handoffs/issues/24), checkpoint A).

**Location keys:**

| key | location | notes |
|---|---|---|
| **Teams** | `genome_entropy:General/…` | rclone remote; 13.44 TB in 308,577 files |
| **Gadi** | `/g/data/ob80/re3494/…` | |
| **Setonix** | `/scratch/pawsey1018/edwa0468/…` | purged after 21 days without access, so it is not an archive |
| **AH** | `linsalrob/agent-handoffs:projects/genome_entropy_logan/` | |
| **GE** | this repo, `main` | |

## 1. Shared definitions

- **ORFs.** `genome_entropy` six-frame ORF finding.
  - `get_orfs -l` took the minimum in amino acids until #103/#104, so the effective floor was **90 aa** for GTDB, phage and mutual information.
  - LOGAN passed `-l 29` through a wrapper, giving a true **30 aa** floor.
  - smORFs are the GMSC 10–99 aa set; no floor applies.
- **`in_genbank` label.** An ORF matches a deposited CDS when all of these hold:
  - same strand;
  - same stop codon;
  - same phase;
  - ≥90% reciprocal overlap;
  - ≥98% amino-acid identity.

  The matcher was fixed in 2cc0e1a (2026-07-31), in 0.2.0. Runs before that fix mislabel minus-strand genes.
- **Entropies.** Plug-in Shannon entropy (bits) of the amino-acid sequence, the 20-state 3Di string and the 12-state reduced alphabet, plus 3Di–12-state mutual information. DNA entropy is computed but carries no signal (GTDB §4.1). Plug-in entropy is biased with length (#101; `Validation/gtdb/scientific_report.md` §4.2).
- **3Di encoder.** `gbouras13/modernprost-50M`, a sequence-only 3Di predictor. Its revision is recorded as `2d68d506` for LOGAN and smORFs, and was not recorded for GTDB or phage.

## 2. Configuration per dataset (Table 1 source; decision D9)

| dataset | genomes / samples | ORFs | ORF floor | labels | package | encoding size | hardware |
|---|---|---|---|---|---|---|---|
| GTDB r232 | 189,715 bac encoded (of 189,801) + 10,122 arc | 2,568,244,984 + 54,858,398 | 90 aa | deposited GenBank CDS | 0.2.0 (exact build unknown), schema 2.2.0 | 10,000 | Gadi V100-SXM2-32GB (`gpuvolta`) |
| LOGAN | 50,009 metagenomes | 89,991,549,208 | 30 aa | Prodigal 2.6.3 `-p meta` | 0.3.0 library + wrapper | 10,000 (default; unstated in the LOGAN draft) | Setonix MI250X, 8 GCD × 4 workers |
| GMSC smORFs | — | 43,726,909 | n/a (10–99 aa) | GMSC curated pos/neg | 0.3.0 | **50,000** | Gadi V100 |
| Phage MI (Setonix) | 36,025 | 6,754,361 | 90 aa | v2 relabel (#108) | ≈0.2.0, schema 2.2.0 | **20,000** | Setonix ROCm fp16 |
| Phage ranking (Setonix) | 31,200 (49 of 361 parts lack JSON) | 5,887,690 | 90 aa | corrected relabel (#110) | ORFs from v0.1.15 | 20,000 | Setonix ROCm |
| Phage (Gadi) | 36,025 | 6,754,361 | 90 aa | current matcher (= v2) | 0.3.0 (v0.3.0, 45dd283) | 10,000 | Gadi V100 |

**Determinism** (Gadi PBS 181181761; `AH/inphared_gadi/README.md` §5): see results_inventory P-8. Hardware, precision and package version cannot be separated.

## 3. Per dataset: inputs, code, outputs

### GTDB r232 (owner: Gadi)

- **Input:**
  - GTDB r232 species representatives;
  - GenBank flatfiles from NCBI `datasets`, at Teams `genbank/{bac,arc}_NNN.tar.zst` (385 GB). These are public: cite the accessions rather than deposit them.
- **Code:** `GE Validation/gtdb/scripts/` (stages 00–42; see its README).
- **Outputs:**

  | data | location |
  |---|---|
  | raw per-genome JSON | Teams `entropy_3di_results/` (1.65 TB) |
  | per-ORF rows | Teams `entropy_rows/` (153 GB) and Gadi `gtdb_entropy/entropy_rows/` |
  | CDS intervals | Teams `cds_intervals/` |
  | per-genome tables | Teams `tables/`. `summary_per_genome_bac.tsv` exists on Gadi only. |
  | figure samples | Teams `figure_samples/` |

- **Missed genes:**
  - v6-era files are at Teams `missed_genes/` (61 GB) and are **superseded**. The exemplar dossiers and `manuscript_examples.tsv` there are still usable.
  - Final v7 matched-comparator tables are in GE `Validation/gtdb/matched_comparator/`; the full set is on Gadi `gtdb_entropy/missed_genes/matched_clean/`.
- **Not on Teams:**
  - #101 length-bias data (Gadi `entropy_null/`);
  - #107/#109 outputs;
  - `summary_per_genome_bac.tsv`.

### LOGAN (owner: LOGAN agent)

- **Input:**
  - LOGAN assembled contigs, public at `s3://logan-pub/c/<ACC>/`;
  - accession selection in Teams `LOGAN/manifests/` (four TSVs to deposit).
- **Code:** `GE Validation/logan/scripts/00–31` (#115).
- **Outputs:**

  | data | location |
  |---|---|
  | per-chunk entropy | Teams `LOGAN/modernprost-50M/v1/` (5.95 TB) |
  | Prodigal | Teams `LOGAN/prodigal/v1/` (3.32 TB) |
  | per-shard analysis tables | Teams `LOGAN/analysis/v1/` (1.74 TB) |
  | final QC | Teams `LOGAN/qc/{analysis_final,analysis_models_final,six_frame_final,figures_final,figures_candidates}` |
  | working copies | Setonix `Logan/` |

- **Teams upload incomplete:** the 16 October AFDB50 and low-score results were still being uploaded (Slurm 50981821, pending on 2026-10-16).
- **Numbers:** `AH/manuscript/manuscript.md` and tables S1–S35, with corrections in #24.

### GMSC smORFs (owner: Gadi)

- **Input:** GMSC v1.0 <100 aa (Teams `Validation/LuisSmallORFs/`, public: cite), and the positive/negative sets (Teams `smORFs/{positive,negative}.tar.gz`, deposit only with GMSC permission).
- **Code:** `GE Validation/smorfs/` (#105).
- **Outputs:**
  - Teams `smORFs/results/`, including `smorf_entropy.tsv.gz` (913 MB, a deposit candidate) and `manuscript/` figures and tables (fig3 missing).
  - Write-up: `AH/smorfs/manuscript.md`.

### Phage (owner: Setonix integrator for MI/ranking/OM869708; Gadi for the Gadi run)

- **Input:**
  - Inphared genomes, annotated with Pharokka + phold: Teams `Validation/Inphared/Pharokka/`, `phold_genbank.tgz`, `phold.gbk(.gz)`, `genbank_split/`.
  - phold and Pharokka versions are **not recorded**.
  - Pyrodigal-gv 0.3.2 positives.
  - "Phynteny" is named in the manuscript but appears in no record (D12).
- **Code:**
  - MI: `GE Validation/mutual_information/` (#108);
  - ranking + Foldseek: `GE Validation/Inphared/scripts/01–08`, `rank_missing_orfs.py --feature-set intrinsic` (#110);
  - Gadi run: `AH/inphared_gadi/scripts/40–50`.
- **Outputs:**
  - MI and ranking tables: `GE Validation/{mutual_information,Inphared}/tables/`;
  - full ranking work: Setonix `genome_entropy/work/phage_rank/` (28 GB, purgeable);
  - Gadi: `/g/data/ob80/re3494/inphared_entropy/`.
- **On Teams, superseded:**
  - `Validation/Inphared/rank_orfs/` (52,057 candidates);
  - `Validation/mutual_information/{phold_modernprost_50m_mutual_information.tsv,in_genbank_pair_analysis/}`.
- **OM869708:** the input is at Teams `Validation/Inphared/separate_genbank/OM869708.gbk`. Corrected-model scores are in Setonix `work/phage_rank/rank_orfs_relabelled_intrinsic/`. The BLASTP result is not on record (D7).

## 4. Gaps to close before submission

1. **Phage annotation tool versions:** Pharokka, phold and Pyrodigal-gv. Owner: Setonix.
2. **GTDB exact package build and model revision:** recorded as unknown. State this as unrecoverable, or recover it from Gadi environment logs. Owner: Gadi.
3. **LOGAN encoding size:** confirm the default (10,000) from the run configuration. Owner: LOGAN agent.
4. **OM869708 BLASTP:** re-run and commit. Owner: Setonix.
5. **Teams copies:** the LOGAN 16 October results (owner: LOGAN agent), and #101/#107/#109/#110 outputs if they are to be archived (owners: Gadi, Setonix).
