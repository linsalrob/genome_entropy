# Inphared ORF ranking on all 36,025 genomes

The ranking in `../README.md` used an earlier encoding of 31,200 genomes (49 of 361 annotation parts had no encoder output).
This rerun uses the mutual-information set instead (`Validation/mutual_information`): all 36,025 genomes and 6,754,361 ORFs ≥ 90 aa,
genome_entropy ≈ 0.2.0, ModernProst-50M, encoding size 20,000. Everything else follows `../README.md`, with three additions:

- **phold DB v1.0.0** (1,363,704 predicted phage protein structures) is a fifth Foldseek target. With BFVD it forms the
  *viral* databases; `non_viral` rates exclude both.
- **Homology-aware cross-validation.** Besides holding out one genome at a time, the models were refitted holding out whole
  skani 95 % ANI clusters and whole NCBI genera.
- **A case-study genome**, the one with the most candidates (`11_case_study.*`).

Setonix work directory: `/scratch/pawsey1018/edwa0468/genome_entropy/work/phage_rank_all` (job IDs in `logs/jobs.txt`).
Run on 2026-10-28.

## Data and backups

| data | Setonix (`work/`) | rclone remote |
|---|---|---|
| encodings (genome_entropy JSON, 145 chunks) | `phage_mi/results/phold_*.json.gz` | `genome_entropy:General/Validation/mutual_information/results/` |
| Pharokka + phold GenBank inputs | `phage_mi/chunks/phold_*.gbk.gz` | `genome_entropy:General/Validation/mutual_information/chunks/` |
| corrected labels, ORF / CDS / sequence tables | `phage_rank_all/tables/{labels,orfs,cds,seqs}.parquet` | `genome_entropy:General/Validation/Inphared/all_genomes/tables/` |
| genome groups (ANI95, genus), skani triangle, genome FASTA, non-ACGT counts | `phage_rank_all/groups/` | `genome_entropy:General/Validation/Inphared/all_genomes/groups/` |

The `in_genbank` field inside the encoding JSONs is from the pre-0.2.0 matcher and is all False; use `labels.parquet`
(`in_genbank`; `in_genbank_orig` is the stale value) or `orfs.parquet`, which carries the corrected label alongside the
entropies. The remote copies of `tables/` and `groups/` were uploaded on 2026-10-30 and checked with `rclone check`.

The labels are the Pharokka v1.9.1 + phold 1.2.5 re-annotation of Inphared `7Apr2026_genomes` (361 parts, run 26–27 May
2026; Pyrodigal-gv 0.3.2 gene calls, `--meta`). The versions are from the run logs inside the part tarballs on the remote
(`Validation/Inphared/Pharokka/7Apr2026_genomes_pharokka_phold_out/*.tar.gz`), checked in parts 001, 180 and 361.

## Pipeline

| step | script | job | notes |
|---|---|---|---|
| labels | `02_relabel.py` | 51707135 | the current CDS matcher, on the MI set's `.json.gz` chunks |
| ORF table | `01_build_orf_table.py` | 51707136 | 6,754,361 ORFs; every genome has its GenBank |
| models, genome-grouped CV | `03_models.py` | 51707137 | all seven feature sets |
| genome groups | `09_ncbi_genus.py` (login node), `10_genome_groups.py` | 51707640 | skani 0.3.2 `triangle --small-genomes`; also writes the per-genome non-ACGT counts |
| models, ANI / genus CV | `03_models.py --groups` | 51707641, 51707643 | intrinsic and length-only |
| ranking | `rank_missing_orfs.py --feature-set intrinsic` | 51707138 | |
| Foldseek arms | `05_select_queries.py` | 51707140 | 20,000 per arm, length-matched |
| Foldseek | `06_foldseek_search.slurm` | 51707141 (1–4), 51710800 (phold) | Foldseek 10.941cd33 |
| evidence | `07_evidence.py` | 51707142 | |
| case study | `11_case_study.*` | 51710801 | NC_042013 |
| tables here | `12_all_genomes_tables.py` | | |

The first phold search (51707141_5) was cancelled. `convertalis` refuses the taxid / taxname fields for a database without
taxonomy files, so the script now fills them with 0 / unknown. A Lustre slowdown on 2026-10-28 (14:00–16:30) stretched the
Foldseek jobs from minutes to two to five hours; the results are unaffected.

## Labels

2,685,268 of 6,754,361 ORFs (39.8 %) match an annotated CDS. The MI analysis's independent relabelling
(`phage_orf_labels_v2.parquet`) agrees on all but 114 ORFs. The labels stored in the MI JSONs are all False. They come from
the pre-0.2.0 matcher, so every analysis here uses the recomputed labels.

## Models (`tables/model_metrics_pooled.tsv`)

| feature set | CV grouping | AUC | AP |
|---|---|---|---|
| intrinsic (4 entropies + lengths) | genome | 0.9763 | 0.9695 |
| intrinsic | 95 % ANI cluster (10,186 groups) | 0.9749 | 0.9676 |
| intrinsic | NCBI genus (7,515 groups) | 0.9748 | 0.9676 |
| length only | genome | 0.7333 | 0.6607 |
| length only | genus | 0.7313 | 0.6576 |

The 31,200-genome set gave 0.9764 / 0.9696 for the intrinsic set. Every feature set matches it to the third decimal.

**Genome groups** (`tables/groups_summary.json`):
- 36,002 accessions have an NCBI taxid; 23,082 have a genus.
- At ≥ 95 % ANI over ≥ 85 % of the shorter genome, the genomes fall into 10,186 clusters (5,926 singletons). 30,099 genomes
  (84 %) have a near-identical relative.
- The genus grouping uses the NCBI genus. A genome without one falls back to its ANI cluster, and genera that share a
  cluster are merged.

Holding out whole clusters or genera costs 0.0015 AUC and 0.002 AP. Near-duplicate genomes therefore do not inflate the
out-of-fold metrics.

## Candidates (`tables/candidates_by_context.tsv`, `candidates_per_genome*`)

52,008 unannotated ORFs have out-of-fold p ≥ 0.8. 4,545 of them share a stop (3,464) or a frame (1,081) with an annotated
gene: they are pieces of that gene and are not candidates. That leaves 47,463 candidates (median 133 aa):
- 25,898 antisense to an annotated gene;
- 16,877 in another frame on the same strand;
- 4,688 intergenic.

They lie in 21,076 genomes (median 2 in those; 90th percentile of all genomes 3; maximum 28), and 14,949 genomes have none.
The count rises with genome length (Spearman ρ = 0.50); the median is 0.15 candidates per 10 kb.

**Ambiguity codes.** 1,311 genomes contain non-ACGT bases. Where they do, the ORF and CDS translations put `X` at different
positions, and the matcher can miss an ORF that *is* an annotated gene. These genomes hold 2,124 (4 %) of the p ≥ 0.8 ORFs.
The 31,200-genome analysis named OR354875 as the genome with the most candidates (30). It has 2,332 ambiguity codes, and 27 of
its 30 are its own annotated genes, left unmatched (`same_stop`).

## Foldseek evidence (`tables/evidence_*.tsv`)

Full-length hits (query and target coverage ≥ 0.8; E ≤ 10⁻³; 3Di + amino acids), 20,000 queries per arm:

| arm | any of 5 DBs | phold | BFVD | non-viral (PDB100, AFDB/SP, CATH50) |
|---|---|---|---|---|
| annotated | 67.9 % | 60.4 % | 64.5 % | 18.1 % |
| candidates | 36.2 % (35.6–36.9) | 31.6 % | 32.9 % | 6.5 % (6.2–6.8) |
| intergenic, p < 0.5 | 2.7 % | 2.0 % | 2.2 % | 0.11 % |
| frameshift, p < 0.5 | 0.31 % | 0.23 % | 0.19 % | 0.010 % |
| antisense, p < 0.5 | 0.14 % | 0.04 % | 0.07 % | 0.045 % |
| candidates, 3Di shuffled | 0.45 % | 0.43 % | 0.025 % | 0 |

**Excess over controls** (`evidence_excess.tsv`). The candidates lie 51–53 % of the way from the controls to the annotated
genes across all databases, and 35–36 % on the non-viral databases.

**phold behaves like BFVD.** Its hits to candidates are mostly near-identical (median 91 % identity; BFVD 98 %), so they
mostly show that a related genome's annotation called the same ORF. Its shuffled-3Di hits (85 of 20,000) are near-identical
amino-acid matches that survive the 3Di shuffle. The non-viral databases remain the independent test.

**By context** (`evidence_by_context.tsv`; any database / non-viral):

| context | any | non-viral |
|---|---|---|
| antisense | 11.3 % | 3.0 % |
| other frame | 65.4 % | 11.2 % |
| intergenic | 67.4 % | 8.5 % |

**Replacement pairs** (4,637):
- the replacement ORF alone has a full-length hit in 1,965 pairs, and the gene it would replace alone in 468;
- the replacement scores more bits in 57 %.

## Case study: Agrobacterium phage Atu_ph07 (NC_042013)

This is the genome with the most candidates in the searched contexts. It is a 490,380 bp jumbo phage with 826 CDS, carrying
28 candidates (17 antisense, 10 other-frame, 1 intergenic; 0.57 per 10 kb, 91st percentile). Its GenBank original, MF403008,
is in the same ANI cluster and ranks second (26).

Five of the 28 have full-length structural hits:
- orf108 (102 aa) is a PAAR-repeat protein: PDB 4JIW and CATH (35 % identity), phold and BFVD;
- orf104, orf844 and orf892 match this phage's own proteins in the submitters' annotation (BFVD, 83–100 %);
- orf875 matches a phold singleton.

Manuscript figure: `manuscript/notebooks/figPS02_atu_ph07.ipynb`.
