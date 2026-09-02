# GTDB r232 → `genome_entropy`, on NCI Gadi

The driver scripts for the full-scale GTDB validation run: **189,715 bacterial
and 10,122 archaeal species representatives, 2.62 billion ORFs**, encoded to
3Di and 12-state on Gadi's `gpuvolta` queue, then analysed for protein-coding
genes that annotation missed.

**Read [`../scientific_report.md`](../scientific_report.md)
first** for what was found. This file covers only what the scripts are and the
order they run in.

> This README described an untested first draft until 2026-09-03. Everything
> below now describes what actually ran. Two claims in that draft were wrong and
> are worth recording because they were plausible: the `--model` flag was real,
> but the cost estimate was not — encoding came in at ~80,000 SU, roughly 8% of
> a 1 MSU grant, and dominates everything else by two orders of magnitude.

## Scale and cost, measured

| | bacteria | archaea |
|---|---:|---:|
| genomes | 189,715 | 10,122 |
| ORF rows | 2,568,244,984 | 54,858,398 |
| chunks | 760 | 41 |
| genomes with ≥1 deposited CDS | 96,875 (51.1%) | 5,492 (54.3%) |
| encoding cost | ~80,000 SU | ~4,000 SU |

`gpuvolta` charges ~36 SU per GPU-hour. Everything downstream of encoding —
counting, classification, Foldseek, the missed-gene analysis — is under 2,000 SU
combined.

## Gadi constraints that shape the design

- **No queue has both internet and GPUs.** `copyq` reaches the network (1 CPU,
  10 h cap); `gpuvolta` does not. Model weights are pre-cached on a login node
  (`03b_download_model.sh`) into `/g/data` before any GPU job runs.
- **Inodes bind before bytes.** Per-genome files must never land on the shared
  filesystem — `02` and `04` stage on `$PBS_JOBFS` and return two files per
  chunk (~10,800 inodes rather than ~600,000).
- `gpuvolta` enforces 12 CPUs per GPU. `max_array_size` is 10, so a 41-subjob
  array needs five submissions.
- Every job needs `-l storage=gdata/ob80`; `/scratch` is auto-swept, so
  genomes, model cache and output all live on `/g/data`.

## The pipeline

**Setup and calibration** — login node, then one-off jobs.

| script | purpose |
|---|---|
| `00_smoke_test.sh`, `00b_smoke_test_gpu.pbs` | correctness on 5 genomes, CPU then GPU |
| `00c`–`00e_calibrate_*.pbs` | `--encoding-size` (leave at default), per-GPU parallelism (`PARALLEL=4`) |
| `01_get_gtdb_reps.sh`, `01b_make_chunks.sh` | GTDB metadata → accession chunks. `01b` verifies the chunk set; every later stage derives its expected inputs from it |
| `03_install_genome_entropy.sh`, `03b_download_model.sh` | environment and offline model cache |

**Encoding** — the expensive part.

| script | purpose |
|---|---|
| `02_download_genomes.pbs` | NCBI `datasets` dehydrated → rehydrate, per chunk, on `copyq` |
| `04_run_entropy.pbs` | `gpuvolta` array; per-chunk `.tar.zst` of per-genome JSON |
| `04b_regenerate_entropy_rows.pbs` | re-emits per-ORF TSVs carrying `contig_length`, which the originals lack |
| `05_aggregate_results.py`, `05b` | per-ORF and per-genome summaries |
| `06`, `07` | failure diagnosis, ORF counts |

**Figures and annotation status.**

| script | purpose |
|---|---|
| `08b_sample_for_figures.pbs`, `08`, `09`, `figstyle.py` | sampling and plots |
| `11_genome_annotation_status.pbs` | per-genome `in_genbank` counts |
| `12_genome_cds_counts.pbs` | **authoritative** CDS counts, parsed from the GenBank records |
| `13_cds_intervals.pbs`, `13b_cds_intervals_all.pbs`, `cds_intervals.py` | deposited CDS coordinates; `13b` parallelises across chunks |

**Missed-gene classification.**

| script | purpose |
|---|---|
| `10_missed_genes.py` | the classifier: shadow / candidate / `unannot_hi` / controls |
| `13_missed_gene_candidates.pbs` | runs it across chunks |
| `16_candidate_burden.py` | per-genome burden against GTDB metadata |

**Structural search.**

| script | purpose |
|---|---|
| `12_foldseek_databases.pbs` | PDB100, AFDB/Swiss-Prot, CATH50, BFVD |
| `14_pilot_queryset.py/.pbs`, `19_full_queryset.pbs` | matched query sets; `19` scales to the full candidate population |
| `14b_extract_orf_seqs.py/.pbs` | pulls AA + 3Di back out of the archives |
| `15_build_query_db.py` | Foldseek query DB from precomputed 3Di |
| `17_pilot_search.pbs` | the searches, plus best-hit reduction into `best/` |
| `18_pilot_analysis.py`, `18b_analysis.pbs` | coverage, mixture estimates, controls |

**Full-population analysis and manuscript output.**

| script | purpose |
|---|---|
| `20_orf_context.py/.pbs` | genomic context, remaining entropy axes, reading-frame class |
| `21_target_descriptions.py` | target id → product, from the Foldseek header DBs |
| `22_rank_candidates.py/.pbs` | ranked table and direct-evidence ladder |
| `23_gtdb_prodigal_coords.pbs` | GTDB Prodigal coordinates (123 GiB archive, streamed) |
| `24_prodigal_overlap.py/.pbs` | ORF-level agreement with an independent gene caller |
| `25_functional_classes.py/.pbs` | functional classes, mobile-element and BFVD-only flags |
| `26_select_examples.py/.pbs`, `27_build_dossiers.py/.pbs` | manuscript examples and dossiers |

## Traps that each cost real time

- **Numbering collides.** `12` and `13` each name two different scripts
  (`12_genome_cds_counts` vs `12_foldseek_databases`; `13_cds_intervals` vs
  `13_missed_gene_candidates`). Always write the full filename.
- **`chunk` means two different things.** In `genome_cds_counts_*.tsv` it is the
  full tag (`arc_038`); in the wanted lists it is the bare number (`038`).
  Prefixing blindly yields `arc_arc_038`.
- **`qsub -v` splits on commas**, so a comma-separated list cannot be passed
  that way. `19` derives its chunk list inside the script instead.
- **`set -o pipefail` inverts `zcat f | head -1 | grep -q X`**: `head` closes the
  pipe, `zcat` dies of SIGPIPE, and the pipeline reports failure even when
  `grep` matched. Use command substitution with `|| true`.
- **`sort` spills to `$TMPDIR`, which is jobfs.** A verification step that sorts
  600 M values will exceed the 100 MB default quota and be killed — after the
  work succeeded. Publish output *before* computing statistics about it.
- **PBS copies the job script at submit time**, so editing a queued script does
  not change what it runs.
- Resume guards must key on the **newest** output column, or a re-run
  republishes stale tables under a clean exit status.

## The defect family this run kept producing

> A stage verifies whatever inputs happen to be present, then publishes an
> artefact downstream consumers treat as covering the whole domain.

It appeared seven times: four in the original stages, once as the coordinate
defect that swapped two analysis arms, once in a diagnostic that independently
made the same coordinate mistake it existed to detect, and once as an analysis
stage that skipped itself and dropped its headline table from the report under
exit status 0. The fix in every case was to derive the expected set from an
authoritative source and make a shortfall fatal, not to add a warning.
