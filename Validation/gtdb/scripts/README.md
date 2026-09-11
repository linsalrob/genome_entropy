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
| `29_population_entropy_summary.pbs`, `population_agg.c` | full-population entropy statistics: one pass over all 2.62 billion ORF rows, 24 CPUs, 5 min, 4 SU |
| `population_summary.py`, `population_tables.py` | combine the partials; render the report tables from the TSVs |

`29` exists because the report's entropy comparisons came from samples, and the
manuscript needs exact population counts. It reads `entropy_rows/` only — no
re-encoding — and stratifies every row by (genome carries a deposited CDS,
`in_genbank`) in one streaming pass, because those four strata compose into
every group anyone has asked for and the alternative is reading 143 GB once per
question. Quantiles come from 1e-4 histograms, so they are bounded rather than
exact; counts and means are exact. It keeps its per-worker partials for the same
reason: the second question should not cost another full read. It did, and they
did not exist yet.

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
- **An aggregation that discards its intermediates buys one answer.** `29`'s
  first run threw away its per-worker partials, so adding a single derived
  statistic meant re-reading all 143 GB. Cheap here (4 SU); the habit is not.
- **A filter's stated rationale is a claim, and can be wrong in the direction
  that matters.** §4 justified excluding never-annotated genomes by saying their
  ORFs sit in the low-3Di band. The population statistics show the opposite —
  they are 23-fold enriched above the candidate threshold — which is a stronger
  reason for the same exclusion and the one §6 had already measured. The filter
  was right; the sentence explaining it was not, and nothing checked it because
  the conclusion was correct.

## Stages 30-33: the length-conditioned entropy references

`30` recovers residue composition from the packed JSON (the entropy rows carry
none), `31` simulates the random reference, `33` builds the empirical one over
all 2.62 billion rows, and `32` scores against either or both. See §4.2.

Three lessons worth carrying:

- **A statistic's null has to match the statistic.** Shannon entropy is
  permutation-invariant, so the obvious shuffle null has zero variance and
  measures nothing. The i.i.d. null is the right shape, but it models only
  multinomial sampling noise, and real between-protein composition variance is
  far larger — so its z has sd 3-7 rather than 1 and its percentile is not a
  significance statement. The reference that calibrates is the empirical
  distribution of real ORFs at the same length.
- **Test calibration on a representative sample, not a convenient one.** Two
  attempts to check the empirical table scored the head of one chunk, then of
  eighteen chunks, and both showed a tilted distribution that looked like a
  defect in the table. The head of a chunk is its first few genomes. Against a
  systematic sample of the population the table is exact (z sd 0.9986). The
  second attempt is the instructive one: spreading over more chunks did not
  help, because the bias was in taking the head of each, not in using too few.
- **Do not carry an approximation's failure mode across from a toy case.** The
  asymptotic entropy variance degenerates for exactly uniform p, and a
  prototype built on uniform p suggested the analytic interval was unusable for
  protein. Against the real composition it is 26% too narrow at 90 aa and 9% at
  300 aa — a real error worth avoiding, but not the one the prototype implied.

### Guards added after the PR #101 review

Codex found three more instances of the family below in stages 30-33, all
valid, all fixed before merge:

- **`33` discovered its inputs with `ls`.** A domain missing chunks would have
  aggregated the rest and published it as complete. Both `30` and `33` now
  derive the expected chunk set from `accessions/`, which `01b_make_chunks.sh`
  writes and verifies, and abort on any shortfall.
- **`set -o pipefail` is not inherited by `bash -c`.** Each `33` worker runs
  its `gzip -dc | length_agg` pipeline in a child shell, so a truncated chunk
  let gzip fail after a valid header while the aggregator consumed the prefix
  and exited 0. Demonstrated on a deliberately truncated chunk: the worker
  reported success having read 29,948 of 50,000 rows. Each worker now sets
  `set -euo pipefail` itself, and a failing worker aborts the domain.
- **`30` treated an unreadable archive as a smaller sample.** It warned and
  carried on, and the composition it produced drives every simulated
  reference. Read failures are now fatal, outputs are staged and only renamed
  into place once every chunk has succeeded, and a non-zero `zstd` exit is
  distinguished from the intended SIGPIPE of abandoning a stream early.

A fourth, in `32`: a custom `--background` with `--fasta` or `--table` did not
require an alphabet, so the stage 30 composition file was pooled across
protein, 3Di and twelve-state into one vector -- plausible-looking scores with
no meaning. The alphabet is now resolved for every input mode and a mismatch
is refused.

None of this changes the published numbers: the run behind §4.2 read all 760
bacterial and 41 archaeal chunks with zero malformed rows, and its marginals
reproduce §4.1 exactly. The guards are there so the next run cannot quietly
do less.

`29_population_entropy_summary.pbs` carried the same two shell-level defects
and gets the same two guards. Its published §4.1 figures are unaffected and
independently corroborated — the stage 33 aggregator reproduces every one of
its stratum totals — but it would have had the same silent failure modes on a
re-run.

Both guards need the accession manifests, which live beside the pipeline
scripts in `claude/` rather than in this repository copy. A run driven from
the repo tree must point `ACC` at them; the scripts say so and fail with that
message rather than a bare `ls` error.

### Second review round on PR #101

Three more, all valid:

- **The domain axis of a custom background was unfiltered.** I had fixed the
  alphabet axis in the first round and left its sibling: `--background` with
  `--domain arc` but no `--background-domain` summed both domains' frequency
  rows, and since each sums to one that averages them. Both axes now default
  from the labelled run, refuse a mismatch, and `load_background` refuses to
  pool a file carrying either column unfiltered.
- **Empirical cells are bins, not samples of a curve.** The generic
  interpolation treated bin midpoints as points on a function, so a 1000 aa
  ORF was scored mostly against the exact-length-999 distribution.
  `EmpiricalTable` now selects the containing bin, using `length_lo`/
  `length_hi` added to the npz; the simulated grid keeps interpolation.
  Aggregate calibration is unchanged to four decimals because 99.5% of ORFs
  are below 1000 aa where bins are 1 aa wide.
- **Malformed rows were counted and then published around.** A short or
  unparseable row is skipped by the aggregator and absent from *both* the
  histogram and the exact counters, so the consistency check cannot see it.
  The summariser now aborts on any malformed row, with `--allow-malformed` as
  the deliberate escape hatch.

The npz was regenerated **from the kept partials**, not by re-reading the
143 GB — which is what keeping them was for. The published TSVs came back
byte-identical in both domains.

## The defect family this run kept producing

> A stage verifies whatever inputs happen to be present, then publishes an
> artefact downstream consumers treat as covering the whole domain.

It appeared seven times: four in the original stages, once as the coordinate
defect that swapped two analysis arms, once in a diagnostic that independently
made the same coordinate mistake it existed to detect, and once as an analysis
stage that skipped itself and dropped its headline table from the report under
exit status 0. The fix in every case was to derive the expected set from an
authoritative source and make a shortfall fatal, not to add a warning.
