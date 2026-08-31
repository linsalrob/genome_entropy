# Validation: GTDB r232 representative genomes

A full-scale run of `genome_entropy` over every bacterial species
representative in GTDB release 232 — **189,715 genomes, 2.57 billion ORFs** —
on NCI Gadi (PBS Pro, Tesla V100). Archaea (10,122 genomes) followed.

The point of keeping this in the repository is not the scripts, which are
site-specific, but the record: what the tool does at this scale, what it costs,
which parameters actually matter, and the defects the run exposed.

**Read [`scientific_report.md`](scientific_report.md) first.** It carries the
results, methods, limitations and outstanding work. This file covers only how to
re-run the thing.

## Findings that generalise beyond this site

- **`--encoding-size` should be left at its default.** Raising it is slower: the
  default was fastest at 119.5 s/genome, 100000 cost 9%, and 800000 cost 14%. A
  single `run` leaves the GPU at 12–48% utilisation on 1.5–2.1 GB of 32 GB,
  which invites the opposite conclusion, but memory was never the binding
  resource. See [`calibration/`](calibration/).
- **Run ~4 `genome_entropy` processes per GPU.** Throughput saturates at 4
  (2.04×), taking utilisation from 12% to 79%. This cut the projected bacterial
  cost from ~227,000 to ~74,000 service units.
- **3Di entropy has hard ceilings at log₂(k)** for k distinct 3Di states, and a
  large population of ORFs encodes to only 2–3 states (76% of their residues are
  `D`), so their entropy is mechanically capped at log₂(3) = 1.585. Any
  threshold analysis on 3Di entropy should use that constant rather than a value
  read off a plot. The ceiling follows from the number of states alone; what
  those states correspond to structurally is not established here, and 3Di
  letters are learned tertiary-interaction states rather than named
  secondary-structure categories. §5 of the report.
- **`in_genbank=False` does not mean "annotation declined this ORF".** At least
  46% of GTDB bacterial representatives carry no CDS annotation at all, so 42.8%
  of all ORFs are `False` by construction. Filter on annotation status before
  interpreting the flag. Take that status from `12_genome_cds_counts.pbs`, which
  reads the GenBank records: `in_genbank` alone cannot tell a genome with no
  annotation from one whose annotations the ORF matcher rejected, so the 46% is
  an upper bound. §6.
- **Long ORFs crash the encoder.** 39 genomes (0.021%) failed with CUDA OOM in
  `_make_sliding_mask`, which materialises an L×L matrix for a banded mask;
  worst case attempted 4,374 GiB for a 766,188-aa ORF call. §7.

## Pipeline

| script | queue | role |
|---|---|---|
| `00_smoke_test.sh` | login | tool checks, 5-genome download |
| `00b_smoke_test_gpu.pbs` | `gpuvolta` | first real GPU run, schema inspection |
| `00c_calibrate_encoding_size.pbs` | `gpuvolta` | batch-budget sweep, peak device memory |
| `00d_calibrate_bacterial.pbs` | `gpuvolta` | the same on bacterial genomes, with repeats |
| `00e_calibrate_parallelism.pbs` | `gpuvolta` | processes per GPU vs throughput |
| `01_get_gtdb_reps.sh` | login | GTDB metadata → per-domain accession lists |
| `01b_make_chunks.sh` | login | split one domain into chunks; prints the `-J` range |
| `02_download_genomes.pbs` | `copyq` | NCBI download, one archive per chunk |
| `03_install_genome_entropy.sh` | login | conda environment (wraps `../../PBS/install.sh`) |
| `03b_download_model.sh` | login | cache the model for offline GPU use (wraps `../../PBS/download_model.sh`) |
| `04_run_entropy.pbs` | `gpuvolta` | encoding; archive + per-ORF TSV per chunk |
| `extract_entropy_rows.py` | (called by 04) | JSON → per-ORF entropy rows, including contig length |
| `05_aggregate_results.py` | login | per-genome summary, one domain at a time |
| `06_diagnose_failures.pbs` | `gpuvolta` | re-run failed genomes, capture the cause |
| `07_count_orfs.pbs` | `normal` | exact ORF / `in_genbank` counts, cached per chunk |
| `11_genome_annotation_status.pbs` | `normal` | per-genome table of whether any ORF matched a CDS |
| `12_genome_cds_counts.pbs` | `normal` | per-genome CDS counts read from the GenBank records — the authoritative annotation-presence answer |
| `13_cds_intervals.pbs` | `normal` | deposited CDS coordinates for named chunks, for the shadow test |
| `cds_intervals.py` | (called by 13) | GenBank CDS locations → interval TSV |
| `08_plot_entropy_scatter.py` | login | four-panel scatter |
| `09_plot_density.py` | login | hexbin and 2D-KDE figures |
| `10_missed_genes.py` | login | candidate genes missed by annotation; needs `--annotation-status` from `12` and `--cds-intervals` from `13` |

### Order

`03` must run before `00_smoke_test.sh`, which activates the environment it
creates.

```bash
bash 03_install_genome_entropy.sh              # creates the conda prefix
bash 00_smoke_test.sh
bash 03b_download_model.sh                     # login node: GPU nodes are offline
bash 01_get_gtdb_reps.sh
bash 01b_make_chunks.sh bac 250                # prints TOTAL_CHUNKS and -J

qsub -v DOMAIN=bac,TOTAL_CHUNKS=760,STRIDE=10 -r y -J 0-9 02_download_genomes.pbs
for s in $(seq 0 10 750); do                   # max_array_size=10, so blocks
  qsub -v DOMAIN=bac,CHUNK_START=$s -r y -J 0-9 \
       -o "logs/entropy_bac_${s}_^array_index^.log" 04_run_entropy.pbs
done

qsub -v DOMAIN=bac 07_count_orfs.pbs
qsub -v DOMAIN=bac 11_genome_annotation_status.pbs
qsub -v DOMAIN=bac 12_genome_cds_counts.pbs
python3 05_aggregate_results.py --domain bac
```

`12` reads the GenBank archives, so run it before `04` deletes them
(`DELETE_GENBANK_AFTER=1`) or re-download with `02`. Its table is what
`10_missed_genes.py` needs:

```bash
qsub -v DOMAIN=bac,CHUNKS="000 051" 13_cds_intervals.pbs

python3 10_missed_genes.py \
    --annotation-status /g/data/.../genome_cds_counts_bac.tsv \
    --cds-intervals     /g/data/.../cds_intervals/bac
```

`13` parses records properly rather than grepping, so it is run only over the
chunks an analysis needs; `12` stays cheap enough for the whole corpus. The
shadow test needs `13` because the spans of ORFs that happened to match a CDS
are not the CDS set: a CDS the matcher rejected is missing from them, and a
matched ORF runs stop to stop and can extend past the deposited CDS.

Then repeat with `DOMAIN=arc`.

## Adapting this to another site

Paths are hardcoded to the machine this ran on. Change:

| what | where |
|---|---|
| `#PBS -P`, `-l storage=` | every `.pbs` header |
| `GDATA_ROOT` | `02`, `04`, `05`, `06`, `07`, `11` |
| `ENV_PREFIX` (conda prefix) | `03`, `03b`, and every job that activates an environment |
| `GET_ORFS_PATH` | `00`, `00b`–`00e`, `04`, `06` |
| `HF_HOME` | `00`, `00b`–`00e`, `03b`, `04`, `06` |
| `$GE_SCRATCH` | `08`, `09`, `10` — read from the environment, defaults to `./work` |
| `GDATA_ROOT`, `GENBANK_DIR` | `12`, `13` — overridable from the environment |

For a portable starting point rather than this record, use the templates in
[`../../PBS/`](../../PBS) instead.

## Site constraints worth knowing

Discovered the hard way on Gadi; most apply to any PBS Pro site.

- **No queue has both GPUs and outbound internet.** The model must be cached
  from a login node and jobs run with `HF_HUB_OFFLINE=1`, so a cache miss fails
  fast instead of hanging on an unreachable network.
- **`max_array_size = 10`** — far smaller than `max_queued = 1000` suggests, and
  a wider array is refused outright. `02` strides chunks across subjobs; `04`
  submits blocks offset by `CHUNK_START`.
- **Array jobs need `-r y`.** Gadi defaults to `-r n` and PBS refuses
  non-rerunable arrays: `qsub: cannot submit non-rerunable Array Job`.
- **`CUDA_VISIBLE_DEVICES` is unset inside a GPU job** — PBS restricts
  visibility by cgroup — so `genome_entropy` falls through to PyTorch's device
  count, which is correct. Do not set it by hand.
- **`copyq` is capped at 10 hours** for a 1-CPU job.
- **A `-o` path containing `^array_index^` collides** when several arrays share
  an array index; include a per-array prefix or lose the logs.
- **Throttle NCBI.** `-W max_run_subjobs=N`; PBS Pro has no `%N` suffix. NCBI
  allows 3 requests/second per IP without an API key, 10 with one.

## Inodes, not bytes

The binding constraint was the inode quota: ~520,000 free against 9 TB of free
space. NCBI `datasets` writes two inodes per genome and the pipeline one JSON per
genome, so the obvious layout wants ~600,000 inodes and fails on a filesystem
with terabytes spare.

`02` and `04` therefore stage all per-genome work on node-local `$PBS_JOBFS` and
return exactly two files per chunk: a `zstd` archive of the JSON and a gzipped
per-ORF TSV. That is **~10,800 inodes instead of ~600,000**.

Measured along the way: entropy JSON is 2.3–4.9× its GenBank input; `zstd -3`
compresses it ~3.4× at roughly ten times gzip's speed; and the entropy values
extracted to TSV are **~42× smaller** than the JSON. Downstream analysis reads
the TSVs and never unpacks an archive.

## Cost

~87,500 service units for bacteria end to end, of which ~80,000 was GPU
encoding. `gpuvolta` charges ~36 SU per GPU-hour. Calibration — the work that
established the two parameter choices above and saved ~115,000 SU — cost about
50 SU.
