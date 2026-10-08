# Validation: LOGAN assembled metagenomes vs Prodigal

Issue [#102](https://github.com/linsalrob/genome_entropy/issues/102). Uses
LOGAN v1.2 contigs as a large, independent metagenomic validation set:
`genome_entropy` over the full ≥30 aa six-frame ORF space, compared against
Prodigal `-p meta` run on the same contigs.

The work is staged. **Phase A** (~100 accessions) validates the workflow
and produces the cost projections. **Phase B** (5–10k) and anything larger
are not run until Phase A has passed its acceptance checks and the
projections have been reviewed on the issue.

## Pipeline

| script | where | role |
|---|---|---|
| `00_install_env.slurm` | gpu-dev | conda env (Prodigal 2.6.3, DuckDB, pandas), ROCm PyTorch, `genome_entropy` from this checkout (commit recorded), model cache |
| `01a_fetch_metadata.slurm` | copy | LOGAN SRA metadata CSV + v1.2 contig seqstats, size-checked |
| `01_select_accessions.py candidates` (`.slurm`) | work | one sequential pass → `meta/logan_metagenomic_candidates.parquet`, census, exclusion ledger |
| `01_select_accessions.py pilot` | login | stratified draw → `manifests/pilot_accessions.tsv`, seeds state files |
| `02_cpu_stage.py` (`.slurm`, array) | work | download → validate → filter ≥120 nt → Prodigal → ORFs → Prodigal matching |
| `03_plan_chunks.py` | login | pack ORFs into ~1 M-ORF chunks; global, never-reused ids |
| `03_encode_chunk.py` (`.slurm`, array) | gpu | ModernProst → validate → `.tsv.zst` → re-read check |
| `04_upload_verify.py` (`.slurm`) | copy | rclone copy → `rclone check` + size/QuickXorHash → manifests → optional cleanup |
| `05_pilot_qc.py` | work | §9 acceptance checks, Prodigal± separation, inspection samples, cost projection |
| `logan_common.py`, `env.sh` | — | paths, states, provenance; environment activation and import check |

```bash
cd $LOGAN_ROOT                                   # /scratch/$PAWSEY_PROJECT/$USER/Logan
S=~/GitHubs/genome_entropy/Validation/logan/scripts
sbatch --wait $S/00_install_env.slurm
sbatch --wait $S/01a_fetch_metadata.slurm
sbatch --wait $S/01_select_accessions.slurm
source $S/env.sh && python $S/01_select_accessions.py pilot --n 100 --seed 102
sbatch --wait --array=1-100 $S/02_cpu_stage.slurm manifests/pilot_accessions.tsv --repro-check
python $S/03_plan_chunks.py --accessions manifests/pilot_accessions.tsv --final
ls manifests/chunks | sed 's/.json$//' > manifests/chunks_to_encode.txt
sbatch --wait --array=1-$(wc -l < manifests/chunks_to_encode.txt) $S/03_encode_chunk.slurm \
       manifests/chunks_to_encode.txt --workers 4
sbatch --wait $S/04_upload_verify.slurm --cleanup
python $S/05_pilot_qc.py --batch pilot
```

### Phase B (tens of thousands of accessions)

The pilot path runs one array element per accession and one GPU per chunk;
neither scales past Setonix's limits (arrays of 1,000, 256 running `work`
jobs, 64 running `gpu` jobs). Phase B uses:

| script | where | role |
|---|---|---|
| `01_select_accessions.py draw` | login | water-filled biome × size strata, ≤25 runs per BioProject, excludes anything already selected |
| `02_cpu_batch.slurm` | work, half node | 100 accessions per element, 12 at a time; per-accession logs in `logs/cpu/`; Prodigal repro check on every 100th |
| `03_encode_node.slurm` | gpu, whole node | 8 streams (one per GCD, `ROCR_VISIBLE_DEVICES`), 4 persistent encoder workers each, zstd -19 overlapped with encoding |
| `04_upload_verify.py` | copy | one rclone copy/check/listing per remote folder; per-chunk Prodigal bundles |
| `06_orchestrate.slurm` | long, 1 core | plans, submits GPU arrays as chunks become ready, retries once, uploads, stops when done |

```bash
python $S/01_select_accessions.py draw --n 25000 --name phaseb1 --seed 102
C=$(sbatch --parsable --array=1-250 $S/02_cpu_batch.slurm manifests/phaseb1_accessions.tsv 100 12)
sbatch $S/06_orchestrate.slurm --batch phaseb1 --cpu-job $C
cat manifests/phaseb1_progress.json        # refreshed every 20 min
```

Phase B remote layout differs from the pilot's in two ways. Chunks and
their manifests sit in 1,000-chunk shard folders
(`modernprost-50M/v1/<shard>/`, `manifests/chunks/<shard>/`). Prodigal
tables travel as one tar per chunk (`prodigal/v1/bundles/<shard>/<chunk>.tar`)
holding every accession whose first row is in that chunk; the 91 pilot
accessions keep their per-accession folders. Always resolve a chunk's files
through its manifest (`cache_file`, `remote`, `bundle`).

Every stage is resumable. Each accession carries a state file
(`state/<acc>.json`) that moves through

```
selected → downloaded → validated → prodigal_done → orfs_called
         → genome_entropy_done → cache_validated → uploaded → remote_verified → complete
```

Each chunk carries a manifest (`manifests/chunks/<id>.json`) that moves
through `planned → cache_validated → uploaded → remote_verified`. Rerunning
any stage skips work already recorded.

## Conventions

**Coordinates.** Every table uses 0-based half-open coordinates on the
contig's forward axis. ORFs are converted exactly once, by
`genome_entropy.io.genbank.normalise_orf_interval`, and Prodigal's 1-based
inclusive coordinates become `(start − 1, end)`. Nothing downstream converts
again. `orf_id` is `<contig>:<start>-<end>:<strand>`.

**Cache rows** (`modernprost-50M/v1/logan_metagenome_NNNNNN.tsv.zst`):
`accession contig orf_id start end strand contig_len stop aa_sequence three_di twelve_state`.
`stop` = 1 if the ORF ends in a stop codon (0 = runs off the contig). AA,
3Di and 12-state sit on one row, so they cannot drift apart. Derived
quantities (entropies, MI, coding probability) are not cached; recompute them.

**Prodigal labels** (`prodigal/v1/<acc>/orf_prodigal.tsv.zst`):

- `Prodigal+` requires both a Prodigal gene on the same contig and strand
  ending at the same stop and acceptance by the package's tolerant matcher
  (`evaluate_orf_genbank_cds_match`: phase, ≥90% overlap, ≥98% AA identity).
- `prodigal_class` labels the rest: `frameshift_overlap`, `antisense_overlap`,
  `frame_undefined_overlap`, `same_frame_unmatched`, `marginal_overlap` (best
  overlap 1–29 bp), `intergenic`. The relation comes from
  `../gtdb/scripts/20_orf_context.py::frame_class`.

## Tool behaviours this depends on

Each of these was verified on LOGAN contigs, and each is handled in
`02_cpu_stage.py` with an assertion.

- **`get_orfs -l` is in amino acids** (#103), so `genome_entropy run
  --min-aa 30` filters at ~90 aa. Here the floor is passed through as
  amino acids (`-l 29`, then `aa_len ≥ 30`).
- **ORFs open at a contig's 3′ end are reported with `end = contig_len + 1`.**
  The wrapper resets `end` to the last complete codon, after asserting that
  the same rule reproduces `end` for every stop-terminated ORF.
- **Prodigal writes `M` for GTG/TTG initiators.** That residue is aligned to
  the ORF's before the identity test, and flagged `initiator_substituted`.
- **Prodigal `-p meta` gives some short contigs a genetic-code-4 model.**
  Their genes read through TGA and cannot share a stop with a table-11 ORF.
  `best_gene_transl_table` records it.
- **LOGAN accession CSV.** DuckDB's parallel CSV reader rejects it, and
  materialising it needs >90 GB, so it is scanned once, sequentially, and
  streamed to Parquet.

## Site notes (Setonix)

- The environment lives on `/scratch`, because `/software` is short of
  inodes. `00_install_env.slurm` rebuilds it if the 21-day purge takes it.
- Encoding jobs request one GCD each (`--gres=gpu:1 --ntasks=1`): the
  allocation pack is 1 GCD, 8 cores and ~29 GB. `--workers` sets how many
  encoder processes share that GCD; it is calibrated in the pilot.
- The remote is OneDrive (Teams). `rclone check` compares QuickXorHash, and
  the manifest records the remote hash.
