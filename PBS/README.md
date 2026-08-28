# PBS Pro examples (NCI Gadi)

Site-specific starting points for PBS Pro clusters, written against
[NCI Gadi](https://opus.nci.org.au). They are examples, not portable job
scripts. Before submitting one, adjust its project code, queue, storage
directives, environment prefix, model cache, input and output paths, wall
time, memory, and GPU request to the target site's policies.

The SLURM equivalents in [`slurm/`](../slurm) are Pawsey-oriented and use
`#SBATCH`. The two are not interchangeable: PBS Pro uses `-P` for the
project rather than `-A` for an account, requires an explicit `-l storage`
directive for every filesystem a job touches, and exposes array indices as
`$PBS_ARRAY_INDEX` rather than `$SLURM_ARRAY_TASK_ID`.

See the [HPC guide](../docs/source/hpc.rst) and
[installation guide](../docs/source/installation.rst) first.

## What makes Gadi awkward

**No queue has both GPUs and internet.** `copyq` reaches the outside world
but has no GPU and a hard cap of 1 CPU and 10 hours. `gpuvolta` and
`gpua100` have GPUs and no route off the machine. Login nodes have direct
outbound internet with no proxy.

Everything that needs the network therefore happens on a login node before
any GPU job is submitted: the conda environment, the PyTorch wheel, and the
encoder model. A GPU job that tries to reach Hugging Face on first use
consumes its allocation waiting for a connection that cannot succeed, which
is why the templates set `HF_HUB_OFFLINE=1` — failing immediately on a cache
miss is much cheaper than hanging.

**Filesystem choice is not cosmetic.** `/home` is capped at 10 GB, and a
CUDA PyTorch installation alone exceeds that. `/scratch` is periodically
swept. The environment, the model cache, the conda and pip caches, and all
output belong on `/g/data`, and each of those filesystems must be named in
`-l storage=` or the job cannot see it.

**`gpuvolta` allocates 12 CPUs per GPU** and rejects a smaller `ncpus`.

**GPU service units cost substantially more than CPU ones.** Measure before
scaling: run `estimate_tokens.pbs`, then time a single `pipeline.pbs`, and
size an array from those numbers rather than from a guess.

## Install

Run on a login node:

```bash
bash PBS/install.sh /g/data/<project>/<user>/conda/genome_entropy
bash PBS/download_model.sh /g/data/<project>/<user>/conda/genome_entropy \
                           /g/data/<project>/<user>/hf_cache
```

`install.sh` builds a conda environment on `/g/data`, keeps the conda and
pip caches off `/home`, installs a CUDA PyTorch wheel, installs this
repository with the `ml` extra, and prints a verification block.

Two Gadi-specific points the generic installation guide does not cover:

- **Python.** `genome_entropy` requires Python 3.10 or newer, and Gadi's
  default `python3` may be older. `install.sh` takes Python from conda; the
  `python3/3.10.4` and later modules are an alternative if you would rather
  use a `venv`.
- **NCBI `datasets`.** There is no Gadi module for NCBI's `datasets` CLI.
  `install.sh` installs `ncbi-datasets-cli` from bioconda. Drop it if you
  are not fetching genomes from NCBI.

### Check the wheel supports your GPU

`gpuvolta` is Tesla V100, compute capability `sm_70`; `gpua100` is A100,
`sm_80`. Recent CUDA toolkits have been dropping Volta, and a wheel without
`sm_70` kernels fails only once it reaches a GPU node, after the job has
queued and been charged. `install.sh` prints the wheel's architecture list
so you can confirm this before submitting anything:

```
torch: 2.13.0+cu126
arch list: ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
  gpuvolta (V100): OK
  gpua100 (A100): OK
```

That combination — `torch 2.13.0+cu126` from the `cu126` index — was
checked on Gadi and covers both queues. Re-check whenever you move to a
newer CUDA index; `cu128` and later are the ones likely to have dropped
Volta.

### `get_orfs`

`get_orfs` is an external executable rather than a Python dependency, and
needs Rust/Cargo. Build it on a login node — compute nodes cannot fetch
crates — and put it on `PATH` or set `GET_ORFS_PATH`.

### Model cache

`download_model.sh` writes to an `HF_HOME` you choose on `/g/data` and
verifies the model actually landed there rather than in `~/.cache`. Every
GPU job must then export the same path. ModernProst is loaded with
`trust_remote_code=True`, so this downloads Python that later executes on
the compute node; review it, and pin a revision, if your project requires
audited or reproducible code.

## Templates

| File | Queue | Purpose |
|---|---|---|
| `install.sh` | login node | conda environment, CUDA PyTorch, `genome_entropy[ml]`, `datasets` CLI |
| `download_model.sh` | login node | cache the encoder model on `/g/data` for offline GPU jobs |
| `estimate_tokens.pbs` | `gpuvolta` | benchmark an encoding token budget on the real device |
| `pipeline.pbs` | `gpuvolta` | one GenBank file to one results JSON |
| `pipeline_array.pbs` | `gpuvolta` | array over per-chunk manifests, skipping completed output |
| `encoder.pbs` | `gpuvolta` | protein JSON to structural-state JSON |
| `pytest.pbs` | `gpuvolta` | test suite inside a GPU allocation |
| `download_genomes.pbs` | `copyq` | fetch GenBank files from NCBI by accession |

The job templates read their paths from the environment so you can override
them at submission without editing the file:

```bash
qsub -v INPUT=/g/data/.../genome.gbff,OUTPUT=/g/data/.../genome.json \
     PBS/pipeline.pbs
```

`ENV_PREFIX`, `HF_CACHE`, `GET_ORFS_PATH`, and `OUT_ROOT` are overridable
the same way. The `YOUR_NCI_PROJECT` placeholders in the `#PBS` directives
must be edited, because PBS does not expand variables in directive lines.

## Multi-GPU under PBS

GPU discovery checks `SLURM_JOB_GPUS`, then `SLURM_GPUS`, then
`CUDA_VISIBLE_DEVICES`. Under PBS Pro only the last of those is set, and it
remaps allocated devices to local indices, so pass local IDs — `--gpu-ids
0,1` — rather than physical device numbers. Request `ngpus=2` with
`ncpus=24` on `gpuvolta` and add `--multi-gpu`.

## Checking a run

`qstat -x <jobid>` shows exit status after a job leaves the queue; an array
reports per-index status with `qstat -xt <jobid>[]`. `pipeline_array.pbs`
exits non-zero when any genome in its chunk failed, so a partially failed
index is visible rather than being mistaken for a clean run, and it skips
genomes whose output already exists so a resubmitted index does not repay
for GPU time already spent. Check exit statuses before launching the next
stage.

## Status

The `#PBS` directives, queue constraints, and offline-cache handling here
reflect Gadi's documented limits, and the login-node install path has been
exercised on Gadi. The GPU job templates themselves have not been run
through a `gpuvolta` allocation, so treat wall times, memory figures, and
the token budget as starting points to measure rather than as tested
defaults.
