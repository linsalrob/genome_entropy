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

`get_orfs` is an external executable rather than a Python dependency. It
is a C project built with CMake, and Gadi's default `gcc` and `cmake`
(3.26.5) are sufficient — no module load and no Rust toolchain:

```bash
git clone https://github.com/linsalrob/get_orfs
cd get_orfs && mkdir build && cd build
cmake .. && make && cmake --install . --prefix ..
```

Build it on a login node, since compute nodes cannot clone the repository,
and put the binary on `PATH` or set `GET_ORFS_PATH`.

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
| `pipeline_array.pbs` | `gpuvolta` | array over `PREFIX`-named chunk archives, resuming partial and completed output |
| `encoder.pbs` | `gpuvolta` | protein JSON to structural-state JSON |
| `pytest.pbs` | `gpuvolta` | test suite inside a GPU allocation |
| `download_genomes.pbs` | `copyq` | fetch GenBank files from NCBI by accession into one archive per chunk |
| `extract_entropy_rows.py` | (called by the array job) | pull per-ORF entropy values out of JSON into a TSV |

The job templates read their paths from the environment so you can override
them at submission without editing the file:

```bash
qsub -v INPUT=/g/data/.../genome.gbff,OUTPUT=/g/data/.../genome.json \
     PBS/pipeline.pbs
```

`ENV_PREFIX`, `HF_CACHE`, `GET_ORFS_PATH`, and `OUT_ROOT` are overridable
the same way. The `YOUR_NCI_PROJECT` placeholders in the `#PBS` directives
must be edited, because PBS does not expand variables in directive lines.

## Array jobs

Two things about PBS Pro arrays that are easy to discover the hard way:

**Arrays must be rerunable.** Gadi defaults jobs to `-r n`, and PBS refuses
an array submitted that way:

```
qsub: cannot submit non-rerunable Array Job
```

Both array templates set `#PBS -r y`. Keep it.

**Check `max_array_size` before designing around arrays.** It can be far
smaller than the queue's `max_queued` suggests — Gadi sets it to **10**:

```bash
qmgr -c "print server" | grep max_array_size     # -> set server max_array_size = 10
```

An array wider than that is refused at submission with `qsub: Array job
exceeds server or queue size limit`. With hundreds of chunks to process,
the workable pattern is to let each subjob *stride* through the chunk list
rather than own a single chunk — subjob `k` takes chunks `k`, `k+STRIDE`,
`k+2*STRIDE`, … so a 10-wide array still covers 760 chunks:

```bash
qsub -v PREFIX=bac,TOTAL_CHUNKS=760,STRIDE=10 -r y -J 0-9 \
     PBS/download_genomes.pbs
```

That also pins concurrent NCBI streams to exactly `STRIDE`, which is what
you want anyway. `download_genomes.pbs` implements this and skips chunks
whose archive already exists, so a resubmitted array resumes.

**`PREFIX` names the dataset end to end.** It is what `download_genomes.pbs`
calls the archives it writes, so `pipeline_array.pbs` needs the same value
to find them; pass it to both:

```bash
qsub -v PREFIX=bac,PARALLEL=4 -r y -J 0-9 PBS/pipeline_array.pbs
```

It defaults to `chunk` in both scripts, so a single unnamed dataset works
without setting it. Give separate domains separate prefixes and their
inputs and outputs never collide.

**Throttle concurrency with `max_run_subjobs`.** Where arrays *are* wide
enough to need it, PBS Pro has no `%N` suffix; the equivalent is a `-W`
option:

```bash
qsub -r y -J 0-9 -W max_run_subjobs=4 PBS/download_genomes.pbs
```

This matters most for the download array. NCBI rate-limits per IP — 3
requests per second without an API key, 10 with one — so letting hundreds
of subjobs rehydrate at once gets the whole site throttled rather than
speeding anything up. Set `NCBI_API_KEY` from the environment for large
runs, and keep concurrency modest regardless.

Queues also cap queued jobs (`max_queued = 1000` on Gadi), which bounds
how many subjobs can sit in the queue at once across all your arrays.

## Multi-GPU under PBS

GPU discovery checks `SLURM_JOB_GPUS`, then `SLURM_GPUS`, then
`CUDA_VISIBLE_DEVICES`. On Gadi **none of the three is set** inside a GPU
job — verified in a `gpuvolta` allocation, where the log recorded
`CUDA_VISIBLE_DEVICES=unset` while `torch.cuda.is_available()` was true and
the V100 was visible. PBS restricts device visibility by cgroup rather than
by that variable, so discovery falls through to PyTorch's device count,
which already reports exactly the GPUs you were allocated. That is the
behaviour you want; do not set the variable by hand to "fix" it.

If you pass `--gpu-ids` explicitly, use local indices (`0,1`), not physical
device numbers. Request `ngpus=2` with `ncpus=24` on `gpuvolta` and add
`--multi-gpu`.

## Checking a run

`qstat -x <jobid>` shows exit status after a job leaves the queue; an array
reports per-index status with `qstat -xt <jobid>[]`. `pipeline_array.pbs`
exits non-zero when any genome in its chunk failed, so a partially failed
index is visible rather than being mistaken for a clean run. Check exit
statuses before launching the next stage.

A chunk counts as failed if a genome failed to encode **or** if its JSON
could not be read back when the entropy TSV was extracted. The second case
matters: the TSV and archive are still published, because they hold real
encoded genomes and discarding them would waste the GPU time, but the TSV
is missing rows it should have had. Either way the affected accessions are
named in `<out>/<prefix>_NNN.failures`, so a chunk that quietly lost a
genome is not possible.

**Resuming after walltime.** When PBS sends `SIGTERM`, the job packs the
genomes it has finished into `<out>/<prefix>_NNN.tar.zst.partial`. A
resubmitted index unpacks that partial first and encodes only what is
missing, so the GPU time already spent is not repaid. Delete the `.partial`
if you want a clean re-run.

## Inodes are usually the binding constraint

On a shared `/g/data` the inode quota bites long before the byte quota. The
project this was written against had ~520k inodes free against 30 TB of
space, while NCBI `datasets` writes two inodes per genome (an accession
directory and a `genomic.gbff`) and the pipeline writes one JSON per genome
on top. A 200k-genome run wants ~600k inodes and fails on a filesystem with
9 TB free.

`download_genomes.pbs` and `pipeline_array.pbs` therefore keep nothing
per-genome on `/g/data`. Each unpacks its input onto node-local
`$PBS_JOBFS`, which has its own filesystem and costs no `/g/data` inodes,
and writes back a single compressed archive per chunk. Two inodes per
chunk, not two per genome.

Measured on real output: entropy JSON is 2.3–4.9x the size of its GenBank
input, `zstd -3` compresses it about 3.4x at roughly ten times gzip's
speed, and the per-ORF entropy values extracted to TSV are about **42x**
smaller than the JSON they came from. So the templates write both — a JSON
archive for sequences and encodings, and a small TSV for the numbers — and
downstream analysis reads the TSV without ever unpacking an archive.

## Status

The `#PBS` directives, queue constraints, and offline-cache handling
reflect Gadi's documented limits. Verified in a real `gpuvolta` allocation:
the conda environment, the offline `HF_HOME` cache with `HF_HUB_OFFLINE=1`,
`torch.cuda` on a V100, and a full `run` over five genomes producing schema
2.2.0 with all five entropy fields populated.

It was charged **4.41 SU for 7m21s** on one GPU with 12 CPUs, which puts
`gpuvolta` at about **36 SU per GPU-hour**.

### Do not raise `--encoding-size` to "use the GPU"

A single `run` leaves the device looking idle — 12–48% utilisation using
1.5–2.1 GB of the V100's 32 GB — which invites the conclusion that the
batch budget should be much larger. Measured on Gadi, that is wrong.

On three bacterial genomes over two rounds (variation under 1%):

| `--encoding-size` | s/genome | vs default |
|---|---|---|
| **10000** (default) | **119.5** | **1.000 — fastest** |
| 25000 | 120.3 | 0.993 |
| 50000 | 124.0 | 0.964 |
| 100000 | 131.5 | 0.909 |

An earlier single-genome sweep went further: 800000 ran 14% slower than
the default, and 400000 and 800000 returned byte-identical peak memory,
meaning the budget already exceeded the genome's entire protein content.

`encoding_size` is a token budget per batch, and a wider budget groups
sequences of more varied length, every one padded to the longest in its
batch. Past a point the extra padding costs more than the wider batch
saves. The default is already at that point; leave it alone.

The idle device is real, but it is not a batch-width problem.

### Run several genomes per GPU instead

One `run` at a time uses about one core of the 12 that `gpuvolta` bills per
GPU, and leaves the device idle much of the time waiting on CPU-side work:
`get_orfs`, GenBank parsing, JSON writing. Running several concurrently
fills those gaps with other processes' CPU work. Measured on 12 bacterial
genomes, same work at every setting:

| processes | genomes/hr | speedup | GPU mem |
|---|---|---|---|
| 1 | 40 | 1.00× | 3.1 GB |
| 2 | 68 | 1.71× | 4.5 GB |
| **4** | **82** | **2.04×** | **7.8 GB** |
| 6 | 83 | 2.07× | 10.5 GB |
| 12 | 82 | 2.07× | 19.2 GB |

Throughput saturates at **4 processes**, taking GPU utilisation from 12% to
79% — past that the device itself is the bottleneck and more processes only
consume VRAM. Roughly 2x, not the 8x that "4% of memory used" might
suggest, because memory was never the binding resource.

`pipeline_array.pbs` takes `PARALLEL=N` for this, keeping up to `N`
`genome_entropy run` processes in flight against the one GPU. It defaults
to `1`; the measurement above says `4`. Re-measure on your own hardware and
genomes: the right number depends on how the CPU-side work compares to
encoding, which varies with genome size and annotation. Watch VRAM, since
each process holds its own copy of the model.

Wall times, memory figures, and chunk sizes in these templates remain
starting points to measure, not tested defaults.
