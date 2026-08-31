# AMD ROCm SLURM templates

These Pawsey-oriented scripts assume SLURM, a `rocm/<version>` environment module, project variables such as `PAWSEY_PROJECT`, and a writable shared scratch filesystem. They must be customised for other clusters.

- `install.slurm` creates a virtual environment and attempts to select PyTorch wheels matching the loaded ROCm major/minor release. Verify the discovered wheel channel against current PyTorch guidance before use; the script may delete and recreate its exact configured virtual-environment directory.
- `download.slurm` caches the default `gbouras13/modernprost-50M` model and requires internet unless cached.
- `estimate_tokens.slurm` runs real synthetic inference on the allocated AMD GPUs.
- `pipeline.slurm` runs the complete pipeline in multi-GPU mode for GenBank alone or FASTA plus GenBank.

Mutual-information workflow, in submission order:

- `split_phold_chunks.slurm` splits one large gzipped GenBank file into fixed-size gzipped chunks.
- `run_mutual_information_chunks.slurm` is a job array; each element re-encodes four chunks by calling `rerun_mutual_information.slurm` in-process.
- `rerun_mutual_information.slurm` converts one GenBank chunk to FASTA and re-runs the dual-head 50M model over it.
- `extract_mutual_information.slurm` and `aggregate_mutual_information.slurm` write per-ORF entropy and mutual-information TSV output for one input or for a directory of results.
- `install_plotting_dependencies.slurm` adds Matplotlib and Seaborn to the virtual environment.
- `analyze_in_genbank_variables.slurm` ranks entropy-variable pairs by their association with `in_genbank` and writes the Seaborn figures.
- `rank_missing_orfs.slurm` ranks putative missed or misannotated ORFs with genome-level cross-validated XGBoost.

These scripts locate the repository through `GENOME_ENTROPY_REPO`, falling back to `SLURM_SUBMIT_DIR`. Submit them from a `genome_entropy` checkout, or export `GENOME_ENTROPY_REPO=/path/to/genome_entropy` first. `install.slurm` installs `genome_entropy` from PyPI, so `rank_missing_orfs.py` puts its own checkout's `src/` ahead of that release on `sys.path`; the script and the package code it calls therefore cannot diverge, and the virtual environment does not need reinstalling for each checkout.

The chunking, extraction, aggregation, analysis, and ranking jobs do no GPU work and therefore request the CPU `work` partition under the non-GPU account. Only the encoding jobs reserve GPUs.

PyTorch exposes ROCm accelerators through `torch.cuda`, so `--device cuda` and `cuda:N` are expected. This does not give XGBoost an AMD GPU backend. Use CPU XGBoost unless the local build is independently verified.

Use site-supported AMD monitoring tools such as `amd-smi monitor` or `rocm-smi`; do not use `nvidia-smi`. Review the hard-coded accounts, partitions, ROCm module, virtual-environment location, eight-GPU requests, encoding budget, and wall time before submission. These templates and the `scripts/` helpers they call are site-local analysis workflows, not part of the supported package interface.

See the [HPC guide](../../docs/source/hpc.rst) for general guidance.
