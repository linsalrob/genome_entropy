# AMD ROCm SLURM templates

These Pawsey-oriented scripts assume SLURM, a `rocm/<version>` environment module, project variables such as `PAWSEY_PROJECT`, and a writable shared scratch filesystem. They must be customised for other clusters.

- `install.slurm` creates a virtual environment and attempts to select PyTorch wheels matching the loaded ROCm major/minor release. Verify the discovered wheel channel against current PyTorch guidance before use; the script may delete and recreate its exact configured virtual-environment directory.
- `download.slurm` caches the default `gbouras13/modernprost-50M` model and requires internet unless cached.
- `estimate_tokens.slurm` runs real synthetic inference on the allocated AMD GPUs.
- `pipeline.slurm` runs the complete pipeline in multi-GPU mode for GenBank alone or FASTA plus GenBank.

PyTorch exposes ROCm accelerators through `torch.cuda`, so `--device cuda` and `cuda:N` are expected. This does not give XGBoost an AMD GPU backend. Use CPU XGBoost unless the local build is independently verified.

Use site-supported AMD monitoring tools such as `amd-smi monitor` or `rocm-smi`; do not use `nvidia-smi`. Review the hard-coded account, partition, ROCm module, virtual-environment location, eight-GPU requests, encoding budget, and wall time before submission. The untracked `rank_missing_orfs.slurm` workflow is site-local and is not part of the supported package interface.

See the [HPC guide](../../docs/source/hpc.rst) for general guidance.
