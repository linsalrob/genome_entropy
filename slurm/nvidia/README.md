# NVIDIA SLURM templates

These scripts are examples for a SLURM cluster with NVIDIA GPUs, Conda, and a `genome_entropy` environment. They are not portable defaults. Before submission, customise the partition, account (if required), wall time, memory, GPU count, environment initialisation, model cache, and input/output paths.

- `pip_install.slurm` installs the repository with development, documentation, and ML extras. Confirm the site's recommended CUDA-enabled PyTorch installation first.
- `download.slurm` caches the default `gbouras13/modernprost-50M` model and requires outbound internet unless already cached.
- `estimate_tokens.slurm` benchmarks a synthetic encoding budget on allocated hardware.
- `pipeline.slurm` runs GenBank input with the 50M model.
- `encoder.slurm` accepts protein JSON and writes structural-state JSON; multitask models include 3Di and 12-state.
- `pytest.slurm` runs the test suite in a GPU allocation; ordinary unit tests do not require a GPU.

The example requests two GPUs in several scripts even when the command is single-device. Reduce resource requests or add `--multi-gpu` as appropriate. Quote all user-supplied paths when adapting the templates. Monitor NVIDIA devices with the site-supported `nvidia-smi` command.

See the [HPC guide](../../docs/source/hpc.rst) for cache, visibility, multi-GPU, and safety details.
