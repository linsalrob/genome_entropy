# SLURM examples

The `nvidia/` and `rocm/` directories contain site-specific starting points for
CUDA/NVIDIA and ROCm/AMD clusters. They are examples, not portable job scripts.
Before submitting one, adjust its account, partition, modules, environment
path, model cache, input/output paths, wall time, memory, and GPU request to the
target site's policies.

PyTorch exposes ROCm accelerators through its `torch.cuda` API, so the
application device remains `cuda` on an AMD PyTorch build. This does not imply
that XGBoost can train on AMD GPUs; the CLI's XGBoost GPU mode requires a
CUDA-capable XGBoost installation. See the
[HPC guide](../docs/source/hpc.rst) and
[installation guide](../docs/source/installation.rst) before use.
