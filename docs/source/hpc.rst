GPU and HPC operation
=====================

Platform semantics
------------------

NVIDIA CUDA and AMD ROCm are separate PyTorch installations. On ROCm, PyTorch
retains the ``torch.cuda`` API and device strings such as ``cuda:0``; this is
expected and does not mean that NVIDIA CUDA is installed. Apple MPS supports a
single-device encoder but not the CUDA-oriented multi-GPU workflow. CPU is the
portable fallback and can be much slower for large encoders.

Automatic selection checks CUDA/ROCm, then MPS, then CPU. Explicit unsupported
devices raise ``DeviceError`` rather than silently moving work. Token-budget
batches constrain the total approximate amino-acid count per inference batch;
use ``estimate-tokens`` on the same model, software stack, and accelerator used
for production.

Scheduler discovery and visibility
----------------------------------

Multi-GPU discovery reads SLURM GPU allocation variables before
``CUDA_VISIBLE_DEVICES`` and finally PyTorch's visible device count. Visible
physical devices may be remapped to local indices, so pass local IDs such as
``--gpu-ids 0,1`` inside the job allocation. Do not request devices outside the
scheduler allocation.

Installation and offline jobs
-----------------------------

#. Install the cluster-supported CUDA or ROCm PyTorch build in an isolated
   environment; do not replace system drivers.
#. Install ``genome_entropy`` and the external ``get_orfs`` executable.
#. On a networked node, run ``genome_entropy download --model MODEL`` with the
   same cache location visible to compute nodes.
#. Estimate a safe batch budget in an interactive allocation.
#. Submit the production command with site-specific account, partition, module,
   environment, cache, memory, wall-time, and GPU requests.

The files in ``slurm/nvidia`` and ``slurm/rocm`` are Pawsey-oriented examples,
not portable cluster profiles. Review every ``#SBATCH`` directive and path.
Model download jobs require outbound internet unless the cache is already
populated. ``get_orfs`` compilation requires Rust/Cargo according to that
project's installation instructions.

Monitoring
----------

Use ``nvidia-smi`` only on NVIDIA systems. On supported AMD systems use the
site-provided ROCm tooling, commonly ``amd-smi monitor`` or ``rocm-smi``;
available commands and flags depend on the installed ROCm release. Scheduler
commands such as ``squeue`` and ``sacct`` remain site dependent.

ML caveat
---------

These accelerator statements describe PyTorch/ModernProst inference. XGBoost
has its own compiled GPU backend. In particular, a ROCm-capable PyTorch
installation does not guarantee AMD GPU acceleration for XGBoost; see :doc:`ml`.
