# Multi-GPU encoder reuse: historical implementation note

This note describes an earlier fix that moved multi-GPU encoder construction
outside the per-record loop so loaded models could be reused across records.
It is retained as project history, not as an API or operational guide; class
names and internal call paths may change.

For current behaviour, supported models, device discovery, visible-device
variables, precision, token-budget batching, and limitations, use the
[HPC and accelerator guide](docs/source/hpc.rst) and
[model guide](docs/source/models.rst). Current multitask ModernProst models emit
both 3Di and 12-state structural encodings. The pipeline selects the encoder
from the model registry rather than assuming ProstT5.

The user-visible outcome of the historical fix was that `--multi-gpu` could
reuse one encoder per selected device while processing multiple input records,
avoiding repeated model initialisation. Tests in `tests/test_multi_gpu_reuse.py`
and related modules define the current expected behaviour.
