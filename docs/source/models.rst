Models and structural-state encodings
=====================================

Authoritative model registry
----------------------------

The following table mirrors ``genome_entropy.config.MODEL_REGISTRY`` and was
checked against the live Hugging Face repositories on 29 July 2026.

.. list-table::
   :header-rows: 1
   :widths: 32 14 10 10 16

   * - Canonical identifier
     - Size
     - 3Di
     - 12-state
     - Status
   * - ``gbouras13/modernprost-50M``
     - 52,555,936 parameters
     - yes
     - yes
     - default
   * - ``gbouras13/modernprost-base``
     - approximately 1B
     - yes
     - yes
     - supported
   * - ``gbouras13/modernprost-base-deprecated``
     - legacy
     - yes
     - no
     - deprecated
   * - ``gbouras13/modernprost-profiles-deprecated``
     - legacy
     - yes
     - no
     - deprecated
   * - ``Rostlab/ProstT5``
     - approximately 3B
     - yes
     - no
     - supported
   * - ``Rostlab/ProstT5_fp16``
     - approximately 3B
     - yes
     - no
     - supported, fp16

The old identifier ``gbouras13/modernprost-profiles`` is accepted as an alias
for ``gbouras13/modernprost-profiles-deprecated`` and emits a
``FutureWarning``. ``gbouras13/modernprost-base`` is *not* a legacy identifier:
it is the current larger multitask model. The current Hugging Face configurations
identify both multitask repositories as ``modernprost_3di12st``.

Output semantics
----------------

3Di is Foldseek's 20-state representation of local tertiary interactions.
The additional 12-state head predicts 12 structural classes. In JSON, 12-state
class IDs 0--11 are deterministically serialised as characters ``A``--``L``;
these letters are storage symbols, not a published biological nomenclature.
For dual-head output, genome_entropy also stores raw per-ORF mutual information
in bits between the aligned discrete 3Di and 12-state sequences.

For legacy models, structural records and unified pipeline output contain:

.. code-block:: json

   {
     "twelve_state": null,
     "twelve_state_entropy": null,
     "three_di_twelve_state_mutual_information": null
   }

Older JSON in which these fields are absent remains readable and is treated as
missing data. A missing 12-state value is not the same as an empty encoding or
zero entropy.

Loading, security, and cache behaviour
--------------------------------------

ModernProst uses Hugging Face ``AutoConfig``, ``AutoTokenizer``, and
``AutoModel`` with ``trust_remote_code=True``. This executes Python code from
the selected model repository. Use only trusted repositories; for controlled or
reproducible environments, review and pin a model revision before deployment.
The application currently selects canonical repository names but does not expose
a CLI revision-pin option.

``genome_entropy download`` populates the normal Hugging Face cache and is useful
on a networked login node before an offline batch job. ModernProst loading uses
``local_files_only=True`` when the repository is already cached. Model downloads
require internet access and can consume substantial storage.

Devices, precision, and multi-GPU
---------------------------------

``--device`` accepts ``cpu``, ``cuda``, or ``mps``; omission performs automatic
detection in that order. On ROCm installations, PyTorch intentionally exposes
AMD accelerators through ``torch.cuda`` and the CLI value remains ``cuda``.
``Rostlab/ProstT5_fp16`` is loaded in half precision. Other precision behaviour
follows the model checkpoint and encoder implementation.

``--multi-gpu`` creates one encoder per selected visible device and distributes
token-budget batches across them. ``--gpu-ids`` supplies comma-separated local
device indices. Discovery considers SLURM GPU variables, then
``CUDA_VISIBLE_DEVICES``, then ``torch.cuda.device_count()``. Multi-GPU mode is
for CUDA/ROCm devices, not Apple MPS.

Citations and provenance
------------------------

ModernProst repositories and integration were contributed by George Bouras:

* `George Bouras's Hugging Face namespace <https://huggingface.co/gbouras13>`_
* `ModernProst 50M <https://huggingface.co/gbouras13/modernprost-50M>`_
* `ModernProst base <https://huggingface.co/gbouras13/modernprost-base>`_

For ProstT5, cite Heinzinger *et al.*, "Bilingual language model for protein
sequence and structure", *NAR Genomics and Bioinformatics* (2024),
doi:10.1093/nargab/lqae150. For 3Di/Foldseek, cite van Kempen *et al.*, "Fast
and accurate protein structure search with Foldseek", *Nature Biotechnology*
(2024), doi:10.1038/s41587-023-01773-0.
