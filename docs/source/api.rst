Python API reference
====================

The CLI is the primary stable interface. Package ``__init__`` modules export the
objects shown below; implementation modules may expose additional semi-public
helpers. Type annotations and generated signatures come from the current code.

ORF discovery
-------------

.. automodule:: genome_entropy.orf.finder
   :members: find_orfs, reverse_complement

.. automodule:: genome_entropy.orf.types
   :members:

Translation
-----------

.. automodule:: genome_entropy.translate.translator
   :members:

Structural-state encoding
-------------------------

.. automodule:: genome_entropy.encode3di.types
   :members:

.. automodule:: genome_entropy.encode3di.encoder
   :members: ProstT5ThreeDiEncoder

.. automodule:: genome_entropy.encode3di.modernprost
   :members: ModernProstThreeDiEncoder

.. automodule:: genome_entropy.encode3di.multi_gpu
   :members: MultiGPUEncoder

.. automodule:: genome_entropy.encode3di.gpu_utils
   :members: discover_available_gpus, select_device_for_gpu, validate_gpu_availability

.. automodule:: genome_entropy.encode3di.token_estimator
   :members:

Entropy
-------

The normalisation helpers are intended for downstream use and are not invoked
by standard serialisation.

.. automodule:: genome_entropy.entropy.shannon
   :members:

Pipeline and schemas
--------------------

.. automodule:: genome_entropy.pipeline.runner
   :members: PipelineResult, run_pipeline, calculate_pipeline_entropy

.. automodule:: genome_entropy.pipeline.types
   :members:

I/O
---

.. automodule:: genome_entropy.io.fasta
   :members:

.. automodule:: genome_entropy.io.genbank
   :members:

.. automodule:: genome_entropy.io.jsonio
   :members:

Machine learning
----------------

Install the ``ml`` extra before importing these modules.

.. automodule:: genome_entropy.ml.classifier
   :members:

.. automodule:: genome_entropy.ml.file_split
   :members:

.. automodule:: genome_entropy.ml.models
   :members:
   :show-inheritance:

Configuration, errors, and logging
----------------------------------

.. automodule:: genome_entropy.config
   :members:

.. automodule:: genome_entropy.errors
   :members:
   :show-inheritance:

.. automodule:: genome_entropy.logging_config
   :members:
