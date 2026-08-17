Quick start
===========

Prerequisites
-------------

Install the package and external ``get_orfs`` executable as described in
:doc:`installation`. The first encoder invocation downloads a model unless it
is already cached.

Complete pipeline
-----------------

.. code-block:: bash

   genome_entropy run --input genome.fasta --output results.json

The default ``gbouras13/modernprost-50M`` model predicts 3Di and 12-state
encodings. The JSON contains raw entropy plus raw 3Di--12-state mutual
information when both encodings are available, and uses unified schema 2.2.0.

GenBank input enables CDS matching:

.. code-block:: bash

   genome_entropy run --genbank genome.gbk.gz --output results.json

Supplying both ``--input`` and ``--genbank`` uses sequences from FASTA and CDS
annotations from GenBank. IDs must correspond for useful matching.

Step-by-step workflow
---------------------

.. code-block:: bash

   genome_entropy orf --input genome.fasta --output orfs.json
   genome_entropy translate --input orfs.json --output proteins.json
   genome_entropy encode3di --input proteins.json --output structures.json
   genome_entropy entropy --input structures.json --output entropy.json

The standalone entropy report cannot reconstruct whole-contig entropy and writes
``dna_entropy_global`` as ``0.0``. Prefer ``run`` when the unified complete
record is required.

Protein-only workflow
---------------------

.. code-block:: bash

   genome_entropy fasta-to-protein --input proteins.faa --output proteins.json
   genome_entropy encode3di --input proteins.faa --output structures.json

``encode3di`` accepts protein FASTA directly, so conversion is only needed when
an intermediate protein-record JSON is useful.

Model and device selection
--------------------------

.. code-block:: bash

   genome_entropy run --input genome.fasta --output results.json \
       --model gbouras13/modernprost-base --device cuda
   genome_entropy encode3di --input proteins.faa --output structures.json \
       --multi-gpu --gpu-ids 0,1 --encoding-size 8000

Use the 50M default for ordinary examples; the larger model needs more memory.
Estimate a safe token budget on the target hardware with ``estimate-tokens``.

Next steps
----------

Read :doc:`models` for encoder details, :doc:`data_formats` for schemas and
normalised entropy, :doc:`cli` for every option, and :doc:`ml` before training
or interpreting a classifier.
