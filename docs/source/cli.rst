Command-line reference
======================

The installed CLI help is authoritative for its version. Inspect it with:

.. code-block:: bash

   genome_entropy --help
   genome_entropy COMMAND --help
   genome_entropy ml train --help

Global options
--------------

``--version, -v`` prints the package version. ``--log-level, -l`` accepts
``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, or ``CRITICAL`` (default ``INFO``).
``--log-file PATH`` redirects logs from standard output. ``run`` and
``estimate-tokens`` also expose command-local logging options because they can
be invoked independently of the global callback.

``run``
-------

Run DNA/GenBank input through ORF discovery, translation, structural-state
encoding, entropy, and optional CDS matching.

Required: ``--output, -o PATH`` and at least one of ``--input, -i FASTA`` or
``--genbank, -g GENBANK``.

Options: ``--table, -t`` (11), ``--min-aa`` (30), ``--model, -m`` (the 50M
default), ``--device, -d``, ``--skip-entropy``, ``--multi-gpu``, ``--gpu-ids``,
``--encoding-size, -e`` (10000), and logging options. When both input sources
are supplied, FASTA provides sequences and GenBank provides annotations.
``--device`` is ignored in multi-GPU mode.

.. code-block:: bash

   genome_entropy run --input genome.fasta --output results.json
   genome_entropy run --genbank genome.gbk.gz --output results.json \
       --model gbouras13/modernprost-50M

Errors include a missing input source, invalid log level or GPU ID list, input
I/O errors, absent ``get_orfs``, model/device failures, and encoding failures.
User-input validation normally exits 2; runtime pipeline failures exit 3.

``orf``
-------

Required: ``--input, -i FASTA`` and ``--output, -o JSON``. Options are
``--table, -t`` (11) and ``--min-nt`` (90). Output is a list of ORF records.
The command requires the external ``get_orfs`` executable and returns exit 3 on
processing failure.

``translate``
-------------

Required: ``--input, -i ORFS.json`` and ``--output, -o PROTEINS.json``. The
optional ``--table, -t`` defaults to 11 and overrides the translation table.
Input must be the list format written by ``orf``.

``fasta-to-protein``
--------------------

Required: ``--input, -i PROTEINS.faa`` and ``--output, -o PROTEINS.json``.
It creates compatibility ORF metadata because no genomic location exists. The
result can be passed to ``encode3di``.

``encode3di``
--------------

Despite its historical name, this command writes structural-state records:
current ModernProst models include 3Di and 12-state; legacy models include 3Di
and ``twelve_state: null``.

Required: ``--input, -i`` (protein FASTA or protein-record JSON) and
``--output, -o JSON``. Options: ``--model, -m``, ``--device, -d``,
``--encoding-size, -e`` (10000), ``--multi-gpu``, and ``--gpu-ids``.
Accepted FASTA suffixes are ``.fasta``, ``.fa``, and ``.faa``; JSON uses
``.json``. Unknown extensions and malformed record layouts fail explicitly.

.. code-block:: bash

   genome_entropy encode3di --input proteins.faa --output structures.json
   genome_entropy encode3di --input proteins.json --output structures.json \
       --multi-gpu --gpu-ids 0,1

``entropy``
-----------

Required: ``--input, -i STRUCTURES.json`` and ``--output, -o ENTROPY.json``.
It writes raw entropy for ORF nucleotide, protein, 3Di, and available 12-state
sequences. There is no ``--normalize`` option and no normalised JSON field. See
:doc:`data_formats` for downstream helpers. Whole-contig DNA is unavailable in
this intermediate input, so ``dna_entropy_global`` is ``0.0``.

``download``
------------

``--model, -m`` defaults to ``gbouras13/modernprost-50M``. ``--test-data`` is
accepted but currently prints that test-data download is not implemented; it
does not create the example path mentioned by older help text. Model download
requires internet access and executes the model-specific tokenizer/configuration
loading described in :doc:`models`.

``estimate-tokens``
-------------------

Generate random proteins, test increasing total lengths, and report 90% of the
largest successful length as a safety recommendation.

Options: ``--model, -m``; ``--device, -d``; ``--start, -s`` (3000);
``--end, -e`` (10000); ``--step`` (1000); ``--trials, -t`` (3);
``--base-length, -b`` (100); and logging options. This is a model-download and
inference benchmark, not a cheap static estimator. Use an accelerator allocation
matching production. See :doc:`token_estimation`.

``ml train``
------------

Exactly one input is required: ``--json PATH``, ``--json-dir, -i DIR``, or
``--split-dir DIR``. ``--output, -o`` is required. Options are
``--model-type, -m`` (``xgboost``), ``--device, -d``,
``--validation-split, -v`` (0.2), ``--test-split, -t`` (0.1),
``--json-output`` (split-directory mode), and ``--random-seed`` (42).

The three splitting modes, persistence files, reproducibility limits, metrics,
and accelerator caveats are documented in :doc:`ml`. Invalid model type,
fractions, paths, or combinations exit 1.

``ml predict``
--------------

Exactly one of ``--json`` and ``--json-dir, -i`` is required, along with
``--model, -m`` and ``--output, -o``. ``--model-type, -t`` defaults to
``xgboost`` and must match the saved model. Output is TSV, not CSV or JSON; see
:doc:`ml` for the six columns and probability semantics.

Exit behaviour
--------------

Typer handles usage and path-existence errors before command execution. Most
standalone bioinformatics commands map import/input problems to exit 2 and
runtime processing failures to exit 3. ML commands currently use exit 1 for
validation, loading, training, and prediction failures. Error-code handling is
not completely uniform, so automation should also capture stderr/log output.

Environment variables
---------------------

``GET_ORFS_PATH`` selects the ORF binary. GPU discovery observes SLURM allocation
variables and ``CUDA_VISIBLE_DEVICES``. Hugging Face and PyTorch honour their own
documented cache/offline variables; ``genome_entropy`` does not redefine them.
