CLI Commands Reference
======================

The **genome_entropy** command-line interface provides modular commands for each step of the pipeline, plus a unified ``run`` command to execute the entire workflow.

Global Options
--------------

All commands support these global options:

.. code-block:: bash

   genome_entropy [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]

**Global Options:**

``--version, -v``
   Show version and exit

``--log-level, -l LEVEL``
   Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   
   Default: INFO

``--log-file PATH``
   Write logs to file instead of STDOUT

**Example:**

.. code-block:: bash

   genome_entropy --log-level DEBUG --log-file debug.log run --input data.fasta --output results.json

Commands
--------

run
^^^

Run the complete pipeline from DNA to 3Di with entropy analysis.

**Usage:**

.. code-block:: bash

   genome_entropy run [OPTIONS]

**Required Options:**

``--input, -i PATH``
   Input FASTA file with DNA sequences. Required if ``--genbank`` is not provided.

``--output, -o PATH``
   Output JSON file for results

**Optional Options:**

``--genbank, -g PATH``
   GenBank file with DNA sequences and CDS annotations. Can be used instead of
   ``--input``, or together with ``--input`` to use FASTA sequences with GenBank
   annotations.

``--table, -t INTEGER``
   NCBI genetic code table ID
   
   Default: 11 (bacterial/archaeal)

``--min-aa INTEGER``
   Minimum protein length in amino acids
   
   Default: 30

``--model, -m TEXT``
   3Di model name from HuggingFace
   
   Default: gbouras13/modernprost-50M

   Supported models include the multitask ``gbouras13/modernprost-50M`` and
   ``gbouras13/modernprost-base`` models, both deprecated ModernProst repositories,
   and both ProstT5 repositories.

``--device, -d TEXT``
   Device for inference (auto, cuda, mps, cpu). Ignored when ``--multi-gpu`` is set.
   
   Default: auto

``--encoding-size INTEGER``
   Total sequence length per encoding batch (in amino acids)
   
   Default: 10000

``--multi-gpu``
   Use multi-GPU parallel encoding when available.

``--gpu-ids TEXT``
   Comma-separated GPU IDs to use with ``--multi-gpu`` (for example, ``0,1,2``).
   If omitted, available GPUs are auto-discovered.

``--skip-entropy``
   Skip entropy calculation

**Examples:**

.. code-block:: bash

   # Basic usage with defaults
   genome_entropy run --input genome.fasta --output results.json

   # Use GPU and custom parameters
   genome_entropy run \
       --input genome.fasta \
       --output results.json \
       --table 1 \
       --min-aa 50 \
       --device cuda

   # Skip entropy for faster processing
   genome_entropy run --input genome.fasta --output results.json --skip-entropy

   # Use GenBank annotations and multi-GPU encoding
   genome_entropy run \
       --genbank annotations.gbk \
       --output results.json \
       --multi-gpu \
       --gpu-ids 0,1

orf
^^^

Extract Open Reading Frames from DNA sequences.

**Usage:**

.. code-block:: bash

   genome_entropy orf [OPTIONS]

**Required Options:**

``--input, -i PATH``
   Input FASTA file with DNA sequences

``--output, -o PATH``
   Output JSON file with ORF records

**Optional Options:**

``--table, -t INTEGER``
   NCBI genetic code table ID
   
   Default: 11

``--min-nt INTEGER``
   Minimum ORF length in nucleotides
   
   Default: 90 (30 amino acids)

**Examples:**

.. code-block:: bash

   # Find ORFs with default settings
   genome_entropy orf --input genome.fasta --output orfs.json

   # Use standard genetic code and longer minimum length
   genome_entropy orf \
       --input genome.fasta \
       --output orfs.json \
       --table 1 \
       --min-nt 150

translate
^^^^^^^^^

Translate ORFs to protein sequences.

**Usage:**

.. code-block:: bash

   genome_entropy translate [OPTIONS]

**Required Options:**

``--input, -i PATH``
   Input JSON file with ORF records

``--output, -o PATH``
   Output JSON file with protein records

**Optional Options:**

``--table, -t INTEGER``
   NCBI genetic code table ID
   
   Default: 11

**Examples:**

.. code-block:: bash

   # Translate ORFs
   genome_entropy translate --input orfs.json --output proteins.json

   # Use different genetic code
   genome_entropy translate \
       --input orfs.json \
       --output proteins.json \
       --table 4

encode3di
^^^^^^^^^

Encode proteins into structural-state sequences. New ModernProst models produce
both 3Di and 12-state encodings; legacy models produce only 3Di.

**Usage:**

.. code-block:: bash

   genome_entropy encode3di [OPTIONS]

**Required Options:**

``--input, -i PATH``
   Input protein file. FASTA files (``.fasta``, ``.fa``, ``.faa``) are read as
   amino acid sequences; other inputs are treated as protein JSON records from
   ``translate`` or ``fasta-to-protein``.

``--output, -o PATH``
   Output JSON file with 3Di records

**Optional Options:**

``--model, -m TEXT``
   3Di model name
   
   Default: gbouras13/modernprost-50M

   Supported models include ``gbouras13/modernprost-50M``,
   ``gbouras13/modernprost-base``, the two ``-deprecated`` ModernProst repositories,
   ``Rostlab/ProstT5``, and ``Rostlab/ProstT5_fp16``.

``--device, -d TEXT``
   Device for inference (auto, cuda, mps, cpu). Ignored when ``--multi-gpu`` is set.
   
   Default: auto

``--encoding-size INTEGER``
   Total amino acids per encoding batch
   
   Default: 10000

``--multi-gpu``
   Use multi-GPU parallel encoding when available.

``--gpu-ids TEXT``
   Comma-separated GPU IDs to use with ``--multi-gpu`` (for example, ``0,1,2``).
   If omitted, available GPUs are auto-discovered.

**Examples:**

.. code-block:: bash

   # Basic encoding
   genome_entropy encode3di --input proteins.json --output 3di.json

   # Use GPU with larger batches
   genome_entropy encode3di \
       --input proteins.json \
       --output 3di.json \
       --device cuda \
       --encoding-size 10000

   # Encode a protein FASTA directly
   genome_entropy encode3di --input proteins.faa --output 3di.json

   # Encode on multiple GPUs
   genome_entropy encode3di \
       --input proteins.json \
       --output 3di.json \
       --multi-gpu \
       --gpu-ids 0,1

   # Force CPU usage
   genome_entropy encode3di \
       --input proteins.json \
       --output 3di.json \
       --device cpu

fasta-to-protein
^^^^^^^^^^^^^^^^

Convert protein FASTA input to the protein JSON format used by ``encode3di``.

This is useful when you already have amino acid sequences and want to bypass ORF
finding and translation. Because the proteins are not derived from ORFs, the
command creates minimal placeholder ORF metadata for compatibility with the
pipeline JSON schema.

**Usage:**

.. code-block:: bash

   genome_entropy fasta-to-protein [OPTIONS]

**Required Options:**

``--input, -i PATH``
   Input protein FASTA file

``--output, -o PATH``
   Output JSON file with protein records

**Examples:**

.. code-block:: bash

   genome_entropy fasta-to-protein --input proteins.faa --output proteins.json

entropy
^^^^^^^

Calculate Shannon entropy at all representation levels.

**Usage:**

.. code-block:: bash

   genome_entropy entropy [OPTIONS]

**Required Options:**

``--input, -i PATH``
   Input JSON file with 3Di records

``--output, -o PATH``
   Output JSON file with entropy report

**Examples:**

.. code-block:: bash

   # Calculate entropy
   genome_entropy entropy --input 3di.json --output entropy.json

The command writes raw Shannon entropy. Use the normalisation helpers described
in the user guide during downstream analysis; normalised entropy is not written
to standard JSON.

download
^^^^^^^^

Pre-download ModernProst or ProstT5 models to the HuggingFace cache.

**Usage:**

.. code-block:: bash

   genome_entropy download [OPTIONS]

**Optional Options:**

``--model, -m TEXT``
   Model name to download
   
   Default: gbouras13/modernprost-50M

``--test-data``
   Request test datasets. This option is currently a placeholder; use
   ``examples/example_small.fasta`` for local testing.

**Examples:**

.. code-block:: bash

   # Download default model
   genome_entropy download

   # Download specific model
   genome_entropy download --model Rostlab/ProstT5

estimate-tokens
^^^^^^^^^^^^^^^

Estimate optimal encoding size for your GPU.

**Usage:**

.. code-block:: bash

   genome_entropy estimate-tokens [OPTIONS]

**Optional Options:**

``--device, -d TEXT``
   Device to test (auto, cuda, mps, cpu)
   
   Default: auto

``--model, -m TEXT``
   3Di model name
   
   Default: gbouras13/modernprost-50M

``--start INTEGER``
   Starting encoding size to test
   
   Default: 3000

``--end INTEGER``
   Ending encoding size to test
   
   Default: 10000

``--step INTEGER``
   Step size for testing
   
   Default: 1000

``--trials INTEGER``
   Number of trials per size
   
   Default: 3

``--base-length, -b INTEGER``
   Approximate length of generated individual proteins in amino acids

   Default: 100

``--log-level, -l LEVEL``
   Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

   Default: INFO

``--log-file PATH``
   Path to log file

**Examples:**

.. code-block:: bash

   # Basic estimation
   genome_entropy estimate-tokens

   # Custom range for powerful GPU
   genome_entropy estimate-tokens \
       --device cuda \
       --start 5000 \
       --end 20000 \
       --step 2000

   # Test CPU limits
   genome_entropy estimate-tokens --device cpu

ml
^^

Train and use machine learning classifiers that predict whether ORFs correspond
to GenBank CDS annotations. The classifier consumes JSON files produced by the
pipeline, extracts tabular ORF-level features such as entropy values, lengths,
position, strand, frame, and start/stop codon flags, then trains either an
XGBoost model or a PyTorch neural network.

The ML dependencies are optional. Install them with:

.. code-block:: bash

   pip install "genome_entropy[ml]"

ml train
""""""""

Train a classifier and save the model.

**Usage:**

.. code-block:: bash

   genome_entropy ml train [OPTIONS]

**Required Options:**

Exactly one input source is required:

``--json PATH``
   One JSON file containing multiple pipeline records. Records are split between
   training and test sets while all ORFs from a record remain together.

``--json-dir, -i PATH``
   Directory containing JSON output files from ``genome_entropy run``. Uses all
   files with random sample-level validation and test splits.

``--split-dir PATH``
   Directory containing JSON output files to split 80/20 by file into training
   and held-out test sets.

``--output, -o PATH``
   Path where the trained model will be saved.

**Optional Options:**

``--model-type, -m TEXT``
   Model type: ``xgboost`` or ``neural_net``

   Default: xgboost

``--device, -d TEXT``
   Training device: ``cuda``, ``cpu``, or auto-detect when omitted.

``--validation-split, -v FLOAT``
   Fraction of data used for validation during training.

   Default: 0.2

``--test-split, -t FLOAT``
   Fraction held out for final testing. In ``--json`` mode this is the fraction
   of top-level records; in ``--json-dir`` mode it is the fraction of ORFs.
   Ignored when ``--split-dir`` is used.

   Default: 0.1

``--json-output PATH``
   Save a detailed JSON report in ``--split-dir`` mode, including file lists and
   test metrics.

``--random-seed INTEGER``
   Random seed for reproducible file splits.

   Default: 42

**Examples:**

.. code-block:: bash

   # Train the recommended XGBoost classifier
   genome_entropy ml train --json-dir results/ --output model.ubj

   # Train from one multi-record JSON without record-level leakage
   genome_entropy ml train --json results.json --output model.ubj

   # File-based train/test split with detailed reporting
   genome_entropy ml train \
       --split-dir results/ \
       --output model.ubj \
       --json-output detailed_results.json

   # Train a neural network on CUDA
   genome_entropy ml train \
       --json-dir results/ \
       --output model.pt \
       --model-type neural_net \
       --device cuda

ml predict
""""""""""

Load a trained classifier and write per-ORF predictions.

**Usage:**

.. code-block:: bash

   genome_entropy ml predict [OPTIONS]

**Required Options:**

Exactly one input source is required:

``--json PATH``
   One JSON file containing one or more pipeline records.

``--json-dir, -i PATH``
   Directory containing JSON files to predict on.

``--model, -m PATH``
   Path to a trained model file.

``--output, -o PATH``
   Path to save predictions in TSV format.

**Optional Options:**

``--model-type, -t TEXT``
   Model type: ``xgboost`` or ``neural_net``

   Default: xgboost

**Output Columns:**

``input_id``
   Identifier of the top-level input record.

``orf_id``
   ORF identifier.

``predicted_label``
   Predicted class, where ``1`` means in GenBank and ``0`` means not in GenBank.

``prob_not_in_genbank`` / ``prob_in_genbank``
   Class probabilities from the model.

``in_genbank``
   Actual label when present in the input metadata, otherwise ``NA``.

**Examples:**

.. code-block:: bash

   genome_entropy ml predict \
       --json-dir new_results/ \
       --model model.ubj \
       --output predictions.tsv

   genome_entropy ml predict \
       --json results.json \
       --model model.ubj \
       --output predictions.tsv

Common Workflows
----------------

Standard Analysis
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Complete pipeline with logging
   genome_entropy --log-file analysis.log run \
       --input genome.fasta \
       --output results.json \
       --table 11 \
       --min-aa 30 \
       --device auto

Step-by-Step Analysis
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Step 1: Find ORFs
   genome_entropy orf --input genome.fasta --output orfs.json --table 11

   # Step 2: Translate
   genome_entropy translate --input orfs.json --output proteins.json --table 11

   # Step 3: Encode to 3Di
   genome_entropy encode3di \
       --input proteins.json \
       --output 3di.json \
       --device cuda

   # Step 4: Calculate entropy
   genome_entropy entropy --input 3di.json --output entropy.json

Optimizing Performance
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # First, find optimal encoding size
   genome_entropy estimate-tokens --device cuda

   # Then use it in the pipeline
   genome_entropy run \
       --input genome.fasta \
       --output results.json \
       --device cuda \
       --encoding-size 15000  # Use recommended value from estimate-tokens

Protein FASTA to 3Di
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Convert first, then encode
   genome_entropy fasta-to-protein --input proteins.faa --output proteins.json
   genome_entropy encode3di --input proteins.json --output 3di.json

   # Or encode the FASTA directly
   genome_entropy encode3di --input proteins.faa --output 3di.json

Training GenBank Annotation Classifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Store annotated pipeline output in the directory used for training.
   mkdir -p results
   genome_entropy run --genbank genome.gbk --output results/annotated_results.json
   genome_entropy ml train --json-dir results/ --output genbank_classifier.ubj
   genome_entropy ml predict \
       --json-dir new_results/ \
       --model genbank_classifier.ubj \
       --output predictions.tsv

Exit Codes
----------

The CLI uses standard exit codes:

* **0**: Success
* **1**: General error
* **2**: User error (bad arguments, missing file)
* **3**: Runtime error (model failure, GPU error)

Examples:

.. code-block:: bash

   # Check exit code
   genome_entropy run --input genome.fasta --output results.json
   echo $?  # Should print 0 on success

Genetic Code Tables
-------------------

Common NCBI genetic code tables:

+--------+-----------------------------------------------+
| Table  | Description                                   |
+========+===============================================+
| 1      | Standard genetic code                         |
+--------+-----------------------------------------------+
| 11     | Bacterial, archaeal, plant plastid (default)  |
+--------+-----------------------------------------------+
| 4      | Mold, protozoan, coelenterate mitochondrial   |
+--------+-----------------------------------------------+
| 2      | Vertebrate mitochondrial                      |
+--------+-----------------------------------------------+
| 5      | Invertebrate mitochondrial                    |
+--------+-----------------------------------------------+

See complete list: https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi

Environment Variables
---------------------

``GET_ORFS_PATH``
   Path to get_orfs binary if not in PATH
   
   Example: ``export GET_ORFS_PATH=/usr/local/bin/get_orfs``

``TRANSFORMERS_CACHE``
   HuggingFace cache directory for models
   
   Default: ``~/.cache/huggingface/``

``CUDA_VISIBLE_DEVICES``
   Select specific GPU(s)
   
   Example: ``export CUDA_VISIBLE_DEVICES=0``

Next Steps
----------

* Read the :doc:`user_guide` for detailed pipeline documentation
* See :doc:`api` for Python API usage
* Learn about :doc:`token_estimation` for performance optimization
