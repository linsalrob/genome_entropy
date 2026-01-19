CLI Commands Reference
======================

The **dna23di** command-line interface provides modular commands for each step of the pipeline, plus a unified ``run`` command to execute the entire workflow.

Global Options
--------------

All commands support these global options:

.. code-block:: bash

   dna23di [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]

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

   dna23di --log-level DEBUG --log-file debug.log run --input data.fasta --output results.json

Commands
--------

run
^^^

Run the complete pipeline from DNA to 3Di with entropy analysis.

**Usage:**

.. code-block:: bash

   dna23di run [OPTIONS]

**Required Options:**

``--input, -i PATH``
   Input FASTA file with DNA sequences

``--output, -o PATH``
   Output JSON file for results

**Optional Options:**

``--table, -t INTEGER``
   NCBI genetic code table ID
   
   Default: 11 (bacterial/archaeal)

``--min-aa INTEGER``
   Minimum protein length in amino acids
   
   Default: 30

``--model, -m TEXT``
   ProstT5 model name from HuggingFace
   
   Default: Rostlab/ProstT5_fp16

``--device, -d TEXT``
   Device for inference (auto, cuda, mps, cpu)
   
   Default: auto

``--batch-size INTEGER``
   Batch size for encoding
   
   Default: 4

``--encoding-size INTEGER``
   Total sequence length per encoding batch (in amino acids)
   
   Default: 5000

``--skip-entropy``
   Skip entropy calculation

**Examples:**

.. code-block:: bash

   # Basic usage with defaults
   dna23di run --input genome.fasta --output results.json

   # Use GPU and custom parameters
   dna23di run \
       --input genome.fasta \
       --output results.json \
       --table 1 \
       --min-aa 50 \
       --device cuda \
       --batch-size 8

   # Skip entropy for faster processing
   dna23di run --input genome.fasta --output results.json --skip-entropy

orf
^^^

Extract Open Reading Frames from DNA sequences.

**Usage:**

.. code-block:: bash

   dna23di orf [OPTIONS]

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
   dna23di orf --input genome.fasta --output orfs.json

   # Use standard genetic code and longer minimum length
   dna23di orf \
       --input genome.fasta \
       --output orfs.json \
       --table 1 \
       --min-nt 150

translate
^^^^^^^^^

Translate ORFs to protein sequences.

**Usage:**

.. code-block:: bash

   dna23di translate [OPTIONS]

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
   dna23di translate --input orfs.json --output proteins.json

   # Use different genetic code
   dna23di translate \
       --input orfs.json \
       --output proteins.json \
       --table 4

encode3di
^^^^^^^^^

Encode protein sequences to 3Di structural tokens using ProstT5.

**Usage:**

.. code-block:: bash

   dna23di encode3di [OPTIONS]

**Required Options:**

``--input, -i PATH``
   Input JSON file with protein records

``--output, -o PATH``
   Output JSON file with 3Di records

**Optional Options:**

``--model, -m TEXT``
   ProstT5 model name
   
   Default: Rostlab/ProstT5_fp16

``--device, -d TEXT``
   Device for inference (auto, cuda, mps, cpu)
   
   Default: auto

``--batch-size INTEGER``
   Number of sequences per batch
   
   Default: 4

``--encoding-size INTEGER``
   Total amino acids per encoding batch
   
   Default: 5000

**Examples:**

.. code-block:: bash

   # Basic encoding
   dna23di encode3di --input proteins.json --output 3di.json

   # Use GPU with larger batches
   dna23di encode3di \
       --input proteins.json \
       --output 3di.json \
       --device cuda \
       --batch-size 8 \
       --encoding-size 10000

   # Force CPU usage
   dna23di encode3di \
       --input proteins.json \
       --output 3di.json \
       --device cpu

entropy
^^^^^^^

Calculate Shannon entropy at all representation levels.

**Usage:**

.. code-block:: bash

   dna23di entropy [OPTIONS]

**Required Options:**

``--input, -i PATH``
   Input JSON file with 3Di records

``--output, -o PATH``
   Output JSON file with entropy report

**Optional Options:**

``--normalize``
   Normalize entropy by alphabet size (scale to [0, 1])

**Examples:**

.. code-block:: bash

   # Calculate entropy
   dna23di entropy --input 3di.json --output entropy.json

   # Calculate normalized entropy
   dna23di entropy \
       --input 3di.json \
       --output entropy.json \
       --normalize

download
^^^^^^^^

Pre-download ProstT5 models to cache.

**Usage:**

.. code-block:: bash

   dna23di download [OPTIONS]

**Optional Options:**

``--model, -m TEXT``
   Model name to download
   
   Default: Rostlab/ProstT5_fp16

**Examples:**

.. code-block:: bash

   # Download default model
   dna23di download

   # Download specific model
   dna23di download --model Rostlab/ProstT5

estimate-tokens
^^^^^^^^^^^^^^^

Estimate optimal encoding size for your GPU.

**Usage:**

.. code-block:: bash

   dna23di estimate-tokens [OPTIONS]

**Optional Options:**

``--device, -d TEXT``
   Device to test (auto, cuda, mps, cpu)
   
   Default: auto

``--model, -m TEXT``
   ProstT5 model name
   
   Default: Rostlab/ProstT5_fp16

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

**Examples:**

.. code-block:: bash

   # Basic estimation
   dna23di estimate-tokens

   # Custom range for powerful GPU
   dna23di estimate-tokens \
       --device cuda \
       --start 5000 \
       --end 20000 \
       --step 2000

   # Test CPU limits
   dna23di estimate-tokens --device cpu

Common Workflows
----------------

Standard Analysis
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Complete pipeline with logging
   dna23di --log-file analysis.log run \
       --input genome.fasta \
       --output results.json \
       --table 11 \
       --min-aa 30 \
       --device auto

Step-by-Step Analysis
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Step 1: Find ORFs
   dna23di orf --input genome.fasta --output orfs.json --table 11

   # Step 2: Translate
   dna23di translate --input orfs.json --output proteins.json --table 11

   # Step 3: Encode to 3Di
   dna23di encode3di \
       --input proteins.json \
       --output 3di.json \
       --device cuda \
       --batch-size 8

   # Step 4: Calculate entropy
   dna23di entropy --input 3di.json --output entropy.json --normalize

Optimizing Performance
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # First, find optimal encoding size
   dna23di estimate-tokens --device cuda

   # Then use it in the pipeline
   dna23di run \
       --input genome.fasta \
       --output results.json \
       --device cuda \
       --encoding-size 15000  # Use recommended value from estimate-tokens

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
   dna23di run --input genome.fasta --output results.json
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
