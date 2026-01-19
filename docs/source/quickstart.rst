Quick Start Guide
=================

This guide will help you get started with **dna23di** in minutes.

Prerequisites
-------------

* orf_entropy installed (see :doc:`installation`)
* get_orfs binary available in PATH
* Sample FASTA file with DNA sequences

Basic Usage
-----------

Complete Pipeline
^^^^^^^^^^^^^^^^^

Run the entire pipeline from DNA to 3Di with a single command:

.. code-block:: bash

   dna23di run --input examples/example_small.fasta --output results.json

This command will:

1. Find all ORFs in the input DNA sequences
2. Translate ORFs to protein sequences
3. Encode proteins to 3Di structural tokens using ProstT5
4. Calculate Shannon entropy at all levels
5. Save results to JSON

Step-by-Step Pipeline
^^^^^^^^^^^^^^^^^^^^^^

Alternatively, run each step individually:

.. code-block:: bash

   # Step 1: Find ORFs
   dna23di orf --input input.fasta --output orfs.json

   # Step 2: Translate ORFs to proteins
   dna23di translate --input orfs.json --output proteins.json

   # Step 3: Encode proteins to 3Di
   dna23di encode3di --input proteins.json --output 3di.json

   # Step 4: Calculate entropy
   dna23di entropy --input 3di.json --output entropy.json

Example Output
--------------

Results are saved in JSON format:

.. code-block:: json

   [
     {
       "input_id": "seq1",
       "input_dna_length": 1500,
       "orfs": [
         {
           "parent_id": "seq1",
           "orf_id": "seq1_orf_1",
           "start": 0,
           "end": 300,
           "strand": "+",
           "frame": 0,
           "nt_sequence": "ATGGCA...",
           "aa_sequence": "MA...",
           "table_id": 11,
           "has_start_codon": true,
           "has_stop_codon": true
         }
       ],
       "proteins": [...],
       "three_dis": [
         {
           "orf_id": "seq1_orf_1",
           "three_di": "AAABBBCCC...",
           "method": "prostt5_aa2fold",
           "model_name": "Rostlab/ProstT5_fp16"
         }
       ],
       "entropy": {
         "dna_entropy_global": 1.95,
         "orf_nt_entropy": {"seq1_orf_1": 1.85},
         "protein_aa_entropy": {"seq1_orf_1": 3.12},
         "three_di_entropy": {"seq1_orf_1": 2.89},
         "alphabet_sizes": {
           "dna": 4,
           "protein": 20,
           "three_di": 20
         }
       }
     }
   ]

Common Use Cases
----------------

Use GPU for Faster Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   dna23di run --input data.fasta --output results.json --device cuda

Use Different Genetic Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Standard genetic code (Table 1)
   dna23di run --input data.fasta --output results.json --table 1

   # Bacterial code (Table 11, default)
   dna23di run --input data.fasta --output results.json --table 11

Filter Short ORFs
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Only keep proteins >= 50 amino acids
   dna23di run --input data.fasta --output results.json --min-aa 50

Enable Debug Logging
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   dna23di --log-level DEBUG run --input data.fasta --output results.json

Log to File
^^^^^^^^^^^

.. code-block:: bash

   dna23di --log-file pipeline.log run --input data.fasta --output results.json

Pre-download Models
^^^^^^^^^^^^^^^^^^^

Download models before running the pipeline:

.. code-block:: bash

   dna23di download --model Rostlab/ProstT5_fp16

Estimate Optimal Token Size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Find the best encoding size for your GPU:

.. code-block:: bash

   dna23di estimate-tokens --device cuda

Input File Format
-----------------

DNA sequences should be in FASTA format:

.. code-block:: text

   >sequence1 Description of sequence 1
   ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
   TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
   >sequence2 Description of sequence 2
   ATGGGGCCCTTTAAAGGGCCCTTTAAAGGGCCCTTTAAAGGG
   CCCTTTAAAGGGCCCTTTAAAGGGCCCTTTAAA

Tips for Large Datasets
-----------------------

1. **Use GPU**: Encoding is much faster on GPU
2. **Adjust batch size**: Increase for faster processing, decrease if OOM errors
3. **Filter short ORFs**: Use ``--min-aa`` to exclude short proteins
4. **Log to file**: Use ``--log-file`` to track progress
5. **Estimate tokens first**: Use ``estimate-tokens`` to find optimal encoding size

Example Workflow
----------------

Complete workflow for analyzing bacterial genomes:

.. code-block:: bash

   # 1. Pre-download the model
   dna23di download --model Rostlab/ProstT5_fp16

   # 2. Estimate optimal token size for your GPU
   dna23di estimate-tokens --device cuda

   # 3. Run the pipeline with bacterial genetic code
   dna23di --log-file analysis.log run \
       --input bacterial_genome.fasta \
       --output results.json \
       --table 11 \
       --min-aa 30 \
       --device cuda

   # 4. Check the log file for any issues
   cat analysis.log

Performance Benchmarks
----------------------

Approximate processing times on different hardware:

+---------------------+---------------+---------------+
| Hardware            | 100 sequences | 1000 sequences|
+=====================+===============+===============+
| CPU (8 cores)       | ~5 minutes    | ~50 minutes   |
+---------------------+---------------+---------------+
| NVIDIA RTX 3090     | ~1 minute     | ~10 minutes   |
+---------------------+---------------+---------------+
| Apple M1 Max (MPS)  | ~2 minutes    | ~20 minutes   |
+---------------------+---------------+---------------+

*Note: Times are approximate and depend on sequence length and system load.*

Next Steps
----------

* Learn about all CLI commands: :doc:`cli`
* Understand the pipeline in detail: :doc:`user_guide`
* Use the Python API: :doc:`api`
* Optimize token estimation: :doc:`token_estimation`
