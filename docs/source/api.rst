API Reference
=============

This page documents the Python API for **genome_entropy**. You can use these modules directly in your Python code for more fine-grained control over the pipeline.

Core Modules
------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   genome_entropy.orf
   genome_entropy.translate
   genome_entropy.encode3di
   genome_entropy.entropy
   genome_entropy.pipeline
   genome_entropy.io

ORF Finding
-----------

.. automodule:: genome_entropy.orf
   :members:
   :undoc-members:
   :show-inheritance:

Types
^^^^^

.. autoclass:: genome_entropy.orf.types.OrfRecord
   :members:
   :undoc-members:
   :show-inheritance:

Finder
^^^^^^

.. automodule:: genome_entropy.orf.finder
   :members:
   :undoc-members:
   :show-inheritance:

Translation
-----------

.. automodule:: genome_entropy.translate
   :members:
   :undoc-members:
   :show-inheritance:

Translator
^^^^^^^^^^

.. automodule:: genome_entropy.translate.translator
   :members:
   :undoc-members:
   :show-inheritance:

3Di Encoding
------------

.. automodule:: genome_entropy.encode3di
   :members:
   :undoc-members:
   :show-inheritance:

Types
^^^^^

.. automodule:: genome_entropy.encode3di.types
   :members:
   :undoc-members:
   :show-inheritance:

Encoder
^^^^^^^

.. automodule:: genome_entropy.encode3di.encoder
   :members:
   :undoc-members:
   :show-inheritance:

Encoding Functions
^^^^^^^^^^^^^^^^^^

.. automodule:: genome_entropy.encode3di.encoding
   :members:
   :undoc-members:
   :show-inheritance:

Token Estimator
^^^^^^^^^^^^^^^

.. automodule:: genome_entropy.encode3di.token_estimator
   :members:
   :undoc-members:
   :show-inheritance:

Entropy Calculation
-------------------

.. automodule:: genome_entropy.entropy
   :members:
   :undoc-members:
   :show-inheritance:

Shannon Entropy
^^^^^^^^^^^^^^^

.. automodule:: genome_entropy.entropy.shannon
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline
--------

.. automodule:: genome_entropy.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Runner
^^^^^^

.. automodule:: genome_entropy.pipeline.runner
   :members:
   :undoc-members:
   :show-inheritance:

I/O
---

.. automodule:: genome_entropy.io
   :members:
   :undoc-members:
   :show-inheritance:

FASTA I/O
^^^^^^^^^

.. automodule:: genome_entropy.io.fasta
   :members:
   :undoc-members:
   :show-inheritance:

JSON I/O
^^^^^^^^

.. automodule:: genome_entropy.io.jsonio
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. automodule:: genome_entropy.config
   :members:
   :undoc-members:
   :show-inheritance:

Errors
------

.. automodule:: genome_entropy.errors
   :members:
   :undoc-members:
   :show-inheritance:

Logging
-------

.. automodule:: genome_entropy.logging_config
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

ORF Finding
^^^^^^^^^^^

.. code-block:: python

   from genome_entropy.orf.finder import find_orfs

   # Find ORFs in a FASTA file
   orfs = find_orfs(
       fasta_path="genome.fasta",
       table_id=11,
       min_length_nt=90
   )

   # Examine results
   for orf in orfs:
       print(f"ORF {orf.orf_id}: {orf.start}-{orf.end} ({orf.strand})")
       print(f"  Nucleotide: {orf.nt_sequence[:50]}...")
       print(f"  Amino acid: {orf.aa_sequence[:50]}...")

Translation
^^^^^^^^^^^

.. code-block:: python

   from genome_entropy.translate.translator import translate_orfs

   # Translate ORFs
   proteins = translate_orfs(orfs, table_id=11)

   for protein in proteins:
       print(f"Protein from {protein.orf.orf_id}: {protein.aa_sequence}")
       print(f"  Length: {protein.aa_length} amino acids")

3Di Encoding
^^^^^^^^^^^^

.. code-block:: python

   from genome_entropy.encode3di import ProstT5ThreeDiEncoder

   # Initialize encoder
   encoder = ProstT5ThreeDiEncoder(
       model_name="Rostlab/ProstT5_fp16",
       device="auto"  # Auto-detect CUDA/MPS/CPU
   )

   # Encode proteins to 3Di
   aa_sequences = [p.aa_sequence for p in proteins]
   three_di_tokens = encoder.encode(
       aa_sequences,
       batch_size=4,
       encoding_size=5000
   )

   for i, tokens in enumerate(three_di_tokens):
       print(f"Protein {i}: {tokens[:50]}...")

Token Estimation
^^^^^^^^^^^^^^^^

.. code-block:: python

   from genome_entropy.encode3di import ProstT5ThreeDiEncoder, estimate_token_size

   # Initialize encoder
   encoder = ProstT5ThreeDiEncoder()

   # Find optimal encoding size
   results = estimate_token_size(
       encoder=encoder,
       start_length=3000,
       end_length=10000,
       step=1000,
       num_trials=3
   )

   print(f"Recommended encoding size: {results['recommended_token_size']} AA")
   
   # Use in encoding
   three_di = encoder.encode(
       sequences,
       encoding_size=results['recommended_token_size']
   )

Shannon Entropy
^^^^^^^^^^^^^^^

.. code-block:: python

   from genome_entropy.entropy.shannon import shannon_entropy, calculate_sequence_entropy

   # Calculate basic entropy
   dna = "ATCGATCGATCG"
   entropy = shannon_entropy(dna)
   print(f"DNA entropy: {entropy:.2f} bits")

   # Normalized entropy
   normalized = shannon_entropy(
       dna,
       alphabet=set("ACGT"),
       normalize=True
   )
   print(f"Normalized: {normalized:.2f}")

   # For biological sequences
   protein = "MKKYTLFLGLLGLVAAGTLWGLSACCA"
   protein_entropy = calculate_sequence_entropy(protein)
   print(f"Protein entropy: {protein_entropy:.2f} bits")

Complete Pipeline
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   from genome_entropy.pipeline.runner import run_pipeline

   # Run complete pipeline
   results = run_pipeline(
       input_fasta=Path("genome.fasta"),
       output_json=Path("results.json"),
       table_id=11,
       min_aa_len=30,
       model_name="Rostlab/ProstT5_fp16",
       device="auto",
       compute_entropy=True
   )

   # Process results
   for result in results:
       print(f"Sequence: {result.input_id}")
       print(f"  DNA length: {result.input_dna_length}")
       print(f"  ORFs found: {len(result.orfs)}")
       print(f"  DNA entropy: {result.entropy.dna_entropy_global:.2f}")
       
       for orf_id, entropy in result.entropy.protein_aa_entropy.items():
           print(f"  Protein {orf_id} entropy: {entropy:.2f}")

I/O Operations
^^^^^^^^^^^^^^

.. code-block:: python

   from genome_entropy.io.fasta import read_fasta, write_fasta
   from genome_entropy.io.jsonio import save_json, load_json

   # Read FASTA
   sequences = read_fasta("genome.fasta")
   for seq_id, seq in sequences:
       print(f"{seq_id}: {len(seq)} bp")

   # Write FASTA
   output_sequences = [
       ("seq1", "ATCGATCG"),
       ("seq2", "GCTAGCTA")
   ]
   write_fasta("output.fasta", output_sequences)

   # Save/load JSON
   data = {"key": "value", "results": [1, 2, 3]}
   save_json(data, "output.json")
   loaded = load_json("output.json")

Error Handling
^^^^^^^^^^^^^^

.. code-block:: python

   from genome_entropy.errors import (
       OrfEntropyError,
       OrfFinderError,
       TranslationError,
       EncodingError
   )

   try:
       orfs = find_orfs("genome.fasta", table_id=11)
   except OrfFinderError as e:
       print(f"ORF finding failed: {e}")
   except OrfEntropyError as e:
       print(f"General error: {e}")

Custom Logging
^^^^^^^^^^^^^^

.. code-block:: python

   from genome_entropy.logging_config import configure_logging
   import logging

   # Configure logging
   configure_logging(level="DEBUG", log_file="debug.log")

   # Get logger for your module
   logger = logging.getLogger(__name__)
   logger.info("Starting analysis")
   logger.debug("Detailed debug information")

Advanced: Custom Batching
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from genome_entropy.encode3di.encoder import ProstT5ThreeDiEncoder

   encoder = ProstT5ThreeDiEncoder()

   # Create batches with token budget
   sequences = ["MKKYTLFLG", "ACDEFGHIK", ...]
   batches = encoder.token_budget_batches(
       sequences,
       max_total_length=5000
   )

   # Process each batch
   all_results = []
   for batch in batches:
       batch_results = encoder._encode_batch(batch)
       all_results.extend(batch_results)

Data Classes
------------

OrfRecord
^^^^^^^^^

.. code-block:: python

   @dataclass
   class OrfRecord:
       parent_id: str          # Source sequence ID
       orf_id: str             # Unique ORF identifier
       start: int              # 0-based, inclusive
       end: int                # 0-based, exclusive
       strand: Literal["+","-"]
       frame: int              # 0, 1, 2
       nt_sequence: str        # Nucleotide sequence
       aa_sequence: str        # Amino acid sequence
       table_id: int           # NCBI translation table
       has_start_codon: bool
       has_stop_codon: bool

ThreeDiRecord
^^^^^^^^^^^^^

.. code-block:: python

   @dataclass
   class ThreeDiRecord:
       orf_id: str
       three_di: str           # 3Di token sequence
       method: Literal["prostt5_aa2fold"]
       model_name: str
       inference_device: str   # "cuda", "mps", or "cpu"

EntropyReport
^^^^^^^^^^^^^

.. code-block:: python

   @dataclass
   class EntropyReport:
       dna_entropy_global: float
       orf_nt_entropy: dict[str, float]     # orf_id → entropy
       protein_aa_entropy: dict[str, float]
       three_di_entropy: dict[str, float]
       alphabet_sizes: dict[str, int]

Type Hints
----------

All modules use comprehensive type hints for better IDE support and type checking:

.. code-block:: python

   from typing import List, Dict, Optional, Tuple
   from pathlib import Path

   def find_orfs(
       fasta_path: Path | str,
       table_id: int = 11,
       min_length_nt: int = 90
   ) -> List[OrfRecord]:
       ...

Next Steps
----------

* See :doc:`user_guide` for conceptual overview
* Check :doc:`cli` for command-line usage
* Read :doc:`development` for contributing
