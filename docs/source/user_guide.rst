User Guide
==========

This guide provides a comprehensive overview of the **genome_entropy** pipeline, explaining concepts, data flow, and best practices.

Pipeline Overview
-----------------

The **genome_entropy** pipeline transforms DNA sequences through multiple representation levels, computing Shannon entropy at each stage:

.. code-block:: text

   DNA (FASTA)
       ↓
   [ORF Finding]
       ↓
   ORFs (nucleotides) → Entropy₁
       ↓
   [Translation]
       ↓
   Proteins (amino acids) → Entropy₂
       ↓
   [3Di Encoding via ProstT5]
       ↓
   3Di Tokens (structural) → Entropy₃
       ↓
   [Entropy Analysis]
       ↓
   Complete Entropy Report

This multi-level analysis enables comparison of information content across different biological sequence representations.

Understanding ORFs
------------------

What is an ORF?
^^^^^^^^^^^^^^^

An **Open Reading Frame (ORF)** is a sequence of DNA between a start codon (typically ATG) and a stop codon (TAA, TAG, or TGA), representing a potential protein-coding region.

Reading Frames
^^^^^^^^^^^^^^

DNA has **six possible reading frames**:

* **Three forward frames** (starting at positions 0, 1, 2)
* **Three reverse frames** (reverse complement, starting at positions 0, 1, 2)

Example:

.. code-block:: text

   DNA:     ATGGCATAGCTAA
   Frame 0: ATG GCA TAG CTA A
   Frame 1: A TGG CAT AGC TAA
   Frame 2: AT GGC ATA GCT AA

ORF Properties
^^^^^^^^^^^^^^

Each ORF has the following properties:

* **Position**: Start and end coordinates (0-based)
* **Strand**: Forward (+) or reverse (-)
* **Frame**: Which reading frame (0, 1, or 2)
* **Codons**: Presence of start/stop codons
* **Sequences**: Both nucleotide and amino acid sequences

Genetic Code Tables
-------------------

The pipeline uses NCBI genetic code tables for translation. Different organisms use different genetic codes:

Common Tables
^^^^^^^^^^^^^

+--------+--------------------------------------------------+------------------+
| Table  | Description                                      | Typical Use      |
+========+==================================================+==================+
| 1      | Standard genetic code                            | Eukaryotes       |
+--------+--------------------------------------------------+------------------+
| 11     | Bacterial, archaeal, plant plastid (default)     | Bacteria, Archaea|
+--------+--------------------------------------------------+------------------+
| 4      | Mold, protozoan, coelenterate mitochondrial      | Some protozoans  |
+--------+--------------------------------------------------+------------------+
| 2      | Vertebrate mitochondrial                         | Mitochondria     |
+--------+--------------------------------------------------+------------------+
| 5      | Invertebrate mitochondrial                       | Mitochondria     |
+--------+--------------------------------------------------+------------------+

Key Differences
^^^^^^^^^^^^^^^

The main differences between genetic codes involve stop codons and rare amino acids:

* **Table 1**: UGA = Stop
* **Table 11**: UGA = Stop (same as standard)
* **Table 4**: UGA = Trp (not a stop!)

**Important**: Always use the correct genetic code table for your organism.

Understanding 3Di
-----------------

What is 3Di?
^^^^^^^^^^^^

**3Di** (3D-interactions) is a structural alphabet that represents local 3D protein backbone geometry using 20 discrete states. It was developed for the Foldseek structural search tool.

Why 3Di?
^^^^^^^^

Traditional approaches require:
1. Amino acid sequence
2. Protein structure prediction (e.g., AlphaFold)
3. Structure → 3Di conversion

**ProstT5** enables direct sequence → 3Di prediction, skipping the expensive structure prediction step:

.. code-block:: text

   Traditional:  AA → AlphaFold → PDB → Foldseek → 3Di
   ProstT5:      AA → ProstT5 → 3Di

Benefits:

* Much faster (no structure prediction)
* Lower computational requirements
* Enables large-scale structural analysis

3Di Alphabet
^^^^^^^^^^^^

The 3Di alphabet has 20 symbols (like amino acids) representing different structural states. Each symbol encodes local backbone geometry.

Shannon Entropy
---------------

What is Entropy?
^^^^^^^^^^^^^^^^

**Shannon entropy** measures the information content or complexity of a sequence:

.. code-block:: text

   H = -Σ(p_i × log₂(p_i))

where p_i is the frequency of symbol i.

Interpretation
^^^^^^^^^^^^^^

* **High entropy**: More complex, diverse, unpredictable
* **Low entropy**: More repetitive, simple, predictable

Examples:

.. code-block:: python

   # Maximum entropy (all symbols equally likely)
   "ACGTACGT" → H ≈ 2.0 bits

   # Minimum entropy (one symbol only)
   "AAAAAAAA" → H = 0.0 bits

   # Intermediate
   "AAAACCCC" → H = 1.0 bits

Normalized Entropy
^^^^^^^^^^^^^^^^^^

Normalized entropy scales values to [0, 1] by dividing by the maximum possible entropy:

.. code-block:: text

   H_norm = H / log₂(|alphabet|)

This allows fair comparison across different alphabets:

* DNA: 4 symbols (max entropy = 2.0)
* Protein: 20 symbols (max entropy ≈ 4.32)
* 3Di: 20 symbols (max entropy ≈ 4.32)

Entropy in Biology
^^^^^^^^^^^^^^^^^^

Biological applications:

* **Low-complexity regions**: Entropy < 2.0 indicates repetitive sequences
* **Sequence quality**: High entropy suggests good diversity
* **Structural complexity**: Compare protein vs. 3Di entropy
* **Functional sites**: Often have distinct entropy patterns

Data Flow
---------

Step 1: Input (FASTA)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   >sequence1
   ATGGCTAGCTAGCTAGCTAG...
   >sequence2
   ATGGGCCCTTTTAAA...

Step 2: ORF Finding
^^^^^^^^^^^^^^^^^^^

Extract all potential coding regions:

.. code-block:: json

   {
     "parent_id": "sequence1",
     "orf_id": "sequence1_orf_1",
     "start": 0,
     "end": 300,
     "strand": "+",
     "frame": 0,
     "nt_sequence": "ATGGCTAGC...",
     "aa_sequence": "MAS...",
     "has_start_codon": true,
     "has_stop_codon": true
   }

Step 3: Translation
^^^^^^^^^^^^^^^^^^^

Convert nucleotides to amino acids:

.. code-block:: text

   Nucleotides: ATGGCTAGC → ATG GCT AGC
   Amino acids:             → M   A   S

Step 4: 3Di Encoding
^^^^^^^^^^^^^^^^^^^^

Predict structural tokens using ProstT5:

.. code-block:: json

   {
     "orf_id": "sequence1_orf_1",
     "three_di": "AAABBBCCCDDD...",
     "method": "prostt5_aa2fold",
     "model_name": "Rostlab/ProstT5_fp16",
     "inference_device": "cuda"
   }

Step 5: Entropy Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute entropy at all levels:

.. code-block:: json

   {
     "dna_entropy_global": 1.95,
     "orf_nt_entropy": {
       "sequence1_orf_1": 1.85,
       "sequence1_orf_2": 1.90
     },
     "protein_aa_entropy": {
       "sequence1_orf_1": 3.12,
       "sequence1_orf_2": 3.25
     },
     "three_di_entropy": {
       "sequence1_orf_1": 2.89,
       "sequence1_orf_2": 2.95
     },
     "alphabet_sizes": {
       "dna": 4,
       "protein": 20,
       "three_di": 20
     }
   }

Performance Considerations
--------------------------

GPU vs CPU
^^^^^^^^^^

ProstT5 encoding is the bottleneck:

* **CPU**: Slow but works everywhere
* **CUDA**: 10-50× faster with NVIDIA GPU
* **MPS**: 5-20× faster on Apple Silicon

Memory Management
^^^^^^^^^^^^^^^^^

GPU memory is limited. Key parameters:

* **batch_size**: Number of sequences processed simultaneously
* **encoding_size**: Total amino acids per batch

If you get "CUDA out of memory":

1. Reduce ``batch_size``
2. Reduce ``encoding_size``
3. Use ``--device cpu``

Token Size Estimation
^^^^^^^^^^^^^^^^^^^^^

Use ``estimate-tokens`` to find optimal settings:

.. code-block:: bash

   genome_entropy estimate-tokens --device cuda

This tests different encoding sizes and recommends the best value for your GPU.

Best Practices
--------------

Choosing Parameters
^^^^^^^^^^^^^^^^^^^

**Genetic code table:**

* Use table 11 for bacteria and archaea (default)
* Use table 1 for eukaryotes
* Check NCBI documentation for unusual organisms

**Minimum length:**

* Default 30 AA filters very short ORFs
* Increase to 50-100 AA for higher confidence
* Decrease to 10-20 AA for viral genomes

**Device selection:**

* Use ``auto`` to automatically detect best device (recommended)
* Use ``cuda`` to force GPU (fails if not available)
* Use ``cpu`` for maximum compatibility

Logging
^^^^^^^

Enable debug logging for troubleshooting:

.. code-block:: bash

   genome_entropy --log-level DEBUG --log-file debug.log run --input data.fasta --output results.json

Log files help diagnose:

* Model loading issues
* Memory problems
* Processing bottlenecks
* Unexpected results

Large Datasets
^^^^^^^^^^^^^^

For processing many sequences:

1. **Estimate tokens first**: Find optimal batch size
2. **Use GPU**: Essential for large datasets
3. **Filter short ORFs**: Use ``--min-aa 50`` or higher
4. **Monitor memory**: Watch for OOM errors
5. **Log to file**: Track progress
6. **Split input**: Process in chunks if too large

Quality Control
^^^^^^^^^^^^^^^

Check your results:

* **ORF count**: Too many or too few might indicate issues
* **Entropy values**: Should be within expected ranges
* **3Di output**: Should be same length as protein input
* **Log messages**: Look for warnings or errors

Common Patterns
---------------

Entropy Comparisons
^^^^^^^^^^^^^^^^^^^

Typical entropy patterns:

.. code-block:: text

   DNA entropy:    ~1.8-2.0 (max 2.0 for 4 symbols)
   Protein entropy: ~3.0-4.0 (max 4.32 for 20 symbols)
   3Di entropy:     ~2.5-3.5 (varies by structure)

Observations:

* Proteins usually have higher entropy than DNA (more symbols)
* 3Di entropy reflects structural complexity
* Low-complexity regions have entropy < 2.0

Structural Predictions
^^^^^^^^^^^^^^^^^^^^^^

3Di tokens enable:

* Fast structural searches (via Foldseek)
* Structural alignment
* Structure-based clustering
* Fold recognition

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**ORF finding fails:**

* Check get_orfs binary is installed and in PATH
* Verify input is valid FASTA format
* Try different genetic code table

**Translation errors:**

* Ensure correct genetic code table
* Check for ambiguous bases (N) in sequences

**Encoding fails:**

* Verify model downloaded: ``genome_entropy download``
* Check GPU memory: Use ``--device cpu`` or reduce batch size
* Update PyTorch/Transformers: ``pip install --upgrade torch transformers``

**Out of memory:**

* Reduce batch size: ``--batch-size 1``
* Reduce encoding size: ``--encoding-size 2000``
* Use CPU: ``--device cpu``
* Process fewer sequences at once

Performance Issues
^^^^^^^^^^^^^^^^^^

**Slow encoding:**

* Use GPU if available
* Increase batch size (if memory allows)
* Use fp16 model: ``Rostlab/ProstT5_fp16``

**Slow ORF finding:**

* This is usually fast; check input file size
* Consider filtering input sequences

Next Steps
----------

* Try the :doc:`quickstart` examples
* Read the :doc:`cli` reference
* Explore the :doc:`api` for Python integration
* Learn about :doc:`token_estimation` optimization
