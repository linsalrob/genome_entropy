Changelog
=========

All notable changes to this project will be documented in this file.

[0.1.0] - 2026-01-19
--------------------

Initial release of genome_entropy.

Added
^^^^^

* Complete pipeline: DNA → ORF → Protein → 3Di → Entropy
* ORF finding using external get_orfs binary
* Protein translation with all NCBI genetic code tables
* 3Di encoding via ProstT5 model
* Shannon entropy calculation at all levels
* Modular CLI with individual commands:
  
  * ``genome_entropy run`` - Complete pipeline
  * ``genome_entropy orf`` - Find ORFs
  * ``genome_entropy translate`` - Translate to proteins
  * ``genome_entropy encode3di`` - Encode to 3Di
  * ``genome_entropy entropy`` - Calculate entropy
  * ``genome_entropy download`` - Pre-download models
  * ``genome_entropy estimate-tokens`` - Estimate optimal encoding size

* GPU acceleration support:
  
  * CUDA (NVIDIA GPUs)
  * MPS (Apple Silicon)
  * CPU fallback

* Comprehensive logging system:
  
  * Configurable log levels
  * File or STDOUT output
  * Progress tracking

* Token size estimation for optimal GPU utilization
* Batch processing for efficient encoding
* JSON I/O for structured data
* FASTA reading and writing
* Complete test suite with unit and integration tests
* Comprehensive documentation
* Example data and scripts

Features
^^^^^^^^

* Auto-detection of best available device (CUDA/MPS/CPU)
* Graceful fallback on GPU memory errors
* Support for all NCBI genetic code tables
* Customizable ORF length filtering
* Normalized and non-normalized entropy
* Type hints throughout codebase
* Google-style docstrings

Known Limitations
^^^^^^^^^^^^^^^^^

* Requires external get_orfs binary
* ProstT5 model is large (~2GB)
* 3Di encoding is memory-intensive
* Integration tests not run in CI

[Unreleased]
------------

Planned features for future releases:

* Additional encoder models
* Parallel processing for large datasets
* Streaming mode for very large files
* Additional entropy metrics
* Web interface
* Pre-built binaries for get_orfs
