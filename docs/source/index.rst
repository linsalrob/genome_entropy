genome_entropy Documentation
============================

.. image:: https://github.com/linsalrob/genome_entropy/workflows/Python%20CI/badge.svg
   :target: https://github.com/linsalrob/genome_entropy/actions
   :alt: Python CI

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

Welcome to the documentation for **genome_entropy**, a complete bioinformatics pipeline that converts DNA sequences → ORFs → proteins → 3Di structural tokens, computing Shannon entropy at each representation level.

Overview
--------

**genome_entropy** enables researchers to:

* Extract Open Reading Frames (ORFs) from DNA sequences
* Translate ORFs to protein sequences using customizable genetic codes
* Predict structural alphabet tokens (3Di) directly from sequences using ModernProst or ProstT5
* Calculate and compare Shannon entropy at DNA, ORF, protein, and 3Di levels
* Train ML classifiers that predict GenBank annotations from ORF-level features
* Process data efficiently with GPU acceleration (CUDA, MPS, CPU, or multiple GPUs)

Key Features
------------

🧬 **ORF Finding**
   Extract Open Reading Frames from DNA sequences using customizable genetic codes

🔄 **Translation**
   Convert ORFs to protein sequences with support for all NCBI genetic code tables

🏗️ **3Di Encoding**
   Predict structural alphabet tokens directly from sequences using ModernProst or ProstT5

📊 **Entropy Analysis**
   Calculate Shannon entropy at DNA, ORF, protein, and 3Di levels

🤖 **ML Classifier**
   Train XGBoost or neural-network models to predict GenBank annotations

⚡ **GPU Acceleration**
   Auto-detect and use CUDA, MPS (Apple Silicon), CPU, or multiple GPUs

🚀 **Multi-GPU Support**
   Parallelize 3Di encoding across available GPUs

🔧 **Modular CLI**
   Run complete pipeline or individual steps

📝 **Comprehensive Logging**
   Configurable log levels and output to file or STDOUT

Getting Started
---------------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   user_guide

Reference
---------

.. toctree::
   :maxdepth: 2

   cli
   api
   token_estimation

Development
-----------

.. toctree::
   :maxdepth: 2

   development
   changelog

Citation
--------

If you use this software, please cite:

* **ProstT5**: Heinzinger et al. (2023), "ProstT5: Bilingual Language Model for Protein Sequence and Structure"
* **get_orfs**: https://github.com/linsalrob/get_orfs
* **pygenetic-code**: https://github.com/linsalrob/genetic_codes

License
-------

MIT License - see `LICENSE <https://github.com/linsalrob/genome_entropy/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
