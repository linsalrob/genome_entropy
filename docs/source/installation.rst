Installation
============

Requirements
------------

``genome_entropy`` requires Python 3.10 or newer. CI tests Python 3.10, 3.11,
and 3.12. Version 0.1.15 declares these runtime minimums:

* PyTorch 2.0
* Transformers 5.14.1
* Accelerate 0.20
* pygenetic-code 0.20
* Typer 0.9
* tqdm 4.65
* protobuf 6.33.1
* sentencepiece 0.2.1
* Biopython 1.80

The constraints are lower bounds, not claims that every future release is
compatible. Use an isolated environment and preserve a working lock or package
record for reproducible analyses.

PyPI
----

.. code-block:: bash

   python -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install genome_entropy

The distribution and import names use an underscore in project metadata
(``genome_entropy``); PyPI normalises underscores and hyphens for lookup.

Optional extras
---------------

.. code-block:: bash

   python -m pip install "genome_entropy[ml]"
   python -m pip install "genome_entropy[docs]"
   python -m pip install "genome_entropy[dev]"

The ML extra installs XGBoost, scikit-learn, and NumPy. The docs extra installs
Sphinx, the Read the Docs theme, MyST, link support, and dependencies needed to
import optional ML modules during autodoc. The dev extra provides pytest,
coverage, Black, Ruff, mypy, and pytest plugins.

Development checkout
--------------------

.. code-block:: bash

   git clone https://github.com/linsalrob/genome_entropy.git
   cd genome_entropy
   python -m venv .venv
   . .venv/bin/activate
   python -m pip install -e ".[dev,docs,ml]"

External ``get_orfs`` dependency
--------------------------------

ORF discovery invokes an external executable from
https://github.com/linsalrob/get_orfs. It is not bundled or installed by
``pip``. Follow that project's Rust/Cargo build instructions, place the binary
on ``PATH``, or set an explicit path before running:

.. code-block:: bash

   export GET_ORFS_PATH=/absolute/path/to/get_orfs

Commands that start from existing protein records, such as ``encode3di``, do
not invoke ``get_orfs``.

CPU, NVIDIA, AMD, and Apple
---------------------------

PyTorch installation is platform specific. Select the official CPU, CUDA, or
ROCm wheel appropriate to the host, then install ``genome_entropy`` without
allowing a generic PyTorch wheel to replace it. On Apple Silicon, the normal
macOS PyTorch package supplies MPS when supported.

ROCm devices appear through PyTorch's ``cuda`` API, so the application uses
``--device cuda`` on AMD as well as NVIDIA. This convention applies to model
inference, not automatically to XGBoost. See :doc:`hpc` and :doc:`ml`.

Model downloads and offline use
-------------------------------

.. code-block:: bash

   genome_entropy download --model gbouras13/modernprost-50M

This requires internet access on the first run and writes to the Hugging Face
cache (normally below ``~/.cache/huggingface`` unless Hugging Face environment
variables configure another location). Pre-cache on a networked node when
compute nodes are offline. ModernProst executes trusted repository code; review
the security notes in :doc:`models`.

Verification
------------

.. code-block:: bash

   genome_entropy --version
   genome_entropy --help
   get_orfs --help
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

Do not use ``nvidia-smi`` as an AMD verification command. Cluster-specific
installation examples are under ``slurm/`` and require local customisation.
