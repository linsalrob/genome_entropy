Installation
============

System Requirements
-------------------

* **Python**: 3.10 or higher
* **Operating System**: Linux, macOS, or Windows with WSL
* **Optional**: NVIDIA GPU with CUDA support for accelerated inference

Python Dependencies
-------------------

The following packages are automatically installed with genome_entropy:

* **PyTorch** >= 2.0.0 (GPU support optional)
* **Transformers** >= 4.30.0 (HuggingFace)
* **pygenetic-code** >= 0.20.0
* **typer** >= 0.9.0
* **tqdm** >= 4.65.0
* **protobuf** >= 6.33.1
* **sentencepiece** >= 0.2.1

Installation Methods
--------------------

From Source (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/linsalrob/genome_entropy.git
      cd genome_entropy

2. Create and activate a virtual environment:

   .. code-block:: bash

      python3 -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the package:

   .. code-block:: bash

      pip install -e .

4. (Optional) Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

Installing External Dependencies
---------------------------------

get_orfs Binary
^^^^^^^^^^^^^^^

The ORF finder requires the ``get_orfs`` binary from https://github.com/linsalrob/get_orfs

**Build Instructions:**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/linsalrob/get_orfs.git /tmp/get_orfs
   cd /tmp/get_orfs

   # Build using CMake
   mkdir build && cd build
   cmake ..
   make
   cmake --install . --prefix ..

   # Add to PATH
   export PATH="/tmp/get_orfs/bin:$PATH"

   # Or set environment variable
   export GET_ORFS_PATH=/tmp/get_orfs/bin/get_orfs

**Requirements for building get_orfs:**

* C++ compiler (g++ or clang++)
* CMake >= 3.10
* Make

Verifying Installation
----------------------

Check that the installation was successful:

.. code-block:: bash

   # Check CLI is available
   genome_entropy --version

   # Check get_orfs is available
   which get_orfs
   # Or check environment variable
   echo $GET_ORFS_PATH

   # Run a simple test
   genome_entropy download --help

GPU Support
-----------

CUDA (NVIDIA GPUs)
^^^^^^^^^^^^^^^^^^

If you have an NVIDIA GPU with CUDA support, install PyTorch with CUDA:

.. code-block:: bash

   # For CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121

Verify CUDA is available:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())  # Should print True

MPS (Apple Silicon)
^^^^^^^^^^^^^^^^^^^

For Apple Silicon Macs (M1, M2, M3), MPS acceleration is automatically detected if PyTorch >= 2.0.0 is installed:

.. code-block:: python

   import torch
   print(torch.backends.mps.is_available())  # Should print True

CPU-Only
^^^^^^^^

For CPU-only systems, no special configuration is needed. The pipeline will automatically use CPU.

Downloading Models
------------------

Pre-download the ProstT5 model to avoid delays during first use:

.. code-block:: bash

   genome_entropy download --model Rostlab/ProstT5_fp16

This downloads the model to your HuggingFace cache directory (typically ``~/.cache/huggingface/``).

Troubleshooting
---------------

ModuleNotFoundError
^^^^^^^^^^^^^^^^^^^

**Error**: ``ModuleNotFoundError: No module named 'genome_entropy'``

**Solution**: Make sure you installed the package:

.. code-block:: bash

   cd /path/to/genome_entropy
   pip install -e .

get_orfs Not Found
^^^^^^^^^^^^^^^^^^

**Error**: ``get_orfs binary not found``

**Solution**: Install get_orfs and add to PATH or set the ``GET_ORFS_PATH`` environment variable:

.. code-block:: bash

   export GET_ORFS_PATH=/path/to/get_orfs/bin/get_orfs

CUDA Out of Memory
^^^^^^^^^^^^^^^^^^

**Error**: ``RuntimeError: CUDA out of memory``

**Solution**: Use CPU or reduce batch size:

.. code-block:: bash

   # Use CPU
   genome_entropy run --input data.fasta --output results.json --device cpu

   # Or reduce batch size
   genome_entropy encode3di --input proteins.json --output 3di.json --batch-size 1

Model Download Fails
^^^^^^^^^^^^^^^^^^^^

**Error**: Connection errors when downloading models

**Solution**: 

* Check internet connection
* Verify HuggingFace cache permissions: ``ls -la ~/.cache/huggingface/``
* Try downloading manually:

  .. code-block:: bash

     python -c "from transformers import AutoModel; AutoModel.from_pretrained('Rostlab/ProstT5_fp16')"

Permission Errors
^^^^^^^^^^^^^^^^^

**Error**: Permission denied when installing packages

**Solution**: Use a virtual environment (recommended) or install with ``--user`` flag:

.. code-block:: bash

   pip install --user -e .

Next Steps
----------

* Continue to :doc:`quickstart` for basic usage examples
* See :doc:`cli` for complete command reference
* Read :doc:`user_guide` for detailed pipeline documentation
