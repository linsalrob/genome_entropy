Development Guide
=================

This guide is for developers who want to contribute to **orf_entropy** or understand its internals.

Setting Up Development Environment
-----------------------------------

Clone and Install
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Clone repository
   git clone https://github.com/linsalrob/orf_entropy.git
   cd orf_entropy

   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

   # Install in editable mode with dev dependencies
   pip install -e ".[dev]"

Install External Tools
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Install get_orfs binary
   git clone https://github.com/linsalrob/get_orfs.git /tmp/get_orfs
   cd /tmp/get_orfs
   mkdir build && cd build
   cmake .. && make
   cmake --install . --prefix ..
   export PATH="/tmp/get_orfs/bin:$PATH"

Project Structure
-----------------

.. code-block:: text

   orf_entropy/
   ├── src/orf_entropy/         # Main package
   │   ├── __init__.py
   │   ├── config.py            # Configuration and constants
   │   ├── errors.py            # Custom exceptions
   │   ├── logging_config.py    # Logging configuration
   │   ├── io/                  # I/O operations
   │   │   ├── fasta.py         # FASTA reading/writing
   │   │   └── jsonio.py        # JSON serialization
   │   ├── orf/                 # ORF finding
   │   │   ├── types.py         # OrfRecord dataclass
   │   │   └── finder.py        # ORF finder wrapper
   │   ├── translate/           # Translation
   │   │   └── translator.py    # Protein translation
   │   ├── encode3di/           # 3Di encoding
   │   │   ├── types.py         # ThreeDiRecord, etc.
   │   │   ├── encoder.py       # ProstT5ThreeDiEncoder
   │   │   ├── encoding.py      # Core encoding logic
   │   │   ├── token_estimator.py  # Token size estimation
   │   │   └── prostt5.py       # Backward compatibility
   │   ├── entropy/             # Entropy calculation
   │   │   └── shannon.py       # Shannon entropy
   │   ├── pipeline/            # Pipeline orchestration
   │   │   └── runner.py        # End-to-end pipeline
   │   └── cli/                 # Command-line interface
   │       ├── main.py          # CLI entry point
   │       └── commands/        # Individual commands
   ├── tests/                   # Test suite
   ├── docs/                    # Documentation
   ├── examples/                # Example scripts and data
   └── pyproject.toml           # Project configuration

Code Style and Standards
------------------------

Type Hints
^^^^^^^^^^

All functions must have complete type hints:

.. code-block:: python

   from typing import List, Dict, Optional
   from pathlib import Path

   def process_sequences(
       sequences: List[str],
       output_path: Optional[Path] = None
   ) -> Dict[str, float]:
       """Process sequences and return results."""
       ...

Docstrings
^^^^^^^^^^

Use Google-style docstrings:

.. code-block:: python

   def calculate_entropy(sequence: str, normalize: bool = False) -> float:
       """Calculate Shannon entropy of a sequence.
       
       Args:
           sequence: Input sequence string
           normalize: Whether to normalize by alphabet size
           
       Returns:
           Shannon entropy in bits (or normalized to [0,1])
           
       Raises:
           ValueError: If sequence is invalid
           
       Examples:
           >>> calculate_entropy("ACGT")
           2.0
           >>> calculate_entropy("AAAA")
           0.0
       """
       ...

Code Formatting
^^^^^^^^^^^^^^^

Use **black** for formatting (88 character line length):

.. code-block:: bash

   # Format code
   black src/ tests/

   # Check formatting
   black --check src/ tests/

Linting
^^^^^^^

Use **ruff** for linting:

.. code-block:: bash

   # Lint code
   ruff check src/ tests/

   # Auto-fix issues
   ruff check --fix src/ tests/

Type Checking
^^^^^^^^^^^^^

Use **mypy** for type checking:

.. code-block:: bash

   # Type check
   mypy src/orf_entropy/

Testing
-------

Test Organization
^^^^^^^^^^^^^^^^^

Tests are organized by module:

.. code-block:: text

   tests/
   ├── conftest.py                    # Shared fixtures
   ├── test_basic.py                  # Basic sanity tests
   ├── test_orf_finder.py             # ORF finding tests
   ├── test_translation.py            # Translation tests
   ├── test_entropy.py                # Entropy tests
   ├── test_encoder_methods.py        # Encoder tests
   ├── test_token_estimator.py        # Token estimation tests
   ├── test_cli_smoke.py              # CLI smoke tests
   └── test_prostt5_integration.py    # Integration tests (slow)

Running Tests
^^^^^^^^^^^^^

.. code-block:: bash

   # Run all unit tests (fast)
   pytest -k "not integration"

   # Run with coverage
   pytest -k "not integration" --cov=orf_entropy --cov-report=html

   # Run specific test file
   pytest tests/test_entropy.py -v

   # Run integration tests (slow, downloads models)
   RUN_INTEGRATION=1 pytest -v -m integration

Writing Tests
^^^^^^^^^^^^^

Use pytest fixtures from ``conftest.py``:

.. code-block:: python

   def test_entropy_calculation(synthetic_dna):
       """Test entropy calculation on synthetic data."""
       from orf_entropy.entropy.shannon import shannon_entropy
       
       # Use fixture
       entropy = shannon_entropy(synthetic_dna)
       
       # Assertions
       assert 0.0 <= entropy <= 2.0  # DNA max entropy
       assert isinstance(entropy, float)

Mock external dependencies:

.. code-block:: python

   def test_encoder_mock(monkeypatch):
       """Test encoder with mocked model."""
       def mock_encode(*args, **kwargs):
           return ["AAA" * 10]  # Fake 3Di output
       
       monkeypatch.setattr(
           "orf_entropy.encode3di.encoder.ProstT5ThreeDiEncoder._encode_batch",
           mock_encode
       )
       
       # Test with mocked encoder
       ...

Integration Tests
^^^^^^^^^^^^^^^^^

Mark slow tests as integration:

.. code-block:: python

   import pytest

   @pytest.mark.integration
   @pytest.mark.skipif(
       not os.getenv("RUN_INTEGRATION"),
       reason="Integration tests disabled"
   )
   def test_real_prostt5_encoding():
       """Test real ProstT5 encoding (slow)."""
       # This downloads models and runs real inference
       ...

Git Workflow
------------

Branching
^^^^^^^^^

.. code-block:: bash

   # Create feature branch
   git checkout -b feature/my-feature

   # Make changes and commit
   git add .
   git commit -m "Add feature: description"

   # Push and create PR
   git push origin feature/my-feature

Commit Messages
^^^^^^^^^^^^^^^

Use clear, descriptive commit messages:

.. code-block:: text

   # Good
   Add token size estimation for optimal batch sizing
   Fix entropy calculation for empty sequences
   Update documentation for CLI commands

   # Bad
   Fix bug
   Update code
   Changes

Pre-commit Checks
^^^^^^^^^^^^^^^^^

Before committing, run:

.. code-block:: bash

   # Format code
   black src/ tests/

   # Lint
   ruff check src/ tests/

   # Type check
   mypy src/orf_entropy/

   # Test
   pytest -k "not integration"

Adding New Features
-------------------

1. Design the API
^^^^^^^^^^^^^^^^^

Define clear interfaces:

.. code-block:: python

   # Bad: Unclear function
   def process(x, y, z):
       ...

   # Good: Clear, typed interface
   def translate_sequence(
       nucleotide: str,
       table_id: int = 11,
       include_stop: bool = False
   ) -> str:
       """Translate nucleotide to amino acid sequence."""
       ...

2. Implement Core Logic
^^^^^^^^^^^^^^^^^^^^^^^^

Keep functions focused:

.. code-block:: python

   # Single Responsibility Principle
   def read_fasta(path: Path) -> List[Tuple[str, str]]:
       """Read FASTA file and return (id, seq) tuples."""
       ...

   def validate_sequence(seq: str, alphabet: str = "ACGT") -> bool:
       """Validate sequence contains only allowed characters."""
       ...

3. Add Tests
^^^^^^^^^^^^

Test normal cases, edge cases, and errors:

.. code-block:: python

   def test_read_fasta_normal():
       """Test reading valid FASTA."""
       ...

   def test_read_fasta_empty():
       """Test reading empty FASTA."""
       ...

   def test_read_fasta_invalid():
       """Test reading invalid FASTA raises error."""
       with pytest.raises(ValueError):
           ...

4. Update Documentation
^^^^^^^^^^^^^^^^^^^^^^^^

Add docstrings and update relevant docs:

* Module docstrings
* Function docstrings
* README examples
* API reference
* User guide

5. Add CLI Command (if needed)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create command in ``cli/commands/``:

.. code-block:: python

   # cli/commands/my_command.py
   import typer

   def my_command(
       input: Path = typer.Option(..., "--input", "-i"),
       output: Path = typer.Option(..., "--output", "-o"),
   ) -> None:
       """Description of command."""
       # Implementation
       ...

Register in ``cli/main.py``:

.. code-block:: python

   from .commands import my_command
   app.command(name="my-command")(my_command.my_command)

Debugging
---------

Using Logging
^^^^^^^^^^^^^

Add logging to your code:

.. code-block:: python

   from orf_entropy.logging_config import get_logger

   logger = get_logger(__name__)

   def process_data(data):
       logger.debug("Processing %d items", len(data))
       logger.info("Starting processing")
       
       try:
           result = expensive_operation(data)
           logger.info("Processing complete")
           return result
       except Exception as e:
           logger.error("Processing failed: %s", e)
           raise

Interactive Debugging
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Add breakpoint
   breakpoint()  # Python 3.7+

   # Or use pdb
   import pdb; pdb.set_trace()

   # Then use debugger commands:
   # n - next line
   # s - step into
   # c - continue
   # p variable - print variable
   # l - list code

Profiling
^^^^^^^^^

Find performance bottlenecks:

.. code-block:: bash

   # Profile script
   python -m cProfile -s cumtime -o profile.stats script.py

   # View results
   python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(20)"

Memory Profiling
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Install memory profiler
   pip install memory_profiler

   # Profile memory
   python -m memory_profiler script.py

CI/CD
-----

GitHub Actions
^^^^^^^^^^^^^^

The CI pipeline runs on every push/PR:

1. **Linting**: ruff check
2. **Formatting**: black check
3. **Type checking**: mypy
4. **Unit tests**: pytest (integration tests skipped)
5. **Coverage**: Upload to Codecov

See ``.github/workflows/python-ci.yml`` for details.

Local CI Emulation
^^^^^^^^^^^^^^^^^^

Run the same checks locally:

.. code-block:: bash

   # Format check
   black --check src/ tests/

   # Lint
   ruff check src/ tests/

   # Type check
   mypy src/orf_entropy/

   # Test
   pytest -k "not integration" -v --cov=orf_entropy

Release Process
---------------

1. Update Version
^^^^^^^^^^^^^^^^^

Edit ``pyproject.toml`` and ``src/orf_entropy/__init__.py``:

.. code-block:: python

   __version__ = "0.2.0"

2. Update Changelog
^^^^^^^^^^^^^^^^^^^

Add release notes to ``CHANGELOG.md`` (create if needed).

3. Create Release
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Tag release
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0

   # Create GitHub release with notes

Common Tasks
------------

Adding a New Encoder
^^^^^^^^^^^^^^^^^^^^

1. Create encoder class in ``encode3di/``
2. Implement interface matching ``ProstT5ThreeDiEncoder``
3. Add tests in ``tests/test_encoder_methods.py``
4. Update documentation
5. Add CLI option to select encoder

Adding a New Genetic Code
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``pygenetic_code`` library handles this. Just use the NCBI table ID.

Optimizing Performance
^^^^^^^^^^^^^^^^^^^^^^

1. Profile to find bottlenecks
2. Consider:
   * Batching improvements
   * Memory optimization
   * GPU utilization
   * Parallel processing
3. Benchmark before and after
4. Document performance improvements

Resources
---------

* **Python Style**: PEP 8, PEP 257
* **Type Hints**: PEP 484, PEP 526
* **Testing**: pytest documentation
* **Git**: Git Flow workflow
* **Documentation**: Sphinx, reStructuredText

Getting Help
------------

* **Issues**: https://github.com/linsalrob/orf_entropy/issues
* **Discussions**: GitHub Discussions
* **Email**: raedwards@gmail.com

Next Steps
----------

* Read the :doc:`api` reference
* Check :doc:`cli` for command details
* See :doc:`user_guide` for pipeline concepts
