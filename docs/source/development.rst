Development guide
=================

Environment
-----------

.. code-block:: bash

   git clone https://github.com/linsalrob/genome_entropy.git
   cd genome_entropy
   python -m venv .venv
   . .venv/bin/activate
   python -m pip install -e ".[dev,docs,ml]"

Install ``get_orfs`` separately when exercising DNA workflows. Real encoder
integration tests also need model downloads and suitable compute; unit tests use
mocks and should not download checkpoints.

Source layout
-------------

``src/genome_entropy`` contains configuration, CLI, encoders, entropy, I/O, ML,
ORF, pipeline, and translation packages. ``tests`` mirrors these areas.
``docs/source`` is the Sphinx source. ``examples`` contains statically checked
examples, and ``slurm`` contains site-specific templates.

Quality commands
----------------

The package and CI minimum is Python 3.10. Black, Ruff, and mypy are configured
in ``pyproject.toml``.

.. code-block:: bash

   black --check src tests
   ruff check src tests
   mypy src/genome_entropy
   pytest -k "not integration"
   python -m sphinx -W --keep-going -b html docs/source docs/_build/html

Run ``pytest`` for the complete locally enabled suite. Tests marked integration
may still be skipped unless their documented environment variables and model
cache are available. The GitHub unit-test workflow currently runs lint, format,
and type commands with ``|| true``; local contributors should treat their
findings as real validation results even though existing CI does not gate on
them.

Documentation
-------------

Use Google-style docstrings for public Python objects. Describe parameters and
return types, defaults, exceptions, missing-value semantics, coordinates, and
side effects. Do not duplicate long CLI option lists in multiple pages: generate
``--help`` from Typer and keep :doc:`cli` aligned with it.

The documentation workflow and Read the Docs configuration build Sphinx with
warnings treated as errors. Generated HTML belongs under ``docs/_build`` or
``docs/build`` and must not be committed.

Integration tests
-----------------

``tests/test_prostt5_integration.py`` uses ``RUN_INTEGRATION=1``. The current
ModernProst 50M smoke test uses ``RUN_HUGGINGFACE_INTEGRATION_TESTS=1``. These
tests download or load large external artefacts and are not part of ordinary CI.

Release metadata
----------------

``pyproject.toml`` is the authoritative package version source; package
``__version__`` is read from installed metadata. Releases trigger the PyPI
workflow. Update :doc:`changelog` without inventing release dates, build the
distributions, and create a reviewed GitHub release according to maintainer
policy. Do not edit generated ``egg-info`` files manually.

Contributing safely
-------------------

Keep changes focused, add tests for behaviour, and preserve backward-compatible
JSON handling. Never commit model caches, environments, credentials, sensitive
sequence data, scheduler logs, or generated documentation. Report issues at
https://github.com/linsalrob/genome_entropy/issues.
