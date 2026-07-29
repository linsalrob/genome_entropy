# Documentation

The maintained Sphinx sources are in `docs/source/`. Install the documentation
extra and build with warnings treated as errors:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs/source docs/build/html
```

Run `make -C docs linkcheck` to check external links. Network failures can make
that check transient; inspect each failure rather than suppressing it.

The documentation workflow publishes GitHub Pages at
<https://linsalrob.github.io/genome_entropy/>. Read the Docs also builds the
same sources as the `genome-entropy` project at
<https://genome-entropy.readthedocs.io/en/latest/>. Both sites are intentional;
`orf-entropy` is not this project's documentation name.

New pages must be reachable from `docs/source/index.rst`. Keep examples small
and offline where practical; model-download integration examples must be
clearly labelled and must not become mandatory documentation tests.
