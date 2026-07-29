# Documentation deployment

The repository publishes the same Sphinx source through two services:

- GitHub Pages: <https://linsalrob.github.io/genome_entropy/>
- Read the Docs project `genome-entropy`: <https://genome-entropy.readthedocs.io/>

`.readthedocs.yaml` selects Python 3.11, installs the `docs` extra, uses `docs/source/conf.py`, and treats warnings as errors. `.github/workflows/docs.yml` builds the same source and deploys `docs/build/html` to GitHub Pages from `main`.

Validate locally before pushing:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

The README badge and target must both use the `genome-entropy` Read the Docs project name. Generated HTML, PDF, and EPUB files are build artefacts and must not be committed.
