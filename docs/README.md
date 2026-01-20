# Documentation

This directory contains the Sphinx documentation for **genome_entropy** (dna23di).

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

Or install with the package:

```bash
pip install -e ".[docs]"
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated HTML will be in `build/html/`. Open `build/html/index.html` in a browser to view.

### Other Formats

```bash
make epub      # Build EPUB
make latexpdf  # Build PDF (requires LaTeX)
make linkcheck # Check all external links
```

### Clean Build

```bash
make clean
```

## Documentation Structure

- `source/conf.py` - Sphinx configuration
- `source/index.rst` - Main documentation index
- `source/installation.rst` - Installation guide
- `source/quickstart.rst` - Quick start guide
- `source/cli.rst` - CLI reference
- `source/api.rst` - Python API reference
- `source/user_guide.rst` - Comprehensive user guide
- `source/development.rst` - Developer guide
- `source/changelog.rst` - Version history
- `source/token_estimation.md` - Token estimation guide

## Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch via the `.github/workflows/docs.yml` GitHub Action.

View the live documentation at: https://linsalrob.github.io/genome_entropy/

## ReadTheDocs Integration

This project is configured for ReadTheDocs with `.readthedocs.yaml`. To enable:

1. Import the project at https://readthedocs.org/
2. The configuration will automatically use the settings in `.readthedocs.yaml`
3. Documentation will be built on every push

## Contributing to Documentation

When adding new features:

1. Update relevant `.rst` files in `source/`
2. Add docstrings to Python code (Google style)
3. Build locally to check for errors: `make html`
4. Commit changes and push to trigger automatic deployment

## Troubleshooting

### Missing Dependencies

If you get import errors, install the package dependencies:

```bash
pip install -e ..
```

### Build Warnings

Some warnings are expected (e.g., duplicate object descriptions with autosummary). Build errors will prevent HTML generation and should be fixed.

### Link Check Failures

External links may fail due to network issues or temporary outages. Use `make linkcheck` to verify all links periodically.
