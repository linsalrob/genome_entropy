# ReadTheDocs Setup Instructions

This document provides instructions for setting up ReadTheDocs for the genome_entropy project.

## What Has Been Implemented

### 1. Comprehensive Sphinx Documentation

The project now has complete Sphinx-based documentation in the `docs/` directory:

#### Documentation Pages
- **index.rst** - Main documentation landing page
- **installation.rst** - Complete installation guide with prerequisites
- **quickstart.rst** - Quick start tutorial with examples
- **cli.rst** - Comprehensive CLI command reference
- **api.rst** - Python API documentation with autodoc
- **user_guide.rst** - Detailed user guide explaining concepts
- **development.rst** - Developer guide for contributors
- **changelog.rst** - Version history and release notes
- **token_estimation.md** - Token size estimation guide (existing)

#### Documentation Features
- Full API reference with autodoc extraction from docstrings
- Code examples in all pages
- Cross-references between pages
- Search functionality
- Multiple output formats (HTML, PDF, EPUB)
- Mobile-responsive theme (Read the Docs theme)

### 2. Build System

#### Local Build
- `docs/Makefile` for easy local building
- `docs/requirements.txt` with all dependencies
- Support for multiple output formats

Build command:
```bash
cd docs && make html
```

#### Dependencies
All Sphinx dependencies are defined in:
- `pyproject.toml` under `[project.optional-dependencies]` → `docs`
- `docs/requirements.txt` for standalone installation

Required packages:
- sphinx >= 7.0.0
- sphinx-rtd-theme >= 2.0.0
- myst-parser >= 2.0.0 (for Markdown support)
- linkify-it-py >= 2.0.0 (for automatic link detection)

### 3. GitHub Pages Deployment

A GitHub Action (`.github/workflows/docs.yml`) automatically:
1. Builds documentation on every push to `main`
2. Runs link checking
3. Deploys to GitHub Pages at: https://linsalrob.github.io/genome_entropy/

The workflow:
- Triggers on push to main and pull requests
- Installs dependencies
- Builds HTML documentation
- Uploads as artifacts (retained 30 days)
- Deploys to GitHub Pages (main branch only)
- Runs linkcheck to verify external links

### 4. ReadTheDocs Configuration

The `.readthedocs.yaml` file configures ReadTheDocs to:
- Use Python 3.11
- Install the package with dependencies
- Build Sphinx documentation from `docs/source/conf.py`
- Generate PDF and EPUB formats
- Use Ubuntu 22.04 build environment

## Setting Up ReadTheDocs (Optional)

If you want to host on ReadTheDocs.org in addition to GitHub Pages:

### Step 1: Import Project

1. Go to https://readthedocs.org/
2. Sign in with your GitHub account
3. Click "Import a Project"
4. Select `linsalrob/genome_entropy`
5. Click "Next"

### Step 2: Configure Project (Auto-configured)

ReadTheDocs will automatically detect `.readthedocs.yaml` and use those settings:
- Python version: 3.11
- Documentation format: Sphinx
- Configuration file: `docs/source/conf.py`
- Output formats: HTML, PDF, EPUB

No additional configuration needed!

### Step 3: Build and Verify

1. ReadTheDocs will automatically trigger the first build
2. View build logs to ensure success
3. Visit your documentation at: `https://orf-entropy.readthedocs.io/`

### Step 4: Add Badge (Optional)

Add ReadTheDocs badge to README.md:

```markdown
[![Documentation Status](https://readthedocs.org/projects/orf-entropy/badge/?version=latest)](https://orf-entropy.readthedocs.io/en/latest/?badge=latest)
```

## Documentation Maintenance

### Updating Documentation

1. Edit `.rst` files in `docs/source/`
2. Build locally to verify: `cd docs && make html`
3. Commit and push changes
4. GitHub Action automatically rebuilds and deploys

### Adding New Pages

1. Create new `.rst` file in `docs/source/`
2. Add to appropriate `toctree` in relevant file (usually `index.rst`)
3. Build locally to verify
4. Commit and push

### Updating API Documentation

API documentation is auto-generated from docstrings:
1. Update docstrings in Python code (Google style)
2. Rebuild documentation: `cd docs && make html`
3. Review generated API docs in `docs/build/html/api.html`

### Checking Links

```bash
cd docs
make linkcheck
```

This verifies all external links are valid.

## Troubleshooting

### Build Fails

Check:
1. All dependencies installed: `pip install -e ".[docs]"`
2. Package can be imported: `python -c "import genome_entropy"`
3. Sphinx configuration is valid: `python docs/source/conf.py`

### Missing Dependencies

Install all dependencies:
```bash
pip install -e .
pip install -e ".[docs]"
```

### ReadTheDocs Build Fails

1. Check build logs on ReadTheDocs dashboard
2. Verify `.readthedocs.yaml` is valid YAML
3. Ensure all dependencies are in `pyproject.toml`
4. Check that documentation builds locally first

### GitHub Pages Not Updating

1. Check GitHub Action logs in "Actions" tab
2. Verify GitHub Pages is enabled (Settings → Pages)
3. Ensure source is set to "gh-pages" branch
4. Check that action has write permissions

## Current Status

✅ Documentation structure complete
✅ All documentation pages written
✅ Local build working
✅ GitHub Action configured
✅ GitHub Pages deployment configured
✅ ReadTheDocs configuration file ready
✅ README updated with documentation link

## Documentation URLs

- **GitHub Pages**: https://linsalrob.github.io/genome_entropy/ (Active)
- **ReadTheDocs**: https://orf-entropy.readthedocs.io/ (Setup required)

## Next Steps for Repository Owner

1. **Enable GitHub Pages** (if not already enabled):
   - Go to repository Settings → Pages
   - Set source to "gh-pages" branch
   - Wait for first deployment from GitHub Action

2. **Optional: Setup ReadTheDocs**:
   - Import project at readthedocs.org
   - Verify automatic build
   - Add ReadTheDocs badge to README

3. **Optional: Configure Custom Domain**:
   - Add CNAME record in DNS
   - Configure in GitHub Pages settings
   - Update documentation URLs

The documentation is production-ready and will automatically deploy on the next push to main!
