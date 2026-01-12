# Copilot Instructions for orf_entropy

## Repository Overview

**Purpose**: Compare and contrast the entropy of sequences, ORFs (Open Reading Frames), proteins, and 3Di encodings for bioinformatics analysis.  
**Type**: Python 3.8+ bioinformatics library | **License**: MIT | **Author**: Rob Edwards (@linsalrob)

**Current State**: New repository with only .gitignore, LICENSE, README.md. Source code and tests not yet added.

**Expected Structure**:
```
orf_entropy/           # Main package (entropy.py, orf.py, utils.py)
tests/                 # pytest tests
.github/workflows/     # CI/CD (when added)
pyproject.toml         # Project config (when added)
requirements.txt       # Dependencies (when added)
```

## Setup and Building

**Python Version**: 3.8+

### Bootstrap (First Time)
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -e .          # When pyproject.toml exists
# OR: pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy  # Dev dependencies
```

**Expected Dependencies**: biopython, numpy, scipy, pytest, black, flake8, mypy

### Running Tests
**ALWAYS run from repository root.**
```bash
pytest                           # All tests
pytest --cov=orf_entropy        # With coverage
pytest tests/test_entropy.py    # Specific file
pytest -v                        # Verbose
```
**Expected time**: < 10 seconds for unit tests.

### Linting and Formatting
```bash
black orf_entropy/ tests/       # Format (modifies files)
flake8 orf_entropy/ tests/      # Lint (check style)
mypy orf_entropy/               # Type check
```
**Style**: PEP 8, black defaults (88 char line length).

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` importing orf_entropy | Run `pip install -e .` from repo root |
| Tests fail with import errors | Activate venv, install dependencies, run from root |
| Permission errors (Linux/Mac) | Use `source venv/bin/activate` |

## CI/CD and Validation

**Expected GitHub Actions** (when added): `.github/workflows/python-ci.yml`
- **Triggers**: push, pull_request on main
- **Matrix**: Python 3.8, 3.9, 3.10, 3.11
- **Steps**: checkout → setup Python → install deps → flake8 → pytest with coverage
- **Time**: 2-5 minutes per version

**Pre-commit checklist**:
1. Format: `black orf_entropy/ tests/`
2. Lint: `flake8 orf_entropy/ tests/`
3. Type check: `mypy orf_entropy/`
4. Test: `pytest`
5. All pass → commit and push

## Project Architecture

**Key Directories**:
- `.github/`: Workflows and this file
- `orf_entropy/`: Main package (entropy.py, orf.py, utils.py)
- `tests/`: pytest tests (test_*.py pattern)

**Configuration** (when added):
- `pyproject.toml`: Use setuptools backend `[build-system] requires = ["setuptools>=61.0"]`
- `requirements.txt`: Pin versions for reproducibility

**Code Principles**:
- Modular design, type hints, NumPy/Google docstrings
- Target >80% test coverage
- Use Biopython for sequence I/O (FASTA, GenBank)

## Development Workflow

**Making Changes**:
1. Branch: `git checkout -b feature-name`
2. Make minimal surgical changes
3. Add/update tests for all code changes
4. Validate: format → lint → type check → test
5. Commit small, focused changes
6. Push and create PR

**Adding Features**:
1. Create module in `orf_entropy/`
2. Add type hints and docstrings
3. Create `tests/test_<module>.py`
4. Import in `__init__.py`
5. Update docs if needed

**Debug Tips**: Use `pytest -v` (verbose), `-s` (print), `-k pattern` (filter), `breakpoint()` (debug)

## Domain Knowledge

**ORF (Open Reading Frame)**: DNA between start codon (ATG) and stop codon (TAA/TAG/TGA). Potential protein-coding region in 6 reading frames (3 forward, 3 reverse).

**Entropy**: Measures sequence complexity. Shannon entropy: H = -Σ(p_i × log2(p_i)). High = complex/diverse, Low = simple/repetitive. Used to identify functional regions and assess quality.

**3Di Encoding**: Structural alphabet for protein 3D structures. Reduces structure to sequence-like representation for comparison.

## Key Guidelines

**Trust these instructions first.** Only search the repo if instructions are incomplete, incorrect, or you need implementation details.

**Minimal changes**:
- Change as few lines as possible
- Don't refactor unrelated code
- Don't fix unrelated bugs/tests
- Focus on specific task

**Testing**:
- Never remove/disable existing tests
- Add tests for new functionality
- Tests must be independent
- Mock external dependencies

**Dependencies**:
- Avoid adding unless necessary
- Choose well-maintained libraries
- Pin versions in requirements.txt
- Prefer stdlib or existing deps

## Quick Reference

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -e .

# Development
black .                    # Format
flake8 orf_entropy/       # Lint
mypy orf_entropy/         # Type check
pytest --cov=orf_entropy  # Test

# Git
git status && git add <files> && git commit -m "msg" && git push
```

**Current State**: Only .gitignore, LICENSE, README.md exist. Package, tests, and CI need creation.

---
*Last Updated: 2026-01-12 | Version 1.0*
