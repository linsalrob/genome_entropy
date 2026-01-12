# Copilot Instructions for orf_entropy

## Repository Overview

**Purpose**: Compare and contrast the entropy of sequences, ORFs (Open Reading Frames), proteins, and 3Di encodings for bioinformatics analysis.

**Type**: Python bioinformatics library/tool  
**Language**: Python 3.8+  
**License**: MIT  
**Author**: Rob Edwards (@linsalrob)

This is a computational genomics project focused on entropy analysis of biological sequences. The repository analyzes complexity and information content in DNA sequences, open reading frames, protein sequences, and structure-based 3Di encodings.

## Repository Structure

This is a new repository. When the project is fully developed, expect this structure:

```
orf_entropy/
├── .github/              # GitHub configuration
│   ├── workflows/        # CI/CD workflows (when added)
│   └── copilot-instructions.md (this file)
├── orf_entropy/          # Main package directory
│   ├── __init__.py
│   ├── entropy.py        # Entropy calculation functions
│   ├── orf.py            # ORF detection and processing
│   └── utils.py          # Utility functions
├── tests/                # Test suite (pytest)
│   └── test_*.py
├── .gitignore            # Python gitignore template
├── LICENSE               # MIT license
├── README.md             # Project overview
├── pyproject.toml        # Project configuration (when added)
└── requirements.txt      # Dependencies (when added)
```

**Current State**: Repository contains only initial files (.gitignore, LICENSE, README.md). Source code and tests need to be added.

## Python Environment Setup

**Python Version**: Use Python 3.8 or higher.

### Initial Setup (When Dependencies Are Added)

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies** (once requirements.txt or pyproject.toml exists):
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt  # or: pip install -e .
   ```

3. **Install development dependencies** (for testing):
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

### Expected Dependencies

Based on typical bioinformatics projects, expect these dependencies:
- **Core**: biopython, numpy, scipy
- **Testing**: pytest, pytest-cov, coverage
- **Quality**: black (formatting), flake8 (linting), mypy (type checking)
- **Optional**: matplotlib (visualization), pandas (data analysis)

## Building and Testing

### Bootstrap Process

Since this is a new repository without existing build infrastructure:

1. **First time setup**:
   ```bash
   cd /path/to/orf_entropy
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

2. **If pyproject.toml exists**:
   ```bash
   pip install -e .  # Editable install
   ```

3. **If only requirements.txt exists**:
   ```bash
   pip install -r requirements.txt
   ```

### Running Tests

**ALWAYS run tests from the repository root directory.**

When tests are added, use pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=orf_entropy --cov-report=term-missing

# Run specific test file
pytest tests/test_entropy.py

# Run with verbose output
pytest -v
```

**Test Discovery**: pytest automatically discovers tests in files matching `test_*.py` or `*_test.py` in the `tests/` directory.

**Expected Test Time**: Unit tests should complete in < 10 seconds. If tests timeout, increase wait time for complex computations.

### Linting and Code Quality

When code is added, use these commands:

```bash
# Format code (modifies files in place)
black orf_entropy/ tests/

# Check code style
flake8 orf_entropy/ tests/

# Type checking
mypy orf_entropy/
```

**Code Style**: Follow PEP 8 conventions. Use black for automatic formatting with default settings (88 character line length).

### Common Issues and Solutions

**Issue**: `ModuleNotFoundError` when importing orf_entropy  
**Solution**: Install package in editable mode: `pip install -e .` from repository root.

**Issue**: Tests fail with import errors  
**Solution**: Ensure virtual environment is activated and dependencies installed. Run from repository root.

**Issue**: Permission errors on Linux/Mac  
**Solution**: Use `source venv/bin/activate` not `venv/bin/activate` directly.

## Project Layout and Architecture

### Key Files and Directories

- **/.github/**: GitHub-specific configuration including workflows and this instruction file
- **/orf_entropy/**: Main package containing all source code modules
- **/tests/**: All test files using pytest framework
- **/.gitignore**: Comprehensive Python gitignore excluding __pycache__, .venv, dist/, build/, *.pyc, etc.
- **/LICENSE**: MIT license file
- **/README.md**: Project description and quick start guide

### Configuration Files (To Be Added)

- **pyproject.toml**: Modern Python project configuration using PEP 621 standard
  - Defines project metadata, dependencies, build system
  - Configures tools: black, mypy, pytest
  - Use setuptools backend: `[build-system] requires = ["setuptools>=61.0"]`

- **requirements.txt**: Pip-installable dependencies for reproducibility
  - Pin versions for CI/CD: `numpy==1.24.0`
  - Use `pip freeze > requirements.txt` to capture exact versions

### Code Organization Principles

- **Modular design**: Separate concerns (entropy calculation, ORF detection, I/O)
- **Type hints**: Use Python type annotations for clarity and mypy checking
- **Docstrings**: Follow NumPy/Google style for all public functions
- **Testing**: Aim for >80% code coverage with meaningful tests
- **Data handling**: Use Biopython for sequence parsing (FASTA, GenBank)

## Continuous Integration (When Added)

### Expected GitHub Actions Workflows

When CI is set up, expect workflow in `.github/workflows/python-ci.yml`:

**Workflow triggers**: push, pull_request on main/master branch

**Test matrix**: Python 3.8, 3.9, 3.10, 3.11

**Steps**:
1. Checkout code
2. Set up Python version
3. Install dependencies: `pip install -r requirements.txt`
4. Run linting: `flake8 orf_entropy/`
5. Run tests: `pytest --cov=orf_entropy`
6. Upload coverage to Codecov (optional)

**Typical execution time**: 2-5 minutes per Python version

### Pre-commit Validation

Before committing code changes:

1. Run formatters: `black orf_entropy/ tests/`
2. Run linters: `flake8 orf_entropy/ tests/`
3. Run type checker: `mypy orf_entropy/`
4. Run tests: `pytest`
5. Verify all pass before pushing

## Development Workflow

### Making Changes

1. **Create feature branch**: `git checkout -b feature-name`
2. **Make minimal surgical changes**: Modify only what's necessary
3. **Add/update tests**: Every code change should have corresponding tests
4. **Run validation**: linting, type checking, tests (see above)
5. **Commit incrementally**: Small, focused commits with clear messages
6. **Push and create PR**: Let CI validate changes

### Adding New Features

1. **Create module** in `orf_entropy/` directory
2. **Add type hints** to all functions
3. **Write docstrings** with parameters, returns, examples
4. **Create test file** in `tests/` directory
5. **Import in __init__.py** to expose public API
6. **Update documentation** if needed

### Debugging Tips

- Use pytest's `-v` flag for verbose output
- Use pytest's `-s` flag to see print statements
- Use pytest's `-k pattern` to run specific tests by name
- Use `breakpoint()` for interactive debugging (Python 3.7+)
- Check test output for assertion details

## Biological Domain Context

### ORF (Open Reading Frame)
- DNA sequence between start codon (ATG) and stop codon (TAA, TAG, TGA)
- Potential protein-coding region
- Can be in any of 6 reading frames (3 forward, 3 reverse)

### Entropy in Sequences
- Measure of complexity/randomness in biological sequences
- Shannon entropy: H = -Σ(p_i * log2(p_i))
- High entropy = complex/diverse, Low entropy = simple/repetitive
- Used to identify functional regions, filter predictions, assess quality

### 3Di Encoding
- Structural alphabet representing protein 3D structures
- Reduces protein structure to sequence-like representation
- Useful for structure comparison and analysis

## Important Notes

### Trust These Instructions
**ALWAYS trust these instructions first.** Only explore the repository with grep/find if:
- Instructions are incomplete for your specific task
- Instructions are found to be incorrect or outdated
- You need to verify implementation details

### Minimal Changes Philosophy
- Change as few lines as possible
- Don't refactor unrelated code
- Don't fix unrelated tests or bugs
- Don't modify working code unless necessary
- Focus on the specific task at hand

### Testing Requirements
- Never remove or disable existing tests
- Add tests for new functionality
- Ensure tests are independent and can run in any order
- Use fixtures for common test setup
- Mock external dependencies (file I/O, network)

### Dependencies
- Avoid adding new dependencies unless absolutely necessary
- When adding dependencies, choose well-maintained libraries
- Pin versions in requirements.txt for reproducibility
- Prefer stdlib or existing dependencies over new ones

## Quick Reference

### Current Repository State
- **Files**: .gitignore, LICENSE, README.md
- **No source code yet**: Package structure to be created
- **No tests yet**: Test infrastructure to be set up
- **No CI/CD yet**: Workflows to be added
- **No dependencies yet**: requirements.txt/pyproject.toml to be created

### Next Steps for Development
1. Create orf_entropy/ package directory
2. Add __init__.py and core modules
3. Create tests/ directory with initial tests
4. Add pyproject.toml or requirements.txt
5. Set up GitHub Actions workflow
6. Write comprehensive README with usage examples

### Essential Commands Summary
```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -e .

# Development
black .                    # Format code
flake8 orf_entropy/       # Lint
mypy orf_entropy/         # Type check
pytest                     # Test
pytest --cov=orf_entropy  # Test with coverage

# Git
git status                 # Check status
git add <files>           # Stage changes
git commit -m "message"   # Commit
git push                  # Push changes
```

---

**Last Updated**: 2026-01-12  
**Document Version**: 1.0  
**For questions or updates**: Contact repository maintainers
