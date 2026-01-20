# Copilot Instructions for genome_entropy (dna23di pipeline)

## Repository Overview

**Purpose**: Complete bioinformatics pipeline that converts DNA sequences → ORFs → proteins → 3Di structural tokens, computing Shannon entropy at each representation level.  
**Type**: Python 3.8+ bioinformatics library with CLI | **License**: MIT | **Author**: Rob Edwards (@linsalrob)

**Project Name**: Repository is `genome_entropy`, but CLI tool is `dna23di` (reflects DNA→3Di transformation pipeline).

**Current State**: Initial repository with basic structure. Full pipeline implementation needed.

**Complete Structure** (to be implemented):
```
dna23di/
  src/
    dna23di/
      __init__.py
      config.py              # Configuration and constants
      errors.py              # Custom exceptions
      io/
        __init__.py
        fasta.py            # FASTA reading/writing
        jsonio.py           # JSON serialization for data models
      orf/
        __init__.py
        types.py            # OrfRecord dataclass
        finder.py           # ORF finding using get_orfs
      translate/
        __init__.py
        translator.py       # Translation using genetic_codes
      encode3di/
        __init__.py
        prostt5.py          # ProstT5 encoder (AA→3Di)
        batching.py         # Batch processing utilities
      entropy/
        __init__.py
        shannon.py          # Shannon entropy calculation
      pipeline/
        __init__.py
        runner.py           # End-to-end pipeline orchestration
      cli/
        __init__.py
        main.py             # CLI entry point (typer-based)
        commands/
          __init__.py
          download.py       # Download models/datasets
          orf.py            # ORF extraction command
          translate.py      # Translation command
          encode3di.py      # 3Di encoding command
          entropy.py        # Entropy calculation command
          run.py            # End-to-end pipeline command
  tests/
    conftest.py             # Pytest fixtures and config
    test_orf_finder.py      # ORF finding tests
    test_translation.py     # Translation tests
    test_entropy.py         # Entropy calculation tests
    test_pipeline_unit.py   # Pipeline tests (mocked)
    test_cli_smoke.py       # CLI smoke tests
    test_prostt5_integration.py  # Integration tests (skipped by default)
  examples/
    example_small.fasta     # Small synthetic test sequences
    run_end_to_end.sh       # Example pipeline execution
  .github/workflows/
    python-ci.yml           # CI/CD pipeline
  pyproject.toml            # Project configuration
  README.md                 # User documentation
  LICENSE
```

## Core Data Models

All data models use Python `dataclasses` for simplicity and must be JSON-serializable.

### `OrfRecord` (orf/types.py)
```python
@dataclass
class OrfRecord:
    parent_id: str          # Source sequence ID
    orf_id: str             # Unique ORF identifier
    start: int              # 0-based, inclusive
    end: int                # 0-based, exclusive
    strand: Literal["+","-"]
    frame: int              # 0, 1, 2 (relative to forward strand)
    nt_sequence: str        # Nucleotide sequence
    table_id: int           # NCBI translation table used
    has_start_codon: bool
    has_stop_codon: bool
```

### `ProteinRecord` (translate/translator.py)
```python
@dataclass
class ProteinRecord:
    orf: OrfRecord
    aa_sequence: str        # Amino acid sequence
    aa_length: int
```

### `ThreeDiRecord` (encode3di/prostt5.py)
```python
@dataclass
class ThreeDiRecord:
    protein: ProteinRecord
    three_di: str           # 3Di token sequence
    method: Literal["prostt5_aa2fold"]
    model_name: str
    inference_device: str   # "cuda", "mps", or "cpu"
```

### `EntropyReport` (entropy/shannon.py)
```python
@dataclass
class EntropyReport:
    dna_entropy_global: float
    orf_nt_entropy: dict[str, float]     # orf_id → entropy
    protein_aa_entropy: dict[str, float]
    three_di_entropy: dict[str, float]
    alphabet_sizes: dict[str, int]
```

### `PipelineResult` (pipeline/runner.py)
```python
@dataclass
class PipelineResult:
    input_id: str
    input_dna_length: int
    orfs: list[OrfRecord]
    proteins: list[ProteinRecord]
    three_dis: list[ThreeDiRecord]
    entropy: EntropyReport
```

## Setup and Building

**Python Version**: 3.8+

### Bootstrap (First Time)
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -e .
pip install -e ".[dev]"   # Install dev dependencies
```

**Core Dependencies**:
- `torch` (PyTorch) - GPU-aware ML framework
- `transformers` - Hugging Face for ProstT5 model loading
- `pygenetic-code` - Translation using NCBI genetic codes (from linsalrob/genetic_codes)
- `typer` - Modern CLI framework
- `get_orfs` (external binary) - Fast ORF finding (from linsalrob/get_orfs)

**Dev Dependencies**: pytest, pytest-cov, pytest-mock, black, ruff, mypy

### External Binary: get_orfs

The ORF finder uses an external binary from https://github.com/linsalrob/get_orfs

**Installation** (required before running ORF commands):
```bash
# Clone and build get_orfs
git clone https://github.com/linsalrob/get_orfs.git /tmp/get_orfs
cd /tmp/get_orfs
mkdir build && cd build
cmake ..
make
cmake --install . --prefix ..
# Binary is now at /tmp/get_orfs/bin/get_orfs
```

**Integration**: `orf/finder.py` wraps `get_orfs` binary via subprocess. User must have binary in PATH or specify location via config.

### Running Tests

**ALWAYS run from repository root.**

**Unit Tests** (fast, models mocked):
```bash
pytest                           # All tests
pytest --cov=genome_entropy        # With coverage
pytest tests/test_entropy.py    # Specific file
pytest -v                        # Verbose
pytest -k "not integration"     # Skip integration tests
```

**Integration Tests** (slow, downloads real models):
```bash
RUN_INTEGRATION=1 pytest -v -m integration
```

**Test Organization**:
- Unit tests mock ProstT5 model outputs for speed
- Integration tests marked with `@pytest.mark.integration` and skipped by default
- Fixtures in `conftest.py` provide synthetic DNA sequences and expected ORFs
- Expected time: <10 seconds for unit tests, 1-5 minutes for integration tests (model download + inference)

### Linting and Formatting
```bash
black src/ tests/               # Format (modifies files)
ruff check src/ tests/          # Lint (fast, replaces flake8)
mypy src/genome_entropy/           # Type check
```

**Style**: PEP 8, black defaults (88 char line length), full type hints required.

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` importing genome_entropy | Run `pip install -e .` from repo root |
| Tests fail with import errors | Activate venv, install dependencies including `[dev]` extras |
| `get_orfs` binary not found | Install get_orfs from GitHub and add to PATH or configure location |
| CUDA errors on CPU-only machine | Code auto-detects device; check torch installation |
| ProstT5 model download fails | Check internet connection, HuggingFace cache permissions |
| Integration tests run unexpectedly | Use `pytest -k "not integration"` or unset RUN_INTEGRATION |
| Type errors in mypy | Ensure all functions have type hints, import from `typing` |

## GPU and Device Handling

**Critical**: Code must auto-detect GPU type and handle gracefully.

**Device Selection Strategy** (in `encode3di/prostt5.py`):
```python
def select_device() -> str:
    """Auto-select best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"
```

**Dtype Selection**:
- CUDA: Try bf16 if supported, else fp16, else fp32
- MPS/CPU: fp32 only
- Must wrap inference in try/except and fallback on dtype errors

**Testing**: Mock torch.cuda.is_available() in tests to simulate different devices.

## CI/CD and Validation

**GitHub Actions**: `.github/workflows/python-ci.yml`
- **Triggers**: push, pull_request on main
- **Matrix**: Python 3.8, 3.9, 3.10, 3.11, 3.12
- **Steps**:
  1. Checkout code
  2. Setup Python
  3. Install dependencies (without get_orfs binary for CI speed)
  4. Run ruff linting
  5. Run mypy type checking
  6. Run pytest (unit tests only, integration skipped)
  7. Upload coverage to Codecov (optional)
- **Time**: 3-7 minutes per Python version
- **Note**: Integration tests NOT run in CI to avoid model downloads

**Pre-commit Checklist**:
1. Format: `black src/ tests/`
2. Lint: `ruff check src/ tests/`
3. Type check: `mypy src/genome_entropy/`
4. Unit tests: `pytest -k "not integration"`
5. All pass → commit and push

**Manual Integration Testing** (before major releases):
```bash
RUN_INTEGRATION=1 pytest -v -m integration
```

## Project Architecture

**Key Directories**:
- `.github/`: Workflows and this instruction file
- `src/genome_entropy/`: Main package source code
  - `io/`: FASTA and JSON I/O
  - `orf/`: ORF finding and types
  - `translate/`: Protein translation
  - `encode3di/`: 3Di encoding via ProstT5
  - `entropy/`: Shannon entropy calculation
  - `pipeline/`: End-to-end orchestration
  - `cli/`: Command-line interface
- `tests/`: pytest tests (test_*.py pattern)
- `examples/`: Runnable examples and sample data

**Configuration**:
- `pyproject.toml`: Modern Python project config (PEP 621)
  - Build system: setuptools
  - Dependencies: torch, transformers, pygenetic-code, typer
  - Dev dependencies: pytest, pytest-cov, pytest-mock, black, ruff, mypy
  - CLI entry point: `[project.scripts] dna23di = "genome_entropy.cli.main:app"`
- No requirements.txt (use pyproject.toml only for modern packaging)

**Code Principles**:
- **Modular design**: Each subpackage has clear responsibility
- **Type hints**: All functions fully typed (Python 3.8+ syntax)
- **Docstrings**: Google style for all public APIs
- **Error handling**: Custom exceptions in `errors.py`
- **Testing**: >80% coverage target, mock external dependencies
- **Performance**: Python-first, optional Rust extensions for hotspots (future)

**Data Flow**:
```
DNA FASTA → ORF Finder → ORFs (nt) → Translator → Proteins (aa) 
          → ProstT5 Encoder → 3Di tokens → Entropy Calculator → Results JSON
```

## CLI Commands (dna23di)

**Entry Point**: `dna23di` (installed via pip, defined in pyproject.toml)

**Subcommands**:

### `dna23di download`
Pre-download models, tokenizers, and optional test datasets.
```bash
dna23di download [--model MODEL_NAME] [--test-data]
```
- Downloads ProstT5 model from HuggingFace to cache
- Optionally downloads small reference datasets (<5MB) with SHA256 checksums
- Prints cache locations

### `dna23di orf`
Extract ORFs from DNA sequences.
```bash
dna23di orf --input in.fasta --output orfs.json [--table 11] [--min-nt 90]
```
- Uses `get_orfs` binary (must be in PATH or configured)
- Outputs JSON with ORF records

### `dna23di translate`
Translate ORFs to proteins.
```bash
dna23di translate --input orfs.json --output proteins.json [--table 11]
```
- Uses `pygenetic-code` for translation
- Handles ambiguous codons (→ X)

### `dna23di encode3di`
Encode proteins to 3Di structural tokens.
```bash
dna23di encode3di --input proteins.json --output 3di.json [--model MODEL] [--device auto|cuda|cpu]
```
- Uses ProstT5 for AA→3Di prediction
- Auto-detects device if `--device auto`
- Supports batching for efficiency

### `dna23di entropy`
Calculate Shannon entropy at all levels.
```bash
dna23di entropy --input 3di.json --output entropy.json [--normalize]
```
- Computes DNA, ORF, protein, and 3Di entropies
- Optional normalization by alphabet size

### `dna23di run`
End-to-end pipeline (all steps combined).
```bash
dna23di run --input in.fasta --output results.json [--table 11] [--min-aa 30] [--model MODEL]
```
- Runs: FASTA → ORF → translate → 3Di → entropy
- Single JSON output with `PipelineResult` objects
- Shows progress bar

**Exit Codes**:
- 0: Success
- 1: General error
- 2: User error (bad arguments, missing file)
- 3: Runtime error (model failure, GPU error)

## Domain Knowledge

### ORF (Open Reading Frame)
DNA sequence between start codon (typically ATG) and stop codon (TAA/TAG/TGA). Potential protein-coding region.

**Reading Frames**: 6 total (3 forward, 3 reverse complement)
- Frame 0: Start at position 0
- Frame 1: Start at position 1  
- Frame 2: Start at position 2
- Each frame on + and - strands

**ORF Finding**: Uses `get_orfs` binary (custom implementation allowing genetic code changes)
- Identifies all valid ORFs in all 6 frames
- Tracks start/stop codon presence
- Outputs 0-based coordinates (Python convention)

### Genetic Code Translation
Uses NCBI translation tables via `pygenetic-code` library.

**Common Tables**:
- Table 1: Standard genetic code
- Table 11: Bacterial, archaeal, and plant plastid code

**Ambiguous Codons**: Codons with N or other IUPAC ambiguous bases translate to X (unknown amino acid) by default.

### 3Di Structural Alphabet
A discrete alphabet of 20 structural states representing local 3D geometry of protein backbones. Developed for Foldseek structural search.

**Two Routes to 3Di**:
1. **Structure→3Di**: Foldseek derives 3Di from 3D coordinates (PDB files)
2. **Sequence→3Di** (this project): ProstT5 predicts 3Di directly from amino acid sequences (AA→3Di)

**ProstT5**: Transformer model trained to predict 3Di and secondary structure from protein sequences. We use the AA→3Di prediction capability.

**Reference**: Heinzinger et al. (2023), "ProstT5: Bilingual Language Model for Protein Sequence and Structure" ([OUP Academic](https://academic.oup.com/))

### Shannon Entropy
Measures information content / complexity of a sequence.

**Formula**: H = -Σ(p_i × log₂(p_i))
- p_i = frequency of symbol i
- Higher entropy = more complex/diverse
- Lower entropy = more repetitive/simple

**Normalization**: H_norm = H / log₂(|alphabet|)
- Scales entropy to [0, 1] range
- Allows comparison across different alphabets

**Applications**:
- Identify low-complexity regions in proteins
- Assess sequence quality (high entropy = good)
- Compare structural vs sequence complexity

## Testing Strategy

### Unit Tests (Fast, Always Run)
**Location**: `tests/test_*.py` (except integration tests)
**Approach**: Mock all external dependencies
- **ORF Finder**: Use synthetic DNA with known ORFs
- **Translation**: Test known codon→AA mappings, ambiguous codon handling
- **ProstT5**: Mock encoder to return deterministic 3Di strings
- **Entropy**: Test with toy strings with known entropy values
- **Pipeline**: Mock encoder, test data flow and JSON output

**Fixtures** (in `conftest.py`):
```python
@pytest.fixture
def synthetic_dna():
    """DNA with known ORFs on both strands."""
    return ">test_seq\nATGGCATAGTGA..."

@pytest.fixture  
def mock_prostt5_encoder(monkeypatch):
    """Mock ProstT5 encoder returning fake 3Di."""
    # Returns 'A' * len(aa_seq) for deterministic testing
```

### Integration Tests (Slow, Optional)
**Location**: `tests/test_prostt5_integration.py`
**Marker**: `@pytest.mark.integration`
**Run**: Only when `RUN_INTEGRATION=1` environment variable set

**Purpose**:
- Download and load real ProstT5 model
- Encode 1-2 short proteins (5-20 AA)
- Verify output format and alphabet correctness
- Test device selection (CUDA/MPS/CPU)

**Example**:
```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Integration tests disabled")
def test_prostt5_real_inference():
    encoder = ProstT5ThreeDiEncoder()
    result = encoder.encode(["ACDEFGHIKLMNPQRSTVWY"])
    assert len(result) == 1
    assert len(result[0]) == 20  # Same length as input
    assert all(c in THREEDDI_ALPHABET for c in result[0])
```

### Control Datasets
**Purpose**: Validate entropy differences and ORF behavior with controlled perturbations.

**Built-in Synthetic**:
- Hand-crafted DNA with specific ORF patterns
- Sequences with ambiguous bases (N)
- Edge cases: no stop, no start, overlapping ORFs

**Generated Controls** (in `examples/`):
1. **Nucleotide-shuffled**: Preserves mono-nucleotide composition
2. **Codon-shuffled**: Preserves codon composition (AA distribution preserved)
3. **Frame-shifted**: Shift by +1 or +2 to disrupt ORFs

**External Integration** (optional, downloaded):
- Small bacterial gene region (<5MB FASTA)
- SHA256 checksum verification
- Used only in integration tests and examples

### Test Coverage Target
- **Overall**: >80%
- **Critical modules** (entropy, translation): >90%
- **CLI**: Smoke tests (help, basic run with mocks)
- **Integration**: Optional, not counted in coverage metrics

## Implementation Details

### ORF Finding (orf/finder.py)
**External Binary**: Uses `get_orfs` from https://github.com/linsalrob/get_orfs
- **Why**: Allows custom genetic code tables, better performance than Biopython
- **Integration**: Subprocess wrapper in Python
- **Input**: FASTA file or string
- **Output**: Parse stdout to `OrfRecord` objects
- **Configuration**: Binary path configurable via env var or config file

**Build get_orfs**:
```bash
git clone https://github.com/linsalrob/get_orfs.git
cd get_orfs && mkdir build && cd build
cmake .. && make
cmake --install . --prefix ..
# Binary: ./bin/get_orfs
```

### Translation (translate/translator.py)
**Library**: `pygenetic-code` from https://github.com/linsalrob/genetic_codes
- **Installation**: `pip install pygenetic-code`
- **Usage**: Provides NCBI translation tables
- **Ambiguous Handling**: Configurable (default: ambiguous codon → X)

**Example**:
```python
from genetic_codes import GeneticCode
gc = GeneticCode.from_ncbi_id(11)  # Bacterial code
aa = gc.translate("ATGGCATAA")  # "MA*"
```

### 3Di Encoding (encode3di/prostt5.py)
**Model**: ProstT5 from HuggingFace Transformers
- **Default Model**: `Rostlab/ProstT5` or similar checkpoint
- **Method**: AA→3Di prediction (no 3D structure required)
- **Framework**: PyTorch + Transformers

**Key Class**:
```python
class ProstT5ThreeDiEncoder:
    def __init__(self, model_name: str, device: str | None = None):
        self.device = self._select_device(device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def encode(self, aa_sequences: list[str], batch_size: int = 4) -> list[str]:
        """Encode amino acid sequences to 3Di tokens."""
        # Batch processing, truncation warnings, 3Di extraction
```

**Device Selection** (critical):
```python
def _select_device(self, device_hint: str | None) -> str:
    if device_hint and device_hint != "auto":
        return device_hint
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

**Dtype Strategy**:
- CUDA: Try `torch.bfloat16` if supported, else `torch.float16`, fallback to `torch.float32`
- MPS/CPU: `torch.float32` only
- Wrap in try/except for graceful fallback

### Shannon Entropy (entropy/shannon.py)
**Implementation**:
```python
def shannon_entropy(s: str, alphabet: set[str] | None = None, normalize: bool = False) -> float:
    """Calculate Shannon entropy of string s."""
    if not s:
        return 0.0
    
    # Count frequencies
    counts = Counter(s)
    total = len(s)
    
    # Calculate entropy
    entropy = -sum((count/total) * math.log2(count/total) for count in counts.values())
    
    # Normalize if requested
    if normalize and alphabet:
        max_entropy = math.log2(len(alphabet))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    return entropy
```

**Special Cases**:
- Empty string → 0
- Single symbol → 0
- DNA: Canonicalize to uppercase before calculation
- 3Di: Handle space-delimited tokens if ProstT5 outputs them

### Pipeline (pipeline/runner.py)
**Orchestration**:
```python
def run_pipeline(
    input_fasta: Path,
    table_id: int = 11,
    min_aa_len: int = 30,
    model_name: str = "Rostlab/ProstT5",
    compute_entropy: bool = True,
    output_json: Path | None = None,
) -> list[PipelineResult]:
    # 1. Read FASTA
    # 2. Find ORFs (filter by min_aa_len * 3)
    # 3. Translate ORFs
    # 4. Encode to 3Di (batch for efficiency)
    # 5. Calculate entropies
    # 6. Build PipelineResult objects
    # 7. Write JSON
```

**Progress Tracking**: Use `tqdm` for user feedback in CLI mode.

## Development Workflow

### Making Changes
1. **Branch**: `git checkout -b feature-name`
2. **Implement**: Make minimal, surgical changes
3. **Test**: Add/update tests for changed code
4. **Validate**: Run pre-commit checks (format → lint → type → test)
5. **Commit**: Small, focused commits with clear messages
6. **Push**: Create PR with description

### Adding New Features
1. **Design**: Define data models and interfaces first
2. **Core Logic**: Implement in appropriate subpackage
3. **Tests**: Unit tests with mocks, integration tests if needed
4. **CLI**: Add command in `cli/commands/` if user-facing
5. **Docs**: Update docstrings and README examples
6. **Integration**: Wire into pipeline if applicable

### Debugging Tips
- **Pytest**: Use `-v` (verbose), `-s` (show prints), `-k pattern` (filter tests)
- **Breakpoints**: Add `breakpoint()` for interactive debugging
- **Logging**: Use `logging` module, not print statements
- **Device Issues**: Set `CUDA_VISIBLE_DEVICES=-1` to force CPU
- **Model Issues**: Check HuggingFace cache (`~/.cache/huggingface/`)

### Performance Optimization (Future)
**Python-First Approach**: All initial implementation in pure Python.

**Optional Rust Extensions** (only if profiling shows bottlenecks):
- ORF finding: Already using fast C++ binary (`get_orfs`)
- Entropy calculation: Could add Rust impl for huge corpora
- Sequence processing: Rust for tight loops on large datasets

**Guidelines**:
- Profile first (`python -m cProfile`)
- Implement as optional extension with Python fallback
- Use PyO3 for Rust-Python bindings
- Add feature flag in pyproject.toml
- Do NOT require Rust for default installation

## Key Guidelines

**Trust these instructions first.** Only search the repo if instructions are incomplete, incorrect, or you need implementation details.

**Minimal Changes**:
- Change as few lines as possible
- Don't refactor unrelated code
- Don't fix unrelated bugs/tests
- Focus on specific task requirements

**Testing**:
- Never remove/disable existing tests
- Add tests for all new functionality
- Tests must be independent and deterministic
- Mock external dependencies (models, network, filesystem)
- Use fixtures for reusable test data

**Dependencies**:
- Avoid adding unless necessary
- Choose well-maintained, widely-used libraries
- Document dependency purpose in comments
- Consider optional dependencies for heavy features (e.g., `pip install genome_entropy[viz]` for visualization)

**Code Quality**:
- Full type hints (Python 3.8+ compatible)
- Google-style docstrings for all public APIs
- Handle errors explicitly (custom exceptions)
- Log important events (not debug prints)
- Keep functions focused (single responsibility)

## Quick Reference

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -e .
pip install -e ".[dev]"

# Install get_orfs binary
git clone https://github.com/linsalrob/get_orfs.git /tmp/get_orfs
cd /tmp/get_orfs && mkdir build && cd build
cmake .. && make && cmake --install . --prefix ..
export PATH="/tmp/get_orfs/bin:$PATH"

# Development
black src/ tests/           # Format
ruff check src/ tests/      # Lint
mypy src/genome_entropy/       # Type check
pytest                      # Unit tests only
pytest -k "not integration" # Explicitly skip integration

# Integration tests (downloads models)
RUN_INTEGRATION=1 pytest -v -m integration

# CLI usage
dna23di download                                    # Pre-download models
dna23di run --input examples/example_small.fasta --output results.json
dna23di orf --input in.fasta --output orfs.json
dna23di translate --input orfs.json --output proteins.json
dna23di encode3di --input proteins.json --output 3di.json
dna23di entropy --input 3di.json --output entropy.json

# Git
git status && git add <files> && git commit -m "msg" && git push
```

## Project Status

**Implemented**:
- [x] Basic repository structure
- [x] pyproject.toml with dependencies defined
- [x] Initial test infrastructure

**To Implement** (priority order):
1. Data models (dataclasses in types.py, jsonio.py)
2. Shannon entropy module (pure Python, well-tested)
3. ORF finder wrapper (subprocess to get_orfs)
4. Translation module (using pygenetic-code)
5. ProstT5 encoder with device auto-detection
6. Pipeline orchestration
7. CLI commands (typer-based)
8. Integration tests (marked, skipped by default)
9. Examples and documentation
10. CI/CD workflow

**External Dependencies to Document in README**:
- `get_orfs` binary (requires manual build, see instructions above)
- GPU drivers (optional, for CUDA/MPS acceleration)
- HuggingFace account (optional, for gated models)

---
*Last Updated: 2026-01-12 | Version 2.0*  
*Major Update: Comprehensive dna23di pipeline specification with ORF→protein→3Di→entropy workflow*
