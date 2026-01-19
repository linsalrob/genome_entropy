# orf_entropy (dna23di)

[![Python CI](https://github.com/linsalrob/orf_entropy/workflows/Python%20CI/badge.svg)](https://github.com/linsalrob/orf_entropy/actions)
[![Documentation](https://github.com/linsalrob/orf_entropy/workflows/Documentation/badge.svg)](https://linsalrob.github.io/orf_entropy/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compare and contrast the entropy of sequences, ORFs, proteins, and 3Di structural encodings.

**dna23di** is a complete bioinformatics pipeline that converts DNA sequences → ORFs → proteins → 3Di structural tokens, computing Shannon entropy at each representation level.

## Documentation

📚 **[Read the full documentation on GitHub Pages](https://linsalrob.github.io/orf_entropy/)**

The documentation includes:
- Installation guide
- Quick start tutorial
- Complete CLI reference
- Python API documentation
- User guide with detailed explanations
- Developer guide for contributors

## Features

- 🧬 **ORF Finding**: Extract Open Reading Frames from DNA sequences using customizable genetic codes
- 🔄 **Translation**: Convert ORFs to protein sequences with support for all NCBI genetic code tables
- 🏗️ **3Di Encoding**: Predict structural alphabet tokens directly from sequences using ProstT5
- 📊 **Entropy Analysis**: Calculate Shannon entropy at DNA, ORF, protein, and 3Di levels
- ⚡ **GPU Acceleration**: Auto-detect and use CUDA, MPS (Apple Silicon), or CPU
- 🔧 **Modular CLI**: Run complete pipeline or individual steps
- 📝 **Comprehensive Logging**: Configurable log levels and output to file or STDOUT

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/linsalrob/orf_entropy.git
cd orf_entropy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Run complete pipeline
dna23di run --input examples/example_small.fasta --output results.json

# Or run individual steps
dna23di orf --input input.fasta --output orfs.json
dna23di translate --input orfs.json --output proteins.json
dna23di encode3di --input proteins.json --output 3di.json
dna23di entropy --input 3di.json --output entropy.json
```

## Requirements

### Python Dependencies

- Python 3.8 or higher
- PyTorch >= 2.0.0 (GPU support optional)
- Transformers >= 4.30.0 (HuggingFace)
- pygenetic-code >= 0.1.0
- typer >= 0.9.0

### External Binary: get_orfs

The ORF finder requires the `get_orfs` binary from https://github.com/linsalrob/get_orfs

**Installation:**

```bash
# Clone and build get_orfs
git clone https://github.com/linsalrob/get_orfs.git /tmp/get_orfs
cd /tmp/get_orfs
mkdir build && cd build
cmake ..
make
cmake --install . --prefix ..

# Add to PATH or set environment variable
export PATH="/tmp/get_orfs/bin:$PATH"
# Or set GET_ORFS_PATH environment variable
export GET_ORFS_PATH=/tmp/get_orfs/bin/get_orfs
```

## CLI Commands

### `dna23di run` - Complete Pipeline

Run all steps from DNA to 3Di with entropy calculation:

```bash
dna23di run \
    --input input.fasta \
    --output results.json \
    --table 11 \
    --min-aa 30 \
    --model Rostlab/ProstT5_fp16 \
    --device auto
```

**Options:**
- `--input, -i`: Input FASTA file (required)
- `--output, -o`: Output JSON file (required)
- `--table, -t`: NCBI genetic code table ID (default: 11)
- `--min-aa`: Minimum protein length in amino acids (default: 30)
- `--model, -m`: ProstT5 model name (default: Rostlab/ProstT5_fp16)
- `--device, -d`: Device for inference (auto/cuda/mps/cpu)
- `--skip-entropy`: Skip entropy calculation

### `dna23di orf` - Find ORFs

Extract Open Reading Frames from DNA sequences:

```bash
dna23di orf --input input.fasta --output orfs.json --table 11 --min-nt 90
```

### `dna23di translate` - Translate ORFs

Translate ORFs to protein sequences:

```bash
dna23di translate --input orfs.json --output proteins.json --table 11
```

### `dna23di encode3di` - Encode to 3Di

Convert proteins to 3Di structural tokens using ProstT5:

```bash
dna23di encode3di \
    --input proteins.json \
    --output 3di.json \
    --model Rostlab/ProstT5_fp16 \
    --device auto \
    --batch-size 4
```

### `dna23di entropy` - Calculate Entropy

Compute Shannon entropy at all representation levels:

```bash
dna23di entropy --input 3di.json --output entropy.json --normalize
```

### `dna23di download` - Pre-download Models

Pre-download ProstT5 models to cache:

```bash
dna23di download --model Rostlab/ProstT5_fp16
```

## Logging

All `dna23di` commands support comprehensive logging with configurable output and verbosity.

### Global Logging Options

Every command accepts these logging options:

```bash
dna23di [OPTIONS] COMMAND [ARGS]

Global Options:
  --log-level, -l  TEXT  Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) [default: INFO]
  --log-file       PATH  Path to log file (default: log to STDOUT)
```

### Usage Examples

**Default logging (INFO level to STDOUT):**
```bash
dna23di run --input data.fasta --output results.json
```

**Debug logging to see detailed progress:**
```bash
dna23di --log-level DEBUG run --input data.fasta --output results.json
```

**Log to a file:**
```bash
dna23di --log-file pipeline.log run --input data.fasta --output results.json
```

**Debug logging to file:**
```bash
dna23di --log-level DEBUG --log-file debug.log run --input data.fasta --output results.json
```

**Quiet mode (only warnings and errors):**
```bash
dna23di --log-level WARNING run --input data.fasta --output results.json
```

### Log Levels

- **DEBUG**: Detailed information for diagnosing problems (sequence lengths, batch info, etc.)
- **INFO**: General informational messages (default - shows major steps and progress)
- **WARNING**: Warning messages for unusual conditions
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical errors that may cause the program to abort

### What Gets Logged

The logging system tracks:

- **File I/O**: Reading/writing FASTA and JSON files with sequence counts
- **ORF Finding**: Number of ORFs found, binary checks, parsing progress
- **Translation**: Translation progress, codon handling, error details
- **3Di Encoding**: Model loading, batch processing, memory usage, timing estimates
- **Entropy Calculation**: Entropy values at each representation level
- **Pipeline Progress**: Step-by-step progress through the complete pipeline

Example log output (INFO level):
```
2026-01-19 10:30:15 - orf_entropy.io.fasta - INFO - Reading FASTA file: input.fasta
2026-01-19 10:30:15 - orf_entropy.io.fasta - INFO - Successfully read 5 sequence(s) from input.fasta
2026-01-19 10:30:15 - orf_entropy.orf.finder - INFO - Starting ORF finding for 5 sequence(s) (table=11, min_length=90)
2026-01-19 10:30:16 - orf_entropy.orf.finder - INFO - Found 47 ORF(s) in 5 sequence(s)
2026-01-19 10:30:16 - orf_entropy.translate.translator - INFO - Translating 47 ORF(s) with table 11
2026-01-19 10:30:16 - orf_entropy.encode3di.encoder - INFO - Loading ProstT5 model: Rostlab/ProstT5_fp16
2026-01-19 10:30:20 - orf_entropy.encode3di.encoder - INFO - Loaded model Rostlab/ProstT5_fp16 on device cuda
2026-01-19 10:30:20 - orf_entropy.encode3di.encoding - INFO - 3Di encoding batch 1 of 12 batches...
```

## Data Flow

```
DNA FASTA → ORF Finder → ORFs (nucleotides)
          ↓
     Translator → Proteins (amino acids)
          ↓
     ProstT5 → 3Di tokens (structural alphabet)
          ↓
     Shannon Entropy → Entropy Report
```

## Genetic Code Tables

The pipeline supports all NCBI genetic code tables. Common ones:

- **Table 1**: Standard genetic code
- **Table 11**: Bacterial, archaeal, and plant plastid code (default)
- **Table 4**: Mold, protozoan, and coelenterate mitochondrial code

See full list at: https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi

## Output Format

Results are saved as JSON with the following structure:

```json
[
  {
    "input_id": "seq1",
    "input_dna_length": 1000,
    "orfs": [...],
    "proteins": [...],
    "three_dis": [...],
    "entropy": {
      "dna_entropy_global": 2.5,
      "orf_nt_entropy": {"orf1": 1.8},
      "protein_aa_entropy": {"orf1": 3.2},
      "three_di_entropy": {"orf1": 2.9},
      "alphabet_sizes": {"dna": 4, "protein": 20, "three_di": 20}
    }
  }
]
```

## Development

### Running Tests

```bash
# Run unit tests
pytest

# Run with coverage
pytest --cov=orf_entropy

# Skip integration tests (default)
pytest -k "not integration"

# Run integration tests (downloads models, slow)
RUN_INTEGRATION=1 pytest -v -m integration
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/orf_entropy/
```

### Project Structure

```
orf_entropy/
├── src/orf_entropy/
│   ├── io/              # FASTA and JSON I/O
│   ├── orf/             # ORF finding and types
│   ├── translate/       # Protein translation
│   ├── encode3di/       # 3Di encoding (ProstT5)
│   ├── entropy/         # Shannon entropy calculation
│   ├── pipeline/        # End-to-end orchestration
│   └── cli/             # Command-line interface
├── tests/               # Unit and integration tests
└── examples/            # Example data and scripts
```

## Citation

If you use this software, please cite:

- **ProstT5**: Heinzinger et al. (2023), "ProstT5: Bilingual Language Model for Protein Sequence and Structure"
- **get_orfs**: https://github.com/linsalrob/get_orfs
- **pygenetic-code**: https://github.com/linsalrob/genetic_codes

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Rob Edwards (@linsalrob)  
Email: raedwards@gmail.com

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'orf_entropy'**
- Run `pip install -e .` from repository root

**get_orfs binary not found**
- Install get_orfs and add to PATH or set GET_ORFS_PATH environment variable

**CUDA out of memory**
- Use CPU with `--device cpu` or reduce batch size with `--batch-size 1`

**Model download fails**
- Check internet connection
- Verify HuggingFace cache permissions (~/.cache/huggingface/)

**Integration tests run unexpectedly**
- Use `pytest -k "not integration"` to skip them

## Acknowledgments

- ProstT5 model by Rostlab
- get_orfs by Rob Edwards
- genetic_codes by Rob Edwards
