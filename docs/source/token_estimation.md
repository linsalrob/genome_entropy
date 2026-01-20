# Token Size Estimation for 3Di Encoding

## Overview

The `encode3di` module has been refactored to improve code organization and add token size estimation functionality. This allows users to find the optimal encoding size for their GPU when converting proteins to 3Di structural tokens.

## Module Structure

The `encode3di` module is now organized into separate files for better clarity:

### New Module Organization

```
src/genome_entropy/encode3di/
├── __init__.py          # Public API exports
├── types.py             # Data types (ThreeDiRecord, IndexedSeq)
├── encoder.py           # ProstT5ThreeDiEncoder class
├── encoding.py          # Core encoding functions
├── token_estimator.py   # Token size estimation utilities
└── prostt5.py          # Backward compatibility exports
```

### Key Components

1. **types.py**: Data structures
   - `ThreeDiRecord`: Represents a 3Di structural encoding
   - `IndexedSeq`: Sequence with original position index

2. **encoder.py**: Main encoder class
   - `ProstT5ThreeDiEncoder`: Converts amino acids to 3Di tokens
   - `token_budget_batches()`: Batch sequences under token budget
   - `_encode_batch()`: Encode a single batch

3. **encoding.py**: Core encoding logic with reduced complexity
   - `preprocess_sequences()`: Prepare sequences for encoding
   - `process_batches()`: Process batches with progress tracking
   - `format_seconds()`: Format time durations
   - `get_memory_info()`: Get GPU memory usage

4. **token_estimator.py**: New token size estimation
   - `generate_random_protein()`: Generate random protein sequences
   - `generate_combined_proteins()`: Generate multiple proteins
   - `estimate_token_size()`: Find optimal token budget

## Token Size Estimation

### Purpose

The token size (encoding size) determines how many amino acids are encoded in each GPU batch. Setting this too high can cause Out of Memory errors, while setting it too low wastes GPU capacity.

The token size estimator helps you find the optimal value for your GPU.

### Usage

#### Via CLI

```bash
# Basic usage
dna23di estimate-tokens

# Custom range and parameters
dna23di estimate-tokens --start 3000 --end 10000 --step 1000 --trials 3

# Specify device
dna23di estimate-tokens --device cuda --model Rostlab/ProstT5_fp16
```

#### Via Python API

```python
from genome_entropy.encode3di import ProstT5ThreeDiEncoder, estimate_token_size

# Initialize encoder
encoder = ProstT5ThreeDiEncoder()

# Run estimation
results = estimate_token_size(
    encoder=encoder,
    start_length=3000,
    end_length=10000,
    step=1000,
    num_trials=3,
    base_protein_length=100,
)

# Use recommended token size
print(f"Recommended token size: {results['recommended_token_size']} AA")

# Use in encoding
encoder.encode(proteins, encoding_size=results['recommended_token_size'])
```

### How It Works

1. **Generates random proteins**: Creates realistic protein sequences of varying lengths
2. **Combines into batches**: Uses the same batching logic as actual encoding
3. **Tests encoding**: Attempts to encode with increasing total lengths
4. **Catches OOM errors**: Detects when GPU memory is exhausted
5. **Recommends size**: Returns 90% of maximum for safety margin

### Output

The estimator returns a dictionary with:
- `max_length`: Maximum length successfully encoded
- `recommended_token_size`: 90% of max for safety (recommended)
- `trials_per_length`: Number of successful trials per length tested
- `device`: Device used for testing

## Backward Compatibility

All existing imports continue to work:

```python
# Old style - still works
from genome_entropy.encode3di.prostt5 import ThreeDiRecord, ProstT5ThreeDiEncoder

# New style - also works
from genome_entropy.encode3di import ThreeDiRecord, ProstT5ThreeDiEncoder
```

## Testing

```bash
# Run all tests (excluding integration)
pytest tests/ -k "not integration"

# Run token estimator tests specifically
pytest tests/test_token_estimator.py -v
```

## Examples

See `examples/token_estimation_example.py` for complete working examples.
