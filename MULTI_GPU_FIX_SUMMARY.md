# Multi-GPU Model Reloading Fix - Summary

## Problem

When processing GenBank files with multiple sequences using multi-GPU mode, the models were being reloaded for each sequence instead of being loaded once and reused. This caused significant performance degradation.

### Observed Behavior (Before Fix)

For a GenBank file with 2 sequences on a 2-GPU system:
```
Processing sequence 1/2:
  - Discovered 2 GPU(s)
  - Created encoder for cuda:0
  - Created encoder for cuda:1
  - Loading models on all GPUs... (10 seconds per GPU = ~20 seconds)
  - Encoded 6 sequences in 49.7s

Processing sequence 2/2:
  - Discovered 2 GPU(s) AGAIN
  - Created encoder for cuda:0 AGAIN
  - Created encoder for cuda:1 AGAIN
  - Loading models on all GPUs AGAIN... (10 seconds per GPU = ~20 seconds)
  - Encoded 2 sequences in 39.5s
```

**Total unnecessary overhead**: ~40 seconds of model loading time for the second sequence!

## Root Cause

The issue was in the code flow:

1. `run_pipeline()` calls `encoder.encode_proteins()` for each sequence
2. `encode_proteins()` calls `encoder.encode()` with `use_multi_gpu=True`
3. Inside `encoder.encode()`, when `use_multi_gpu=True`, it **creates a new MultiGPUEncoder instance**
4. `MultiGPUEncoder.__init__()` discovers GPUs and creates new encoder instances
5. `encode_multi_gpu()` loads models on all GPUs

This happened **for every sequence** because `MultiGPUEncoder` was created inside the encoding method.

## Solution

Move the `MultiGPUEncoder` initialization to the pipeline level:

### Changes Made

#### 1. Pipeline Runner (`runner.py`)

Initialize `MultiGPUEncoder` once before the sequence loop:

```python
# Initialize multi-GPU encoder once if multi-GPU mode is enabled
multi_gpu_encoder = None
if use_multi_gpu:
    from ..encode3di.multi_gpu import MultiGPUEncoder
    logger.info("Initializing multi-GPU encoder (will be reused for all sequences)...")
    multi_gpu_encoder = MultiGPUEncoder(
        model_name=model_name,
        encoder_class=ProstT5ThreeDiEncoder,
        gpu_ids=gpu_ids,
    )
    # Load models on all GPUs once at initialization
    logger.info("Loading models on all GPUs...")
    for gpu_encoder in multi_gpu_encoder.encoders:
        gpu_encoder._load_model()

# ... sequence processing loop ...
for seq_idx, (seq_id, dna_sequence) in enumerate(sequences.items(), 1):
    # ...
    three_dis = encoder.encode_proteins(
        proteins,
        encoding_size=actual_encoding_size,
        use_multi_gpu=use_multi_gpu,
        gpu_ids=gpu_ids,
        multi_gpu_encoder=multi_gpu_encoder,  # Pass pre-initialized encoder
    )
```

#### 2. Encoder (`encoder.py`)

Accept optional pre-initialized encoder:

```python
def encode(
    self,
    aa_sequences: List[str],
    encoding_size: int = DEFAULT_ENCODING_SIZE,
    use_multi_gpu: bool = False,
    gpu_ids: Optional[List[int]] = None,
    multi_gpu_encoder: Optional[Any] = None,  # New parameter
) -> List[str]:
    if use_multi_gpu:
        # Use pre-initialized encoder if provided, otherwise create new one
        if multi_gpu_encoder is None:
            logger.info("Initializing multi-GPU encoding")
            multi_gpu_encoder = MultiGPUEncoder(...)
            # Load models if not already loaded
            for gpu_encoder in multi_gpu_encoder.encoders:
                gpu_encoder._load_model()
        
        # Encode using multi-GPU (models already loaded)
        return multi_gpu_encoder.encode_multi_gpu(
            processed_seqs,
            self.token_budget_batches,
            encoding_size,
            skip_model_loading=True,  # Skip loading since already done
        )
```

#### 3. Multi-GPU Encoder (`multi_gpu.py`)

Add `skip_model_loading` parameter:

```python
def encode_multi_gpu(
    self,
    aa_sequences: List[str],
    token_budget_batches_fn: Callable[[List[str], int], Iterator[Any]],
    encoding_size: int,
    skip_model_loading: bool = False,  # New parameter
) -> List[str]:
    # Load models for all encoders (unless already loaded)
    if not skip_model_loading:
        logger.info("Loading models on all GPUs...")
        for encoder in self.encoders:
            encoder._load_model()
    
    # ... encoding logic ...
```

## Expected Behavior (After Fix)

For a GenBank file with 2 sequences on a 2-GPU system:
```
Pipeline initialization:
  - Discovered 2 GPU(s)
  - Created encoder for cuda:0
  - Created encoder for cuda:1
  - Loading models on all GPUs... (10 seconds per GPU = ~20 seconds)

Processing sequence 1/2:
  - Encoded 6 sequences in 49.7s

Processing sequence 2/2:
  - Encoded 2 sequences in 39.5s
```

**Result**: Models loaded only ONCE at the start, not for each sequence!

## Testing

Added comprehensive tests in `test_multi_gpu_reuse.py`:

1. **test_multi_gpu_encoder_created_once_for_multiple_sequences**: Verifies `MultiGPUEncoder` is instantiated only once when processing multiple sequences
2. **test_multi_gpu_encoder_models_loaded_once**: Verifies `_load_model` is called only once per GPU
3. **test_multi_gpu_encoder_reused_genbank**: Verifies encoder reuse for GenBank files with multiple entries

All existing tests continue to pass:
- `test_encoder_reuse.py`: 3 passed
- `test_multi_gpu_encoding.py`: 9 passed
- All unit tests: 133 passed, 9 skipped

## Backward Compatibility

The changes are fully backward compatible:

1. **Single-GPU mode**: Unchanged, still works as before
2. **Multi-GPU without pre-initialized encoder**: Falls back to old behavior (creates encoder on demand)
3. **Multi-GPU with pre-initialized encoder**: New optimized path (encoder reused)

## Performance Impact

For a GenBank file with N sequences on M GPUs:

- **Before**: Model loading time = N × M × ~10 seconds
- **After**: Model loading time = M × ~10 seconds (once at start)

For the example with 2 sequences and 2 GPUs:
- **Before**: 2 × 2 × 10 = 40 seconds
- **After**: 2 × 10 = 20 seconds
- **Savings**: 20 seconds (50% reduction in model loading overhead)

For larger files with 10 sequences:
- **Before**: 10 × 2 × 10 = 200 seconds
- **After**: 2 × 10 = 20 seconds
- **Savings**: 180 seconds (90% reduction!)

## Files Modified

1. `src/genome_entropy/pipeline/runner.py` - Initialize multi-GPU encoder at pipeline level
2. `src/genome_entropy/encode3di/encoder.py` - Accept pre-initialized encoder parameter
3. `src/genome_entropy/encode3di/multi_gpu.py` - Add skip_model_loading parameter
4. `tests/conftest.py` - Update mock to accept new parameter
5. `tests/test_multi_gpu_reuse.py` - Add comprehensive tests for encoder reuse
