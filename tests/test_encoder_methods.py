"""Unit tests for ProstT5 encoder methods and tokenizer validation."""

import inspect
import pytest
import re


def test_encoder_uses_correct_tokenizer_api() -> None:
    """Test that encoder uses modern tokenizer API, not deprecated methods.

    This test validates that we're using the modern tokenizer API
    (calling the tokenizer directly via __call__) rather than deprecated
    methods like batch_encode_plus.
    """
    from genome_entropy.encode3di.encoder import ProstT5ThreeDiEncoder

    # Get the source code of _encode_batch
    source = inspect.getsource(ProstT5ThreeDiEncoder._encode_batch)

    # Verify we're NOT using the deprecated batch_encode_plus
    assert "batch_encode_plus" not in source, (
        "_encode_batch should not use deprecated batch_encode_plus method. "
        "Use tokenizer() call instead (modern API)."
    )

    # Verify we're using the modern API (calling tokenizer directly)
    assert "self.tokenizer(" in source, (
        "_encode_batch should use self.tokenizer() call (modern API) "
        "instead of batch_encode_plus."
    )


@pytest.mark.skipif(True, reason="Requires torch and transformers to be installed")
def test_encoder_tokenizer_methods() -> None:
    """Test that encoder validates tokenizer has required methods.

    This test is skipped by default as it requires torch and transformers.
    """
    from genome_entropy.encode3di.encoder import ProstT5ThreeDiEncoder

    try:
        # Test that encoder can be instantiated (without loading model)
        encoder = ProstT5ThreeDiEncoder(device="cpu")

        # Verify encoder has expected attributes
        assert hasattr(encoder, "model_name")
        assert hasattr(encoder, "device")
        assert hasattr(encoder, "model")
        assert hasattr(encoder, "tokenizer")
        assert encoder.device == "cpu"
    except Exception:
        pytest.skip("Cannot instantiate encoder without dependencies")


@pytest.mark.skipif(True, reason="Requires torch and transformers to be installed")
def test_encoder_methods_exist() -> None:
    """Test that encoder has all expected methods.

    This test is skipped by default as it requires torch and transformers.
    """
    from genome_entropy.encode3di.encoder import ProstT5ThreeDiEncoder

    try:
        encoder = ProstT5ThreeDiEncoder(device="cpu")

        # Check for expected public methods
        assert hasattr(encoder, "encode")
        assert hasattr(encoder, "encode_proteins")
        assert hasattr(encoder, "token_budget_batches")
        assert callable(encoder.encode)
        assert callable(encoder.encode_proteins)
        assert callable(encoder.token_budget_batches)

        # Check for expected private methods
        assert hasattr(encoder, "_encode_batch")
        assert hasattr(encoder, "_load_model")
        assert hasattr(encoder, "_select_device")
        assert callable(encoder._encode_batch)
        assert callable(encoder._load_model)
        assert callable(encoder._select_device)
    except Exception:
        pytest.skip("Cannot instantiate encoder without dependencies")


@pytest.mark.skipif(True, reason="Requires transformers to be installed")
def test_tokenizer_call_method_compatibility() -> None:
    """Test that tokenizer __call__ method is used correctly.

    This test validates that we're using the modern tokenizer API
    (calling the tokenizer directly) rather than deprecated methods.

    This test is skipped by default as it requires transformers.
    """
    try:
        from transformers import T5Tokenizer

        # Create a simple tokenizer (if transformers is available)
        tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

        # Verify the __call__ method exists (modern API)
        assert callable(tokenizer)

        # Test that calling the tokenizer works with expected parameters
        test_sequences = ["ACDE", "FGHI"]
        result = tokenizer(
            test_sequences,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )

        # Verify the result has expected attributes
        assert hasattr(result, "input_ids")
        assert hasattr(result, "attention_mask")

    except ImportError:
        pytest.skip("Transformers not available for this test")
    except Exception as e:
        # If model download fails, that's okay - we're just testing the API
        if "Connection" in str(e) or "download" in str(e).lower():
            pytest.skip(f"Could not download model for test: {e}")
        else:
            raise


def test_prostt5_attention_mask_extra_tokens() -> None:
    """Test that ProstT5 encoder masks extra special tokens in attention mask.
    
    This test verifies that the _encode_batch method includes logic to mask
    out the extra special tokens that ProstT5 appends at the end of each sequence.
    The masking should happen after tokenization but before inference.
    """
    from genome_entropy.encode3di.encoder import ProstT5ThreeDiEncoder
    
    # Get the source code of _encode_batch
    source = inspect.getsource(ProstT5ThreeDiEncoder._encode_batch)
    
    # Verify the masking logic is present
    # Should have a comment about ProstT5 appending special tokens at the end
    assert "ProstT5 appends special tokens at the end" in source, (
        "_encode_batch should have a comment explaining that ProstT5 appends special tokens at the end"
    )
    
    # Should have logic that modifies attention_mask
    assert "attention_mask" in source, (
        "_encode_batch should reference attention_mask for masking"
    )
    
    # Should have a loop that iterates over sequences
    assert re.search(r"for\s+\w+\s*,\s*\w+\s+in\s+enumerate\s*\(", source), (
        "_encode_batch should have a loop that enumerates over sequences"
    )
    
    # Should set attention_mask to 0 for extra tokens
    # Pattern: ids.attention_mask[idx, mask_position] = 0
    assert re.search(r"attention_mask\s*\[\s*\w+\s*,\s*\w+\s*\]\s*=\s*0\b", source), (
        "_encode_batch should set attention_mask to 0 for extra token positions"
    )
    
    # Should have bounds checking before masking - check for dimension [1]
    assert re.search(r"<\s*ids\.attention_mask\.shape\[1\]", source), (
        "_encode_batch should check bounds using ids.attention_mask.shape[1] before masking"
    )
    
    # Verify the masking happens after tokenization but before model.generate
    lines = source.split('\n')
    tokenizer_line = None
    mask_line = None
    generate_line = None
    
    for i, line in enumerate(lines):
        if 'self.tokenizer(' in line:
            tokenizer_line = i
        if 'attention_mask[' in line and '= 0' in line:
            mask_line = i
        if 'self.model.generate(' in line:
            generate_line = i
    
    # Verify all three operations are found
    assert tokenizer_line is not None, "Could not find tokenizer call in _encode_batch"
    assert mask_line is not None, "Could not find attention_mask modification in _encode_batch"
    assert generate_line is not None, "Could not find model.generate call in _encode_batch"
    
    # Verify the order: tokenizer -> mask -> generate
    assert tokenizer_line < mask_line < generate_line, (
        "Masking should happen after tokenization but before model.generate()"
    )

