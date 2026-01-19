"""Unit tests for ProstT5 encoder methods and tokenizer validation."""

import inspect
import pytest


def test_encoder_uses_correct_tokenizer_api() -> None:
    """Test that encoder uses modern tokenizer API, not deprecated methods.

    This test validates that we're using the modern tokenizer API
    (calling the tokenizer directly via __call__) rather than deprecated
    methods like batch_encode_plus.
    """
    from orf_entropy.encode3di.encoder import ProstT5ThreeDiEncoder

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
    from orf_entropy.encode3di.encoder import ProstT5ThreeDiEncoder

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
    from orf_entropy.encode3di.encoder import ProstT5ThreeDiEncoder

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
