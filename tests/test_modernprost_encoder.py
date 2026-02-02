"""Unit tests for ModernProst encoder methods."""

import inspect
import pytest
import re


def test_modernprost_encoder_uses_correct_tokenizer_api() -> None:
    """Test that ModernProst encoder uses modern tokenizer API.

    This test validates that we're using the modern tokenizer API
    (calling the tokenizer directly via __call__) rather than deprecated
    methods like batch_encode_plus.
    """
    from genome_entropy.encode3di.modernprost import ModernProstThreeDiEncoder

    # Get the source code of _encode_batch
    source = inspect.getsource(ModernProstThreeDiEncoder._encode_batch)

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


def test_modernprost_encoder_structure() -> None:
    """Test that ModernProst encoder has expected structure."""
    from genome_entropy.encode3di.modernprost import ModernProstThreeDiEncoder

    # Check that the class exists and has expected methods
    assert hasattr(ModernProstThreeDiEncoder, "__init__")
    assert hasattr(ModernProstThreeDiEncoder, "_load_model")
    assert hasattr(ModernProstThreeDiEncoder, "_encode_batch")
    assert hasattr(ModernProstThreeDiEncoder, "encode")
    assert hasattr(ModernProstThreeDiEncoder, "encode_proteins")
    assert hasattr(ModernProstThreeDiEncoder, "token_budget_batches")
    assert hasattr(ModernProstThreeDiEncoder, "_select_device")


def test_modernprost_encoder_no_special_tokens() -> None:
    """Test that ModernProst encoder does NOT use special tokens.

    ModernProst models do not use special tokens in tokenization,
    unlike ProstT5 which does.
    """
    from genome_entropy.encode3di.modernprost import ModernProstThreeDiEncoder

    # Get the source code of _encode_batch
    source = inspect.getsource(ModernProstThreeDiEncoder._encode_batch)

    # Verify we set add_special_tokens=False for ModernProst
    assert "add_special_tokens=False" in source, (
        "ModernProst encoder should use add_special_tokens=False in tokenization"
    )


def test_modernprost_encoder_handles_nonstandard_aa() -> None:
    """Test that ModernProst encoder replaces non-standard amino acids."""
    from genome_entropy.encode3di.modernprost import ModernProstThreeDiEncoder

    # Get the source code of _encode_batch
    source = inspect.getsource(ModernProstThreeDiEncoder._encode_batch)

    # Verify we replace U, Z, O with X
    assert 'replace("U", "X")' in source, (
        "ModernProst encoder should replace non-standard amino acid U with X"
    )
    assert 'replace("Z", "X")' in source, (
        "ModernProst encoder should replace non-standard amino acid Z with X"
    )
    assert 'replace("O", "X")' in source, (
        "ModernProst encoder should replace non-standard amino acid O with X"
    )


def test_config_has_modernprost_models() -> None:
    """Test that config defines ModernProst model constants."""
    from genome_entropy.config import (
        MODERNPROST_BASE_MODEL,
        MODERNPROST_PROFILES_MODEL,
        MODERNPROST_MODELS,
    )

    # Check that constants are defined
    assert MODERNPROST_BASE_MODEL == "gbouras13/modernprost-base"
    assert MODERNPROST_PROFILES_MODEL == "gbouras13/modernprost-profiles"
    assert isinstance(MODERNPROST_MODELS, set)
    assert MODERNPROST_BASE_MODEL in MODERNPROST_MODELS
    assert MODERNPROST_PROFILES_MODEL in MODERNPROST_MODELS


def test_modernprost_encoder_imports() -> None:
    """Test that ModernProst encoder can be imported."""
    from genome_entropy.encode3di import ModernProstThreeDiEncoder

    # Verify the class is accessible from the package
    assert ModernProstThreeDiEncoder is not None


@pytest.mark.skipif(True, reason="Requires torch and transformers to be installed")
def test_modernprost_encoder_instantiation() -> None:
    """Test that ModernProst encoder can be instantiated.

    This test is skipped by default as it requires torch and transformers.
    """
    from genome_entropy.encode3di.modernprost import ModernProstThreeDiEncoder

    try:
        # Test that encoder can be instantiated (without loading model)
        encoder = ModernProstThreeDiEncoder(
            model_name="gbouras13/modernprost-base",
            device="cpu"
        )

        # Verify encoder has expected attributes
        assert hasattr(encoder, "model_name")
        assert hasattr(encoder, "device")
        assert hasattr(encoder, "model")
        assert hasattr(encoder, "tokenizer")
        assert encoder.device == "cpu"
        assert encoder.model_name == "gbouras13/modernprost-base"
    except Exception:
        pytest.skip("Cannot instantiate encoder without dependencies")


def test_modernprost_no_attention_mask_modification() -> None:
    """Test that ModernProst encoder does NOT modify attention mask.
    
    Unlike ProstT5, ModernProst uses add_special_tokens=False, so it doesn't
    need to mask out extra tokens. This test verifies that there's no
    attention mask modification logic in _encode_batch.
    """
    from genome_entropy.encode3di.modernprost import ModernProstThreeDiEncoder
    
    # Get the source code of _encode_batch
    source = inspect.getsource(ModernProstThreeDiEncoder._encode_batch)
    
    # Verify add_special_tokens=False is used
    assert "add_special_tokens=False" in source, (
        "ModernProst should use add_special_tokens=False"
    )
    
    # Verify there's NO attention mask modification (setting to 0)
    # Pattern: attention_mask[...] = 0 (exactly 0, not 0.5 or 01)
    assert not re.search(r"attention_mask\s*\[.*\]\s*=\s*0\b", source), (
        "ModernProst should NOT modify attention_mask (no extra tokens to mask)"
    )

