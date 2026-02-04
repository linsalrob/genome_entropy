"""Tests for estimate_tokens command model selection."""

import pytest


def test_estimate_tokens_imports_modernprost() -> None:
    """Test that estimate_tokens command can import ModernProstThreeDiEncoder."""
    try:
        from genome_entropy.cli.commands.estimate_tokens import estimate_token_size_command
        from genome_entropy.encode3di import ModernProstThreeDiEncoder
        from genome_entropy.config import MODERNPROST_MODELS
        
        # Verify imports work
        assert ModernProstThreeDiEncoder is not None
        assert MODERNPROST_MODELS is not None
        assert "gbouras13/modernprost-base" in MODERNPROST_MODELS
        assert "gbouras13/modernprost-profiles" in MODERNPROST_MODELS
    except ImportError as e:
        pytest.skip(f"Required imports not available: {e}")


def test_model_selection_logic() -> None:
    """Test that correct encoder class is selected based on model name."""
    try:
        from genome_entropy.encode3di import ProstT5ThreeDiEncoder, ModernProstThreeDiEncoder
        from genome_entropy.config import MODERNPROST_MODELS
        
        # Test model names
        prostt5_model = "Rostlab/ProstT5_fp16"
        modernprost_base = "gbouras13/modernprost-base"
        modernprost_profiles = "gbouras13/modernprost-profiles"
        
        # Verify model detection logic (same as in estimate_tokens.py)
        assert modernprost_base in MODERNPROST_MODELS
        assert modernprost_profiles in MODERNPROST_MODELS
        assert prostt5_model not in MODERNPROST_MODELS
        
        # Test encoder instantiation (without loading models)
        # ProstT5 encoder
        encoder_prostt5 = ProstT5ThreeDiEncoder(model_name=prostt5_model, device="cpu")
        assert encoder_prostt5.model_name == prostt5_model
        assert isinstance(encoder_prostt5, ProstT5ThreeDiEncoder)
        assert not isinstance(encoder_prostt5, ModernProstThreeDiEncoder)
        
        # ModernProst base encoder
        encoder_modernprost_base = ModernProstThreeDiEncoder(model_name=modernprost_base, device="cpu")
        assert encoder_modernprost_base.model_name == modernprost_base
        assert isinstance(encoder_modernprost_base, ModernProstThreeDiEncoder)
        assert not isinstance(encoder_modernprost_base, ProstT5ThreeDiEncoder)
        
        # ModernProst profiles encoder
        encoder_modernprost_profiles = ModernProstThreeDiEncoder(model_name=modernprost_profiles, device="cpu")
        assert encoder_modernprost_profiles.model_name == modernprost_profiles
        assert isinstance(encoder_modernprost_profiles, ModernProstThreeDiEncoder)
        
    except ImportError as e:
        pytest.skip(f"Required imports not available: {e}")
