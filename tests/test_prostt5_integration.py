"""Integration tests for ProstT5 encoding (skipped by default)."""

import os

import pytest

# These tests require the full model and are slow
pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Integration tests disabled")
def test_prostt5_real_inference() -> None:
    """Test real ProstT5 inference with a small protein.
    
    This test downloads the actual model and performs inference.
    Only run when RUN_INTEGRATION=1 environment variable is set.
    """
    from genome_entropy.encode3di.prostt5 import ProstT5ThreeDiEncoder
    
    # Initialize encoder (will download model on first run)
    encoder = ProstT5ThreeDiEncoder()
    
    # Test with a short protein sequence
    test_sequences = [
        "ACDEFGHIKLMNPQRSTVWY",  # All 20 standard amino acids
        "MKTAYIAKQR",  # A real protein N-terminus
    ]
    
    # Encode
    results = encoder.encode(test_sequences)
    
    # Verify results
    assert len(results) == 2
    # Results should be approximately same length as input (may vary due to tokenization)
    assert 15 <= len(results[0]) <= 25
    assert 8 <= len(results[1]) <= 15
    
    # Verify all characters are lowercase (3Di format)
    for three_di_seq in results:
        assert three_di_seq.islower() or three_di_seq == ""


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Integration tests disabled")
def test_prostt5_device_selection() -> None:
    """Test that device selection works correctly."""
    from genome_entropy.encode3di.prostt5 import ProstT5ThreeDiEncoder
    
    # Test auto device selection
    encoder = ProstT5ThreeDiEncoder(device=None)
    assert encoder.device in ["cuda", "mps", "cpu"]
    
    # Test explicit CPU
    encoder_cpu = ProstT5ThreeDiEncoder(device="cpu")
    assert encoder_cpu.device == "cpu"


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Integration tests disabled")
def test_prostt5_batch_processing() -> None:
    """Test batch processing of multiple sequences."""
    from genome_entropy.encode3di.prostt5 import ProstT5ThreeDiEncoder
    
    encoder = ProstT5ThreeDiEncoder()
    
    # Create a batch of test sequences
    sequences = ["ACDEFG"] * 10  # 10 identical short sequences
    
    # Encode with small encoding size to force multiple batches
    results = encoder.encode(sequences, encoding_size=20)
    
    assert len(results) == 10
    for result in results:
        # Results should be approximately same length as input
        # (exact length may vary slightly due to tokenization)
        assert 4 <= len(result) <= 10
