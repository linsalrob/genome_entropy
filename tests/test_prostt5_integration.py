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
    from orf_entropy.config import THREEDDI_ALPHABET
    from orf_entropy.encode3di.prostt5 import ProstT5ThreeDiEncoder
    
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
    assert len(results[0]) == len(test_sequences[0])
    assert len(results[1]) == len(test_sequences[1])
    
    # Verify all characters are from 3Di alphabet
    for three_di_seq in results:
        assert all(c in THREEDDI_ALPHABET for c in three_di_seq)


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Integration tests disabled")
def test_prostt5_device_selection() -> None:
    """Test that device selection works correctly."""
    from orf_entropy.encode3di.prostt5 import ProstT5ThreeDiEncoder
    
    # Test auto device selection
    encoder = ProstT5ThreeDiEncoder(device=None)
    assert encoder.device in ["cuda", "mps", "cpu"]
    
    # Test explicit CPU
    encoder_cpu = ProstT5ThreeDiEncoder(device="cpu")
    assert encoder_cpu.device == "cpu"


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Integration tests disabled")
def test_prostt5_batch_processing() -> None:
    """Test batch processing of multiple sequences."""
    from orf_entropy.encode3di.prostt5 import ProstT5ThreeDiEncoder
    
    encoder = ProstT5ThreeDiEncoder()
    
    # Create a batch of test sequences
    sequences = ["ACDEFG"] * 10  # 10 identical short sequences
    
    # Encode with different batch sizes
    results = encoder.encode(sequences, batch_size=2)
    
    assert len(results) == 10
    for result in results:
        assert len(result) == 6  # Same length as input
