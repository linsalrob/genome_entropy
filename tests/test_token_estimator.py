"""Tests for token size estimation functionality."""

import pytest

from orf_entropy.encode3di.token_estimator import (
    generate_random_protein,
    generate_combined_proteins,
)


def test_generate_random_protein() -> None:
    """Test random protein generation."""
    # Test basic generation
    protein = generate_random_protein(100)
    assert len(protein) == 100
    assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in protein)


def test_generate_random_protein_with_seed() -> None:
    """Test that seed produces reproducible results."""
    protein1 = generate_random_protein(50, seed=42)
    protein2 = generate_random_protein(50, seed=42)
    assert protein1 == protein2

    # Different seed should produce different result
    protein3 = generate_random_protein(50, seed=43)
    assert protein1 != protein3


def test_generate_combined_proteins() -> None:
    """Test generation of combined proteins."""
    proteins = generate_combined_proteins(target_length=500, base_length=100)

    # Should generate approximately 5 proteins
    assert 3 <= len(proteins) <= 7  # Allow some variation

    # Total length should be approximately target
    total_len = sum(len(p) for p in proteins)
    assert 450 <= total_len <= 550  # Allow some variation

    # Each protein should be valid
    for protein in proteins:
        assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in protein)


def test_generate_combined_proteins_with_seed() -> None:
    """Test reproducibility of combined proteins."""
    proteins1 = generate_combined_proteins(300, base_length=100, seed=42)
    proteins2 = generate_combined_proteins(300, base_length=100, seed=42)

    assert len(proteins1) == len(proteins2)
    for p1, p2 in zip(proteins1, proteins2):
        assert p1 == p2


def test_generate_combined_proteins_small_target() -> None:
    """Test with target smaller than base length."""
    proteins = generate_combined_proteins(target_length=50, base_length=100)

    # Should generate just one protein with adjusted length
    assert len(proteins) >= 1
    assert sum(len(p) for p in proteins) <= 70  # Should be close to target


@pytest.mark.skipif(
    not hasattr(pytest, "importorskip")
    or pytest.importorskip("torch", reason="torch not available"),
    reason="PyTorch not available",
)
def test_estimate_token_size_basic() -> None:
    """Test basic token size estimation functionality.

    This is a minimal test that doesn't require a real GPU.
    """
    # Import here to avoid import errors when torch is not available
    try:
        import torch
        from orf_entropy.encode3di import ProstT5ThreeDiEncoder
    except ImportError:
        pytest.skip("PyTorch or encoder not available")

    # Skip if CUDA is not available (to avoid very slow tests)
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU-dependent test")

    # This test would need a real encoder and GPU, so we just test imports work
    encoder = ProstT5ThreeDiEncoder()
    assert hasattr(encoder, "encode")
    assert hasattr(encoder, "device")


def test_estimate_token_size_validates_encoder() -> None:
    """Test that estimate_token_size validates the encoder parameter."""
    from orf_entropy.encode3di.token_estimator import estimate_token_size

    # Should raise ValueError for invalid encoder
    with pytest.raises(ValueError, match="encoder must be"):
        estimate_token_size("not an encoder")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="encoder must be"):
        estimate_token_size({})  # type: ignore[arg-type]
