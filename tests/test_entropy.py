"""Tests for Shannon entropy calculation."""

import math

import pytest

from genome_entropy.entropy.shannon import (
    EntropyReport,
    calculate_entropies_for_sequences,
    calculate_sequence_entropy,
    shannon_entropy,
)


def test_shannon_entropy_empty_string() -> None:
    """Test entropy of empty string is 0."""
    assert shannon_entropy("") == 0.0


def test_shannon_entropy_single_symbol() -> None:
    """Test entropy of uniform string is 0."""
    assert shannon_entropy("AAAA") == 0.0
    assert shannon_entropy("TTTTTT") == 0.0


def test_shannon_entropy_two_symbols_equal() -> None:
    """Test entropy of two symbols with equal probability."""
    # ATAT has H = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
    entropy = shannon_entropy("ATAT")
    assert abs(entropy - 1.0) < 1e-10


def test_shannon_entropy_four_symbols_equal() -> None:
    """Test entropy of four symbols with equal probability."""
    # ACGT has H = -4 * 0.25*log2(0.25) = 2.0
    entropy = shannon_entropy("ACGT")
    assert abs(entropy - 2.0) < 1e-10


def test_shannon_entropy_dna_sequence() -> None:
    """Test entropy calculation for a DNA sequence."""
    # Sequence with known distribution
    sequence = "AAAACCCCGGGGTTTT"  # Equal distribution
    entropy = shannon_entropy(sequence)
    expected = 2.0  # log2(4) for equal distribution of 4 bases
    assert abs(entropy - expected) < 1e-10


def test_shannon_entropy_biased_distribution() -> None:
    """Test entropy of biased distribution."""
    # 8 A's and 2 C's: p(A)=0.8, p(C)=0.2
    sequence = "AAAAAAACCC"
    entropy = shannon_entropy(sequence)
    # H = -0.7*log2(0.7) - 0.3*log2(0.3) ≈ 0.881
    expected = -(0.7 * math.log2(0.7) + 0.3 * math.log2(0.3))
    assert abs(entropy - expected) < 1e-10


def test_shannon_entropy_normalization() -> None:
    """Test normalized entropy calculation."""
    # Perfect uniform distribution over 4 symbols
    entropy_norm = shannon_entropy("ACGT", alphabet=set("ACGT"), normalize=True)
    assert abs(entropy_norm - 1.0) < 1e-10
    
    # Low entropy sequence
    entropy_norm = shannon_entropy("AAAA", alphabet=set("ACGT"), normalize=True)
    assert abs(entropy_norm - 0.0) < 1e-10
    
    # Half entropy
    entropy_norm = shannon_entropy("ATAT", alphabet=set("ACGT"), normalize=True)
    expected = 1.0 / 2.0  # H=1.0, max=2.0
    assert abs(entropy_norm - expected) < 1e-10


def test_calculate_sequence_entropy_uppercase() -> None:
    """Test that sequence entropy handles case normalization."""
    entropy_upper = calculate_sequence_entropy("ACGT")
    entropy_lower = calculate_sequence_entropy("acgt")
    assert abs(entropy_upper - entropy_lower) < 1e-10


def test_calculate_entropies_for_sequences() -> None:
    """Test batch entropy calculation."""
    sequences = {
        "seq1": "AAAA",
        "seq2": "ACGT",
        "seq3": "ATAT",
    }
    
    entropies = calculate_entropies_for_sequences(sequences)
    
    assert len(entropies) == 3
    assert abs(entropies["seq1"] - 0.0) < 1e-10
    assert abs(entropies["seq2"] - 2.0) < 1e-10
    assert abs(entropies["seq3"] - 1.0) < 1e-10


def test_entropy_report_dataclass() -> None:
    """Test EntropyReport dataclass creation."""
    report = EntropyReport(
        dna_entropy_global=2.5,
        orf_nt_entropy={"orf1": 1.8},
        protein_aa_entropy={"orf1": 3.2},
        three_di_entropy={"orf1": 2.9},
        alphabet_sizes={"dna": 4, "protein": 20, "three_di": 20},
    )
    
    assert report.dna_entropy_global == 2.5
    assert report.orf_nt_entropy["orf1"] == 1.8
    assert report.protein_aa_entropy["orf1"] == 3.2
    assert report.three_di_entropy["orf1"] == 2.9
    assert report.alphabet_sizes["dna"] == 4


def test_shannon_entropy_protein_sequence() -> None:
    """Test entropy for protein sequences."""
    # All alanine
    assert shannon_entropy("AAAA") == 0.0
    
    # Simple protein sequence
    protein = "ACDEFGHIKLMNPQRSTVWY"  # All 20 standard amino acids
    entropy = shannon_entropy(protein)
    expected = math.log2(20)  # Equal distribution
    assert abs(entropy - expected) < 1e-10
