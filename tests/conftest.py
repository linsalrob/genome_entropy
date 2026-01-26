"""Pytest configuration and fixtures for genome_entropy tests."""

import os
from typing import Dict

import pytest


@pytest.fixture
def synthetic_dna() -> Dict[str, str]:
    """Provide synthetic DNA sequences with known ORFs for testing.
    
    Returns:
        Dictionary mapping sequence IDs to DNA sequences
    """
    sequences = {
        "test_seq1": (
            # Forward strand ORF: positions 0-90 (ATG...TGA)
            "ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTGA"
            # Some non-ORF sequence
            "NNNNNNNNNN"
            # Reverse complement ORF would be here
        ),
        "test_seq2": (
            # Multiple ORFs on same strand
            "ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG"  # ORF 1 (no stop)
            "TAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAGTAG"
            "ATGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATAA"  # ORF 2 with stop
        ),
        "test_seq_short": (
            # Too short to be an ORF with default min length
            "ATGGCTAGCTGA"
        ),
    }
    return sequences


@pytest.fixture
def sample_fasta_content() -> str:
    """Provide sample FASTA file content for testing.
    
    Returns:
        FASTA formatted string
    """
    return """>seq1
ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
CTAGCTAGCTAGCTAGCTAGCTGA
>seq2
ATGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATAA
"""


@pytest.fixture
def mock_prostt5_encoder(monkeypatch):
    """Mock ProstT5 encoder to return deterministic 3Di sequences.
    
    Returns 'a' repeated for the length of each input sequence.
    This avoids needing to download models during testing.
    """
    from genome_entropy.encode3di.prostt5 import ProstT5ThreeDiEncoder
    
    def mock_encode(self, aa_sequences, encoding_size=10, use_multi_gpu=False, gpu_ids=None, multi_gpu_encoder=None):
        # Return 'a' * length for each sequence (lowercase for 3Di)
        return ['a' * len(seq.replace(' ', '').replace('<AA2fold>', '').strip()) for seq in aa_sequences]
    
    monkeypatch.setattr(ProstT5ThreeDiEncoder, "encode", mock_encode)
    monkeypatch.setattr(ProstT5ThreeDiEncoder, "_load_model", lambda self: None)


@pytest.fixture
def skip_integration():
    """Skip integration tests unless RUN_INTEGRATION environment variable is set."""
    if not os.getenv("RUN_INTEGRATION"):
        pytest.skip("Integration tests disabled (set RUN_INTEGRATION=1 to run)")


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires models, slow)",
    )
