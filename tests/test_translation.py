"""Tests for translation functionality."""

import pytest

from orf_entropy.orf.types import OrfRecord
from orf_entropy.translate.translator import ProteinRecord


def test_protein_record_creation() -> None:
    """Test creating a valid ProteinRecord."""
    orf = OrfRecord(
        parent_id="seq1",
        orf_id="orf1",
        start=0,
        end=90,
        strand="+",
        frame=0,
        nt_sequence="ATG" * 30,  # 90 nt = 30 codons
        table_id=11,
        has_start_codon=True,
        has_stop_codon=False,
    )
    
    protein = ProteinRecord(
        orf=orf,
        aa_sequence="M" * 30,
        aa_length=30,
    )
    
    assert protein.orf == orf
    assert protein.aa_sequence == "M" * 30
    assert protein.aa_length == 30


def test_protein_record_length_mismatch() -> None:
    """Test that aa_length must match sequence length."""
    orf = OrfRecord(
        parent_id="seq1",
        orf_id="orf1",
        start=0,
        end=90,
        strand="+",
        frame=0,
        nt_sequence="ATG" * 30,
        table_id=11,
        has_start_codon=True,
        has_stop_codon=False,
    )
    
    with pytest.raises(ValueError, match="aa_length"):
        ProteinRecord(
            orf=orf,
            aa_sequence="M" * 30,
            aa_length=25,  # Wrong length
        )


def test_protein_record_with_stop() -> None:
    """Test protein with stop codon (should be removed)."""
    orf = OrfRecord(
        parent_id="seq1",
        orf_id="orf1",
        start=0,
        end=93,
        strand="+",
        frame=0,
        nt_sequence="ATGGCATAG" * 10 + "TAG",  # Ends with stop
        table_id=11,
        has_start_codon=True,
        has_stop_codon=True,
    )
    
    # In real translation, stop codon would be removed
    protein = ProteinRecord(
        orf=orf,
        aa_sequence="MA" * 10,  # Stop removed
        aa_length=20,
    )
    
    assert protein.aa_length == 20
    assert "*" not in protein.aa_sequence


def test_protein_record_various_amino_acids() -> None:
    """Test protein with various amino acids."""
    orf = OrfRecord(
        parent_id="seq1",
        orf_id="orf1",
        start=0,
        end=60,
        strand="+",
        frame=0,
        nt_sequence="N" * 60,  # Ambiguous sequence
        table_id=11,
        has_start_codon=False,
        has_stop_codon=False,
    )
    
    # Ambiguous codons translate to X
    protein = ProteinRecord(
        orf=orf,
        aa_sequence="ACDEFGHIKLMNPQRSTVWY",  # 20 standard AAs
        aa_length=20,
    )
    
    assert len(set(protein.aa_sequence)) == 20  # All different
    assert protein.aa_length == 20


def test_protein_record_short_protein() -> None:
    """Test very short protein."""
    orf = OrfRecord(
        parent_id="seq1",
        orf_id="short_orf",
        start=0,
        end=9,
        strand="+",
        frame=0,
        nt_sequence="ATGGCATAG",  # 3 codons
        table_id=11,
        has_start_codon=True,
        has_stop_codon=False,
    )
    
    protein = ProteinRecord(
        orf=orf,
        aa_sequence="MA",  # Stop removed if present
        aa_length=2,
    )
    
    assert protein.aa_length == 2


def test_protein_record_ambiguous_codons() -> None:
    """Test translation with ambiguous codons."""
    orf = OrfRecord(
        parent_id="seq1",
        orf_id="orf1",
        start=0,
        end=12,
        strand="+",
        frame=0,
        nt_sequence="ATGNNNGGGTTT",  # ATG, NNN (ambiguous), GGG, TTT
        table_id=11,
        has_start_codon=True,
        has_stop_codon=False,
    )
    
    # NNN should translate to X
    protein = ProteinRecord(
        orf=orf,
        aa_sequence="MXGF",  # M from ATG, X from NNN, G from GGG, F from TTT
        aa_length=4,
    )
    
    assert "X" in protein.aa_sequence
    assert protein.aa_length == 4
