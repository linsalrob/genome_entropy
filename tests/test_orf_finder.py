"""Tests for ORF finding functionality."""

import pytest

from orf_entropy.orf.types import OrfRecord


def test_orf_record_creation() -> None:
    """Test creating a valid OrfRecord."""
    orf = OrfRecord(
        parent_id="seq1",
        orf_id="seq1_orf_0_90_+_f0",
        start=0,
        end=90,
        strand="+",
        frame=0,
        nt_sequence="A" * 90,
        table_id=11,
        has_start_codon=True,
        has_stop_codon=True,
    )
    
    assert orf.parent_id == "seq1"
    assert orf.orf_id == "seq1_orf_0_90_+_f0"
    assert orf.start == 0
    assert orf.end == 90
    assert orf.strand == "+"
    assert orf.frame == 0
    assert len(orf.nt_sequence) == 90
    assert orf.table_id == 11
    assert orf.has_start_codon is True
    assert orf.has_stop_codon is True


def test_orf_record_invalid_strand() -> None:
    """Test that invalid strand raises ValueError."""
    with pytest.raises(ValueError, match="Invalid strand"):
        OrfRecord(
            parent_id="seq1",
            orf_id="orf1",
            start=0,
            end=90,
            strand="*",  # Invalid
            frame=0,
            nt_sequence="A" * 90,
            table_id=11,
            has_start_codon=True,
            has_stop_codon=True,
        )


def test_orf_record_invalid_frame() -> None:
    """Test that invalid frame raises ValueError."""
    with pytest.raises(ValueError, match="Invalid frame"):
        OrfRecord(
            parent_id="seq1",
            orf_id="orf1",
            start=0,
            end=90,
            strand="+",
            frame=3,  # Invalid (must be 0, 1, or 2)
            nt_sequence="A" * 90,
            table_id=11,
            has_start_codon=True,
            has_stop_codon=True,
        )


def test_orf_record_invalid_coordinates() -> None:
    """Test that invalid coordinates raise ValueError."""
    # Start < 0
    with pytest.raises(ValueError, match="Invalid start position"):
        OrfRecord(
            parent_id="seq1",
            orf_id="orf1",
            start=-1,
            end=90,
            strand="+",
            frame=0,
            nt_sequence="A" * 91,
            table_id=11,
            has_start_codon=True,
            has_stop_codon=True,
        )
    
    # End <= start
    with pytest.raises(ValueError, match="Invalid end position"):
        OrfRecord(
            parent_id="seq1",
            orf_id="orf1",
            start=90,
            end=90,
            strand="+",
            frame=0,
            nt_sequence="",
            table_id=11,
            has_start_codon=True,
            has_stop_codon=True,
        )


def test_orf_record_sequence_length_mismatch() -> None:
    """Test that sequence length must match coordinates."""
    with pytest.raises(ValueError, match="Sequence length"):
        OrfRecord(
            parent_id="seq1",
            orf_id="orf1",
            start=0,
            end=90,
            strand="+",
            frame=0,
            nt_sequence="A" * 60,  # Wrong length
            table_id=11,
            has_start_codon=True,
            has_stop_codon=True,
        )


def test_orf_record_both_strands() -> None:
    """Test creating ORFs on both strands."""
    orf_plus = OrfRecord(
        parent_id="seq1",
        orf_id="orf_plus",
        start=0,
        end=90,
        strand="+",
        frame=0,
        nt_sequence="A" * 90,
        table_id=11,
        has_start_codon=True,
        has_stop_codon=True,
    )
    
    orf_minus = OrfRecord(
        parent_id="seq1",
        orf_id="orf_minus",
        start=100,
        end=190,
        strand="-",
        frame=1,
        nt_sequence="T" * 90,
        table_id=11,
        has_start_codon=True,
        has_stop_codon=False,
    )
    
    assert orf_plus.strand == "+"
    assert orf_minus.strand == "-"
    assert orf_minus.frame == 1


def test_orf_record_all_frames() -> None:
    """Test creating ORFs in all three frames."""
    for frame in [0, 1, 2]:
        orf = OrfRecord(
            parent_id="seq1",
            orf_id=f"orf_f{frame}",
            start=frame,
            end=frame + 90,
            strand="+",
            frame=frame,
            nt_sequence="A" * 90,
            table_id=11,
            has_start_codon=True,
            has_stop_codon=True,
        )
        assert orf.frame == frame


def test_orf_record_no_start_or_stop() -> None:
    """Test ORF without start or stop codons."""
    orf = OrfRecord(
        parent_id="seq1",
        orf_id="partial_orf",
        start=10,
        end=100,
        strand="+",
        frame=1,
        nt_sequence="C" * 90,
        table_id=11,
        has_start_codon=False,
        has_stop_codon=False,
    )
    
    assert orf.has_start_codon is False
    assert orf.has_stop_codon is False
