"""Tests for ORF finding functionality."""

import pytest

from genome_entropy.orf.types import OrfRecord
from genome_entropy.orf.finder import _parse_orf_header_line


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
        aa_sequence="K" * 30,  # Added aa_sequence field
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
    assert len(orf.aa_sequence) == 30
    assert orf.table_id == 11
    assert orf.has_start_codon is True
    assert orf.has_stop_codon is True
    assert orf.in_genbank is False  # Default value


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
            aa_sequence="K" * 30,
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
            frame=4,  # Invalid (must be 0, 1, 2, or 3)
            nt_sequence="A" * 90,
            aa_sequence="K" * 30,
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
            aa_sequence="K" * 30,
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
            aa_sequence="",
            table_id=11,
            has_start_codon=True,
            has_stop_codon=True,
        )


def test_orf_record_sequence_length_mismatch() -> None:
    """Test that sequence length can be different from coordinates.

    Note: The OrfRecord allows sequences to be empty or shorter than coordinates
    because sequences may be filled in later (e.g., by the ORF finder).
    This test now verifies that mismatched lengths are allowed.
    """
    # This should NOT raise an error
    orf = OrfRecord(
        parent_id="seq1",
        orf_id="orf1",
        start=0,
        end=90,
        strand="+",
        frame=0,
        nt_sequence="A" * 60,  # Different length than coordinates
        aa_sequence="K" * 20,
        table_id=11,
        has_start_codon=True,
        has_stop_codon=True,
    )
    assert len(orf.nt_sequence) == 60
    assert orf.end - orf.start == 90


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
        aa_sequence="K" * 30,
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
        aa_sequence="F" * 30,
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
            aa_sequence="K" * 30,
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
        aa_sequence="P" * 30,
        table_id=11,
        has_start_codon=False,
        has_stop_codon=False,
    )

    assert orf.has_start_codon is False
    assert orf.has_stop_codon is False


def test_parse_orf_header_standard_format() -> None:
    """Test parsing standard get_orfs header format with parent_id prefix."""
    # Standard format: >parent_id-orf_id [parent_id frame frame_num start end]
    header = ">JQ995537-orf14635 [JQ995537 frame -3 96951 97093]"
    orf = _parse_orf_header_line(header, table_id=11)

    assert orf.parent_id == "JQ995537"
    assert orf.orf_id == "orf14635"
    assert orf.start == 96951
    assert orf.end == 97093
    assert orf.strand == "-"
    assert orf.frame == 3
    assert orf.table_id == 11


def test_parse_orf_header_simple_format() -> None:
    """Test parsing simple header format without parent_id prefix."""
    # Simple format: >orf_id [parent_id frame frame_num start end]
    header = ">orf1 [JQ995537 frame +1 388 1230]"
    orf = _parse_orf_header_line(header, table_id=11)

    assert orf.parent_id == "JQ995537"
    assert orf.orf_id == "orf1"
    assert orf.start == 388
    assert orf.end == 1230
    assert orf.strand == "+"
    assert orf.frame == 1
    assert orf.table_id == 11


def test_parse_orf_header_positive_frame() -> None:
    """Test parsing header with positive frame."""
    header = ">NC_000913-orf1 [NC_000913 frame +2 100 400]"
    orf = _parse_orf_header_line(header, table_id=11)

    assert orf.strand == "+"
    assert orf.frame == 2


def test_parse_orf_header_negative_frame() -> None:
    """Test parsing header with negative frame."""
    header = ">NC_000913-orf2 [NC_000913 frame -1 500 800]"
    orf = _parse_orf_header_line(header, table_id=11)

    assert orf.strand == "-"
    assert orf.frame == 1


def test_parse_orf_header_invalid_format() -> None:
    """Test that invalid header format raises ValueError."""
    # Missing required components
    invalid_headers = [
        ">orf1",  # No brackets
        ">orf1 []",  # Empty brackets
        ">orf1 [JQ995537]",  # Missing frame info
        "orf1 [JQ995537 frame +1 388 1230]",  # Missing >
    ]

    for header in invalid_headers:
        with pytest.raises(ValueError, match="Header did not match expected format"):
            _parse_orf_header_line(header, table_id=11)
