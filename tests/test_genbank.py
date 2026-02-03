"""Tests for GenBank file parsing and CDS matching functionality."""

import tempfile
from pathlib import Path

import pytest

from genome_entropy.io.genbank import (
    GenBankCDS,
    extract_cds_features,
    match_orf_to_genbank_cds,
    read_genbank,
)


def create_test_genbank_file() -> str:
    """Create a minimal GenBank file for testing."""
    genbank_content = """LOCUS       TEST_SEQ                 300 bp    DNA     linear   BCT 01-JAN-2024
DEFINITION  Test sequence for GenBank parsing.
ACCESSION   TEST_SEQ
VERSION     TEST_SEQ.1
KEYWORDS    .
SOURCE      Test organism
  ORGANISM  Test organism
            Bacteria.
FEATURES             Location/Qualifiers
     source          1..300
                     /organism="Test organism"
     CDS             10..100
                     /codon_start=1
                     /transl_table=11
                     /product="test protein 1"
                     /translation="MKSLLTSLAVVSGFLATCVAETKQEQ"
     CDS             complement(150..250)
                     /codon_start=1
                     /transl_table=11
                     /product="test protein 2"
                     /translation="MQLLVLSCGQEDPKHLLKLRQF"
ORIGIN
        1 atgaaatccc ttctgacttc cctcgctgtc gtctccggct tcctcgccac ctgcgtggcc
       61 gagaccaagc aggagcagtg atagctcgat tatcgatcga tcgatcgatc gatcgatcga
      121 tcgatcgatc gatcgatcga tcgatcgatc tcgaaacgca gtttaaactt gagcaggcgc
      181 ttgaacttgg tctcgttcag agcgccgctg agcagcagca tgacgtagct agctagctag
      241 ctagctagct gtcaatacgg atcgatcgac gtatcagtac ggacatgcat acgtacgtac
      301
//
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gb", delete=False) as tmp:
        tmp.write(genbank_content)
        return tmp.name


def test_read_genbank() -> None:
    """Test reading DNA sequences from GenBank file."""
    genbank_file = create_test_genbank_file()

    try:
        sequences = read_genbank(genbank_file)

        assert len(sequences) == 1
        assert "TEST_SEQ.1" in sequences

        # Check that sequence is uppercase
        seq = sequences["TEST_SEQ.1"]
        assert seq == seq.upper()
        assert len(seq) == 300
    finally:
        Path(genbank_file).unlink(missing_ok=True)


def test_read_genbank_nonexistent_file() -> None:
    """Test that reading nonexistent GenBank file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_genbank("/nonexistent/file.gb")


def test_extract_cds_features() -> None:
    """Test extracting CDS features from GenBank file."""
    genbank_file = create_test_genbank_file()

    try:
        cds_features = extract_cds_features(genbank_file)

        assert len(cds_features) == 2

        # Check first CDS (forward strand)
        cds1 = cds_features[0]
        assert cds1.parent_id == "TEST_SEQ.1"
        assert cds1.start == 9  # 0-based
        assert cds1.end == 100  # exclusive
        assert cds1.strand == "+"
        assert cds1.protein_sequence == "MKSLLTSLAVVSGFLATCVAETKQEQ"

        # Check second CDS (reverse strand)
        cds2 = cds_features[1]
        assert cds2.parent_id == "TEST_SEQ.1"
        assert cds2.start == 149  # 0-based
        assert cds2.end == 250  # exclusive
        assert cds2.strand == "-"
        assert cds2.protein_sequence == "MQLLVLSCGQEDPKHLLKLRQF"
    finally:
        Path(genbank_file).unlink(missing_ok=True)


def test_match_orf_to_genbank_cds_exact_match() -> None:
    """Test matching ORF to GenBank CDS with exact C-terminal match."""
    # Create test CDS features
    cds_list = [
        GenBankCDS(
            parent_id="seq1",
            start=0,
            end=100,
            strand="+",
            protein_sequence="MKSLLTSLAVVSGFLATCVAETKQEQ",
        ),
    ]

    # ORF with same C-terminal sequence
    orf_aa = "MKSLLTSLAVVSGFLATCVAETKQEQ"

    result = match_orf_to_genbank_cds(orf_aa, cds_list, min_c_terminal_match=10)
    assert result is True


def test_match_orf_to_genbank_cds_partial_match() -> None:
    """Test matching ORF with different N-terminal but same C-terminal."""
    cds_list = [
        GenBankCDS(
            parent_id="seq1",
            start=0,
            end=100,
            strand="+",
            protein_sequence="MKSLLTSLAVVSGFLATCVAETKQEQ",
        ),
    ]

    # ORF with different start but same C-terminal (last 10 amino acids)
    orf_aa = "XXXXXXXXVVSGFLATCVAETKQEQ"  # Different N-terminal, same C-terminal

    result = match_orf_to_genbank_cds(orf_aa, cds_list, min_c_terminal_match=10)
    assert result is True


def test_match_orf_to_genbank_cds_no_match() -> None:
    """Test ORF that doesn't match any GenBank CDS."""
    cds_list = [
        GenBankCDS(
            parent_id="seq1",
            start=0,
            end=100,
            strand="+",
            protein_sequence="MKSLLTSLAVVSGFLATCVAETKQEQ",
        ),
    ]

    # ORF with completely different sequence
    orf_aa = "AAAAAAAAAAAAAAAAAAAAAAAAAA"

    result = match_orf_to_genbank_cds(orf_aa, cds_list, min_c_terminal_match=10)
    assert result is False


def test_match_orf_to_genbank_cds_with_stop_codon() -> None:
    """Test matching ORF with stop codon to CDS."""
    cds_list = [
        GenBankCDS(
            parent_id="seq1",
            start=0,
            end=100,
            strand="+",
            protein_sequence="MKSLLTSLAVVSGFLATCVAETKQEQ",
        ),
    ]

    # ORF with stop codon at end
    orf_aa = "MKSLLTSLAVVSGFLATCVAETKQEQ*"

    result = match_orf_to_genbank_cds(orf_aa, cds_list, min_c_terminal_match=10)
    assert result is True


def test_match_orf_to_genbank_cds_short_sequence() -> None:
    """Test matching short ORF sequence."""
    cds_list = [
        GenBankCDS(
            parent_id="seq1",
            start=0,
            end=30,
            strand="+",
            protein_sequence="MKSLLTSLA",
        ),
    ]

    # ORF shorter than min_c_terminal_match
    orf_aa = "MKSLLTSLA"

    result = match_orf_to_genbank_cds(orf_aa, cds_list, min_c_terminal_match=10)
    assert result is True


def test_match_orf_to_genbank_cds_empty_inputs() -> None:
    """Test matching with empty inputs."""
    # Empty ORF sequence
    result = match_orf_to_genbank_cds("", [], min_c_terminal_match=10)
    assert result is False

    # Empty CDS list
    result = match_orf_to_genbank_cds("MKSLLTSLA", [], min_c_terminal_match=10)
    assert result is False


def test_match_orf_to_genbank_cds_multiple_cds() -> None:
    """Test matching ORF against multiple CDS features."""
    cds_list = [
        GenBankCDS(
            parent_id="seq1",
            start=0,
            end=100,
            strand="+",
            protein_sequence="AAAAAAAAAAAAAAAAAAAAAAAAAA",
        ),
        GenBankCDS(
            parent_id="seq1",
            start=200,
            end=300,
            strand="+",
            protein_sequence="MKSLLTSLAVVSGFLATCVAETKQEQ",
        ),
    ]

    # ORF matching second CDS
    orf_aa = "MKSLLTSLAVVSGFLATCVAETKQEQ"

    result = match_orf_to_genbank_cds(orf_aa, cds_list, min_c_terminal_match=10)
    assert result is True


def test_genbank_cds_dataclass() -> None:
    """Test GenBankCDS dataclass creation."""
    cds = GenBankCDS(
        parent_id="TEST_SEQ",
        start=10,
        end=100,
        strand="+",
        protein_sequence="MKSLLTSLA",
    )

    assert cds.parent_id == "TEST_SEQ"
    assert cds.start == 10
    assert cds.end == 100
    assert cds.strand == "+"
    assert cds.protein_sequence == "MKSLLTSLA"
