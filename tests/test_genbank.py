"""Tests for GenBank file parsing and CDS matching functionality."""

import tempfile
from pathlib import Path
from typing import Literal

import pytest

from genome_entropy.io.genbank import (
    GenBankCDS,
    extract_cds_features,
    match_orf_to_genbank_cds,
    orf_matches_genbank_cds,
    read_genbank,
)
from genome_entropy.orf.types import OrfRecord


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


def make_orf(
    sequence: str,
    *,
    parent_id: str = "seq1",
    start: int = 10,
    end: int = 100,
    strand: Literal["+", "-"] = "+",
) -> OrfRecord:
    """Create an ORF with get_orfs-style one-based inclusive coordinates."""
    return OrfRecord(
        parent_id=parent_id,
        orf_id="orf1",
        start=start,
        end=end,
        strand=strand,
        frame=1,
        nt_sequence="ATG",
        aa_sequence=sequence,
        table_id=11,
        has_start_codon=True,
        has_stop_codon=True,
    )


def make_cds(
    sequence: str,
    *,
    parent_id: str = "seq1",
    start: int = 9,
    end: int = 100,
    strand: str = "+",
) -> GenBankCDS:
    """Create a CDS with Biopython-style zero-based half-open coordinates."""
    return GenBankCDS(parent_id, start, end, strand, sequence)


@pytest.mark.parametrize(
    ("orf_sequence", "cds_sequence"),
    [
        ("MABCDEFGHIJK", "MABCDEFGHIJK"),
        ("MABCDEFGHIJK", "XABCDEFGHIJK"),
        ("MABCDEFGHIJK", "XDEFGHIJK"),
        ("XDEFGHIJK", "MABCDEFGHIJK"),
        ("MABCDEFGHIJK*", "MABCDEFGHIJK"),
        ("MABCDEFGHIJK", "MABCDEFGHIJK*"),
        (" MABCD EFGHIJK* ", "mabcdefghijk*"),
    ],
)
def test_orf_matches_genbank_cds_valid_c_terminal_matches(
    orf_sequence: str,
    cds_sequence: str,
) -> None:
    """Match identical, alternative-start, suffix, stop, case, and space forms."""
    assert orf_matches_genbank_cds(
        make_orf(orf_sequence),
        make_cds(cds_sequence),
    )


@pytest.mark.parametrize(
    ("orf_sequence", "cds_sequence"),
    [
        ("MABCDEFGHIJK", "MABCDEFGHIJX"),
        ("MABCDEFGHIJK", "MABCXEFGHIJK"),
        ("MABCDEFGHIJK", "XEFGHIJ"),
        ("MABCDEFGHIJK", "MABCDE"),
    ],
)
def test_orf_matches_genbank_cds_rejects_non_terminal_matches(
    orf_sequence: str,
    cds_sequence: str,
) -> None:
    """Reject final mismatches, region mismatches, substrings, and prefixes."""
    assert not orf_matches_genbank_cds(
        make_orf(orf_sequence),
        make_cds(cds_sequence),
    )


def test_orf_matches_genbank_cds_requires_same_stop_coordinate() -> None:
    """Matching sequences at different biological stops are distinct proteins."""
    assert not orf_matches_genbank_cds(
        make_orf("MABCDEFGHIJK", end=100),
        make_cds("MABCDEFGHIJK", end=101),
    )


def test_orf_matches_genbank_cds_requires_same_strand() -> None:
    """Matching sequences on different strands are distinct proteins."""
    assert not orf_matches_genbank_cds(
        make_orf("MABCDEFGHIJK", strand="+"),
        make_cds("MABCDEFGHIJK", strand="-"),
    )


def test_orf_matches_genbank_cds_requires_same_parent_sequence() -> None:
    """Coordinates on different parent sequences do not identify one protein."""
    assert not orf_matches_genbank_cds(
        make_orf("MABCDEFGHIJK", parent_id="seq1"),
        make_cds("MABCDEFGHIJK", parent_id="seq2"),
    )


def test_orf_matches_genbank_cds_uses_reverse_strand_low_coordinate() -> None:
    """A reverse-strand stop maps from one-based ORF to zero-based CDS start."""
    orf = make_orf("MABCDEFGHIJK", start=150, end=250, strand="-")

    assert orf_matches_genbank_cds(
        orf,
        make_cds("MABCDEFGHIJK", start=149, end=250, strand="-"),
    )
    assert not orf_matches_genbank_cds(
        orf,
        make_cds("MABCDEFGHIJK", start=150, end=250, strand="-"),
    )


@pytest.mark.parametrize(
    ("orf_sequence", "cds_sequence"),
    [
        ("", "MABC"),
        ("MABC", ""),
        ("M", "MABC"),
        ("MABC", "M"),
        ("*", "MABC"),
        ("MABC", "*"),
    ],
)
def test_orf_matches_genbank_cds_rejects_empty_comparison_sequences(
    orf_sequence: str,
    cds_sequence: str,
) -> None:
    """Empty and one-residue proteins cannot establish a C-terminal match."""
    assert not orf_matches_genbank_cds(
        make_orf(orf_sequence),
        make_cds(cds_sequence),
    )


def test_match_orf_to_genbank_cds_matches_any_eligible_cds() -> None:
    """Preserve successful matching when the eligible CDS is later in the list."""
    orf = make_orf("MKSLLTSLAVVSGFLATCVAETKQEQ")
    cds_list = [
        make_cds("MAAAAAAAAAAAAAAAAAAAAAAAAA", end=100),
        make_cds("XKSLLTSLAVVSGFLATCVAETKQEQ", end=100),
    ]

    assert match_orf_to_genbank_cds(orf, cds_list)
    assert not match_orf_to_genbank_cds(orf, [])


def test_extracted_forward_and_reverse_cds_match_detected_orfs() -> None:
    """Regression-test parsed CDS coordinates with partial and altered starts."""
    genbank_file = create_test_genbank_file()

    try:
        forward_cds, reverse_cds = extract_cds_features(genbank_file)
        forward_cds.protein_sequence = "XETKQEQ"
        reverse_cds.protein_sequence = "XQEDPKHLLKLRQF"

        forward_orf = make_orf(
            "MKSLLTSLAVVSGFLATCVAETKQEQ",
            parent_id="TEST_SEQ.1",
            start=10,
            end=100,
            strand="+",
        )
        reverse_orf = make_orf(
            "MQLLVLSCGQEDPKHLLKLRQF",
            parent_id="TEST_SEQ.1",
            start=150,
            end=250,
            strand="-",
        )

        assert match_orf_to_genbank_cds(forward_orf, [forward_cds])
        assert match_orf_to_genbank_cds(reverse_orf, [reverse_cds])
    finally:
        Path(genbank_file).unlink(missing_ok=True)


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
