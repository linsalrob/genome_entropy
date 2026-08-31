"""Tests for GenBank parsing and coordinate-anchored CDS matching."""

import tempfile
from pathlib import Path
from typing import Literal, Optional

import pytest

from genome_entropy.io.genbank import (
    MIN_CDS_OVERLAP_FRACTION,
    MIN_SHARED_AA_IDENTITY,
    CodingInterval,
    GenBankCDS,
    amino_acids_are_compatible,
    calculate_interval_overlap,
    coding_phase_is_compatible,
    evaluate_orf_genbank_cds_match,
    extract_cds_features,
    match_orf_to_genbank_cds,
    normalise_orf_coordinates,
    normalise_orf_interval,
    normalise_protein_sequence,
    orf_matches_genbank_cds,
    read_genbank,
)
from genome_entropy.orf.types import OrfRecord


def create_test_genbank_file() -> str:
    """Create a minimal file with qualifier and fallback CDS translations."""
    genbank_content = """LOCUS       TEST_SEQ                  30 bp    DNA     linear   BCT 01-JAN-2024
DEFINITION  Test sequence for GenBank parsing.
ACCESSION   TEST_SEQ
VERSION     TEST_SEQ.1
KEYWORDS    .
SOURCE      Test organism
  ORGANISM  Test organism
            Bacteria.
FEATURES             Location/Qualifiers
     source          1..30
                     /organism="Test organism"
     CDS             2..10
                     /codon_start=1
                     /transl_table=15
                     /locus_tag="WITH_TRANSLATION"
                     /translation="MK*"
     CDS             complement(21..29)
                     /codon_start=1
                     /transl_table=11
                     /locus_tag="FALLBACK_TRANSLATION"
ORIGIN
        1 aatgaaataa cccccccccc ttatttcat c
//
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gb", delete=False) as tmp:
        tmp.write(genbank_content)
        return tmp.name


def make_orf(
    sequence: str,
    *,
    parent_id: str = "seq1",
    genomic_start: int = 99,
    genomic_end: Optional[int] = None,
    strand: Literal["+", "-"] = "+",
    record_length: int = 5000,
    orf_id: str = "orf1",
    table_id: int = 11,
) -> OrfRecord:
    """Create an ORF from a desired zero-based, half-open genomic interval."""
    if genomic_end is None:
        genomic_end = genomic_start + 3 * len(normalise_protein_sequence(sequence))
    if strand == "+":
        start = genomic_start + 1
        end = genomic_end
    else:
        start = record_length - genomic_end + 1
        end = record_length - genomic_start
    return OrfRecord(
        parent_id=parent_id,
        orf_id=orf_id,
        start=start,
        end=end,
        strand=strand,
        frame=1,
        nt_sequence="",
        aa_sequence=sequence,
        table_id=table_id,
        has_start_codon="M" in sequence,
        has_stop_codon="*" in sequence,
    )


def make_cds(
    sequence: str,
    *,
    parent_id: str = "seq1",
    start: int = 99,
    end: Optional[int] = None,
    strand: Literal["+", "-"] = "+",
    record_length: int = 5000,
    feature_id: str = "CDS_1",
    translation_table: int = 11,
    partial: bool = False,
) -> GenBankCDS:
    """Create a CDS with zero-based, half-open genomic coordinates."""
    if end is None:
        end = start + 3 * len(normalise_protein_sequence(sequence))
    return GenBankCDS(
        parent_id,
        start,
        end,
        strand,
        sequence,
        record_length=record_length,
        feature_id=feature_id,
        translation_table=translation_table,
        partial=partial,
    )


def test_read_genbank() -> None:
    """Read sequences with uppercase bases and versioned record identifiers."""
    path = create_test_genbank_file()
    try:
        sequences = read_genbank(path)
        assert list(sequences) == ["TEST_SEQ.1"]
        assert sequences["TEST_SEQ.1"].isupper()
        assert len(sequences["TEST_SEQ.1"]) == 30
    finally:
        Path(path).unlink(missing_ok=True)


def test_read_genbank_nonexistent_file() -> None:
    """A missing input file is reported directly."""
    with pytest.raises(FileNotFoundError):
        read_genbank("/nonexistent/file.gb")


def test_extract_cds_uses_qualifier_and_fallback_translations() -> None:
    """Prefer /translation and otherwise translate the feature-specific DNA."""
    path = create_test_genbank_file()
    try:
        first, second = extract_cds_features(path)
        assert first.feature_id == "WITH_TRANSLATION"
        assert first.protein_sequence == "MK*"
        assert first.translation_table == 15
        assert first.record_length == 30
        assert (first.start, first.end, first.strand) == (1, 10, "+")

        assert second.feature_id == "FALLBACK_TRANSLATION"
        assert second.protein_sequence == "MK*"
        assert second.translation_table == 11
        assert (second.start, second.end, second.strand) == (20, 29, "-")
    finally:
        Path(path).unlink(missing_ok=True)


def test_codon_start_is_applied_to_interval_and_fallback_translation() -> None:
    """codon_start shifts the biological coding start before translation."""
    content = """LOCUS       CODON_START               10 bp    DNA     linear   BCT 01-JAN-2024
ACCESSION   CODON_START
VERSION     CODON_START.1
FEATURES             Location/Qualifiers
     CDS             1..10
                     /codon_start=2
                     /transl_table=11
                     /locus_tag="OFFSET"
ORIGIN
        1 aatgaaataa
//
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gb", delete=False) as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        cds = extract_cds_features(path)[0]
        assert (cds.start, cds.end, cds.codon_start) == (1, 10, 2)
        assert cds.protein_sequence == "MK*"
        orf = make_orf(
            "MK*",
            parent_id="CODON_START.1",
            genomic_start=1,
            genomic_end=10,
            record_length=10,
        )
        assert orf_matches_genbank_cds(orf, cds)
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("source_qualifier", "pipeline_table", "expected_table"),
    [
        ("                     /transl_table=15\n", 11, 15),
        ("", 15, 15),
    ],
)
def test_translation_table_fallback_order(
    source_qualifier: str,
    pipeline_table: int,
    expected_table: int,
) -> None:
    """A record/source table precedes the pipeline table for fallback translation."""
    content = f"""LOCUS       TABLE_FALLBACK              6 bp    DNA     linear   BCT 01-JAN-2024
ACCESSION   TABLE_FALLBACK
VERSION     TABLE_FALLBACK.1
FEATURES             Location/Qualifiers
     source          1..6
{source_qualifier}     CDS             1..6
                     /locus_tag="TABLE_CDS"
ORIGIN
        1 atgtag
//
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gb", delete=False) as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        cds = extract_cds_features(path, pipeline_table_id=pipeline_table)[0]
        assert cds.translation_table == expected_table
        assert cds.protein_sequence == "MQ"
    finally:
        Path(path).unlink(missing_ok=True)


def test_partial_simple_cds_can_match() -> None:
    """A safe contiguous partial location retains its boundary information."""
    content = """LOCUS       PARTIAL                    10 bp    DNA     linear   BCT 01-JAN-2024
ACCESSION   PARTIAL
VERSION     PARTIAL.1
FEATURES             Location/Qualifiers
     CDS             <2..10
                     /locus_tag="PARTIAL_CDS"
                     /translation="MK*"
ORIGIN
        1 aatgaaataa
//
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gb", delete=False) as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        cds = extract_cds_features(path)[0]
        assert cds.partial
        orf = make_orf(
            "MK*",
            parent_id="PARTIAL.1",
            genomic_start=1,
            genomic_end=10,
            record_length=10,
        )
        assert orf_matches_genbank_cds(orf, cds)
    finally:
        Path(path).unlink(missing_ok=True)


def test_compound_location_is_skipped() -> None:
    """Joined and origin-crossing CDS locations are not silently flattened."""
    content = """LOCUS       JOINED                     30 bp    DNA     circular BCT 01-JAN-2024
ACCESSION   JOINED
VERSION     JOINED.1
FEATURES             Location/Qualifiers
     CDS             join(25..30,1..6)
                     /locus_tag="JOINED_CDS"
                     /translation="MKK"
ORIGIN
        1 atgaaaaaaa aaaaaaaaaa aaaaaaataa
//
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gb", delete=False) as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        assert extract_cds_features(path) == []
    finally:
        Path(path).unlink(missing_ok=True)


def test_coordinate_normalisation_on_both_strands() -> None:
    """Convert public ORF coordinates to one genomic coordinate convention."""
    forward = make_orf("MABC", genomic_start=99, genomic_end=111)
    reverse = make_orf(
        "MABC",
        genomic_start=399,
        genomic_end=411,
        strand="-",
        record_length=1000,
    )
    assert normalise_orf_coordinates(forward, 1000) == CodingInterval(99, 111, "+")
    assert normalise_orf_coordinates(reverse, 1000) == CodingInterval(399, 411, "-")


def test_coordinate_only_normalisation_matches_record_normalisation() -> None:
    """Callers holding serialised locations share one normalisation rule."""
    assert normalise_orf_interval(100, 111, "+", 1000) == CodingInterval(99, 111, "+")
    assert normalise_orf_interval(590, 601, "-", 1000) == CodingInterval(399, 411, "-")

    with pytest.raises(ValueError):
        normalise_orf_interval(100, 111, "?", 1000)
    with pytest.raises(ValueError):
        normalise_orf_interval(590, 601, "-", None)


def test_invalid_coding_intervals_are_rejected() -> None:
    """Central interval validation rejects negative and empty intervals."""
    with pytest.raises(ValueError):
        CodingInterval(-1, 3, "+")
    with pytest.raises(ValueError):
        CodingInterval(3, 3, "+")


def test_exact_full_length_match() -> None:
    """An existing exact same-locus match remains successful."""
    sequence = "MABCDEFGHI"
    assert orf_matches_genbank_cds(make_orf(sequence), make_cds(sequence))


def test_alternative_start_and_contained_cds_match() -> None:
    """A same-frame N-terminal extension does not prevent a CDS match."""
    shared = "M" + "A" * 99
    extension = "QRSKLMNP"
    orf = make_orf(
        extension + shared,
        genomic_start=75,
        genomic_end=399,
    )
    cds = make_cds(shared, start=99, end=399)
    assert orf_matches_genbank_cds(orf, cds)


def test_shorter_cds_contained_within_longer_orf_matches() -> None:
    """Both terminal boundaries may differ when overlap remains substantial."""
    core = "A" * 100
    orf = make_orf("QQQQQ" + core + "RRRRR", genomic_start=84, genomic_end=414)
    cds = make_cds(core, start=99, end=399)
    assert orf_matches_genbank_cds(orf, cds)


@pytest.mark.parametrize(
    ("orf_sequence", "cds_sequence"),
    [
        ("MABCDKFGHI", "MABCDXFGHI"),
        ("MABCDXFGHI", "MABCDKFGHI"),
        ("MABCDKFGHI*", "MABCDXFGHI"),
        ("MABCDKFGHI", "MABCDXFGHI*"),
        (" mabcdkfghi* ", "MABCDXFGHI"),
    ],
)
def test_x_terminal_stop_case_and_whitespace_are_supported(
    orf_sequence: str,
    cds_sequence: str,
) -> None:
    """X is an aligned wildcard and terminal formatting is ignored."""
    assert orf_matches_genbank_cds(
        make_orf(orf_sequence, genomic_end=129),
        make_cds(cds_sequence, end=129),
    )


def test_cds_specific_translation_table_does_not_need_pipeline_table() -> None:
    """Matching preserves a feature's annotation-provider translation."""
    orf = make_orf("MPEPTIDE", table_id=11)
    cds = make_cds("MPEPTIDE", translation_table=15)
    assert cds.translation_table == 15
    assert orf_matches_genbank_cds(orf, cds)


def test_one_codon_difference_at_each_end_matches() -> None:
    """Coordinate alignment tolerates table-dependent terminal differences."""
    shared = "A" * 100
    orf = make_orf("V" + shared, genomic_start=96, genomic_end=399)
    cds = make_cds(shared + "Q", start=99, end=402, translation_table=15)
    result = evaluate_orf_genbank_cds_match(orf, cds)
    assert result.matched
    assert result.compared_aa == 100
    assert result.overlap_fraction > 0.99


def test_reverse_strand_alternative_start_matches() -> None:
    """Phase and protein offsets follow high-to-low biological translation."""
    shared = "M" + "A" * 99
    orf = make_orf(
        "Q" * 10 + shared,
        genomic_start=99,
        genomic_end=429,
        strand="-",
        record_length=1000,
    )
    cds = make_cds(
        shared,
        start=99,
        end=399,
        strand="-",
        record_length=1000,
    )
    assert orf_matches_genbank_cds(orf, cds)


def test_same_coordinates_different_strands_do_not_match() -> None:
    """Strand is a mandatory genomic constraint."""
    assert not orf_matches_genbank_cds(
        make_orf("MABCDEFGHI", strand="+"),
        make_cds("MABCDEFGHI", strand="-"),
    )


def test_different_parent_sequences_do_not_match() -> None:
    """Coordinates cannot identify the same gene on different records."""
    assert not orf_matches_genbank_cds(
        make_orf("MABCDEFGHI", parent_id="one"),
        make_cds("MABCDEFGHI", parent_id="two"),
    )


def test_strong_overlap_in_different_codon_phase_does_not_match() -> None:
    """Nearly identical intervals offset by one nucleotide fail phase checking."""
    orf = make_orf("A" * 100, genomic_start=99, genomic_end=399)
    cds = make_cds("A" * 99, start=100, end=397)
    result = evaluate_orf_genbank_cds_match(orf, cds)
    assert not result.matched
    assert not result.phase_compatible
    assert result.reason == "different codon phase"


@pytest.mark.parametrize(
    ("overlap_nt", "expected"),
    [
        (270, True),
        (267, False),
    ],
)
def test_overlap_threshold_boundary(overlap_nt: int, expected: bool) -> None:
    """Exactly 90% overlap passes while the next whole codon below it fails."""
    orf = make_orf("A" * 100, genomic_start=99, genomic_end=399)
    cds_start = 399 - overlap_nt
    cds = make_cds("A" * 100, start=cds_start, end=cds_start + 300)
    result = evaluate_orf_genbank_cds_match(orf, cds)
    assert result.overlap_fraction == pytest.approx(overlap_nt / 300)
    assert result.matched is expected
    assert MIN_CDS_OVERLAP_FRACTION == 0.90


def test_conserved_domain_without_near_complete_overlap_does_not_match() -> None:
    """A shared local domain cannot override insufficient genomic overlap."""
    orf = make_orf("A" * 100, genomic_start=99, genomic_end=399)
    cds = make_cds("A" * 100, start=309, end=609)
    assert not orf_matches_genbank_cds(orf, cds)


@pytest.mark.parametrize(
    ("mismatches", "expected"),
    [
        (2, True),
        (3, False),
    ],
)
def test_identity_threshold_boundary(mismatches: int, expected: bool) -> None:
    """Exactly 98% compatibility passes and 97% fails."""
    orf_sequence = "A" * 100
    cds_sequence = "B" * mismatches + "A" * (100 - mismatches)
    result = evaluate_orf_genbank_cds_match(
        make_orf(orf_sequence),
        make_cds(cds_sequence),
    )
    assert result.identity == pytest.approx((100 - mismatches) / 100)
    assert result.matched is expected
    assert MIN_SHARED_AA_IDENTITY == 0.98


def test_high_overlap_with_poor_identity_does_not_match() -> None:
    """Genomic agreement cannot compensate for unrelated translations."""
    assert not orf_matches_genbank_cds(
        make_orf("A" * 100),
        make_cds("R" * 100),
    )


def test_translation_must_cover_coordinate_aligned_overlap() -> None:
    """A short local protein fragment cannot match a much larger interval."""
    orf = make_orf("A" * 100, genomic_start=99, genomic_end=399)
    cds = make_cds("A" * 20, start=99, end=399)
    result = evaluate_orf_genbank_cds_match(orf, cds)
    assert not result.matched
    assert "do not cover" in result.reason


def test_existing_true_negative_remains_false() -> None:
    """An unrelated same-coordinate protein remains unmatched."""
    orf = make_orf("MABCDEFGHI")
    cds_list = [make_cds("MRRRRRRRRR"), make_cds("MABCDEFGHI", parent_id="other")]
    assert not match_orf_to_genbank_cds(orf, cds_list)


def test_match_orf_to_genbank_cds_checks_later_candidates() -> None:
    """A failed candidate does not prevent a later valid CDS match."""
    orf = make_orf("MABCDEFGHI")
    candidates = [make_cds("MRRRRRRRRR"), make_cds("MABXDEFGHI")]
    assert match_orf_to_genbank_cds(orf, candidates)


@pytest.mark.parametrize(
    ("residue_a", "residue_b", "expected"),
    [
        ("A", "A", True),
        ("X", "K", True),
        ("K", "X", True),
        ("B", "D", False),
        ("Z", "Q", False),
        ("*", "X", False),
    ],
)
def test_amino_acids_are_compatible(
    residue_a: str,
    residue_b: str,
    expected: bool,
) -> None:
    """Only X is a wildcard; other ambiguous symbols remain specific."""
    assert amino_acids_are_compatible(residue_a, residue_b) is expected


def test_interval_overlap_and_phase_helpers() -> None:
    """Helper calculations expose their zero-based genomic semantics."""
    first = CodingInterval(99, 399, "+")
    second = CodingInterval(129, 399, "+")
    assert calculate_interval_overlap(first, second) == (270, 1.0)
    assert coding_phase_is_compatible(first, second)
    assert not coding_phase_is_compatible(first, CodingInterval(100, 400, "+"))


def test_kj206559_orf110_regression() -> None:
    """KJ206559 orf110 matches CDS_0138 across 8-aa extension and four Xs."""
    shared = list("M" + "A" * 703)
    cds_shared = shared.copy()
    for position in (80, 320, 350, 370):
        cds_shared[position] = "X"
    orf = make_orf(
        "SSVKDERY" + "".join(shared),
        parent_id="KJ206559",
        genomic_start=99452,
        genomic_end=101591,
        record_length=142096,
        orf_id="orf110",
    )
    cds = make_cds(
        "".join(cds_shared) + "*",
        parent_id="KJ206559",
        start=99476,
        end=101591,
        record_length=142096,
        feature_id="KJ206559_CDS_0138",
    )
    result = evaluate_orf_genbank_cds_match(orf, cds)
    assert result.matched
    assert result.wildcard_aa == 4
    assert result.compared_aa == 704


def test_px673949_orf76_regression() -> None:
    """PX673949.1 orf76 matches CDS_0126 across 37-aa extension and four Xs."""
    shared = list("M" + "A" * 691)
    cds_shared = shared.copy()
    for position in (220, 280, 400, 650):
        cds_shared[position] = "X"
    orf = make_orf(
        "Q" * 37 + "".join(shared),
        parent_id="PX673949.1",
        genomic_start=84411,
        genomic_end=86601,
        record_length=100813,
        orf_id="orf76",
    )
    cds = make_cds(
        "".join(cds_shared) + "*",
        parent_id="PX673949.1",
        start=84522,
        end=86601,
        record_length=100813,
        feature_id="PX673949.1_CDS_0126",
    )
    result = evaluate_orf_genbank_cds_match(orf, cds)
    assert result.matched
    assert result.wildcard_aa == 4
    assert result.compared_aa == 692


def test_nc055901_orf52_regression() -> None:
    """NC_055901 orf52 matches table-15 CDS_0090 across shifted termini."""
    shared = "A" * 626
    orf = make_orf(
        "V" + shared,
        parent_id="NC_055901",
        genomic_start=92434,
        genomic_end=94318,
        record_length=94826,
        orf_id="orf52",
        table_id=11,
    )
    cds = make_cds(
        shared + "Q*",
        parent_id="NC_055901",
        start=92437,
        end=94321,
        record_length=94826,
        feature_id="NC_055901_CDS_0090",
        translation_table=15,
    )
    result = evaluate_orf_genbank_cds_match(orf, cds)
    assert result.matched
    assert result.compared_aa == 626
    assert result.identity == 1.0
