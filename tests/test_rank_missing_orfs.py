"""Tests for the site-local putative-missing-ORF ranking script."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

import pandas as pd  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rank_missing_orfs.py"


def load_script():
    """Import the standalone ranking script by path."""
    spec = importlib.util.spec_from_file_location("rank_missing_orfs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_record(orfs):
    """Build one unified record from (orf_id, start, end, strand) tuples."""
    return {
        "schema_version": "2.2.0",
        "input_id": "genome_1",
        "input_dna_length": 1000,
        "features": {
            orf_id: {
                "orf_id": orf_id,
                "location": {
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "frame": 0,
                },
            }
            for orf_id, start, end, strand in orfs
        },
    }


def test_reverse_strand_coordinates_are_placed_on_the_genomic_axis() -> None:
    """Negative-strand get_orfs coordinates index the reverse complement."""
    module = load_script()

    intervals, unusable = module.collect_genomic_intervals(
        [[make_record([("orf_1", 100, 111, "+"), ("orf_2", 100, 111, "-")])]]
    )

    assert unusable == 0
    by_orf = intervals.set_index("orf_id")
    assert by_orf.loc["orf_1", "genomic_start"] == 99
    assert by_orf.loc["orf_1", "genomic_end"] == 111
    # The same raw coordinates on the reverse strand sit elsewhere entirely.
    assert by_orf.loc["orf_2", "genomic_start"] == 889
    assert by_orf.loc["orf_2", "genomic_end"] == 901


def test_unnormalisable_orfs_are_counted_not_silently_dropped() -> None:
    """A reverse-strand ORF with no record length is reported, not guessed."""
    module = load_script()

    record = make_record([("orf_1", 100, 111, "-")])
    del record["input_dna_length"]

    intervals, unusable = module.collect_genomic_intervals([[record]])

    assert intervals.empty
    assert unusable == 1


def test_opposite_strand_orfs_do_not_produce_spurious_overlaps() -> None:
    """Raw coordinates would collide; normalised genomic intervals do not."""
    module = load_script()

    predictions = pd.DataFrame(
        [
            {
                "input_id": "genome_1",
                "orf_id": "annotated",
                "genomic_start": 99,
                "genomic_end": 111,
                "in_genbank": True,
                "probability_in_genbank": 0.1,
                "protein_length": 40,
            },
            {
                "input_id": "genome_1",
                "orf_id": "candidate",
                "genomic_start": 889,
                "genomic_end": 901,
                "in_genbank": False,
                "probability_in_genbank": 0.9,
                "protein_length": 40,
            },
        ]
    )

    assert module.find_overlapping_annotated_orfs(predictions).empty


def test_genuine_overlap_is_still_detected() -> None:
    """Overlapping normalised intervals still yield a competing annotation."""
    module = load_script()

    predictions = pd.DataFrame(
        [
            {
                "input_id": "genome_1",
                "orf_id": "annotated",
                "genomic_start": 100,
                "genomic_end": 400,
                "genomic_strand": "+",
                "in_genbank": True,
                "probability_in_genbank": 0.1,
                "protein_length": 100,
            },
            {
                "input_id": "genome_1",
                "orf_id": "candidate",
                "genomic_start": 150,
                "genomic_end": 450,
                "genomic_strand": "-",
                "in_genbank": False,
                "probability_in_genbank": 0.9,
                "protein_length": 100,
            },
        ]
    )

    result = module.find_overlapping_annotated_orfs(predictions)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["competing_orf_id"] == "annotated"
    assert row["overlap_nt"] == 250
    assert row["competing_genomic_strand"] == "+"
    assert row["overlap_fraction_shorter"] == pytest.approx(250 / 300)


def test_orfs_without_a_genomic_interval_are_excluded() -> None:
    """Missing normalised coordinates must not be compared as raw values."""
    module = load_script()

    predictions = pd.DataFrame(
        [
            {
                "input_id": "genome_1",
                "orf_id": "annotated",
                "genomic_start": 100.0,
                "genomic_end": 400.0,
                "in_genbank": True,
                "probability_in_genbank": 0.1,
                "protein_length": 100,
            },
            {
                "input_id": "genome_1",
                "orf_id": "candidate",
                "genomic_start": float("nan"),
                "genomic_end": float("nan"),
                "in_genbank": False,
                "probability_in_genbank": 0.9,
                "protein_length": 100,
            },
        ]
    )

    assert module.find_overlapping_annotated_orfs(predictions).empty
