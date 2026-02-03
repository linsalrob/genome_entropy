"""Tests for unified JSON output format."""

import pytest
from genome_entropy.pipeline.runner import PipelineResult
from genome_entropy.pipeline.types import (
    UnifiedPipelineResult,
    UnifiedFeature,
    FeatureLocation,
    FeatureDNA,
    FeatureProtein,
    FeatureThreeDi,
    FeatureMetadata,
    FeatureEntropy,
)
from genome_entropy.orf.types import OrfRecord
from genome_entropy.translate.translator import ProteinRecord
from genome_entropy.encode3di.types import ThreeDiRecord
from genome_entropy.entropy.shannon import EntropyReport
from genome_entropy.io.jsonio import convert_pipeline_result_to_unified, to_json_dict


def test_unified_feature_structure():
    """Test that UnifiedFeature has the expected hierarchical structure."""
    feature = UnifiedFeature(
        orf_id="test_orf_1",
        location=FeatureLocation(start=0, end=30, strand="+", frame=0),
        dna=FeatureDNA(nt_sequence="ATGGCTAGC", length=9),
        protein=FeatureProtein(aa_sequence="MAS", length=3),
        three_di=FeatureThreeDi(
            encoding="ABC",
            length=3,
            method="prostt5_aa2fold",
            model_name="test_model",
            inference_device="cpu",
        ),
        metadata=FeatureMetadata(
            parent_id="test_seq",
            table_id=11,
            has_start_codon=True,
            has_stop_codon=False,
            in_genbank=False,
        ),
        entropy=FeatureEntropy(
            dna_entropy=1.5, protein_entropy=1.2, three_di_entropy=0.8
        ),
    )

    # Verify structure
    assert feature.orf_id == "test_orf_1"
    assert feature.location.start == 0
    assert feature.dna.nt_sequence == "ATGGCTAGC"
    assert feature.protein.aa_sequence == "MAS"
    assert feature.three_di.encoding == "ABC"
    assert feature.metadata.parent_id == "test_seq"
    assert feature.entropy.dna_entropy == 1.5


def test_convert_pipeline_result_to_unified():
    """Test conversion from old PipelineResult to new UnifiedPipelineResult."""
    # Create test data in old format
    orf = OrfRecord(
        parent_id="test_seq",
        orf_id="orf_1",
        start=0,
        end=30,
        strand="+",
        frame=0,
        nt_sequence="ATGGCTAGCTAGCTAGCTAGCTAGCTAG",
        aa_sequence="MASSSSSS",
        table_id=11,
        has_start_codon=True,
        has_stop_codon=False,
        in_genbank=False,
    )

    protein = ProteinRecord(orf=orf, aa_sequence="MASSSSSS", aa_length=8)

    three_di = ThreeDiRecord(
        protein=protein,
        three_di="AAAAAAAA",
        method="prostt5_aa2fold",
        model_name="test_model",
        inference_device="cpu",
    )

    entropy = EntropyReport(
        dna_entropy_global=1.8,
        orf_nt_entropy={"orf_1": 1.2},
        protein_aa_entropy={"orf_1": 0.8},
        three_di_entropy={"orf_1": 0.0},
        alphabet_sizes={"dna": 4, "protein": 20, "three_di": 20},
    )

    old_result = PipelineResult(
        input_id="test_seq",
        input_dna_length=100,
        orfs=[orf],
        proteins=[protein],
        three_dis=[three_di],
        entropy=entropy,
    )

    # Convert to unified format
    unified = convert_pipeline_result_to_unified(old_result)

    # Verify top-level fields
    assert isinstance(unified, UnifiedPipelineResult)
    assert unified.schema_version == "2.0.0"
    assert unified.input_id == "test_seq"
    assert unified.input_dna_length == 100
    assert unified.dna_entropy_global == 1.8
    assert unified.alphabet_sizes == {"dna": 4, "protein": 20, "three_di": 20}

    # Verify features dictionary
    assert len(unified.features) == 1
    assert "orf_1" in unified.features

    # Verify feature structure
    feature = unified.features["orf_1"]
    assert feature.orf_id == "orf_1"

    # Verify location
    assert feature.location.start == 0
    assert feature.location.end == 30
    assert feature.location.strand == "+"
    assert feature.location.frame == 0

    # Verify DNA
    assert feature.dna.nt_sequence == "ATGGCTAGCTAGCTAGCTAGCTAGCTAG"
    assert feature.dna.length == 28

    # Verify protein
    assert feature.protein.aa_sequence == "MASSSSSS"
    assert feature.protein.length == 8

    # Verify 3Di
    assert feature.three_di.encoding == "AAAAAAAA"
    assert feature.three_di.length == 8
    assert feature.three_di.method == "prostt5_aa2fold"
    assert feature.three_di.model_name == "test_model"
    assert feature.three_di.inference_device == "cpu"

    # Verify metadata
    assert feature.metadata.parent_id == "test_seq"
    assert feature.metadata.table_id == 11
    assert feature.metadata.has_start_codon is True
    assert feature.metadata.has_stop_codon is False
    assert feature.metadata.in_genbank is False

    # Verify entropy
    assert feature.entropy.dna_entropy == 1.2
    assert feature.entropy.protein_entropy == 0.8
    assert feature.entropy.three_di_entropy == 0.0


def test_conversion_handles_multiple_features():
    """Test that conversion handles multiple ORFs correctly."""
    # Create multiple ORFs
    orfs = []
    proteins = []
    three_dis = []

    for i in range(3):
        orf_id = f"orf_{i}"
        orf = OrfRecord(
            parent_id="test_seq",
            orf_id=orf_id,
            start=i * 30,
            end=(i + 1) * 30,
            strand="+",
            frame=0,
            nt_sequence="A" * 30,
            aa_sequence="K" * 10,
            table_id=11,
            has_start_codon=True,
            has_stop_codon=True,
            in_genbank=False,
        )
        orfs.append(orf)

        protein = ProteinRecord(orf=orf, aa_sequence="K" * 10, aa_length=10)
        proteins.append(protein)

        three_di = ThreeDiRecord(
            protein=protein,
            three_di="A" * 10,
            method="prostt5_aa2fold",
            model_name="test",
            inference_device="cpu",
        )
        three_dis.append(three_di)

    entropy = EntropyReport(
        dna_entropy_global=2.0,
        orf_nt_entropy={f"orf_{i}": 1.0 for i in range(3)},
        protein_aa_entropy={f"orf_{i}": 0.5 for i in range(3)},
        three_di_entropy={f"orf_{i}": 0.0 for i in range(3)},
        alphabet_sizes={"dna": 4, "protein": 20, "three_di": 20},
    )

    old_result = PipelineResult(
        input_id="test_seq",
        input_dna_length=300,
        orfs=orfs,
        proteins=proteins,
        three_dis=three_dis,
        entropy=entropy,
    )

    # Convert
    unified = convert_pipeline_result_to_unified(old_result)

    # Verify all features are present
    assert len(unified.features) == 3
    for i in range(3):
        assert f"orf_{i}" in unified.features
        feature = unified.features[f"orf_{i}"]
        assert feature.orf_id == f"orf_{i}"
        assert feature.location.start == i * 30
        assert feature.entropy.dna_entropy == 1.0


def test_no_data_loss_in_conversion():
    """Verify that no fields are lost during conversion."""
    # Create comprehensive test data
    orf = OrfRecord(
        parent_id="parent_123",
        orf_id="orf_test",
        start=100,
        end=250,
        strand="-",
        frame=2,
        nt_sequence="ATGCGATCGATCG",
        aa_sequence="MRSID",
        table_id=4,
        has_start_codon=False,
        has_stop_codon=True,
        in_genbank=True,
    )

    protein = ProteinRecord(orf=orf, aa_sequence="MRSID", aa_length=5)

    three_di = ThreeDiRecord(
        protein=protein,
        three_di="ABCDE",
        method="prostt5_aa2fold",
        model_name="Rostlab/ProstT5",
        inference_device="cuda",
    )

    entropy = EntropyReport(
        dna_entropy_global=3.5,
        orf_nt_entropy={"orf_test": 2.1},
        protein_aa_entropy={"orf_test": 1.8},
        three_di_entropy={"orf_test": 1.5},
        alphabet_sizes={"dna": 4, "protein": 20, "three_di": 20},
    )

    old_result = PipelineResult(
        input_id="seq_xyz",
        input_dna_length=500,
        orfs=[orf],
        proteins=[protein],
        three_dis=[three_di],
        entropy=entropy,
    )

    # Convert
    unified = convert_pipeline_result_to_unified(old_result)
    feature = unified.features["orf_test"]

    # Verify ALL fields are present
    # Top level
    assert unified.input_id == "seq_xyz"
    assert unified.input_dna_length == 500
    assert unified.dna_entropy_global == 3.5

    # Location
    assert feature.location.start == 100
    assert feature.location.end == 250
    assert feature.location.strand == "-"
    assert feature.location.frame == 2

    # DNA
    assert feature.dna.nt_sequence == "ATGCGATCGATCG"

    # Protein
    assert feature.protein.aa_sequence == "MRSID"

    # 3Di
    assert feature.three_di.encoding == "ABCDE"
    assert feature.three_di.method == "prostt5_aa2fold"
    assert feature.three_di.model_name == "Rostlab/ProstT5"
    assert feature.three_di.inference_device == "cuda"

    # Metadata
    assert feature.metadata.parent_id == "parent_123"
    assert feature.metadata.table_id == 4
    assert feature.metadata.has_start_codon is False
    assert feature.metadata.has_stop_codon is True
    assert feature.metadata.in_genbank is True

    # Entropy
    assert feature.entropy.dna_entropy == 2.1
    assert feature.entropy.protein_entropy == 1.8
    assert feature.entropy.three_di_entropy == 1.5


def test_unified_json_serialization():
    """Test that unified format serializes correctly to JSON."""
    orf = OrfRecord(
        parent_id="test_seq",
        orf_id="orf_1",
        start=0,
        end=30,
        strand="+",
        frame=0,
        nt_sequence="ATGGCTAGC",
        aa_sequence="MAS",
        table_id=11,
        has_start_codon=True,
        has_stop_codon=False,
        in_genbank=False,
    )

    protein = ProteinRecord(orf=orf, aa_sequence="MAS", aa_length=3)
    three_di = ThreeDiRecord(
        protein=protein,
        three_di="ABC",
        method="prostt5_aa2fold",
        model_name="test",
        inference_device="cpu",
    )

    entropy = EntropyReport(
        dna_entropy_global=1.5,
        orf_nt_entropy={"orf_1": 1.2},
        protein_aa_entropy={"orf_1": 0.8},
        three_di_entropy={"orf_1": 0.5},
        alphabet_sizes={"dna": 4, "protein": 20, "three_di": 20},
    )

    old_result = PipelineResult(
        input_id="test_seq",
        input_dna_length=100,
        orfs=[orf],
        proteins=[protein],
        three_dis=[three_di],
        entropy=entropy,
    )

    # Convert and serialize
    unified = convert_pipeline_result_to_unified(old_result)
    json_dict = to_json_dict(unified)

    # Verify JSON structure
    assert "schema_version" in json_dict
    assert json_dict["schema_version"] == "2.0.0"
    assert "input_id" in json_dict
    assert "input_dna_length" in json_dict
    assert "dna_entropy_global" in json_dict
    assert "alphabet_sizes" in json_dict
    assert "features" in json_dict

    # Verify features structure
    assert "orf_1" in json_dict["features"]
    feature_dict = json_dict["features"]["orf_1"]

    # Verify hierarchical organization
    assert "location" in feature_dict
    assert "dna" in feature_dict
    assert "protein" in feature_dict
    assert "three_di" in feature_dict
    assert "metadata" in feature_dict
    assert "entropy" in feature_dict

    # Verify no redundancy (nt_sequence appears only once)
    import json as json_module

    json_str = json_module.dumps(json_dict)
    # Count occurrences of the DNA sequence
    sequence_count = json_str.count("ATGGCTAGC")
    assert (
        sequence_count == 1
    ), f"DNA sequence appears {sequence_count} times, expected 1"

    # Count occurrences of the AA sequence
    aa_count = json_str.count("MAS")
    assert aa_count == 1, f"Protein sequence appears {aa_count} times, expected 1"


def test_conversion_handles_list_of_results():
    """Test that conversion handles a list of PipelineResult objects."""
    results = []
    for i in range(2):
        orf = OrfRecord(
            parent_id=f"seq_{i}",
            orf_id=f"orf_{i}",
            start=0,
            end=30,
            strand="+",
            frame=0,
            nt_sequence="ATG" * 10,
            aa_sequence="M" * 10,
            table_id=11,
            has_start_codon=True,
            has_stop_codon=True,
            in_genbank=False,
        )
        protein = ProteinRecord(orf=orf, aa_sequence="M" * 10, aa_length=10)
        three_di = ThreeDiRecord(
            protein=protein,
            three_di="A" * 10,
            method="prostt5_aa2fold",
            model_name="test",
            inference_device="cpu",
        )
        entropy = EntropyReport(
            dna_entropy_global=1.0,
            orf_nt_entropy={f"orf_{i}": 0.5},
            protein_aa_entropy={f"orf_{i}": 0.3},
            three_di_entropy={f"orf_{i}": 0.0},
            alphabet_sizes={"dna": 4, "protein": 20, "three_di": 20},
        )
        results.append(
            PipelineResult(
                input_id=f"seq_{i}",
                input_dna_length=100,
                orfs=[orf],
                proteins=[protein],
                three_dis=[three_di],
                entropy=entropy,
            )
        )

    # Convert list
    unified_list = convert_pipeline_result_to_unified(results)

    # Verify
    assert isinstance(unified_list, list)
    assert len(unified_list) == 2
    for i, unified in enumerate(unified_list):
        assert unified.input_id == f"seq_{i}"
        assert f"orf_{i}" in unified.features
