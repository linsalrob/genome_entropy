"""Tests for file-based train/test splitting."""

import json
import tempfile
from pathlib import Path

import pytest

from genome_entropy.ml.file_split import (
    split_json_files,
    load_json_files,
    train_with_file_split,
)


@pytest.fixture
def sample_json_unified():
    """Create sample JSON data in unified format."""
    return {
        "schema_version": "2.0.0",
        "input_id": "test_seq",
        "input_dna_length": 1000,
        "dna_entropy_global": 1.8,
        "alphabet_sizes": {"dna": 4, "protein": 20, "three_di": 20},
        "features": {
            "orf_1": {
                "orf_id": "orf_1",
                "location": {"start": 0, "end": 300, "strand": "+", "frame": 0},
                "dna": {"nt_sequence": "ATG" * 100, "length": 300},
                "protein": {"aa_sequence": "M" * 100, "length": 100},
                "three_di": {
                    "encoding": "A" * 100,
                    "length": 100,
                    "method": "prostt5_aa2fold",
                    "model_name": "test_model",
                    "inference_device": "cpu",
                },
                "metadata": {
                    "parent_id": "test_seq",
                    "table_id": 11,
                    "has_start_codon": True,
                    "has_stop_codon": True,
                    "in_genbank": True,
                },
                "entropy": {
                    "dna_entropy": 1.2,
                    "protein_entropy": 0.5,
                    "three_di_entropy": 0.3,
                },
            },
            "orf_2": {
                "orf_id": "orf_2",
                "location": {"start": 400, "end": 700, "strand": "-", "frame": 1},
                "dna": {"nt_sequence": "ATG" * 100, "length": 300},
                "protein": {"aa_sequence": "K" * 100, "length": 100},
                "three_di": {
                    "encoding": "B" * 100,
                    "length": 100,
                    "method": "prostt5_aa2fold",
                    "model_name": "test_model",
                    "inference_device": "cpu",
                },
                "metadata": {
                    "parent_id": "test_seq",
                    "table_id": 11,
                    "has_start_codon": True,
                    "has_stop_codon": False,
                    "in_genbank": False,
                },
                "entropy": {
                    "dna_entropy": 1.5,
                    "protein_entropy": 0.8,
                    "three_di_entropy": 0.6,
                },
            },
        },
    }


@pytest.fixture
def temp_json_dir_for_split(sample_json_unified):
    """Create temporary directory with multiple JSON files for splitting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create 10 JSON files
        for i in range(10):
            data = sample_json_unified.copy()
            data["input_id"] = f"test_seq_{i}"

            # Wrap in list as expected by the loader
            json_file = tmpdir / f"test_{i:02d}.json"
            with open(json_file, "w") as f:
                json.dump([data], f)

        yield tmpdir


def test_split_json_files(temp_json_dir_for_split):
    """Test splitting JSON files into train and test sets."""
    train_files, test_files = split_json_files(
        temp_json_dir_for_split, train_ratio=0.8, random_seed=42
    )

    # Check correct split
    assert len(train_files) == 8  # 80% of 10
    assert len(test_files) == 2  # 20% of 10

    # Check all files are unique
    all_files = set(train_files + test_files)
    assert len(all_files) == 10

    # Check reproducibility
    train_files2, test_files2 = split_json_files(
        temp_json_dir_for_split, train_ratio=0.8, random_seed=42
    )

    assert train_files == train_files2
    assert test_files == test_files2


def test_split_json_files_different_seed(temp_json_dir_for_split):
    """Test that different seeds produce different splits."""
    train_files1, test_files1 = split_json_files(
        temp_json_dir_for_split, train_ratio=0.8, random_seed=42
    )

    train_files2, test_files2 = split_json_files(
        temp_json_dir_for_split, train_ratio=0.8, random_seed=123
    )

    # Splits should be different (with very high probability)
    assert train_files1 != train_files2 or test_files1 != test_files2


def test_split_json_files_invalid_ratio(temp_json_dir_for_split):
    """Test error handling for invalid train ratio."""
    with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
        split_json_files(temp_json_dir_for_split, train_ratio=1.5)

    with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
        split_json_files(temp_json_dir_for_split, train_ratio=0)


def test_split_json_files_empty_dir():
    """Test error handling for directory with no JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No JSON files found"):
            split_json_files(Path(tmpdir))


def test_load_json_files(temp_json_dir_for_split):
    """Test loading JSON files from a file list."""
    json_files = list(temp_json_dir_for_split.glob("*.json"))[:3]

    data = load_json_files(json_files)

    # Should have loaded 3 files
    assert len(data) == 3

    # Each element should be a list (as loaded from JSON)
    assert all(isinstance(d, list) for d in data)


def test_train_with_file_split(temp_json_dir_for_split):
    """Test complete file-based training workflow."""
    pytest.importorskip("xgboost")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_model = Path(tmpdir) / "model.ubj"
        json_output = Path(tmpdir) / "results.json"

        # Run file-based training
        result = train_with_file_split(
            split_dir=temp_json_dir_for_split,
            output=output_model,
            model_type="xgboost",
            device="cpu",
            validation_split=0.2,
            random_seed=42,
            json_output=json_output,
        )

        # Check result structure
        assert "training_files" in result
        assert "test_files" in result
        assert "training_parameters" in result
        assert "training_metrics" in result
        assert "test_metrics" in result
        assert "test_predictions" in result

        # Check file counts
        assert len(result["training_files"]) == 8
        assert len(result["test_files"]) == 2

        # Check training parameters
        assert result["training_parameters"]["model_type"] == "xgboost"
        assert result["training_parameters"]["random_seed"] == 42
        assert result["training_parameters"]["n_features"] == 12

        # Check metrics exist
        assert "accuracy" in result["test_metrics"]
        assert "precision" in result["test_metrics"]
        assert "recall" in result["test_metrics"]
        assert "f1" in result["test_metrics"]

        # Check model was saved
        assert output_model.exists()

        # Check JSON output was saved
        assert json_output.exists()

        # Verify JSON output is valid
        with open(json_output) as f:
            saved_result = json.load(f)

        assert saved_result["training_files"] == result["training_files"]
        assert saved_result["test_files"] == result["test_files"]

        # Check test predictions structure
        assert len(result["test_predictions"]) > 0
        for pred in result["test_predictions"]:
            assert "orf_id" in pred
            assert "predicted_label" in pred
            assert "probability_in_genbank" in pred
            assert "probability_not_in_genbank" in pred
            assert "actual_label" in pred
            assert "correct" in pred
            assert pred["predicted_label"] in [0, 1]
            assert pred["actual_label"] in [0, 1]
            assert isinstance(pred["correct"], bool)


def test_train_with_file_split_without_json_output(temp_json_dir_for_split):
    """Test file-based training without JSON output."""
    pytest.importorskip("xgboost")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_model = Path(tmpdir) / "model.ubj"

        # Run without JSON output
        result = train_with_file_split(
            split_dir=temp_json_dir_for_split,
            output=output_model,
            model_type="xgboost",
            device="cpu",
            validation_split=0.2,
            random_seed=42,
            json_output=None,
        )

        # Should still return complete result
        assert "training_files" in result
        assert "test_files" in result
        assert "test_metrics" in result

        # Model should be saved
        assert output_model.exists()


def test_train_with_file_split_feature_importance(temp_json_dir_for_split):
    """Test that feature importance is included in results."""
    pytest.importorskip("xgboost")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_model = Path(tmpdir) / "model.ubj"

        result = train_with_file_split(
            split_dir=temp_json_dir_for_split,
            output=output_model,
            model_type="xgboost",
            device="cpu",
            validation_split=0.2,
            random_seed=42,
        )

        # XGBoost should provide feature importance
        assert "feature_importance" in result
        assert len(result["feature_importance"]) == 12

        # Check that all expected features are present
        expected_features = [
            "dna_entropy",
            "protein_entropy",
            "three_di_entropy",
            "dna_length",
            "protein_length",
            "three_di_length",
            "start",
            "end",
            "strand_plus",
            "frame",
            "has_start_codon",
            "has_stop_codon",
        ]
        for feat in expected_features:
            assert feat in result["feature_importance"]
