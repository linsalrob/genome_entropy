"""Tests for the ML classifier module."""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from genome_entropy.ml.classifier import (
    load_json_data,
    extract_features,
    GenbankClassifier,
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
                "location": {
                    "start": 0,
                    "end": 300,
                    "strand": "+",
                    "frame": 0
                },
                "dna": {
                    "nt_sequence": "ATG" * 100,
                    "length": 300
                },
                "protein": {
                    "aa_sequence": "M" * 100,
                    "length": 100
                },
                "three_di": {
                    "encoding": "A" * 100,
                    "length": 100,
                    "method": "prostt5_aa2fold",
                    "model_name": "test_model",
                    "inference_device": "cpu"
                },
                "metadata": {
                    "parent_id": "test_seq",
                    "table_id": 11,
                    "has_start_codon": True,
                    "has_stop_codon": True,
                    "in_genbank": True
                },
                "entropy": {
                    "dna_entropy": 1.2,
                    "protein_entropy": 0.5,
                    "three_di_entropy": 0.3
                }
            },
            "orf_2": {
                "orf_id": "orf_2",
                "location": {
                    "start": 400,
                    "end": 700,
                    "strand": "-",
                    "frame": 1
                },
                "dna": {
                    "nt_sequence": "ATG" * 100,
                    "length": 300
                },
                "protein": {
                    "aa_sequence": "K" * 100,
                    "length": 100
                },
                "three_di": {
                    "encoding": "B" * 100,
                    "length": 100,
                    "method": "prostt5_aa2fold",
                    "model_name": "test_model",
                    "inference_device": "cpu"
                },
                "metadata": {
                    "parent_id": "test_seq",
                    "table_id": 11,
                    "has_start_codon": True,
                    "has_stop_codon": False,
                    "in_genbank": False
                },
                "entropy": {
                    "dna_entropy": 1.5,
                    "protein_entropy": 0.8,
                    "three_di_entropy": 0.6
                }
            }
        }
    }


@pytest.fixture
def temp_json_dir(sample_json_unified):
    """Create temporary directory with sample JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create multiple JSON files
        for i in range(3):
            data = sample_json_unified.copy()
            data["input_id"] = f"test_seq_{i}"
            
            json_file = tmpdir / f"test_{i}.json"
            with open(json_file, "w") as f:
                json.dump(data, f)
        
        yield tmpdir


def test_load_json_data(temp_json_dir):
    """Test loading JSON files from directory."""
    data = load_json_data(temp_json_dir)
    
    assert len(data) == 3
    # Each element is now a list containing a dict
    assert all(isinstance(d, list) for d in data)
    assert all(len(d) == 1 for d in data)
    assert all("schema_version" in d[0] for d in data)


def test_load_json_data_empty_dir():
    """Test error handling for empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No JSON files found"):
            load_json_data(Path(tmpdir))


def test_extract_features_unified_format(sample_json_unified):
    """Test feature extraction from unified format."""
    X, y, feature_names, _ = extract_features([[sample_json_unified]])
    
    # Should have 2 ORFs
    assert X.shape[0] == 2
    # Should have 12 features
    assert X.shape[1] == 12
    
    # Check feature names
    assert len(feature_names) == 12
    assert "dna_entropy" in feature_names
    assert "protein_entropy" in feature_names
    assert "three_di_entropy" in feature_names
    
    # Check labels
    assert len(y) == 2
    assert y[0] == 1  # orf_1 is in GenBank
    assert y[1] == 0  # orf_2 is not in GenBank
    
    # Check feature values for orf_1 (first row)
    assert X[0, feature_names.index("dna_entropy")] == 1.2
    assert X[0, feature_names.index("protein_entropy")] == 0.5
    assert X[0, feature_names.index("three_di_entropy")] == 0.3
    assert X[0, feature_names.index("dna_length")] == 300
    assert X[0, feature_names.index("start")] == 0
    assert X[0, feature_names.index("end")] == 300
    assert X[0, feature_names.index("strand_plus")] == 1.0
    assert X[0, feature_names.index("frame")] == 0.0
    assert X[0, feature_names.index("has_start_codon")] == 1.0
    assert X[0, feature_names.index("has_stop_codon")] == 1.0
    
    # Check feature values for orf_2 (second row)
    assert X[1, feature_names.index("strand_plus")] == 0.0  # - strand
    assert X[1, feature_names.index("has_stop_codon")] == 0.0


def test_extract_features_old_format():
    """Test feature extraction from old format for backward compatibility."""
    old_format_data = {
        "input_id": "test_seq",
        "input_dna_length": 1000,
        "orfs": [
            {
                "orf_id": "orf_1",
                "start": 0,
                "end": 300,
                "strand": "+",
                "frame": 0,
                "nt_sequence": "ATG" * 100,
                "aa_sequence": "M" * 100,
                "has_start_codon": True,
                "has_stop_codon": True,
                "in_genbank": True
            }
        ],
        "proteins": [
            {
                "orf": {
                    "orf_id": "orf_1",
                    "start": 0,
                    "end": 300,
                    "strand": "+",
                    "frame": 0,
                    "nt_sequence": "ATG" * 100,
                    "aa_sequence": "M" * 100,
                    "has_start_codon": True,
                    "has_stop_codon": True,
                    "in_genbank": True
                },
                "aa_sequence": "M" * 100,
                "aa_length": 100
            }
        ],
        "three_dis": [
            {
                "protein": {
                    "orf": {
                        "orf_id": "orf_1",
                        "start": 0,
                        "end": 300,
                        "strand": "+",
                        "frame": 0,
                        "nt_sequence": "ATG" * 100,
                        "aa_sequence": "M" * 100,
                        "has_start_codon": True,
                        "has_stop_codon": True,
                        "in_genbank": True
                    },
                    "aa_sequence": "M" * 100,
                    "aa_length": 100
                },
                "three_di": "A" * 100,
                "method": "prostt5_aa2fold",
                "model_name": "test_model",
                "inference_device": "cpu"
            }
        ],
        "entropy": {
            "dna_entropy_global": 1.8,
            "orf_nt_entropy": {"orf_1": 1.2},
            "protein_aa_entropy": {"orf_1": 0.5},
            "three_di_entropy": {"orf_1": 0.3},
            "alphabet_sizes": {"dna": 4, "protein": 20, "three_di": 20}
        }
    }
    
    X, y, feature_names, _ = extract_features([[old_format_data]])
    
    assert X.shape[0] == 1
    assert X.shape[1] == 12
    assert y[0] == 1


def test_genbank_classifier_initialization():
    """Test GenbankClassifier initialization."""
    # XGBoost classifier
    classifier = GenbankClassifier(model_type="xgboost")
    assert classifier.model_type == "xgboost"
    assert not classifier.is_trained
    
    # Neural network classifier
    classifier_nn = GenbankClassifier(model_type="neural_net", device="cpu")
    assert classifier_nn.model_type == "neural_net"
    assert classifier_nn.device == "cpu"
    
    # Invalid model type - error occurs during fit, not initialization
    classifier_invalid = GenbankClassifier(model_type="invalid")
    assert classifier_invalid.model_type == "invalid"  # Accepted during init
    
    # But should fail during fit
    X = np.random.randn(10, 12).astype(np.float32)
    y = np.random.randint(0, 2, 10).astype(np.int32)
    with pytest.raises(ValueError, match="Unknown model_type"):
        classifier_invalid.fit(X, y)


def test_genbank_classifier_synthetic_data():
    """Test training and prediction with synthetic data."""
    # Skip if xgboost not available
    pytest.importorskip("xgboost")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 12
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Create labels with some correlation to features
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Initialize and train
    classifier = GenbankClassifier(model_type="xgboost", device="cpu")
    metrics = classifier.fit(X, y, feature_names=feature_names, validation_split=0.2)
    
    assert classifier.is_trained
    assert "val_accuracy" in metrics
    assert 0 <= metrics["val_accuracy"] <= 1
    
    # Test prediction
    predictions = classifier.predict(X[:10])
    assert predictions.shape == (10,)
    assert all(p in [0, 1] for p in predictions)
    
    # Test probability prediction
    probas = classifier.predict_proba(X[:10])
    assert probas.shape == (10, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)
    
    # Test evaluation
    eval_metrics = classifier.evaluate(X, y)
    assert "accuracy" in eval_metrics
    assert "precision" in eval_metrics
    assert "recall" in eval_metrics
    assert "f1" in eval_metrics
    
    # Test feature importance
    importance = classifier.get_feature_importance()
    assert importance is not None
    assert len(importance) == n_features


def test_classifier_not_trained_errors():
    """Test that operations on untrained model raise appropriate errors."""
    classifier = GenbankClassifier(model_type="xgboost")
    
    X = np.random.randn(10, 12).astype(np.float32)
    y = np.random.randint(0, 2, 10).astype(np.int32)
    
    with pytest.raises(RuntimeError, match="Model not trained"):
        classifier.predict(X)
    
    with pytest.raises(RuntimeError, match="Model not trained"):
        classifier.predict_proba(X)
    
    with pytest.raises(RuntimeError, match="Model not trained"):
        classifier.evaluate(X, y)
    
    with pytest.raises(RuntimeError, match="Model not trained"):
        classifier.save(Path("/tmp/model.xgb"))


def test_classifier_save_load():
    """Test model saving and loading."""
    pytest.importorskip("xgboost")
    
    # Create and train a simple model
    np.random.seed(42)
    X = np.random.randn(50, 12).astype(np.float32)
    y = np.random.randint(0, 2, 50).astype(np.int32)
    
    classifier = GenbankClassifier(model_type="xgboost", device="cpu")
    classifier.fit(X, y, validation_split=0.2)
    
    # Get predictions before saving
    pred_before = classifier.predict(X[:10])
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.xgb"
        classifier.save(model_path)
        
        assert model_path.exists()
        
        # Load model
        classifier_loaded = GenbankClassifier(model_type="xgboost", device="cpu")
        classifier_loaded.load(model_path)
        
        # Get predictions after loading
        pred_after = classifier_loaded.predict(X[:10])
        
        # Predictions should be identical
        np.testing.assert_array_equal(pred_before, pred_after)


def test_extract_features_multiple_files(temp_json_dir):
    """Test feature extraction from multiple JSON files."""
    data = load_json_data(temp_json_dir)
    X, y, feature_names, _ = extract_features(data)
    
    # Should have 3 files * 2 ORFs each = 6 samples
    assert X.shape[0] == 6
    assert len(y) == 6
    
    # Check label distribution (each file has 1 true, 1 false)
    assert (y == 1).sum() == 3
    assert (y == 0).sum() == 3


def test_feature_extraction_robustness():
    """Test feature extraction handles missing/invalid data gracefully."""
    # Data with missing entropy values
    incomplete_data = {
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
                    "inference_device": "cpu"
                },
                "metadata": {
                    "parent_id": "test_seq",
                    "table_id": 11,
                    "has_start_codon": True,
                    "has_stop_codon": True,
                    "in_genbank": True
                },
                "entropy": {
                    "dna_entropy": 1.2,
                    "protein_entropy": 0.5,
                    "three_di_entropy": 0.3
                }
            },
            "orf_bad": {
                # Missing required fields - should be skipped with warning
                "orf_id": "orf_bad",
                "location": {"start": 400, "end": 700},  # Missing strand, frame
            }
        }
    }
    
    # Should extract only the valid ORF
    X, y, _, _ = extract_features([[incomplete_data]])
    assert X.shape[0] == 1  # Only orf_1 extracted
    assert y[0] == 1
