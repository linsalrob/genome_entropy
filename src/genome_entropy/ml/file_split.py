"""File-based train/test splitting for ML classifier.

This module provides functionality to randomly split JSON files into
training and test sets, train a classifier on the training set, and
evaluate on the test set.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from ..logging_config import get_logger
from .classifier import GenbankClassifier, extract_features

logger = get_logger(__name__)


def split_json_files(
    directory: Path, train_ratio: float = 0.8, random_seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """Split JSON files in directory into train and test sets.

    Args:
        directory: Path to directory containing JSON files
        train_ratio: Fraction of files to use for training (default: 0.8)
        random_seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_files, test_files) as lists of Path objects

    Raises:
        ValueError: If no JSON files found or invalid train_ratio
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    # Find all JSON files
    json_files = sorted(directory.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {directory}")

    logger.info(f"Found {len(json_files)} JSON files in {directory}")

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle files
    shuffled_files = json_files.copy()
    random.shuffle(shuffled_files)

    # Split into train and test
    n_train = int(len(shuffled_files) * train_ratio)
    train_files = shuffled_files[:n_train]
    test_files = shuffled_files[n_train:]

    logger.info(
        f"Split: {len(train_files)} training files, {len(test_files)} test files"
    )

    return train_files, test_files


def load_json_files(file_list: List[Path]) -> List[List[Dict[str, Any]]]:
    """Load JSON data from a list of files.

    Args:
        file_list: List of paths to JSON files

    Returns:
        List of lists of parsed JSON data (same format as load_json_data)

    Raises:
        ValueError: If no valid JSON files could be loaded
    """
    data = []

    for json_file in file_list:
        try:
            with open(json_file, "r") as f:
                content = json.load(f)
                data.append(content)
                logger.debug(f"Loaded {json_file.name}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {json_file.name}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error loading {json_file.name}: {e}")
            continue

    if not data:
        raise ValueError("No valid JSON files could be loaded")

    logger.info(f"Successfully loaded {len(data)} JSON files")
    return data


def train_with_file_split(
    split_dir: Path,
    output: Path,
    model_type: str = "xgboost",
    device: Optional[str] = None,
    validation_split: float = 0.2,
    random_seed: int = 42,
    json_output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train classifier with file-based train/test split.

    This function:
    1. Randomly splits JSON files in directory 80/20
    2. Trains classifier on training files
    3. Evaluates on test files
    4. Returns/saves detailed results

    Args:
        split_dir: Directory containing JSON files to split
        output: Path to save trained model
        model_type: "xgboost" or "neural_net"
        device: Device for training (None for auto-detect)
        validation_split: Fraction of training data for validation
        random_seed: Random seed for reproducible splits
        json_output: Optional path to save detailed JSON report

    Returns:
        Dictionary with training results, test results, and file lists
    """
    logger.info("Starting file-based train/test split...")

    # Split files 80/20
    train_files, test_files = split_json_files(
        split_dir, train_ratio=0.8, random_seed=random_seed
    )

    # Load training data
    logger.info("\nLoading training files...")
    train_data = load_json_files(train_files)

    # Extract training features
    logger.info("Extracting features from training data...")
    X_train, y_train, feature_names, _ = extract_features(train_data)

    logger.info(
        f"Training set: {len(X_train)} ORF samples with {len(feature_names)} features"
    )
    logger.info(f"Features: {', '.join(feature_names)}")
    logger.info(
        f"Label distribution: {(y_train == 1).sum()} in GenBank, {(y_train == 0).sum()} not in GenBank"
    )

    # Check for class imbalance
    pos_ratio = (y_train == 1).sum() / len(y_train)
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        logger.warning(f"Class imbalance detected: {pos_ratio:.1%} positive samples")
        logger.warning("Model may have difficulty with minority class")

    # Initialize and train classifier
    logger.info(f"\nInitializing {model_type} classifier...")
    classifier = GenbankClassifier(model_type=model_type, device=device)

    logger.info("Training classifier on training files...")
    train_metrics = classifier.fit(
        X_train, y_train, feature_names=feature_names, validation_split=validation_split
    )

    # Log training results
    logger.info("\n" + "=" * 60)
    logger.info("Training Results")
    logger.info("=" * 60)
    for key, value in train_metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    # Load test data
    logger.info("\n" + "=" * 60)
    logger.info("Test Set Evaluation")
    logger.info("=" * 60)
    logger.info("Loading test files...")
    test_data = load_json_files(test_files)

    # Extract test features
    logger.info("Extracting features from test data...")
    X_test, y_test, _, test_metadata = extract_features(test_data, return_metadata=True)

    logger.info(f"Test set: {len(X_test)} ORF samples")
    logger.info(
        f"Label distribution: {(y_test == 1).sum()} in GenBank, {(y_test == 0).sum()} not in GenBank"
    )

    # Evaluate on test set
    logger.info("Evaluating on test files...")
    test_metrics = classifier.evaluate(X_test, y_test)

    for key, value in test_metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    # Get test predictions
    test_predictions = classifier.predict(X_test)
    test_probabilities = classifier.predict_proba(X_test)

    # Show feature importance if available
    feature_importance = classifier.get_feature_importance()
    if feature_importance:
        logger.info("\n" + "=" * 60)
        logger.info("Feature Importance (Top 10)")
        logger.info("=" * 60)

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        for feat_name, importance in sorted_features[:10]:
            logger.info(f"{feat_name:20s}: {importance:.4f}")

    # Save model
    logger.info(f"\nSaving model to: {output}")
    classifier.save(output)

    # Build result dictionary
    result = {
        "training_files": [str(f.name) for f in train_files],
        "test_files": [str(f.name) for f in test_files],
        "training_parameters": {
            "model_type": model_type,
            "device": str(device) if device else "auto",
            "validation_split": validation_split,
            "random_seed": random_seed,
            "n_features": len(feature_names),
            "feature_names": feature_names,
        },
        "training_samples": {
            "n_samples": int(len(X_train)),
            "n_in_genbank": int((y_train == 1).sum()),
            "n_not_in_genbank": int((y_train == 0).sum()),
        },
        "training_metrics": {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in train_metrics.items()
        },
        "test_samples": {
            "n_samples": int(len(X_test)),
            "n_in_genbank": int((y_test == 1).sum()),
            "n_not_in_genbank": int((y_test == 0).sum()),
        },
        "test_metrics": {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in test_metrics.items()
        },
        "test_predictions": [
            {
                "orf_id": test_metadata[i]["orf_id"] if test_metadata else f"orf_{i}",
                "predicted_label": int(test_predictions[i]),
                "probability_in_genbank": float(test_probabilities[i, 1]),
                "probability_not_in_genbank": float(test_probabilities[i, 0]),
                "actual_label": int(y_test[i]),
                "correct": bool(test_predictions[i] == y_test[i]),
            }
            for i in range(len(test_predictions))
        ],
    }

    # Add feature importance if available
    if feature_importance:
        result["feature_importance"] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in feature_importance.items()
        }

    # Save JSON output if requested
    if json_output:
        logger.info(f"Saving detailed report to: {json_output}")
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)

    return result
