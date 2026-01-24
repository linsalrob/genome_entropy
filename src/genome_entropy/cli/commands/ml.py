"""CLI command for training and using the GenBank ML classifier."""

from pathlib import Path
from typing import Optional

import typer

from ..logging_config import get_logger
from ..ml.classifier import GenbankClassifier, load_json_data, extract_features

logger = get_logger(__name__)

app = typer.Typer(help="Train ML classifier to predict GenBank annotations")


@app.command("train")
def train_classifier(
    json_dir: Path = typer.Option(
        ...,
        "--json-dir",
        "-i",
        help="Directory containing JSON output files from genome_entropy pipeline"
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to save the trained model"
    ),
    model_type: str = typer.Option(
        "xgboost",
        "--model-type",
        "-m",
        help="Model type: 'xgboost' (recommended) or 'neural_net'"
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device for training: 'cuda', 'cpu', or None for auto-detect"
    ),
    validation_split: float = typer.Option(
        0.2,
        "--validation-split",
        "-v",
        help="Fraction of data to use for validation"
    ),
    test_split: float = typer.Option(
        0.1,
        "--test-split",
        "-t",
        help="Fraction of data to hold out for final testing"
    ),
) -> None:
    """Train a machine learning classifier to predict GenBank annotations.
    
    This command trains a model to predict whether an ORF was annotated in the
    original GenBank file (in_genbank: True/False) based on sequence features
    including entropy values, length, position, and other characteristics.
    
    MODEL JUSTIFICATION:
    --------------------
    
    The default model is XGBoost (Gradient Boosted Trees) because:
    
    1. **Excellent performance on tabular data**: XGBoost consistently achieves
       state-of-the-art results on structured/tabular datasets like ORF features.
    
    2. **Handles mixed feature types**: Our features include continuous (entropy,
       length), categorical (strand, frame), and boolean (has_start/stop_codon)
       variables. XGBoost handles these naturally without extensive preprocessing.
    
    3. **Built-in GPU support**: Matches the requirement "we are already working
       on GPUs" - XGBoost can leverage GPU acceleration for faster training.
    
    4. **Feature importance**: Provides interpretability by showing which features
       are most predictive of GenBank annotations (e.g., are entropy values or
       structural features more important?).
    
    5. **Robust and fast**: Less prone to overfitting than neural networks on
       small-to-medium datasets, trains quickly, and doesn't require extensive
       hyperparameter tuning to get good results.
    
    6. **Handles imbalanced data**: GenBank annotations are often imbalanced
       (more non-annotated ORFs than annotated ones). XGBoost handles this well.
    
    ALTERNATIVE: Neural Network
    ----------------------------
    
    The neural_net option is also available, which uses PyTorch for:
    - GPU acceleration via PyTorch's CUDA support
    - Modeling complex non-linear relationships
    - Flexibility in architecture
    
    However, it generally performs worse than XGBoost on this type of structured
    data unless you have a very large dataset (10k+ samples) and can tune it well.
    
    USAGE EXAMPLES:
    ---------------
    
    Basic usage with XGBoost (recommended):
        genome_entropy ml train --json-dir results/ --output model.xgb
    
    Use neural network with GPU:
        genome_entropy ml train --json-dir results/ --output model.pt \\
            --model-type neural_net --device cuda
    
    Custom validation split:
        genome_entropy ml train --json-dir results/ --output model.xgb \\
            --validation-split 0.3 --test-split 0.15
    """
    logger.info("="*60)
    logger.info("GenBank ORF Classification - Model Training")
    logger.info("="*60)
    
    # Validate inputs
    if not json_dir.is_dir():
        logger.error(f"JSON directory not found: {json_dir}")
        raise typer.Exit(1)
    
    if model_type not in ["xgboost", "neural_net"]:
        logger.error(f"Invalid model type: {model_type}")
        logger.error("Choose 'xgboost' or 'neural_net'")
        raise typer.Exit(1)
    
    # Log model choice justification
    logger.info(f"Model type: {model_type}")
    if model_type == "xgboost":
        logger.info("Using XGBoost (Gradient Boosted Trees) - recommended for:")
        logger.info("  ✓ Excellent performance on structured/tabular data")
        logger.info("  ✓ Handles mixed feature types naturally")
        logger.info("  ✓ Built-in GPU support for acceleration")
        logger.info("  ✓ Provides feature importance for interpretability")
        logger.info("  ✓ Robust to overfitting, fast training")
    else:
        logger.info("Using Neural Network (PyTorch) - suitable for:")
        logger.info("  ✓ GPU acceleration via PyTorch CUDA")
        logger.info("  ✓ Complex non-linear relationships (if large dataset)")
        logger.info("  ! May require more data and tuning than XGBoost")
    
    # Load data
    logger.info(f"\nLoading JSON files from: {json_dir}")
    try:
        json_data = load_json_data(json_dir)
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        raise typer.Exit(1)
    
    # Extract features
    logger.info("\nExtracting features from JSON data...")
    try:
        X, y, feature_names = extract_features(json_data)
    except Exception as e:
        logger.error(f"Failed to extract features: {e}")
        raise typer.Exit(1)
    
    logger.info(f"Extracted {len(X)} ORF samples with {len(feature_names)} features")
    logger.info(f"Features: {', '.join(feature_names)}")
    logger.info(f"Label distribution: {(y == 1).sum()} in GenBank, {(y == 0).sum()} not in GenBank")
    
    # Check for class imbalance
    pos_ratio = (y == 1).sum() / len(y)
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        logger.warning(f"Class imbalance detected: {pos_ratio:.1%} positive samples")
        logger.warning("Model may have difficulty with minority class")
    
    # Split data into train+val and test sets
    import numpy as np
    n_samples = len(X)
    n_test = int(n_samples * test_split)
    indices = np.random.permutation(n_samples)
    
    X_trainval = X[indices[n_test:]]
    y_trainval = y[indices[n_test:]]
    X_test = X[indices[:n_test]]
    y_test = y[indices[:n_test]]
    
    logger.info(f"\nData split: {len(X_trainval)} train+val, {len(X_test)} test")
    
    # Initialize and train classifier
    logger.info(f"\nInitializing {model_type} classifier...")
    classifier = GenbankClassifier(model_type=model_type, device=device)
    
    logger.info("Training classifier...")
    try:
        train_metrics = classifier.fit(
            X_trainval,
            y_trainval,
            feature_names=feature_names,
            validation_split=validation_split
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise typer.Exit(1)
    
    # Log training results
    logger.info("\n" + "="*60)
    logger.info("Training Results")
    logger.info("="*60)
    for key, value in train_metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    
    # Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("Test Set Evaluation")
    logger.info("="*60)
    try:
        test_metrics = classifier.evaluate(X_test, y_test)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise typer.Exit(1)
    
    for key, value in test_metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    
    # Show feature importance if available
    feature_importance = classifier.get_feature_importance()
    if feature_importance:
        logger.info("\n" + "="*60)
        logger.info("Feature Importance (Top 10)")
        logger.info("="*60)
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for feat_name, importance in sorted_features[:10]:
            logger.info(f"{feat_name:20s}: {importance:.4f}")
    
    # Save model
    logger.info(f"\nSaving model to: {output}")
    try:
        classifier.save(output)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise typer.Exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Model saved to: {output}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 score: {test_metrics['f1']:.4f}")


@app.command("predict")
def predict_with_classifier(
    json_dir: Path = typer.Option(
        ...,
        "--json-dir",
        "-i",
        help="Directory containing JSON files to predict on"
    ),
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model file"
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to save predictions (CSV format)"
    ),
    model_type: str = typer.Option(
        "xgboost",
        "--model-type",
        "-t",
        help="Model type: 'xgboost' or 'neural_net'"
    ),
) -> None:
    """Make predictions using a trained classifier.
    
    Loads a previously trained model and makes predictions on new JSON files.
    Outputs predictions in CSV format with ORF IDs and predicted probabilities.
    """
    logger.info("Loading JSON files...")
    try:
        json_data = load_json_data(json_dir)
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        raise typer.Exit(1)
    
    logger.info("Extracting features...")
    try:
        X, _, feature_names = extract_features(json_data)
    except Exception as e:
        logger.error(f"Failed to extract features: {e}")
        raise typer.Exit(1)
    
    logger.info(f"Loading model from: {model}")
    classifier = GenbankClassifier(model_type=model_type)
    try:
        classifier.load(model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise typer.Exit(1)
    
    logger.info("Making predictions...")
    try:
        predictions = classifier.predict(X)
        probabilities = classifier.predict_proba(X)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise typer.Exit(1)
    
    # TODO: Extract ORF IDs to match with predictions
    # For now, just save predictions and probabilities
    
    logger.info(f"Saving predictions to: {output}")
    import csv
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "predicted_label", "prob_not_in_genbank", "prob_in_genbank"])
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            writer.writerow([i, pred, prob[0], prob[1]])
    
    logger.info(f"Predictions saved. {len(predictions)} samples processed.")
    logger.info(f"Predicted in GenBank: {(predictions == 1).sum()}")
    logger.info(f"Predicted not in GenBank: {(predictions == 0).sum()}")
