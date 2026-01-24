"""Main classifier for predicting GenBank ORF annotations.

This module provides functionality to train machine learning models that predict
whether an ORF was annotated in the original GenBank file (in_genbank: True/False)
based on various sequence features including entropy values, length, position, etc.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)


def load_json_data(json_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON files from a directory.
    
    Handles both old PipelineResult format and new unified format.
    
    Args:
        json_dir: Directory containing JSON output files
        
    Returns:
        List of parsed JSON data (each is a dictionary)
        
    Raises:
        ValueError: If no JSON files found or if files are invalid
    """
    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise ValueError(f"Not a directory: {json_dir}")
    
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    logger.info(f"Found {len(json_files)} JSON files in {json_dir}")
    
    data = []
    for json_file in json_files:
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


def extract_features(
    json_data: List[Dict[str, Any]],
    include_sequences: bool = False,
    return_metadata: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[List[Dict[str, Any]]]]:
    """Extract features and labels from JSON data.
    
    Extracts numerical and categorical features from the unified JSON format
    to predict the in_genbank boolean target.
    
    Features extracted:
    - Numerical: dna_entropy, protein_entropy, three_di_entropy
    - Numerical: dna_length, protein_length, three_di_length
    - Numerical: start, end (genomic position)
    - Categorical (encoded): strand (+/-), frame (0-3)
    - Boolean (encoded): has_start_codon, has_stop_codon
    
    Args:
        json_data: List of parsed JSON dictionaries from load_json_data()
        include_sequences: If True, include sequence-based features (default: False)
                          This can make feature vectors very large
        return_metadata: If True, return metadata for each ORF including orf_id and 
                        actual in_genbank value (default: False)
        
    Returns:
        Tuple of (features, labels, feature_names, metadata):
        - features: numpy array of shape (n_samples, n_features)
        - labels: numpy array of shape (n_samples,) with 0/1 labels
        - feature_names: list of feature names in order
        - metadata: list of dicts with orf_id and in_genbank (if return_metadata=True), else None
        
    Raises:
        ValueError: If data format is invalid or no features found
    """
    features_list = []
    labels_list = []
    metadata_list = [] if return_metadata else None
    feature_names = [
        "dna_entropy",
        "protein_entropy",
        "three_di_entropy",
        "dna_length",
        "protein_length",
        "three_di_length",
        "start",
        "end",
        "strand_plus",  # 1 if +, 0 if -
        "frame",
        "has_start_codon",
        "has_stop_codon",
    ]
    
    orf_count = 0
    for data in json_data:
            # Handle unified format (schema_version 2.0.0)
            if "schema_version" in data and "features" in data:
                # New unified format
                features_dict = data["features"]
                
                for orf_id, feature in features_dict.items():
                    try:
                        # Extract feature values
                        feature_vector = [
                            feature["entropy"]["dna_entropy"],
                            feature["entropy"]["protein_entropy"],
                            feature["entropy"]["three_di_entropy"],
                            feature["dna"]["length"],
                            feature["protein"]["length"],
                            feature["three_di"]["length"],
                            feature["location"]["start"],
                            feature["location"]["end"],
                            1.0 if feature["location"]["strand"] == "+" else 0.0,
                            float(feature["location"]["frame"]),
                            1.0 if feature["metadata"]["has_start_codon"] else 0.0,
                            1.0 if feature["metadata"]["has_stop_codon"] else 0.0,
                        ]
                        
                        # Extract label
                        label = 1 if feature["metadata"]["in_genbank"] else 0
                        
                        features_list.append(feature_vector)
                        labels_list.append(label)
                        
                        # Store metadata if requested
                        if return_metadata:
                            metadata_list.append({
                                "orf_id": orf_id,
                                "in_genbank": feature["metadata"]["in_genbank"]
                            })
                        
                        orf_count += 1
                        
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Failed to extract features for {orf_id}: {e}")
                        continue
            
            # Handle old format (for backward compatibility)
            elif "orfs" in data and "entropy" in data:
                logger.warning("Old format detected - consider regenerating with new pipeline")
                # Old format: extract from orfs, proteins, three_dis lists
                orfs = data.get("orfs", [])
                entropy = data.get("entropy", {})
                
                # Build lookup dicts
                orf_nt_entropy = entropy.get("orf_nt_entropy", {})
                protein_aa_entropy = entropy.get("protein_aa_entropy", {})
                three_di_entropy = entropy.get("three_di_entropy", {})
                
                # Get proteins and three_dis by orf_id
                proteins = {p["orf"]["orf_id"]: p for p in data.get("proteins", [])}
                three_dis = {
                    td["protein"]["orf"]["orf_id"]: td 
                    for td in data.get("three_dis", [])
                }
                
                for orf in orfs:
                    try:
                        orf_id = orf["orf_id"]
                        protein = proteins.get(orf_id)
                        three_di = three_dis.get(orf_id)
                        
                        if not protein or not three_di:
                            continue
                        
                        feature_vector = [
                            orf_nt_entropy.get(orf_id, 0.0),
                            protein_aa_entropy.get(orf_id, 0.0),
                            three_di_entropy.get(orf_id, 0.0),
                            len(orf["nt_sequence"]),
                            protein["aa_length"],
                            len(three_di["three_di"]),
                            orf["start"],
                            orf["end"],
                            1.0 if orf["strand"] == "+" else 0.0,
                            float(orf["frame"]),
                            1.0 if orf["has_start_codon"] else 0.0,
                            1.0 if orf["has_stop_codon"] else 0.0,
                        ]
                        
                        label = 1 if orf.get("in_genbank", False) else 0
                        
                        features_list.append(feature_vector)
                        labels_list.append(label)
                        
                        # Store metadata if requested
                        if return_metadata:
                            metadata_list.append({
                                "orf_id": orf_id,
                                "in_genbank": orf.get("in_genbank", False)
                            })
                        
                        orf_count += 1
                        
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Failed to extract features from old format: {e}")
                        continue
            else:
                logger.warning(f"Unknown JSON format in data")
                continue
    
    if not features_list:
        raise ValueError("No features could be extracted from the JSON data")
    
    # Convert to numpy arrays
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    
    logger.info(f"Extracted {len(X)} ORF samples with {X.shape[1]} features")
    logger.info(f"Label distribution: {np.sum(y == 1)} in GenBank, {np.sum(y == 0)} not in GenBank")
    
    if return_metadata:
        return X, y, feature_names, metadata_list
    else:
        return X, y, feature_names, None


class GenbankClassifier:
    """Machine learning classifier for predicting GenBank ORF annotations.
    
    This classifier trains a model to predict whether an ORF was annotated
    in the original GenBank file based on various sequence features.
    
    Supports multiple model types:
    - "xgboost": Gradient boosted trees (default, recommended)
    - "neural_net": Simple neural network using PyTorch
    
    Example:
        >>> classifier = GenbankClassifier(model_type="xgboost")
        >>> data = load_json_data(Path("results/"))
        >>> X, y, feature_names = extract_features(data)
        >>> classifier.fit(X, y, feature_names)
        >>> metrics = classifier.evaluate(X, y)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        device: Optional[str] = None,
        **model_kwargs: Any
    ):
        """Initialize the classifier.
        
        Args:
            model_type: Type of model to use ("xgboost" or "neural_net")
            device: Device for computation (None for auto-detect, "cuda", "cpu")
            **model_kwargs: Additional arguments passed to the model
        """
        self.model_type = model_type
        self.device = device
        self.model_kwargs = model_kwargs
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        logger.info(f"Initialized GenbankClassifier with model_type={model_type}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the classifier on the provided data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            feature_names: Optional list of feature names
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training metrics
        """
        from .models import XGBoostModel, NeuralNetModel
        
        self.feature_names = feature_names
        
        # Initialize the model
        if self.model_type == "xgboost":
            self.model = XGBoostModel(device=self.device, **self.model_kwargs)
        elif self.model_type == "neural_net":
            self.model = NeuralNetModel(
                input_dim=X.shape[1],
                device=self.device,
                **self.model_kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        # Train the model
        logger.info(f"Training {self.model_type} model on {len(X)} samples")
        metrics = self.model.fit(X, y, validation_split=validation_split)
        self.is_trained = True
        
        logger.info(f"Training completed. Validation metrics: {metrics}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities of shape (n_samples, 2)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self.model.evaluate(X, y)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if model doesn't support feature importance
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        importance = self.model.get_feature_importance()
        
        if importance is not None and self.feature_names is not None:
            return dict(zip(self.feature_names, importance))
        
        return importance
    
    def save(self, path: Path) -> None:
        """Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        from .models import XGBoostModel, NeuralNetModel
        
        # Initialize the appropriate model
        if self.model_type == "xgboost":
            self.model = XGBoostModel(device=self.device, **self.model_kwargs)
        elif self.model_type == "neural_net":
            # Need to know input_dim - will be loaded from model file
            self.model = NeuralNetModel(input_dim=12, device=self.device, **self.model_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        self.model.load(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
