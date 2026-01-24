"""Machine learning model implementations.

This module provides wrapper classes for different ML model types
that can be used for predicting GenBank annotations.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod

import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the model."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the model."""
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance. Returns None if not supported."""
        return None


class XGBoostModel(BaseModel):
    """XGBoost gradient boosted tree classifier.
    
    Recommended model for this task because:
    1. Excellent performance on tabular data
    2. Handles mixed feature types well
    3. Built-in GPU support for acceleration
    4. Provides feature importance for interpretability
    5. Robust to overfitting with proper parameters
    6. Fast training and inference
    
    This model automatically uses GPU if available, falling back to CPU.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs: Any
    ):
        """Initialize XGBoost model.
        
        Args:
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            **kwargs: Additional XGBoost parameters
        """
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )
        
        # Auto-detect device if not specified
        if device is None:
            device = self._auto_detect_device()
        
        self.device = device
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        self.model = None
        
        logger.info(f"XGBoost model initialized with device={device}")
    
    def _auto_detect_device(self) -> str:
        """Auto-detect available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train XGBoost model.
        
        Args:
            X: Feature matrix
            y: Labels
            validation_split: Fraction for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        X_train = X[indices[n_val:]]
        y_train = y[indices[n_val:]]
        X_val = X[indices[:n_val]]
        y_val = y[indices[:n_val]]
        
        # Set up parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error", "auc"],
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "tree_method": "hist",  # Fast histogram method
            **self.kwargs
        }
        
        # Add GPU support if available
        if self.device == "cuda":
            params["device"] = "cuda"
            logger.info("Using GPU acceleration for XGBoost")
        else:
            params["device"] = "cpu"
        
        # Create DMatrix objects
        dtrain = self.xgb.DMatrix(X_train, label=y_train)
        dval = self.xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        evals = [(dtrain, "train"), (dval, "val")]
        evals_result = {}
        
        logger.info(f"Training XGBoost with {len(X_train)} train, {len(X_val)} val samples")
        
        self.model = self.xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            evals_result=evals_result,
            verbose_eval=False
        )
        
        # Get final metrics
        val_preds = self.model.predict(dval)
        val_preds_binary = (val_preds > 0.5).astype(int)
        val_accuracy = np.mean(val_preds_binary == y_val)
        
        metrics = {
            "val_accuracy": float(val_accuracy),
            "val_logloss": float(evals_result["val"]["logloss"][-1]),
            "val_auc": float(evals_result["val"]["auc"][-1]),
            "train_samples": len(X_train),
            "val_samples": len(X_val)
        }
        
        logger.info(f"Training complete. Val accuracy: {val_accuracy:.4f}, Val AUC: {metrics['val_auc']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels (0 or 1)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        dmatrix = self.xgb.DMatrix(X)
        probs = self.model.predict(dmatrix)
        return (probs > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probabilities of shape (n_samples, 2)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        dmatrix = self.xgb.DMatrix(X)
        probs_pos = self.model.predict(dmatrix)
        probs_neg = 1 - probs_pos
        return np.column_stack([probs_neg, probs_pos])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y)
        
        # Confusion matrix components
        tp = np.sum((y_pred == 1) & (y == 1))
        tn = np.sum((y_pred == 0) & (y == 0))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC (if sklearn available)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, y_proba)
        except ImportError:
            auc = 0.0
            logger.warning("sklearn not available, AUC not calculated")
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores.
        
        Returns:
            Array of importance scores
        """
        if self.model is None:
            return None
        
        # Get importance scores
        importance_dict = self.model.get_score(importance_type="gain")
        
        # Convert to array (ordered by feature index)
        n_features = self.model.num_features()
        importance = np.zeros(n_features)
        
        for feat_name, score in importance_dict.items():
            # Feature names are like 'f0', 'f1', etc.
            feat_idx = int(feat_name[1:])
            importance[feat_idx] = score
        
        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()
        
        return importance
    
    def save(self, path: Path) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(path))
        
        # Save metadata
        metadata = {
            "device": self.device,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "kwargs": self.kwargs
        }
        
        metadata_path = path.with_suffix(path.suffix + ".meta")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"XGBoost model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk.
        
        Args:
            path: Path to saved model
        """
        path = Path(path)
        
        # Load XGBoost model
        self.model = self.xgb.Booster()
        self.model.load_model(str(path))
        
        # Load metadata if available
        metadata_path = path.with_suffix(path.suffix + ".meta")
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            self.device = metadata.get("device", "cpu")
            self.n_estimators = metadata.get("n_estimators", 100)
            self.max_depth = metadata.get("max_depth", 6)
            self.learning_rate = metadata.get("learning_rate", 0.1)
            self.kwargs = metadata.get("kwargs", {})
        
        logger.info(f"XGBoost model loaded from {path}")


class NeuralNetModel(BaseModel):
    """Simple neural network classifier using PyTorch.
    
    Alternative to XGBoost. Uses a simple feedforward network with:
    - 2 hidden layers with ReLU activation
    - Dropout for regularization
    - Binary cross-entropy loss
    - GPU support via PyTorch
    
    Generally less suitable than XGBoost for this task because:
    - Requires more data and careful tuning
    - Less interpretable (no feature importance)
    - More prone to overfitting on small datasets
    
    However, it provides GPU acceleration and can model complex non-linear
    relationships if sufficient data is available.
    """
    
    def __init__(
        self,
        input_dim: int,
        device: Optional[str] = None,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """Initialize neural network model.
        
        Args:
            input_dim: Number of input features
            device: Device to use (None for auto-detect)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        import torch
        import torch.nn as nn
        
        self.torch = torch
        self.nn = nn
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            device = self._auto_detect_device()
        
        self.device = torch.device(device)
        self.model = None
        
        logger.info(f"Neural network initialized with device={device}")
    
    def _auto_detect_device(self) -> str:
        """Auto-detect available device."""
        if self.torch.cuda.is_available():
            return "cuda"
        elif hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _build_model(self) -> Any:
        """Build the neural network architecture."""
        class SimpleNN(self.nn.Module):
            def __init__(self_nn, input_dim, hidden_dim, dropout):
                super().__init__()
                self_nn.layers = self.nn.Sequential(
                    self.nn.Linear(input_dim, hidden_dim),
                    self.nn.ReLU(),
                    self.nn.Dropout(dropout),
                    self.nn.Linear(hidden_dim, hidden_dim // 2),
                    self.nn.ReLU(),
                    self.nn.Dropout(dropout),
                    self.nn.Linear(hidden_dim // 2, 1),
                    self.nn.Sigmoid()
                )
            
            def forward(self_nn, x):
                return self_nn.layers(x)
        
        return SimpleNN(self.input_dim, self.hidden_dim, self.dropout)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train neural network.
        
        Args:
            X: Feature matrix
            y: Labels
            validation_split: Fraction for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Build model
        self.model = self._build_model().to(self.device)
        
        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        X_train = X[indices[n_val:]]
        y_train = y[indices[n_val:]]
        X_val = X[indices[:n_val]]
        y_val = y[indices[:n_val]]
        
        # Convert to tensors
        X_train_t = self.torch.tensor(X_train, dtype=self.torch.float32).to(self.device)
        y_train_t = self.torch.tensor(y_train, dtype=self.torch.float32).unsqueeze(1).to(self.device)
        X_val_t = self.torch.tensor(X_val, dtype=self.torch.float32).to(self.device)
        y_val_t = self.torch.tensor(y_val, dtype=self.torch.float32).unsqueeze(1).to(self.device)
        
        # Setup optimizer and loss
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = self.nn.BCELoss()
        
        # Training loop
        logger.info(f"Training neural network for {self.epochs} epochs")
        
        best_val_loss = float("inf")
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # Mini-batch training
            for i in range(0, len(X_train_t), self.batch_size):
                batch_X = X_train_t[i:i+self.batch_size]
                batch_y = y_train_t[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            with self.torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                val_preds = (val_outputs > 0.5).float()
                val_accuracy = (val_preds == y_val_t).float().mean()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}: val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")
        
        metrics = {
            "val_accuracy": float(val_accuracy),
            "val_loss": float(val_loss),
            "train_samples": len(X_train),
            "val_samples": len(X_val)
        }
        
        logger.info(f"Training complete. Val accuracy: {val_accuracy:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels (0 or 1)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        with self.torch.no_grad():
            X_t = self.torch.tensor(X, dtype=self.torch.float32).to(self.device)
            outputs = self.model(X_t)
            preds = (outputs > 0.5).cpu().numpy().astype(int).flatten()
        
        return preds
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probabilities of shape (n_samples, 2)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        with self.torch.no_grad():
            X_t = self.torch.tensor(X, dtype=self.torch.float32).to(self.device)
            outputs = self.model(X_t)
            probs_pos = outputs.cpu().numpy().flatten()
        
        probs_neg = 1 - probs_pos
        return np.column_stack([probs_neg, probs_pos])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        y_pred = self.predict(X)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y)
        
        tp = np.sum((y_pred == 1) & (y == 1))
        tn = np.sum((y_pred == 0) & (y == 0))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }
    
    def save(self, path: Path) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        self.torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }, path)
        
        logger.info(f"Neural network saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk.
        
        Args:
            path: Path to saved model
        """
        path = Path(path)
        
        checkpoint = self.torch.load(path, map_location=self.device)
        
        # Update parameters
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.dropout = checkpoint["dropout"]
        self.learning_rate = checkpoint.get("learning_rate", 0.001)
        self.epochs = checkpoint.get("epochs", 100)
        self.batch_size = checkpoint.get("batch_size", 32)
        
        # Build and load model
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        logger.info(f"Neural network loaded from {path}")
