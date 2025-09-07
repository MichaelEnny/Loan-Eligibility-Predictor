"""
Base Model Trainer Class
Provides the foundation for all ML model implementations with comprehensive
training, validation, and monitoring capabilities.
"""

import os
import time
import json
import pickle
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import joblib

# FeatureEngineeringPipeline import handled separately to avoid circular imports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingMetrics:
    """Comprehensive training metrics tracking."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.training_time = 0.0
        self.validation_scores = {}
        self.test_scores = {}
        self.cross_val_scores = {}
        self.feature_importance = None
        self.confusion_matrix = None
        self.classification_report = ""
        self.model_size_mb = 0.0
        self.inference_time_ms = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'training_time': self.training_time,
            'validation_scores': self.validation_scores,
            'test_scores': self.test_scores,
            'cross_val_scores': self.cross_val_scores,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'classification_report': self.classification_report,
            'model_size_mb': self.model_size_mb,
            'inference_time_ms': self.inference_time_ms
        }


class BaseModelTrainer(ABC):
    """
    Abstract base class for all model trainers.
    
    Provides standardized interface for model training, validation, evaluation,
    and deployment with comprehensive monitoring and artifact management.
    """
    
    def __init__(self, 
                 model_name: str,
                 random_state: int = 42,
                 verbose: bool = True,
                 model_dir: str = "trained_models",
                 enable_monitoring: bool = True):
        """
        Initialize base model trainer.
        
        Args:
            model_name: Name of the model
            random_state: Random state for reproducibility
            verbose: Enable verbose logging
            model_dir: Directory to save trained models
            enable_monitoring: Enable training monitoring
        """
        self.model_name = model_name
        self.random_state = random_state
        self.verbose = verbose
        self.enable_monitoring = enable_monitoring
        
        # Setup directories
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.feature_pipeline = None
        self.metrics = TrainingMetrics()
        self.is_trained = False
        
        # Training metadata
        self.training_metadata = {
            'model_name': model_name,
            'version': '1.0.0',
            'created_at': None,
            'training_data_shape': None,
            'feature_names': [],
            'target_classes': [],
            'hyperparameters': {},
            'training_config': {}
        }
        
        if self.verbose:
            logger.info(f"Initialized {model_name} trainer")
    
    @abstractmethod
    def _create_model(self, **hyperparameters) -> BaseEstimator:
        """Create and return the model instance."""
        pass
    
    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Return default hyperparameters for the model."""
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """Return hyperparameter space for tuning."""
        pass
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray = None, 
                          metric_prefix: str = "") -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics[f'{metric_prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{metric_prefix}precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics[f'{metric_prefix}recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics[f'{metric_prefix}f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC if probabilities are available
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics[f'{metric_prefix}roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                else:  # Multi-class
                    metrics[f'{metric_prefix}roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics[f'{metric_prefix}roc_auc'] = 0.0
        
        return metrics
    
    def _measure_inference_time(self, X: np.ndarray, n_samples: int = 1000) -> float:
        """Measure inference time in milliseconds."""
        if not self.is_trained:
            return 0.0
        
        # Use a subset for timing
        X_sample = X[:min(n_samples, len(X))]
        
        start_time = time.time()
        _ = self.model.predict(X_sample)
        end_time = time.time()
        
        # Calculate ms per prediction
        total_time_ms = (end_time - start_time) * 1000
        return total_time_ms / len(X_sample)
    
    def _get_model_size(self) -> float:
        """Get model size in MB."""
        if not self.is_trained:
            return 0.0
        
        try:
            # Serialize model to estimate size
            import pickle
            model_bytes = pickle.dumps(self.model)
            return len(model_bytes) / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def train(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray],
              validation_split: float = 0.2,
              hyperparameters: Optional[Dict[str, Any]] = None,
              feature_pipeline: Optional[Any] = None,
              **kwargs) -> 'BaseModelTrainer':
        """
        Train the model with comprehensive validation and monitoring.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Proportion for validation split
            hyperparameters: Model hyperparameters
            feature_pipeline: Optional feature engineering pipeline
            **kwargs: Additional training arguments
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"Starting {self.model_name} training...")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.training_metadata['feature_names'] = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Store metadata
        self.training_metadata['training_data_shape'] = X.shape
        self.training_metadata['target_classes'] = list(np.unique(y))
        self.training_metadata['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Apply feature pipeline if provided
        if feature_pipeline is not None:
            self.feature_pipeline = feature_pipeline
            X_train = feature_pipeline.fit_transform(X_train, y_train)
            X_val = feature_pipeline.transform(X_val)
        
        # Get hyperparameters
        if hyperparameters is None:
            hyperparameters = self.get_default_hyperparameters()
        self.training_metadata['hyperparameters'] = hyperparameters
        
        # Create and train model
        self.model = self._create_model(**hyperparameters)
        
        if self.verbose:
            logger.info(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training time
        self.metrics.training_time = time.time() - start_time
        
        # Validation predictions
        val_pred = self.model.predict(X_val)
        val_pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            val_pred_proba = self.model.predict_proba(X_val)
            if val_pred_proba.shape[1] == 2:  # Binary classification
                val_pred_proba = val_pred_proba[:, 1]
        
        # Calculate metrics
        self.metrics.validation_scores = self._calculate_metrics(
            y_val, val_pred, val_pred_proba, "val_"
        )
        
        # Store confusion matrix and classification report
        self.metrics.confusion_matrix = confusion_matrix(y_val, val_pred)
        self.metrics.classification_report = classification_report(
            y_val, val_pred, target_names=[f'Class_{i}' for i in np.unique(y)]
        )
        
        # Feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.metrics.feature_importance = self.model.feature_importances_
        
        # Model size and inference time
        self.metrics.model_size_mb = self._get_model_size()
        self.metrics.inference_time_ms = self._measure_inference_time(X_val)
        
        self.is_trained = True
        
        if self.verbose:
            logger.info(f"Training completed in {self.metrics.training_time:.2f}s")
            logger.info(f"Validation accuracy: {self.metrics.validation_scores.get('val_accuracy', 0):.4f}")
            logger.info(f"Validation F1-score: {self.metrics.validation_scores.get('val_f1_score', 0):.4f}")
            if 'val_roc_auc' in self.metrics.validation_scores:
                logger.info(f"Validation ROC-AUC: {self.metrics.validation_scores['val_roc_auc']:.4f}")
        
        return self
    
    def evaluate(self, X_test: Union[pd.DataFrame, np.ndarray], 
                 y_test: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Test metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Convert to numpy if needed
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            X_test = self.feature_pipeline.transform(X_test)
        
        # Predictions
        test_pred = self.model.predict(X_test)
        test_pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            test_pred_proba = self.model.predict_proba(X_test)
            if test_pred_proba.shape[1] == 2:  # Binary classification
                test_pred_proba = test_pred_proba[:, 1]
        
        # Calculate metrics
        self.metrics.test_scores = self._calculate_metrics(
            y_test, test_pred, test_pred_proba, "test_"
        )
        
        if self.verbose:
            logger.info(f"Test accuracy: {self.metrics.test_scores.get('test_accuracy', 0):.4f}")
            logger.info(f"Test F1-score: {self.metrics.test_scores.get('test_f1_score', 0):.4f}")
            if 'test_roc_auc' in self.metrics.test_scores:
                logger.info(f"Test ROC-AUC: {self.metrics.test_scores['test_roc_auc']:.4f}")
        
        return self.metrics.test_scores
    
    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray],
                      cv_folds: int = 5,
                      scoring: List[str] = None) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            cv_folds: Number of CV folds
            scoring: List of scoring metrics
            
        Returns:
            Cross-validation scores
        """
        if not self.is_trained:
            # Create model for CV
            hyperparams = self.get_default_hyperparameters()
            model = self._create_model(**hyperparams)
        else:
            model = self.model
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)
        
        # Default scoring metrics
        if scoring is None:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        
        for metric in scoring:
            try:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
                cv_results[f'cv_{metric}_mean'] = scores.mean()
                cv_results[f'cv_{metric}_std'] = scores.std()
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Could not calculate {metric}: {e}")
                cv_results[f'cv_{metric}_mean'] = 0.0
                cv_results[f'cv_{metric}_std'] = 0.0
        
        self.metrics.cross_val_scores = cv_results
        
        if self.verbose:
            logger.info(f"Cross-validation results ({cv_folds} folds):")
            for metric, score in cv_results.items():
                if '_mean' in metric:
                    logger.info(f"  {metric}: {score:.4f}")
        
        return cv_results
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"{self.model_name} does not support probability predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        else:
            return None
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save trained model and metadata.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = self.model_dir / f"{self.model_name}_{timestamp}.pkl"
        else:
            filepath = Path(filepath)
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and pipeline
        model_data = {
            'model': self.model,
            'feature_pipeline': self.feature_pipeline,
            'metrics': self.metrics.to_dict(),
            'metadata': self.training_metadata,
            'model_name': self.model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata separately
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'metrics': self.metrics.to_dict(),
                'metadata': self.training_metadata
            }, f, indent=2, default=str)
        
        if self.verbose:
            logger.info(f"Model saved to {filepath}")
        
        return str(filepath)
    
    @classmethod
    def load_model(cls, filepath: str, model_name: str = None) -> 'BaseModelTrainer':
        """
        Load trained model.
        
        Args:
            filepath: Path to model file
            model_name: Optional model name override
            
        Returns:
            Loaded model trainer
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        if model_name is None:
            model_name = model_data.get('model_name', 'loaded_model')
        
        # This is a base class method, so we need to determine the correct subclass
        # In practice, subclasses should override this method
        instance = cls(model_name=model_name)
        
        # Load components
        instance.model = model_data['model']
        instance.feature_pipeline = model_data.get('feature_pipeline')
        instance.training_metadata = model_data.get('metadata', {})
        
        # Load metrics
        metrics_dict = model_data.get('metrics', {})
        instance.metrics = TrainingMetrics()
        for key, value in metrics_dict.items():
            if hasattr(instance.metrics, key):
                setattr(instance.metrics, key, value)
        
        instance.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'metadata': self.training_metadata,
            'metrics': self.metrics.to_dict() if self.is_trained else {},
        }
        
        if self.is_trained and hasattr(self.model, '__dict__'):
            summary['model_parameters'] = str(self.model.get_params())
        
        return summary