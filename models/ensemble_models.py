"""
Ensemble Model Implementations
Provides comprehensive ensemble methods for achieving maximum prediction accuracy
including voting, stacking, blending, and weight optimization.
"""

import os
import time
import json
import pickle
import logging
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
)
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# Import our base trainer
from .base_trainer import BaseModelTrainer, TrainingMetrics

# Import existing model trainers
from .random_forest_model import RandomForestTrainer
from .neural_network_model import NeuralNetworkTrainer
from .xgboost_model import XGBoostTrainer

logger = logging.getLogger(__name__)


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Custom weighted ensemble classifier."""
    
    def __init__(self, estimators: List[Tuple[str, BaseEstimator]], weights: np.ndarray = None):
        self.estimators = estimators
        self.weights = weights
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X, y):
        """Fit all base estimators."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Fit all estimators
        for name, estimator in self.estimators:
            estimator.fit(X, y)
        
        # Set equal weights if not provided
        if self.weights is None:
            self.weights = np.ones(len(self.estimators)) / len(self.estimators)
            
        return self
    
    def predict(self, X):
        """Make predictions using weighted voting."""
        predictions = []
        for name, estimator in self.estimators:
            predictions.append(estimator.predict(X))
        
        predictions = np.array(predictions)
        
        # Weighted voting
        weighted_predictions = np.zeros((X.shape[0], self.n_classes_))
        
        for i, (name, estimator) in enumerate(self.estimators):
            for j, class_label in enumerate(self.classes_):
                mask = predictions[i] == class_label
                weighted_predictions[mask, j] += self.weights[i]
        
        return self.classes_[np.argmax(weighted_predictions, axis=1)]
    
    def predict_proba(self, X):
        """Make probability predictions using weighted averaging."""
        probabilities = []
        for name, estimator in self.estimators:
            if hasattr(estimator, 'predict_proba'):
                probabilities.append(estimator.predict_proba(X))
            else:
                # Convert predictions to probabilities
                preds = estimator.predict(X)
                prob_matrix = np.zeros((X.shape[0], self.n_classes_))
                for i, pred in enumerate(preds):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    prob_matrix[i, class_idx] = 1.0
                probabilities.append(prob_matrix)
        
        # Weighted average of probabilities
        weighted_probs = np.zeros_like(probabilities[0])
        for i, probs in enumerate(probabilities):
            weighted_probs += self.weights[i] * probs
            
        return weighted_probs


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Custom stacking ensemble with cross-validation."""
    
    def __init__(self, base_estimators: List[Tuple[str, BaseEstimator]], 
                 meta_learner: BaseEstimator = None, cv: int = 5):
        self.base_estimators = base_estimators
        self.meta_learner = meta_learner if meta_learner is not None else LogisticRegression(random_state=42)
        self.cv = cv
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X, y):
        """Fit base estimators and meta-learner."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Fit all base estimators on full dataset
        for name, estimator in self.base_estimators:
            estimator.fit(X, y)
        
        # Fit meta-learner
        self.meta_learner.fit(meta_features, y)
        
        return self
    
    def _generate_meta_features(self, X, y):
        """Generate meta-features using cross-validation."""
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.base_estimators)))
        
        for i, (name, estimator) in enumerate(self.base_estimators):
            predictions = cross_val_predict(estimator, X, y, cv=skf, method='predict_proba')
            if predictions.shape[1] == 2:  # Binary classification
                meta_features[:, i] = predictions[:, 1]
            else:  # Multi-class - use max probability
                meta_features[:, i] = np.max(predictions, axis=1)
        
        return meta_features
    
    def predict(self, X):
        """Make predictions using stacked approach."""
        meta_features = self._get_base_predictions(X)
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X):
        """Make probability predictions using stacked approach."""
        meta_features = self._get_base_predictions(X)
        return self.meta_learner.predict_proba(meta_features)
    
    def _get_base_predictions(self, X):
        """Get predictions from base estimators."""
        meta_features = np.zeros((X.shape[0], len(self.base_estimators)))
        
        for i, (name, estimator) in enumerate(self.base_estimators):
            if hasattr(estimator, 'predict_proba'):
                probs = estimator.predict_proba(X)
                if probs.shape[1] == 2:  # Binary classification
                    meta_features[:, i] = probs[:, 1]
                else:  # Multi-class
                    meta_features[:, i] = np.max(probs, axis=1)
            else:
                # Use prediction confidence or binary predictions
                meta_features[:, i] = estimator.predict(X)
        
        return meta_features


class EnsembleTrainer(BaseModelTrainer):
    """
    Comprehensive Ensemble Model Trainer
    Implements voting, stacking, blending, and weight optimization for maximum accuracy.
    """
    
    def __init__(self, 
                 ensemble_type: str = "voting",
                 base_models: List[str] = None,
                 random_state: int = 42,
                 verbose: bool = True,
                 model_dir: str = "trained_models"):
        """
        Initialize ensemble trainer.
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'stacking', 'blending', 'weighted')
            base_models: List of base model names to use
            random_state: Random state for reproducibility
            verbose: Enable verbose logging
            model_dir: Directory for saving models
        """
        super().__init__(
            model_name=f"Ensemble_{ensemble_type.title()}",
            random_state=random_state,
            verbose=verbose,
            model_dir=model_dir
        )
        
        self.ensemble_type = ensemble_type
        self.base_models = base_models or ['RandomForest', 'XGBoost', 'NeuralNetwork']
        self.base_estimators = []
        self.optimal_weights = None
        
        # Ensemble-specific attributes
        self.meta_learner = None
        self.blend_holdout_size = 0.2
        self.cv_folds = 5
        
        logger.info(f"Initialized {ensemble_type} ensemble with models: {self.base_models}")
    
    def _create_base_estimators(self) -> List[Tuple[str, BaseEstimator]]:
        """Create base estimator instances."""
        estimators = []
        
        for model_name in self.base_models:
            if model_name == 'RandomForest':
                trainer = RandomForestTrainer(random_state=self.random_state, verbose=False)
                model = trainer._create_model(**trainer.get_default_hyperparameters())
                estimators.append((model_name, model))
                
            elif model_name == 'XGBoost':
                trainer = XGBoostTrainer(random_state=self.random_state, verbose=False)
                model = trainer._create_model(**trainer.get_default_hyperparameters())
                estimators.append((model_name, model))
                
            elif model_name == 'NeuralNetwork':
                trainer = NeuralNetworkTrainer(random_state=self.random_state, verbose=False)
                model = trainer._create_model(**trainer.get_default_hyperparameters())
                estimators.append((model_name, model))
        
        return estimators
    
    def _create_model(self, **hyperparameters) -> BaseEstimator:
        """Create ensemble model based on type."""
        self.base_estimators = self._create_base_estimators()
        
        if self.ensemble_type == "voting":
            voting_type = hyperparameters.get('voting', 'soft')
            return VotingClassifier(
                estimators=self.base_estimators,
                voting=voting_type
            )
        
        elif self.ensemble_type == "stacking":
            meta_learner = hyperparameters.get('meta_learner', LogisticRegression(random_state=self.random_state))
            cv_folds = hyperparameters.get('cv_folds', self.cv_folds)
            return StackingEnsemble(
                base_estimators=self.base_estimators,
                meta_learner=meta_learner,
                cv=cv_folds
            )
        
        elif self.ensemble_type == "weighted":
            weights = hyperparameters.get('weights', None)
            return WeightedEnsemble(
                estimators=self.base_estimators,
                weights=weights
            )
        
        elif self.ensemble_type == "blending":
            # Blending is implemented in the training process
            return WeightedEnsemble(estimators=self.base_estimators)
        
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for ensemble."""
        if self.ensemble_type == "voting":
            return {"voting": "soft"}
        elif self.ensemble_type == "stacking":
            return {
                "meta_learner": LogisticRegression(random_state=self.random_state),
                "cv_folds": 5
            }
        elif self.ensemble_type in ["weighted", "blending"]:
            return {"weights": None}
        else:
            return {}
    
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """Get hyperparameter space for tuning."""
        if self.ensemble_type == "voting":
            return {"voting": ["soft", "hard"]}
        elif self.ensemble_type == "stacking":
            return {
                "cv_folds": [3, 5, 7],
                "meta_learner": [
                    LogisticRegression(random_state=self.random_state),
                    RandomForestTrainer(random_state=self.random_state)._create_model(**{"n_estimators": 100})
                ]
            }
        else:
            return {}
    
    def _optimize_weights_differential_evolution(self, X_val: np.ndarray, y_val: np.ndarray, 
                                               base_predictions: np.ndarray) -> np.ndarray:
        """Optimize ensemble weights using differential evolution."""
        from scipy.optimize import differential_evolution
        
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize weights
            ensemble_pred = np.zeros(len(y_val))
            
            for i in range(len(y_val)):
                weighted_votes = {}
                for j, pred in enumerate(base_predictions[:, i]):
                    if pred not in weighted_votes:
                        weighted_votes[pred] = 0
                    weighted_votes[pred] += weights[j]
                
                ensemble_pred[i] = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
            
            return -accuracy_score(y_val, ensemble_pred)  # Negative for minimization
        
        # Ensure we have the correct number of models for bounds
        n_models = base_predictions.shape[0]  # Number of base models
        bounds = [(0.1, 1.0) for _ in range(n_models)]
        
        result = differential_evolution(
            objective, 
            bounds, 
            seed=self.random_state,
            maxiter=100
        )
        
        optimal_weights = result.x
        return optimal_weights / np.sum(optimal_weights)
    
    def _optimize_weights_bayesian(self, X_val: np.ndarray, y_val: np.ndarray, 
                                  base_predictions: np.ndarray) -> np.ndarray:
        """Optimize ensemble weights using Bayesian optimization."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.utils import use_named_args
            
            # Define search space
            dimensions = [Real(0.1, 1.0, name=f'weight_{i}') for i in range(len(self.base_estimators))]
            
            @use_named_args(dimensions)
            def objective(**params):
                weights = np.array([params[f'weight_{i}'] for i in range(len(self.base_estimators))])
                weights = weights / np.sum(weights)  # Normalize
                
                ensemble_pred = np.zeros(len(y_val))
                for i in range(len(y_val)):
                    weighted_votes = {}
                    for j, pred in enumerate(base_predictions[:, i]):
                        if pred not in weighted_votes:
                            weighted_votes[pred] = 0
                        weighted_votes[pred] += weights[j]
                    
                    ensemble_pred[i] = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
                
                return -accuracy_score(y_val, ensemble_pred)  # Negative for minimization
            
            result = gp_minimize(objective, dimensions, n_calls=50, random_state=self.random_state)
            optimal_weights = np.array(result.x)
            return optimal_weights / np.sum(optimal_weights)
            
        except ImportError:
            logger.warning("scikit-optimize not available, falling back to grid search")
            return self._optimize_weights_grid_search(X_val, y_val, base_predictions)
    
    def _optimize_weights_grid_search(self, X_val: np.ndarray, y_val: np.ndarray, 
                                     base_predictions: np.ndarray) -> np.ndarray:
        """Optimize ensemble weights using grid search."""
        best_weights = None
        best_score = 0
        
        # Generate weight combinations
        weight_steps = 5
        weight_values = np.linspace(0.1, 1.0, weight_steps)
        
        from itertools import product
        
        for weight_combo in product(weight_values, repeat=len(self.base_estimators)):
            weights = np.array(weight_combo)
            weights = weights / np.sum(weights)  # Normalize
            
            ensemble_pred = np.zeros(len(y_val))
            for i in range(len(y_val)):
                weighted_votes = {}
                for j, pred in enumerate(base_predictions[:, i]):
                    if pred not in weighted_votes:
                        weighted_votes[pred] = 0
                    weighted_votes[pred] += weights[j]
                
                ensemble_pred[i] = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
            
            score = accuracy_score(y_val, ensemble_pred)
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
        
        return best_weights if best_weights is not None else np.ones(len(self.base_estimators)) / len(self.base_estimators)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray],
              validation_split: float = 0.2,
              hyperparameters: Optional[Dict[str, Any]] = None,
              feature_pipeline: Optional[Any] = None,
              weight_optimization: str = "differential_evolution",
              **kwargs) -> 'EnsembleTrainer':
        """
        Train ensemble model with weight optimization.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Proportion for validation split
            hyperparameters: Ensemble hyperparameters
            feature_pipeline: Optional feature engineering pipeline
            weight_optimization: Weight optimization method ('differential_evolution', 'bayesian', 'grid_search')
            **kwargs: Additional training arguments
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
        if self.ensemble_type == "blending":
            # For blending, we need a separate holdout set
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y, test_size=self.blend_holdout_size, 
                random_state=self.random_state, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, 
                random_state=self.random_state, stratify=y_train
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, 
                random_state=self.random_state, stratify=y
            )
        
        # Apply feature pipeline if provided
        if feature_pipeline is not None:
            self.feature_pipeline = feature_pipeline
            X_train = feature_pipeline.fit_transform(X_train, y_train)
            X_val = feature_pipeline.transform(X_val)
            if self.ensemble_type == "blending":
                X_holdout = feature_pipeline.transform(X_holdout)
        
        # Get hyperparameters
        if hyperparameters is None:
            hyperparameters = self.get_default_hyperparameters()
        self.training_metadata['hyperparameters'] = hyperparameters
        
        # Handle different ensemble types
        if self.ensemble_type == "blending":
            self.model = self._train_blending_ensemble(
                X_train, y_train, X_val, y_val, X_holdout, y_holdout, 
                weight_optimization
            )
        elif self.ensemble_type == "weighted":
            self.model = self._train_weighted_ensemble(
                X_train, y_train, X_val, y_val, weight_optimization
            )
        else:
            # Standard training for voting and stacking
            self.model = self._create_model(**hyperparameters)
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
            
            if self.optimal_weights is not None:
                logger.info(f"Optimal weights: {self.optimal_weights}")
        
        return self
    
    def _train_blending_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                X_holdout: np.ndarray, y_holdout: np.ndarray,
                                weight_optimization: str) -> BaseEstimator:
        """Train blending ensemble."""
        # Train base models on training set
        base_models = []
        for name, estimator in self.base_estimators:
            estimator.fit(X_train, y_train)
            base_models.append((name, estimator))
        
        # Generate predictions on holdout set
        holdout_predictions = []
        for name, estimator in base_models:
            pred = estimator.predict(X_holdout)
            holdout_predictions.append(pred)
        
        holdout_predictions = np.array(holdout_predictions).T  # Shape: (n_samples, n_models)
        
        # Optimize weights on holdout set
        if weight_optimization == "bayesian":
            self.optimal_weights = self._optimize_weights_bayesian(
                X_holdout, y_holdout, holdout_predictions.T
            )
        elif weight_optimization == "grid_search":
            self.optimal_weights = self._optimize_weights_grid_search(
                X_holdout, y_holdout, holdout_predictions.T
            )
        else:  # differential_evolution
            self.optimal_weights = self._optimize_weights_differential_evolution(
                X_holdout, y_holdout, holdout_predictions.T
            )
        
        # Create final weighted ensemble
        return WeightedEnsemble(estimators=base_models, weights=self.optimal_weights)
    
    def _train_weighted_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                weight_optimization: str) -> BaseEstimator:
        """Train weighted ensemble with optimization."""
        # Train base models
        base_models = []
        for name, estimator in self.base_estimators:
            estimator.fit(X_train, y_train)
            base_models.append((name, estimator))
        
        # Generate predictions on validation set
        val_predictions = []
        for name, estimator in base_models:
            pred = estimator.predict(X_val)
            val_predictions.append(pred)
        
        val_predictions = np.array(val_predictions).T  # Shape: (n_samples, n_models)
        
        # Optimize weights
        if weight_optimization == "bayesian":
            self.optimal_weights = self._optimize_weights_bayesian(
                X_val, y_val, val_predictions.T
            )
        elif weight_optimization == "grid_search":
            self.optimal_weights = self._optimize_weights_grid_search(
                X_val, y_val, val_predictions.T
            )
        else:  # differential_evolution
            self.optimal_weights = self._optimize_weights_differential_evolution(
                X_val, y_val, val_predictions.T
            )
        
        # Retrain on full training + validation data
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
        
        final_models = []
        for name, estimator in self.base_estimators:
            # Create fresh estimator to avoid data leakage
            if name == 'RandomForest':
                trainer = RandomForestTrainer(random_state=self.random_state, verbose=False)
                fresh_model = trainer._create_model(**trainer.get_default_hyperparameters())
            elif name == 'XGBoost':
                trainer = XGBoostTrainer(random_state=self.random_state, verbose=False)
                fresh_model = trainer._create_model(**trainer.get_default_hyperparameters())
            elif name == 'NeuralNetwork':
                trainer = NeuralNetworkTrainer(random_state=self.random_state, verbose=False)
                fresh_model = trainer._create_model(**trainer.get_default_hyperparameters())
            
            fresh_model.fit(X_full, y_full)
            final_models.append((name, fresh_model))
        
        return WeightedEnsemble(estimators=final_models, weights=self.optimal_weights)
    
    def get_base_model_predictions(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get individual base model predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before getting base predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)
        
        predictions = {}
        
        if hasattr(self.model, 'estimators'):
            for name, estimator in self.model.estimators:
                predictions[name] = estimator.predict(X)
        elif hasattr(self.model, 'base_estimators'):
            for name, estimator in self.model.base_estimators:
                predictions[name] = estimator.predict(X)
        
        return predictions
    
    def get_ensemble_weights(self) -> Optional[np.ndarray]:
        """Get ensemble weights."""
        if hasattr(self.model, 'weights'):
            return self.model.weights
        return self.optimal_weights