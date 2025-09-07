"""
Random Forest Model Trainer
Implements Random Forest classifier with hyperparameter optimization
and comprehensive evaluation for loan eligibility prediction.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from .base_trainer import BaseModelTrainer
import logging

logger = logging.getLogger(__name__)


class RandomForestTrainer(BaseModelTrainer):
    """
    Random Forest model trainer with advanced hyperparameter tuning.
    
    Provides robust ensemble learning with feature importance analysis,
    out-of-bag scoring, and comprehensive hyperparameter optimization.
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 verbose: bool = True,
                 model_dir: str = "trained_models",
                 enable_monitoring: bool = True,
                 n_jobs: int = -1):
        """
        Initialize Random Forest trainer.
        
        Args:
            random_state: Random state for reproducibility
            verbose: Enable verbose logging
            model_dir: Directory to save trained models
            enable_monitoring: Enable training monitoring
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        super().__init__(
            model_name="RandomForest",
            random_state=random_state,
            verbose=verbose,
            model_dir=model_dir,
            enable_monitoring=enable_monitoring
        )
        self.n_jobs = n_jobs
    
    def _create_model(self, **hyperparameters) -> RandomForestClassifier:
        """Create Random Forest model with specified hyperparameters."""
        # Set default n_jobs if not provided
        if 'n_jobs' not in hyperparameters:
            hyperparameters['n_jobs'] = self.n_jobs
        
        if 'random_state' not in hyperparameters:
            hyperparameters['random_state'] = self.random_state
        
        return RandomForestClassifier(**hyperparameters)
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Return production-ready default hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': 'sqrt',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': 0,
            'warm_start': False,
            'class_weight': 'balanced',  # Handle imbalanced data
            'ccp_alpha': 0.0,
            'max_samples': None
        }
    
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """Return hyperparameter space for tuning."""
        return {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None],
            'min_impurity_decrease': [0.0, 0.01, 0.02, 0.05],
            'ccp_alpha': [0.0, 0.01, 0.02, 0.05]
        }
    
    def get_loan_specific_hyperparameters(self) -> Dict[str, Any]:
        """Return loan-specific optimized hyperparameters."""
        return {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'class_weight': 'balanced',
            'min_impurity_decrease': 0.01,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': 0
        }
    
    def train_with_hyperparameter_search(self,
                                       X: Union[pd.DataFrame, np.ndarray],
                                       y: Union[pd.Series, np.ndarray],
                                       validation_split: float = 0.2,
                                       n_iter: int = 50,
                                       cv_folds: int = 5,
                                       scoring: str = 'roc_auc',
                                       **kwargs) -> 'RandomForestTrainer':
        """
        Train with automated hyperparameter search.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Validation split ratio
            n_iter: Number of hyperparameter combinations to try
            cv_folds: Cross-validation folds
            scoring: Scoring metric for optimization
            **kwargs: Additional training arguments
            
        Returns:
            Self for method chaining
        """
        if self.verbose:
            logger.info(f"Starting hyperparameter search with {n_iter} iterations...")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.training_metadata['feature_names'] = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Create base model
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # Setup parameter search
        param_distributions = self.get_hyperparameter_space()
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0,
            return_train_score=True
        )
        
        random_search.fit(X, y)
        
        # Store best hyperparameters
        best_params = random_search.best_params_
        self.training_metadata['hyperparameters'] = best_params
        self.training_metadata['best_cv_score'] = random_search.best_score_
        
        if self.verbose:
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")
            logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        return self.train(X, y, validation_split=validation_split, 
                         hyperparameters=best_params, **kwargs)
    
    def get_feature_importance_ranking(self, 
                                     feature_names: Optional[list] = None,
                                     top_k: int = 20) -> pd.DataFrame:
        """
        Get ranked feature importance with names.
        
        Args:
            feature_names: List of feature names
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance rankings
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance = self.get_feature_importance()
        if importance is None:
            raise ValueError("Model does not have feature importance")
        
        # Create feature names if not provided
        if feature_names is None:
            if self.training_metadata.get('feature_names'):
                feature_names = self.training_metadata['feature_names']
            else:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Add ranking
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        # Return top k
        return importance_df.head(top_k)
    
    def get_oob_score(self) -> Optional[float]:
        """Get out-of-bag score if available."""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        return None
    
    def analyze_tree_stats(self) -> Dict[str, Any]:
        """Analyze Random Forest tree statistics."""
        if not self.is_trained:
            raise ValueError("Model must be trained to analyze trees")
        
        trees = self.model.estimators_
        
        # Calculate tree statistics
        depths = [tree.tree_.max_depth for tree in trees]
        node_counts = [tree.tree_.node_count for tree in trees]
        leaf_counts = [tree.tree_.n_leaves for tree in trees]
        
        stats = {
            'n_trees': len(trees),
            'avg_depth': np.mean(depths),
            'max_depth': np.max(depths),
            'min_depth': np.min(depths),
            'std_depth': np.std(depths),
            'avg_nodes': np.mean(node_counts),
            'max_nodes': np.max(node_counts),
            'min_nodes': np.min(node_counts),
            'avg_leaves': np.mean(leaf_counts),
            'max_leaves': np.max(leaf_counts),
            'min_leaves': np.min(leaf_counts),
            'oob_score': self.get_oob_score()
        }
        
        return stats
    
    def get_prediction_confidence(self, 
                                X: Union[pd.DataFrame, np.ndarray],
                                confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Analyze prediction confidence using ensemble voting.
        
        Args:
            X: Input features
            confidence_threshold: Threshold for high-confidence predictions
            
        Returns:
            Confidence analysis results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained for confidence analysis")
        
        # Get predictions from all trees
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)
        max_probabilities = np.max(probabilities, axis=1)
        
        # Analyze confidence
        high_confidence_mask = max_probabilities >= confidence_threshold
        high_confidence_count = np.sum(high_confidence_mask)
        
        confidence_analysis = {
            'total_predictions': len(X),
            'high_confidence_predictions': high_confidence_count,
            'high_confidence_ratio': high_confidence_count / len(X),
            'avg_confidence': np.mean(max_probabilities),
            'min_confidence': np.min(max_probabilities),
            'max_confidence': np.max(max_probabilities),
            'confidence_std': np.std(max_probabilities),
            'confidence_threshold': confidence_threshold
        }
        
        return confidence_analysis
    
    def explain_predictions(self, 
                           X: Union[pd.DataFrame, np.ndarray],
                           feature_names: Optional[list] = None,
                           n_samples: int = 5) -> Dict[str, Any]:
        """
        Explain predictions using feature contributions.
        
        Args:
            X: Input features
            feature_names: Feature names
            n_samples: Number of sample explanations
            
        Returns:
            Prediction explanations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained for explanations")
        
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X = X.values
        
        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)
        
        # Get feature names
        if feature_names is None:
            if self.training_metadata.get('feature_names'):
                feature_names = self.training_metadata['feature_names']
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Get predictions and importance
        predictions = self.model.predict(X[:n_samples])
        probabilities = self.model.predict_proba(X[:n_samples])
        importance = self.get_feature_importance()
        
        explanations = []
        
        for i in range(min(n_samples, len(X))):
            # Get top contributing features
            feature_values = X[i]
            feature_contributions = feature_values * importance
            
            # Sort by contribution
            contribution_indices = np.argsort(np.abs(feature_contributions))[::-1][:10]
            
            explanation = {
                'sample_index': i,
                'prediction': predictions[i],
                'probability': probabilities[i],
                'top_features': [
                    {
                        'feature': feature_names[idx],
                        'value': feature_values[idx],
                        'importance': importance[idx],
                        'contribution': feature_contributions[idx]
                    }
                    for idx in contribution_indices
                ]
            }
            
            explanations.append(explanation)
        
        return {
            'explanations': explanations,
            'global_feature_importance': self.get_feature_importance_ranking(feature_names)
        }
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RandomForestTrainer':
        """Load Random Forest model from file."""
        instance = super().load_model(filepath, "RandomForest")
        
        # Convert to correct class
        rf_instance = cls(
            random_state=instance.training_metadata.get('random_state', 42),
            verbose=True
        )
        
        # Copy all attributes
        for attr_name in ['model', 'feature_pipeline', 'metrics', 'training_metadata', 'is_trained']:
            setattr(rf_instance, attr_name, getattr(instance, attr_name))
        
        return rf_instance


# Convenience function for quick training
def train_random_forest(X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       hyperparameter_search: bool = False,
                       loan_optimized: bool = True,
                       **kwargs) -> RandomForestTrainer:
    """
    Quick training function for Random Forest.
    
    Args:
        X: Training features
        y: Training targets
        hyperparameter_search: Whether to perform hyperparameter search
        loan_optimized: Use loan-specific hyperparameters
        **kwargs: Additional training arguments
        
    Returns:
        Trained Random Forest model
    """
    trainer = RandomForestTrainer(**kwargs)
    
    if hyperparameter_search:
        return trainer.train_with_hyperparameter_search(X, y, **kwargs)
    else:
        hyperparams = trainer.get_loan_specific_hyperparameters() if loan_optimized else None
        return trainer.train(X, y, hyperparameters=hyperparams, **kwargs)