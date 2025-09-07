"""
XGBoost Model Trainer
Implements XGBoost classifier with advanced hyperparameter optimization,
early stopping, and comprehensive evaluation for loan eligibility prediction.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import xgboost as xgb
import time
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import log_loss
import logging

from .base_trainer import BaseModelTrainer

logger = logging.getLogger(__name__)


class XGBoostTrainer(BaseModelTrainer):
    """
    XGBoost model trainer with advanced gradient boosting optimization.
    
    Provides state-of-the-art gradient boosting with early stopping,
    feature importance analysis, and comprehensive hyperparameter tuning
    specifically optimized for loan eligibility prediction.
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 verbose: bool = True,
                 model_dir: str = "trained_models",
                 enable_monitoring: bool = True,
                 n_jobs: int = -1,
                 use_gpu: bool = False):
        """
        Initialize XGBoost trainer.
        
        Args:
            random_state: Random state for reproducibility
            verbose: Enable verbose logging
            model_dir: Directory to save trained models
            enable_monitoring: Enable training monitoring
            n_jobs: Number of parallel jobs (-1 for all cores)
            use_gpu: Whether to use GPU acceleration
        """
        super().__init__(
            model_name="XGBoost",
            random_state=random_state,
            verbose=verbose,
            model_dir=model_dir,
            enable_monitoring=enable_monitoring
        )
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.early_stopping_patience = 50
        self.eval_results = {}
    
    def _create_model(self, **hyperparameters) -> xgb.XGBClassifier:
        """Create XGBoost model with specified hyperparameters."""
        # Set default parameters if not provided
        if 'n_jobs' not in hyperparameters:
            hyperparameters['n_jobs'] = self.n_jobs
        
        if 'random_state' not in hyperparameters:
            hyperparameters['random_state'] = self.random_state
        
        if 'tree_method' not in hyperparameters:
            hyperparameters['tree_method'] = 'gpu_hist' if self.use_gpu else 'hist'
        
        if 'verbosity' not in hyperparameters:
            hyperparameters['verbosity'] = 1 if self.verbose else 0
        
        # Enable evaluation metrics
        hyperparameters['eval_metric'] = ['logloss', 'auc', 'error']
        
        return xgb.XGBClassifier(**hyperparameters)
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Return production-ready default hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'colsample_bylevel': 1.0,
            'colsample_bynode': 1.0,
            'gamma': 0,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,  # Will be auto-calculated for imbalanced data
            'max_delta_step': 0,
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbosity': 1 if self.verbose else 0,
            'eval_metric': ['logloss', 'auc', 'error']
        }
    
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """Return hyperparameter space for tuning."""
        return {
            'n_estimators': [100, 200, 300, 500, 800, 1000],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5, 1, 2, 5],
            'min_child_weight': [1, 2, 3, 5, 7, 10],
            'reg_alpha': [0, 0.01, 0.1, 1, 10, 100],
            'reg_lambda': [1, 1.5, 2, 5, 10, 50, 100],
            'scale_pos_weight': [1, 2, 3, 5, 10]  # For imbalanced datasets
        }
    
    def get_loan_specific_hyperparameters(self) -> Dict[str, Any]:
        """Return loan-specific optimized hyperparameters."""
        return {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'gamma': 0.1,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'scale_pos_weight': 1,  # Will be calculated automatically
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbosity': 1 if self.verbose else 0,
            'eval_metric': ['logloss', 'auc', 'error']
        }
    
    def _calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """Calculate scale_pos_weight for imbalanced datasets."""
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        
        if pos_count == 0 or neg_count == 0:
            return 1.0
        
        return neg_count / pos_count
    
    def train(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray],
              validation_split: float = 0.2,
              hyperparameters: Optional[Dict[str, Any]] = None,
              feature_pipeline: Optional[Any] = None,
              early_stopping: bool = True,
              **kwargs) -> 'XGBoostTrainer':
        """
        Train XGBoost model with early stopping and validation monitoring.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Proportion for validation split
            hyperparameters: Model hyperparameters
            feature_pipeline: Optional feature engineering pipeline
            early_stopping: Enable early stopping
            **kwargs: Additional training arguments
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"Starting {self.model_name} training with early stopping...")
        
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
        from sklearn.model_selection import train_test_split
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
        
        # Calculate scale_pos_weight for imbalanced data
        if hyperparameters.get('scale_pos_weight', 1) == 1:
            scale_pos_weight = self._calculate_scale_pos_weight(y_train)
            hyperparameters['scale_pos_weight'] = scale_pos_weight
            if self.verbose:
                logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
        
        self.training_metadata['hyperparameters'] = hyperparameters
        
        # Create model
        self.model = self._create_model(**hyperparameters)
        
        if self.verbose:
            logger.info(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
        
        # Train model (using basic API for compatibility)
        self.model.fit(X_train, y_train)
        
        # Calculate training time
        self.metrics.training_time = time.time() - start_time
        
        # Validation predictions
        val_pred = self.model.predict(X_val)
        val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        self.metrics.validation_scores = self._calculate_metrics(
            y_val, val_pred, val_pred_proba, "val_"
        )
        
        # Store confusion matrix and classification report
        from sklearn.metrics import confusion_matrix, classification_report
        self.metrics.confusion_matrix = confusion_matrix(y_val, val_pred)
        self.metrics.classification_report = classification_report(
            y_val, val_pred, target_names=[f'Class_{i}' for i in np.unique(y)]
        )
        
        # Feature importance
        self.metrics.feature_importance = self.model.feature_importances_
        
        # Model size and inference time
        self.metrics.model_size_mb = self._get_model_size()
        self.metrics.inference_time_ms = self._measure_inference_time(X_val)
        
        self.is_trained = True
        
        if self.verbose:
            logger.info(f"Training completed in {self.metrics.training_time:.2f}s")
            logger.info(f"Validation accuracy: {self.metrics.validation_scores.get('val_accuracy', 0):.4f}")
            logger.info(f"Validation F1-score: {self.metrics.validation_scores.get('val_f1_score', 0):.4f}")
            logger.info(f"Validation ROC-AUC: {self.metrics.validation_scores.get('val_roc_auc', 0):.4f}")
        
        return self
    
    def train_with_hyperparameter_search(self,
                                       X: Union[pd.DataFrame, np.ndarray],
                                       y: Union[pd.Series, np.ndarray],
                                       validation_split: float = 0.2,
                                       n_iter: int = 50,
                                       cv_folds: int = 5,
                                       scoring: str = 'roc_auc',
                                       **kwargs) -> 'XGBoostTrainer':
        """
        Train with automated hyperparameter search using cross-validation.
        
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
        base_model = xgb.XGBClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method='gpu_hist' if self.use_gpu else 'hist',
            verbosity=0,
            eval_metric='auc'
        )
        
        # Setup parameter search
        param_distributions = self.get_hyperparameter_space()
        
        # Calculate scale_pos_weight
        scale_pos_weight = self._calculate_scale_pos_weight(y)
        param_distributions['scale_pos_weight'] = [1, scale_pos_weight, scale_pos_weight * 0.5, scale_pos_weight * 2]
        
        # Create stratified K-fold for CV
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
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
        self.training_metadata['cv_results'] = {
            'best_score': random_search.best_score_,
            'best_params': best_params,
            'cv_results_df': pd.DataFrame(random_search.cv_results_)
        }
        
        if self.verbose:
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")
            logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        return self.train(X, y, validation_split=validation_split, 
                         hyperparameters=best_params, **kwargs)
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history from evaluation results."""
        if not self.eval_results:
            return {}
        
        history = {}
        for dataset_name, metrics in self.eval_results.items():
            for metric_name, values in metrics.items():
                history[f'{dataset_name}_{metric_name}'] = values
        
        return history
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """Plot training curves if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
            
            history = self.get_training_history()
            if not history:
                logger.warning("No training history available for plotting")
                return
            
            # Create subplots for different metrics
            metrics = set([key.split('_', 1)[1] for key in history.keys()])
            n_metrics = len(metrics)
            
            if n_metrics == 0:
                return
            
            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                for dataset in ['train', 'val']:
                    key = f'{dataset}_{metric}'
                    if key in history:
                        ax.plot(history[key], label=f'{dataset}_{metric}')
                
                ax.set_title(f'Training {metric}')
                ax.set_xlabel('Iteration')
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if self.verbose:
                    logger.info(f"Training curves saved to {save_path}")
            else:
                plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting training curves")
    
    def get_feature_importance_types(self) -> Dict[str, np.ndarray]:
        """Get different types of feature importance."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_types = {}
        
        # Weight-based importance (default)
        importance_types['weight'] = self.model.feature_importances_
        
        # Get other importance types if available
        try:
            # Gain-based importance
            importance_types['gain'] = self.model.get_booster().get_score(importance_type='gain')
            
            # Cover-based importance  
            importance_types['cover'] = self.model.get_booster().get_score(importance_type='cover')
            
            # Convert dictionaries to arrays
            n_features = len(importance_types['weight'])
            
            for imp_type in ['gain', 'cover']:
                if imp_type in importance_types:
                    imp_dict = importance_types[imp_type]
                    imp_array = np.zeros(n_features)
                    for i in range(n_features):
                        feature_name = f'f{i}'
                        if feature_name in imp_dict:
                            imp_array[i] = imp_dict[feature_name]
                    importance_types[imp_type] = imp_array
                    
        except Exception as e:
            if self.verbose:
                logger.warning(f"Could not get additional importance types: {e}")
        
        return importance_types
    
    def get_prediction_leaf_indices(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get leaf indices for each prediction (useful for similarity analysis)."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get leaf indices")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)
        
        return self.model.apply(X)
    
    def analyze_tree_structure(self) -> Dict[str, Any]:
        """Analyze XGBoost tree structure and complexity."""
        if not self.is_trained:
            raise ValueError("Model must be trained to analyze tree structure")
        
        booster = self.model.get_booster()
        
        # Get model dump
        tree_dump = booster.get_dump(dump_format='json')
        
        # Analyze trees
        tree_depths = []
        tree_nodes = []
        
        import json
        for tree_json in tree_dump:
            tree_data = json.loads(tree_json)
            
            def get_tree_depth(node, depth=0):
                if 'children' not in node:
                    return depth
                return max(get_tree_depth(child, depth + 1) for child in node['children'])
            
            def count_nodes(node):
                if 'children' not in node:
                    return 1
                return 1 + sum(count_nodes(child) for child in node['children'])
            
            tree_depths.append(get_tree_depth(tree_data))
            tree_nodes.append(count_nodes(tree_data))
        
        structure_stats = {
            'n_trees': len(tree_dump),
            'avg_depth': np.mean(tree_depths),
            'max_depth': np.max(tree_depths),
            'min_depth': np.min(tree_depths),
            'avg_nodes': np.mean(tree_nodes),
            'total_nodes': np.sum(tree_nodes),
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'best_score': getattr(self.model, 'best_score', None)
        }
        
        return structure_stats
    
    @classmethod
    def load_model(cls, filepath: str) -> 'XGBoostTrainer':
        """Load XGBoost model from file."""
        instance = super().load_model(filepath, "XGBoost")
        
        # Convert to correct class
        xgb_instance = cls(
            random_state=instance.training_metadata.get('random_state', 42),
            verbose=True
        )
        
        # Copy all attributes
        for attr_name in ['model', 'feature_pipeline', 'metrics', 'training_metadata', 'is_trained', 'eval_results']:
            if hasattr(instance, attr_name):
                setattr(xgb_instance, attr_name, getattr(instance, attr_name))
        
        return xgb_instance


# Convenience function for quick training
def train_xgboost(X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, np.ndarray],
                  hyperparameter_search: bool = False,
                  loan_optimized: bool = True,
                  early_stopping: bool = True,
                  use_gpu: bool = False,
                  **kwargs) -> XGBoostTrainer:
    """
    Quick training function for XGBoost.
    
    Args:
        X: Training features
        y: Training targets
        hyperparameter_search: Whether to perform hyperparameter search
        loan_optimized: Use loan-specific hyperparameters
        early_stopping: Enable early stopping
        use_gpu: Use GPU acceleration
        **kwargs: Additional training arguments
        
    Returns:
        Trained XGBoost model
    """
    trainer = XGBoostTrainer(use_gpu=use_gpu, **kwargs)
    
    if hyperparameter_search:
        return trainer.train_with_hyperparameter_search(X, y, **kwargs)
    else:
        hyperparams = trainer.get_loan_specific_hyperparameters() if loan_optimized else None
        return trainer.train(X, y, hyperparameters=hyperparams, 
                           early_stopping=early_stopping, **kwargs)