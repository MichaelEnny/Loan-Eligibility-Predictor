"""
Neural Network Model Trainer
Implements neural network classifier using scikit-learn's MLPClassifier
with advanced hyperparameter optimization and regularization techniques.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import time
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import logging

from .base_trainer import BaseModelTrainer

logger = logging.getLogger(__name__)

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)


class NeuralNetworkTrainer(BaseModelTrainer):
    """
    Neural Network model trainer using MLPClassifier.
    
    Provides deep learning capabilities with multiple hidden layers,
    various activation functions, regularization techniques, and 
    comprehensive hyperparameter optimization for loan eligibility prediction.
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 verbose: bool = True,
                 model_dir: str = "trained_models",
                 enable_monitoring: bool = True,
                 auto_scale: bool = True,
                 max_iter: int = 1000):
        """
        Initialize Neural Network trainer.
        
        Args:
            random_state: Random state for reproducibility
            verbose: Enable verbose logging
            model_dir: Directory to save trained models
            enable_monitoring: Enable training monitoring
            auto_scale: Automatically scale features
            max_iter: Maximum number of training iterations
        """
        super().__init__(
            model_name="NeuralNetwork",
            random_state=random_state,
            verbose=verbose,
            model_dir=model_dir,
            enable_monitoring=enable_monitoring
        )
        self.auto_scale = auto_scale
        self.max_iter = max_iter
        self.scaler = None
        self.training_curves = {}
    
    def _create_model(self, **hyperparameters) -> MLPClassifier:
        """Create Neural Network model with specified hyperparameters."""
        # Set default parameters if not provided
        if 'random_state' not in hyperparameters:
            hyperparameters['random_state'] = self.random_state
        
        if 'max_iter' not in hyperparameters:
            hyperparameters['max_iter'] = self.max_iter
        
        if 'verbose' not in hyperparameters:
            hyperparameters['verbose'] = self.verbose
        
        return MLPClassifier(**hyperparameters)
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Return production-ready default hyperparameters."""
        return {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,  # L2 regularization
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'power_t': 0.5,
            'max_iter': self.max_iter,
            'shuffle': True,
            'random_state': self.random_state,
            'tol': 1e-4,
            'verbose': self.verbose,
            'warm_start': False,
            'momentum': 0.9,
            'nesterovs_momentum': True,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-8,
            'n_iter_no_change': 10,
            'max_fun': 15000
        }
    
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """Return hyperparameter space for tuning."""
        return {
            'hidden_layer_sizes': [
                (50,), (100,), (150,), (200,),
                (50, 25), (100, 50), (150, 75), (200, 100),
                (100, 50, 25), (150, 100, 50), (200, 100, 50),
                (300, 200, 100), (400, 200, 100)
            ],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
            'momentum': [0.8, 0.9, 0.95, 0.99],
            'beta_1': [0.85, 0.9, 0.95],
            'beta_2': [0.99, 0.999, 0.9999],
            'early_stopping': [True, False],
            'validation_fraction': [0.1, 0.15, 0.2]
        }
    
    def get_loan_specific_hyperparameters(self) -> Dict[str, Any]:
        """Return loan-specific optimized hyperparameters."""
        return {
            'hidden_layer_sizes': (150, 100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.01,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': self.max_iter,
            'shuffle': True,
            'random_state': self.random_state,
            'tol': 1e-4,
            'verbose': self.verbose,
            'early_stopping': True,
            'validation_fraction': 0.15,
            'momentum': 0.9,
            'nesterovs_momentum': True,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-8,
            'n_iter_no_change': 15
        }
    
    def _setup_feature_scaling(self, X_train: np.ndarray, X_val: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Setup and apply feature scaling for neural networks."""
        if self.auto_scale:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
            else:
                X_train_scaled = self.scaler.transform(X_train)
            
            X_val_scaled = None
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
            
            return X_train_scaled, X_val_scaled
        else:
            return X_train, X_val
    
    def train(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray],
              validation_split: float = 0.2,
              hyperparameters: Optional[Dict[str, Any]] = None,
              feature_pipeline: Optional[Any] = None,
              **kwargs) -> 'NeuralNetworkTrainer':
        """
        Train Neural Network model with feature scaling and validation monitoring.
        
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
        
        # Apply feature scaling
        X_train_scaled, X_val_scaled = self._setup_feature_scaling(X_train, X_val)
        
        # Get hyperparameters
        if hyperparameters is None:
            hyperparameters = self.get_default_hyperparameters()
        self.training_metadata['hyperparameters'] = hyperparameters
        
        # Create and train model
        self.model = self._create_model(**hyperparameters)
        
        if self.verbose:
            logger.info(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
            logger.info(f"Network architecture: {hyperparameters.get('hidden_layer_sizes', 'default')}")
            logger.info(f"Activation function: {hyperparameters.get('activation', 'relu')}")
            logger.info(f"Solver: {hyperparameters.get('solver', 'adam')}")
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Store training curves if available
        if hasattr(self.model, 'loss_curve_'):
            self.training_curves['loss_curve'] = self.model.loss_curve_
        if hasattr(self.model, 'validation_scores_'):
            self.training_curves['validation_scores'] = self.model.validation_scores_
        
        # Calculate training time
        self.metrics.training_time = time.time() - start_time
        
        # Validation predictions
        val_pred = self.model.predict(X_val_scaled)
        val_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
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
        
        # Model size and inference time
        self.metrics.model_size_mb = self._get_model_size()
        self.metrics.inference_time_ms = self._measure_inference_time(X_val_scaled)
        
        self.is_trained = True
        
        if self.verbose:
            logger.info(f"Training completed in {self.metrics.training_time:.2f}s")
            logger.info(f"Final training iterations: {self.model.n_iter_}")
            logger.info(f"Converged: {'Yes' if self.model.n_iter_ < self.model.max_iter else 'No'}")
            logger.info(f"Validation accuracy: {self.metrics.validation_scores.get('val_accuracy', 0):.4f}")
            logger.info(f"Validation F1-score: {self.metrics.validation_scores.get('val_f1_score', 0):.4f}")
            logger.info(f"Validation ROC-AUC: {self.metrics.validation_scores.get('val_roc_auc', 0):.4f}")
        
        return self
    
    def train_with_hyperparameter_search(self,
                                       X: Union[pd.DataFrame, np.ndarray],
                                       y: Union[pd.Series, np.ndarray],
                                       validation_split: float = 0.2,
                                       n_iter: int = 30,
                                       cv_folds: int = 3,  # Lower CV folds due to computational cost
                                       scoring: str = 'roc_auc',
                                       **kwargs) -> 'NeuralNetworkTrainer':
        """
        Train with automated hyperparameter search.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Validation split ratio
            n_iter: Number of hyperparameter combinations to try
            cv_folds: Cross-validation folds (lower for neural networks)
            scoring: Scoring metric for optimization
            **kwargs: Additional training arguments
            
        Returns:
            Self for method chaining
        """
        if self.verbose:
            logger.info(f"Starting hyperparameter search with {n_iter} iterations...")
            logger.warning("Neural network hyperparameter search may take longer than other models")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.training_metadata['feature_names'] = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Apply feature scaling
        if self.auto_scale:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Create base model with reduced max_iter for search
        base_model = MLPClassifier(
            random_state=self.random_state,
            max_iter=500,  # Reduced for faster search
            verbose=False,
            early_stopping=True
        )
        
        # Setup parameter search with reduced complexity
        param_distributions = {
            'hidden_layer_sizes': [
                (50,), (100,), (150,),
                (50, 25), (100, 50), (150, 75),
                (100, 50, 25)
            ],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'early_stopping': [True],
            'validation_fraction': [0.1, 0.15]
        }
        
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
            n_jobs=1,  # Neural networks don't parallelize well in sklearn
            verbose=1 if self.verbose else 0,
            return_train_score=True
        )
        
        random_search.fit(X_scaled, y)
        
        # Store best hyperparameters and increase max_iter for final training
        best_params = random_search.best_params_
        best_params['max_iter'] = self.max_iter  # Use full iterations for final model
        
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
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions with automatic feature scaling."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)
        
        # Apply feature scaling
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make probability predictions with automatic feature scaling."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)
        
        # Apply feature scaling
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def get_network_weights(self) -> Dict[str, np.ndarray]:
        """Get neural network weights and biases."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get weights")
        
        weights = {}
        for i, (coef, intercept) in enumerate(zip(self.model.coefs_, self.model.intercepts_)):
            weights[f'layer_{i}_weights'] = coef
            weights[f'layer_{i}_bias'] = intercept
        
        return weights
    
    def analyze_network_complexity(self) -> Dict[str, Any]:
        """Analyze neural network complexity and structure."""
        if not self.is_trained:
            raise ValueError("Model must be trained to analyze complexity")
        
        weights = self.get_network_weights()
        
        # Calculate network statistics
        total_params = sum(w.size for w in self.model.coefs_) + sum(b.size for b in self.model.intercepts_)
        layer_sizes = [self.model.coefs_[0].shape[0]] + [coef.shape[1] for coef in self.model.coefs_]
        
        complexity_stats = {
            'architecture': layer_sizes,
            'hidden_layers': len(self.model.coefs_) - 1,
            'total_parameters': total_params,
            'trainable_parameters': total_params,  # All parameters are trainable in MLPClassifier
            'activation_function': self.model.activation,
            'solver': self.model.solver,
            'learning_rate_init': self.model.learning_rate_init,
            'alpha_regularization': self.model.alpha,
            'n_iterations': self.model.n_iter_,
            'converged': self.model.n_iter_ < self.model.max_iter,
            'loss_curve_length': len(self.training_curves.get('loss_curve', [])),
            'final_loss': self.training_curves.get('loss_curve', [None])[-1] if self.training_curves.get('loss_curve') else None
        }
        
        return complexity_stats
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """Plot training curves if available."""
        if not self.training_curves:
            logger.warning("No training curves available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot loss curve
            if 'loss_curve' in self.training_curves:
                axes[0].plot(self.training_curves['loss_curve'], label='Training Loss')
                axes[0].set_title('Training Loss Curve')
                axes[0].set_xlabel('Iteration')
                axes[0].set_ylabel('Loss')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
            
            # Plot validation scores if available
            if 'validation_scores' in self.training_curves:
                axes[1].plot(self.training_curves['validation_scores'], label='Validation Score', color='orange')
                axes[1].set_title('Validation Score Curve')
                axes[1].set_xlabel('Iteration')
                axes[1].set_ylabel('Score')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
            else:
                axes[1].text(0.5, 0.5, 'No validation scores\navailable', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Validation Score')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if self.verbose:
                    logger.info(f"Training curves saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available for plotting training curves")
    
    def get_layer_activations(self, X: Union[pd.DataFrame, np.ndarray], layer_idx: int = -1) -> np.ndarray:
        """Get activations from a specific layer (requires sklearn >= 0.24)."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get activations")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply feature pipeline and scaling
        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # This is a simplified version - full implementation would require
        # manually forward passing through the network
        logger.warning("Layer activation extraction is limited in scikit-learn MLPClassifier")
        
        # Return prediction probabilities as proxy
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """Save model including scaler."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Save the base model
        model_path = super().save_model(filepath)
        
        # Save scaler separately if it exists
        if self.scaler is not None:
            import pickle
            from pathlib import Path
            scaler_path = Path(model_path).with_name(Path(model_path).stem + '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            if self.verbose:
                logger.info(f"Scaler saved to {scaler_path}")
        
        return model_path
    
    @classmethod
    def load_model(cls, filepath: str) -> 'NeuralNetworkTrainer':
        """Load Neural Network model with scaler."""
        instance = super().load_model(filepath, "NeuralNetwork")
        
        # Convert to correct class
        nn_instance = cls(
            random_state=instance.training_metadata.get('random_state', 42),
            verbose=True
        )
        
        # Copy all attributes
        for attr_name in ['model', 'feature_pipeline', 'metrics', 'training_metadata', 'is_trained', 'training_curves']:
            if hasattr(instance, attr_name):
                setattr(nn_instance, attr_name, getattr(instance, attr_name))
        
        # Load scaler if it exists
        import pickle
        from pathlib import Path
        scaler_path = Path(filepath).with_name(Path(filepath).stem + '_scaler.pkl')
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                nn_instance.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        return nn_instance


# Convenience function for quick training
def train_neural_network(X: Union[pd.DataFrame, np.ndarray],
                        y: Union[pd.Series, np.ndarray],
                        hyperparameter_search: bool = False,
                        loan_optimized: bool = True,
                        auto_scale: bool = True,
                        max_iter: int = 1000,
                        **kwargs) -> NeuralNetworkTrainer:
    """
    Quick training function for Neural Network.
    
    Args:
        X: Training features
        y: Training targets
        hyperparameter_search: Whether to perform hyperparameter search
        loan_optimized: Use loan-specific hyperparameters
        auto_scale: Automatically scale features
        max_iter: Maximum training iterations
        **kwargs: Additional training arguments
        
    Returns:
        Trained Neural Network model
    """
    trainer = NeuralNetworkTrainer(auto_scale=auto_scale, max_iter=max_iter, **kwargs)
    
    if hyperparameter_search:
        return trainer.train_with_hyperparameter_search(X, y, **kwargs)
    else:
        hyperparams = trainer.get_loan_specific_hyperparameters() if loan_optimized else None
        return trainer.train(X, y, hyperparameters=hyperparams, **kwargs)