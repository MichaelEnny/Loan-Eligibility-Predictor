"""
Advanced Hyperparameter Tuning Framework with Optuna
Provides intelligent hyperparameter optimization for all model types
with Bayesian optimization, pruning, and multi-objective optimization.
"""

from typing import Dict, Any, Optional, Union, List, Callable, Tuple
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
import json
import pickle
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
    from optuna.study import MaxTrialsCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna>=3.0.0")

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


class OptimizationObjective:
    """Defines optimization objectives for hyperparameter tuning."""
    
    def __init__(self, 
                 trainer_class,
                 X: np.ndarray,
                 y: np.ndarray,
                 cv_folds: int = 5,
                 scoring: str = 'roc_auc',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 timeout_per_trial: Optional[int] = None):
        """
        Initialize optimization objective.
        
        Args:
            trainer_class: Model trainer class
            X: Training features
            y: Training targets
            cv_folds: Number of CV folds
            scoring: Scoring metric
            test_size: Test set size for validation
            random_state: Random state
            timeout_per_trial: Timeout per trial in seconds
        """
        self.trainer_class = trainer_class
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state
        self.timeout_per_trial = timeout_per_trial
        
        # Split data once
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.trial_count = 0
        self.best_score = -np.inf
        self.trial_results = []
    
    def __call__(self, trial) -> float:
        """Objective function for Optuna optimization."""
        self.trial_count += 1
        start_time = time.time()
        
        try:
            # Get hyperparameters from trial
            hyperparams = self._suggest_hyperparameters(trial)
            
            # Create trainer
            trainer = self.trainer_class(random_state=self.random_state, verbose=False)
            
            # Perform cross-validation
            model = trainer._create_model(**hyperparams)
            
            # Use stratified K-fold
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=self.scoring)
            
            score = scores.mean()
            std = scores.std()
            
            # Store trial results
            trial_time = time.time() - start_time
            self.trial_results.append({
                'trial': self.trial_count,
                'params': hyperparams,
                'score': score,
                'std': std,
                'time': trial_time
            })
            
            # Update best score
            if score > self.best_score:
                self.best_score = score
                logger.info(f"Trial {self.trial_count}: New best {self.scoring} = {score:.4f} (±{std:.4f})")
            
            return score
            
        except Exception as e:
            logger.warning(f"Trial {self.trial_count} failed: {e}")
            return -np.inf
    
    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters based on model type."""
        # This will be overridden by specific model objectives
        return {}


class RandomForestObjective(OptimizationObjective):
    """Optuna objective for Random Forest hyperparameter tuning."""
    
    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest Random Forest hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 30) if trial.suggest_categorical('max_depth_none', [True, False]) else None,
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1, step=0.01),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1, step=0.01),
            'random_state': self.random_state,
            'n_jobs': -1,
            'oob_score': True
        }


class XGBoostObjective(OptimizationObjective):
    """Optuna objective for XGBoost hyperparameter tuning."""
    
    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest XGBoost hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0, step=0.1),
            'gamma': trial.suggest_float('gamma', 0, 5, step=0.1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 100, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 100, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10, step=0.5),
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
            'verbosity': 0,
            'eval_metric': 'auc'
        }


class NeuralNetworkObjective(OptimizationObjective):
    """Optuna objective for Neural Network hyperparameter tuning."""
    
    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest Neural Network hyperparameters."""
        # Suggest architecture
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_sizes = []
        
        for i in range(n_layers):
            size = trial.suggest_int(f'n_units_l{i}', 32, 512, log=True)
            hidden_sizes.append(size)
        
        return {
            'hidden_layer_sizes': tuple(hidden_sizes),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
            'solver': trial.suggest_categorical('solver', ['adam', 'sgd', 'lbfgs']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
            'momentum': trial.suggest_float('momentum', 0.8, 0.99, step=0.01),
            'beta_1': trial.suggest_float('beta_1', 0.85, 0.95, step=0.01),
            'beta_2': trial.suggest_float('beta_2', 0.99, 0.999, step=0.001),
            'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
            'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.2, step=0.05),
            'max_iter': 500,  # Reduced for faster optimization
            'random_state': self.random_state,
            'verbose': False
        }


class HyperparameterTuner:
    """
    Advanced hyperparameter tuning framework using Optuna.
    
    Provides Bayesian optimization, early stopping, pruning, and 
    multi-objective optimization for all model types.
    """
    
    def __init__(self, 
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None,
                 sampler: str = 'tpe',
                 pruner: str = 'median',
                 direction: str = 'maximize',
                 random_state: int = 42):
        """
        Initialize hyperparameter tuner.
        
        Args:
            study_name: Name of the optimization study
            storage: Storage backend for study persistence
            sampler: Sampling algorithm ('tpe', 'random', 'cmaes')
            pruner: Pruning algorithm ('median', 'hyperband', 'successive_halving')
            direction: Optimization direction ('maximize' or 'minimize')
            random_state: Random state for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna>=3.0.0")
        
        self.study_name = study_name or f"loan_eligibility_study_{int(time.time())}"
        self.storage = storage
        self.direction = direction
        self.random_state = random_state
        
        # Create sampler
        self.sampler = self._create_sampler(sampler)
        
        # Create pruner
        self.pruner = self._create_pruner(pruner)
        
        # Study and results
        self.study = None
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
    
    def _create_sampler(self, sampler_name: str):
        """Create Optuna sampler."""
        samplers = {
            'tpe': TPESampler(seed=self.random_state),
            'random': RandomSampler(seed=self.random_state),
            'cmaes': CmaEsSampler(seed=self.random_state)
        }
        
        if sampler_name not in samplers:
            logger.warning(f"Unknown sampler '{sampler_name}', using TPE")
            sampler_name = 'tpe'
        
        return samplers[sampler_name]
    
    def _create_pruner(self, pruner_name: str):
        """Create Optuna pruner."""
        pruners = {
            'median': MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            'hyperband': HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3),
            'successive_halving': SuccessiveHalvingPruner(min_resource=1, reduction_factor=4)
        }
        
        if pruner_name not in pruners:
            logger.warning(f"Unknown pruner '{pruner_name}', using median")
            pruner_name = 'median'
        
        return pruners[pruner_name]
    
    def optimize_model(self,
                      trainer_class,
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      n_trials: int = 100,
                      timeout: Optional[int] = None,
                      cv_folds: int = 5,
                      scoring: str = 'roc_auc',
                      test_size: float = 0.2,
                      callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model.
        
        Args:
            trainer_class: Model trainer class to optimize
            X: Training features
            y: Training targets
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            cv_folds: Cross-validation folds
            scoring: Scoring metric
            test_size: Test set size
            callbacks: Optional callbacks for optimization
            
        Returns:
            Optimization results
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info(f"Starting hyperparameter optimization for {trainer_class.__name__}")
        logger.info(f"Trials: {n_trials}, CV folds: {cv_folds}, Scoring: {scoring}")
        
        # Create study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            direction=self.direction,
            load_if_exists=True
        )
        
        # Create objective function
        objective = self._create_objective(
            trainer_class, X, y, cv_folds, scoring, test_size
        )
        
        # Setup callbacks
        if callbacks is None:
            callbacks = []
        
        # Add timeout callback if specified
        if timeout:
            callbacks.append(MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,)))
        
        # Run optimization
        start_time = time.time()
        
        try:
            self.study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=callbacks,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        optimization_time = time.time() - start_time
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # Create optimization results
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_trial': self.study.best_trial,
            'n_trials': len(self.study.trials),
            'optimization_time': optimization_time,
            'study_statistics': {
                'completed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'failed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
                'pruned_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            }
        }
        
        logger.info(f"Optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return results
    
    def _create_objective(self, trainer_class, X, y, cv_folds, scoring, test_size):
        """Create appropriate objective function for the trainer class."""
        if 'RandomForest' in trainer_class.__name__:
            return RandomForestObjective(trainer_class, X, y, cv_folds, scoring, test_size, self.random_state)
        elif 'XGBoost' in trainer_class.__name__:
            return XGBoostObjective(trainer_class, X, y, cv_folds, scoring, test_size, self.random_state)
        elif 'NeuralNetwork' in trainer_class.__name__:
            return NeuralNetworkObjective(trainer_class, X, y, cv_folds, scoring, test_size, self.random_state)
        else:
            raise ValueError(f"Unsupported trainer class: {trainer_class.__name__}")
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            return pd.DataFrame()
        
        trials_df = self.study.trials_dataframe()
        
        # Add additional information
        if not trials_df.empty:
            trials_df['trial_number'] = range(len(trials_df))
            trials_df['cumulative_best'] = trials_df['value'].cummax() if self.direction == 'maximize' else trials_df['value'].cummin()
        
        return trials_df
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if self.study is None:
            logger.warning("No study available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Optimization history
            plot_optimization_history(self.study, ax=axes[0, 0])
            axes[0, 0].set_title('Optimization History')
            
            # Parameter importances
            try:
                plot_param_importances(self.study, ax=axes[0, 1])
                axes[0, 1].set_title('Parameter Importances')
            except:
                axes[0, 1].text(0.5, 0.5, 'Parameter importances\nnot available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # Score distribution
            scores = [trial.value for trial in self.study.trials if trial.value is not None]
            axes[1, 0].hist(scores, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(self.best_score, color='red', linestyle='--', label=f'Best: {self.best_score:.4f}')
            axes[1, 0].set_xlabel('Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Score Distribution')
            axes[1, 0].legend()
            
            # Trial states
            states = [trial.state.name for trial in self.study.trials]
            state_counts = pd.Series(states).value_counts()
            axes[1, 1].pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Trial States')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization plots saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available for plotting optimization history")
    
    def save_study(self, filepath: str):
        """Save optimization study."""
        if self.study is None:
            logger.warning("No study to save")
            return
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save study data
        study_data = {
            'study_name': self.study_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'trials': len(self.study.trials),
            'trials_dataframe': self.get_optimization_history().to_dict('records'),
            'sampler': str(self.sampler),
            'pruner': str(self.pruner),
            'direction': self.direction
        }
        
        with open(filepath, 'w') as f:
            json.dump(study_data, f, indent=2, default=str)
        
        logger.info(f"Study saved to {filepath}")
    
    def load_study(self, filepath: str):
        """Load optimization study."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Study file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            study_data = json.load(f)
        
        self.study_name = study_data['study_name']
        self.best_params = study_data['best_params']
        self.best_score = study_data['best_score']
        
        logger.info(f"Study loaded from {filepath}")
    
    def suggest_next_trials(self, n_suggestions: int = 5) -> List[Dict[str, Any]]:
        """Suggest next hyperparameter combinations to try."""
        if self.study is None:
            logger.warning("No study available for suggestions")
            return []
        
        suggestions = []
        
        # Create temporary trials to get suggestions
        for _ in range(n_suggestions):
            trial = self.study.ask()
            suggestions.append(dict(trial.params))
            # Don't tell the study about these trials
        
        return suggestions
    
    def compare_models(self,
                      trainer_classes: List[type],
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      n_trials_per_model: int = 50,
                      **kwargs) -> pd.DataFrame:
        """
        Compare multiple models with hyperparameter optimization.
        
        Args:
            trainer_classes: List of trainer classes to compare
            X: Training features
            y: Training targets
            n_trials_per_model: Number of trials per model
            **kwargs: Additional optimization arguments
            
        Returns:
            Comparison results DataFrame
        """
        results = []
        
        for trainer_class in trainer_classes:
            logger.info(f"Optimizing {trainer_class.__name__}...")
            
            # Create new tuner for each model
            tuner = HyperparameterTuner(
                study_name=f"{trainer_class.__name__}_{int(time.time())}",
                sampler='tpe',
                pruner='median',
                random_state=self.random_state
            )
            
            # Optimize model
            opt_results = tuner.optimize_model(
                trainer_class, X, y, 
                n_trials=n_trials_per_model,
                **kwargs
            )
            
            # Store results
            results.append({
                'model': trainer_class.__name__,
                'best_score': opt_results['best_score'],
                'best_params': opt_results['best_params'],
                'optimization_time': opt_results['optimization_time'],
                'completed_trials': opt_results['study_statistics']['completed_trials'],
                'failed_trials': opt_results['study_statistics']['failed_trials'],
                'pruned_trials': opt_results['study_statistics']['pruned_trials']
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('best_score', ascending=False)
        
        return comparison_df


# Convenience functions
def tune_random_forest(X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      n_trials: int = 100,
                      **kwargs):
    """Quick Random Forest hyperparameter tuning."""
    from .random_forest_model import RandomForestTrainer
    
    tuner = HyperparameterTuner(study_name="RandomForest_optimization")
    return tuner.optimize_model(RandomForestTrainer, X, y, n_trials=n_trials, **kwargs)


def tune_xgboost(X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray],
                n_trials: int = 100,
                **kwargs):
    """Quick XGBoost hyperparameter tuning."""
    from .xgboost_model import XGBoostTrainer
    
    tuner = HyperparameterTuner(study_name="XGBoost_optimization")
    return tuner.optimize_model(XGBoostTrainer, X, y, n_trials=n_trials, **kwargs)


def tune_neural_network(X: Union[pd.DataFrame, np.ndarray],
                       y: Union[pd.Series, np.ndarray],
                       n_trials: int = 50,
                       **kwargs):
    """Quick Neural Network hyperparameter tuning."""
    from .neural_network_model import NeuralNetworkTrainer
    
    tuner = HyperparameterTuner(study_name="NeuralNetwork_optimization")
    return tuner.optimize_model(NeuralNetworkTrainer, X, y, n_trials=n_trials, **kwargs)


def compare_all_models(X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      n_trials_per_model: int = 50,
                      **kwargs):
    """Compare all available models with hyperparameter optimization."""
    from .random_forest_model import RandomForestTrainer
    from .xgboost_model import XGBoostTrainer
    from .neural_network_model import NeuralNetworkTrainer
    
    tuner = HyperparameterTuner(study_name="model_comparison")
    return tuner.compare_models(
        [RandomForestTrainer, XGBoostTrainer, NeuralNetworkTrainer],
        X, y, n_trials_per_model=n_trials_per_model, **kwargs
    )