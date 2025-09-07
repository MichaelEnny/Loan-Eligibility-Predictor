"""
Advanced Cross-Validation Framework
Provides comprehensive cross-validation strategies with stratification,
time series splits, and custom validation schemes for robust model evaluation.
"""

from typing import Dict, Any, Optional, Union, List, Tuple, Generator
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from sklearn.model_selection import (
    StratifiedKFold, KFold, TimeSeriesSplit, GroupKFold,
    LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedShuffleSplit,
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    mean_squared_error, mean_absolute_error
)
import warnings

logger = logging.getLogger(__name__)


class CrossValidationResults:
    """Container for cross-validation results with comprehensive metrics."""
    
    def __init__(self, 
                 scores: Dict[str, np.ndarray],
                 fold_results: List[Dict[str, Any]],
                 cv_strategy: str,
                 n_splits: int):
        """
        Initialize CV results.
        
        Args:
            scores: Dictionary of metric scores across folds
            fold_results: Detailed results for each fold
            cv_strategy: Cross-validation strategy used
            n_splits: Number of splits
        """
        self.scores = scores
        self.fold_results = fold_results
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        
        # Calculate summary statistics
        self.mean_scores = {metric: np.mean(values) for metric, values in scores.items()}
        self.std_scores = {metric: np.std(values) for metric, values in scores.items()}
        self.min_scores = {metric: np.min(values) for metric, values in scores.items()}
        self.max_scores = {metric: np.max(values) for metric, values in scores.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of CV results."""
        return {
            'cv_strategy': self.cv_strategy,
            'n_splits': self.n_splits,
            'mean_scores': self.mean_scores,
            'std_scores': self.std_scores,
            'min_scores': self.min_scores,
            'max_scores': self.max_scores,
            'score_ranges': {
                metric: (self.min_scores[metric], self.max_scores[metric])
                for metric in self.mean_scores.keys()
            }
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for i, fold_result in enumerate(self.fold_results):
            row = {'fold': i}
            row.update(fold_result.get('scores', {}))
            row.update({
                'train_size': fold_result.get('train_size', 0),
                'test_size': fold_result.get('test_size', 0),
                'fit_time': fold_result.get('fit_time', 0),
                'score_time': fold_result.get('score_time', 0)
            })
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_scores(self, save_path: Optional[str] = None):
        """Plot cross-validation scores."""
        try:
            import matplotlib.pyplot as plt
            
            metrics = list(self.mean_scores.keys())
            n_metrics = len(metrics)
            
            if n_metrics == 0:
                logger.warning("No metrics to plot")
                return
            
            fig, axes = plt.subplots(1, min(n_metrics, 4), figsize=(4 * min(n_metrics, 4), 5))
            if n_metrics == 1:
                axes = [axes]
            elif n_metrics > 4:
                axes = axes.flatten()[:4]
                metrics = metrics[:4]
            
            for i, metric in enumerate(metrics):
                scores = self.scores[metric]
                mean_score = self.mean_scores[metric]
                std_score = self.std_scores[metric]
                
                # Box plot for fold scores
                axes[i].boxplot(scores, labels=[metric])
                axes[i].axhline(y=mean_score, color='red', linestyle='--', alpha=0.8, 
                               label=f'Mean: {mean_score:.3f}')
                
                axes[i].set_title(f'{metric} (CV={self.n_splits})')
                axes[i].set_ylabel('Score')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"CV plots saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available for plotting CV results")


class BaseCrossValidator(ABC):
    """Base class for cross-validation strategies."""
    
    def __init__(self, 
                 n_splits: int = 5,
                 random_state: int = 42,
                 scoring: Union[str, List[str]] = 'accuracy'):
        """
        Initialize base cross-validator.
        
        Args:
            n_splits: Number of cross-validation splits
            random_state: Random state for reproducibility
            scoring: Scoring metrics to compute
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.scoring = scoring if isinstance(scoring, list) else [scoring]
    
    @abstractmethod
    def _create_cv_splitter(self, X, y, groups=None):
        """Create sklearn cross-validation splitter."""
        pass
    
    def validate(self, 
                model,
                X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray],
                groups: Optional[np.ndarray] = None,
                return_estimator: bool = False,
                verbose: bool = True) -> CrossValidationResults:
        """
        Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            groups: Group labels for GroupKFold
            return_estimator: Whether to return trained estimators
            verbose: Whether to print progress
            
        Returns:
            CrossValidationResults object
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Create cross-validation splitter
        cv_splitter = self._create_cv_splitter(X, y, groups)
        
        if verbose:
            logger.info(f"Starting {self.__class__.__name__} with {self.n_splits} splits")
        
        # Perform cross-validation with detailed results
        cv_results = cross_validate(
            model, X, y,
            cv=cv_splitter,
            scoring=self.scoring,
            return_train_score=True,
            return_estimator=return_estimator,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        # Process results
        scores = {}
        fold_results = []
        
        for metric in self.scoring:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            
            if test_key in cv_results:
                scores[f'{metric}_test'] = cv_results[test_key]
            if train_key in cv_results:
                scores[f'{metric}_train'] = cv_results[train_key]
        
        # Create fold-level results
        for i in range(len(cv_results['fit_time'])):
            fold_result = {
                'fold': i,
                'fit_time': cv_results['fit_time'][i],
                'score_time': cv_results['score_time'][i],
                'scores': {}
            }
            
            for metric in self.scoring:
                test_key = f'test_{metric}'
                train_key = f'train_{metric}'
                
                if test_key in cv_results:
                    fold_result['scores'][f'{metric}_test'] = cv_results[test_key][i]
                if train_key in cv_results:
                    fold_result['scores'][f'{metric}_train'] = cv_results[train_key][i]
            
            fold_results.append(fold_result)
        
        results = CrossValidationResults(
            scores=scores,
            fold_results=fold_results,
            cv_strategy=self.__class__.__name__,
            n_splits=self.n_splits
        )
        
        if verbose:
            self._print_results_summary(results)
        
        return results
    
    def _print_results_summary(self, results: CrossValidationResults):
        """Print results summary."""
        logger.info(f"Cross-validation completed ({results.cv_strategy})")
        logger.info(f"Splits: {results.n_splits}")
        
        for metric, mean_score in results.mean_scores.items():
            std_score = results.std_scores[metric]
            logger.info(f"{metric}: {mean_score:.4f} ± {std_score:.4f}")


class StratifiedCrossValidator(BaseCrossValidator):
    """Stratified K-Fold cross-validation for classification."""
    
    def __init__(self, 
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: int = 42,
                 scoring: Union[str, List[str]] = 'accuracy'):
        """
        Initialize stratified cross-validator.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random state for reproducibility
            scoring: Scoring metrics
        """
        super().__init__(n_splits, random_state, scoring)
        self.shuffle = shuffle
    
    def _create_cv_splitter(self, X, y, groups=None):
        """Create stratified K-fold splitter."""
        return StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )


class TimeSeriesCrossValidator(BaseCrossValidator):
    """Time series cross-validation with forward chaining."""
    
    def __init__(self, 
                 n_splits: int = 5,
                 max_train_size: Optional[int] = None,
                 test_size: Optional[int] = None,
                 gap: int = 0,
                 scoring: Union[str, List[str]] = 'neg_mean_squared_error'):
        """
        Initialize time series cross-validator.
        
        Args:
            n_splits: Number of splits
            max_train_size: Maximum training set size
            test_size: Test set size for each split
            gap: Gap between train and test sets
            scoring: Scoring metrics
        """
        super().__init__(n_splits, random_state=42, scoring=scoring)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
    
    def _create_cv_splitter(self, X, y, groups=None):
        """Create time series splitter."""
        return TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            gap=self.gap
        )


class GroupCrossValidator(BaseCrossValidator):
    """Group-based cross-validation."""
    
    def __init__(self, 
                 n_splits: int = 5,
                 random_state: int = 42,
                 scoring: Union[str, List[str]] = 'accuracy'):
        """
        Initialize group cross-validator.
        
        Args:
            n_splits: Number of splits
            random_state: Random state
            scoring: Scoring metrics
        """
        super().__init__(n_splits, random_state, scoring)
    
    def _create_cv_splitter(self, X, y, groups=None):
        """Create group K-fold splitter."""
        if groups is None:
            raise ValueError("Groups must be provided for GroupKFold")
        return GroupKFold(n_splits=self.n_splits)
    
    def validate(self, model, X, y, groups, **kwargs):
        """Validate with groups."""
        if groups is None:
            raise ValueError("Groups must be provided for group cross-validation")
        return super().validate(model, X, y, groups=groups, **kwargs)


class BootstrapCrossValidator(BaseCrossValidator):
    """Bootstrap cross-validation using ShuffleSplit."""
    
    def __init__(self, 
                 n_splits: int = 10,
                 test_size: float = 0.2,
                 train_size: Optional[float] = None,
                 random_state: int = 42,
                 scoring: Union[str, List[str]] = 'accuracy'):
        """
        Initialize bootstrap cross-validator.
        
        Args:
            n_splits: Number of bootstrap samples
            test_size: Proportion of test set
            train_size: Proportion of training set
            random_state: Random state
            scoring: Scoring metrics
        """
        super().__init__(n_splits, random_state, scoring)
        self.test_size = test_size
        self.train_size = train_size
    
    def _create_cv_splitter(self, X, y, groups=None):
        """Create shuffle split splitter."""
        # Use stratified shuffle split for classification
        if len(np.unique(y)) < 10:  # Likely classification
            return StratifiedShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                train_size=self.train_size,
                random_state=self.random_state
            )
        else:
            return ShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                train_size=self.train_size,
                random_state=self.random_state
            )


class LoanEligibilityCrossValidator(StratifiedCrossValidator):
    """Specialized cross-validator for loan eligibility models."""
    
    def __init__(self, 
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: int = 42):
        """Initialize loan eligibility cross-validator."""
        # Use loan-specific scoring metrics
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        super().__init__(n_splits, shuffle, random_state, scoring)
    
    def validate_with_business_metrics(self,
                                     model,
                                     X: Union[pd.DataFrame, np.ndarray],
                                     y: Union[pd.Series, np.ndarray],
                                     loan_amounts: Optional[np.ndarray] = None,
                                     **kwargs) -> Tuple[CrossValidationResults, Dict[str, float]]:
        """
        Validate with additional business metrics for loan models.
        
        Args:
            model: Model to validate
            X: Features
            y: Targets (1 for approved, 0 for denied)
            loan_amounts: Loan amounts for business metric calculation
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (CV results, business metrics)
        """
        # Perform standard CV
        cv_results = self.validate(model, X, y, **kwargs)
        
        # Calculate business metrics if loan amounts provided
        business_metrics = {}
        if loan_amounts is not None:
            business_metrics = self._calculate_business_metrics(
                model, X, y, loan_amounts
            )
        
        return cv_results, business_metrics
    
    def _calculate_business_metrics(self, 
                                  model, 
                                  X: np.ndarray, 
                                  y: np.ndarray,
                                  loan_amounts: np.ndarray) -> Dict[str, float]:
        """Calculate business-specific metrics."""
        # This is a simplified business metrics calculation
        # In practice, you would use actual business rules
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else predictions
        
        # Calculate potential revenue impact
        approved_mask = predictions == 1
        actual_approved_mask = y == 1
        
        # True positives (correctly approved loans)
        tp_mask = approved_mask & actual_approved_mask
        tp_revenue = np.sum(loan_amounts[tp_mask]) * 0.05  # Assume 5% interest margin
        
        # False positives (incorrectly approved loans - potential losses)
        fp_mask = approved_mask & ~actual_approved_mask
        fp_losses = np.sum(loan_amounts[fp_mask]) * 0.3  # Assume 30% default loss
        
        # False negatives (missed opportunities)
        fn_mask = ~approved_mask & actual_approved_mask
        fn_opportunity_cost = np.sum(loan_amounts[fn_mask]) * 0.05
        
        total_portfolio_value = np.sum(loan_amounts[actual_approved_mask])
        
        business_metrics = {
            'potential_revenue': tp_revenue,
            'potential_losses': fp_losses,
            'opportunity_cost': fn_opportunity_cost,
            'net_business_impact': tp_revenue - fp_losses,
            'revenue_efficiency': tp_revenue / (total_portfolio_value * 0.05) if total_portfolio_value > 0 else 0,
            'risk_adjusted_return': (tp_revenue - fp_losses) / (tp_revenue + fp_losses) if (tp_revenue + fp_losses) > 0 else 0
        }
        
        return business_metrics


class CrossValidator:
    """
    Main cross-validation orchestrator with multiple strategies.
    
    Provides unified interface for different cross-validation approaches
    with comprehensive reporting and comparison capabilities.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize cross-validator."""
        self.random_state = random_state
        
        # Available validators
        self.validators = {
            'stratified': StratifiedCrossValidator,
            'time_series': TimeSeriesCrossValidator,
            'group': GroupCrossValidator,
            'bootstrap': BootstrapCrossValidator,
            'loan_eligibility': LoanEligibilityCrossValidator
        }
    
    def validate(self,
                model,
                X: Union[pd.DataFrame, np.ndarray],
                y: Union[pd.Series, np.ndarray],
                cv_strategy: str = 'stratified',
                n_splits: int = 5,
                groups: Optional[np.ndarray] = None,
                **kwargs) -> CrossValidationResults:
        """
        Perform cross-validation with specified strategy.
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            cv_strategy: Cross-validation strategy
            n_splits: Number of splits
            groups: Group labels (for group CV)
            **kwargs: Additional validator arguments
            
        Returns:
            CrossValidationResults object
        """
        if cv_strategy not in self.validators:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}. "
                           f"Available: {list(self.validators.keys())}")
        
        # Create validator
        validator_class = self.validators[cv_strategy]
        validator = validator_class(
            n_splits=n_splits,
            random_state=self.random_state,
            **kwargs
        )
        
        # Perform validation
        if cv_strategy == 'group':
            return validator.validate(model, X, y, groups=groups)
        else:
            return validator.validate(model, X, y)
    
    def compare_strategies(self,
                          model,
                          X: Union[pd.DataFrame, np.ndarray],
                          y: Union[pd.Series, np.ndarray],
                          strategies: List[str] = None,
                          n_splits: int = 5,
                          **kwargs) -> Dict[str, CrossValidationResults]:
        """
        Compare multiple CV strategies.
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            strategies: List of CV strategies to compare
            n_splits: Number of splits
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of results by strategy
        """
        if strategies is None:
            strategies = ['stratified', 'bootstrap']
        
        results = {}
        
        for strategy in strategies:
            try:
                logger.info(f"Running {strategy} cross-validation...")
                results[strategy] = self.validate(
                    model, X, y, cv_strategy=strategy, n_splits=n_splits, **kwargs
                )
            except Exception as e:
                logger.warning(f"Failed to run {strategy} CV: {e}")
                continue
        
        return results
    
    def create_validation_report(self, 
                               results: Dict[str, CrossValidationResults],
                               save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create comprehensive validation report.
        
        Args:
            results: Dictionary of CV results by strategy
            save_path: Path to save report
            
        Returns:
            Summary DataFrame
        """
        report_data = []
        
        for strategy, result in results.items():
            summary = result.get_summary()
            
            for metric, mean_score in summary['mean_scores'].items():
                std_score = summary['std_scores'][metric]
                min_score = summary['min_scores'][metric]
                max_score = summary['max_scores'][metric]
                
                report_data.append({
                    'cv_strategy': strategy,
                    'metric': metric,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'min_score': min_score,
                    'max_score': max_score,
                    'n_splits': summary['n_splits'],
                    'score_range': max_score - min_score
                })
        
        report_df = pd.DataFrame(report_data)
        
        if save_path:
            report_df.to_csv(save_path, index=False)
            logger.info(f"Validation report saved to {save_path}")
        
        return report_df


# Convenience functions
def quick_cv_score(model,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, np.ndarray],
                  cv: int = 5,
                  scoring: str = 'accuracy',
                  stratify: bool = True) -> Tuple[float, float]:
    """
    Quick cross-validation score.
    
    Returns:
        Tuple of (mean_score, std_score)
    """
    validator = StratifiedCrossValidator(n_splits=cv, scoring=scoring) if stratify else BaseCrossValidator(n_splits=cv, scoring=scoring)
    results = validator.validate(model, X, y, verbose=False)
    
    metric_key = f'{scoring}_test' if f'{scoring}_test' in results.mean_scores else list(results.mean_scores.keys())[0]
    return results.mean_scores[metric_key], results.std_scores[metric_key]


def compare_models_cv(models: Dict[str, Any],
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray],
                     cv: int = 5,
                     scoring: str = 'accuracy') -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.
    
    Args:
        models: Dictionary of model name -> model instance
        X: Features
        y: Targets
        cv: Number of CV folds
        scoring: Scoring metric
        
    Returns:
        Comparison DataFrame
    """
    validator = CrossValidator()
    comparison_data = []
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        try:
            results = validator.validate(
                model, X, y, cv_strategy='stratified', n_splits=cv, 
                scoring=scoring, verbose=False
            )
            
            metric_key = f'{scoring}_test'
            if metric_key in results.mean_scores:
                mean_score = results.mean_scores[metric_key]
                std_score = results.std_scores[metric_key]
            else:
                # Fallback to first available metric
                first_metric = list(results.mean_scores.keys())[0]
                mean_score = results.mean_scores[first_metric]
                std_score = results.std_scores[first_metric]
            
            comparison_data.append({
                'model': model_name,
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': min(results.scores[list(results.scores.keys())[0]]),
                'max_score': max(results.scores[list(results.scores.keys())[0]])
            })
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {model_name}: {e}")
            continue
    
    comparison_df = pd.DataFrame(comparison_data)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values('mean_score', ascending=False)
    
    return comparison_df