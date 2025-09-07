"""
Comprehensive Model Evaluation Framework
Provides detailed evaluation metrics, visualization, and business impact analysis
for loan eligibility prediction models with fairness and bias assessment.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, log_loss, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
import warnings

logger = logging.getLogger(__name__)


class ModelEvaluationMetrics:
    """Container for comprehensive model evaluation metrics."""
    
    def __init__(self):
        self.classification_metrics = {}
        self.probability_metrics = {}
        self.business_metrics = {}
        self.fairness_metrics = {}
        self.stability_metrics = {}
        self.calibration_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'classification_metrics': self.classification_metrics,
            'probability_metrics': self.probability_metrics,
            'business_metrics': self.business_metrics,
            'fairness_metrics': self.fairness_metrics,
            'stability_metrics': self.stability_metrics,
            'calibration_metrics': self.calibration_metrics
        }
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of key metrics."""
        summary = {}
        
        # Key classification metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in key_metrics:
            if metric in self.classification_metrics:
                summary[metric] = self.classification_metrics[metric]
        
        # Key business metrics
        if 'net_business_value' in self.business_metrics:
            summary['business_value'] = self.business_metrics['net_business_value']
        
        # Key fairness metric
        if 'demographic_parity_diff' in self.fairness_metrics:
            summary['fairness_score'] = abs(self.fairness_metrics['demographic_parity_diff'])
        
        return summary


class BusinessImpactCalculator:
    """Calculate business impact metrics for loan eligibility models."""
    
    def __init__(self, 
                 default_rate: float = 0.1,
                 profit_margin: float = 0.05,
                 collection_rate: float = 0.3,
                 processing_cost: float = 100):
        """
        Initialize business impact calculator.
        
        Args:
            default_rate: Expected default rate for approved loans
            profit_margin: Profit margin on performing loans
            collection_rate: Recovery rate on defaulted loans
            processing_cost: Cost to process each application
        """
        self.default_rate = default_rate
        self.profit_margin = profit_margin
        self.collection_rate = collection_rate
        self.processing_cost = processing_cost
    
    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         loan_amounts: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate business impact metrics.
        
        Args:
            y_true: True labels (1 for should approve, 0 for should deny)
            y_pred: Predicted labels (1 for approve, 0 for deny)
            loan_amounts: Loan amounts (use median if not provided)
            
        Returns:
            Dictionary of business metrics
        """
        if loan_amounts is None:
            # Use median loan amount if not provided
            loan_amounts = np.full(len(y_true), 25000)  # Assumed median
        
        # Calculate confusion matrix components
        tp = np.sum((y_pred == 1) & (y_true == 1))  # Correctly approved
        fp = np.sum((y_pred == 1) & (y_true == 0))  # Incorrectly approved (will default)
        tn = np.sum((y_pred == 0) & (y_true == 0))  # Correctly denied
        fn = np.sum((y_pred == 0) & (y_true == 1))  # Incorrectly denied (missed opportunity)
        
        # Loan amounts for each category
        tp_amounts = loan_amounts[(y_pred == 1) & (y_true == 1)]
        fp_amounts = loan_amounts[(y_pred == 1) & (y_true == 0)]
        fn_amounts = loan_amounts[(y_pred == 0) & (y_true == 1)]
        
        # Revenue calculations
        tp_revenue = np.sum(tp_amounts) * self.profit_margin * (1 - self.default_rate)
        fp_losses = np.sum(fp_amounts) * (1 - self.collection_rate)  # Loss from defaults
        fn_opportunity_cost = np.sum(fn_amounts) * self.profit_margin * (1 - self.default_rate)
        
        # Processing costs
        total_processing_cost = len(y_pred) * self.processing_cost
        
        # Net business value
        net_business_value = tp_revenue - fp_losses - total_processing_cost
        
        # Additional metrics
        total_approved = tp + fp
        total_should_approve = tp + fn
        
        # Portfolio metrics
        approved_portfolio_value = np.sum(tp_amounts) + np.sum(fp_amounts)
        expected_portfolio_value = np.sum(tp_amounts) + np.sum(fn_amounts)
        
        return {
            'net_business_value': net_business_value,
            'revenue_from_approved': tp_revenue,
            'losses_from_defaults': fp_losses,
            'opportunity_cost': fn_opportunity_cost,
            'processing_costs': total_processing_cost,
            'approved_portfolio_value': approved_portfolio_value,
            'expected_portfolio_value': expected_portfolio_value,
            'portfolio_efficiency': approved_portfolio_value / expected_portfolio_value if expected_portfolio_value > 0 else 0,
            'average_loan_amount': np.mean(loan_amounts),
            'approved_applications': total_approved,
            'approval_rate': total_approved / len(y_pred) if len(y_pred) > 0 else 0,
            'precision_weighted_value': tp_revenue / (tp_revenue + fp_losses) if (tp_revenue + fp_losses) > 0 else 0
        }


class FairnessEvaluator:
    """Evaluate model fairness across demographic groups."""
    
    def __init__(self):
        self.protected_attributes = []
        self.fairness_metrics = {}
    
    def evaluate_fairness(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray] = None,
                         protected_attributes: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Evaluate fairness metrics across protected groups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            protected_attributes: Dictionary of protected attribute arrays
            
        Returns:
            Dictionary of fairness metrics
        """
        if protected_attributes is None:
            return {}
        
        fairness_metrics = {}
        
        for attr_name, attr_values in protected_attributes.items():
            # Get unique groups
            groups = np.unique(attr_values)
            
            if len(groups) < 2:
                continue
            
            # Calculate metrics for each group
            group_metrics = {}
            
            for group in groups:
                group_mask = attr_values == group
                if np.sum(group_mask) == 0:
                    continue
                
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                group_metrics[group] = {
                    'accuracy': accuracy_score(group_y_true, group_y_pred),
                    'precision': precision_score(group_y_true, group_y_pred, zero_division=0),
                    'recall': recall_score(group_y_true, group_y_pred, zero_division=0),
                    'approval_rate': np.mean(group_y_pred),
                    'base_rate': np.mean(group_y_true),
                    'sample_size': np.sum(group_mask)
                }
                
                if y_prob is not None:
                    group_y_prob = y_prob[group_mask]
                    try:
                        group_metrics[group]['auc'] = roc_auc_score(group_y_true, group_y_prob)
                    except:
                        group_metrics[group]['auc'] = 0.0
            
            # Calculate fairness metrics
            if len(group_metrics) >= 2:
                groups_list = list(group_metrics.keys())
                
                # Demographic parity (equal approval rates)
                approval_rates = [group_metrics[g]['approval_rate'] for g in groups_list]
                fairness_metrics[f'{attr_name}_demographic_parity_diff'] = max(approval_rates) - min(approval_rates)
                fairness_metrics[f'{attr_name}_demographic_parity_ratio'] = min(approval_rates) / max(approval_rates) if max(approval_rates) > 0 else 0
                
                # Equal opportunity (equal TPR)
                tprs = [group_metrics[g]['recall'] for g in groups_list]
                fairness_metrics[f'{attr_name}_equal_opportunity_diff'] = max(tprs) - min(tprs)
                
                # Equalized odds (equal TPR and FPR)
                # Note: This is simplified - full equalized odds requires FPR calculation
                
                # Accuracy parity
                accuracies = [group_metrics[g]['accuracy'] for g in groups_list]
                fairness_metrics[f'{attr_name}_accuracy_parity_diff'] = max(accuracies) - min(accuracies)
                
                # Store group metrics
                fairness_metrics[f'{attr_name}_group_metrics'] = group_metrics
        
        # Overall fairness score (lower is better)
        fairness_violations = []
        for metric, value in fairness_metrics.items():
            if 'diff' in metric and isinstance(value, (int, float)):
                fairness_violations.append(abs(value))
        
        if fairness_violations:
            fairness_metrics['overall_fairness_score'] = max(fairness_violations)
        
        return fairness_metrics


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    
    Provides detailed evaluation including classification metrics, business impact,
    fairness assessment, model stability, and calibration analysis.
    """
    
    def __init__(self,
                 business_calculator: Optional[BusinessImpactCalculator] = None,
                 fairness_evaluator: Optional[FairnessEvaluator] = None):
        """
        Initialize model evaluator.
        
        Args:
            business_calculator: Business impact calculator
            fairness_evaluator: Fairness evaluator
        """
        self.business_calculator = business_calculator or BusinessImpactCalculator()
        self.fairness_evaluator = fairness_evaluator or FairnessEvaluator()
        
        self.evaluation_history = []
    
    def evaluate_model(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_prob: Optional[np.ndarray] = None,
                      model_name: str = "model",
                      loan_amounts: Optional[np.ndarray] = None,
                      protected_attributes: Optional[Dict[str, np.ndarray]] = None,
                      sample_weights: Optional[np.ndarray] = None) -> ModelEvaluationMetrics:
        """
        Perform comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            model_name: Name of the model
            loan_amounts: Loan amounts for business metrics
            protected_attributes: Protected attributes for fairness evaluation
            sample_weights: Sample weights for weighted metrics
            
        Returns:
            ModelEvaluationMetrics object
        """
        logger.info(f"Evaluating model: {model_name}")
        
        metrics = ModelEvaluationMetrics()
        
        # 1. Classification Metrics
        metrics.classification_metrics = self._calculate_classification_metrics(
            y_true, y_pred, sample_weights
        )
        
        # 2. Probability Metrics
        if y_prob is not None:
            metrics.probability_metrics = self._calculate_probability_metrics(
                y_true, y_prob, sample_weights
            )
            
            # 3. Calibration Metrics
            metrics.calibration_metrics = self._calculate_calibration_metrics(
                y_true, y_prob
            )
        
        # 4. Business Metrics
        metrics.business_metrics = self.business_calculator.calculate_metrics(
            y_true, y_pred, loan_amounts
        )
        
        # 5. Fairness Metrics
        if protected_attributes:
            metrics.fairness_metrics = self.fairness_evaluator.evaluate_fairness(
                y_true, y_pred, y_prob, protected_attributes
            )
        
        # 6. Stability Metrics (requires historical data)
        # This would be populated when comparing against previous evaluations
        
        # Store in history
        evaluation_record = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.to_dict(),
            'sample_size': len(y_true),
            'positive_rate': np.mean(y_true)
        }
        self.evaluation_history.append(evaluation_record)
        
        logger.info(f"Evaluation completed for {model_name}")
        return metrics
    
    def _calculate_classification_metrics(self,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        metrics['precision'] = precision_score(y_true, y_pred, sample_weight=sample_weights, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, sample_weight=sample_weights, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, sample_weight=sample_weights, zero_division=0)
        
        # Additional metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        
        # Derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Positive and negative predictive values
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Likelihood ratios
        if metrics['false_positive_rate'] > 0:
            metrics['positive_likelihood_ratio'] = metrics['sensitivity'] / metrics['false_positive_rate']
        else:
            metrics['positive_likelihood_ratio'] = float('inf')
        
        if metrics['sensitivity'] < 1:
            metrics['negative_likelihood_ratio'] = (1 - metrics['sensitivity']) / metrics['specificity']
        else:
            metrics['negative_likelihood_ratio'] = 0
        
        return metrics
    
    def _calculate_probability_metrics(self,
                                     y_true: np.ndarray,
                                     y_prob: np.ndarray,
                                     sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate probability-based metrics."""
        metrics = {}
        
        # ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, sample_weight=sample_weights)
        except:
            metrics['roc_auc'] = 0.0
        
        # Precision-Recall AUC
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_prob, sample_weight=sample_weights)
        except:
            metrics['pr_auc'] = 0.0
        
        # Log Loss
        try:
            metrics['log_loss'] = log_loss(y_true, y_prob, sample_weight=sample_weights)
        except:
            metrics['log_loss'] = float('inf')
        
        # Brier Score
        metrics['brier_score'] = np.mean((y_prob - y_true) ** 2)
        
        return metrics
    
    def _calculate_calibration_metrics(self,
                                     y_true: np.ndarray,
                                     y_prob: np.ndarray,
                                     n_bins: int = 10) -> Dict[str, Any]:
        """Calculate calibration metrics."""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )
            
            # Calibration error
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Maximum calibration error
            max_calibration_error = np.max(np.abs(fraction_of_positives - mean_predicted_value))
            
            return {
                'calibration_error': calibration_error,
                'max_calibration_error': max_calibration_error,
                'calibration_curve_data': {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                }
            }
        except Exception as e:
            logger.warning(f"Could not calculate calibration metrics: {e}")
            return {}
    
    def compare_models(self,
                      evaluations: Dict[str, ModelEvaluationMetrics],
                      key_metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple model evaluations.
        
        Args:
            evaluations: Dictionary of model name -> ModelEvaluationMetrics
            key_metrics: List of key metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        if key_metrics is None:
            key_metrics = [
                'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
                'net_business_value', 'overall_fairness_score'
            ]
        
        comparison_data = []
        
        for model_name, metrics in evaluations.items():
            row = {'model': model_name}
            
            # Extract metrics from different categories
            all_metrics = {
                **metrics.classification_metrics,
                **metrics.probability_metrics,
                **metrics.business_metrics,
                **metrics.fairness_metrics
            }
            
            for metric in key_metrics:
                row[metric] = all_metrics.get(metric, np.nan)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by a composite score (if possible)
        if 'f1_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        return comparison_df
    
    def generate_evaluation_report(self,
                                  metrics: ModelEvaluationMetrics,
                                  model_name: str,
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: Model evaluation metrics
            model_name: Name of the model
            save_path: Path to save report
            
        Returns:
            Report dictionary
        """
        report = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'summary': metrics.get_summary(),
            'detailed_metrics': metrics.to_dict(),
            'recommendations': self._generate_recommendations(metrics)
        }
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def _generate_recommendations(self, metrics: ModelEvaluationMetrics) -> List[str]:
        """Generate recommendations based on evaluation metrics."""
        recommendations = []
        
        # Classification performance recommendations
        if metrics.classification_metrics.get('accuracy', 0) < 0.85:
            recommendations.append("Model accuracy is below 85%. Consider feature engineering or hyperparameter tuning.")
        
        if metrics.classification_metrics.get('f1_score', 0) < 0.8:
            recommendations.append("F1-score is below 0.8. Review precision-recall trade-off.")
        
        # Business impact recommendations
        if metrics.business_metrics.get('net_business_value', 0) < 0:
            recommendations.append("Model has negative business value. Review approval thresholds.")
        
        # Fairness recommendations
        fairness_score = metrics.fairness_metrics.get('overall_fairness_score', 0)
        if fairness_score > 0.1:
            recommendations.append("Model shows potential fairness issues. Consider bias mitigation techniques.")
        
        # Calibration recommendations
        calibration_error = metrics.calibration_metrics.get('calibration_error', 0)
        if calibration_error > 0.1:
            recommendations.append("Model probabilities are poorly calibrated. Consider calibration techniques.")
        
        if not recommendations:
            recommendations.append("Model performance looks good across all evaluated metrics.")
        
        return recommendations
    
    def plot_evaluation_results(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None):
        """Plot comprehensive evaluation results."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[0, 0].figure.colorbar(im, ax=axes[0, 0])
            axes[0, 0].set(xticks=np.arange(cm.shape[1]),
                          yticks=np.arange(cm.shape[0]),
                          xticklabels=['Deny', 'Approve'],
                          yticklabels=['Deny', 'Approve'],
                          title='Confusion Matrix',
                          ylabel='True Label',
                          xlabel='Predicted Label')
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > thresh else "black")
            
            if y_prob is not None:
                # ROC Curve
                RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0, 1])
                axes[0, 1].set_title('ROC Curve')
                
                # Precision-Recall Curve
                PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[1, 0])
                axes[1, 0].set_title('Precision-Recall Curve')
                
                # Calibration Plot
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_prob, n_bins=10
                )
                axes[1, 1].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
                axes[1, 1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                axes[1, 1].set_xlabel('Mean Predicted Probability')
                axes[1, 1].set_ylabel('Fraction of Positives')
                axes[1, 1].set_title('Calibration Plot')
                axes[1, 1].legend()
            else:
                # If no probabilities, show class distribution
                axes[1, 0].bar(['Actual Deny', 'Actual Approve'], 
                              [np.sum(y_true == 0), np.sum(y_true == 1)],
                              alpha=0.7, label='Actual')
                axes[1, 0].bar(['Predicted Deny', 'Predicted Approve'], 
                              [np.sum(y_pred == 0), np.sum(y_pred == 1)],
                              alpha=0.7, label='Predicted')
                axes[1, 0].set_title('Class Distribution')
                axes[1, 0].legend()
                
                axes[1, 1].text(0.5, 0.5, 'Probability plots\nnot available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Evaluation plots saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available for plotting evaluation results")
    
    def get_evaluation_history(self) -> pd.DataFrame:
        """Get evaluation history as DataFrame."""
        if not self.evaluation_history:
            return pd.DataFrame()
        
        # Flatten history for DataFrame
        flattened_data = []
        
        for record in self.evaluation_history:
            row = {
                'model_name': record['model_name'],
                'timestamp': record['timestamp'],
                'sample_size': record['sample_size'],
                'positive_rate': record['positive_rate']
            }
            
            # Add key metrics
            metrics = record['metrics']
            for category, category_metrics in metrics.items():
                if isinstance(category_metrics, dict):
                    for metric_name, metric_value in category_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            row[f'{category}_{metric_name}'] = metric_value
            
            flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)


# Convenience functions
def quick_evaluate(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_prob: Optional[np.ndarray] = None,
                  model_name: str = "model") -> Dict[str, float]:
    """Quick model evaluation with key metrics."""
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(y_true, y_pred, y_prob, model_name)
    return metrics.get_summary()


def compare_model_performance(models_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """
    Compare performance of multiple models.
    
    Args:
        models_results: Dict of model_name -> (y_true, y_pred, y_prob)
        
    Returns:
        Comparison DataFrame
    """
    evaluator = ModelEvaluator()
    evaluations = {}
    
    for model_name, (y_true, y_pred, y_prob) in models_results.items():
        evaluations[model_name] = evaluator.evaluate_model(
            y_true, y_pred, y_prob, model_name
        )
    
    return evaluator.compare_models(evaluations)