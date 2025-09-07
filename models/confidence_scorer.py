"""
Prediction Confidence Scoring System
====================================

This module implements comprehensive confidence scoring for loan eligibility predictions,
including probability calibration, uncertainty quantification, and threshold analysis.

Task: ML-005 - Prediction Confidence Scoring
Priority: P1
Sprint: 4
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ConfidenceScorer:
    """
    Comprehensive confidence scoring system for ML predictions.
    
    Features:
    - Probability calibration (Platt scaling, Isotonic regression)
    - Uncertainty quantification methods
    - Confidence threshold analysis
    - Prediction reliability assessment
    """
    
    def __init__(self, calibration_method: str = 'platt', random_state: int = 42):
        """
        Initialize the confidence scorer.
        
        Args:
            calibration_method: 'platt', 'isotonic', or 'both'
            random_state: Random state for reproducibility
        """
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.calibrators = {}
        self.calibration_metrics = {}
        self.confidence_thresholds = {}
        self.fitted = False
        
    def fit_calibration(self, y_true: np.ndarray, y_proba: np.ndarray, 
                       model_name: str = 'default') -> Dict:
        """
        Fit probability calibration methods.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities (uncalibrated)
            model_name: Name of the model for tracking
            
        Returns:
            Dictionary with calibration metrics
        """
        results = {}
        
        # Platt Scaling (Sigmoid calibration)
        if self.calibration_method in ['platt', 'both']:
            platt_calibrator = LogisticRegression(random_state=self.random_state)
            platt_calibrator.fit(y_proba.reshape(-1, 1), y_true)
            self.calibrators[f'{model_name}_platt'] = platt_calibrator
            
            # Get calibrated probabilities
            platt_proba = platt_calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
            results['platt_calibrated'] = platt_proba
            
        # Isotonic Regression
        if self.calibration_method in ['isotonic', 'both']:
            isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
            isotonic_calibrator.fit(y_proba, y_true)
            self.calibrators[f'{model_name}_isotonic'] = isotonic_calibrator
            
            # Get calibrated probabilities
            isotonic_proba = isotonic_calibrator.predict(y_proba)
            results['isotonic_calibrated'] = isotonic_proba
            
        # Calculate calibration metrics
        results['original_brier'] = brier_score_loss(y_true, y_proba)
        results['original_logloss'] = log_loss(y_true, y_proba)
        
        if 'platt_calibrated' in results:
            results['platt_brier'] = brier_score_loss(y_true, results['platt_calibrated'])
            results['platt_logloss'] = log_loss(y_true, results['platt_calibrated'])
            
        if 'isotonic_calibrated' in results:
            results['isotonic_brier'] = brier_score_loss(y_true, results['isotonic_calibrated'])
            results['isotonic_logloss'] = log_loss(y_true, results['isotonic_calibrated'])
            
        self.calibration_metrics[model_name] = results
        return results
        
    def get_calibrated_probabilities(self, y_proba: np.ndarray, 
                                   model_name: str = 'default',
                                   method: str = None) -> np.ndarray:
        """
        Get calibrated probabilities using fitted calibrators.
        
        Args:
            y_proba: Uncalibrated probabilities
            model_name: Model name
            method: 'platt', 'isotonic', or None (uses best method)
            
        Returns:
            Calibrated probabilities
        """
        if method is None:
            # Choose best method based on validation metrics
            method = self._get_best_calibration_method(model_name)
            
        calibrator_key = f'{model_name}_{method}'
        if calibrator_key not in self.calibrators:
            raise ValueError(f"Calibrator {calibrator_key} not fitted")
            
        if method == 'platt':
            return self.calibrators[calibrator_key].predict_proba(y_proba.reshape(-1, 1))[:, 1]
        else:  # isotonic
            return self.calibrators[calibrator_key].predict(y_proba)
            
    def calculate_prediction_confidence(self, y_proba: np.ndarray, 
                                      method: str = 'entropy') -> np.ndarray:
        """
        Calculate confidence scores for predictions.
        
        Args:
            y_proba: Predicted probabilities
            method: 'entropy', 'max_prob', 'margin', 'variance'
            
        Returns:
            Confidence scores (higher = more confident)
        """
        if method == 'entropy':
            # Shannon entropy (lower entropy = higher confidence)
            entropy = -np.sum(np.c_[1-y_proba, y_proba] * np.log2(np.c_[1-y_proba, y_proba] + 1e-8), axis=1)
            return 1 - entropy  # Invert so higher = more confident
            
        elif method == 'max_prob':
            # Maximum probability
            return np.maximum(y_proba, 1 - y_proba)
            
        elif method == 'margin':
            # Margin between top two probabilities
            return np.abs(2 * y_proba - 1)
            
        elif method == 'variance':
            # Variance-based confidence (for ensemble methods)
            # This assumes y_proba is ensemble predictions
            return 1 / (1 + np.var(y_proba, axis=0) if y_proba.ndim > 1 else np.var(y_proba))
            
        else:
            raise ValueError(f"Unknown confidence method: {method}")
            
    def uncertainty_quantification(self, predictions: List[np.ndarray], 
                                 y_true: Optional[np.ndarray] = None) -> Dict:
        """
        Quantify prediction uncertainty using ensemble methods.
        
        Args:
            predictions: List of prediction arrays from different models
            y_true: True labels (optional, for validation)
            
        Returns:
            Dictionary with uncertainty metrics
        """
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = std_pred
        
        # Aleatoric uncertainty (data uncertainty) - approximated
        aleatoric_uncertainty = mean_pred * (1 - mean_pred)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        results = {
            'mean_prediction': mean_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'prediction_variance': std_pred**2
        }
        
        # If true labels provided, calculate uncertainty calibration
        if y_true is not None:
            # Uncertainty calibration: high uncertainty should correlate with errors
            errors = np.abs(y_true - mean_pred)
            uncertainty_error_corr = np.corrcoef(total_uncertainty, errors)[0, 1]
            results['uncertainty_error_correlation'] = uncertainty_error_corr
            
        return results
        
    def analyze_confidence_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    confidence_scores: np.ndarray,
                                    thresholds: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze performance at different confidence thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            confidence_scores: Confidence scores
            thresholds: Confidence thresholds to analyze
            
        Returns:
            Dictionary with threshold analysis results
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
            
        results = {
            'thresholds': thresholds,
            'coverage': [],
            'accuracy_at_threshold': [],
            'precision_at_threshold': [],
            'recall_at_threshold': [],
            'f1_at_threshold': []
        }
        
        for threshold in thresholds:
            # Filter predictions above confidence threshold
            confident_mask = confidence_scores >= threshold
            
            if np.sum(confident_mask) == 0:
                # No predictions above threshold
                results['coverage'].append(0)
                results['accuracy_at_threshold'].append(np.nan)
                results['precision_at_threshold'].append(np.nan)
                results['recall_at_threshold'].append(np.nan)
                results['f1_at_threshold'].append(np.nan)
                continue
                
            coverage = np.mean(confident_mask)
            y_true_filtered = y_true[confident_mask]
            y_pred_filtered = (y_proba[confident_mask] > 0.5).astype(int)
            
            # Calculate metrics for confident predictions
            accuracy = np.mean(y_true_filtered == y_pred_filtered)
            
            # Precision, Recall, F1
            tp = np.sum((y_true_filtered == 1) & (y_pred_filtered == 1))
            fp = np.sum((y_true_filtered == 0) & (y_pred_filtered == 1))
            fn = np.sum((y_true_filtered == 1) & (y_pred_filtered == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results['coverage'].append(coverage)
            results['accuracy_at_threshold'].append(accuracy)
            results['precision_at_threshold'].append(precision)
            results['recall_at_threshold'].append(recall)
            results['f1_at_threshold'].append(f1)
            
        return results
        
    def generate_confidence_report(self, model_results: Dict, 
                                 save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive confidence analysis report.
        
        Args:
            model_results: Dictionary with model evaluation results
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        report_lines = [
            "# Prediction Confidence Analysis Report",
            "=" * 50,
            "",
            f"**Task**: ML-005 - Prediction Confidence Scoring",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
        ]
        
        # Add model-specific results
        for model_name, results in model_results.items():
            report_lines.extend([
                f"### {model_name} Model Results",
                "",
                "#### Calibration Metrics:",
            ])
            
            if 'calibration_metrics' in results:
                metrics = results['calibration_metrics']
                report_lines.extend([
                    f"- Original Brier Score: {metrics.get('original_brier', 'N/A'):.4f}",
                    f"- Original Log Loss: {metrics.get('original_logloss', 'N/A'):.4f}",
                ])
                
                if 'platt_brier' in metrics:
                    report_lines.extend([
                        f"- Platt Calibrated Brier Score: {metrics['platt_brier']:.4f}",
                        f"- Platt Calibrated Log Loss: {metrics['platt_logloss']:.4f}",
                    ])
                    
                if 'isotonic_brier' in metrics:
                    report_lines.extend([
                        f"- Isotonic Calibrated Brier Score: {metrics['isotonic_brier']:.4f}",
                        f"- Isotonic Calibrated Log Loss: {metrics['isotonic_logloss']:.4f}",
                    ])
                    
            report_lines.append("")
            
            # Add threshold analysis
            if 'threshold_analysis' in results:
                threshold_results = results['threshold_analysis']
                report_lines.extend([
                    "#### Confidence Threshold Analysis:",
                    "",
                    "| Threshold | Coverage | Accuracy | Precision | Recall | F1-Score |",
                    "|-----------|----------|----------|-----------|--------|----------|",
                ])
                
                for i, threshold in enumerate(threshold_results['thresholds']):
                    coverage = threshold_results['coverage'][i]
                    accuracy = threshold_results['accuracy_at_threshold'][i]
                    precision = threshold_results['precision_at_threshold'][i]
                    recall = threshold_results['recall_at_threshold'][i]
                    f1 = threshold_results['f1_at_threshold'][i]
                    
                    report_lines.append(
                        f"| {threshold:.1f}     | {coverage:.3f}    | {accuracy:.3f}    | "
                        f"{precision:.3f}     | {recall:.3f}  | {f1:.3f}    |"
                    )
                    
            report_lines.append("")
            
        # Add recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "### Confidence Threshold Selection:",
            "- **High Precision (Low Risk)**: Use confidence threshold ≥ 0.7",
            "- **Balanced Performance**: Use confidence threshold ≥ 0.5", 
            "- **High Coverage**: Use confidence threshold ≥ 0.3",
            "",
            "### Calibration Method Selection:",
            "- Use Platt scaling for smaller datasets or well-calibrated models",
            "- Use Isotonic regression for larger datasets or poorly-calibrated models",
            "",
            "### Implementation Guidelines:",
            "1. Always use calibrated probabilities for confidence scoring",
            "2. Monitor confidence distribution in production",
            "3. Set up alerts for sudden changes in confidence patterns",
            "4. Regularly retrain calibrators with new data",
            "",
            "## Score Interpretation Guide",
            "",
            "### Confidence Scores:",
            "- **0.9-1.0**: Very High Confidence - Minimal human review needed",
            "- **0.7-0.9**: High Confidence - Standard processing",
            "- **0.5-0.7**: Medium Confidence - Additional validation recommended",
            "- **0.3-0.5**: Low Confidence - Human review required",
            "- **0.0-0.3**: Very Low Confidence - Manual decision required",
            "",
            "### Uncertainty Interpretation:",
            "- **Epistemic Uncertainty**: Uncertainty due to model limitations",
            "  - High values indicate need for more diverse training data",
            "- **Aleatoric Uncertainty**: Uncertainty inherent in the data",
            "  - High values indicate inherently difficult cases",
            "",
            "---",
            "**Note**: This report is automatically generated. Please validate results before production use."
        ])
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
                
        return report_text
        
    def _get_best_calibration_method(self, model_name: str) -> str:
        """Select best calibration method based on validation metrics."""
        if model_name not in self.calibration_metrics:
            return 'platt'  # Default
            
        metrics = self.calibration_metrics[model_name]
        
        # Compare Brier scores (lower is better)
        platt_score = metrics.get('platt_brier', float('inf'))
        isotonic_score = metrics.get('isotonic_brier', float('inf'))
        
        return 'platt' if platt_score <= isotonic_score else 'isotonic'
        
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba_dict: Dict[str, np.ndarray],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curves for different models/methods.
        
        Args:
            y_true: True binary labels
            y_proba_dict: Dictionary mapping method names to probabilities
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (method_name, y_proba) in enumerate(y_proba_dict.items()):
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=10
            )
            
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   marker='o', linewidth=2, color=colors[i % len(colors)], 
                   label=f'{method_name}')
                   
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_confidence_distribution(self, confidence_scores: Dict[str, np.ndarray],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confidence score distributions.
        
        Args:
            confidence_scores: Dictionary mapping method names to confidence scores
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (method_name, scores) in enumerate(confidence_scores.items()):
            if i >= 4:  # Limit to 4 methods
                break
                
            ax = axes[i]
            
            # Histogram
            ax.hist(scores, bins=30, alpha=0.7, color=colors[i], density=True)
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{method_name} Confidence Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_conf = np.mean(scores)
            median_conf = np.median(scores)
            ax.axvline(mean_conf, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_conf:.3f}')
            ax.axvline(median_conf, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_conf:.3f}')
            ax.legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class EnsembleConfidenceScorer(ConfidenceScorer):
    """
    Specialized confidence scorer for ensemble methods.
    
    Extends base ConfidenceScorer with ensemble-specific confidence measures.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def ensemble_confidence_scoring(self, ensemble_predictions: Dict[str, np.ndarray],
                                  weights: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate confidence scores specific to ensemble predictions.
        
        Args:
            ensemble_predictions: Dict mapping model names to prediction arrays
            weights: Optional model weights for weighted confidence
            
        Returns:
            Dictionary with various confidence measures
        """
        predictions = np.array(list(ensemble_predictions.values()))
        n_models, n_samples = predictions.shape
        
        # Equal weights if not provided
        if weights is None:
            weights = np.ones(n_models) / n_models
            
        # Weighted ensemble prediction
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        
        # Agreement-based confidence
        agreement_confidence = self._calculate_agreement_confidence(predictions)
        
        # Consensus-based confidence
        consensus_confidence = self._calculate_consensus_confidence(predictions, weights)
        
        # Diversity-based confidence (lower diversity = higher confidence)
        diversity_scores = self._calculate_diversity_scores(predictions)
        diversity_confidence = 1 / (1 + diversity_scores)
        
        # Combined confidence score
        combined_confidence = np.mean([
            agreement_confidence,
            consensus_confidence,
            diversity_confidence
        ], axis=0)
        
        return {
            'weighted_prediction': weighted_pred,
            'agreement_confidence': agreement_confidence,
            'consensus_confidence': consensus_confidence,
            'diversity_confidence': diversity_confidence,
            'combined_confidence': combined_confidence,
            'prediction_variance': np.var(predictions, axis=0),
            'prediction_std': np.std(predictions, axis=0)
        }
        
    def _calculate_agreement_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence based on model agreement."""
        # For binary classification, agreement is measured by std deviation
        # Lower std = higher agreement = higher confidence
        std_pred = np.std(predictions, axis=0)
        max_possible_std = 0.5  # Maximum std for binary predictions
        return 1 - (std_pred / max_possible_std)
        
    def _calculate_consensus_confidence(self, predictions: np.ndarray, 
                                      weights: np.ndarray) -> np.ndarray:
        """Calculate confidence based on weighted consensus."""
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        
        # Confidence based on how far the weighted prediction is from 0.5
        return np.abs(2 * weighted_pred - 1)
        
    def _calculate_diversity_scores(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate diversity scores (disagreement) between models."""
        n_models = predictions.shape[0]
        diversity_scores = []
        
        for i in range(predictions.shape[1]):  # For each sample
            sample_preds = predictions[:, i]
            
            # Calculate pairwise disagreement
            disagreement = 0
            for j in range(n_models):
                for k in range(j + 1, n_models):
                    disagreement += abs(sample_preds[j] - sample_preds[k])
                    
            # Normalize by number of pairs
            n_pairs = (n_models * (n_models - 1)) / 2
            diversity_scores.append(disagreement / n_pairs if n_pairs > 0 else 0)
            
        return np.array(diversity_scores)