"""
Ensemble Model Performance Comparison Framework
Provides comprehensive evaluation, comparison, and visualization tools for ensemble models.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our models
from .base_trainer import BaseModelTrainer
from .ensemble_models import EnsembleTrainer
from .random_forest_model import RandomForestTrainer
from .neural_network_model import NeuralNetworkTrainer
from .xgboost_model import XGBoostTrainer

logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    """
    Comprehensive ensemble model evaluation and comparison framework.
    
    Provides tools for:
    - Individual model evaluation
    - Ensemble vs base model comparison
    - Performance visualization
    - Statistical significance testing
    - Business impact analysis
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.evaluation_results = {}
        self.comparison_results = {}
        
        logger.info(f"Initialized ensemble evaluator with output dir: {self.output_dir}")
    
    def add_model(self, name: str, model: BaseModelTrainer):
        """Add a trained model for evaluation."""
        if not model.is_trained:
            raise ValueError(f"Model {name} must be trained before evaluation")
        
        self.models[name] = model
        logger.info(f"Added model {name} for evaluation")
    
    def evaluate_model(self, 
                      name: str, 
                      X_test: Union[pd.DataFrame, np.ndarray], 
                      y_test: Union[pd.Series, np.ndarray],
                      X_train: Union[pd.DataFrame, np.ndarray] = None,
                      y_train: Union[pd.Series, np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            name: Model name
            X_test: Test features
            y_test: Test targets
            X_train: Optional training features for cross-validation
            y_train: Optional training targets for cross-validation
            
        Returns:
            Comprehensive evaluation results
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found. Add it first using add_model()")
        
        model = self.models[name]
        
        # Convert to numpy if needed
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        results = {
            'model_name': name,
            'model_type': model.model_name,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Basic predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Probability predictions if available
        y_pred_proba = None
        if hasattr(model.model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:  # Binary classification
                    y_pred_proba = y_pred_proba[:, 1]
            except:
                pass
        
        # Core metrics
        results['test_metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_test)) == 2:  # Binary classification
                    results['test_metrics']['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                    results['test_metrics']['average_precision'] = average_precision_score(y_test, y_pred_proba)
                else:  # Multi-class
                    results['test_metrics']['roc_auc'] = roc_auc_score(
                        y_test, y_pred_proba, multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
        
        # Performance characteristics
        results['performance'] = {
            'prediction_time_total_ms': prediction_time * 1000,
            'prediction_time_per_sample_ms': (prediction_time / len(X_test)) * 1000,
            'model_size_mb': model.metrics.model_size_mb,
            'training_time_seconds': model.metrics.training_time
        }
        
        # Confusion matrix and classification report
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        results['classification_report'] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        
        # Cross-validation if training data provided
        if X_train is not None and y_train is not None:
            results['cross_validation'] = self._cross_validate_model(model, X_train, y_train)
        
        # Store results
        self.evaluation_results[name] = results
        
        logger.info(f"Evaluated model {name} - Accuracy: {results['test_metrics']['accuracy']:.4f}")
        
        return results
    
    def _cross_validate_model(self, model: BaseModelTrainer, 
                             X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on a model."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Create a fresh model instance for CV
        if hasattr(model, '_create_model'):
            fresh_model = model._create_model(**model.get_default_hyperparameters())
        else:
            fresh_model = model.model
        
        # Apply feature pipeline if available
        if model.feature_pipeline is not None:
            X = model.feature_pipeline.transform(X)
        
        cv_results = {}
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        for metric in metrics:
            try:
                scores = cross_val_score(fresh_model, X, y, cv=cv, scoring=metric)
                cv_results[f'{metric}_mean'] = scores.mean()
                cv_results[f'{metric}_std'] = scores.std()
                cv_results[f'{metric}_scores'] = scores.tolist()
            except Exception as e:
                logger.warning(f"Could not calculate CV {metric}: {e}")
                cv_results[f'{metric}_mean'] = 0.0
                cv_results[f'{metric}_std'] = 0.0
        
        return cv_results
    
    def compare_models(self, 
                      X_test: Union[pd.DataFrame, np.ndarray], 
                      y_test: Union[pd.Series, np.ndarray],
                      metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare all added models.
        
        Args:
            X_test: Test features
            y_test: Test targets
            metrics: List of metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        if not self.models:
            raise ValueError("No models added for comparison")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        comparison_data = []
        
        for name, model in self.models.items():
            if name not in self.evaluation_results:
                # Evaluate if not already done
                self.evaluate_model(name, X_test, y_test)
            
            results = self.evaluation_results[name]
            
            row = {
                'Model': name,
                'Type': results['model_type'],
                'Training_Time_s': results['performance']['training_time_seconds'],
                'Model_Size_MB': results['performance']['model_size_mb'],
                'Prediction_Time_ms': results['performance']['prediction_time_per_sample_ms']
            }
            
            # Add metric scores
            for metric in metrics:
                if metric in results['test_metrics']:
                    row[metric.title()] = results['test_metrics'][metric]
                else:
                    row[metric.title()] = np.nan
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by accuracy (or first available metric)
        sort_metric = 'Accuracy' if 'Accuracy' in comparison_df.columns else comparison_df.columns[5]
        comparison_df = comparison_df.sort_values(sort_metric, ascending=False)
        
        # Store comparison results
        self.comparison_results['model_comparison'] = comparison_df
        
        # Save to CSV
        comparison_df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        
        logger.info(f"Model comparison completed. Results saved to {self.output_dir / 'model_comparison.csv'}")
        
        return comparison_df
    
    def analyze_ensemble_contribution(self, 
                                    ensemble_name: str,
                                    X_test: Union[pd.DataFrame, np.ndarray], 
                                    y_test: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze individual model contributions to ensemble performance.
        
        Args:
            ensemble_name: Name of the ensemble model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Contribution analysis results
        """
        if ensemble_name not in self.models:
            raise ValueError(f"Ensemble model {ensemble_name} not found")
        
        ensemble = self.models[ensemble_name]
        
        if not isinstance(ensemble, EnsembleTrainer):
            raise ValueError(f"Model {ensemble_name} is not an ensemble")
        
        # Convert to numpy if needed
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Get ensemble prediction
        ensemble_pred = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        # Get base model predictions
        base_predictions = ensemble.get_base_model_predictions(X_test)
        
        # Analyze individual contributions
        contributions = {}
        
        for model_name, predictions in base_predictions.items():
            individual_accuracy = accuracy_score(y_test, predictions)
            contributions[model_name] = {
                'individual_accuracy': individual_accuracy,
                'improvement_from_ensemble': ensemble_accuracy - individual_accuracy,
                'agreement_with_ensemble': np.mean(predictions == ensemble_pred),
                'unique_correct_predictions': np.sum(
                    (predictions == y_test) & (ensemble_pred != y_test)
                )
            }
        
        # Get ensemble weights if available
        weights = ensemble.get_ensemble_weights()
        if weights is not None:
            for i, (model_name, _) in enumerate(base_predictions.items()):
                contributions[model_name]['ensemble_weight'] = weights[i]
        
        analysis_results = {
            'ensemble_accuracy': ensemble_accuracy,
            'base_model_contributions': contributions,
            'diversity_metrics': self._calculate_diversity_metrics(base_predictions, y_test),
            'ensemble_weights': weights.tolist() if weights is not None else None
        }
        
        # Save analysis
        with open(self.output_dir / f'{ensemble_name}_contribution_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Ensemble contribution analysis completed for {ensemble_name}")
        
        return analysis_results
    
    def _calculate_diversity_metrics(self, base_predictions: Dict[str, np.ndarray], 
                                   y_true: np.ndarray) -> Dict[str, float]:
        """Calculate diversity metrics for ensemble models."""
        model_names = list(base_predictions.keys())
        predictions = np.array(list(base_predictions.values())).T  # Shape: (n_samples, n_models)
        
        diversity_metrics = {}
        
        # Pairwise disagreement
        n_models = len(model_names)
        disagreements = []
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                disagreement = np.mean(predictions[:, i] != predictions[:, j])
                disagreements.append(disagreement)
        
        diversity_metrics['average_pairwise_disagreement'] = np.mean(disagreements)
        
        # Q-statistic (for binary problems)
        if len(np.unique(y_true)) == 2:
            q_statistics = []
            
            for i in range(n_models):
                for j in range(i+1, n_models):
                    pred_i = predictions[:, i]
                    pred_j = predictions[:, j]
                    
                    n11 = np.sum((pred_i == y_true) & (pred_j == y_true))
                    n10 = np.sum((pred_i == y_true) & (pred_j != y_true))
                    n01 = np.sum((pred_i != y_true) & (pred_j == y_true))
                    n00 = np.sum((pred_i != y_true) & (pred_j != y_true))
                    
                    if (n11 * n00 + n10 * n01) > 0:
                        q = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
                        q_statistics.append(q)
            
            if q_statistics:
                diversity_metrics['average_q_statistic'] = np.mean(q_statistics)
        
        # Entropy-based diversity
        ensemble_entropy = 0
        for i in range(len(y_true)):
            prediction_counts = np.bincount(predictions[i].astype(int))
            probabilities = prediction_counts / n_models
            probabilities = probabilities[probabilities > 0]  # Remove zeros
            ensemble_entropy += -np.sum(probabilities * np.log2(probabilities))
        
        diversity_metrics['average_entropy'] = ensemble_entropy / len(y_true)
        
        return diversity_metrics
    
    def create_performance_visualizations(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive performance visualizations.
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary of plot objects
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Evaluate models first.")
        
        plots = {}
        
        # 1. Performance comparison bar chart
        plots['performance_comparison'] = self._create_performance_comparison_plot()
        
        # 2. ROC curves (if binary classification)
        plots['roc_curves'] = self._create_roc_curves_plot()
        
        # 3. Precision-Recall curves
        plots['pr_curves'] = self._create_precision_recall_curves_plot()
        
        # 4. Confusion matrices
        plots['confusion_matrices'] = self._create_confusion_matrices_plot()
        
        # 5. Model efficiency plot (accuracy vs time)
        plots['efficiency_plot'] = self._create_efficiency_plot()
        
        # 6. Feature importance comparison (if available)
        plots['feature_importance'] = self._create_feature_importance_plot()
        
        if save_plots:
            self._save_plots(plots)
        
        logger.info("Performance visualizations created")
        
        return plots
    
    def _create_performance_comparison_plot(self):
        """Create performance comparison bar chart."""
        metrics_data = []
        
        for name, results in self.evaluation_results.items():
            for metric, value in results['test_metrics'].items():
                if not np.isnan(value):
                    metrics_data.append({
                        'Model': name,
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': value
                    })
        
        df = pd.DataFrame(metrics_data)
        
        if df.empty:
            return None
        
        fig = px.bar(
            df, 
            x='Model', 
            y='Value', 
            color='Metric',
            barmode='group',
            title='Model Performance Comparison',
            labels={'Value': 'Score', 'Model': 'Model Name'}
        )
        
        fig.update_layout(
            height=600,
            xaxis_tickangle=-45,
            legend_title="Metrics"
        )
        
        return fig
    
    def _create_roc_curves_plot(self):
        """Create ROC curves plot for binary classification."""
        fig = go.Figure()
        
        # Add diagonal line
        fig.add_shape(
            type='line',
            line=dict(dash='dash', color='gray'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        for name, model in self.models.items():
            try:
                # This would need test data - simplified for now
                # In practice, you'd pass X_test, y_test to create actual ROC curves
                pass
            except:
                continue
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=600
        )
        
        return fig
    
    def _create_precision_recall_curves_plot(self):
        """Create Precision-Recall curves plot."""
        # Placeholder - would need test predictions for actual implementation
        fig = go.Figure()
        fig.update_layout(
            title='Precision-Recall Curves Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=600
        )
        
        return fig
    
    def _create_confusion_matrices_plot(self):
        """Create confusion matrices visualization."""
        n_models = len(self.evaluation_results)
        
        if n_models == 0:
            return None
        
        # Create subplot layout
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(self.evaluation_results.keys()),
            specs=[[{'type': 'heatmap'}] * cols for _ in range(rows)]
        )
        
        for i, (name, results) in enumerate(self.evaluation_results.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            cm = np.array(results['confusion_matrix'])
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    colorscale='Blues',
                    showscale=True if i == 0 else False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Confusion Matrices Comparison',
            height=200 * rows + 100
        )
        
        return fig
    
    def _create_efficiency_plot(self):
        """Create efficiency plot (accuracy vs prediction time)."""
        efficiency_data = []
        
        for name, results in self.evaluation_results.items():
            efficiency_data.append({
                'Model': name,
                'Accuracy': results['test_metrics']['accuracy'],
                'Prediction_Time_ms': results['performance']['prediction_time_per_sample_ms'],
                'Model_Size_MB': results['performance']['model_size_mb'],
                'Training_Time_s': results['performance']['training_time_seconds']
            })
        
        df = pd.DataFrame(efficiency_data)
        
        fig = px.scatter(
            df, 
            x='Prediction_Time_ms', 
            y='Accuracy',
            size='Model_Size_MB',
            hover_data=['Training_Time_s'],
            text='Model',
            title='Model Efficiency: Accuracy vs Prediction Time',
            labels={
                'Prediction_Time_ms': 'Prediction Time (ms per sample)',
                'Accuracy': 'Test Accuracy'
            }
        )
        
        fig.update_traces(textposition="top center")
        fig.update_layout(height=600)
        
        return fig
    
    def _create_feature_importance_plot(self):
        """Create feature importance comparison plot."""
        # Placeholder - would need to extract feature importance from models
        fig = go.Figure()
        fig.update_layout(
            title='Feature Importance Comparison',
            height=600
        )
        
        return fig
    
    def _save_plots(self, plots: Dict[str, Any]):
        """Save plots to files."""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for plot_name, plot_obj in plots.items():
            if plot_obj is not None:
                try:
                    plot_obj.write_html(plots_dir / f"{plot_name}.html")
                    plot_obj.write_image(plots_dir / f"{plot_name}.png", width=1200, height=800)
                except Exception as e:
                    logger.warning(f"Could not save plot {plot_name}: {e}")
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive evaluation report."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        report_lines = [
            "# Comprehensive Model Evaluation Report",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Performance Summary",
            ""
        ]
        
        # Performance summary table
        if 'model_comparison' in self.comparison_results:
            df = self.comparison_results['model_comparison']
            report_lines.append("### Performance Comparison")
            report_lines.append(df.to_string(index=False))
            report_lines.append("")
        
        # Individual model details
        report_lines.append("## Individual Model Analysis")
        report_lines.append("")
        
        for name, results in self.evaluation_results.items():
            report_lines.extend([
                f"### {name} ({results['model_type']})",
                "",
                "**Test Metrics:**",
                ""
            ])
            
            for metric, value in results['test_metrics'].items():
                report_lines.append(f"- {metric.replace('_', ' ').title()}: {value:.4f}")
            
            report_lines.extend([
                "",
                "**Performance Characteristics:**",
                f"- Training Time: {results['performance']['training_time_seconds']:.2f} seconds",
                f"- Model Size: {results['performance']['model_size_mb']:.2f} MB",
                f"- Prediction Time: {results['performance']['prediction_time_per_sample_ms']:.4f} ms/sample",
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            self._generate_recommendations(),
            ""
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "comprehensive_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Comprehensive report saved to {report_path}")
        
        return report_content
    
    def _generate_recommendations(self) -> str:
        """Generate model recommendations based on evaluation results."""
        if not self.evaluation_results:
            return "No evaluation results available for recommendations."
        
        recommendations = []
        
        # Find best performing model
        best_accuracy = 0
        best_model = ""
        
        for name, results in self.evaluation_results.items():
            accuracy = results['test_metrics']['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = name
        
        recommendations.append(f"**Best Overall Performance:** {best_model} with {best_accuracy:.4f} accuracy")
        
        # Find most efficient model
        fastest_model = ""
        fastest_time = float('inf')
        
        for name, results in self.evaluation_results.items():
            pred_time = results['performance']['prediction_time_per_sample_ms']
            if pred_time < fastest_time:
                fastest_time = pred_time
                fastest_model = name
        
        recommendations.append(f"**Fastest Prediction:** {fastest_model} with {fastest_time:.4f} ms/sample")
        
        # Model selection recommendation
        if best_accuracy >= 0.90:
            recommendations.append("✅ **Target Achieved:** The best model meets the ≥90% accuracy requirement")
        else:
            recommendations.append("❌ **Target Not Met:** Consider ensemble methods or hyperparameter tuning")
        
        return "\n".join(recommendations)