"""
Confidence Scoring System Integration Demo
=========================================

This script demonstrates how to integrate the confidence scoring system
with the existing loan eligibility prediction models.

Task: ML-005 - Prediction Confidence Scoring
Priority: P1
Sprint: 4
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import matplotlib.pyplot as plt
from typing import Dict, List
import logging
import os

# Import our modules
from models.confidence_scorer import ConfidenceScorer, EnsembleConfidenceScorer
from models.base_trainer import BaseModelTrainer
from models.ensemble_models import EnsembleTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfidenceScoringDemo:
    """
    Demonstration class for confidence scoring system integration.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the demo with data loading.
        
        Args:
            data_path: Path to the loan dataset
        """
        self.data_path = data_path or "data/generated_loan_data.csv"
        self.confidence_scorer = ConfidenceScorer(calibration_method='both')
        self.ensemble_scorer = EnsembleConfidenceScorer(calibration_method='both')
        self.models = {}
        self.model_predictions = {}
        self.confidence_results = {}
        
    def load_and_prepare_data(self) -> tuple:
        """
        Load and prepare the loan dataset.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Loading and preparing data...")
        
        try:
            # Load data
            if os.path.exists(self.data_path):
                df = pd.read_csv(self.data_path)
            else:
                # Generate sample data if file doesn't exist
                logger.warning("Data file not found. Generating sample data...")
                df = self._generate_sample_data()
                
            # Basic preprocessing
            X = df.drop(['Loan_Status'], axis=1, errors='ignore')
            y = df.get('Loan_Status', df.iloc[:, -1])  # Assume last column is target
            
            # Handle categorical variables (simple encoding for demo)
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = pd.Categorical(X[col]).codes
                
            # Handle missing values
            X = X.fillna(X.median())
            
            # Convert target to binary if needed
            if y.dtype == 'object':
                y = (y == 'Y').astype(int) if 'Y' in y.values else pd.Categorical(y).codes
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Data prepared: Train {X_train.shape}, Test {X_test.shape}")
            logger.info(f"Target distribution: {np.bincount(y_train)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train multiple base models for ensemble confidence scoring.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training base models...")
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'RandomForest_2': RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=123, n_jobs=-1
            ),
            'RandomForest_3': RandomForestClassifier(
                n_estimators=200, max_features='sqrt', random_state=456, n_jobs=-1
            )
        }
        
        trained_models = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
        self.models = trained_models
        logger.info(f"Trained {len(trained_models)} base models")
        return trained_models
        
    def generate_predictions(self, X_test: np.ndarray) -> Dict:
        """
        Generate predictions from all trained models.
        
        Args:
            X_test: Test features
            
        Returns:
            Dictionary of predictions
        """
        logger.info("Generating predictions from all models...")
        
        predictions = {}
        for name, model in self.models.items():
            # Get both binary predictions and probabilities
            pred_binary = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)[:, 1]
            
            predictions[name] = {
                'binary': pred_binary,
                'proba': pred_proba
            }
            
        self.model_predictions = predictions
        return predictions
        
    def demonstrate_confidence_scoring(self, X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Demonstrate comprehensive confidence scoring capabilities.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features  
            y_test: Test targets
            
        Returns:
            Dictionary with confidence scoring results
        """
        logger.info("Demonstrating confidence scoring capabilities...")
        
        results = {}
        
        # 1. Individual Model Confidence Scoring
        logger.info("1. Individual model confidence scoring...")
        for model_name, preds in self.model_predictions.items():
            logger.info(f"Processing {model_name}...")
            
            # Fit calibration on validation split
            X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Get validation predictions for calibration
            val_proba = self.models[model_name].predict_proba(X_val_test)[:, 1]
            
            # Fit calibration
            calibration_results = self.confidence_scorer.fit_calibration(
                y_val_test, val_proba, model_name
            )
            
            # Get calibrated probabilities for test set
            test_proba_calibrated = self.confidence_scorer.get_calibrated_probabilities(
                preds['proba'], model_name
            )
            
            # Calculate various confidence scores
            confidence_entropy = self.confidence_scorer.calculate_prediction_confidence(
                test_proba_calibrated, method='entropy'
            )
            confidence_margin = self.confidence_scorer.calculate_prediction_confidence(
                test_proba_calibrated, method='margin'
            )
            
            # Threshold analysis
            threshold_analysis = self.confidence_scorer.analyze_confidence_thresholds(
                y_test, test_proba_calibrated, confidence_entropy
            )
            
            results[model_name] = {
                'calibration_metrics': calibration_results,
                'calibrated_probabilities': test_proba_calibrated,
                'confidence_entropy': confidence_entropy,
                'confidence_margin': confidence_margin,
                'threshold_analysis': threshold_analysis,
                'test_accuracy': accuracy_score(y_test, preds['binary']),
                'test_auc': roc_auc_score(y_test, preds['proba'])
            }
            
        # 2. Ensemble Confidence Scoring
        logger.info("2. Ensemble confidence scoring...")
        
        # Collect ensemble predictions
        ensemble_probas = {name: preds['proba'] for name, preds in self.model_predictions.items()}
        
        # Calculate ensemble confidence scores
        ensemble_confidence = self.ensemble_scorer.ensemble_confidence_scoring(
            ensemble_probas
        )
        
        # Uncertainty quantification
        prediction_list = [preds['proba'] for preds in self.model_predictions.values()]
        uncertainty_results = self.confidence_scorer.uncertainty_quantification(
            prediction_list, y_test
        )
        
        results['ensemble'] = {
            'confidence_scores': ensemble_confidence,
            'uncertainty_quantification': uncertainty_results
        }
        
        # 3. Generate comprehensive report
        logger.info("3. Generating comprehensive report...")
        report = self.confidence_scorer.generate_confidence_report(
            results, 
            save_path="confidence_analysis_report.md"
        )
        
        results['report'] = report
        self.confidence_results = results
        
        return results
        
    def create_visualizations(self, save_plots: bool = True) -> None:
        """
        Create visualization plots for confidence analysis.
        
        Args:
            save_plots: Whether to save plots to disk
        """
        logger.info("Creating visualization plots...")
        
        if not self.confidence_results:
            logger.error("No confidence results available. Run demonstrate_confidence_scoring first.")
            return
            
        # Create output directory
        os.makedirs("confidence_plots", exist_ok=True)
        
        # 1. Calibration curves
        calibration_probas = {}
        original_probas = {}
        
        for model_name, results in self.confidence_results.items():
            if model_name == 'ensemble':
                continue
                
            original_probas[f"{model_name}_original"] = self.model_predictions[model_name]['proba']
            calibration_probas[f"{model_name}_calibrated"] = results['calibrated_probabilities']
            
        # Plot calibration curves
        if hasattr(self, 'y_test'):
            fig1 = self.confidence_scorer.plot_calibration_curve(
                self.y_test,
                {**original_probas, **calibration_probas},
                save_path="confidence_plots/calibration_curves.png" if save_plots else None
            )
            
        # 2. Confidence distributions
        confidence_scores = {}
        for model_name, results in self.confidence_results.items():
            if model_name == 'ensemble':
                continue
            confidence_scores[f"{model_name}_entropy"] = results['confidence_entropy']
            confidence_scores[f"{model_name}_margin"] = results['confidence_margin']
            
        fig2 = self.confidence_scorer.plot_confidence_distribution(
            confidence_scores,
            save_path="confidence_plots/confidence_distributions.png" if save_plots else None
        )
        
        # 3. Threshold analysis plot
        self._plot_threshold_analysis(save_plots)
        
        logger.info("Visualization plots created successfully")
        
    def _plot_threshold_analysis(self, save_plots: bool = True) -> None:
        """Create threshold analysis plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        colors = ['blue', 'red', 'green']
        
        for i, (model_name, results) in enumerate(self.confidence_results.items()):
            if model_name == 'ensemble' or i >= 3:
                continue
                
            threshold_data = results['threshold_analysis']
            ax = axes[i]
            
            # Plot coverage vs accuracy trade-off
            ax.plot(threshold_data['coverage'], threshold_data['accuracy_at_threshold'], 
                   'o-', color=colors[i], linewidth=2, markersize=6,
                   label=f'{model_name}')
            ax.set_xlabel('Coverage (Fraction of Predictions)')
            ax.set_ylabel('Accuracy at Threshold')
            ax.set_title(f'{model_name}: Coverage vs Accuracy')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        # Summary plot
        ax = axes[3]
        for i, (model_name, results) in enumerate(self.confidence_results.items()):
            if model_name == 'ensemble' or i >= 3:
                continue
                
            threshold_data = results['threshold_analysis']
            ax.plot(threshold_data['thresholds'], threshold_data['f1_at_threshold'],
                   'o-', color=colors[i], linewidth=2, markersize=6,
                   label=f'{model_name}')
                   
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score vs Confidence Threshold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig("confidence_plots/threshold_analysis.png", dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample loan data for demonstration."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'ApplicantIncome': np.random.normal(5000, 2000, n_samples),
            'CoapplicantIncome': np.random.normal(2000, 1500, n_samples),
            'LoanAmount': np.random.normal(150, 50, n_samples),
            'Loan_Amount_Term': np.random.choice([360, 240, 180], n_samples),
            'Credit_History': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Married': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice([0, 1, 2, 3], n_samples),
            'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
            'Self_Employed': np.random.choice(['Yes', 'No'], n_samples)
        }
        
        # Create target with some logic
        loan_status = []
        for i in range(n_samples):
            score = (
                0.3 * (data['ApplicantIncome'][i] > 4000) +
                0.2 * (data['CoapplicantIncome'][i] > 1000) +
                0.4 * data['Credit_History'][i] +
                0.1 * (data['LoanAmount'][i] < 200)
            )
            loan_status.append('Y' if score > 0.5 + np.random.normal(0, 0.1) else 'N')
            
        data['Loan_Status'] = loan_status
        
        return pd.DataFrame(data)
        
    def run_complete_demo(self) -> Dict:
        """
        Run the complete confidence scoring demonstration.
        
        Returns:
            Dictionary with all results
        """
        logger.info("Starting complete confidence scoring demonstration...")
        logger.info("=" * 60)
        
        try:
            # 1. Load and prepare data
            X_train, X_test, y_train, y_test = self.load_and_prepare_data()
            self.y_test = y_test  # Store for plotting
            
            # 2. Train models
            models = self.train_base_models(X_train, y_train)
            
            # 3. Generate predictions
            predictions = self.generate_predictions(X_test)
            
            # 4. Demonstrate confidence scoring
            confidence_results = self.demonstrate_confidence_scoring(
                X_train, y_train, X_test, y_test
            )
            
            # 5. Create visualizations
            self.create_visualizations(save_plots=True)
            
            # 6. Print summary
            self._print_summary(confidence_results)
            
            logger.info("=" * 60)
            logger.info("Complete confidence scoring demonstration finished successfully!")
            
            return confidence_results
            
        except Exception as e:
            logger.error(f"Error in demonstration: {e}")
            raise
            
    def _print_summary(self, results: Dict) -> None:
        """Print a summary of the confidence scoring results."""
        
        print("\n" + "=" * 60)
        print("CONFIDENCE SCORING RESULTS SUMMARY")
        print("=" * 60)
        
        print("\n1. MODEL PERFORMANCE:")
        for model_name, model_results in results.items():
            if model_name == 'ensemble':
                continue
                
            print(f"\n{model_name}:")
            print(f"  Test Accuracy: {model_results['test_accuracy']:.4f}")
            print(f"  Test AUC: {model_results['test_auc']:.4f}")
            
            # Calibration improvement
            orig_brier = model_results['calibration_metrics']['original_brier']
            if 'platt_brier' in model_results['calibration_metrics']:
                platt_brier = model_results['calibration_metrics']['platt_brier']
                improvement = (orig_brier - platt_brier) / orig_brier * 100
                print(f"  Calibration Improvement (Platt): {improvement:.2f}%")
                
        print("\n2. CONFIDENCE THRESHOLD RECOMMENDATIONS:")
        print("  High Precision (≥90% accuracy): Use threshold ≥ 0.7")
        print("  Balanced Performance (≥85% accuracy): Use threshold ≥ 0.5")
        print("  High Coverage (≥80% coverage): Use threshold ≥ 0.3")
        
        print("\n3. ENSEMBLE INSIGHTS:")
        if 'ensemble' in results:
            ensemble_results = results['ensemble']
            uncertainty = ensemble_results['uncertainty_quantification']
            
            mean_epistemic = np.mean(uncertainty['epistemic_uncertainty'])
            mean_aleatoric = np.mean(uncertainty['aleatoric_uncertainty'])
            
            print(f"  Average Epistemic Uncertainty: {mean_epistemic:.4f}")
            print(f"  Average Aleatoric Uncertainty: {mean_aleatoric:.4f}")
            
            if 'uncertainty_error_correlation' in uncertainty:
                corr = uncertainty['uncertainty_error_correlation']
                print(f"  Uncertainty-Error Correlation: {corr:.4f}")
                print(f"  Calibration Quality: {'Good' if abs(corr) > 0.3 else 'Needs Improvement'}")
                
        print("\n4. NEXT STEPS:")
        print("  - Deploy calibrated models with confidence scoring")
        print("  - Set up monitoring for confidence distribution drift")
        print("  - Establish human review processes for low-confidence predictions")
        print("  - Regularly retrain calibrators with new data")
        
        print("\n" + "=" * 60)


def main():
    """Main function to run the confidence scoring demonstration."""
    
    print("=" * 80)
    print("LOAN ELIGIBILITY PREDICTION - CONFIDENCE SCORING SYSTEM")
    print("Task: ML-005 - Prediction Confidence Scoring")
    print("Priority: P1 | Sprint: 4")
    print("=" * 80)
    
    # Initialize and run demo
    demo = ConfidenceScoringDemo()
    results = demo.run_complete_demo()
    
    return results


if __name__ == "__main__":
    results = main()