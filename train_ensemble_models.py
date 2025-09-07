"""
Comprehensive Ensemble Model Training Script
Trains multiple ensemble methods to achieve ≥90% accuracy on loan eligibility prediction.
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import our models and evaluation framework
sys.path.append(str(Path(__file__).parent))
from models.ensemble_models import EnsembleTrainer
from models.ensemble_evaluator import EnsembleEvaluator
from models.random_forest_model import RandomForestTrainer
from models.neural_network_model import NeuralNetworkTrainer
from models.xgboost_model import XGBoostTrainer
from feature_engineering.pipeline import FeatureEngineeringPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ensemble_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str = "loan_dataset.csv") -> tuple:
    """Load and prepare the loan dataset."""
    logger.info("Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Separate features and target
    target_col = 'loan_approved'
    feature_cols = [col for col in df.columns if col not in [target_col, 'application_date']]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Basic preprocessing
    # Handle missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        else:
            X[col] = X[col].fillna(X[col].median())
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Ensure target is binary
    if y.dtype != 'int64':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    logger.info(f"Prepared features shape: {X.shape}")
    logger.info(f"Target distribution: {np.bincount(y)}")
    
    return X, y, label_encoders


def train_base_models(X_train, y_train, X_val, y_val, feature_pipeline=None) -> dict:
    """Train individual base models for comparison."""
    logger.info("Training base models...")
    
    base_models = {}
    
    # Random Forest
    logger.info("Training Random Forest...")
    rf_trainer = RandomForestTrainer(random_state=42, verbose=True)
    rf_trainer.train(X_train, y_train, validation_split=0.2, feature_pipeline=feature_pipeline)
    rf_trainer.evaluate(X_val, y_val)
    base_models['RandomForest'] = rf_trainer
    
    # XGBoost
    logger.info("Training XGBoost...")
    xgb_trainer = XGBoostTrainer(random_state=42, verbose=True)
    xgb_trainer.train(X_train, y_train, validation_split=0.2, feature_pipeline=feature_pipeline)
    xgb_trainer.evaluate(X_val, y_val)
    base_models['XGBoost'] = xgb_trainer
    
    # Neural Network
    logger.info("Training Neural Network...")
    nn_trainer = NeuralNetworkTrainer(random_state=42, verbose=True)
    nn_trainer.train(X_train, y_train, validation_split=0.2, feature_pipeline=feature_pipeline)
    nn_trainer.evaluate(X_val, y_val)
    base_models['NeuralNetwork'] = nn_trainer
    
    return base_models


def train_ensemble_models(X_train, y_train, X_val, y_val, feature_pipeline=None) -> dict:
    """Train various ensemble models."""
    logger.info("Training ensemble models...")
    
    ensemble_models = {}
    base_model_names = ['RandomForest', 'XGBoost', 'NeuralNetwork']
    
    # 1. Voting Classifier (Soft)
    logger.info("Training Voting Classifier (Soft)...")
    voting_soft = EnsembleTrainer(
        ensemble_type="voting",
        base_models=base_model_names,
        random_state=42,
        verbose=True
    )
    voting_soft.train(
        X_train, y_train, validation_split=0.2,
        feature_pipeline=feature_pipeline,
        hyperparameters={"voting": "soft"}
    )
    voting_soft.evaluate(X_val, y_val)
    ensemble_models['Voting_Soft'] = voting_soft
    
    # 2. Voting Classifier (Hard)
    logger.info("Training Voting Classifier (Hard)...")
    voting_hard = EnsembleTrainer(
        ensemble_type="voting",
        base_models=base_model_names,
        random_state=42,
        verbose=True
    )
    voting_hard.train(
        X_train, y_train, validation_split=0.2,
        feature_pipeline=feature_pipeline,
        hyperparameters={"voting": "hard"}
    )
    voting_hard.evaluate(X_val, y_val)
    ensemble_models['Voting_Hard'] = voting_hard
    
    # 3. Stacking Ensemble
    logger.info("Training Stacking Ensemble...")
    stacking = EnsembleTrainer(
        ensemble_type="stacking",
        base_models=base_model_names,
        random_state=42,
        verbose=True
    )
    stacking.train(
        X_train, y_train, validation_split=0.2,
        feature_pipeline=feature_pipeline,
        hyperparameters={"cv_folds": 5}
    )
    stacking.evaluate(X_val, y_val)
    ensemble_models['Stacking'] = stacking
    
    # 4. Weighted Ensemble (Differential Evolution)
    logger.info("Training Weighted Ensemble (Differential Evolution)...")
    weighted_de = EnsembleTrainer(
        ensemble_type="weighted",
        base_models=base_model_names,
        random_state=42,
        verbose=True
    )
    weighted_de.train(
        X_train, y_train, validation_split=0.2,
        feature_pipeline=feature_pipeline,
        weight_optimization="differential_evolution"
    )
    weighted_de.evaluate(X_val, y_val)
    ensemble_models['Weighted_DE'] = weighted_de
    
    # 5. Blending Ensemble
    logger.info("Training Blending Ensemble...")
    blending = EnsembleTrainer(
        ensemble_type="blending",
        base_models=base_model_names,
        random_state=42,
        verbose=True
    )
    blending.train(
        X_train, y_train, validation_split=0.2,
        feature_pipeline=feature_pipeline,
        weight_optimization="differential_evolution"
    )
    blending.evaluate(X_val, y_val)
    ensemble_models['Blending'] = blending
    
    return ensemble_models


def comprehensive_evaluation(all_models: dict, X_test, y_test, X_train, y_train) -> EnsembleEvaluator:
    """Perform comprehensive evaluation of all models."""
    logger.info("Performing comprehensive evaluation...")
    
    # Initialize evaluator
    evaluator = EnsembleEvaluator(output_dir="evaluation_results")
    
    # Add all models to evaluator
    for name, model in all_models.items():
        evaluator.add_model(name, model)
    
    # Evaluate all models
    for name, model in all_models.items():
        evaluator.evaluate_model(name, X_test, y_test, X_train, y_train)
    
    # Create comparison
    comparison_df = evaluator.compare_models(X_test, y_test)
    logger.info("\nModel Comparison Results:")
    logger.info(comparison_df.to_string(index=False))
    
    # Analyze ensemble contributions for ensemble models
    ensemble_model_names = [name for name in all_models.keys() if 'Ensemble' in name or any(ens_type in name for ens_type in ['Voting', 'Stacking', 'Weighted', 'Blending'])]
    
    for ensemble_name in ensemble_model_names:
        try:
            contribution_analysis = evaluator.analyze_ensemble_contribution(
                ensemble_name, X_test, y_test
            )
            logger.info(f"\nEnsemble Contribution Analysis for {ensemble_name}:")
            logger.info(f"Ensemble Accuracy: {contribution_analysis['ensemble_accuracy']:.4f}")
            
            for model_name, contrib in contribution_analysis['base_model_contributions'].items():
                logger.info(f"  {model_name}: Individual={contrib['individual_accuracy']:.4f}, "
                          f"Improvement={contrib['improvement_from_ensemble']:+.4f}")
        except Exception as e:
            logger.warning(f"Could not analyze contributions for {ensemble_name}: {e}")
    
    # Create visualizations
    try:
        evaluator.create_performance_visualizations(save_plots=True)
        logger.info("Performance visualizations created and saved")
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")
    
    # Generate comprehensive report
    try:
        report = evaluator.generate_comprehensive_report()
        logger.info("Comprehensive evaluation report generated")
    except Exception as e:
        logger.warning(f"Could not generate report: {e}")
    
    return evaluator


def check_accuracy_target(all_models: dict, target_accuracy: float = 0.90) -> dict:
    """Check which models meet the accuracy target."""
    logger.info(f"\nChecking accuracy target of {target_accuracy:.1%}...")
    
    results = {}
    meeting_target = []
    
    for name, model in all_models.items():
        if hasattr(model, 'metrics') and hasattr(model.metrics, 'test_scores'):
            test_accuracy = model.metrics.test_scores.get('test_accuracy', 0)
            results[name] = test_accuracy
            
            if test_accuracy >= target_accuracy:
                meeting_target.append((name, test_accuracy))
                logger.info(f"✅ {name}: {test_accuracy:.4f} (MEETS TARGET)")
            else:
                logger.info(f"❌ {name}: {test_accuracy:.4f} (below target)")
    
    if meeting_target:
        logger.info(f"\n🎯 {len(meeting_target)} model(s) meet the ≥{target_accuracy:.1%} accuracy target:")
        for name, accuracy in sorted(meeting_target, key=lambda x: x[1], reverse=True):
            logger.info(f"   {name}: {accuracy:.4f}")
        
        best_model_name, best_accuracy = max(meeting_target, key=lambda x: x[1])
        logger.info(f"\n🏆 Best performing model: {best_model_name} with {best_accuracy:.4f} accuracy")
    else:
        logger.info(f"\n❌ No models meet the ≥{target_accuracy:.1%} accuracy target")
        
        # Show best performing model
        if results:
            best_name, best_acc = max(results.items(), key=lambda x: x[1])
            logger.info(f"Best performing model: {best_name} with {best_acc:.4f} accuracy")
    
    return results


def save_best_models(all_models: dict, target_accuracy: float = 0.90):
    """Save models that meet the accuracy target."""
    logger.info("Saving best performing models...")
    
    best_models_dir = Path("best_models")
    best_models_dir.mkdir(exist_ok=True)
    
    saved_models = []
    
    for name, model in all_models.items():
        if hasattr(model, 'metrics') and hasattr(model.metrics, 'test_scores'):
            test_accuracy = model.metrics.test_scores.get('test_accuracy', 0)
            
            if test_accuracy >= target_accuracy:
                try:
                    model_path = model.save_model(str(best_models_dir / f"{name}_best.pkl"))
                    saved_models.append((name, test_accuracy, model_path))
                    logger.info(f"Saved {name} (accuracy: {test_accuracy:.4f}) to {model_path}")
                except Exception as e:
                    logger.warning(f"Could not save {name}: {e}")
    
    # Save model summary
    if saved_models:
        summary_path = best_models_dir / "model_summary.json"
        summary = {
            "target_accuracy": target_accuracy,
            "models_meeting_target": len(saved_models),
            "saved_models": [
                {
                    "name": name,
                    "accuracy": accuracy,
                    "model_path": model_path
                }
                for name, accuracy, model_path in saved_models
            ]
        }
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Model summary saved to {summary_path}")


def main():
    """Main training pipeline."""
    logger.info("Starting comprehensive ensemble model training pipeline")
    logger.info("=" * 60)
    
    # Load and prepare data
    X, y, label_encoders = load_and_prepare_data()
    
    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )  # 0.25 * 0.8 = 0.2 of total data for validation
    
    # Convert back to DataFrames for feature pipeline compatibility
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_val = pd.DataFrame(X_val, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.Series(y_train)
    y_val = pd.Series(y_val)
    y_test = pd.Series(y_test)
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Validation: {X_val.shape[0]} samples")
    logger.info(f"  Test: {X_test.shape[0]} samples")
    
    # Create feature engineering pipeline - disabled for now due to compatibility issues
    feature_pipeline = None
    logger.info("Feature pipeline disabled for compatibility")
    
    # Train base models
    base_models = train_base_models(X_train, y_train, X_val, y_val, feature_pipeline)
    
    # Train ensemble models
    ensemble_models = train_ensemble_models(X_train, y_train, X_val, y_val, feature_pipeline)
    
    # Combine all models
    all_models = {**base_models, **ensemble_models}
    
    logger.info(f"\nTrained {len(all_models)} models:")
    for name in all_models.keys():
        logger.info(f"  - {name}")
    
    # Comprehensive evaluation
    evaluator = comprehensive_evaluation(all_models, X_test, y_test, X_train, y_train)
    
    # Check accuracy target
    accuracy_results = check_accuracy_target(all_models, target_accuracy=0.90)
    
    # Save best models
    save_best_models(all_models, target_accuracy=0.90)
    
    logger.info("=" * 60)
    logger.info("Ensemble model training pipeline completed!")
    logger.info(f"Check 'evaluation_results/' for detailed analysis")
    logger.info(f"Check 'best_models/' for saved high-performing models")


if __name__ == "__main__":
    start_time = time.time()
    
    try:
        main()
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {total_time:.2f} seconds")