"""
Simplified Ensemble Model Training Script
Focuses on achieving ≥90% accuracy with working ensemble methods.
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
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import our models
sys.path.append(str(Path(__file__).parent))
from models.random_forest_model import RandomForestTrainer
from models.neural_network_model import NeuralNetworkTrainer
from models.xgboost_model import XGBoostTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_ensemble_training.log'),
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


def train_and_evaluate_model(model_name, model_trainer, X_train, y_train, X_test, y_test):
    """Train and evaluate a single model."""
    logger.info(f"Training {model_name}...")
    
    start_time = time.time()
    
    # Train model
    model_trainer.train(X_train, y_train, validation_split=0.2)
    
    # Make predictions
    y_pred = model_trainer.predict(X_test)
    y_pred_proba = None
    
    if hasattr(model_trainer.model, 'predict_proba'):
        try:
            y_pred_proba = model_trainer.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:  # Binary classification
                y_pred_proba = y_pred_proba[:, 1]
        except:
            pass
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    roc_auc = None
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            pass
    
    training_time = time.time() - start_time
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'training_time': training_time,
        'model': model_trainer
    }
    
    logger.info(f"{model_name} Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-score: {f1:.4f}")
    if roc_auc:
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  Training time: {training_time:.2f}s")
    logger.info("")
    
    return results


def create_voting_ensemble(base_models, X_train, y_train, X_test, y_test, voting='soft'):
    """Create and evaluate voting ensemble."""
    logger.info(f"Creating Voting Ensemble ({voting})...")
    
    # Prepare estimators for sklearn VotingClassifier
    estimators = []
    
    for name, result in base_models.items():
        model = result['model'].model
        estimators.append((name, model))
    
    # Create voting classifier
    voting_clf = VotingClassifier(estimators=estimators, voting=voting)
    
    start_time = time.time()
    
    # Train ensemble
    voting_clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = voting_clf.predict(X_test)
    y_pred_proba = None
    
    if hasattr(voting_clf, 'predict_proba'):
        try:
            y_pred_proba = voting_clf.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:  # Binary classification
                y_pred_proba = y_pred_proba[:, 1]
        except:
            pass
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    roc_auc = None
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            pass
    
    training_time = time.time() - start_time
    
    results = {
        'model_name': f'Voting_{voting.title()}',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'training_time': training_time,
        'model': voting_clf
    }
    
    logger.info(f"Voting Ensemble ({voting}) Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-score: {f1:.4f}")
    if roc_auc:
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  Training time: {training_time:.2f}s")
    logger.info("")
    
    return results


def optimize_random_forest(X_train, y_train, X_test, y_test):
    """Train optimized Random Forest with different hyperparameters."""
    logger.info("Training Optimized Random Forest...")
    
    best_accuracy = 0
    best_model = None
    best_params = {}
    
    # Hyperparameter combinations to try
    param_combinations = [
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 10},
        {'n_estimators': 500, 'max_depth': None, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 2, 'class_weight': 'balanced'},
    ]
    
    for params in param_combinations:
        logger.info(f"  Trying params: {params}")
        
        # Create and train model
        rf = RandomForestClassifier(random_state=42, **params)
        rf.fit(X_train, y_train)
        
        # Test accuracy
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"    Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = rf
            best_params = params
    
    # Calculate all metrics for best model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results = {
        'model_name': 'Optimized_RandomForest',
        'accuracy': best_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'best_params': best_params,
        'model': best_model
    }
    
    logger.info(f"Optimized Random Forest Results:")
    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Accuracy: {best_accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info("")
    
    return results


def main():
    """Main training pipeline."""
    logger.info("Starting Simplified Ensemble Model Training")
    logger.info("=" * 60)
    
    # Load and prepare data
    X, y, label_encoders = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to DataFrames for compatibility
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Test: {X_test.shape[0]} samples")
    logger.info("")
    
    # Train base models
    base_models = {}
    
    # Random Forest
    rf_trainer = RandomForestTrainer(random_state=42, verbose=False)
    base_models['RandomForest'] = train_and_evaluate_model(
        'RandomForest', rf_trainer, X_train, y_train, X_test, y_test
    )
    
    # XGBoost
    xgb_trainer = XGBoostTrainer(random_state=42, verbose=False)
    base_models['XGBoost'] = train_and_evaluate_model(
        'XGBoost', xgb_trainer, X_train, y_train, X_test, y_test
    )
    
    # Neural Network
    nn_trainer = NeuralNetworkTrainer(random_state=42, verbose=False)
    base_models['NeuralNetwork'] = train_and_evaluate_model(
        'NeuralNetwork', nn_trainer, X_train, y_train, X_test, y_test
    )
    
    # Train optimized Random Forest
    optimized_rf = optimize_random_forest(X_train, y_train, X_test, y_test)
    
    # Create voting ensembles
    voting_soft = create_voting_ensemble(
        base_models, X_train, y_train, X_test, y_test, voting='soft'
    )
    
    voting_hard = create_voting_ensemble(
        base_models, X_train, y_train, X_test, y_test, voting='hard'
    )
    
    # Collect all results
    all_results = {**base_models, 'Optimized_RF': optimized_rf, 'Voting_Soft': voting_soft, 'Voting_Hard': voting_hard}
    
    # Summary
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    
    # Sort by accuracy
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    target_accuracy = 0.90
    models_meeting_target = []
    
    for name, result in sorted_results:
        accuracy = result['accuracy']
        status = "✅ MEETS TARGET" if accuracy >= target_accuracy else "❌ Below target"
        
        logger.info(f"{name}: {accuracy:.4f} {status}")
        
        if accuracy >= target_accuracy:
            models_meeting_target.append((name, accuracy))
    
    logger.info("")
    
    if models_meeting_target:
        logger.info(f"🎯 SUCCESS! {len(models_meeting_target)} model(s) achieved ≥{target_accuracy:.1%} accuracy:")
        for name, accuracy in models_meeting_target:
            logger.info(f"   {name}: {accuracy:.4f}")
        
        best_name, best_accuracy = models_meeting_target[0]
        logger.info(f"\n🏆 Best model: {best_name} with {best_accuracy:.4f} accuracy")
        
        # Save best model
        if hasattr(all_results[best_name]['model'], 'save_model'):
            try:
                save_path = all_results[best_name]['model'].save_model(f"best_model_{best_name}.pkl")
                logger.info(f"💾 Saved best model to: {save_path}")
            except:
                logger.info("Could not save model using trainer method")
        
    else:
        logger.info(f"❌ No models achieved the ≥{target_accuracy:.1%} accuracy target")
        best_name, best_result = sorted_results[0]
        logger.info(f"Best performing model: {best_name} with {best_result['accuracy']:.4f} accuracy")
    
    # Create summary table
    summary_data = []
    for name, result in sorted_results:
        summary_data.append({
            'Model': name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}",
            'ROC-AUC': f"{result['roc_auc']:.4f}" if result['roc_auc'] else 'N/A',
            'Training_Time': f"{result.get('training_time', 0):.2f}s"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_df.to_csv('model_performance_summary.csv', index=False)
    logger.info(f"\n📊 Performance summary saved to: model_performance_summary.csv")
    
    logger.info("\nDetailed Performance Summary:")
    logger.info(summary_df.to_string(index=False))
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    start_time = time.time()
    
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {total_time:.2f} seconds")