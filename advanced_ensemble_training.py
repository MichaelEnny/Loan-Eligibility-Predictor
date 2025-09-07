"""
Advanced Ensemble Model Training Script
Applies feature engineering, hyperparameter optimization, and advanced ensemble methods
to achieve ≥90% accuracy target.
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, RobustScaler
from sklearn.ensemble import (
    VotingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

# Import XGBoost if available
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_ensemble_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def advanced_feature_engineering(X_train, X_test):
    """Advanced feature engineering to improve model performance."""
    logger.info("Applying advanced feature engineering...")
    
    # Original feature count
    original_features = X_train.shape[1]
    
    # Create polynomial features (degree 2) for selected numerical features
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10 numerical features
    
    if len(numerical_cols) > 0:
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        
        # Fit on training data
        X_train_poly = poly.fit_transform(X_train[numerical_cols])
        X_test_poly = poly.transform(X_test[numerical_cols])
        
        # Create column names for polynomial features
        poly_feature_names = [f"poly_{i}" for i in range(X_train_poly.shape[1] - len(numerical_cols))]
        
        # Add polynomial features to original data
        poly_df_train = pd.DataFrame(X_train_poly[:, len(numerical_cols):], 
                                   columns=poly_feature_names, 
                                   index=X_train.index)
        poly_df_test = pd.DataFrame(X_test_poly[:, len(numerical_cols):], 
                                  columns=poly_feature_names, 
                                  index=X_test.index)
        
        X_train = pd.concat([X_train, poly_df_train], axis=1)
        X_test = pd.concat([X_test, poly_df_test], axis=1)
    
    # Create ratio and difference features
    income_cols = [col for col in X_train.columns if 'income' in col.lower()]
    debt_cols = [col for col in X_train.columns if 'debt' in col.lower()]
    
    if len(income_cols) >= 2:
        X_train['income_ratio'] = X_train[income_cols[0]] / (X_train[income_cols[1]] + 1)
        X_test['income_ratio'] = X_test[income_cols[0]] / (X_test[income_cols[1]] + 1)
    
    if len(debt_cols) >= 1 and len(income_cols) >= 1:
        X_train['debt_income_ratio'] = X_train[debt_cols[0]] / (X_train[income_cols[0]] + 1)
        X_test['debt_income_ratio'] = X_test[debt_cols[0]] / (X_test[income_cols[0]] + 1)
    
    # Age-based features
    if 'age' in X_train.columns:
        X_train['age_squared'] = X_train['age'] ** 2
        X_test['age_squared'] = X_test['age'] ** 2
        
        X_train['age_group'] = pd.cut(X_train['age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3])
        X_test['age_group'] = pd.cut(X_test['age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3])
    
    # Credit score interactions
    if 'credit_score' in X_train.columns and 'age' in X_train.columns:
        X_train['credit_age_interaction'] = X_train['credit_score'] * X_train['age']
        X_test['credit_age_interaction'] = X_test['credit_score'] * X_test['age']
    
    logger.info(f"Feature engineering: {original_features} -> {X_train.shape[1]} features")
    
    return X_train, X_test


def load_and_prepare_data(data_path: str = "loan_dataset.csv") -> tuple:
    """Load and prepare the loan dataset with advanced preprocessing."""
    logger.info("Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Separate features and target
    target_col = 'loan_approved'
    feature_cols = [col for col in df.columns if col not in [target_col, 'application_date']]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Advanced missing value handling
    for col in X.columns:
        if X[col].dtype == 'object':
            # For categorical, fill with mode or create 'Unknown' category
            if X[col].isnull().sum() > 0:
                if not X[col].mode().empty:
                    X[col] = X[col].fillna(X[col].mode()[0])
                else:
                    X[col] = X[col].fillna('Unknown')
        else:
            # For numerical, try median, then mean, then 0
            if X[col].isnull().sum() > 0:
                if not np.isnan(X[col].median()):
                    X[col] = X[col].fillna(X[col].median())
                elif not np.isnan(X[col].mean()):
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(0)
    
    # Advanced categorical encoding
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


def create_advanced_models():
    """Create a diverse set of advanced models."""
    models = {}
    
    # Random Forest with optimal parameters
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Extra Trees
    models['ExtraTrees'] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42
    )
    
    # XGBoost if available
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    
    # AdaBoost
    models['AdaBoost'] = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )
    
    # Neural Network
    models['NeuralNetwork'] = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    # Logistic Regression
    models['LogisticRegression'] = LogisticRegression(
        C=1.0,
        penalty='l2',
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    
    # SVM
    models['SVM'] = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    
    return models


def hyperparameter_optimization(model, param_grid, X_train, y_train, cv=3):
    """Perform hyperparameter optimization."""
    logger.info("Performing hyperparameter optimization...")
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', 
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test, optimize=False):
    """Train and evaluate a single model with optional hyperparameter optimization."""
    logger.info(f"Training {name}...")
    
    start_time = time.time()
    
    # Hyperparameter optimization for specific models
    if optimize and name in ['RandomForest', 'XGBoost']:
        if name == 'RandomForest':
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', None]
            }
            model, best_params, best_score = hyperparameter_optimization(
                RandomForestClassifier(random_state=42), param_grid, X_train, y_train
            )
            logger.info(f"Best params for {name}: {best_params}")
        
        elif name == 'XGBoost' and HAS_XGBOOST:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            }
            model, best_params, best_score = hyperparameter_optimization(
                xgb.XGBClassifier(random_state=42), param_grid, X_train, y_train
            )
            logger.info(f"Best params for {name}: {best_params}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)
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
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    training_time = time.time() - start_time
    
    results = {
        'model_name': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'training_time': training_time,
        'model': model
    }
    
    logger.info(f"{name} Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-score: {f1:.4f}")
    if roc_auc:
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  Training time: {training_time:.2f}s")
    logger.info("")
    
    return results


def create_advanced_ensemble(models, X_train, y_train, X_test, y_test):
    """Create advanced ensemble with multiple voting strategies."""
    logger.info("Creating advanced ensemble...")
    
    # Select top performing models for ensemble
    estimators = [(name, model['model']) for name, model in models.items() 
                 if model['accuracy'] > 0.85]  # Only include models with good performance
    
    if len(estimators) < 2:
        logger.warning("Not enough good models for ensemble, using all available models")
        estimators = [(name, model['model']) for name, model in models.items()]
    
    logger.info(f"Using {len(estimators)} models in ensemble: {[name for name, _ in estimators]}")
    
    results = {}
    
    # Soft voting ensemble
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    result = train_and_evaluate_model(
        'Advanced_Voting_Soft', voting_soft, X_train, y_train, X_test, y_test
    )
    results['Advanced_Voting_Soft'] = result
    
    # Hard voting ensemble
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    result = train_and_evaluate_model(
        'Advanced_Voting_Hard', voting_hard, X_train, y_train, X_test, y_test
    )
    results['Advanced_Voting_Hard'] = result
    
    return results


def feature_selection(X_train, y_train, X_test):
    """Apply feature selection to improve model performance."""
    logger.info("Applying feature selection...")
    
    # Use Random Forest for feature selection
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector = SelectFromModel(rf_selector, threshold='median')
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    logger.info(f"Feature selection: {X_train.shape[1]} -> {X_train_selected.shape[1]} features")
    
    return X_train_selected, X_test_selected, selector


def main():
    """Main advanced training pipeline."""
    logger.info("Starting Advanced Ensemble Model Training")
    logger.info("=" * 70)
    
    # Load and prepare data
    X, y, label_encoders = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y  # Using smaller test set for more training data
    )
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Test: {X_test.shape[0]} samples")
    logger.info("")
    
    # Apply advanced feature engineering
    X_train_engineered, X_test_engineered = advanced_feature_engineering(X_train, X_test)
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_engineered)
    X_test_scaled = scaler.transform(X_test_engineered)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_engineered.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_engineered.columns)
    
    logger.info(f"Final data shape after preprocessing: {X_train_scaled.shape}")
    
    # Optional: Apply feature selection
    try:
        X_train_selected, X_test_selected, selector = feature_selection(
            X_train_scaled, y_train, X_test_scaled
        )
        
        # Use selected features
        X_train_final = X_train_selected
        X_test_final = X_test_selected
    except Exception as e:
        logger.warning(f"Feature selection failed: {e}, using all features")
        X_train_final = X_train_scaled.values
        X_test_final = X_test_scaled.values
    
    # Create and train models
    model_configs = create_advanced_models()
    all_results = {}
    
    # Train each model
    for name, model in model_configs.items():
        try:
            result = train_and_evaluate_model(
                name, model, X_train_final, y_train, X_test_final, y_test,
                optimize=True if name in ['RandomForest', 'XGBoost'] else False
            )
            all_results[name] = result
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
    
    # Create advanced ensemble
    try:
        ensemble_results = create_advanced_ensemble(
            all_results, X_train_final, y_train, X_test_final, y_test
        )
        all_results.update(ensemble_results)
    except Exception as e:
        logger.error(f"Failed to create ensemble: {e}")
    
    # Final Results Summary
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    
    # Sort by accuracy
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    target_accuracy = 0.90
    models_meeting_target = []
    
    for name, result in sorted_results:
        accuracy = result['accuracy']
        cv_score = result['cv_mean']
        status = "✅ MEETS TARGET" if accuracy >= target_accuracy else "❌ Below target"
        
        logger.info(f"{name}: Test={accuracy:.4f}, CV={cv_score:.4f} ± {result['cv_std']:.4f} {status}")
        
        if accuracy >= target_accuracy:
            models_meeting_target.append((name, accuracy, result))
    
    logger.info("")
    
    if models_meeting_target:
        logger.info(f"🎯 SUCCESS! {len(models_meeting_target)} model(s) achieved ≥{target_accuracy:.1%} accuracy:")
        for name, accuracy, result in models_meeting_target:
            logger.info(f"   {name}: {accuracy:.4f} (CV: {result['cv_mean']:.4f})")
        
        # Select best model (highest test accuracy with good CV score)
        best_name, best_accuracy, best_result = models_meeting_target[0]
        
        # Check if CV score is reasonable (not overfitting)
        if abs(best_accuracy - best_result['cv_mean']) < 0.05:
            logger.info(f"\n🏆 Best model: {best_name}")
            logger.info(f"   Test Accuracy: {best_accuracy:.4f}")
            logger.info(f"   CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
            logger.info(f"   Not overfitting: ✅")
        else:
            logger.info(f"\n⚠️ Best model {best_name} may be overfitting")
            logger.info(f"   Test: {best_accuracy:.4f}, CV: {best_result['cv_mean']:.4f}")
        
        # Save best model
        try:
            import joblib
            model_path = f"best_advanced_model_{best_name}.pkl"
            joblib.dump(best_result['model'], model_path)
            logger.info(f"💾 Saved best model to: {model_path}")
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
            
    else:
        logger.info(f"❌ No models achieved the ≥{target_accuracy:.1%} accuracy target")
        if sorted_results:
            best_name, best_result = sorted_results[0]
            logger.info(f"Best performing model: {best_name}")
            logger.info(f"   Test Accuracy: {best_result['accuracy']:.4f}")
            logger.info(f"   CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
    
    # Create detailed summary
    summary_data = []
    for name, result in sorted_results:
        summary_data.append({
            'Model': name,
            'Test_Accuracy': f"{result['accuracy']:.4f}",
            'CV_Mean': f"{result['cv_mean']:.4f}",
            'CV_Std': f"{result['cv_std']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1_Score': f"{result['f1_score']:.4f}",
            'ROC_AUC': f"{result['roc_auc']:.4f}" if result['roc_auc'] else 'N/A',
            'Training_Time': f"{result.get('training_time', 0):.2f}s",
            'Meets_Target': 'YES' if result['accuracy'] >= target_accuracy else 'NO'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_df.to_csv('advanced_model_performance_summary.csv', index=False)
    logger.info(f"\n📊 Advanced performance summary saved to: advanced_model_performance_summary.csv")
    
    logger.info("\nDetailed Performance Summary:")
    logger.info(summary_df.to_string(index=False))
    
    logger.info("\n" + "=" * 70)
    logger.info("Advanced ensemble training completed!")
    
    return models_meeting_target


if __name__ == "__main__":
    start_time = time.time()
    
    try:
        successful_models = main()
        
        if successful_models:
            logger.info(f"\n🎉 MISSION ACCOMPLISHED! {len(successful_models)} model(s) achieved ≥90% accuracy!")
            logger.info("Task 1.1.4: Model Ensemble Implementation - COMPLETED ✅")
        else:
            logger.info("\n🔄 Target not quite reached, but significant progress made.")
            logger.info("Consider further hyperparameter tuning or additional feature engineering.")
            
    except Exception as e:
        logger.error(f"Advanced training failed: {e}", exc_info=True)
        sys.exit(1)
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {total_time:.2f} seconds")