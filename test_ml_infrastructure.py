"""
Comprehensive ML Infrastructure Test Script
Tests all components of the ML training infrastructure and validates
the >85% accuracy requirement for loan eligibility prediction.
"""

import sys
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

# Add models directory to path
sys.path.append('models')

# Import our ML infrastructure
from models import (
    RandomForestTrainer, XGBoostTrainer, NeuralNetworkTrainer,
    HyperparameterTuner, ModelRegistry, TrainingMonitor,
    CrossValidator, ModelEvaluator
)
from feature_engineering import FeatureEngineeringPipeline, create_default_loan_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load and prepare the loan dataset."""
    logger.info("Loading and preparing loan dataset...")
    
    # Load data
    data_path = Path("loan_dataset.csv")
    if not data_path.exists():
        logger.error("Dataset not found. Please ensure loan_dataset.csv exists.")
        return None, None, None, None
    
    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded: {df.shape}")
    
    # Basic data preprocessing
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    if 'loan_approved' not in df.columns:
        # Find likely target column
        target_candidates = [col for col in df.columns if 'approved' in col.lower() or 'target' in col.lower()]
        if target_candidates:
            target_col = target_candidates[0]
            logger.info(f"Using target column: {target_col}")
        else:
            # Create synthetic target based on common loan criteria
            logger.warning("No clear target column found. Creating synthetic target.")
            df['loan_approved'] = (
                (df.get('credit_score', 700) >= 650) & 
                (df.get('annual_income', 50000) >= 30000) &
                (df.get('debt_to_income_ratio', 0.3) <= 0.4)
            ).astype(int)
            target_col = 'loan_approved'
    else:
        target_col = 'loan_approved'
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical variables for basic models
    X_encoded = X.copy()
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def test_feature_engineering(X_train, y_train):
    """Test feature engineering pipeline."""
    logger.info("=== Testing Feature Engineering Pipeline ===")
    
    try:
        # Create and configure pipeline
        config = create_default_loan_config()
        pipeline = FeatureEngineeringPipeline(config=config)
        
        # Fit and transform
        X_transformed = pipeline.fit_transform(X_train, y_train)
        
        logger.info(f"Original features: {X_train.shape[1]}")
        logger.info(f"Engineered features: {X_transformed.shape[1]}")
        
        # Get feature info
        feature_info = pipeline.get_feature_info()
        logger.info(f"Processing steps completed: {feature_info['processing_stats']['steps_completed']}")
        
        return pipeline, X_transformed
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {e}")
        return None, X_train


def test_basic_training(X_train, X_test, y_train, y_test):
    """Test basic model training."""
    logger.info("=== Testing Basic Model Training ===")
    
    results = {}
    models_to_test = [
        ("RandomForest", RandomForestTrainer),
        ("XGBoost", XGBoostTrainer),
        ("NeuralNetwork", NeuralNetworkTrainer)
    ]
    
    for model_name, trainer_class in models_to_test:
        try:
            logger.info(f"Training {model_name}...")
            
            # Create and train model
            trainer = trainer_class(verbose=True)
            trainer.train(X_train, y_train, validation_split=0.2)
            
            # Evaluate on test set
            test_metrics = trainer.evaluate(X_test, y_test)
            
            # Perform cross-validation
            cv_metrics = trainer.cross_validate(X_train, y_train, cv_folds=3)
            
            results[model_name] = {
                'trainer': trainer,
                'test_metrics': test_metrics,
                'cv_metrics': cv_metrics,
                'training_time': trainer.metrics.training_time
            }
            
            # Log key metrics
            logger.info(f"{model_name} Results:")
            logger.info(f"  Test Accuracy: {test_metrics.get('test_accuracy', 0):.4f}")
            logger.info(f"  Test F1-Score: {test_metrics.get('test_f1_score', 0):.4f}")
            logger.info(f"  Test ROC-AUC: {test_metrics.get('test_roc_auc', 0):.4f}")
            logger.info(f"  Training Time: {results[model_name]['training_time']:.2f}s")
            
            # Check accuracy target
            accuracy = test_metrics.get('test_accuracy', 0)
            if accuracy >= 0.85:
                logger.info(f"✅ {model_name} meets >85% accuracy target: {accuracy:.4f}")
            else:
                logger.warning(f"⚠️ {model_name} below accuracy target: {accuracy:.4f}")
        
        except Exception as e:
            logger.error(f"Training {model_name} failed: {e}")
            continue
    
    return results


def test_hyperparameter_tuning(X_train, y_train):
    """Test hyperparameter tuning with Optuna."""
    logger.info("=== Testing Hyperparameter Tuning ===")
    
    try:
        # Test with RandomForest (faster than others)
        tuner = HyperparameterTuner(study_name="test_optimization")
        
        logger.info("Running hyperparameter optimization (limited trials for testing)...")
        results = tuner.optimize_model(
            RandomForestTrainer,
            X_train, y_train,
            n_trials=10,  # Limited for testing
            cv_folds=3,
            scoring='roc_auc'
        )
        
        logger.info(f"Best score: {results['best_score']:.4f}")
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Optimization time: {results['optimization_time']:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning test failed: {e}")
        return None


def test_model_registry(trained_models, X_train, y_train):
    """Test model registry and versioning."""
    logger.info("=== Testing Model Registry ===")
    
    try:
        # Create registry
        registry = ModelRegistry(registry_path="test_model_registry")
        
        # Register models
        model_ids = []
        for model_name, model_info in trained_models.items():
            trainer = model_info['trainer']
            
            # Register model
            model_id = registry.register_model(
                trainer,
                model_name=f"loan_eligibility_{model_name.lower()}",
                description=f"Test {model_name} model for loan eligibility",
                tags=["test", "loan_eligibility", model_name.lower()],
                training_data=(X_train.values, y_train.values)
            )
            
            model_ids.append(model_id)
            logger.info(f"Registered {model_name} as {model_id}")
        
        # List models
        models_df = registry.list_models()
        logger.info(f"Registry contains {len(models_df)} models")
        
        # Test loading a model
        if model_ids:
            test_model_id = model_ids[0]
            loaded_model = registry.get_model(test_model_id)
            logger.info(f"Successfully loaded model: {test_model_id}")
        
        # Get registry statistics
        stats = registry.get_registry_statistics()
        logger.info(f"Registry statistics: {stats}")
        
        return registry, model_ids
        
    except Exception as e:
        logger.error(f"Model registry test failed: {e}")
        return None, []


def test_training_monitor():
    """Test training monitor."""
    logger.info("=== Testing Training Monitor ===")
    
    try:
        # Create monitor
        monitor = TrainingMonitor(session_name="test_session")
        
        # Start session
        session_id = monitor.start_session(
            model_name="test_model",
            model_type="RandomForest",
            hyperparameters={'n_estimators': 100, 'max_depth': 10}
        )
        
        # Simulate training steps
        for step in range(10):
            metrics = {
                'accuracy': 0.7 + (step * 0.02),
                'loss': 0.5 - (step * 0.03),
                'f1_score': 0.65 + (step * 0.025)
            }
            
            should_continue = monitor.log_step(
                step=step,
                metrics=metrics,
                loss=metrics['loss'],
                duration_ms=100 + np.random.randint(0, 50)
            )
            
            if not should_continue:
                logger.info("Early stopping triggered")
                break
        
        # End session
        final_metrics = {'final_accuracy': 0.88, 'final_f1': 0.85}
        monitor.end_session(final_metrics=final_metrics)
        
        # Get summary
        summary = monitor.get_session_summary()
        logger.info(f"Training session summary: {summary}")
        
        return monitor
        
    except Exception as e:
        logger.error(f"Training monitor test failed: {e}")
        return None


def test_cross_validation(X_train, y_train):
    """Test cross-validation framework."""
    logger.info("=== Testing Cross-Validation Framework ===")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Create model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Create cross-validator
        cv = CrossValidator(random_state=42)
        
        # Test different CV strategies
        strategies = ['stratified', 'bootstrap']
        
        for strategy in strategies:
            logger.info(f"Testing {strategy} cross-validation...")
            
            results = cv.validate(
                model, X_train, y_train,
                cv_strategy=strategy,
                n_splits=3,
                scoring=['accuracy', 'f1', 'roc_auc'],
                verbose=False
            )
            
            summary = results.get_summary()
            logger.info(f"{strategy} CV Results:")
            for metric, score in summary['mean_scores'].items():
                std = summary['std_scores'][metric]
                logger.info(f"  {metric}: {score:.4f} ± {std:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Cross-validation test failed: {e}")
        return False


def test_model_evaluator(trained_models, X_test, y_test):
    """Test model evaluator."""
    logger.info("=== Testing Model Evaluator ===")
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator()
        
        # Evaluate each model
        evaluations = {}
        
        for model_name, model_info in trained_models.items():
            trainer = model_info['trainer']
            
            # Get predictions
            y_pred = trainer.predict(X_test)
            y_prob = trainer.predict_proba(X_test)[:, 1] if hasattr(trainer.model, 'predict_proba') else None
            
            # Evaluate
            metrics = evaluator.evaluate_model(
                y_test.values, y_pred, y_prob,
                model_name=model_name
            )
            
            evaluations[model_name] = metrics
            
            # Log summary
            summary = metrics.get_summary()
            logger.info(f"{model_name} Evaluation Summary:")
            for metric, value in summary.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Compare models
        comparison_df = evaluator.compare_models(evaluations)
        logger.info("Model Comparison:")
        logger.info(comparison_df.to_string())
        
        return evaluator, evaluations
        
    except Exception as e:
        logger.error(f"Model evaluator test failed: {e}")
        return None, {}


def run_comprehensive_test():
    """Run comprehensive test of the entire ML infrastructure."""
    logger.info("🚀 Starting Comprehensive ML Infrastructure Test")
    logger.info("=" * 60)
    
    start_time = time.time()
    test_results = {}
    
    try:
        # 1. Load and prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_data()
        if X_train is None:
            logger.error("Failed to load data. Exiting.")
            return
        
        test_results['data_loaded'] = True
        
        # 2. Test feature engineering
        feature_pipeline, X_train_processed = test_feature_engineering(X_train, y_train)
        test_results['feature_engineering'] = feature_pipeline is not None
        
        # Use processed features if available
        if X_train_processed is not None:
            X_test_processed = feature_pipeline.transform(X_test)
        else:
            X_train_processed, X_test_processed = X_train, X_test
        
        # 3. Test basic training
        trained_models = test_basic_training(X_train_processed, X_test_processed, y_train, y_test)
        test_results['basic_training'] = len(trained_models) > 0
        
        # Check if any model meets accuracy target
        accuracy_target_met = False
        for model_name, model_info in trained_models.items():
            accuracy = model_info['test_metrics'].get('test_accuracy', 0)
            if accuracy >= 0.85:
                accuracy_target_met = True
                break
        
        test_results['accuracy_target_met'] = accuracy_target_met
        
        # 4. Test hyperparameter tuning
        tuning_results = test_hyperparameter_tuning(X_train_processed, y_train)
        test_results['hyperparameter_tuning'] = tuning_results is not None
        
        # 5. Test model registry
        registry, model_ids = test_model_registry(trained_models, X_train_processed, y_train)
        test_results['model_registry'] = registry is not None
        
        # 6. Test training monitor
        monitor = test_training_monitor()
        test_results['training_monitor'] = monitor is not None
        
        # 7. Test cross-validation
        cv_success = test_cross_validation(X_train_processed, y_train)
        test_results['cross_validation'] = cv_success
        
        # 8. Test model evaluator
        evaluator, evaluations = test_model_evaluator(trained_models, X_test_processed, y_test)
        test_results['model_evaluator'] = evaluator is not None
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
    
    # Report results
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🎯 ML Infrastructure Test Results")
    logger.info("=" * 60)
    
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name:25}: {status}")
    
    # Overall summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info("-" * 60)
    logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
    logger.info(f"Total Time: {total_time:.2f} seconds")
    
    # Critical requirement check
    if test_results.get('accuracy_target_met', False):
        logger.info("🎉 SUCCESS: >85% accuracy requirement MET!")
    else:
        logger.warning("⚠️ WARNING: >85% accuracy requirement NOT MET")
    
    # Overall status
    critical_tests = ['data_loaded', 'basic_training', 'accuracy_target_met']
    critical_passed = all(test_results.get(test, False) for test in critical_tests)
    
    if critical_passed and passed_tests >= total_tests * 0.8:
        logger.info("🚀 OVERALL STATUS: INFRASTRUCTURE READY FOR PRODUCTION")
    else:
        logger.warning("⚠️ OVERALL STATUS: ISSUES FOUND - REVIEW REQUIRED")
    
    return test_results


def create_sample_usage_demo():
    """Create a sample usage demonstration."""
    logger.info("=== Creating Sample Usage Demo ===")
    
    demo_code = '''
# Sample usage of the ML Training Infrastructure

from models import RandomForestTrainer, XGBoostTrainer, ModelRegistry, HyperparameterTuner
from feature_engineering import FeatureEngineeringPipeline, create_default_loan_config

# 1. Load and prepare your data
# X_train, X_test, y_train, y_test = load_your_data()

# 2. Feature engineering
config = create_default_loan_config()
pipeline = FeatureEngineeringPipeline(config=config)
X_train_processed = pipeline.fit_transform(X_train, y_train)
X_test_processed = pipeline.transform(X_test)

# 3. Train models
rf_trainer = RandomForestTrainer()
rf_trainer.train(X_train_processed, y_train)

# 4. Hyperparameter tuning
tuner = HyperparameterTuner()
best_results = tuner.optimize_model(RandomForestTrainer, X_train_processed, y_train)

# 5. Model registry
registry = ModelRegistry()
model_id = registry.register_model(rf_trainer, "loan_eligibility_rf")

# 6. Load model for inference
loaded_model = registry.get_model(model_id)
predictions = loaded_model.predict(X_new)
'''
    
    demo_file = Path("sample_ml_usage.py")
    with open(demo_file, 'w') as f:
        f.write(demo_code)
    
    logger.info(f"Sample usage demo saved to: {demo_file}")


if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    # Create sample usage demo
    create_sample_usage_demo()
    
    logger.info("🏁 ML Infrastructure test completed!")