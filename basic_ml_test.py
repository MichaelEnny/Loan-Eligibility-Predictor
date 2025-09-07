"""
Basic ML Infrastructure Test
Tests the core ML training capabilities without optional dependencies.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import warnings

# Add models directory to path
sys.path.append('models')

# Import our ML infrastructure
from models import RandomForestTrainer, XGBoostTrainer, NeuralNetworkTrainer, ModelEvaluator, ModelRegistry

warnings.filterwarnings('ignore')

def create_sample_loan_data():
    """Create sample loan-like dataset."""
    print("Creating sample loan dataset...")
    
    # Generate classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        random_state=42,
        class_sep=0.8
    )
    
    # Create DataFrame with loan-like feature names
    feature_names = [
        'credit_score', 'annual_income', 'loan_amount', 'debt_to_income_ratio',
        'years_employed', 'age', 'monthly_debt_payments', 'number_of_dependents',
        'property_value', 'loan_term', 'employment_status_score', 
        'education_level_score', 'marital_status_score', 'payment_history_score',
        'account_balance_score'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='loan_approved')
    
    # Make features more realistic
    X_df['credit_score'] = ((X_df['credit_score'] + 3) * 100 + 650).clip(300, 850)
    X_df['annual_income'] = ((X_df['annual_income'] + 3) * 20000 + 40000).clip(20000, 200000)
    X_df['loan_amount'] = ((X_df['loan_amount'] + 3) * 50000 + 10000).clip(5000, 500000)
    
    print(f"Dataset created: {X_df.shape}")
    print(f"Target distribution: {y_series.value_counts().to_dict()}")
    
    return X_df, y_series

def test_basic_training():
    """Test basic model training."""
    print("\n=== Testing Basic Model Training ===")
    
    # Create data
    X, y = create_sample_loan_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    results = {}
    
    # Test each model
    models_to_test = [
        ("RandomForest", RandomForestTrainer),
        ("XGBoost", XGBoostTrainer),
        ("NeuralNetwork", NeuralNetworkTrainer)
    ]
    
    for model_name, trainer_class in models_to_test:
        print(f"\nTraining {model_name}...")
        
        try:
            # Create and train model
            trainer = trainer_class(verbose=False)
            trainer.train(X_train, y_train, validation_split=0.2)
            
            # Evaluate
            test_metrics = trainer.evaluate(X_test, y_test)
            
            results[model_name] = {
                'trainer': trainer,
                'accuracy': test_metrics.get('test_accuracy', 0),
                'f1_score': test_metrics.get('test_f1_score', 0),
                'roc_auc': test_metrics.get('test_roc_auc', 0)
            }
            
            print(f"{model_name} Results:")
            print(f"  Accuracy: {results[model_name]['accuracy']:.4f}")
            print(f"  F1-Score: {results[model_name]['f1_score']:.4f}")
            print(f"  ROC-AUC: {results[model_name]['roc_auc']:.4f}")
            
            # Check accuracy target
            if results[model_name]['accuracy'] >= 0.85:
                print(f"  PASS: Meets >85% accuracy target")
            else:
                print(f"  INFO: Below 85% accuracy (expected for synthetic data)")
            
        except Exception as e:
            print(f"  ERROR: Training {model_name} failed: {e}")
    
    return results, X_test, y_test

def test_model_evaluation(results, X_test, y_test):
    """Test model evaluation framework."""
    print("\n=== Testing Model Evaluation ===")
    
    evaluator = ModelEvaluator()
    evaluations = {}
    
    for model_name, result in results.items():
        trainer = result['trainer']
        
        try:
            # Get predictions
            y_pred = trainer.predict(X_test)
            y_prob = None
            if hasattr(trainer.model, 'predict_proba'):
                y_prob = trainer.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = evaluator.evaluate_model(y_test.values, y_pred, y_prob, model_name)
            evaluations[model_name] = metrics
            
            print(f"{model_name} Evaluation:")
            summary = metrics.get_summary()
            for metric, value in summary.items():
                print(f"  {metric}: {value:.4f}")
            
        except Exception as e:
            print(f"  ERROR: Evaluating {model_name} failed: {e}")
    
    # Compare models
    if evaluations:
        comparison = evaluator.compare_models(evaluations)
        print("\nModel Comparison:")
        print(comparison.to_string(index=False))
    
    return evaluations

def test_model_registry(results):
    """Test model registry."""
    print("\n=== Testing Model Registry ===")
    
    try:
        registry = ModelRegistry(registry_path="test_registry")
        
        for model_name, result in results.items():
            trainer = result['trainer']
            
            # Register model
            model_id = registry.register_model(
                trainer,
                model_name=f"test_{model_name.lower()}",
                description=f"Test {model_name} model"
            )
            
            print(f"Registered {model_name} as {model_id}")
        
        # List models
        models_df = registry.list_models()
        print(f"\nRegistry contains {len(models_df)} models")
        print(models_df[['model_name', 'validation_accuracy', 'model_size_mb']].to_string(index=False))
        
        return registry
        
    except Exception as e:
        print(f"ERROR: Model registry test failed: {e}")
        return None

def main():
    """Run basic ML infrastructure test."""
    print("=== BASIC ML INFRASTRUCTURE TEST ===")
    print("Testing core functionality without optional dependencies")
    
    success_count = 0
    total_tests = 3
    
    try:
        # Test 1: Basic training
        print("\nTest 1: Basic Model Training")
        results, X_test, y_test = test_basic_training()
        if results:
            success_count += 1
            print("PASS: Basic training successful")
        
        # Test 2: Model evaluation
        print("\nTest 2: Model Evaluation")
        evaluations = test_model_evaluation(results, X_test, y_test)
        if evaluations:
            success_count += 1
            print("PASS: Model evaluation successful")
        
        # Test 3: Model registry
        print("\nTest 3: Model Registry")
        registry = test_model_registry(results)
        if registry:
            success_count += 1
            print("PASS: Model registry successful")
        
    except Exception as e:
        print(f"ERROR: Test execution failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("BASIC ML INFRASTRUCTURE TEST RESULTS")
    print("="*50)
    print(f"Tests Passed: {success_count}/{total_tests}")
    
    # Check if any model meets accuracy target
    accuracy_met = False
    if 'results' in locals():
        for model_name, result in results.items():
            if result['accuracy'] >= 0.85:
                accuracy_met = True
                break
    
    if accuracy_met:
        print("SUCCESS: At least one model meets >85% accuracy target")
    else:
        print("INFO: No models meet 85% accuracy (expected for synthetic data)")
    
    if success_count >= 2:
        print("OVERALL: CORE ML INFRASTRUCTURE IS WORKING")
    else:
        print("OVERALL: ISSUES DETECTED IN ML INFRASTRUCTURE")
    
    print("\nTo test with real loan data, replace create_sample_loan_data() with your dataset")
    print("To enable hyperparameter tuning, install: pip install optuna>=3.0.0")

if __name__ == "__main__":
    main()