"""
Example usage of the Feature Engineering Pipeline for Loan Eligibility Prediction.

This script demonstrates how to use the comprehensive feature engineering pipeline
with the loan dataset to prepare data for machine learning models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from feature_engineering import (
    FeatureEngineeringPipeline,
    create_default_loan_config,
    FeaturePipelineConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_loan_data():
    """Load the loan dataset."""
    try:
        data_path = Path("loan_dataset.csv")
        if not data_path.exists():
            logger.error(f"Dataset not found at {data_path}")
            return None
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None


def demonstrate_basic_usage():
    """Demonstrate basic pipeline usage with default configuration."""
    logger.info("=== BASIC USAGE DEMONSTRATION ===")
    
    # Load data
    df = load_loan_data()
    if df is None:
        return
    
    # Separate features and target
    X = df.drop(columns=['loan_approved'])
    y = df['loan_approved']
    
    # Create pipeline with default loan configuration
    config = create_default_loan_config()
    pipeline = FeatureEngineeringPipeline(config=config)
    
    # Fit and transform
    logger.info("Fitting and transforming data...")
    X_transformed = pipeline.fit_transform(X, y)
    
    logger.info(f"Original features: {X.shape[1]}")
    logger.info(f"Engineered features: {X_transformed.shape[1]}")
    
    # Get feature information
    feature_info = pipeline.get_feature_info()
    logger.info("Feature engineering steps completed:")
    for step in feature_info['processing_stats']['steps_completed']:
        logger.info(f"  ✓ {step}")
    
    # Save pipeline for later use
    pipeline_path = Path("trained_feature_pipeline.pkl")
    pipeline.save_pipeline(pipeline_path)
    logger.info(f"Pipeline saved to {pipeline_path}")
    
    return X_transformed, y, pipeline


def demonstrate_custom_configuration():
    """Demonstrate custom pipeline configuration."""
    logger.info("=== CUSTOM CONFIGURATION DEMONSTRATION ===")
    
    # Load data
    df = load_loan_data()
    if df is None:
        return
    
    X = df.drop(columns=['loan_approved'])
    y = df['loan_approved']
    
    # Create custom configuration
    config = FeaturePipelineConfig(
        target_column='loan_approved',
        random_state=42,
        verbose=True
    )
    
    # Customize categorical encoding
    config.categorical.onehot_features = ['gender', 'marital_status', 'area_type']
    config.categorical.target_features = ['education', 'employment_status', 'loan_purpose']
    config.categorical.label_features = ['state']
    
    # Customize numerical preprocessing
    config.numerical.scaling_features = [
        'age', 'years_employed', 'annual_income', 'credit_score', 
        'loan_amount', 'property_value'
    ]
    config.numerical.scaling_method = 'robust'
    config.numerical.outlier_features = ['annual_income', 'loan_amount', 'property_value']
    config.numerical.outlier_action = 'clip'
    
    # Customize feature interactions
    config.interactions.ratio_pairs = [
        ('loan_amount', 'annual_income'),
        ('existing_debt', 'annual_income'),
        ('monthly_debt_payments', 'monthly_income'),
        ('property_value', 'loan_amount')
    ]
    config.interactions.polynomial_features = ['credit_score', 'debt_to_income_ratio']
    config.interactions.polynomial_degree = 2
    
    # Customize dimensionality reduction
    config.dimensionality.feature_selection_enabled = True
    config.dimensionality.selection_method = 'mutual_info'
    config.dimensionality.selection_k = 20
    config.dimensionality.variance_threshold = 0.01
    
    # Create and fit pipeline
    pipeline = FeatureEngineeringPipeline(config=config)
    X_transformed = pipeline.fit_transform(X, y)
    
    logger.info(f"Custom pipeline - Original: {X.shape[1]}, Engineered: {X_transformed.shape[1]}")
    
    return X_transformed, y, pipeline


def demonstrate_model_training(X_transformed, y):
    """Demonstrate model training with engineered features."""
    logger.info("=== MODEL TRAINING DEMONSTRATION ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"\nTraining {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[model_name] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  AUC Score: {auc_score:.4f}")
        
        # Detailed classification report
        report = classification_report(y_test, y_pred, target_names=['Denied', 'Approved'])
        logger.info(f"Classification Report:\n{report}")
    
    return results


def demonstrate_pipeline_validation():
    """Demonstrate pipeline validation capabilities."""
    logger.info("=== PIPELINE VALIDATION DEMONSTRATION ===")
    
    # Load data
    df = load_loan_data()
    if df is None:
        return
    
    X = df.drop(columns=['loan_approved'])
    y = df['loan_approved']
    
    # Create pipeline
    config = create_default_loan_config()
    pipeline = FeatureEngineeringPipeline(config=config)
    
    # Validate pipeline
    validation_results = pipeline.validate_pipeline(X, y, test_size=0.3)
    
    logger.info("Pipeline Validation Results:")
    logger.info(f"  Status: {validation_results['status']}")
    
    if validation_results['status'] == 'success':
        metrics = validation_results['metrics']
        logger.info(f"  Fit Time: {metrics['fit_time']:.3f}s")
        logger.info(f"  Transform Time: {metrics['transform_time']:.3f}s")
        logger.info(f"  Input Shape: {metrics['input_shape']}")
        logger.info(f"  Output Shape: {metrics['output_shape']}")
        logger.info(f"  Feature Reduction: {metrics['feature_reduction_ratio']:.2%}")
        logger.info(f"  Memory Usage: {metrics['memory_usage_mb']:.2f} MB")
    
    if validation_results['warnings']:
        logger.warning("Validation Warnings:")
        for warning in validation_results['warnings']:
            logger.warning(f"  - {warning}")
    
    if validation_results['errors']:
        logger.error("Validation Errors:")
        for error in validation_results['errors']:
            logger.error(f"  - {error}")


def demonstrate_pipeline_reuse():
    """Demonstrate loading and reusing a saved pipeline."""
    logger.info("=== PIPELINE REUSE DEMONSTRATION ===")
    
    pipeline_path = Path("trained_feature_pipeline.pkl")
    if not pipeline_path.exists():
        logger.warning("No saved pipeline found. Run basic usage first.")
        return
    
    # Load data (simulate new data)
    df = load_loan_data()
    if df is None:
        return
    
    # Take a sample as "new data"
    new_data = df.drop(columns=['loan_approved']).sample(100, random_state=123)
    
    # Load saved pipeline
    logger.info("Loading saved pipeline...")
    pipeline = FeatureEngineeringPipeline.load_pipeline(pipeline_path)
    
    # Transform new data
    logger.info("Transforming new data...")
    new_data_transformed = pipeline.transform(new_data)
    
    logger.info(f"New data transformed: {new_data.shape} -> {new_data_transformed.shape}")
    
    # Get pipeline info
    feature_info = pipeline.get_feature_info()
    logger.info("Pipeline Information:")
    logger.info(f"  Input Features: {feature_info['input_features']}")
    logger.info(f"  Output Features: {feature_info['output_features']}")
    logger.info(f"  Processing Steps: {len(feature_info['processing_stats']['steps_completed'])}")


def main():
    """Main demonstration function."""
    logger.info("🚀 Feature Engineering Pipeline Demonstration")
    logger.info("=" * 60)
    
    try:
        # Basic usage
        X_transformed, y, pipeline = demonstrate_basic_usage()
        
        if X_transformed is not None:
            # Model training
            demonstrate_model_training(X_transformed, y)
            
            # Custom configuration
            demonstrate_custom_configuration()
            
            # Pipeline validation
            demonstrate_pipeline_validation()
            
            # Pipeline reuse
            demonstrate_pipeline_reuse()
            
            logger.info("\n✅ All demonstrations completed successfully!")
        
        else:
            logger.error("Could not load data. Please ensure loan_dataset.csv exists.")
    
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()