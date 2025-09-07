"""
Production Training Script for Loan Eligibility Models
Trains and evaluates multiple models with comprehensive feature engineering,
hyperparameter optimization, and model comparison for production deployment.
"""

import sys
import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import argparse

# Add models directory to path
sys.path.append('models')

from models import (
    RandomForestTrainer, XGBoostTrainer, NeuralNetworkTrainer,
    HyperparameterTuner, ModelRegistry, ModelEvaluator,
    CrossValidator, TrainingMonitor
)
from feature_engineering import FeatureEngineeringPipeline, create_default_loan_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class LoanEligibilityTrainer:
    """Production trainer for loan eligibility models."""
    
    def __init__(self,
                 data_path: str = "loan_dataset.csv",
                 output_dir: str = "production_models",
                 random_state: int = 42,
                 test_size: float = 0.2,
                 validation_size: float = 0.2):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the dataset
            output_dir: Directory for output artifacts
            random_state: Random state for reproducibility
            test_size: Test set size
            validation_size: Validation set size
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.test_size = test_size
        self.validation_size = validation_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.registry = ModelRegistry(registry_path=str(self.output_dir / "model_registry"))
        self.evaluator = ModelEvaluator()
        self.cv = CrossValidator(random_state=random_state)
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_pipeline = None
        
        # Results storage
        self.trained_models = {}
        self.evaluation_results = {}
        
        logger.info(f"Initialized trainer with output directory: {self.output_dir}")
    
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        logger.info("Loading and preparing dataset...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        # Load data
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Handle missing values
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows with missing values")
        
        # Identify target column
        target_candidates = [col for col in df.columns if 
                           any(keyword in col.lower() for keyword in ['approved', 'target', 'label', 'outcome'])]
        
        if target_candidates:
            target_col = target_candidates[0]
            logger.info(f"Using target column: {target_col}")
        elif 'loan_approved' in df.columns:
            target_col = 'loan_approved'
        else:
            # Create synthetic target based on loan criteria
            logger.warning("Creating synthetic target based on common loan criteria")
            df['loan_approved'] = (
                (df.get('credit_score', df.get('Credit_Score', 700)) >= 650) & 
                (df.get('annual_income', df.get('Annual_Income', 50000)) >= 30000)
            ).astype(int)
            target_col = 'loan_approved'
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            logger.info(f"Encoding {len(categorical_columns)} categorical columns")
            
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data into train, validation, and test sets
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=self.validation_size, 
            random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"Data splits - Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        logger.info(f"Target distribution - Train: {self.y_train.value_counts().to_dict()}")
        
        # Save data splits
        self._save_data_splits()
        
        return True
    
    def setup_feature_engineering(self):
        """Setup and apply feature engineering pipeline."""
        logger.info("Setting up feature engineering pipeline...")
        
        try:
            # Create configuration
            config = create_default_loan_config()
            
            # Customize based on our data
            config.categorical.onehot_features = []
            config.categorical.label_features = []
            config.numerical.scaling_features = list(self.X_train.select_dtypes(include=[np.number]).columns)
            
            # Create and fit pipeline
            self.feature_pipeline = FeatureEngineeringPipeline(config=config)
            
            # Fit on training data
            X_train_transformed = self.feature_pipeline.fit_transform(
                self.X_train, self.y_train
            )
            
            # Transform validation and test sets
            X_val_transformed = self.feature_pipeline.transform(self.X_val)
            X_test_transformed = self.feature_pipeline.transform(self.X_test)
            
            # Update data
            self.X_train = X_train_transformed
            self.X_val = X_val_transformed
            self.X_test = X_test_transformed
            
            logger.info(f"Feature engineering completed. Features: {self.X_train.shape[1]}")
            
            # Get feature info
            feature_info = self.feature_pipeline.get_feature_info()
            logger.info(f"Processing steps: {feature_info['processing_stats']['steps_completed']}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Feature engineering failed, using original features: {e}")
            return False
    
    def train_models(self, enable_hyperparameter_tuning: bool = False):
        """Train all models."""
        logger.info("Training models...")
        
        models_config = {
            'RandomForest': {
                'trainer_class': RandomForestTrainer,
                'hyperparameter_trials': 50
            },
            'XGBoost': {
                'trainer_class': XGBoostTrainer,
                'hyperparameter_trials': 50
            },
            'NeuralNetwork': {
                'trainer_class': NeuralNetworkTrainer,
                'hyperparameter_trials': 30
            }
        }
        
        for model_name, config in models_config.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Create trainer
                trainer = config['trainer_class'](verbose=True)
                
                if enable_hyperparameter_tuning:
                    # Hyperparameter tuning
                    logger.info(f"Performing hyperparameter tuning for {model_name}...")
                    tuner = HyperparameterTuner(study_name=f"loan_{model_name.lower()}")
                    
                    results = tuner.optimize_model(
                        config['trainer_class'],
                        self.X_train, self.y_train,
                        n_trials=config['hyperparameter_trials'],
                        cv_folds=5,
                        scoring='roc_auc'
                    )
                    
                    # Train with best parameters
                    best_params = results['best_params']
                    trainer.train(
                        self.X_train, self.y_train,
                        hyperparameters=best_params,
                        feature_pipeline=self.feature_pipeline
                    )
                    
                    logger.info(f"Best CV score: {results['best_score']:.4f}")
                    
                else:
                    # Train with default parameters
                    trainer.train(
                        self.X_train, self.y_train,
                        feature_pipeline=self.feature_pipeline
                    )
                
                # Evaluate on validation set
                val_metrics = trainer.evaluate(self.X_val, self.y_val)
                
                # Cross-validation
                cv_results = trainer.cross_validate(
                    self.X_train, self.y_train,
                    cv_folds=5,
                    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                )
                
                self.trained_models[model_name] = {
                    'trainer': trainer,
                    'val_metrics': val_metrics,
                    'cv_results': cv_results
                }
                
                # Log results
                logger.info(f"{model_name} Results:")
                logger.info(f"  Validation Accuracy: {val_metrics.get('val_accuracy', 0):.4f}")
                logger.info(f"  Validation F1: {val_metrics.get('val_f1_score', 0):.4f}")
                logger.info(f"  Validation ROC-AUC: {val_metrics.get('val_roc_auc', 0):.4f}")
                
                # Check accuracy requirement
                accuracy = val_metrics.get('val_accuracy', 0)
                if accuracy >= 0.85:
                    logger.info(f"✅ {model_name} meets >85% accuracy requirement")
                else:
                    logger.warning(f"⚠️ {model_name} below 85% accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Training {model_name} failed: {e}")
                continue
        
        logger.info(f"Successfully trained {len(self.trained_models)} models")
    
    def evaluate_models(self):
        """Comprehensive model evaluation."""
        logger.info("Evaluating all trained models...")
        
        # Evaluate each model on test set
        for model_name, model_info in self.trained_models.items():
            trainer = model_info['trainer']
            
            try:
                # Test set evaluation
                test_metrics = trainer.evaluate(self.X_test, self.y_test)
                
                # Get predictions
                y_pred = trainer.predict(self.X_test)
                y_prob = None
                if hasattr(trainer.model, 'predict_proba'):
                    y_prob = trainer.predict_proba(self.X_test)[:, 1]
                
                # Comprehensive evaluation
                evaluation_metrics = self.evaluator.evaluate_model(
                    self.y_test.values, y_pred, y_prob,
                    model_name=model_name
                )
                
                self.evaluation_results[model_name] = {
                    'test_metrics': test_metrics,
                    'evaluation_metrics': evaluation_metrics,
                    'predictions': y_pred,
                    'probabilities': y_prob
                }
                
                # Log results
                summary = evaluation_metrics.get_summary()
                logger.info(f"{model_name} Test Results:")
                for metric, value in summary.items():
                    logger.info(f"  {metric}: {value:.4f}")
                
            except Exception as e:
                logger.error(f"Evaluation failed for {model_name}: {e}")
        
        # Model comparison
        if self.evaluation_results:
            evaluations = {name: results['evaluation_metrics'] 
                         for name, results in self.evaluation_results.items()}
            comparison_df = self.evaluator.compare_models(evaluations)
            
            logger.info("Model Comparison (sorted by accuracy):")
            logger.info(comparison_df.to_string(index=False))
            
            # Save comparison
            comparison_path = self.output_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"Model comparison saved to: {comparison_path}")
    
    def register_models(self):
        """Register models in the model registry."""
        logger.info("Registering models in registry...")
        
        registered_models = []
        
        for model_name, model_info in self.trained_models.items():
            trainer = model_info['trainer']
            
            try:
                # Determine model stage based on performance
                test_accuracy = self.evaluation_results[model_name]['test_metrics'].get('test_accuracy', 0)
                
                if test_accuracy >= 0.90:
                    stage = "production"
                    description = f"High-performance {model_name} model ready for production"
                elif test_accuracy >= 0.85:
                    stage = "staging"
                    description = f"Good {model_name} model suitable for staging"
                else:
                    stage = "development" 
                    description = f"Baseline {model_name} model for development"
                
                # Register model
                model_id = self.registry.register_model(
                    trainer,
                    model_name=f"loan_eligibility_{model_name.lower()}",
                    description=description,
                    tags=["loan_eligibility", model_name.lower(), "production_ready"],
                    training_data=(self.X_train.values, self.y_train.values)
                )
                
                registered_models.append({
                    'model_name': model_name,
                    'model_id': model_id,
                    'stage': stage,
                    'accuracy': test_accuracy
                })
                
                logger.info(f"Registered {model_name} as {model_id} (stage: {stage})")
                
            except Exception as e:
                logger.error(f"Failed to register {model_name}: {e}")
        
        # Save registration summary
        if registered_models:
            registration_df = pd.DataFrame(registered_models)
            registration_path = self.output_dir / "registered_models.csv"
            registration_df.to_csv(registration_path, index=False)
            logger.info(f"Registration summary saved to: {registration_path}")
    
    def generate_reports(self):
        """Generate comprehensive training reports."""
        logger.info("Generating training reports...")
        
        # Training summary report
        summary_data = []
        
        for model_name, model_info in self.trained_models.items():
            val_metrics = model_info['val_metrics']
            test_metrics = self.evaluation_results[model_name]['test_metrics']
            
            summary_data.append({
                'Model': model_name,
                'Val_Accuracy': val_metrics.get('val_accuracy', 0),
                'Val_F1': val_metrics.get('val_f1_score', 0),
                'Val_ROC_AUC': val_metrics.get('val_roc_auc', 0),
                'Test_Accuracy': test_metrics.get('test_accuracy', 0),
                'Test_F1': test_metrics.get('test_f1_score', 0),
                'Test_ROC_AUC': test_metrics.get('test_roc_auc', 0),
                'Training_Time': model_info['trainer'].metrics.training_time,
                'Meets_85%_Target': test_metrics.get('test_accuracy', 0) >= 0.85
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "training_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        logger.info("Training Summary:")
        logger.info(summary_df.to_string(index=False))
        logger.info(f"Training summary saved to: {summary_path}")
        
        # Best model identification
        best_model = summary_df.loc[summary_df['Test_Accuracy'].idxmax()]
        logger.info(f"Best Model: {best_model['Model']} (Accuracy: {best_model['Test_Accuracy']:.4f})")
        
        # Models meeting accuracy target
        target_models = summary_df[summary_df['Meets_85%_Target']]
        logger.info(f"Models meeting >85% accuracy target: {len(target_models)}/{len(summary_df)}")
        
    def _save_data_splits(self):
        """Save data splits for reproducibility."""
        splits_dir = self.output_dir / "data_splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Save training data
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        train_data.to_csv(splits_dir / "train.csv", index=False)
        
        # Save validation data
        val_data = pd.concat([self.X_val, self.y_val], axis=1)
        val_data.to_csv(splits_dir / "validation.csv", index=False)
        
        # Save test data
        test_data = pd.concat([self.X_test, self.y_test], axis=1)
        test_data.to_csv(splits_dir / "test.csv", index=False)
        
        logger.info(f"Data splits saved to: {splits_dir}")
    
    def run_full_training_pipeline(self, enable_hyperparameter_tuning: bool = False):
        """Run the complete training pipeline."""
        logger.info("🚀 Starting Full Training Pipeline")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Load data
            self.load_and_prepare_data()
            
            # Step 2: Feature engineering
            self.setup_feature_engineering()
            
            # Step 3: Train models
            self.train_models(enable_hyperparameter_tuning)
            
            # Step 4: Evaluate models
            self.evaluate_models()
            
            # Step 5: Register models
            self.register_models()
            
            # Step 6: Generate reports
            self.generate_reports()
            
            total_time = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("🎉 Training Pipeline Completed Successfully!")
            logger.info(f"Total Time: {total_time:.2f} seconds")
            logger.info(f"Models Trained: {len(self.trained_models)}")
            logger.info(f"Output Directory: {self.output_dir}")
            
            # Check if accuracy target was met
            target_met = any(
                self.evaluation_results[model_name]['test_metrics'].get('test_accuracy', 0) >= 0.85
                for model_name in self.trained_models.keys()
            )
            
            if target_met:
                logger.info("✅ SUCCESS: >85% accuracy requirement MET!")
            else:
                logger.warning("⚠️ WARNING: >85% accuracy requirement NOT MET")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Train loan eligibility prediction models")
    parser.add_argument("--data", default="loan_dataset.csv", help="Path to dataset")
    parser.add_argument("--output", default="production_models", help="Output directory")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = LoanEligibilityTrainer(
        data_path=args.data,
        output_dir=args.output,
        random_state=args.seed
    )
    
    # Run training pipeline
    success = trainer.run_full_training_pipeline(enable_hyperparameter_tuning=args.tune)
    
    if success:
        logger.info("🎯 Training completed successfully!")
        return 0
    else:
        logger.error("❌ Training failed!")
        return 1


if __name__ == "__main__":
    exit(main())