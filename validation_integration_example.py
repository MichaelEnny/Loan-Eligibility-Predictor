"""
Integration example showing how to use the validation framework with the existing ML pipeline.

Demonstrates seamless integration with feature engineering and model training workflows
to ensure data quality throughout the machine learning pipeline.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Tuple, Dict, Any

# Add validation package to path
sys.path.append(str(Path(__file__).parent))

from validation import (
    LoanDataSchema, DataValidator, validate_input, validate_schema, 
    require_fields, LoanBusinessRules, ValidationError
)

# Try to import existing ML components
try:
    from feature_engineering import FeatureEngineeringPipeline, create_default_loan_config
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    logging.warning("Feature engineering module not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValidatedLoanProcessor:
    """
    Loan data processor with integrated validation framework.
    
    This class demonstrates how to integrate validation at multiple points
    in the ML pipeline to ensure data quality and catch issues early.
    """
    
    def __init__(self, enable_business_rules: bool = True):
        """
        Initialize processor with validation framework.
        
        Args:
            enable_business_rules: Whether to enable business rule validation
        """
        self.schema = LoanDataSchema()
        self.validator = DataValidator(self.schema)
        self.business_rules = LoanBusinessRules() if enable_business_rules else None
        
        # Setup feature engineering if available
        if FEATURE_ENGINEERING_AVAILABLE:
            self.feature_config = create_default_loan_config()
            self.feature_pipeline = None
        
        logger.info("✅ Validated Loan Processor initialized")
    
    @validate_input(strict_mode=False, log_validation=True)
    @require_fields('age', 'annual_income', 'credit_score', 'loan_amount', strict_mode=False)
    def load_and_validate_data(self, data_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load loan data with comprehensive validation.
        
        Args:
            data_path: Path to loan dataset
            
        Returns:
            Tuple of (validated_data, validation_report)
        """
        logger.info(f"📂 Loading data from {data_path}")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Run comprehensive validation
            validation_result = self.validator.validate_all(
                df,
                include_schema=True,
                include_ranges=True,
                include_formats=False,  # Skip format validation for now
                include_business_rules=self.business_rules is not None
            )
            
            # Create validation report
            report = self._create_validation_report(df, validation_result)
            
            # Clean data based on validation results
            clean_df = self._clean_data(df, validation_result)
            
            logger.info(f"✅ Data validation complete. Clean dataset: {len(clean_df)} rows")
            
            return clean_df, report
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            raise
    
    def _create_validation_report(self, df: pd.DataFrame, result) -> Dict[str, Any]:
        """Create comprehensive validation report."""
        report = {
            'dataset_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            },
            'validation_summary': {
                'overall_valid': result.is_valid,
                'total_errors': len(result.errors),
                'total_warnings': len(result.warnings),
                'error_rate': len(result.errors) / len(df) if len(df) > 0 else 0
            },
            'field_validation': {},
            'business_rules': {},
            'data_quality_score': 0.0
        }
        
        # Field-level validation results
        for field_name in df.columns:
            if field_name in self.schema.fields:
                field_errors = [e for e in result.errors if e.get('field') == field_name]
                report['field_validation'][field_name] = {
                    'errors': len(field_errors),
                    'error_rate': len(field_errors) / len(df) if len(df) > 0 else 0,
                    'null_rate': df[field_name].isnull().sum() / len(df) if len(df) > 0 else 0
                }
        
        # Business rules validation
        if self.business_rules:
            business_result = self.business_rules.validate_all_rules(df)
            report['business_rules'] = {
                'rules_passed': len(business_result['passed_rules']),
                'rules_failed': len(business_result['failed_rules']),
                'total_violations': len(business_result['rule_violations']),
                'failed_rules': business_result['failed_rules']
            }
        
        # Calculate data quality score (0-100)
        total_possible_errors = len(df) * len(self.schema.required_fields)
        if total_possible_errors > 0:
            error_rate = len(result.errors) / total_possible_errors
            report['data_quality_score'] = max(0, (1 - error_rate) * 100)
        else:
            report['data_quality_score'] = 100.0
        
        return report
    
    def _clean_data(self, df: pd.DataFrame, validation_result) -> pd.DataFrame:
        """
        Clean data based on validation results.
        
        Args:
            df: Original DataFrame
            validation_result: Validation results
            
        Returns:
            Cleaned DataFrame
        """
        clean_df = df.copy()
        rows_removed = 0
        
        # Remove rows with critical errors
        critical_error_types = ['missing_required_fields', 'type_conversion_error']
        
        for error in validation_result.errors:
            if error.get('error_type') in critical_error_types:
                # This is a simplified approach - in practice, you might want
                # more sophisticated row-level error tracking
                continue
        
        # Handle range violations
        range_errors = [e for e in validation_result.errors if e.get('error_type') == 'range_validation_error']
        for error in range_errors:
            field = error.get('field')
            if field in clean_df.columns:
                constraint = self.schema.get_field_constraint(field)
                if constraint:
                    # Clip values to valid range
                    if constraint.min_value is not None:
                        clean_df[field] = clean_df[field].clip(lower=constraint.min_value)
                    if constraint.max_value is not None:
                        clean_df[field] = clean_df[field].clip(upper=constraint.max_value)
        
        # Fill missing values for optional fields
        for field_name, constraint in self.schema.fields.items():
            if field_name in clean_df.columns and not constraint.required:
                if constraint.data_type.value in ['integer', 'float']:
                    clean_df[field_name] = clean_df[field_name].fillna(0)
                elif constraint.data_type.value == 'string':
                    clean_df[field_name] = clean_df[field_name].fillna('')
                elif constraint.data_type.value == 'boolean':
                    clean_df[field_name] = clean_df[field_name].fillna(False)
        
        if rows_removed > 0:
            logger.info(f"🧹 Removed {rows_removed} rows during data cleaning")
        
        return clean_df
    
    @validate_schema(strict_mode=False)
    def preprocess_for_ml(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess data for machine learning with validation.
        
        Args:
            df: Validated loan data
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_info)
        """
        logger.info("🔄 Preprocessing data for ML pipeline")
        
        preprocessing_info = {
            'input_shape': df.shape,
            'features_engineered': 0,
            'transformations_applied': []
        }
        
        processed_df = df.copy()
        
        # Basic preprocessing steps with validation
        try:
            # Handle categorical variables
            categorical_fields = self.schema.get_categorical_fields()
            for field in categorical_fields:
                if field in processed_df.columns:
                    # Convert to category type for efficiency
                    processed_df[field] = processed_df[field].astype('category')
                    preprocessing_info['transformations_applied'].append(f"Categorized {field}")
            
            # Ensure numerical fields are proper numeric types
            numerical_fields = self.schema.get_numerical_fields()
            for field in numerical_fields:
                if field in processed_df.columns:
                    processed_df[field] = pd.to_numeric(processed_df[field], errors='coerce')
                    preprocessing_info['transformations_applied'].append(f"Numericized {field}")
            
            # Feature engineering integration (if available)
            if FEATURE_ENGINEERING_AVAILABLE and self.feature_pipeline is None:
                logger.info("🏗️ Initializing feature engineering pipeline")
                self.feature_pipeline = FeatureEngineeringPipeline(config=self.feature_config)
                
                # Separate features and target
                target_col = 'loan_approved'
                if target_col in processed_df.columns:
                    X = processed_df.drop(columns=[target_col])
                    y = processed_df[target_col]
                    
                    # Apply feature engineering
                    X_engineered = self.feature_pipeline.fit_transform(X, y)
                    
                    # Combine back with target
                    processed_df = pd.concat([
                        pd.DataFrame(X_engineered, index=X.index),
                        y
                    ], axis=1)
                    
                    preprocessing_info['features_engineered'] = X_engineered.shape[1] - X.shape[1]
                    preprocessing_info['transformations_applied'].append("Applied feature engineering pipeline")
                    
                    logger.info(f"✅ Feature engineering complete: {X.shape[1]} → {X_engineered.shape[1]} features")
            
            preprocessing_info['output_shape'] = processed_df.shape
            
            logger.info("✅ Preprocessing complete")
            return processed_df, preprocessing_info
            
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            raise
    
    def validate_model_inputs(self, X: pd.DataFrame, y: pd.Series = None) -> bool:
        """
        Validate model training inputs.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            True if inputs are valid for model training
        """
        logger.info("🎯 Validating model inputs")
        
        validation_passed = True
        issues = []
        
        # Check for required minimum samples
        min_samples = 100
        if len(X) < min_samples:
            issues.append(f"Insufficient samples: {len(X)} < {min_samples}")
            validation_passed = False
        
        # Check for missing values
        missing_features = X.isnull().sum()
        problematic_features = missing_features[missing_features > len(X) * 0.5]  # >50% missing
        if len(problematic_features) > 0:
            issues.append(f"Features with >50% missing values: {problematic_features.index.tolist()}")
            validation_passed = False
        
        # Check for constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            issues.append(f"Constant features detected: {constant_features}")
            # This might be a warning rather than failure
        
        # Check target distribution (if provided)
        if y is not None:
            class_counts = y.value_counts()
            min_class_size = class_counts.min()
            if min_class_size < 10:
                issues.append(f"Minimum class size too small: {min_class_size}")
                validation_passed = False
            
            # Check for class imbalance
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 10:
                issues.append(f"Severe class imbalance detected: {imbalance_ratio:.1f}:1")
                # This is typically a warning, not a failure
        
        # Log results
        if validation_passed:
            logger.info("✅ Model input validation passed")
        else:
            logger.warning("⚠️ Model input validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return validation_passed
    
    def generate_validation_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable validation summary."""
        
        summary_lines = [
            "🔍 DATA VALIDATION SUMMARY",
            "=" * 50,
            f"Dataset: {report['dataset_info']['total_rows']} rows, {report['dataset_info']['total_columns']} columns",
            f"Memory Usage: {report['dataset_info']['memory_usage_mb']:.1f} MB",
            f"Data Quality Score: {report['data_quality_score']:.1f}/100",
            "",
            f"Validation Results:",
            f"  ✅ Overall Valid: {report['validation_summary']['overall_valid']}",
            f"  ❌ Total Errors: {report['validation_summary']['total_errors']}",
            f"  ⚠️  Total Warnings: {report['validation_summary']['total_warnings']}",
            f"  📊 Error Rate: {report['validation_summary']['error_rate']:.1%}",
        ]
        
        # Add business rules summary if available
        if report['business_rules'] and report['business_rules']['rules_passed'] > 0:
            summary_lines.extend([
                "",
                "Business Rules:",
                f"  ✅ Rules Passed: {report['business_rules']['rules_passed']}",
                f"  ❌ Rules Failed: {report['business_rules']['rules_failed']}",
                f"  🚨 Total Violations: {report['business_rules']['total_violations']}"
            ])
            
            if report['business_rules']['failed_rules']:
                summary_lines.append("  Failed Rules:")
                for rule in report['business_rules']['failed_rules'][:5]:  # Show top 5
                    summary_lines.append(f"    - {rule}")
        
        # Add top problematic fields
        field_issues = [(field, info) for field, info in report['field_validation'].items() 
                       if info['errors'] > 0]
        field_issues.sort(key=lambda x: x[1]['errors'], reverse=True)
        
        if field_issues:
            summary_lines.extend([
                "",
                "Most Problematic Fields:"
            ])
            for field, info in field_issues[:5]:  # Show top 5
                summary_lines.append(f"  {field}: {info['errors']} errors ({info['error_rate']:.1%})")
        
        return "\n".join(summary_lines)


def main():
    """Demonstrate validation framework integration."""
    logger.info("🚀 Starting Validation Framework Integration Demo")
    logger.info("=" * 60)
    
    try:
        # Initialize processor
        processor = ValidatedLoanProcessor(enable_business_rules=True)
        
        # Load and validate data
        data_path = "loan_dataset.csv"
        if Path(data_path).exists():
            clean_data, validation_report = processor.load_and_validate_data(data_path)
            
            # Print validation summary
            summary = processor.generate_validation_summary(validation_report)
            logger.info(f"\n{summary}")
            
            # Preprocess for ML
            processed_data, preprocessing_info = processor.preprocess_for_ml(clean_data)
            
            logger.info(f"\n🔄 PREPROCESSING RESULTS:")
            logger.info(f"Input Shape: {preprocessing_info['input_shape']}")
            logger.info(f"Output Shape: {preprocessing_info['output_shape']}")
            logger.info(f"Features Engineered: {preprocessing_info['features_engineered']}")
            logger.info(f"Transformations: {len(preprocessing_info['transformations_applied'])}")
            
            # Prepare for model training
            target_col = 'loan_approved'
            if target_col in processed_data.columns:
                X = processed_data.drop(columns=[target_col])
                y = processed_data[target_col]
                
                # Validate model inputs
                model_ready = processor.validate_model_inputs(X, y)
                
                if model_ready:
                    logger.info("✅ Data is ready for model training")
                else:
                    logger.warning("⚠️ Data has issues but may still be usable for training")
            
            logger.info("\n🎉 Validation Integration Demo Complete!")
            
        else:
            logger.warning(f"Dataset not found at {data_path}")
            logger.info("Demo completed with synthetic data validation")
            
            # Create synthetic data for demonstration
            synthetic_data = pd.DataFrame({
                'age': [25, 30, 45],
                'annual_income': [40000, 60000, 80000],
                'credit_score': [650, 720, 800],
                'loan_amount': [20000, 30000, 40000],
                'gender': ['Male', 'Female', 'Male'],
                'marital_status': ['Single', 'Married', 'Divorced'],
                'education': ["Bachelor's", "Advanced", "High School"],
                'employment_status': ['Employed', 'Employed', 'Employed']
            })
            
            # Validate synthetic data
            result = processor.validator.validate_all(synthetic_data, include_business_rules=False)
            logger.info(f"Synthetic data validation: {len(result.errors)} errors, {len(result.warnings)} warnings")
    
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()