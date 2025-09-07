"""
Comprehensive Test and Demo of Data Cleaning & Preprocessing Pipeline
Task ID: DP-002 - Testing and validation
"""

import pandas as pd
import numpy as np
import logging
from preprocessing import PreprocessingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with loan dataset"""
    
    print("🚀 Testing Data Cleaning & Preprocessing Pipeline")
    print("=" * 60)
    
    # Load the loan dataset
    try:
        df = pd.read_csv('loan_dataset.csv')
        print(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except FileNotFoundError:
        print("❌ loan_dataset.csv not found. Creating sample data...")
        df = create_sample_data()
        print(f"✅ Sample dataset created: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print("\n1. Initial Data Assessment")
    print("-" * 30)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Test 1: Default Pipeline Configuration
    print("\n2. Testing Default Pipeline")
    print("-" * 30)
    
    pipeline = PreprocessingPipeline()
    
    # Validate data before processing
    validation_results = pipeline.validate_data(df)
    print(f"Data validation: {'✅ PASSED' if validation_results['valid'] else '❌ FAILED'}")
    if validation_results['issues']:
        print(f"Issues: {validation_results['issues']}")
    if validation_results['warnings']:
        print(f"Warnings: {validation_results['warnings']}")
    
    # Fit and transform
    print("\nFitting pipeline...")
    pipeline.fit(df, target_column='Loan_Status' if 'Loan_Status' in df.columns else None)
    
    print("Transforming data...")
    processed_df = pipeline.transform(df)
    
    print(f"✅ Processing complete!")
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {processed_df.shape}")
    print(f"Missing values after: {processed_df.isnull().sum().sum()}")
    
    # Test 2: Custom Pipeline Configuration
    print("\n3. Testing Custom Pipeline Configuration")
    print("-" * 30)
    
    custom_config = {
        'steps': ['quality_check', 'duplicates', 'imputation', 'outliers', 'normalization'],
        'imputation': {
            'strategies': {
                'LoanAmount': 'median',
                'Credit_History': 'mode',
                'Self_Employed': 'mode'
            },
            'default_numeric': 'median',
            'default_categorical': 'mode'
        },
        'outliers': {
            'methods': {
                'ApplicantIncome': 'zscore',
                'LoanAmount': 'iqr',
                'Loan_Amount_Term': 'iqr'
            },
            'handle_strategy': 'clip',
            'thresholds': {
                'zscore': {'threshold': 2.5},
                'iqr': {'multiplier': 1.5}
            }
        },
        'normalization': {
            'scaling_methods': {
                'ApplicantIncome': 'robust',
                'CoapplicantIncome': 'robust',
                'LoanAmount': 'standard'
            },
            'categorical_encoding': 'onehot'
        },
        'duplicates': {
            'strategy': 'fuzzy',
            'threshold': 0.85,
            'keep': 'first'
        },
        'quality': {
            'min_quality_score': 0.7,
            'generate_report': True
        }
    }
    
    custom_pipeline = PreprocessingPipeline(custom_config)
    processed_custom_df = custom_pipeline.fit_transform(df, target_column='Loan_Status' if 'Loan_Status' in df.columns else None)
    
    print(f"✅ Custom processing complete!")
    print(f"Custom processed shape: {processed_custom_df.shape}")
    
    # Test 3: Component Testing
    print("\n4. Testing Individual Components")
    print("-" * 30)
    
    # Test missing value imputation
    from preprocessing.imputation import MissingValueImputer
    imputer = MissingValueImputer()
    missing_summary = imputer.get_missing_summary(df)
    print(f"Missing value analysis: {len(missing_summary)} columns analyzed")
    print(f"Columns with missing values: {len(missing_summary[missing_summary['Missing_Count'] > 0])}")
    
    # Test outlier detection
    from preprocessing.outliers import OutlierDetector
    outlier_detector = OutlierDetector()
    outlier_summary = outlier_detector.get_outlier_summary(df)
    print(f"Outlier analysis: {len(outlier_summary)} numeric columns analyzed")
    
    # Test duplicate detection
    from preprocessing.duplicates import DuplicateHandler
    duplicate_handler = DuplicateHandler()
    duplicate_summary = duplicate_handler.get_duplicate_summary(df)
    print(f"Duplicate analysis: {duplicate_summary.iloc[0]['Duplicate_Records']} duplicates found")
    
    # Test data quality scoring
    from preprocessing.quality import DataQualityScorer
    quality_scorer = DataQualityScorer()
    quality_score = quality_scorer.get_quality_score(df)
    print(f"Data quality score: {quality_score:.3f}")
    
    # Test 4: Performance and Memory Usage
    print("\n5. Performance Analysis")
    print("-" * 30)
    
    import time
    
    # Time the processing
    start_time = time.time()
    test_pipeline = PreprocessingPipeline()
    result = test_pipeline.fit_transform(df)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing rate: {len(df) / processing_time:.0f} records/second")
    
    memory_before = df.memory_usage(deep=True).sum() / 1024 / 1024
    memory_after = result.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Memory usage - Before: {memory_before:.2f} MB, After: {memory_after:.2f} MB")
    print(f"Memory change: {memory_after - memory_before:+.2f} MB")
    
    # Test 5: Pipeline Reports
    print("\n6. Pipeline Reports")
    print("-" * 30)
    
    # Get processing summary
    summary = test_pipeline.get_preprocessing_summary()
    print(f"Pipeline fitted: {summary['fitted']}")
    print(f"Steps applied: {len(summary['steps_applied'])}")
    
    # Generate comprehensive report
    report = test_pipeline.get_processing_report()
    print("\nProcessing Report:")
    print(report)
    
    # Get quality report if available
    if test_pipeline.quality_reports:
        print("\nLatest Quality Report:")
        latest_quality = test_pipeline.quality_reports[-1]
        print(f"Quality Score: {latest_quality['score']:.3f}")
    
    # Test 6: Save and Load Configuration
    print("\n7. Configuration Management")
    print("-" * 30)
    
    # Save pipeline configuration
    config_file = 'preprocessing_config.json'
    test_pipeline.save_config(config_file)
    print(f"✅ Configuration saved to {config_file}")
    
    # Load configuration
    try:
        loaded_pipeline = PreprocessingPipeline.load_config(config_file)
        print(f"✅ Configuration loaded from {config_file}")
        print(f"Loaded pipeline fitted: {loaded_pipeline.fitted}")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
    
    # Test 7: Integration with Validation Framework
    print("\n8. Integration Testing")
    print("-" * 30)
    
    try:
        # Try to integrate with validation framework
        from validation import ValidationFramework
        validator = ValidationFramework()
        
        # Validate processed data
        validation_result = validator.validate_dataframe(processed_df)
        print(f"✅ Validation framework integration: {validation_result['valid']}")
        print(f"Validation score: {validation_result.get('score', 'N/A')}")
    except ImportError:
        print("ℹ️ Validation framework not available for integration test")
    except Exception as e:
        print(f"❌ Integration test error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 DATA CLEANING & PREPROCESSING PIPELINE TEST COMPLETE")
    print("=" * 60)
    
    return {
        'original_shape': df.shape,
        'processed_shape': processed_df.shape,
        'processing_time': processing_time,
        'quality_score': quality_score,
        'pipeline_fitted': test_pipeline.fitted
    }

def create_sample_data():
    """Create sample loan data for testing if original dataset is not available"""
    np.random.seed(42)
    
    n_samples = 1000
    
    # Create sample loan data
    data = {
        'Loan_ID': [f'LP{i:06d}' for i in range(1, n_samples + 1)],
        'Gender': np.random.choice(['Male', 'Female', np.nan], n_samples, p=[0.8, 0.15, 0.05]),
        'Married': np.random.choice(['Yes', 'No', np.nan], n_samples, p=[0.65, 0.3, 0.05]),
        'Dependents': np.random.choice(['0', '1', '2', '3+', np.nan], n_samples, p=[0.35, 0.25, 0.2, 0.15, 0.05]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.78, 0.22]),
        'Self_Employed': np.random.choice(['Yes', 'No', np.nan], n_samples, p=[0.15, 0.8, 0.05]),
        'ApplicantIncome': np.random.lognormal(8, 1, n_samples).astype(int),
        'CoapplicantIncome': np.random.lognormal(6, 1.5, n_samples).astype(int),
        'LoanAmount': np.random.normal(140, 60, n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 240, 300, np.nan], n_samples, p=[0.85, 0.05, 0.05, 0.03, 0.02]),
        'Credit_History': np.random.choice([1.0, 0.0, np.nan], n_samples, p=[0.8, 0.15, 0.05]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.3, 0.4, 0.3]),
        'Loan_Status': np.random.choice(['Y', 'N'], n_samples, p=[0.69, 0.31])
    }
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    for idx in outlier_indices:
        data['ApplicantIncome'][idx] *= 10  # Create income outliers
    
    # Add some duplicates
    duplicate_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    for i, idx in enumerate(duplicate_indices[1:], 1):
        # Make this row similar to the previous duplicate
        prev_idx = duplicate_indices[i-1]
        for col in ['Gender', 'Married', 'Education', 'Property_Area']:
            data[col][idx] = data[col][prev_idx]
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    for col in ['LoanAmount', 'Credit_History', 'Self_Employed']:
        missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        for idx in missing_indices:
            df.loc[idx, col] = np.nan
    
    return df

def run_comprehensive_tests():
    """Run comprehensive testing suite"""
    
    print("🧪 COMPREHENSIVE PREPROCESSING PIPELINE TESTS")
    print("=" * 70)
    
    test_results = []
    
    try:
        # Main test
        result = test_preprocessing_pipeline()
        test_results.append(('Main Pipeline Test', 'PASSED', result))
        print("✅ Main pipeline test completed successfully")
    except Exception as e:
        test_results.append(('Main Pipeline Test', 'FAILED', str(e)))
        print(f"❌ Main pipeline test failed: {e}")
    
    # Additional specific tests
    try:
        # Test error handling
        empty_df = pd.DataFrame()
        pipeline = PreprocessingPipeline()
        validation_result = pipeline.validate_data(empty_df)
        assert not validation_result['valid'], "Empty dataset should fail validation"
        test_results.append(('Empty Data Validation', 'PASSED', None))
        print("✅ Empty data validation test passed")
    except Exception as e:
        test_results.append(('Empty Data Validation', 'FAILED', str(e)))
        print(f"❌ Empty data validation test failed: {e}")
    
    try:
        # Test configuration validation
        invalid_config = {'steps': ['invalid_step']}
        pipeline = PreprocessingPipeline(invalid_config)
        # This should not crash, just skip invalid steps
        test_results.append(('Invalid Config Handling', 'PASSED', None))
        print("✅ Invalid configuration handling test passed")
    except Exception as e:
        test_results.append(('Invalid Config Handling', 'FAILED', str(e)))
        print(f"❌ Invalid configuration handling test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, status, _ in test_results if status == 'PASSED')
    total = len(test_results)
    
    print(f"Tests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, status, result in test_results:
        status_emoji = "✅" if status == "PASSED" else "❌"
        print(f"{status_emoji} {test_name}: {status}")
        if status == "FAILED":
            print(f"   Error: {result}")
    
    print("\n🏁 COMPREHENSIVE TESTING COMPLETE")
    print("=" * 70)
    
    return test_results

if __name__ == "__main__":
    # Run the comprehensive test suite
    results = run_comprehensive_tests()
    
    # Exit with appropriate code
    failed_tests = sum(1 for _, status, _ in results if status == 'FAILED')
    exit(0 if failed_tests == 0 else 1)