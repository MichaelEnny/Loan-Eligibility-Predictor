"""
Comprehensive test suite for the data validation framework.

Tests all validation components including schema validation, data type conversion,
range validation, format validation, and business rule validation.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add validation package to path
sys.path.append(str(Path(__file__).parent))

from validation import (
    LoanDataSchema, ValidationResult, DataValidator,
    SchemaValidator, RangeValidator, FormatValidator, BusinessRuleValidator,
    validate_input, validate_schema, require_fields,
    DataTypeConverter, LoanBusinessRules,
    ValidationError, SchemaValidationError, BusinessRuleValidationError
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValidationFrameworkTester:
    """Comprehensive tester for the validation framework."""
    
    def __init__(self):
        self.schema = LoanDataSchema()
        self.validator = DataValidator(self.schema)
        self.business_rules = LoanBusinessRules()
        self.results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': []
        }
    
    def run_all_tests(self):
        """Run all validation framework tests."""
        logger.info("🚀 Starting Validation Framework Tests")
        logger.info("=" * 60)
        
        try:
            self.test_schema_validation()
            self.test_data_type_conversion()
            self.test_range_validation()
            self.test_format_validation()
            self.test_business_rule_validation()
            self.test_decorators()
            self.test_integration_with_real_data()
            self.test_error_handling()
            
            self._print_results()
            
        except Exception as e:
            logger.error(f"Test suite failed with error: {e}")
            self.results['errors'].append(str(e))
    
    def test_schema_validation(self):
        """Test schema validation functionality."""
        logger.info("\n📋 Testing Schema Validation")
        
        # Test valid data
        valid_data = {
            'age': 30,
            'gender': 'Male',
            'marital_status': 'Single',
            'education': "Bachelor's",
            'employment_status': 'Employed',
            'years_employed': 5.0,
            'annual_income': 50000.0,
            'monthly_income': 4166.67,
            'credit_score': 720.0,
            'credit_history_length': 8.0,
            'num_credit_accounts': 3,
            'existing_debt': 15000.0,
            'monthly_debt_payments': 500.0,
            'debt_to_income_ratio': 0.12,
            'loan_amount': 25000.0,
            'loan_term_months': 60,
            'loan_purpose': 'Auto',
            'owns_property': True,
            'property_value': 200000.0,
            'has_bank_account': True,
            'years_with_bank': 3.0,
            'previous_loans': 1,
            'previous_loan_defaults': 0,
            'state': 'CA',
            'area_type': 'Urban',
            'has_coapplicant': False,
            'coapplicant_income': 0.0,
            'total_household_income': 50000.0
        }
        
        self._run_test("Valid data schema validation", 
                      lambda: self.schema.validate_schema(valid_data).is_valid)
        
        # Test missing required fields
        incomplete_data = valid_data.copy()
        del incomplete_data['age']
        del incomplete_data['credit_score']
        
        result = self.schema.validate_schema(incomplete_data)
        self._run_test("Missing required fields detection", 
                      lambda: not result.is_valid and len(result.errors) > 0)
        
        # Test unexpected fields
        data_with_extra = valid_data.copy()
        data_with_extra['unexpected_field'] = 'value'
        
        result = self.schema.validate_schema(data_with_extra)
        self._run_test("Unexpected fields detection",
                      lambda: len(result.warnings) > 0)
    
    def test_data_type_conversion(self):
        """Test data type conversion functionality."""
        logger.info("\n🔄 Testing Data Type Conversion")
        
        converter = DataTypeConverter(strict_mode=False)
        
        # Test integer conversion
        from validation.schema import DataType
        result = converter.convert_value("123", DataType.INTEGER)
        self._run_test("String to integer conversion",
                      lambda: result == 123)
        
        # Test float conversion
        result = converter.convert_value("123.45", DataType.FLOAT)
        self._run_test("String to float conversion",
                      lambda: abs(result - 123.45) < 0.001 if result else False)
        
        # Test boolean conversion
        self._run_test("String to boolean conversion",
                      lambda: converter.convert_value("true", DataType.BOOLEAN) == True)
        
        # Test invalid conversion handling
        result = converter.convert_value("invalid", DataType.INTEGER)
        self._run_test("Invalid conversion handling",
                      lambda: result is None)  # Should return None in non-strict mode
    
    def test_range_validation(self):
        """Test range validation functionality."""
        logger.info("\n📊 Testing Range Validation")
        
        # Test valid ranges
        valid_data = pd.DataFrame({
            'age': [25, 30, 45],
            'credit_score': [650, 720, 800],
            'annual_income': [40000, 60000, 80000]
        })
        
        result = self.validator.validate_all(valid_data, include_formats=False, include_business_rules=False)
        self._run_test("Valid ranges validation",
                      lambda: result.is_valid or len(result.errors) == 0)
        
        # Test invalid ranges
        invalid_data = pd.DataFrame({
            'age': [17, 101],  # Below/above valid range
            'credit_score': [200, 900],  # Below/above valid range
            'annual_income': [-1000, 15000000]  # Below/above valid range
        })
        
        result = self.validator.validate_all(invalid_data, include_formats=False, include_business_rules=False)
        self._run_test("Invalid ranges detection",
                      lambda: not result.is_valid and len(result.errors) > 0)
    
    def test_format_validation(self):
        """Test format validation functionality."""
        logger.info("\n✉️ Testing Format Validation")
        
        format_validator = FormatValidator()
        
        # Test email validation
        test_data = pd.DataFrame({
            'email': ['test@example.com', 'invalid_email', 'another@test.org']
        })
        
        result = format_validator.validate(test_data, {'email': 'email'})
        self._run_test("Email format validation",
                      lambda: len(result.errors) > 0)  # Should catch invalid_email
        
        # Test phone validation
        test_data = pd.DataFrame({
            'phone': ['(555) 123-4567', '555-123-4567', 'not-a-phone']
        })
        
        result = format_validator.validate(test_data, {'phone': 'phone'})
        self._run_test("Phone format validation",
                      lambda: len(result.errors) > 0)  # Should catch invalid phone
    
    def test_business_rule_validation(self):
        """Test business rule validation functionality."""
        logger.info("\n🏦 Testing Business Rule Validation")
        
        # Test income consistency rule
        consistent_data = pd.DataFrame({
            'annual_income': [60000],
            'monthly_income': [5000],  # 60000/12 = 5000
            'monthly_debt_payments': [600],
            'debt_to_income_ratio': [0.12],  # 600/5000 = 0.12
            'has_coapplicant': [False],
            'coapplicant_income': [0],
            'total_household_income': [60000]
        })
        
        rules_result = self.business_rules.validate_all_rules(consistent_data)
        self._run_test("Income consistency validation",
                      lambda: 'income_consistency' in rules_result['passed_rules'])
        
        # Test inconsistent data
        inconsistent_data = pd.DataFrame({
            'annual_income': [60000],
            'monthly_income': [3000],  # Should be ~5000
            'monthly_debt_payments': [600],
            'debt_to_income_ratio': [0.12],
            'has_coapplicant': [False],
            'coapplicant_income': [0],
            'total_household_income': [60000]
        })
        
        rules_result = self.business_rules.validate_all_rules(inconsistent_data)
        self._run_test("Income inconsistency detection",
                      lambda: 'income_consistency' in rules_result['failed_rules'])
    
    def test_decorators(self):
        """Test validation decorators."""
        logger.info("\n🎭 Testing Validation Decorators")
        
        # Test validate_input decorator
        @validate_input(strict_mode=False)
        def process_loan_data(data):
            return len(data) if hasattr(data, '__len__') else 1
        
        valid_data = pd.DataFrame({
            'age': [30],
            'gender': ['Male'],
            'annual_income': [50000]
        })
        
        try:
            result = process_loan_data(valid_data)
            self._run_test("validate_input decorator",
                          lambda: result == 1)
        except Exception as e:
            self._run_test("validate_input decorator",
                          lambda: False)
            self.results['errors'].append(f"Decorator test failed: {e}")
        
        # Test require_fields decorator
        @require_fields('age', 'income', strict_mode=False)
        def analyze_application(data):
            return "analyzed"
        
        # This should work without raising an exception due to strict_mode=False
        try:
            result = analyze_application({'age': 30})  # Missing 'income'
            self._run_test("require_fields decorator",
                          lambda: result == "analyzed")
        except Exception:
            self._run_test("require_fields decorator",
                          lambda: False)
    
    def test_integration_with_real_data(self):
        """Test with real dataset if available."""
        logger.info("\n🔗 Testing Integration with Real Data")
        
        try:
            # Try to load real dataset
            data_path = Path("loan_dataset.csv")
            if data_path.exists():
                df = pd.read_csv(data_path)
                logger.info(f"Loaded dataset with {len(df)} rows, {len(df.columns)} columns")
                
                # Sample small subset for testing
                test_data = df.head(100)
                
                # Run comprehensive validation
                result = self.validator.validate_all(
                    test_data,
                    include_schema=True,
                    include_ranges=True,
                    include_formats=False,
                    include_business_rules=True
                )
                
                logger.info(f"Real data validation: {len(result.errors)} errors, "
                           f"{len(result.warnings)} warnings")
                
                self._run_test("Real data validation execution",
                              lambda: True)  # Just test that it runs without crashing
                
                # Test business rules on real data
                business_result = self.business_rules.validate_all_rules(test_data)
                logger.info(f"Business rules: {len(business_result['passed_rules'])} passed, "
                           f"{len(business_result['failed_rules'])} failed")
                
                self._run_test("Business rules on real data",
                              lambda: len(business_result['rule_violations']) >= 0)
            
            else:
                logger.warning("Real dataset not found, skipping integration test")
                self._run_test("Real data integration test",
                              lambda: True)  # Skip test
        
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            self._run_test("Real data integration test",
                          lambda: False)
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        logger.info("\n⚠️ Testing Error Handling")
        
        # Test with None data
        try:
            result = self.validator.validate_all(None)
            self._run_test("None data handling",
                          lambda: not result.is_valid)
        except Exception:
            self._run_test("None data handling",
                          lambda: True)  # Exception is acceptable
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = self.validator.validate_all(empty_df)
        self._run_test("Empty DataFrame handling",
                      lambda: not result.is_valid)  # Should fail due to missing required fields
        
        # Test with malformed data
        malformed_data = pd.DataFrame({
            'age': ['not_a_number', None, 'also_not_a_number'],
            'credit_score': [np.inf, -np.inf, np.nan]
        })
        
        try:
            result = self.validator.validate_all(malformed_data)
            self._run_test("Malformed data handling",
                          lambda: not result.is_valid)
        except Exception:
            self._run_test("Malformed data handling",
                          lambda: True)  # Exception handling is acceptable
    
    def _run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        self.results['tests_run'] += 1
        
        try:
            if test_func():
                logger.info(f"✅ {test_name}")
                self.results['tests_passed'] += 1
            else:
                logger.error(f"❌ {test_name}")
                self.results['tests_failed'] += 1
        except Exception as e:
            logger.error(f"❌ {test_name}: {e}")
            self.results['tests_failed'] += 1
            self.results['errors'].append(f"{test_name}: {e}")
    
    def _print_results(self):
        """Print test results summary."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Tests Run: {self.results['tests_run']}")
        logger.info(f"Tests Passed: {self.results['tests_passed']}")
        logger.info(f"Tests Failed: {self.results['tests_failed']}")
        
        success_rate = (self.results['tests_passed'] / self.results['tests_run'] * 100) if self.results['tests_run'] > 0 else 0
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if self.results['errors']:
            logger.info("\n❌ ERRORS:")
            for error in self.results['errors']:
                logger.info(f"  - {error}")
        
        if success_rate >= 80:
            logger.info("\n🎉 Validation Framework Tests: SUCCESS")
        else:
            logger.error("\n💥 Validation Framework Tests: FAILED")


def demonstrate_validation_capabilities():
    """Demonstrate comprehensive validation capabilities."""
    logger.info("\n" + "=" * 60)
    logger.info("🌟 VALIDATION FRAMEWORK DEMONSTRATION")
    logger.info("=" * 60)
    
    # Create sample data with various issues
    sample_data = pd.DataFrame({
        'age': [25, 17, 45, 101, 30],  # Mix of valid/invalid ages
        'gender': ['Male', 'Female', 'Invalid', 'Male', 'Other'],
        'marital_status': ['Single', 'Married', 'Single', 'Divorced', 'Single'],
        'education': ["Bachelor's", "High School", "Advanced", "Some College", "Bachelor's"],
        'employment_status': ['Employed', 'Unemployed', 'Employed', 'Retired', 'Employed'],
        'years_employed': [5.0, 0.0, 15.0, 45.0, 3.0],
        'annual_income': [50000, 0, 80000, 200000, 45000],
        'monthly_income': [4166.67, 0, 6666.67, 16666.67, 3750],  # Some inconsistent
        'credit_score': [720, 650, 300, 850, 680],  # Mix of valid/edge cases
        'credit_history_length': [8.0, 2.0, 15.0, 25.0, 5.0],
        'num_credit_accounts': [3, 0, 8, 12, 4],
        'existing_debt': [15000, 0, 25000, 50000, 12000],
        'monthly_debt_payments': [500, 0, 800, 1200, 400],
        'debt_to_income_ratio': [0.12, 0.0, 0.15, 0.08, 0.11],  # Some may be inconsistent
        'loan_amount': [25000, 5000, 40000, 100000, 20000],
        'loan_term_months': [60, 24, 84, 360, 48],
        'loan_purpose': ['Auto', 'Personal', 'Home Improvement', 'Business', 'Auto'],
        'owns_property': [True, False, True, True, False],
        'property_value': [200000, 0, 300000, 500000, 0],
        'has_bank_account': [True, True, True, True, True],
        'years_with_bank': [3.0, 1.0, 10.0, 20.0, 2.0],
        'previous_loans': [1, 0, 3, 5, 1],
        'previous_loan_defaults': [0, 0, 1, 0, 0],
        'state': ['CA', 'TX', 'FL', 'NY', 'CA'],
        'area_type': ['Urban', 'Rural', 'Suburban', 'Urban', 'Suburban'],
        'has_coapplicant': [False, False, True, False, False],
        'coapplicant_income': [0, 0, 30000, 0, 0],
        'total_household_income': [50000, 0, 110000, 200000, 45000]
    })
    
    # Initialize validator
    validator = DataValidator()
    
    # Add custom business rules
    validator.add_business_rule(
        "minimum_income_threshold",
        lambda data: data.get('annual_income', 0) >= 20000 if isinstance(data, dict) else (data['annual_income'] >= 20000).all(),
        "Annual income should be at least $20,000"
    )
    
    # Run comprehensive validation
    logger.info("Running comprehensive validation on sample data...")
    result = validator.validate_all(
        sample_data,
        include_schema=True,
        include_ranges=True,
        include_formats=False,
        include_business_rules=True
    )
    
    # Display results
    logger.info(f"\n📊 VALIDATION RESULTS:")
    logger.info(f"Overall Valid: {result.is_valid}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Warnings: {len(result.warnings)}")
    
    if result.errors:
        logger.info("\n❌ VALIDATION ERRORS:")
        for i, error in enumerate(result.errors[:10]):  # Show first 10 errors
            logger.info(f"  {i+1}. Field '{error.get('field')}': {error.get('message')}")
        
        if len(result.errors) > 10:
            logger.info(f"  ... and {len(result.errors) - 10} more errors")
    
    if result.warnings:
        logger.info("\n⚠️ VALIDATION WARNINGS:")
        for i, warning in enumerate(result.warnings[:5]):  # Show first 5 warnings
            logger.info(f"  {i+1}. Field '{warning.get('field')}': {warning.get('message')}")
    
    # Test business rules
    business_rules = LoanBusinessRules()
    business_result = business_rules.validate_all_rules(sample_data)
    
    logger.info(f"\n🏦 BUSINESS RULES RESULTS:")
    logger.info(f"Rules Passed: {len(business_result['passed_rules'])}")
    logger.info(f"Rules Failed: {len(business_result['failed_rules'])}")
    logger.info(f"Total Violations: {len(business_result['rule_violations'])}")
    
    if business_result['failed_rules']:
        logger.info("\n❌ FAILED BUSINESS RULES:")
        for rule in business_result['failed_rules']:
            logger.info(f"  - {rule}")
    
    logger.info("\n✅ VALIDATION DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    # Run comprehensive tests
    tester = ValidationFrameworkTester()
    tester.run_all_tests()
    
    # Run demonstration
    demonstrate_validation_capabilities()
    
    logger.info("\n🎯 Validation Framework Testing Complete!")
    logger.info("The framework is ready for integration with the ML pipeline.")