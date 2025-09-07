"""
Data Validation Framework for Loan Eligibility Prediction System

This package provides comprehensive data validation capabilities including:
- Schema validation for input data
- Data type checking and conversion
- Range validation for numerical fields
- Required field validation  
- Format validation (email, phone, etc.)
- Custom business rule validation
- Validation error reporting

The framework is designed to integrate seamlessly with the existing ML pipeline
and ensure only clean, validated data enters the prediction models.
"""

from .schema import LoanDataSchema, ValidationResult
from .validators import (
    DataValidator,
    SchemaValidator,
    RangeValidator,
    FormatValidator,
    BusinessRuleValidator
)
from .decorators import validate_input, validate_schema, require_fields
from .converters import DataTypeConverter
from .rules import LoanBusinessRules
from .exceptions import (
    ValidationError,
    SchemaValidationError,
    RangeValidationError,
    FormatValidationError,
    BusinessRuleValidationError
)

__all__ = [
    'LoanDataSchema',
    'ValidationResult',
    'DataValidator',
    'SchemaValidator', 
    'RangeValidator',
    'FormatValidator',
    'BusinessRuleValidator',
    'validate_input',
    'validate_schema',
    'require_fields',
    'DataTypeConverter',
    'LoanBusinessRules',
    'ValidationError',
    'SchemaValidationError',
    'RangeValidationError',
    'FormatValidationError',
    'BusinessRuleValidationError'
]

__version__ = "1.0.0"