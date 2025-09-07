"""
Custom exception classes for data validation framework.

Provides specific exception types for different validation failures
to enable precise error handling and reporting.
"""

from typing import Any, Dict, List, Optional


class ValidationError(Exception):
    """Base exception for all validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, details: Optional[Dict] = None):
        self.message = message
        self.field = field
        self.value = value
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'field': self.field,
            'value': self.value,
            'details': self.details
        }


class SchemaValidationError(ValidationError):
    """Raised when data doesn't match expected schema."""
    
    def __init__(self, message: str, missing_fields: Optional[List[str]] = None,
                 extra_fields: Optional[List[str]] = None, **kwargs):
        self.missing_fields = missing_fields or []
        self.extra_fields = extra_fields or []
        details = kwargs.get('details', {})
        details.update({
            'missing_fields': self.missing_fields,
            'extra_fields': self.extra_fields
        })
        super().__init__(message, details=details, **kwargs)


class DataTypeValidationError(ValidationError):
    """Raised when data type conversion fails."""
    
    def __init__(self, message: str, expected_type: str, actual_type: str, **kwargs):
        self.expected_type = expected_type
        self.actual_type = actual_type
        details = kwargs.get('details', {})
        details.update({
            'expected_type': self.expected_type,
            'actual_type': self.actual_type
        })
        super().__init__(message, details=details, **kwargs)


class RangeValidationError(ValidationError):
    """Raised when numeric values are outside acceptable ranges."""
    
    def __init__(self, message: str, min_value: Optional[float] = None,
                 max_value: Optional[float] = None, **kwargs):
        self.min_value = min_value
        self.max_value = max_value
        details = kwargs.get('details', {})
        details.update({
            'min_value': self.min_value,
            'max_value': self.max_value
        })
        super().__init__(message, details=details, **kwargs)


class FormatValidationError(ValidationError):
    """Raised when data format doesn't match expected pattern."""
    
    def __init__(self, message: str, expected_format: str, pattern: Optional[str] = None, **kwargs):
        self.expected_format = expected_format
        self.pattern = pattern
        details = kwargs.get('details', {})
        details.update({
            'expected_format': self.expected_format,
            'pattern': self.pattern
        })
        super().__init__(message, details=details, **kwargs)


class BusinessRuleValidationError(ValidationError):
    """Raised when business rules are violated."""
    
    def __init__(self, message: str, rule_name: str, rule_description: Optional[str] = None, **kwargs):
        self.rule_name = rule_name
        self.rule_description = rule_description
        details = kwargs.get('details', {})
        details.update({
            'rule_name': self.rule_name,
            'rule_description': self.rule_description
        })
        super().__init__(message, details=details, **kwargs)


class RequiredFieldError(ValidationError):
    """Raised when required fields are missing."""
    
    def __init__(self, message: str, required_fields: List[str], **kwargs):
        self.required_fields = required_fields
        details = kwargs.get('details', {})
        details.update({
            'required_fields': self.required_fields
        })
        super().__init__(message, details=details, **kwargs)


class ValidationSummaryError(ValidationError):
    """Raised when multiple validation errors occur."""
    
    def __init__(self, message: str, errors: List[ValidationError], **kwargs):
        self.errors = errors
        self.error_count = len(errors)
        details = kwargs.get('details', {})
        details.update({
            'error_count': self.error_count,
            'errors': [error.to_dict() for error in self.errors]
        })
        super().__init__(message, details=details, **kwargs)
    
    def get_errors_by_type(self, error_type: type) -> List[ValidationError]:
        """Get all errors of a specific type."""
        return [error for error in self.errors if isinstance(error, error_type)]
    
    def get_errors_by_field(self, field_name: str) -> List[ValidationError]:
        """Get all errors for a specific field."""
        return [error for error in self.errors if error.field == field_name]