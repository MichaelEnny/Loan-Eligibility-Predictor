"""
Core validation classes for the data validation framework.

Provides comprehensive validation capabilities including schema validation,
range checking, format validation, and business rule validation.
"""

from typing import Any, Dict, List, Optional, Union, Pattern, Callable
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

from .exceptions import (
    ValidationError, SchemaValidationError, RangeValidationError,
    FormatValidationError, BusinessRuleValidationError
)
from .schema import LoanDataSchema, ValidationResult, FieldConstraint, DataType
from .converters import DataTypeConverter


logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """Abstract base class for all validators."""
    
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or f"{name} validator"
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate data and return results."""
        pass


class SchemaValidator(BaseValidator):
    """Validates data against predefined schema."""
    
    def __init__(self, schema: LoanDataSchema):
        super().__init__("schema_validator", "Validates data structure and types")
        self.schema = schema
        self.converter = DataTypeConverter(strict_mode=False)
    
    def validate(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                convert_types: bool = True) -> ValidationResult:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate (DataFrame or dict)
            convert_types: Whether to attempt type conversion
            
        Returns:
            ValidationResult with validation details
        """
        result = self.schema.validate_schema(data)
        
        if not result.is_valid:
            return result
        
        # Validate individual fields
        if isinstance(data, pd.DataFrame):
            self._validate_dataframe_fields(data, result, convert_types)
        elif isinstance(data, dict):
            self._validate_dict_fields(data, result, convert_types)
        
        return result
    
    def _validate_dataframe_fields(self, df: pd.DataFrame, result: ValidationResult,
                                 convert_types: bool):
        """Validate DataFrame fields against schema constraints."""
        for column in df.columns:
            if column not in self.schema.fields:
                continue
            
            constraint = self.schema.fields[column]
            series = df[column]
            
            # Validate field values
            self._validate_field_series(series, constraint, result, convert_types)
    
    def _validate_dict_fields(self, data: Dict[str, Any], result: ValidationResult,
                            convert_types: bool):
        """Validate dictionary fields against schema constraints."""
        for field_name, value in data.items():
            if field_name not in self.schema.fields:
                continue
            
            constraint = self.schema.fields[field_name]
            
            # Convert single value to series for uniform processing
            series = pd.Series([value], name=field_name)
            self._validate_field_series(series, constraint, result, convert_types)
    
    def _validate_field_series(self, series: pd.Series, constraint: FieldConstraint,
                             result: ValidationResult, convert_types: bool):
        """Validate a pandas Series against field constraint."""
        field_name = constraint.name
        
        # Check for required field with all null values
        if constraint.required and series.isna().all():
            result.add_error(
                field=field_name,
                message=f"Required field '{field_name}' contains only null values",
                error_type="required_field_error"
            )
            return
        
        # Skip validation for null values if field is nullable
        valid_values = series.dropna() if constraint.nullable else series
        
        if len(valid_values) == 0:
            return
        
        # Type validation and conversion
        if convert_types:
            try:
                converted_series = self.converter.convert_series(
                    valid_values, constraint.data_type, field_name
                )
                valid_values = converted_series.dropna()
            except Exception as e:
                result.add_error(
                    field=field_name,
                    message=f"Type conversion failed: {str(e)}",
                    error_type="type_conversion_error"
                )
                return
        
        # Value constraints validation
        self._validate_value_constraints(valid_values, constraint, result)
    
    def _validate_value_constraints(self, series: pd.Series, constraint: FieldConstraint,
                                  result: ValidationResult):
        """Validate value constraints for a field."""
        field_name = constraint.name
        
        # Range validation for numeric types
        if constraint.data_type in [DataType.INTEGER, DataType.FLOAT]:
            if constraint.min_value is not None:
                invalid_min = series < constraint.min_value
                if invalid_min.any():
                    invalid_values = series[invalid_min].tolist()
                    result.add_error(
                        field=field_name,
                        message=f"Values below minimum {constraint.min_value}: {invalid_values[:5]}",
                        error_type="range_validation_error",
                        constraint_type="min_value",
                        min_value=constraint.min_value
                    )
            
            if constraint.max_value is not None:
                invalid_max = series > constraint.max_value
                if invalid_max.any():
                    invalid_values = series[invalid_max].tolist()
                    result.add_error(
                        field=field_name,
                        message=f"Values above maximum {constraint.max_value}: {invalid_values[:5]}",
                        error_type="range_validation_error",
                        constraint_type="max_value",
                        max_value=constraint.max_value
                    )
        
        # Length validation for string types
        if constraint.data_type == DataType.STRING:
            if constraint.min_length is not None:
                invalid_min_len = series.str.len() < constraint.min_length
                if invalid_min_len.any():
                    result.add_error(
                        field=field_name,
                        message=f"Values shorter than minimum length {constraint.min_length}",
                        error_type="length_validation_error",
                        constraint_type="min_length",
                        min_length=constraint.min_length
                    )
            
            if constraint.max_length is not None:
                invalid_max_len = series.str.len() > constraint.max_length
                if invalid_max_len.any():
                    result.add_error(
                        field=field_name,
                        message=f"Values longer than maximum length {constraint.max_length}",
                        error_type="length_validation_error",
                        constraint_type="max_length",
                        max_length=constraint.max_length
                    )
        
        # Allowed values validation
        if constraint.allowed_values is not None:
            invalid_values = ~series.isin(constraint.allowed_values)
            if invalid_values.any():
                bad_values = series[invalid_values].unique().tolist()
                result.add_error(
                    field=field_name,
                    message=f"Invalid values found: {bad_values}. Allowed: {constraint.allowed_values}",
                    error_type="allowed_values_error",
                    invalid_values=bad_values,
                    allowed_values=constraint.allowed_values
                )
        
        # Pattern validation for strings
        if constraint.pattern is not None and constraint.data_type == DataType.STRING:
            pattern = re.compile(constraint.pattern)
            invalid_pattern = ~series.str.match(pattern, na=False)
            if invalid_pattern.any():
                bad_values = series[invalid_pattern].tolist()
                result.add_error(
                    field=field_name,
                    message=f"Values don't match pattern {constraint.pattern}: {bad_values[:5]}",
                    error_type="pattern_validation_error",
                    pattern=constraint.pattern,
                    invalid_values=bad_values[:5]
                )
        
        # Custom validator
        if constraint.custom_validator is not None:
            try:
                invalid_custom = ~series.apply(constraint.custom_validator)
                if invalid_custom.any():
                    bad_values = series[invalid_custom].tolist()
                    result.add_error(
                        field=field_name,
                        message=f"Custom validation failed for values: {bad_values[:5]}",
                        error_type="custom_validation_error",
                        invalid_values=bad_values[:5]
                    )
            except Exception as e:
                result.add_error(
                    field=field_name,
                    message=f"Custom validator error: {str(e)}",
                    error_type="custom_validator_error"
                )


class RangeValidator(BaseValidator):
    """Validates numeric values against specified ranges."""
    
    def __init__(self, range_config: Dict[str, Dict[str, float]]):
        """
        Initialize range validator.
        
        Args:
            range_config: Dict mapping field names to range constraints
                         e.g., {'age': {'min': 18, 'max': 100}}
        """
        super().__init__("range_validator", "Validates numeric ranges")
        self.range_config = range_config
    
    def validate(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> ValidationResult:
        """Validate numeric ranges."""
        result = ValidationResult(is_valid=True)
        
        for field_name, constraints in self.range_config.items():
            if isinstance(data, pd.DataFrame):
                if field_name not in data.columns:
                    continue
                values = data[field_name].dropna()
            elif isinstance(data, dict):
                if field_name not in data:
                    continue
                values = pd.Series([data[field_name]], name=field_name).dropna()
            else:
                continue
            
            if len(values) == 0:
                continue
            
            self._validate_range_constraints(values, field_name, constraints, result)
        
        return result
    
    def _validate_range_constraints(self, values: pd.Series, field_name: str,
                                  constraints: Dict[str, float], result: ValidationResult):
        """Validate range constraints for a field."""
        min_val = constraints.get('min')
        max_val = constraints.get('max')
        
        if min_val is not None:
            below_min = values < min_val
            if below_min.any():
                invalid_values = values[below_min].tolist()
                result.add_error(
                    field=field_name,
                    message=f"Values below minimum {min_val}: {invalid_values[:5]}",
                    error_type="range_validation_error",
                    min_value=min_val,
                    invalid_count=below_min.sum()
                )
        
        if max_val is not None:
            above_max = values > max_val
            if above_max.any():
                invalid_values = values[above_max].tolist()
                result.add_error(
                    field=field_name,
                    message=f"Values above maximum {max_val}: {invalid_values[:5]}",
                    error_type="range_validation_error",
                    max_value=max_val,
                    invalid_count=above_max.sum()
                )


class FormatValidator(BaseValidator):
    """Validates data formats using regular expressions and custom rules."""
    
    def __init__(self):
        super().__init__("format_validator", "Validates data formats")
        self.patterns = self._define_patterns()
    
    def _define_patterns(self) -> Dict[str, Pattern]:
        """Define common validation patterns."""
        return {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'phone': re.compile(r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'),
            'ssn': re.compile(r'^\d{3}-?\d{2}-?\d{4}$'),
            'zip_code': re.compile(r'^\d{5}(-\d{4})?$'),
            'credit_card': re.compile(r'^[0-9]{13,19}$'),
            'routing_number': re.compile(r'^[0-9]{9}$'),
            'account_number': re.compile(r'^[0-9]{8,17}$')
        }
    
    def validate(self, data: Union[pd.DataFrame, Dict[str, Any]],
                format_config: Dict[str, str]) -> ValidationResult:
        """
        Validate data formats.
        
        Args:
            data: Data to validate
            format_config: Dict mapping field names to format types
                          e.g., {'email': 'email', 'phone': 'phone'}
        """
        result = ValidationResult(is_valid=True)
        
        for field_name, format_type in format_config.items():
            if format_type not in self.patterns:
                result.add_warning(
                    field=field_name,
                    message=f"Unknown format type: {format_type}"
                )
                continue
            
            if isinstance(data, pd.DataFrame):
                if field_name not in data.columns:
                    continue
                values = data[field_name].dropna().astype(str)
            elif isinstance(data, dict):
                if field_name not in data:
                    continue
                values = pd.Series([str(data[field_name])], name=field_name)
            else:
                continue
            
            if len(values) == 0:
                continue
            
            self._validate_format(values, field_name, format_type, result)
        
        return result
    
    def _validate_format(self, values: pd.Series, field_name: str,
                        format_type: str, result: ValidationResult):
        """Validate format for a field."""
        pattern = self.patterns[format_type]
        invalid_format = ~values.str.match(pattern, na=False)
        
        if invalid_format.any():
            invalid_values = values[invalid_format].tolist()
            result.add_error(
                field=field_name,
                message=f"Invalid {format_type} format: {invalid_values[:5]}",
                error_type="format_validation_error",
                format_type=format_type,
                invalid_values=invalid_values[:5],
                invalid_count=invalid_format.sum()
            )
    
    def add_custom_pattern(self, name: str, pattern: str):
        """Add custom validation pattern."""
        self.patterns[name] = re.compile(pattern)


class BusinessRuleValidator(BaseValidator):
    """Validates custom business rules."""
    
    def __init__(self):
        super().__init__("business_rule_validator", "Validates business logic rules")
        self.rules = {}
    
    def add_rule(self, name: str, rule_func: Callable, description: str = ""):
        """
        Add a business rule.
        
        Args:
            name: Rule name
            rule_func: Function that takes data and returns bool (True if valid)
            description: Rule description
        """
        self.rules[name] = {
            'function': rule_func,
            'description': description
        }
    
    def validate(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> ValidationResult:
        """Validate business rules."""
        result = ValidationResult(is_valid=True)
        
        for rule_name, rule_info in self.rules.items():
            try:
                is_valid = rule_info['function'](data)
                if not is_valid:
                    result.add_error(
                        field="business_rules",
                        message=f"Business rule '{rule_name}' failed",
                        error_type="business_rule_error",
                        rule_name=rule_name,
                        rule_description=rule_info['description']
                    )
            except Exception as e:
                result.add_error(
                    field="business_rules",
                    message=f"Business rule '{rule_name}' execution failed: {str(e)}",
                    error_type="business_rule_execution_error",
                    rule_name=rule_name
                )
        
        return result


class DataValidator:
    """Main validator class that orchestrates all validation types."""
    
    def __init__(self, schema: Optional[LoanDataSchema] = None):
        """Initialize with schema and sub-validators."""
        self.schema = schema or LoanDataSchema()
        self.schema_validator = SchemaValidator(self.schema)
        self.range_validator = None
        self.format_validator = FormatValidator()
        self.business_rule_validator = BusinessRuleValidator()
        
        # Initialize range validator with schema constraints
        self._setup_range_validator()
    
    def _setup_range_validator(self):
        """Setup range validator based on schema."""
        range_config = {}
        for field_name, constraint in self.schema.fields.items():
            if constraint.min_value is not None or constraint.max_value is not None:
                range_config[field_name] = {}
                if constraint.min_value is not None:
                    range_config[field_name]['min'] = constraint.min_value
                if constraint.max_value is not None:
                    range_config[field_name]['max'] = constraint.max_value
        
        if range_config:
            self.range_validator = RangeValidator(range_config)
    
    def validate_all(self, data: Union[pd.DataFrame, Dict[str, Any]],
                    include_schema: bool = True,
                    include_ranges: bool = True,
                    include_formats: bool = True,
                    include_business_rules: bool = True,
                    format_config: Optional[Dict[str, str]] = None) -> ValidationResult:
        """
        Perform comprehensive validation.
        
        Args:
            data: Data to validate
            include_schema: Whether to include schema validation
            include_ranges: Whether to include range validation
            include_formats: Whether to include format validation
            include_business_rules: Whether to include business rule validation
            format_config: Format validation configuration
            
        Returns:
            Combined validation result
        """
        combined_result = ValidationResult(is_valid=True)
        
        # Schema validation
        if include_schema:
            schema_result = self.schema_validator.validate(data)
            self._merge_results(combined_result, schema_result)
            
            # Stop if schema validation fails critically
            if not schema_result.is_valid and len(schema_result.errors) > 10:
                combined_result.add_error(
                    field="validation",
                    message="Too many schema errors, skipping other validations",
                    error_type="validation_aborted"
                )
                return combined_result
        
        # Range validation
        if include_ranges and self.range_validator:
            range_result = self.range_validator.validate(data)
            self._merge_results(combined_result, range_result)
        
        # Format validation
        if include_formats and format_config:
            format_result = self.format_validator.validate(data, format_config)
            self._merge_results(combined_result, format_result)
        
        # Business rule validation
        if include_business_rules and self.business_rule_validator.rules:
            business_result = self.business_rule_validator.validate(data)
            self._merge_results(combined_result, business_result)
        
        # Add summary metadata
        combined_result.metadata.update({
            'total_validations': sum([
                include_schema, include_ranges, include_formats, include_business_rules
            ]),
            'validation_timestamp': datetime.now().isoformat()
        })
        
        return combined_result
    
    def _merge_results(self, target: ValidationResult, source: ValidationResult):
        """Merge validation results."""
        target.errors.extend(source.errors)
        target.warnings.extend(source.warnings)
        target.field_results.update(source.field_results)
        target.metadata.update(source.metadata)
        
        if not source.is_valid:
            target.is_valid = False
    
    def add_business_rule(self, name: str, rule_func: Callable, description: str = ""):
        """Add business rule to validator."""
        self.business_rule_validator.add_rule(name, rule_func, description)
    
    def add_format_pattern(self, name: str, pattern: str):
        """Add custom format pattern."""
        self.format_validator.add_custom_pattern(name, pattern)