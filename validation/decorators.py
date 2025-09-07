"""
Validation decorators for seamless integration with existing functions.

Provides decorators that can be applied to functions to automatically
validate inputs and outputs, ensuring data quality throughout the ML pipeline.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import functools
import logging
import pandas as pd
from datetime import datetime

from .schema import LoanDataSchema, ValidationResult
from .validators import DataValidator
from .exceptions import ValidationError, ValidationSummaryError


logger = logging.getLogger(__name__)


def validate_input(schema: Optional[LoanDataSchema] = None,
                  required_fields: Optional[List[str]] = None,
                  strict_mode: bool = False,
                  convert_types: bool = True,
                  log_validation: bool = True):
    """
    Decorator to validate function inputs against schema.
    
    Args:
        schema: Data schema to validate against (uses default if None)
        required_fields: List of required field names
        strict_mode: If True, raises exception on validation failure
        convert_types: Whether to attempt type conversion
        log_validation: Whether to log validation results
        
    Raises:
        ValidationError: If strict_mode=True and validation fails
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract data from first argument (assuming it's the data)
            if not args:
                return func(*args, **kwargs)
            
            data = args[0]
            if not isinstance(data, (pd.DataFrame, dict)):
                return func(*args, **kwargs)
            
            # Setup validator
            validation_schema = schema or LoanDataSchema()
            validator = DataValidator(validation_schema)
            
            # Validate data
            result = validator.validate_all(
                data,
                include_schema=True,
                include_ranges=True,
                include_formats=False,  # Skip format validation by default
                include_business_rules=False
            )
            
            # Check required fields if specified
            if required_fields:
                missing_required = _check_required_fields(data, required_fields)
                if missing_required:
                    result.add_error(
                        field="required_fields",
                        message=f"Missing required fields: {missing_required}",
                        error_type="missing_required_fields"
                    )
                    result.is_valid = False
            
            # Log validation results
            if log_validation:
                if result.is_valid:
                    logger.info(f"Input validation passed for {func.__name__}")
                else:
                    logger.warning(f"Input validation failed for {func.__name__}: "
                                 f"{len(result.errors)} errors, {len(result.warnings)} warnings")
            
            # Handle validation failure
            if not result.is_valid and strict_mode:
                error_msg = f"Input validation failed for {func.__name__}"
                raise ValidationSummaryError(
                    message=error_msg,
                    errors=[ValidationError(error['message'], error.get('field'))
                           for error in result.errors]
                )
            
            # Add validation result to kwargs for function access
            kwargs['_validation_result'] = result
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_schema(schema: Optional[LoanDataSchema] = None,
                   strict_mode: bool = True):
    """
    Decorator to validate data schema only.
    
    Args:
        schema: Data schema to validate against
        strict_mode: If True, raises exception on validation failure
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                return func(*args, **kwargs)
            
            data = args[0]
            if not isinstance(data, (pd.DataFrame, dict)):
                return func(*args, **kwargs)
            
            validation_schema = schema or LoanDataSchema()
            result = validation_schema.validate_schema(data)
            
            if not result.is_valid and strict_mode:
                error_msg = f"Schema validation failed for {func.__name__}"
                raise ValidationSummaryError(
                    message=error_msg,
                    errors=[ValidationError(error['message'], error.get('field'))
                           for error in result.errors]
                )
            
            kwargs['_schema_validation_result'] = result
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_fields(*required_fields: str, strict_mode: bool = True):
    """
    Decorator to ensure required fields are present.
    
    Args:
        required_fields: Field names that must be present
        strict_mode: If True, raises exception if fields missing
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                return func(*args, **kwargs)
            
            data = args[0]
            if not isinstance(data, (pd.DataFrame, dict)):
                return func(*args, **kwargs)
            
            missing_fields = _check_required_fields(data, list(required_fields))
            
            if missing_fields:
                error_msg = f"Required fields missing in {func.__name__}: {missing_fields}"
                logger.error(error_msg)
                
                if strict_mode:
                    raise ValidationError(
                        message=error_msg,
                        field="required_fields",
                        details={'missing_fields': missing_fields}
                    )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_output(expected_columns: Optional[List[str]] = None,
                   expected_types: Optional[Dict[str, type]] = None,
                   min_rows: Optional[int] = None,
                   max_rows: Optional[int] = None):
    """
    Decorator to validate function outputs.
    
    Args:
        expected_columns: Expected column names for DataFrame output
        expected_types: Expected data types for columns
        min_rows: Minimum number of rows expected
        max_rows: Maximum number of rows expected
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if isinstance(result, pd.DataFrame):
                _validate_dataframe_output(
                    result, func.__name__, expected_columns,
                    expected_types, min_rows, max_rows
                )
            
            return result
        
        return wrapper
    return decorator


def log_validation_metrics(include_timing: bool = True,
                          include_data_summary: bool = True):
    """
    Decorator to log validation metrics and performance.
    
    Args:
        include_timing: Whether to log execution timing
        include_data_summary: Whether to log data summary statistics
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now() if include_timing else None
            
            # Log input data summary
            if include_data_summary and args:
                data = args[0]
                if isinstance(data, pd.DataFrame):
                    logger.info(f"{func.__name__} input: {data.shape[0]} rows, "
                              f"{data.shape[1]} columns")
                elif isinstance(data, dict):
                    logger.info(f"{func.__name__} input: dict with {len(data)} keys")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log timing
            if include_timing and start_time:
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"{func.__name__} completed in {duration:.3f} seconds")
            
            # Log validation results if available
            validation_result = kwargs.get('_validation_result')
            if validation_result:
                logger.info(f"{func.__name__} validation: "
                          f"{len(validation_result.errors)} errors, "
                          f"{len(validation_result.warnings)} warnings")
            
            return result
        
        return wrapper
    return decorator


def handle_validation_errors(fallback_value: Any = None,
                           log_errors: bool = True,
                           reraise_critical: bool = True):
    """
    Decorator to handle validation errors gracefully.
    
    Args:
        fallback_value: Value to return if validation fails
        log_errors: Whether to log validation errors
        reraise_critical: Whether to reraise critical validation errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except ValidationError as e:
                if log_errors:
                    logger.error(f"Validation error in {func.__name__}: {e.message}")
                
                # Reraise critical errors
                if reraise_critical and _is_critical_error(e):
                    raise
                
                return fallback_value
            
            except ValidationSummaryError as e:
                if log_errors:
                    logger.error(f"Multiple validation errors in {func.__name__}: "
                               f"{e.error_count} errors")
                
                # Reraise if too many critical errors
                if reraise_critical and e.error_count > 10:
                    raise
                
                return fallback_value
        
        return wrapper
    return decorator


def validate_and_clean(schema: Optional[LoanDataSchema] = None,
                      drop_invalid_rows: bool = False,
                      fill_missing: bool = False,
                      fill_values: Optional[Dict[str, Any]] = None):
    """
    Decorator to validate and clean data automatically.
    
    Args:
        schema: Data schema to validate against
        drop_invalid_rows: Whether to drop rows with validation errors
        fill_missing: Whether to fill missing values
        fill_values: Dictionary of fill values for specific columns
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args or not isinstance(args[0], pd.DataFrame):
                return func(*args, **kwargs)
            
            data = args[0].copy()
            validation_schema = schema or LoanDataSchema()
            validator = DataValidator(validation_schema)
            
            # Validate data
            result = validator.validate_all(data)
            
            # Clean data based on validation results
            if not result.is_valid:
                if drop_invalid_rows:
                    # This is a simplified approach - in practice, you'd want
                    # more sophisticated row-level validation
                    data = data.dropna()
                
                if fill_missing:
                    fill_vals = fill_values or {}
                    data = data.fillna(fill_vals)
            
            # Call function with cleaned data
            new_args = (data,) + args[1:]
            kwargs['_validation_result'] = result
            
            return func(*new_args, **kwargs)
        
        return wrapper
    return decorator


# Utility functions
def _check_required_fields(data: Union[pd.DataFrame, dict], 
                         required_fields: List[str]) -> List[str]:
    """Check for missing required fields."""
    if isinstance(data, pd.DataFrame):
        available_fields = set(data.columns)
    elif isinstance(data, dict):
        available_fields = set(data.keys())
    else:
        return required_fields
    
    return [field for field in required_fields if field not in available_fields]


def _validate_dataframe_output(df: pd.DataFrame, func_name: str,
                             expected_columns: Optional[List[str]],
                             expected_types: Optional[Dict[str, type]],
                             min_rows: Optional[int],
                             max_rows: Optional[int]):
    """Validate DataFrame output against expectations."""
    errors = []
    
    # Check columns
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing expected columns: {missing_cols}")
    
    # Check types
    if expected_types:
        for col, expected_type in expected_types.items():
            if col in df.columns and not df[col].dtype == expected_type:
                errors.append(f"Column '{col}' has type {df[col].dtype}, "
                            f"expected {expected_type}")
    
    # Check row count
    if min_rows is not None and len(df) < min_rows:
        errors.append(f"Output has {len(df)} rows, expected at least {min_rows}")
    
    if max_rows is not None and len(df) > max_rows:
        errors.append(f"Output has {len(df)} rows, expected at most {max_rows}")
    
    if errors:
        error_msg = f"Output validation failed for {func_name}: {'; '.join(errors)}"
        raise ValidationError(error_msg, field="output_validation")


def _is_critical_error(error: ValidationError) -> bool:
    """Determine if a validation error is critical."""
    critical_types = [
        'schema_error',
        'missing_required_fields',
        'type_conversion_error'
    ]
    
    return error.details.get('error_type') in critical_types