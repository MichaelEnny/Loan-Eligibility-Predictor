"""
Data schema definitions for loan eligibility prediction system.

Defines the expected structure, types, and constraints for loan application data
to ensure consistency and data quality throughout the ML pipeline.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types for validation."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class FieldConstraint:
    """Defines constraints for a single field."""
    
    name: str
    data_type: DataType
    required: bool = True
    nullable: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate constraint parameters."""
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError(f"min_value ({self.min_value}) cannot be greater than max_value ({self.max_value})")
        
        if self.min_length is not None and self.max_length is not None:
            if self.min_length > self.max_length:
                raise ValueError(f"min_length ({self.min_length}) cannot be greater than max_length ({self.max_length})")


@dataclass
class ValidationResult:
    """Result of data validation operation."""
    
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    field_results: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, field: str, message: str, value: Any = None, 
                  error_type: str = "validation_error", **kwargs):
        """Add validation error."""
        self.errors.append({
            'field': field,
            'message': message,
            'value': value,
            'error_type': error_type,
            'severity': ValidationSeverity.ERROR.value,
            **kwargs
        })
        self.field_results[field] = False
        self.is_valid = False
    
    def add_warning(self, field: str, message: str, value: Any = None, **kwargs):
        """Add validation warning."""
        self.warnings.append({
            'field': field,
            'message': message,
            'value': value,
            'severity': ValidationSeverity.WARNING.value,
            **kwargs
        })
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of errors by type."""
        error_types = {}
        for error in self.errors:
            error_type = error.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        return error_types
    
    def get_failed_fields(self) -> List[str]:
        """Get list of fields that failed validation."""
        return [field for field, result in self.field_results.items() if not result]


class LoanDataSchema:
    """Schema definition for loan application data."""
    
    def __init__(self):
        """Initialize loan data schema with field constraints."""
        self.fields = self._define_fields()
        self.required_fields = [name for name, constraint in self.fields.items() if constraint.required]
        self.optional_fields = [name for name, constraint in self.fields.items() if not constraint.required]
    
    def _define_fields(self) -> Dict[str, FieldConstraint]:
        """Define all field constraints for loan data."""
        return {
            # Personal Information
            'age': FieldConstraint(
                name='age',
                data_type=DataType.INTEGER,
                required=True,
                min_value=18,
                max_value=100,
                description='Applicant age in years'
            ),
            
            'gender': FieldConstraint(
                name='gender',
                data_type=DataType.CATEGORICAL,
                required=True,
                allowed_values=['Male', 'Female', 'Other'],
                description='Applicant gender'
            ),
            
            'marital_status': FieldConstraint(
                name='marital_status',
                data_type=DataType.CATEGORICAL,
                required=True,
                allowed_values=['Single', 'Married', 'Divorced', 'Widowed'],
                description='Marital status'
            ),
            
            'education': FieldConstraint(
                name='education',
                data_type=DataType.CATEGORICAL,
                required=True,
                allowed_values=['High School', 'Some College', "Bachelor's", 'Advanced'],
                description='Education level'
            ),
            
            # Employment Information
            'employment_status': FieldConstraint(
                name='employment_status',
                data_type=DataType.CATEGORICAL,
                required=True,
                allowed_values=['Employed', 'Self-Employed', 'Unemployed', 'Retired', 'Student'],
                description='Employment status'
            ),
            
            'years_employed': FieldConstraint(
                name='years_employed',
                data_type=DataType.FLOAT,
                required=True,
                min_value=0.0,
                max_value=50.0,
                description='Years of employment experience'
            ),
            
            # Financial Information
            'annual_income': FieldConstraint(
                name='annual_income',
                data_type=DataType.FLOAT,
                required=True,
                min_value=0.0,
                max_value=10000000.0,
                description='Annual income in USD'
            ),
            
            'monthly_income': FieldConstraint(
                name='monthly_income',
                data_type=DataType.FLOAT,
                required=True,
                min_value=0.0,
                max_value=1000000.0,
                description='Monthly income in USD'
            ),
            
            'credit_score': FieldConstraint(
                name='credit_score',
                data_type=DataType.FLOAT,
                required=True,
                min_value=300.0,
                max_value=850.0,
                description='Credit score (FICO scale)'
            ),
            
            'credit_history_length': FieldConstraint(
                name='credit_history_length',
                data_type=DataType.FLOAT,
                required=True,
                min_value=0.0,
                max_value=50.0,
                description='Length of credit history in years'
            ),
            
            'num_credit_accounts': FieldConstraint(
                name='num_credit_accounts',
                data_type=DataType.INTEGER,
                required=True,
                min_value=0,
                max_value=50,
                description='Number of active credit accounts'
            ),
            
            'existing_debt': FieldConstraint(
                name='existing_debt',
                data_type=DataType.FLOAT,
                required=True,
                min_value=0.0,
                max_value=10000000.0,
                description='Total existing debt in USD'
            ),
            
            'monthly_debt_payments': FieldConstraint(
                name='monthly_debt_payments',
                data_type=DataType.FLOAT,
                required=True,
                min_value=0.0,
                max_value=100000.0,
                description='Monthly debt payments in USD'
            ),
            
            'debt_to_income_ratio': FieldConstraint(
                name='debt_to_income_ratio',
                data_type=DataType.FLOAT,
                required=True,
                min_value=0.0,
                max_value=1.0,
                description='Debt-to-income ratio (0-1)'
            ),
            
            # Loan Information
            'loan_amount': FieldConstraint(
                name='loan_amount',
                data_type=DataType.FLOAT,
                required=True,
                min_value=1000.0,
                max_value=10000000.0,
                description='Requested loan amount in USD'
            ),
            
            'loan_term_months': FieldConstraint(
                name='loan_term_months',
                data_type=DataType.INTEGER,
                required=True,
                min_value=6,
                max_value=480,
                description='Loan term in months'
            ),
            
            'loan_purpose': FieldConstraint(
                name='loan_purpose',
                data_type=DataType.CATEGORICAL,
                required=True,
                allowed_values=['Home Improvement', 'Debt Consolidation', 'Auto', 'Personal', 
                              'Business', 'Medical', 'Education', 'Other'],
                description='Purpose of the loan'
            ),
            
            # Property Information
            'owns_property': FieldConstraint(
                name='owns_property',
                data_type=DataType.BOOLEAN,
                required=True,
                description='Whether applicant owns property'
            ),
            
            'property_value': FieldConstraint(
                name='property_value',
                data_type=DataType.FLOAT,
                required=False,
                nullable=True,
                min_value=0.0,
                max_value=100000000.0,
                description='Property value in USD (if owned)'
            ),
            
            # Banking Information
            'has_bank_account': FieldConstraint(
                name='has_bank_account',
                data_type=DataType.BOOLEAN,
                required=True,
                description='Whether applicant has bank account'
            ),
            
            'years_with_bank': FieldConstraint(
                name='years_with_bank',
                data_type=DataType.FLOAT,
                required=False,
                nullable=True,
                min_value=0.0,
                max_value=50.0,
                description='Years with current bank'
            ),
            
            # Credit History
            'previous_loans': FieldConstraint(
                name='previous_loans',
                data_type=DataType.INTEGER,
                required=True,
                min_value=0,
                max_value=20,
                description='Number of previous loans'
            ),
            
            'previous_loan_defaults': FieldConstraint(
                name='previous_loan_defaults',
                data_type=DataType.INTEGER,
                required=True,
                min_value=0,
                max_value=10,
                description='Number of previous loan defaults'
            ),
            
            # Geographic Information
            'state': FieldConstraint(
                name='state',
                data_type=DataType.CATEGORICAL,
                required=True,
                allowed_values=['CA', 'TX', 'FL', 'NY', 'Other'],
                description='State of residence'
            ),
            
            'area_type': FieldConstraint(
                name='area_type',
                data_type=DataType.CATEGORICAL,
                required=True,
                allowed_values=['Urban', 'Suburban', 'Rural'],
                description='Area type of residence'
            ),
            
            # Co-applicant Information
            'has_coapplicant': FieldConstraint(
                name='has_coapplicant',
                data_type=DataType.BOOLEAN,
                required=True,
                description='Whether there is a co-applicant'
            ),
            
            'coapplicant_income': FieldConstraint(
                name='coapplicant_income',
                data_type=DataType.FLOAT,
                required=False,
                nullable=True,
                min_value=0.0,
                max_value=10000000.0,
                description='Co-applicant income in USD'
            ),
            
            'total_household_income': FieldConstraint(
                name='total_household_income',
                data_type=DataType.FLOAT,
                required=True,
                min_value=0.0,
                max_value=20000000.0,
                description='Total household income in USD'
            ),
            
            # Target Variable
            'loan_approved': FieldConstraint(
                name='loan_approved',
                data_type=DataType.INTEGER,
                required=False,
                allowed_values=[0, 1],
                description='Loan approval status (0=denied, 1=approved)'
            ),
            
            # Metadata
            'application_date': FieldConstraint(
                name='application_date',
                data_type=DataType.DATETIME,
                required=False,
                nullable=True,
                description='Application submission date'
            )
        }
    
    def get_field_constraint(self, field_name: str) -> Optional[FieldConstraint]:
        """Get constraint for specific field."""
        return self.fields.get(field_name)
    
    def get_required_fields(self) -> List[str]:
        """Get list of required field names."""
        return self.required_fields.copy()
    
    def get_optional_fields(self) -> List[str]:
        """Get list of optional field names."""
        return self.optional_fields.copy()
    
    def get_fields_by_type(self, data_type: DataType) -> List[str]:
        """Get fields of specific data type."""
        return [name for name, constraint in self.fields.items() 
                if constraint.data_type == data_type]
    
    def get_categorical_fields(self) -> List[str]:
        """Get all categorical field names."""
        return self.get_fields_by_type(DataType.CATEGORICAL)
    
    def get_numerical_fields(self) -> List[str]:
        """Get all numerical field names (integer and float)."""
        return (self.get_fields_by_type(DataType.INTEGER) + 
                self.get_fields_by_type(DataType.FLOAT))
    
    def validate_schema(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> ValidationResult:
        """Validate data against schema."""
        result = ValidationResult(is_valid=True)
        
        if isinstance(data, pd.DataFrame):
            field_names = set(data.columns)
        elif isinstance(data, dict):
            field_names = set(data.keys())
        else:
            result.add_error(
                field='schema',
                message=f"Unsupported data type: {type(data)}. Expected DataFrame or dict.",
                error_type='schema_error'
            )
            return result
        
        # Check for missing required fields
        missing_required = set(self.required_fields) - field_names
        if missing_required:
            result.add_error(
                field='schema',
                message=f"Missing required fields: {sorted(missing_required)}",
                error_type='missing_required_fields',
                missing_fields=sorted(missing_required)
            )
        
        # Check for unexpected fields
        expected_fields = set(self.fields.keys())
        unexpected_fields = field_names - expected_fields
        if unexpected_fields:
            result.add_warning(
                field='schema',
                message=f"Unexpected fields found: {sorted(unexpected_fields)}",
                unexpected_fields=sorted(unexpected_fields)
            )
        
        # Set field validation results for existing fields
        for field_name in field_names:
            if field_name in self.fields:
                result.field_results[field_name] = True
        
        # Add metadata
        result.metadata.update({
            'total_fields': len(field_names),
            'expected_fields': len(self.fields),
            'required_fields': len(self.required_fields),
            'missing_required_count': len(missing_required),
            'unexpected_fields_count': len(unexpected_fields)
        })
        
        return result