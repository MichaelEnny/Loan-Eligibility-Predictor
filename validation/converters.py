"""
Data type conversion utilities for the validation framework.

Provides safe and robust data type conversion with proper error handling
and support for various input formats commonly encountered in loan data.
"""

from typing import Any, Union, Optional, Dict, List, Callable
import pandas as pd
import numpy as np
from datetime import datetime, date
import re
import logging
from decimal import Decimal, InvalidOperation

from .exceptions import DataTypeValidationError
from .schema import DataType


logger = logging.getLogger(__name__)


class DataTypeConverter:
    """Handles safe data type conversions with validation."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize converter.
        
        Args:
            strict_mode: If True, raises errors for failed conversions.
                        If False, returns None for failed conversions.
        """
        self.strict_mode = strict_mode
        self._conversion_functions = {
            DataType.INTEGER: self._to_integer,
            DataType.FLOAT: self._to_float,
            DataType.STRING: self._to_string,
            DataType.BOOLEAN: self._to_boolean,
            DataType.DATETIME: self._to_datetime,
            DataType.CATEGORICAL: self._to_categorical
        }
    
    def convert_value(self, value: Any, target_type: DataType, 
                     field_name: Optional[str] = None) -> Any:
        """
        Convert a single value to target type.
        
        Args:
            value: Value to convert
            target_type: Target data type
            field_name: Name of field for error reporting
            
        Returns:
            Converted value or None if conversion fails in non-strict mode
            
        Raises:
            DataTypeValidationError: If conversion fails in strict mode
        """
        if pd.isna(value) or value is None:
            return None
        
        try:
            converter = self._conversion_functions.get(target_type)
            if converter is None:
                raise ValueError(f"Unsupported data type: {target_type}")
            
            return converter(value)
        
        except Exception as e:
            error_msg = f"Failed to convert '{value}' to {target_type.value}"
            if field_name:
                error_msg = f"Field '{field_name}': {error_msg}"
            
            if self.strict_mode:
                raise DataTypeValidationError(
                    message=error_msg,
                    field=field_name,
                    value=value,
                    expected_type=target_type.value,
                    actual_type=type(value).__name__
                )
            else:
                logger.warning(f"{error_msg}: {e}")
                return None
    
    def convert_series(self, series: pd.Series, target_type: DataType,
                      field_name: Optional[str] = None) -> pd.Series:
        """Convert pandas Series to target type."""
        if field_name is None:
            field_name = series.name
        
        converted_values = []
        for idx, value in series.items():
            converted_value = self.convert_value(value, target_type, field_name)
            converted_values.append(converted_value)
        
        return pd.Series(converted_values, index=series.index, name=series.name)
    
    def convert_dataframe(self, df: pd.DataFrame, 
                         type_mapping: Dict[str, DataType]) -> pd.DataFrame:
        """Convert DataFrame columns according to type mapping."""
        df_converted = df.copy()
        
        for column, target_type in type_mapping.items():
            if column in df_converted.columns:
                df_converted[column] = self.convert_series(
                    df_converted[column], target_type, column
                )
        
        return df_converted
    
    def _to_integer(self, value: Any) -> int:
        """Convert value to integer."""
        if isinstance(value, (int, np.integer)):
            return int(value)
        
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                raise ValueError("Cannot convert NaN to integer")
            if value != int(value):
                raise ValueError(f"Float {value} cannot be converted to integer without loss")
            return int(value)
        
        if isinstance(value, str):
            # Clean string
            cleaned = value.strip().replace(',', '')
            
            # Handle percentage
            if cleaned.endswith('%'):
                cleaned = cleaned[:-1]
                return int(float(cleaned))
            
            # Handle decimal strings
            try:
                float_val = float(cleaned)
                if float_val != int(float_val):
                    raise ValueError(f"String '{value}' represents non-integer value")
                return int(float_val)
            except ValueError:
                raise ValueError(f"Cannot convert string '{value}' to integer")
        
        if isinstance(value, bool):
            return int(value)
        
        if isinstance(value, Decimal):
            if value % 1 != 0:
                raise ValueError(f"Decimal {value} has fractional part")
            return int(value)
        
        raise ValueError(f"Cannot convert {type(value).__name__} to integer")
    
    def _to_float(self, value: Any) -> float:
        """Convert value to float."""
        if isinstance(value, (float, np.floating)):
            return float(value)
        
        if isinstance(value, (int, np.integer)):
            return float(value)
        
        if isinstance(value, str):
            # Clean string
            cleaned = value.strip().replace(',', '')
            
            # Handle percentage
            if cleaned.endswith('%'):
                cleaned = cleaned[:-1]
                return float(cleaned) / 100.0
            
            # Handle currency symbols
            cleaned = re.sub(r'[$€£¥]', '', cleaned)
            
            # Convert to float
            try:
                return float(cleaned)
            except ValueError:
                raise ValueError(f"Cannot convert string '{value}' to float")
        
        if isinstance(value, bool):
            return float(value)
        
        if isinstance(value, Decimal):
            return float(value)
        
        raise ValueError(f"Cannot convert {type(value).__name__} to float")
    
    def _to_string(self, value: Any) -> str:
        """Convert value to string."""
        if isinstance(value, str):
            return value.strip()
        
        if pd.isna(value):
            return ""
        
        return str(value)
    
    def _to_boolean(self, value: Any) -> bool:
        """Convert value to boolean."""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, (int, np.integer)):
            if value in [0, 1]:
                return bool(value)
            else:
                raise ValueError(f"Integer {value} cannot be converted to boolean (must be 0 or 1)")
        
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                raise ValueError("Cannot convert NaN to boolean")
            if value in [0.0, 1.0]:
                return bool(value)
            else:
                raise ValueError(f"Float {value} cannot be converted to boolean (must be 0.0 or 1.0)")
        
        if isinstance(value, str):
            cleaned = value.strip().lower()
            
            # True values
            if cleaned in ['true', 'yes', 'y', '1', 'on', 'enabled']:
                return True
            
            # False values
            if cleaned in ['false', 'no', 'n', '0', 'off', 'disabled', '']:
                return False
            
            raise ValueError(f"Cannot convert string '{value}' to boolean")
        
        raise ValueError(f"Cannot convert {type(value).__name__} to boolean")
    
    def _to_datetime(self, value: Any) -> datetime:
        """Convert value to datetime."""
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        
        if isinstance(value, str):
            # Try common datetime formats
            formats = [
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%m/%d/%Y %H:%M:%S',
                '%d/%m/%Y',
                '%d/%m/%Y %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S.%fZ'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(value.strip(), fmt)
                except ValueError:
                    continue
            
            # Try pandas datetime parsing as last resort
            try:
                return pd.to_datetime(value)
            except Exception:
                raise ValueError(f"Cannot parse datetime string '{value}'")
        
        if isinstance(value, (int, float)):
            # Assume unix timestamp
            try:
                return datetime.fromtimestamp(value)
            except (ValueError, OSError):
                raise ValueError(f"Cannot convert {value} to datetime (invalid timestamp)")
        
        raise ValueError(f"Cannot convert {type(value).__name__} to datetime")
    
    def _to_categorical(self, value: Any) -> str:
        """Convert value to categorical string."""
        if pd.isna(value):
            return None
        
        return str(value).strip()
    
    def infer_types(self, df: pd.DataFrame, 
                   sample_size: Optional[int] = 1000) -> Dict[str, DataType]:
        """
        Infer data types for DataFrame columns.
        
        Args:
            df: DataFrame to analyze
            sample_size: Number of rows to sample for type inference
            
        Returns:
            Dictionary mapping column names to inferred data types
        """
        if sample_size and len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df
        
        type_mapping = {}
        
        for column in sample_df.columns:
            series = sample_df[column].dropna()
            
            if len(series) == 0:
                type_mapping[column] = DataType.STRING
                continue
            
            # Check for boolean
            if self._is_boolean_like(series):
                type_mapping[column] = DataType.BOOLEAN
                continue
            
            # Check for datetime
            if self._is_datetime_like(series):
                type_mapping[column] = DataType.DATETIME
                continue
            
            # Check for integer
            if self._is_integer_like(series):
                type_mapping[column] = DataType.INTEGER
                continue
            
            # Check for float
            if self._is_float_like(series):
                type_mapping[column] = DataType.FLOAT
                continue
            
            # Default to categorical for strings with limited unique values
            unique_ratio = len(series.unique()) / len(series)
            if unique_ratio < 0.1 or len(series.unique()) < 20:
                type_mapping[column] = DataType.CATEGORICAL
            else:
                type_mapping[column] = DataType.STRING
        
        return type_mapping
    
    def _is_boolean_like(self, series: pd.Series) -> bool:
        """Check if series contains boolean-like values."""
        unique_values = set(series.astype(str).str.lower().str.strip().unique())
        boolean_values = {'true', 'false', 'yes', 'no', 'y', 'n', '0', '1', 
                         'on', 'off', 'enabled', 'disabled'}
        return unique_values.issubset(boolean_values) and len(unique_values) <= 2
    
    def _is_datetime_like(self, series: pd.Series) -> bool:
        """Check if series contains datetime-like values."""
        try:
            pd.to_datetime(series.head(10))
            return True
        except Exception:
            return False
    
    def _is_integer_like(self, series: pd.Series) -> bool:
        """Check if series contains integer-like values."""
        try:
            # Try converting to float first
            float_series = pd.to_numeric(series, errors='raise')
            # Check if all values are integers
            return (float_series == float_series.astype(int)).all()
        except Exception:
            return False
    
    def _is_float_like(self, series: pd.Series) -> bool:
        """Check if series contains float-like values."""
        try:
            pd.to_numeric(series, errors='raise')
            return True
        except Exception:
            return False