"""
Data Normalization and Standardization Utilities
Implements various scaling and normalization techniques
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import logging

logger = logging.getLogger(__name__)

class DataNormalizer:
    """
    Comprehensive data normalization with multiple scaling methods
    """
    
    def __init__(self, 
                 scaling_methods: Dict[str, str] = None,
                 categorical_encoding: str = 'onehot'):
        """
        Initialize data normalizer
        
        Args:
            scaling_methods: Dict mapping column names to scaling methods
                           Options: 'standard', 'minmax', 'robust', 'power', 'none'
            categorical_encoding: Method for categorical encoding ('onehot', 'label', 'target')
        """
        self.scaling_methods = scaling_methods or {}
        self.categorical_encoding = categorical_encoding
        self.scalers = {}
        self.encoders = {}
        self.fitted = False
        
        # Default scaling methods by data type and distribution
        self.default_methods = {
            'normal_distribution': 'standard',
            'skewed_distribution': 'power',
            'bounded_range': 'minmax',
            'outliers_present': 'robust'
        }
    
    def fit(self, df: pd.DataFrame, target_column: str = None) -> 'DataNormalizer':
        """
        Fit normalizers on the data
        
        Args:
            df: Input DataFrame
            target_column: Target column for target encoding
            
        Returns:
            Self for method chaining
        """
        self.scalers = {}
        self.encoders = {}
        self.target_column = target_column
        
        # Fit scalers for numeric columns
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                self._fit_scaler(column, df[column])
            elif df[column].dtype == 'object':
                self._fit_encoder(column, df[column], df.get(target_column) if target_column else None)
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Normalized DataFrame
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        result_df = df.copy()
        transform_log = {}
        
        # Apply scaling to numeric columns
        for column, scaler in self.scalers.items():
            if column in result_df.columns and scaler is not None:
                try:
                    original_shape = result_df[column].shape
                    scaled_data = scaler.transform(result_df[[column]])
                    result_df[column] = scaled_data.ravel()
                    
                    transform_log[column] = {
                        'type': 'scaling',
                        'method': type(scaler).__name__,
                        'original_range': (df[column].min(), df[column].max()) if column in df.columns else None
                    }
                except Exception as e:
                    logger.warning(f"Failed to scale {column}: {e}")
        
        # Apply encoding to categorical columns
        encoded_columns = []
        for column, encoder in self.encoders.items():
            if column in result_df.columns and encoder is not None:
                try:
                    if isinstance(encoder, OneHotEncoder):
                        # One-hot encoding
                        encoded_data = encoder.transform(result_df[[column]])
                        feature_names = encoder.get_feature_names_out([column])
                        encoded_df = pd.DataFrame(
                            encoded_data.toarray(), 
                            columns=feature_names,
                            index=result_df.index
                        )
                        
                        # Add encoded columns and remove original
                        result_df = pd.concat([result_df.drop(columns=[column]), encoded_df], axis=1)
                        encoded_columns.extend(feature_names)
                        
                        transform_log[column] = {
                            'type': 'onehot_encoding',
                            'new_columns': list(feature_names),
                            'n_categories': len(feature_names)
                        }
                    
                    elif isinstance(encoder, LabelEncoder):
                        # Label encoding
                        result_df[column] = encoder.transform(result_df[column])
                        transform_log[column] = {
                            'type': 'label_encoding',
                            'n_categories': len(encoder.classes_)
                        }
                    
                    else:
                        # Target encoding or other custom encoders
                        result_df[column] = encoder.transform(result_df[[column]])
                        transform_log[column] = {
                            'type': 'target_encoding'
                        }
                
                except Exception as e:
                    logger.warning(f"Failed to encode {column}: {e}")
        
        if transform_log:
            logger.info(f"Normalization summary: {transform_log}")
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """
        Fit and transform in one step
        
        Args:
            df: Input DataFrame
            target_column: Target column for target encoding
            
        Returns:
            Normalized DataFrame
        """
        return self.fit(df, target_column).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale
        
        Args:
            df: Normalized DataFrame
            columns: Columns to inverse transform (if None, all fitted columns)
            
        Returns:
            DataFrame with original scales
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse transform")
        
        result_df = df.copy()
        
        if columns is None:
            columns = list(self.scalers.keys())
        
        for column in columns:
            if column in self.scalers and column in result_df.columns:
                scaler = self.scalers[column]
                if scaler is not None:
                    try:
                        original_data = scaler.inverse_transform(result_df[[column]])
                        result_df[column] = original_data.ravel()
                    except Exception as e:
                        logger.warning(f"Failed to inverse transform {column}: {e}")
        
        return result_df
    
    def _fit_scaler(self, column: str, series: pd.Series):
        """Fit scaler for a numeric column"""
        method = self._get_scaling_method(column, series)
        
        if method == 'none':
            self.scalers[column] = None
            return
        
        # Remove missing values for fitting
        clean_data = series.dropna().values.reshape(-1, 1)
        
        if len(clean_data) == 0:
            self.scalers[column] = None
            return
        
        try:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'power':
                # Power transformer for skewed data
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            else:
                scaler = StandardScaler()  # Default fallback
            
            scaler.fit(clean_data)
            self.scalers[column] = scaler
            
        except Exception as e:
            logger.warning(f"Failed to fit scaler for {column}: {e}")
            self.scalers[column] = None
    
    def _fit_encoder(self, column: str, series: pd.Series, target: pd.Series = None):
        """Fit encoder for a categorical column"""
        # Remove missing values for fitting
        clean_mask = series.notna()
        clean_data = series[clean_mask]
        
        if len(clean_data) == 0:
            self.encoders[column] = None
            return
        
        try:
            if self.categorical_encoding == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(clean_data.values.reshape(-1, 1))
            
            elif self.categorical_encoding == 'label':
                encoder = LabelEncoder()
                encoder.fit(clean_data)
            
            elif self.categorical_encoding == 'target' and target is not None:
                # Simple target encoding (mean of target per category)
                encoder = self._create_target_encoder(clean_data, target[clean_mask])
            
            else:
                # Default to label encoding
                encoder = LabelEncoder()
                encoder.fit(clean_data)
            
            self.encoders[column] = encoder
            
        except Exception as e:
            logger.warning(f"Failed to fit encoder for {column}: {e}")
            self.encoders[column] = None
    
    def _create_target_encoder(self, series: pd.Series, target: pd.Series):
        """Create simple target encoder"""
        class TargetEncoder:
            def __init__(self):
                self.mapping = {}
                self.global_mean = 0
            
            def fit(self, X, y):
                df = pd.DataFrame({'category': X.ravel(), 'target': y})
                self.mapping = df.groupby('category')['target'].mean().to_dict()
                self.global_mean = y.mean()
                return self
            
            def transform(self, X):
                if hasattr(X, 'ravel'):
                    X = X.ravel()
                return np.array([self.mapping.get(x, self.global_mean) for x in X])
        
        encoder = TargetEncoder()
        encoder.fit(series, target)
        return encoder
    
    def _get_scaling_method(self, column: str, series: pd.Series) -> str:
        """Determine appropriate scaling method for a column"""
        if column in self.scaling_methods:
            return self.scaling_methods[column]
        
        # Auto-detect based on data characteristics
        if series.dtype not in ['int64', 'float64']:
            return 'none'
        
        # Remove missing values for analysis
        clean_data = series.dropna()
        if len(clean_data) < 10:
            return 'standard'  # Default for small samples
        
        # Check for skewness
        skewness = abs(clean_data.skew())
        if skewness > 2:
            return 'power'  # Highly skewed
        
        # Check for outliers using IQR
        q1 = clean_data.quantile(0.25)
        q3 = clean_data.quantile(0.75)
        iqr = q3 - q1
        outlier_threshold = 1.5 * iqr
        outliers = ((clean_data < q1 - outlier_threshold) | 
                   (clean_data > q3 + outlier_threshold)).sum()
        outlier_ratio = outliers / len(clean_data)
        
        if outlier_ratio > 0.1:  # More than 10% outliers
            return 'robust'
        
        # Check if data is naturally bounded (e.g., percentages, probabilities)
        data_min, data_max = clean_data.min(), clean_data.max()
        if (data_min >= 0 and data_max <= 1) or (data_min >= 0 and data_max <= 100):
            return 'minmax'
        
        # Default to standard scaling
        return 'standard'
    
    def get_normalization_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of normalization applied to the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalization statistics
        """
        summary_data = []
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                method = self._get_scaling_method(column, df[column])
                scaler = self.scalers.get(column)
                
                summary_data.append({
                    'Column': column,
                    'Data_Type': 'Numeric',
                    'Method': method,
                    'Original_Min': df[column].min(),
                    'Original_Max': df[column].max(),
                    'Original_Mean': df[column].mean(),
                    'Original_Std': df[column].std(),
                    'Skewness': df[column].skew(),
                    'Fitted': scaler is not None
                })
            
            elif df[column].dtype == 'object':
                encoder = self.encoders.get(column)
                unique_values = df[column].nunique()
                
                summary_data.append({
                    'Column': column,
                    'Data_Type': 'Categorical',
                    'Method': self.categorical_encoding,
                    'Unique_Values': unique_values,
                    'Most_Frequent': df[column].mode().iloc[0] if len(df[column].mode()) > 0 else None,
                    'Fitted': encoder is not None
                })
        
        return pd.DataFrame(summary_data)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after transformation
        
        Returns:
            List of feature names after encoding
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted to get feature names")
        
        feature_names = []
        
        # Add scaled numeric columns
        for column in self.scalers.keys():
            feature_names.append(column)
        
        # Add encoded categorical columns
        for column, encoder in self.encoders.items():
            if isinstance(encoder, OneHotEncoder):
                feature_names.extend(encoder.get_feature_names_out([column]))
            else:
                feature_names.append(column)
        
        return feature_names
    
    def detect_data_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect potential data quality issues that affect normalization
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict with detected issues
        """
        issues = {
            'high_skewness': [],
            'many_outliers': [],
            'constant_columns': [],
            'high_cardinality': [],
            'imbalanced_categories': []
        }
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # Check for constant columns
                if df[column].nunique() <= 1:
                    issues['constant_columns'].append(column)
                    continue
                
                # Check for high skewness
                skewness = abs(df[column].skew())
                if skewness > 3:
                    issues['high_skewness'].append({
                        'column': column,
                        'skewness': skewness
                    })
                
                # Check for many outliers
                clean_data = df[column].dropna()
                if len(clean_data) > 10:
                    q1 = clean_data.quantile(0.25)
                    q3 = clean_data.quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((clean_data < q1 - 1.5 * iqr) | 
                              (clean_data > q3 + 1.5 * iqr)).sum()
                    outlier_ratio = outliers / len(clean_data)
                    
                    if outlier_ratio > 0.2:  # More than 20% outliers
                        issues['many_outliers'].append({
                            'column': column,
                            'outlier_ratio': outlier_ratio
                        })
            
            elif df[column].dtype == 'object':
                unique_count = df[column].nunique()
                total_count = len(df[column].dropna())
                
                # Check for high cardinality
                if unique_count > total_count * 0.8:  # More than 80% unique
                    issues['high_cardinality'].append({
                        'column': column,
                        'unique_count': unique_count,
                        'cardinality_ratio': unique_count / total_count
                    })
                
                # Check for imbalanced categories
                if unique_count > 1:
                    value_counts = df[column].value_counts()
                    most_frequent_ratio = value_counts.iloc[0] / total_count
                    if most_frequent_ratio > 0.9:  # Most frequent > 90%
                        issues['imbalanced_categories'].append({
                            'column': column,
                            'most_frequent_ratio': most_frequent_ratio,
                            'most_frequent_value': value_counts.index[0]
                        })
        
        return issues