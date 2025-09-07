"""
Missing Value Imputation Strategies
Implements multiple imputation methods for handling missing data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class MissingValueImputer:
    """
    Comprehensive missing value imputation with multiple strategies
    """
    
    def __init__(self, strategies: Dict[str, str] = None):
        """
        Initialize imputer with strategies for different columns
        
        Args:
            strategies: Dict mapping column names to imputation strategies
                       Options: 'mean', 'median', 'mode', 'knn', 'constant', 'drop'
        """
        self.strategies = strategies or {}
        self.imputers = {}
        self.encoders = {}
        self.fitted = False
        
        # Default strategies by data type
        self.default_strategies = {
            'numeric': 'median',
            'categorical': 'mode',
            'boolean': 'mode'
        }
    
    def fit(self, df: pd.DataFrame) -> 'MissingValueImputer':
        """
        Fit imputers on the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        self.imputers = {}
        self.encoders = {}
        
        for column in df.columns:
            if df[column].isnull().sum() == 0:
                continue
                
            strategy = self._get_strategy(column, df[column])
            
            if strategy == 'drop':
                continue
            elif strategy == 'knn':
                # Handle categorical columns for KNN
                if df[column].dtype == 'object':
                    encoder = LabelEncoder()
                    # Fit on non-null values
                    non_null_mask = df[column].notnull()
                    if non_null_mask.sum() > 0:
                        encoder.fit(df.loc[non_null_mask, column])
                        self.encoders[column] = encoder
            else:
                # Simple imputation
                if strategy in ['mean', 'median']:
                    imputer = SimpleImputer(strategy=strategy)
                elif strategy == 'mode':
                    imputer = SimpleImputer(strategy='most_frequent')
                elif strategy == 'constant':
                    # Use appropriate constant based on data type
                    if df[column].dtype in ['int64', 'float64']:
                        fill_value = 0
                    else:
                        fill_value = 'Unknown'
                    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                
                # Fit imputer
                try:
                    imputer.fit(df[[column]])
                    self.imputers[column] = imputer
                except Exception as e:
                    logger.warning(f"Failed to fit imputer for {column}: {e}")
                    continue
        
        # Fit KNN imputer if needed
        knn_columns = [col for col in df.columns 
                      if self._get_strategy(col, df[col]) == 'knn']
        
        if knn_columns:
            # Prepare data for KNN imputation
            knn_data = df[knn_columns].copy()
            
            # Encode categorical columns
            for col in knn_columns:
                if df[col].dtype == 'object' and col in self.encoders:
                    encoder = self.encoders[col]
                    non_null_mask = knn_data[col].notnull()
                    if non_null_mask.sum() > 0:
                        knn_data.loc[non_null_mask, col] = encoder.transform(
                            knn_data.loc[non_null_mask, col]
                        )
            
            # Fit KNN imputer
            self.knn_imputer = KNNImputer(n_neighbors=5)
            self.knn_imputer.fit(knn_data)
            self.knn_columns = knn_columns
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation to the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values imputed
        """
        if not self.fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        result_df = df.copy()
        imputation_log = {}
        
        # Apply simple imputation strategies
        for column, imputer in self.imputers.items():
            if column in result_df.columns:
                missing_count = result_df[column].isnull().sum()
                if missing_count > 0:
                    try:
                        result_df[column] = imputer.transform(result_df[[column]]).ravel()
                        imputation_log[column] = {
                            'strategy': 'simple',
                            'missing_count': missing_count
                        }
                    except Exception as e:
                        logger.warning(f"Failed to impute {column}: {e}")
        
        # Apply KNN imputation
        if hasattr(self, 'knn_imputer') and hasattr(self, 'knn_columns'):
            knn_missing = result_df[self.knn_columns].isnull().sum().sum()
            if knn_missing > 0:
                try:
                    # Prepare data for KNN
                    knn_data = result_df[self.knn_columns].copy()
                    
                    # Encode categorical columns
                    for col in self.knn_columns:
                        if col in self.encoders:
                            encoder = self.encoders[col]
                            non_null_mask = knn_data[col].notnull()
                            if non_null_mask.sum() > 0:
                                knn_data.loc[non_null_mask, col] = encoder.transform(
                                    knn_data.loc[non_null_mask, col]
                                )
                    
                    # Apply KNN imputation
                    imputed_data = self.knn_imputer.transform(knn_data)
                    
                    # Decode categorical columns back
                    for i, col in enumerate(self.knn_columns):
                        if col in self.encoders:
                            encoder = self.encoders[col]
                            try:
                                # Round to nearest integer for categorical data
                                rounded_data = np.round(imputed_data[:, i]).astype(int)
                                # Ensure values are within encoder range
                                rounded_data = np.clip(rounded_data, 0, len(encoder.classes_) - 1)
                                result_df[col] = encoder.inverse_transform(rounded_data)
                            except Exception as e:
                                logger.warning(f"Failed to decode KNN imputed {col}: {e}")
                        else:
                            result_df[col] = imputed_data[:, i]
                        
                        imputation_log[col] = {
                            'strategy': 'knn',
                            'missing_count': result_df[col].isnull().sum()
                        }
                
                except Exception as e:
                    logger.warning(f"KNN imputation failed: {e}")
        
        # Drop columns marked for dropping
        drop_columns = [col for col in result_df.columns 
                       if self._get_strategy(col, df[col]) == 'drop']
        if drop_columns:
            result_df = result_df.drop(columns=drop_columns)
            for col in drop_columns:
                imputation_log[col] = {'strategy': 'drop'}
        
        # Log imputation summary
        if imputation_log:
            logger.info(f"Imputation summary: {imputation_log}")
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values imputed
        """
        return self.fit(df).transform(df)
    
    def _get_strategy(self, column: str, series: pd.Series) -> str:
        """
        Get imputation strategy for a column
        
        Args:
            column: Column name
            series: Pandas series for the column
            
        Returns:
            Imputation strategy
        """
        # Use explicitly defined strategy
        if column in self.strategies:
            return self.strategies[column]
        
        # Determine default strategy based on data type
        if series.dtype in ['int64', 'float64']:
            return self.default_strategies['numeric']
        elif series.dtype == 'bool':
            return self.default_strategies['boolean']
        else:
            return self.default_strategies['categorical']
    
    def get_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing value statistics
        """
        missing_data = []
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            data_type = str(df[column].dtype)
            strategy = self._get_strategy(column, df[column])
            
            missing_data.append({
                'Column': column,
                'Missing_Count': missing_count,
                'Missing_Percent': missing_percent,
                'Data_Type': data_type,
                'Imputation_Strategy': strategy
            })
        
        summary_df = pd.DataFrame(missing_data)
        summary_df = summary_df.sort_values('Missing_Percent', ascending=False)
        
        return summary_df