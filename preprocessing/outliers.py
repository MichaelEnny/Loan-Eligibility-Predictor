"""
Outlier Detection and Handling
Implements multiple outlier detection methods: IQR, Z-score, Isolation Forest
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class OutlierDetector:
    """
    Comprehensive outlier detection with multiple methods
    """
    
    def __init__(self, 
                 methods: Dict[str, str] = None,
                 thresholds: Dict[str, Dict[str, Any]] = None,
                 handle_strategy: str = 'clip'):
        """
        Initialize outlier detector
        
        Args:
            methods: Dict mapping column names to detection methods
                    Options: 'iqr', 'zscore', 'isolation_forest', 'none'
            thresholds: Dict with method-specific thresholds
            handle_strategy: How to handle outliers ('clip', 'remove', 'flag')
        """
        self.methods = methods or {}
        self.thresholds = thresholds or {
            'iqr': {'multiplier': 1.5},
            'zscore': {'threshold': 3},
            'isolation_forest': {'contamination': 0.1}
        }
        self.handle_strategy = handle_strategy
        self.fitted_params = {}
        self.fitted = False
        
        # Default methods by data type
        self.default_methods = {
            'numeric': 'iqr',
            'categorical': 'none'
        }
    
    def fit(self, df: pd.DataFrame) -> 'OutlierDetector':
        """
        Fit outlier detection parameters
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        self.fitted_params = {}
        
        for column in df.columns:
            if df[column].dtype not in ['int64', 'float64']:
                continue
                
            method = self._get_method(column, df[column])
            
            if method == 'iqr':
                self._fit_iqr(column, df[column])
            elif method == 'zscore':
                self._fit_zscore(column, df[column])
            elif method == 'isolation_forest':
                self._fit_isolation_forest(column, df[column])
        
        self.fitted = True
        return self
    
    def detect(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Detect outliers in the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict mapping column names to boolean arrays (True = outlier)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before detection")
        
        outliers = {}
        
        for column in df.columns:
            if column not in self.fitted_params:
                continue
                
            method = self._get_method(column, df[column])
            
            if method == 'iqr':
                outliers[column] = self._detect_iqr(column, df[column])
            elif method == 'zscore':
                outliers[column] = self._detect_zscore(column, df[column])
            elif method == 'isolation_forest':
                outliers[column] = self._detect_isolation_forest(column, df[column])
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, outliers: Dict[str, np.ndarray] = None) -> pd.DataFrame:
        """
        Handle detected outliers
        
        Args:
            df: Input DataFrame
            outliers: Dict of outlier masks (if None, will detect first)
            
        Returns:
            DataFrame with outliers handled
        """
        if outliers is None:
            outliers = self.detect(df)
        
        result_df = df.copy()
        outlier_log = {}
        
        for column, outlier_mask in outliers.items():
            if column not in result_df.columns:
                continue
                
            outlier_count = outlier_mask.sum()
            if outlier_count == 0:
                continue
            
            if self.handle_strategy == 'clip':
                result_df = self._clip_outliers(result_df, column, outlier_mask)
            elif self.handle_strategy == 'remove':
                # Remove rows with outliers
                result_df = result_df[~outlier_mask]
            elif self.handle_strategy == 'flag':
                # Add outlier flag column
                result_df[f'{column}_outlier'] = outlier_mask
            
            outlier_log[column] = {
                'method': self._get_method(column, df[column]),
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100,
                'strategy': self.handle_strategy
            }
        
        if outlier_log:
            logger.info(f"Outlier handling summary: {outlier_log}")
        
        return result_df
    
    def fit_detect_handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit, detect, and handle outliers in one step
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        self.fit(df)
        outliers = self.detect(df)
        return self.handle_outliers(df, outliers)
    
    def _get_method(self, column: str, series: pd.Series) -> str:
        """
        Get outlier detection method for a column
        """
        if column in self.methods:
            return self.methods[column]
        
        # Default method based on data type
        if series.dtype in ['int64', 'float64']:
            return self.default_methods['numeric']
        else:
            return self.default_methods['categorical']
    
    def _fit_iqr(self, column: str, series: pd.Series):
        """Fit IQR method parameters"""
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        multiplier = self.thresholds['iqr']['multiplier']
        
        self.fitted_params[column] = {
            'method': 'iqr',
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'lower_bound': q25 - multiplier * iqr,
            'upper_bound': q75 + multiplier * iqr,
            'multiplier': multiplier
        }
    
    def _detect_iqr(self, column: str, series: pd.Series) -> np.ndarray:
        """Detect outliers using IQR method"""
        params = self.fitted_params[column]
        lower_bound = params['lower_bound']
        upper_bound = params['upper_bound']
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _fit_zscore(self, column: str, series: pd.Series):
        """Fit Z-score method parameters"""
        mean = series.mean()
        std = series.std()
        threshold = self.thresholds['zscore']['threshold']
        
        self.fitted_params[column] = {
            'method': 'zscore',
            'mean': mean,
            'std': std,
            'threshold': threshold
        }
    
    def _detect_zscore(self, column: str, series: pd.Series) -> np.ndarray:
        """Detect outliers using Z-score method"""
        params = self.fitted_params[column]
        mean = params['mean']
        std = params['std']
        threshold = params['threshold']
        
        if std == 0:
            return np.zeros(len(series), dtype=bool)
        
        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold
    
    def _fit_isolation_forest(self, column: str, series: pd.Series):
        """Fit Isolation Forest parameters"""
        contamination = self.thresholds['isolation_forest']['contamination']
        
        # Handle missing values for isolation forest
        clean_data = series.dropna().values.reshape(-1, 1)
        
        if len(clean_data) < 10:  # Need minimum samples
            self.fitted_params[column] = {
                'method': 'isolation_forest',
                'model': None,
                'contamination': contamination
            }
            return
        
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(clean_data)
        
        self.fitted_params[column] = {
            'method': 'isolation_forest',
            'model': model,
            'contamination': contamination
        }
    
    def _detect_isolation_forest(self, column: str, series: pd.Series) -> np.ndarray:
        """Detect outliers using Isolation Forest"""
        params = self.fitted_params[column]
        model = params['model']
        
        if model is None:
            return np.zeros(len(series), dtype=bool)
        
        # Handle missing values
        outlier_mask = np.zeros(len(series), dtype=bool)
        valid_mask = series.notna()
        
        if valid_mask.sum() == 0:
            return outlier_mask
        
        valid_data = series[valid_mask].values.reshape(-1, 1)
        predictions = model.predict(valid_data)
        
        # Isolation Forest returns -1 for outliers, 1 for inliers
        outlier_mask[valid_mask] = predictions == -1
        
        return outlier_mask
    
    def _clip_outliers(self, df: pd.DataFrame, column: str, outlier_mask: np.ndarray) -> pd.DataFrame:
        """Clip outliers to bounds"""
        if column not in self.fitted_params:
            return df
        
        params = self.fitted_params[column]
        method = params['method']
        
        if method == 'iqr':
            lower_bound = params['lower_bound']
            upper_bound = params['upper_bound']
            df.loc[:, column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'zscore':
            mean = params['mean']
            std = params['std']
            threshold = params['threshold']
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            df.loc[:, column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'isolation_forest':
            # For isolation forest, use percentile-based clipping
            q05 = df[column].quantile(0.05)
            q95 = df[column].quantile(0.95)
            df.loc[:, column] = df[column].clip(lower=q05, upper=q95)
        
        return df
    
    def get_outlier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of outliers in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outlier statistics
        """
        if not self.fitted:
            self.fit(df)
        
        outliers = self.detect(df)
        summary_data = []
        
        for column in df.columns:
            if df[column].dtype not in ['int64', 'float64']:
                continue
            
            method = self._get_method(column, df[column])
            outlier_mask = outliers.get(column, np.zeros(len(df), dtype=bool))
            outlier_count = outlier_mask.sum()
            outlier_percent = (outlier_count / len(df)) * 100
            
            summary_data.append({
                'Column': column,
                'Method': method,
                'Outlier_Count': outlier_count,
                'Outlier_Percent': outlier_percent,
                'Min_Value': df[column].min(),
                'Max_Value': df[column].max(),
                'Mean': df[column].mean(),
                'Std': df[column].std()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Outlier_Percent', ascending=False)
        
        return summary_df
    
    def visualize_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """
        Get outlier visualization data
        
        Args:
            df: Input DataFrame
            columns: List of columns to analyze (if None, use all numeric)
            
        Returns:
            Dict with visualization data for plotting
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not self.fitted:
            self.fit(df)
        
        outliers = self.detect(df)
        viz_data = {}
        
        for column in columns:
            if column not in df.columns:
                continue
            
            outlier_mask = outliers.get(column, np.zeros(len(df), dtype=bool))
            
            viz_data[column] = {
                'values': df[column].values,
                'outliers': outlier_mask,
                'method': self._get_method(column, df[column]),
                'bounds': self._get_bounds(column) if column in self.fitted_params else None
            }
        
        return viz_data
    
    def _get_bounds(self, column: str) -> Dict[str, float]:
        """Get outlier bounds for a column"""
        if column not in self.fitted_params:
            return None
        
        params = self.fitted_params[column]
        method = params['method']
        
        if method == 'iqr':
            return {
                'lower': params['lower_bound'],
                'upper': params['upper_bound']
            }
        elif method == 'zscore':
            mean = params['mean']
            std = params['std']
            threshold = params['threshold']
            return {
                'lower': mean - threshold * std,
                'upper': mean + threshold * std
            }
        
        return None