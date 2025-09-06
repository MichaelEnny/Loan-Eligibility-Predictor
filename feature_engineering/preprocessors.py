"""
Numerical Preprocessing Components for Feature Engineering Pipeline
Handles scaling, normalization, outlier detection/treatment, and imputation.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, KBinsDiscretizer
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
import logging

logger = logging.getLogger(__name__)


class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    Comprehensive outlier detection and treatment component.
    
    Supports multiple outlier detection methods including IQR, Isolation Forest,
    and Local Outlier Factor with configurable treatment strategies.
    """
    
    def __init__(self,
                 method: str = 'iqr',
                 threshold: float = 1.5,
                 action: str = 'clip',
                 contamination: float = 0.1,
                 n_neighbors: int = 20,
                 random_state: int = 42):
        """
        Initialize outlier detector.
        
        Args:
            method: Detection method ('iqr', 'isolation_forest', 'local_outlier_factor')
            threshold: Threshold for IQR method (multiplier for IQR)
            action: Treatment action ('clip', 'remove', 'transform')
            contamination: Expected proportion of outliers
            n_neighbors: Number of neighbors for LOF
            random_state: Random state for reproducibility
        """
        self.method = method
        self.threshold = threshold
        self.action = action
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        
        # Fitted parameters
        self.bounds_ = {}
        self.detector_ = None
        self.outlier_mask_ = None
        
    def _iqr_bounds(self, X: pd.Series) -> Tuple[float, float]:
        """Calculate IQR-based bounds for outlier detection."""
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        return lower_bound, upper_bound
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'OutlierDetector':
        """
        Fit outlier detection model.
        
        Args:
            X: Input numerical features
            y: Target variable (not used)
            
        Returns:
            Self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if self.method == 'iqr':
            # Calculate IQR bounds for each feature
            for column in X.columns:
                if X[column].dtype in ['int64', 'float64']:
                    self.bounds_[column] = self._iqr_bounds(X[column])
                    
        elif self.method == 'isolation_forest':
            # Fit Isolation Forest
            self.detector_ = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )
            # Only fit on numerical columns
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                self.detector_.fit(X[numerical_cols])
            else:
                warnings.warn("No numerical columns found for Isolation Forest")
                
        elif self.method == 'local_outlier_factor':
            # Fit Local Outlier Factor
            self.detector_ = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination,
                n_jobs=-1
            )
            # LOF requires fit_predict, so we store the fitted data for transform
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                self.outlier_mask_ = self.detector_.fit_predict(X[numerical_cols]) == -1
            else:
                warnings.warn("No numerical columns found for Local Outlier Factor")
                
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by treating outliers.
        
        Args:
            X: Input numerical features
            
        Returns:
            Transformed features with outliers treated
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        
        if self.method == 'iqr':
            for column in X.columns:
                if column in self.bounds_:
                    lower_bound, upper_bound = self.bounds_[column]
                    
                    if self.action == 'clip':
                        X_transformed[column] = X_transformed[column].clip(lower_bound, upper_bound)
                    elif self.action == 'remove':
                        # Mark outliers for removal (return mask separately if needed)
                        outlier_mask = ((X[column] < lower_bound) | 
                                      (X[column] > upper_bound))
                        X_transformed.loc[outlier_mask, column] = np.nan
                    elif self.action == 'transform':
                        # Apply log transformation to reduce outlier impact
                        X_transformed[column] = np.sign(X[column]) * np.log1p(np.abs(X[column]))
                        
        elif self.method == 'isolation_forest' and self.detector_ is not None:
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                outlier_scores = self.detector_.predict(X[numerical_cols])
                outlier_mask = outlier_scores == -1
                
                if self.action == 'clip':
                    # Clip based on quantiles
                    for column in numerical_cols:
                        q05, q95 = X[column].quantile([0.05, 0.95])
                        X_transformed[column] = X_transformed[column].clip(q05, q95)
                elif self.action == 'remove':
                    X_transformed.loc[outlier_mask] = np.nan
                elif self.action == 'transform':
                    for column in numerical_cols:
                        X_transformed.loc[outlier_mask, column] = np.log1p(
                            np.abs(X_transformed.loc[outlier_mask, column])
                        ) * np.sign(X_transformed.loc[outlier_mask, column])
                        
        elif self.method == 'local_outlier_factor' and self.outlier_mask_ is not None:
            if self.action == 'clip':
                # Clip based on quantiles
                numerical_cols = X.select_dtypes(include=[np.number]).columns
                for column in numerical_cols:
                    q05, q95 = X[column].quantile([0.05, 0.95])
                    X_transformed[column] = X_transformed[column].clip(q05, q95)
            elif self.action == 'remove':
                X_transformed.loc[self.outlier_mask_] = np.nan
            elif self.action == 'transform':
                numerical_cols = X.select_dtypes(include=[np.number]).columns
                for column in numerical_cols:
                    X_transformed.loc[self.outlier_mask_, column] = np.log1p(
                        np.abs(X_transformed.loc[self.outlier_mask_, column])
                    ) * np.sign(X_transformed.loc[self.outlier_mask_, column])
                    
        return X_transformed


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive numerical preprocessing component.
    
    Handles scaling, normalization, outlier treatment, binning, and imputation
    with intelligent feature selection and preprocessing strategy assignment.
    """
    
    def __init__(self,
                 scaling_features: List[str] = None,
                 scaling_method: str = 'standard',
                 normalization_features: List[str] = None,
                 normalization_method: str = 'yeo-johnson',
                 outlier_features: List[str] = None,
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 1.5,
                 outlier_action: str = 'clip',
                 binning_features: Dict[str, Dict[str, Any]] = None,
                 imputation_strategy: str = 'median',
                 imputation_constant: float = 0.0,
                 quantile_range: Tuple[float, float] = (25.0, 75.0),
                 random_state: int = 42,
                 verbose: bool = False):
        """
        Initialize numerical preprocessor.
        
        Args:
            scaling_features: Features for scaling
            scaling_method: Scaling method ('standard', 'minmax', 'robust', 'quantile')
            normalization_features: Features for normalization
            normalization_method: Normalization method ('yeo-johnson', 'box-cox', 'quantile')
            outlier_features: Features for outlier treatment
            outlier_method: Outlier detection method
            outlier_threshold: Outlier threshold
            outlier_action: Outlier treatment action
            binning_features: Features and configurations for binning
            imputation_strategy: Strategy for missing value imputation
            imputation_constant: Constant value for imputation
            quantile_range: Range for robust scaling
            random_state: Random state for reproducibility
            verbose: Whether to print processing information
        """
        self.scaling_features = scaling_features or []
        self.scaling_method = scaling_method
        self.normalization_features = normalization_features or []
        self.normalization_method = normalization_method
        self.outlier_features = outlier_features or []
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.outlier_action = outlier_action
        self.binning_features = binning_features or {}
        self.imputation_strategy = imputation_strategy
        self.imputation_constant = imputation_constant
        self.quantile_range = quantile_range
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted components
        self.scaler_ = None
        self.normalizer_ = None
        self.outlier_detector_ = None
        self.binning_transformers_ = {}
        self.imputer_ = None
        self.feature_names_out_ = []
        self.feature_stats_ = {}
        
    def _get_feature_stats(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive feature statistics."""
        stats = {}
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for column in numerical_cols:
            col_stats = {
                'mean': X[column].mean(),
                'std': X[column].std(),
                'min': X[column].min(),
                'max': X[column].max(),
                'q25': X[column].quantile(0.25),
                'q50': X[column].quantile(0.50),
                'q75': X[column].quantile(0.75),
                'skewness': X[column].skew(),
                'kurtosis': X[column].kurtosis(),
                'missing_pct': (X[column].isnull().sum() / len(X)) * 100,
                'unique_values': X[column].nunique(),
                'outliers_iqr': self._count_outliers_iqr(X[column])
            }
            stats[column] = col_stats
            
        return stats
    
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def _auto_assign_preprocessing(self, X: pd.DataFrame) -> None:
        """Automatically assign preprocessing strategies based on data characteristics."""
        if not hasattr(self, 'feature_stats_') or not self.feature_stats_:
            self.feature_stats_ = self._get_feature_stats(X)
        
        for column, stats in self.feature_stats_.items():
            # Auto-assign scaling if not specified
            if not self.scaling_features and column in X.columns:
                if stats['std'] > 0:  # Has variance
                    self.scaling_features.append(column)
                    
            # Auto-assign outlier treatment if significant outliers
            if not self.outlier_features and stats['outliers_iqr'] > len(X) * 0.05:
                self.outlier_features.append(column)
                
            # Auto-assign normalization for highly skewed features
            if (not self.normalization_features and 
                abs(stats['skewness']) > 2.0 and 
                column not in self.outlier_features):
                self.normalization_features.append(column)
                
        if self.verbose:
            logger.info(f"Auto-assigned preprocessing:")
            logger.info(f"  Scaling features: {len(self.scaling_features)}")
            logger.info(f"  Outlier features: {len(self.outlier_features)}")
            logger.info(f"  Normalization features: {len(self.normalization_features)}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'NumericalPreprocessor':
        """
        Fit numerical preprocessing components.
        
        Args:
            X: Input numerical features
            y: Target variable (not used)
            
        Returns:
            Self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Calculate feature statistics
        self.feature_stats_ = self._get_feature_stats(X)
        
        # Auto-assign preprocessing strategies
        self._auto_assign_preprocessing(X)
        
        # Fit imputer first (if needed)
        if X.isnull().any().any():
            self.imputer_ = SimpleImputer(
                strategy=self.imputation_strategy,
                fill_value=self.imputation_constant if self.imputation_strategy == 'constant' else None
            )
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                self.imputer_.fit(X[numerical_cols])
        
        # Apply imputation for fitting other components
        X_imputed = X.copy()
        if self.imputer_ is not None:
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                X_imputed[numerical_cols] = self.imputer_.transform(X[numerical_cols])
        
        # Fit outlier detector
        if self.outlier_features:
            outlier_cols = [col for col in self.outlier_features if col in X_imputed.columns]
            if outlier_cols:
                self.outlier_detector_ = OutlierDetector(
                    method=self.outlier_method,
                    threshold=self.outlier_threshold,
                    action=self.outlier_action,
                    random_state=self.random_state
                )
                self.outlier_detector_.fit(X_imputed[outlier_cols])
        
        # Apply outlier treatment for fitting other components
        X_processed = X_imputed.copy()
        if self.outlier_detector_ is not None:
            outlier_cols = [col for col in self.outlier_features if col in X_processed.columns]
            if outlier_cols:
                X_processed[outlier_cols] = self.outlier_detector_.transform(X_processed[outlier_cols])
        
        # Fit normalizer
        if self.normalization_features:
            norm_cols = [col for col in self.normalization_features if col in X_processed.columns]
            if norm_cols:
                if self.normalization_method == 'yeo-johnson':
                    self.normalizer_ = PowerTransformer(method='yeo-johnson', standardize=False)
                elif self.normalization_method == 'box-cox':
                    self.normalizer_ = PowerTransformer(method='box-cox', standardize=False)
                elif self.normalization_method == 'quantile':
                    self.normalizer_ = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
                
                # Ensure positive values for box-cox
                if self.normalization_method == 'box-cox':
                    for col in norm_cols:
                        if (X_processed[col] <= 0).any():
                            X_processed[col] = X_processed[col] - X_processed[col].min() + 1
                
                self.normalizer_.fit(X_processed[norm_cols])
        
        # Apply normalization for fitting scaler
        if self.normalizer_ is not None:
            norm_cols = [col for col in self.normalization_features if col in X_processed.columns]
            if norm_cols:
                X_processed[norm_cols] = self.normalizer_.transform(X_processed[norm_cols])
        
        # Fit scaler
        if self.scaling_features:
            scaling_cols = [col for col in self.scaling_features if col in X_processed.columns]
            if scaling_cols:
                if self.scaling_method == 'standard':
                    self.scaler_ = StandardScaler()
                elif self.scaling_method == 'minmax':
                    self.scaler_ = MinMaxScaler()
                elif self.scaling_method == 'robust':
                    self.scaler_ = RobustScaler(quantile_range=self.quantile_range)
                elif self.scaling_method == 'quantile':
                    self.scaler_ = QuantileTransformer(output_distribution='uniform', random_state=self.random_state)
                    
                self.scaler_.fit(X_processed[scaling_cols])
        
        # Fit binning transformers
        for feature, config in self.binning_features.items():
            if feature in X_processed.columns:
                binning_transformer = KBinsDiscretizer(
                    n_bins=config.get('n_bins', 5),
                    encode=config.get('encode', 'ordinal'),
                    strategy=config.get('strategy', 'quantile'),
                    random_state=self.random_state
                )
                binning_transformer.fit(X_processed[[feature]])
                self.binning_transformers_[feature] = binning_transformer
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical features using fitted preprocessors.
        
        Args:
            X: Input numerical features
            
        Returns:
            Preprocessed features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        
        # Apply imputation
        if self.imputer_ is not None:
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                X_transformed[numerical_cols] = self.imputer_.transform(X[numerical_cols])
        
        # Apply outlier treatment
        if self.outlier_detector_ is not None:
            outlier_cols = [col for col in self.outlier_features if col in X_transformed.columns]
            if outlier_cols:
                X_transformed[outlier_cols] = self.outlier_detector_.transform(X_transformed[outlier_cols])
        
        # Apply normalization
        if self.normalizer_ is not None:
            norm_cols = [col for col in self.normalization_features if col in X_transformed.columns]
            if norm_cols:
                # Handle box-cox requirement for positive values
                if self.normalization_method == 'box-cox':
                    for col in norm_cols:
                        if (X_transformed[col] <= 0).any():
                            X_transformed[col] = X_transformed[col] - X_transformed[col].min() + 1
                
                X_transformed[norm_cols] = self.normalizer_.transform(X_transformed[norm_cols])
        
        # Apply scaling
        if self.scaler_ is not None:
            scaling_cols = [col for col in self.scaling_features if col in X_transformed.columns]
            if scaling_cols:
                X_transformed[scaling_cols] = self.scaler_.transform(X_transformed[scaling_cols])
        
        # Apply binning
        binning_results = []
        for feature, transformer in self.binning_transformers_.items():
            if feature in X_transformed.columns:
                binned_data = transformer.transform(X_transformed[[feature]])
                binned_df = pd.DataFrame(
                    binned_data, 
                    columns=[f"{feature}_binned"],
                    index=X_transformed.index
                )
                binning_results.append(binned_df)
        
        if binning_results:
            binning_df = pd.concat(binning_results, axis=1)
            X_transformed = pd.concat([X_transformed, binning_df], axis=1)
        
        self.feature_names_out_ = list(X_transformed.columns)
        return X_transformed
    
    def get_feature_names_out(self, input_features: List[str] = None) -> List[str]:
        """Get output feature names."""
        return self.feature_names_out_
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about applied preprocessing."""
        info = {
            'scaling_features': self.scaling_features,
            'scaling_method': self.scaling_method,
            'normalization_features': self.normalization_features,
            'normalization_method': self.normalization_method,
            'outlier_features': self.outlier_features,
            'outlier_method': self.outlier_method,
            'binning_features': list(self.binning_features.keys()),
            'feature_stats': self.feature_stats_,
            'output_features': len(self.feature_names_out_)
        }
        return info