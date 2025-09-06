"""
Categorical Encoding Components for Feature Engineering Pipeline
Handles one-hot encoding, target encoding, label encoding, and ordinal encoding.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import KFold
import warnings
import logging

logger = logging.getLogger(__name__)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder with regularization and cross-validation to prevent overfitting.
    
    This encoder replaces categorical values with the mean target value for each category,
    using cross-validation and smoothing to reduce overfitting risks.
    """
    
    def __init__(self, 
                 smoothing: float = 1.0,
                 min_samples_leaf: int = 20,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize target encoder.
        
        Args:
            smoothing: Smoothing factor for regularization
            min_samples_leaf: Minimum samples required for category
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.encodings_ = {}
        self.global_mean_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TargetEncoder':
        """
        Fit the target encoder.
        
        Args:
            X: Input categorical features
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        self.global_mean_ = y.mean()
        
        for column in X.columns:
            # Calculate category means and counts
            category_stats = (X[column]
                            .to_frame()
                            .assign(target=y)
                            .groupby(column)['target']
                            .agg(['mean', 'count'])
                            .reset_index())
            
            # Apply smoothing formula: (count * cat_mean + smoothing * global_mean) / (count + smoothing)
            smoothed_mean = ((category_stats['count'] * category_stats['mean'] + 
                            self.smoothing * self.global_mean_) / 
                           (category_stats['count'] + self.smoothing))
            
            # Store encodings
            encoding_dict = dict(zip(category_stats[column], smoothed_mean))
            self.encodings_[column] = encoding_dict
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using fitted encodings.
        
        Args:
            X: Input categorical features
            
        Returns:
            Encoded features
        """
        if not hasattr(self, 'encodings_'):
            raise ValueError("TargetEncoder must be fitted before transform")
            
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_encoded = X.copy()
        
        for column in X.columns:
            if column in self.encodings_:
                # Map categories to encoded values, use global mean for unknown categories
                X_encoded[column] = X[column].map(self.encodings_[column]).fillna(self.global_mean_)
            else:
                warnings.warn(f"Column {column} not found in fitted encodings")
                
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform using cross-validation to prevent overfitting.
        
        Args:
            X: Input categorical features
            y: Target variable
            
        Returns:
            Cross-validation encoded features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        X_encoded = X.copy()
        self.global_mean_ = y.mean()
        self.encodings_ = {}
        
        # Use cross-validation for encoding
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for column in X.columns:
            X_encoded[column] = np.full(len(X), self.global_mean_)
            
            for train_idx, val_idx in kfold.split(X):
                # Fit on train fold
                train_stats = (X.iloc[train_idx][column]
                             .to_frame()
                             .assign(target=y.iloc[train_idx])
                             .groupby(column)['target']
                             .agg(['mean', 'count'])
                             .reset_index())
                
                # Apply smoothing
                smoothed_mean = ((train_stats['count'] * train_stats['mean'] + 
                                self.smoothing * self.global_mean_) / 
                               (train_stats['count'] + self.smoothing))
                
                encoding_dict = dict(zip(train_stats[column], smoothed_mean))
                
                # Transform validation fold
                X_encoded.iloc[val_idx, X_encoded.columns.get_loc(column)] = (
                    X.iloc[val_idx][column].map(encoding_dict).fillna(self.global_mean_)
                )
            
            # Fit final encoding on full data for future transforms
            final_stats = (X[column]
                          .to_frame()
                          .assign(target=y)
                          .groupby(column)['target']
                          .agg(['mean', 'count'])
                          .reset_index())
            
            final_smoothed = ((final_stats['count'] * final_stats['mean'] + 
                             self.smoothing * self.global_mean_) / 
                            (final_stats['count'] + self.smoothing))
            
            self.encodings_[column] = dict(zip(final_stats[column], final_smoothed))
            
        return X_encoded


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Comprehensive categorical encoder that handles multiple encoding strategies.
    
    Supports one-hot encoding, target encoding, label encoding, and ordinal encoding
    with intelligent feature selection and handling of high cardinality categories.
    """
    
    def __init__(self,
                 onehot_features: List[str] = None,
                 target_features: List[str] = None,
                 label_features: List[str] = None,
                 ordinal_features: Dict[str, List[str]] = None,
                 onehot_max_categories: int = 50,
                 onehot_drop_first: bool = True,
                 target_smoothing: float = 1.0,
                 target_min_samples_leaf: int = 20,
                 target_cv_folds: int = 5,
                 handle_unknown: str = 'ignore',
                 random_state: int = 42):
        """
        Initialize categorical encoder.
        
        Args:
            onehot_features: Features for one-hot encoding
            target_features: Features for target encoding
            label_features: Features for label encoding
            ordinal_features: Features for ordinal encoding with category orders
            onehot_max_categories: Maximum categories for one-hot encoding
            onehot_drop_first: Whether to drop first category in one-hot
            target_smoothing: Smoothing factor for target encoding
            target_min_samples_leaf: Minimum samples for target encoding
            target_cv_folds: CV folds for target encoding
            handle_unknown: How to handle unknown categories
            random_state: Random state for reproducibility
        """
        self.onehot_features = onehot_features or []
        self.target_features = target_features or []
        self.label_features = label_features or []
        self.ordinal_features = ordinal_features or {}
        self.onehot_max_categories = onehot_max_categories
        self.onehot_drop_first = onehot_drop_first
        self.target_smoothing = target_smoothing
        self.target_min_samples_leaf = target_min_samples_leaf
        self.target_cv_folds = target_cv_folds
        self.handle_unknown = handle_unknown
        self.random_state = random_state
        
        # Initialize encoders
        self.onehot_encoder_ = None
        self.target_encoder_ = None
        self.label_encoders_ = {}
        self.ordinal_encoder_ = None
        self.feature_names_out_ = []
        self.cardinality_check_ = {}
        
    def _check_cardinality(self, X: pd.DataFrame) -> Dict[str, int]:
        """Check cardinality of categorical features."""
        cardinality = {}
        for column in X.columns:
            if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                cardinality[column] = X[column].nunique()
        return cardinality
    
    def _auto_assign_encoding(self, X: pd.DataFrame) -> None:
        """Automatically assign encoding strategies based on cardinality."""
        self.cardinality_check_ = self._check_cardinality(X)
        
        for column, cardinality in self.cardinality_check_.items():
            # Skip if already assigned
            if (column in self.onehot_features or 
                column in self.target_features or 
                column in self.label_features or 
                column in self.ordinal_features):
                continue
                
            # Auto-assignment logic
            if cardinality <= 5:
                self.onehot_features.append(column)
            elif cardinality <= self.onehot_max_categories:
                self.onehot_features.append(column)
            else:
                self.label_features.append(column)
                
        logger.info(f"Auto-assigned encoding strategies based on cardinality: {self.cardinality_check_}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CategoricalEncoder':
        """
        Fit categorical encoders.
        
        Args:
            X: Input categorical features
            y: Target variable (required for target encoding)
            
        Returns:
            Self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Auto-assign encoding strategies if not specified
        self._auto_assign_encoding(X)
        
        # Fit one-hot encoder
        if self.onehot_features:
            onehot_cols = [col for col in self.onehot_features if col in X.columns]
            if onehot_cols:
                self.onehot_encoder_ = OneHotEncoder(
                    drop='first' if self.onehot_drop_first else None,
                    handle_unknown=self.handle_unknown,
                    sparse_output=False
                )
                self.onehot_encoder_.fit(X[onehot_cols])
                
        # Fit target encoder
        if self.target_features and y is not None:
            target_cols = [col for col in self.target_features if col in X.columns]
            if target_cols:
                self.target_encoder_ = TargetEncoder(
                    smoothing=self.target_smoothing,
                    min_samples_leaf=self.target_min_samples_leaf,
                    cv_folds=self.target_cv_folds,
                    random_state=self.random_state
                )
                self.target_encoder_.fit(X[target_cols], y)
                
        # Fit label encoders
        if self.label_features:
            label_cols = [col for col in self.label_features if col in X.columns]
            for col in label_cols:
                encoder = LabelEncoder()
                encoder.fit(X[col].astype(str))
                self.label_encoders_[col] = encoder
                
        # Fit ordinal encoder
        if self.ordinal_features:
            ordinal_cols = [col for col in self.ordinal_features.keys() if col in X.columns]
            if ordinal_cols:
                categories = [self.ordinal_features[col] for col in ordinal_cols]
                self.ordinal_encoder_ = OrdinalEncoder(
                    categories=categories,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                self.ordinal_encoder_.fit(X[ordinal_cols])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using fitted encoders.
        
        Args:
            X: Input categorical features
            
        Returns:
            Encoded features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        result_dfs = []
        feature_names = []
        
        # Transform one-hot features
        if self.onehot_encoder_ is not None:
            onehot_cols = [col for col in self.onehot_features if col in X.columns]
            if onehot_cols:
                onehot_encoded = self.onehot_encoder_.transform(X[onehot_cols])
                onehot_names = self.onehot_encoder_.get_feature_names_out(onehot_cols)
                onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_names, index=X.index)
                result_dfs.append(onehot_df)
                feature_names.extend(onehot_names)
                
        # Transform target features
        if self.target_encoder_ is not None:
            target_cols = [col for col in self.target_features if col in X.columns]
            if target_cols:
                target_encoded = self.target_encoder_.transform(X[target_cols])
                target_names = [f"{col}_target_encoded" for col in target_cols]
                target_encoded.columns = target_names
                result_dfs.append(target_encoded)
                feature_names.extend(target_names)
                
        # Transform label features
        if self.label_encoders_:
            label_cols = [col for col in self.label_features if col in X.columns and col in self.label_encoders_]
            if label_cols:
                label_encoded_data = {}
                for col in label_cols:
                    # Handle unknown categories
                    col_data = X[col].astype(str)
                    known_categories = set(self.label_encoders_[col].classes_)
                    unknown_mask = ~col_data.isin(known_categories)
                    
                    if unknown_mask.any():
                        # Use most frequent category for unknowns
                        most_frequent = self.label_encoders_[col].classes_[0]
                        col_data = col_data.copy()
                        col_data.loc[unknown_mask] = most_frequent
                        
                    encoded_values = self.label_encoders_[col].transform(col_data)
                    label_encoded_data[f"{col}_label_encoded"] = encoded_values
                    
                label_df = pd.DataFrame(label_encoded_data, index=X.index)
                result_dfs.append(label_df)
                feature_names.extend(label_df.columns)
                
        # Transform ordinal features
        if self.ordinal_encoder_ is not None:
            ordinal_cols = [col for col in self.ordinal_features.keys() if col in X.columns]
            if ordinal_cols:
                ordinal_encoded = self.ordinal_encoder_.transform(X[ordinal_cols])
                ordinal_names = [f"{col}_ordinal_encoded" for col in ordinal_cols]
                ordinal_df = pd.DataFrame(ordinal_encoded, columns=ordinal_names, index=X.index)
                result_dfs.append(ordinal_df)
                feature_names.extend(ordinal_names)
        
        # Combine all encoded features
        if result_dfs:
            result = pd.concat(result_dfs, axis=1)
        else:
            result = pd.DataFrame(index=X.index)
            
        self.feature_names_out_ = feature_names
        return result
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit and transform categorical features.
        
        Args:
            X: Input categorical features
            y: Target variable
            
        Returns:
            Encoded features
        """
        # Special handling for target encoding with CV
        if self.target_features and y is not None:
            # Fit other encoders normally
            temp_target_features = self.target_features.copy()
            self.target_features = []  # Temporarily remove target features
            self.fit(X, y)
            self.target_features = temp_target_features  # Restore target features
            
            # Handle target encoding separately with CV
            target_cols = [col for col in self.target_features if col in X.columns]
            if target_cols:
                self.target_encoder_ = TargetEncoder(
                    smoothing=self.target_smoothing,
                    min_samples_leaf=self.target_min_samples_leaf,
                    cv_folds=self.target_cv_folds,
                    random_state=self.random_state
                )
                target_encoded = self.target_encoder_.fit_transform(X[target_cols], y)
                target_names = [f"{col}_target_encoded" for col in target_cols]
                target_encoded.columns = target_names
                
                # Transform other features normally
                other_result = self.transform(X)
                
                # Remove target encoded columns from other_result if they exist
                target_encoded_cols = [col for col in other_result.columns if col.endswith('_target_encoded')]
                other_result = other_result.drop(columns=target_encoded_cols, errors='ignore')
                
                # Combine results
                if not other_result.empty:
                    result = pd.concat([other_result, target_encoded], axis=1)
                else:
                    result = target_encoded
                    
                self.feature_names_out_ = list(result.columns)
                return result
        
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features: List[str] = None) -> List[str]:
        """Get output feature names."""
        return self.feature_names_out_
    
    def get_encoding_info(self) -> Dict[str, Any]:
        """Get information about applied encodings."""
        info = {
            'onehot_features': self.onehot_features,
            'target_features': self.target_features, 
            'label_features': self.label_features,
            'ordinal_features': list(self.ordinal_features.keys()),
            'cardinality_check': self.cardinality_check_,
            'output_features': len(self.feature_names_out_)
        }
        return info