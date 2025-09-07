"""
Complete Feature Engineering Pipeline
Orchestrates all feature engineering components for production-ready ML workflows.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
import logging
import time
from pathlib import Path
import pickle
import json

from .config import FeaturePipelineConfig
from .encoders import CategoricalEncoder
from .preprocessors import NumericalPreprocessor
from .interactions import FeatureInteractionGenerator
from .dimensionality import DimensionalityReducer

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """
    Complete feature engineering pipeline for production ML systems.
    
    Orchestrates categorical encoding, numerical preprocessing, feature interactions,
    and dimensionality reduction with comprehensive monitoring, validation, and
    performance optimization.
    """
    
    def __init__(self, 
                 config: Optional[FeaturePipelineConfig] = None,
                 **kwargs):
        """
        Initialize feature engineering pipeline.
        
        Args:
            config: Pipeline configuration object
            **kwargs: Configuration parameters (used if config not provided)
        """
        if config is not None:
            self.config = config
        else:
            # Create config from kwargs
            self.config = FeaturePipelineConfig(**kwargs)
        
        # Validate configuration
        self.config.validate()
        
        # Pipeline components
        self.categorical_encoder_ = None
        self.numerical_preprocessor_ = None
        self.interaction_generator_ = None
        self.dimensionality_reducer_ = None
        
        # Pipeline metadata
        self.feature_names_in_ = []
        self.feature_names_out_ = []
        self.categorical_features_ = []
        self.numerical_features_ = []
        self.processing_stats_ = {}
        self.fit_time_ = None
        self.transform_time_ = None
        self.is_fitted_ = False
        
    def _identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify categorical and numerical features."""
        categorical_features = []
        numerical_features = []
        
        for column in X.columns:
            if column == self.config.target_column:
                continue
            if column in self.config.exclude_columns:
                continue
                
            # Check data type and unique values
            if (X[column].dtype == 'object' or 
                X[column].dtype.name == 'category' or
                (X[column].dtype in ['int64', 'float64'] and X[column].nunique() <= 20)):
                categorical_features.append(column)
            else:
                numerical_features.append(column)
        
        return categorical_features, numerical_features
    
    def _validate_input(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and clean input data."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
        
        if y is not None and not isinstance(y, pd.Series):
            if hasattr(y, '__iter__'):
                y = pd.Series(y, index=X.index)
            else:
                raise ValueError("Target y must be a pandas Series or array-like")
        
        # Check for target column in features
        if self.config.target_column in X.columns:
            if y is None:
                y = X[self.config.target_column].copy()
            X = X.drop(columns=[self.config.target_column])
        
        # Remove excluded columns
        X_cleaned = X.drop(columns=self.config.exclude_columns, errors='ignore')
        
        # Handle missing target values
        if y is not None:
            valid_mask = ~y.isnull()
            if not valid_mask.all():
                logger.warning(f"Removing {(~valid_mask).sum()} rows with missing target values")
                X_cleaned = X_cleaned.loc[valid_mask]
                y = y.loc[valid_mask]
        
        return X_cleaned, y
    
    def _create_categorical_encoder(self) -> CategoricalEncoder:
        """Create categorical encoder from configuration."""
        return CategoricalEncoder(
            onehot_features=self.config.categorical.onehot_features,
            target_features=self.config.categorical.target_features,
            label_features=self.config.categorical.label_features,
            ordinal_features=self.config.categorical.ordinal_features,
            onehot_max_categories=self.config.categorical.onehot_max_categories,
            onehot_drop_first=self.config.categorical.onehot_drop_first,
            target_smoothing=self.config.categorical.target_smoothing,
            target_min_samples_leaf=self.config.categorical.target_min_samples_leaf,
            target_cv_folds=self.config.categorical.target_cv_folds,
            handle_unknown=self.config.categorical.onehot_handle_unknown,
            random_state=self.config.random_state
        )
    
    def _create_numerical_preprocessor(self) -> NumericalPreprocessor:
        """Create numerical preprocessor from configuration."""
        return NumericalPreprocessor(
            scaling_features=self.config.numerical.scaling_features,
            scaling_method=self.config.numerical.scaling_method,
            normalization_features=self.config.numerical.normalization_features,
            normalization_method=self.config.numerical.normalization_method,
            outlier_features=self.config.numerical.outlier_features,
            outlier_method=self.config.numerical.outlier_method,
            outlier_threshold=self.config.numerical.outlier_threshold,
            outlier_action=self.config.numerical.outlier_action,
            binning_features=self.config.numerical.binning_features,
            imputation_strategy=self.config.numerical.imputation_strategy,
            imputation_constant=self.config.numerical.imputation_constant,
            quantile_range=self.config.numerical.quantile_range,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )
    
    def _create_interaction_generator(self) -> FeatureInteractionGenerator:
        """Create feature interaction generator from configuration."""
        return FeatureInteractionGenerator(
            polynomial_features=self.config.interactions.polynomial_features,
            polynomial_degree=self.config.interactions.polynomial_degree,
            polynomial_include_bias=self.config.interactions.polynomial_include_bias,
            polynomial_interaction_only=self.config.interactions.polynomial_interaction_only,
            math_features=self.config.interactions.math_features,
            math_operations=self.config.interactions.math_operations,
            ratio_pairs=self.config.interactions.ratio_pairs,
            arithmetic_pairs=self.config.interactions.arithmetic_pairs,
            arithmetic_operations=self.config.interactions.arithmetic_operations,
            domain_features=self.config.interactions.domain_features,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )
    
    def _create_dimensionality_reducer(self) -> DimensionalityReducer:
        """Create dimensionality reducer from configuration."""
        return DimensionalityReducer(
            pca_enabled=self.config.dimensionality.pca_enabled,
            pca_n_components=self.config.dimensionality.pca_n_components,
            pca_whiten=self.config.dimensionality.pca_whiten,
            pca_svd_solver=self.config.dimensionality.pca_svd_solver,
            feature_selection_enabled=self.config.dimensionality.feature_selection_enabled,
            selection_method=self.config.dimensionality.selection_method,
            selection_k=self.config.dimensionality.selection_k,
            selection_percentile=self.config.dimensionality.selection_percentile,
            rfe_estimator=self.config.dimensionality.rfe_estimator,
            rfe_n_features=self.config.dimensionality.rfe_n_features,
            rfe_step=self.config.dimensionality.rfe_step,
            variance_threshold=self.config.dimensionality.variance_threshold,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeatureEngineeringPipeline':
        """
        Fit the complete feature engineering pipeline.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        
        if self.config.verbose:
            logger.info("Starting feature engineering pipeline fit...")
        
        # Validate input
        if self.config.validate_input:
            X, y = self._validate_input(X, y)
        
        # Store input feature names
        self.feature_names_in_ = list(X.columns)
        
        # Identify feature types
        self.categorical_features_, self.numerical_features_ = self._identify_feature_types(X)
        
        if self.config.verbose:
            logger.info(f"Identified {len(self.categorical_features_)} categorical and {len(self.numerical_features_)} numerical features")
        
        # Initialize processing stats
        self.processing_stats_ = {
            'input_shape': X.shape,
            'categorical_features': len(self.categorical_features_),
            'numerical_features': len(self.numerical_features_),
            'missing_values': X.isnull().sum().sum(),
            'steps_completed': []
        }
        
        # Step 1: Categorical Encoding
        if self.categorical_features_:
            if self.config.verbose:
                logger.info("Fitting categorical encoder...")
            
            # Auto-assign categorical features if not specified in config
            if not any([self.config.categorical.onehot_features,
                       self.config.categorical.target_features,
                       self.config.categorical.label_features,
                       self.config.categorical.ordinal_features]):
                # Let the encoder auto-assign based on cardinality
                pass
            
            self.categorical_encoder_ = self._create_categorical_encoder()
            X_categorical = X[self.categorical_features_]
            self.categorical_encoder_.fit(X_categorical, y)
            
            self.processing_stats_['steps_completed'].append('categorical_encoding')
        
        # Step 2: Numerical Preprocessing
        if self.numerical_features_:
            if self.config.verbose:
                logger.info("Fitting numerical preprocessor...")
            
            self.numerical_preprocessor_ = self._create_numerical_preprocessor()
            X_numerical = X[self.numerical_features_]
            self.numerical_preprocessor_.fit(X_numerical, y)
            
            self.processing_stats_['steps_completed'].append('numerical_preprocessing')
        
        # Get initial processed features for interaction generation
        processed_dfs = []
        
        if self.categorical_encoder_ is not None:
            X_cat_encoded = self.categorical_encoder_.transform(X[self.categorical_features_])
            processed_dfs.append(X_cat_encoded)
        
        if self.numerical_preprocessor_ is not None:
            X_num_processed = self.numerical_preprocessor_.transform(X[self.numerical_features_])
            processed_dfs.append(X_num_processed)
        
        if processed_dfs:
            X_processed = pd.concat(processed_dfs, axis=1)
        else:
            X_processed = X.copy()
        
        # Step 3: Feature Interactions
        if (self.config.interactions.polynomial_features or
            self.config.interactions.math_features or
            self.config.interactions.ratio_pairs or
            self.config.interactions.arithmetic_pairs or
            self.config.interactions.domain_features):
            
            if self.config.verbose:
                logger.info("Fitting interaction generator...")
            
            self.interaction_generator_ = self._create_interaction_generator()
            self.interaction_generator_.fit(X_processed, y)
            
            # Apply interactions
            X_processed = self.interaction_generator_.transform(X_processed)
            
            self.processing_stats_['steps_completed'].append('feature_interactions')
        
        # Step 4: Dimensionality Reduction
        if (self.config.dimensionality.feature_selection_enabled or 
            self.config.dimensionality.pca_enabled or
            self.config.dimensionality.variance_threshold > 0):
            
            if self.config.verbose:
                logger.info("Fitting dimensionality reducer...")
            
            self.dimensionality_reducer_ = self._create_dimensionality_reducer()
            self.dimensionality_reducer_.fit(X_processed, y)
            
            # Apply dimensionality reduction to get final feature names
            X_final = self.dimensionality_reducer_.transform(X_processed)
            self.feature_names_out_ = list(X_final.columns)
            
            self.processing_stats_['steps_completed'].append('dimensionality_reduction')
        else:
            self.feature_names_out_ = list(X_processed.columns)
        
        # Final statistics
        self.fit_time_ = time.time() - start_time
        self.processing_stats_['fit_time'] = self.fit_time_
        self.processing_stats_['final_features'] = len(self.feature_names_out_)
        self.is_fitted_ = True
        
        if self.config.verbose:
            logger.info(f"Pipeline fitting completed in {self.fit_time_:.2f}s")
            logger.info(f"Final feature count: {len(self.feature_names_out_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before transform")
        
        start_time = time.time()
        
        # Validate input
        if self.config.validate_input:
            X, _ = self._validate_input(X, None)
        
        processed_dfs = []
        
        # Step 1: Categorical Encoding
        if self.categorical_encoder_ is not None:
            X_categorical = X[self.categorical_features_]
            X_cat_encoded = self.categorical_encoder_.transform(X_categorical)
            processed_dfs.append(X_cat_encoded)
        
        # Step 2: Numerical Preprocessing
        if self.numerical_preprocessor_ is not None:
            X_numerical = X[self.numerical_features_]
            X_num_processed = self.numerical_preprocessor_.transform(X_numerical)
            processed_dfs.append(X_num_processed)
        
        # Combine categorical and numerical features
        if processed_dfs:
            X_processed = pd.concat(processed_dfs, axis=1)
        else:
            X_processed = X.copy()
        
        # Step 3: Feature Interactions
        if self.interaction_generator_ is not None:
            X_processed = self.interaction_generator_.transform(X_processed)
        
        # Step 4: Dimensionality Reduction
        if self.dimensionality_reducer_ is not None:
            X_processed = self.dimensionality_reducer_.transform(X_processed)
        
        self.transform_time_ = time.time() - start_time
        
        return X_processed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit pipeline and transform features in one step.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features: List[str] = None) -> List[str]:
        """Get output feature names."""
        return self.feature_names_out_
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get comprehensive feature engineering information."""
        if not self.is_fitted_:
            return {"error": "Pipeline not fitted yet"}
        
        info = {
            'pipeline_config': {
                'target_column': self.config.target_column,
                'exclude_columns': self.config.exclude_columns,
                'random_state': self.config.random_state
            },
            'input_features': len(self.feature_names_in_),
            'output_features': len(self.feature_names_out_),
            'processing_stats': self.processing_stats_.copy(),
            'feature_types': {
                'categorical': len(self.categorical_features_),
                'numerical': len(self.numerical_features_)
            }
        }
        
        # Add component-specific information
        if self.categorical_encoder_ is not None:
            info['categorical_encoding'] = self.categorical_encoder_.get_encoding_info()
        
        if self.numerical_preprocessor_ is not None:
            info['numerical_preprocessing'] = self.numerical_preprocessor_.get_preprocessing_info()
        
        if self.interaction_generator_ is not None:
            info['feature_interactions'] = self.interaction_generator_.get_interaction_info()
        
        if self.dimensionality_reducer_ is not None:
            reduction_info = self.dimensionality_reducer_.get_reduction_info()
            reduction_info['original_features'] = len(self.feature_names_in_)
            info['dimensionality_reduction'] = reduction_info
        
        return info
    
    def save_pipeline(self, filepath: Union[str, Path]) -> None:
        """Save the fitted pipeline to disk."""
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted pipeline")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the entire pipeline
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        # Save configuration and metadata separately
        config_path = filepath.with_suffix('.json')
        pipeline_info = {
            'config': self.config.to_dict(),
            'feature_info': self.get_feature_info(),
            'feature_names_in': self.feature_names_in_,
            'feature_names_out': self.feature_names_out_
        }
        
        with open(config_path, 'w') as f:
            json.dump(pipeline_info, f, indent=2, default=str)
        
        if self.config.verbose:
            logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: Union[str, Path]) -> 'FeatureEngineeringPipeline':
        """Load a fitted pipeline from disk."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline
    
    def create_sample_config(self, X: pd.DataFrame, y: pd.Series = None, 
                           save_path: Optional[str] = None) -> FeaturePipelineConfig:
        """
        Create a sample configuration based on the input data.
        
        Args:
            X: Sample input data
            y: Sample target data
            save_path: Path to save the configuration
            
        Returns:
            Generated configuration
        """
        # Analyze the data to create appropriate configuration
        categorical_features, numerical_features = self._identify_feature_types(X)
        
        # Create configuration based on data characteristics
        from .config import create_default_loan_config
        config = create_default_loan_config()
        
        # Override with detected features
        config.categorical.onehot_features = [col for col in categorical_features 
                                            if X[col].nunique() <= 10]
        config.categorical.label_features = [col for col in categorical_features 
                                           if X[col].nunique() > 10]
        config.numerical.scaling_features = numerical_features
        
        # Save if path provided
        if save_path:
            config.save(save_path)
            logger.info(f"Sample configuration saved to {save_path}")
        
        return config
    
    def validate_pipeline(self, X: pd.DataFrame, y: pd.Series = None, 
                         test_size: float = 0.2) -> Dict[str, Any]:
        """
        Validate the pipeline with comprehensive checks.
        
        Args:
            X: Input features
            y: Target variable
            test_size: Proportion of data for testing
            
        Returns:
            Validation results
        """
        validation_results = {
            'status': 'success',
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Split data for validation
            if y is not None and len(X) > 100:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.config.random_state
                )
            else:
                X_train, X_test = X, X
                y_train, y_test = y, y
            
            # Test fitting
            start_fit = time.time()
            self.fit(X_train, y_train)
            fit_time = time.time() - start_fit
            
            # Test transformation
            start_transform = time.time()
            X_transformed = self.transform(X_test)
            transform_time = time.time() - start_transform
            
            # Collect metrics
            validation_results['metrics'] = {
                'fit_time': fit_time,
                'transform_time': transform_time,
                'input_shape': X.shape,
                'output_shape': X_transformed.shape,
                'feature_reduction_ratio': (X.shape[1] - X_transformed.shape[1]) / X.shape[1],
                'memory_usage_mb': X_transformed.memory_usage(deep=True).sum() / 1024**2,
                'missing_values_handled': X.isnull().sum().sum() - X_transformed.isnull().sum().sum()
            }
            
            # Check for common issues
            if X_transformed.isnull().any().any():
                validation_results['warnings'].append("Output contains missing values")
            
            if X_transformed.shape[1] == 0:
                validation_results['errors'].append("No features remain after processing")
                validation_results['status'] = 'error'
            
            if X_transformed.var().min() == 0:
                validation_results['warnings'].append("Some output features have zero variance")
            
        except Exception as e:
            validation_results['status'] = 'error'
            validation_results['errors'].append(str(e))
        
        return validation_results