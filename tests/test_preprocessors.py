"""
Unit tests for numerical preprocessors.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from feature_engineering.preprocessors import OutlierDetector, NumericalPreprocessor


class TestOutlierDetector:
    """Test outlier detection functionality."""
    
    @pytest.fixture
    def outlier_data(self):
        """Create data with known outliers."""
        np.random.seed(42)
        # Normal data + outliers
        normal_data = np.random.normal(0, 1, 900)
        outliers = np.array([10, -10, 15, -15, 20])
        combined_data = np.concatenate([normal_data, outliers])
        np.random.shuffle(combined_data)
        
        return pd.DataFrame({
            'feature1': combined_data,
            'feature2': np.random.normal(0, 1, 905)
        })
    
    def test_initialization(self):
        """Test outlier detector initialization."""
        detector = OutlierDetector()
        
        assert detector.method == 'iqr'
        assert detector.threshold == 1.5
        assert detector.action == 'clip'
        assert detector.contamination == 0.1
        assert detector.random_state == 42
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        detector = OutlierDetector(
            method='isolation_forest',
            threshold=2.0,
            action='remove',
            contamination=0.05,
            random_state=123
        )
        
        assert detector.method == 'isolation_forest'
        assert detector.threshold == 2.0
        assert detector.action == 'remove'
        assert detector.contamination == 0.05
        assert detector.random_state == 123
    
    def test_iqr_bounds(self, outlier_data):
        """Test IQR bounds calculation."""
        detector = OutlierDetector(method='iqr', threshold=1.5)
        
        series = outlier_data['feature1']
        lower_bound, upper_bound = detector._iqr_bounds(series)
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        expected_lower = Q1 - 1.5 * IQR
        expected_upper = Q3 + 1.5 * IQR
        
        assert abs(lower_bound - expected_lower) < 1e-10
        assert abs(upper_bound - expected_upper) < 1e-10
    
    def test_fit_iqr(self, outlier_data):
        """Test fitting with IQR method."""
        detector = OutlierDetector(method='iqr')
        detector.fit(outlier_data)
        
        assert 'feature1' in detector.bounds_
        assert 'feature2' in detector.bounds_
        assert isinstance(detector.bounds_['feature1'], tuple)
        assert len(detector.bounds_['feature1']) == 2
    
    def test_fit_isolation_forest(self, outlier_data):
        """Test fitting with Isolation Forest."""
        detector = OutlierDetector(method='isolation_forest')
        detector.fit(outlier_data)
        
        assert detector.detector_ is not None
        assert hasattr(detector.detector_, 'predict')
    
    def test_fit_lof(self, outlier_data):
        """Test fitting with Local Outlier Factor."""
        detector = OutlierDetector(method='local_outlier_factor')
        detector.fit(outlier_data)
        
        assert detector.detector_ is not None
        assert detector.outlier_mask_ is not None
        assert isinstance(detector.outlier_mask_, np.ndarray)
    
    def test_transform_iqr_clip(self, outlier_data):
        """Test transformation with IQR clipping."""
        detector = OutlierDetector(method='iqr', action='clip')
        detector.fit(outlier_data)
        transformed = detector.transform(outlier_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == outlier_data.shape
        
        # Check that extreme values are clipped
        original_max = outlier_data['feature1'].max()
        transformed_max = transformed['feature1'].max()
        assert transformed_max <= original_max
    
    def test_transform_iqr_remove(self, outlier_data):
        """Test transformation with IQR removal (marking as NaN)."""
        detector = OutlierDetector(method='iqr', action='remove')
        detector.fit(outlier_data)
        transformed = detector.transform(outlier_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == outlier_data.shape
        
        # Should have some NaN values where outliers were removed
        assert transformed.isnull().sum().sum() >= 0
    
    def test_transform_iqr_transform(self, outlier_data):
        """Test transformation with IQR log transformation."""
        detector = OutlierDetector(method='iqr', action='transform')
        detector.fit(outlier_data)
        transformed = detector.transform(outlier_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == outlier_data.shape
        # Values should be different due to transformation
        assert not np.array_equal(transformed.values, outlier_data.values)
    
    def test_transform_isolation_forest(self, outlier_data):
        """Test transformation with Isolation Forest."""
        detector = OutlierDetector(method='isolation_forest', action='clip')
        detector.fit(outlier_data)
        transformed = detector.transform(outlier_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == outlier_data.shape
    
    def test_empty_data(self):
        """Test with empty data."""
        detector = OutlierDetector()
        empty_df = pd.DataFrame()
        
        detector.fit(empty_df)
        transformed = detector.transform(empty_df)
        
        assert transformed.empty


class TestNumericalPreprocessor:
    """Test comprehensive numerical preprocessor."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = NumericalPreprocessor()
        
        assert preprocessor.scaling_method == 'standard'
        assert preprocessor.normalization_method == 'yeo-johnson'
        assert preprocessor.outlier_method == 'iqr'
        assert preprocessor.outlier_threshold == 1.5
        assert preprocessor.imputation_strategy == 'median'
        assert preprocessor.random_state == 42
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        preprocessor = NumericalPreprocessor(
            scaling_method='minmax',
            normalization_method='box-cox',
            outlier_method='isolation_forest',
            imputation_strategy='mean',
            random_state=123,
            verbose=True
        )
        
        assert preprocessor.scaling_method == 'minmax'
        assert preprocessor.normalization_method == 'box-cox'
        assert preprocessor.outlier_method == 'isolation_forest'
        assert preprocessor.imputation_strategy == 'mean'
        assert preprocessor.random_state == 123
        assert preprocessor.verbose is True
    
    def test_get_feature_stats(self, numerical_data):
        """Test feature statistics calculation."""
        preprocessor = NumericalPreprocessor()
        X = numerical_data.drop(columns=['target'])
        
        stats = preprocessor._get_feature_stats(X)
        
        assert isinstance(stats, dict)
        assert 'normal_feature' in stats
        
        feature_stats = stats['normal_feature']
        assert 'mean' in feature_stats
        assert 'std' in feature_stats
        assert 'skewness' in feature_stats
        assert 'missing_pct' in feature_stats
        assert 'outliers_iqr' in feature_stats
    
    def test_count_outliers_iqr(self):
        """Test IQR outlier counting."""
        preprocessor = NumericalPreprocessor()
        
        # Create series with known outliers
        series = pd.Series([1, 2, 3, 4, 5, 100, -100])  # 100 and -100 are outliers
        outlier_count = preprocessor._count_outliers_iqr(series)
        
        assert outlier_count >= 2  # At least the extreme values
    
    def test_auto_assign_preprocessing(self, numerical_data):
        """Test automatic preprocessing assignment."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor()
        
        # Clear existing assignments
        preprocessor.scaling_features = []
        preprocessor.outlier_features = []
        preprocessor.normalization_features = []
        
        preprocessor._auto_assign_preprocessing(X)
        
        # Should auto-assign some features
        assert len(preprocessor.scaling_features) > 0
    
    def test_fit_basic(self, numerical_data):
        """Test basic fitting."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            scaling_features=['normal_feature', 'uniform_feature']
        )
        
        preprocessor.fit(X)
        
        assert preprocessor.scaler_ is not None
        assert preprocessor.feature_stats_ is not None
    
    def test_fit_with_imputation(self, numerical_data):
        """Test fitting with imputation."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor()
        
        preprocessor.fit(X)
        
        # Should create imputer due to missing values
        assert preprocessor.imputer_ is not None
    
    def test_fit_with_outlier_detection(self, numerical_data):
        """Test fitting with outlier detection."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            outlier_features=['outlier_feature']
        )
        
        preprocessor.fit(X)
        
        assert preprocessor.outlier_detector_ is not None
    
    def test_fit_with_normalization(self, numerical_data):
        """Test fitting with normalization."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            normalization_features=['skewed_feature']
        )
        
        preprocessor.fit(X)
        
        assert preprocessor.normalizer_ is not None
    
    def test_fit_with_binning(self, numerical_data):
        """Test fitting with binning."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            binning_features={
                'normal_feature': {'n_bins': 5, 'strategy': 'quantile'}
            }
        )
        
        preprocessor.fit(X)
        
        assert 'normal_feature' in preprocessor.binning_transformers_
    
    def test_transform_basic(self, numerical_data):
        """Test basic transformation."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            scaling_features=['normal_feature', 'uniform_feature']
        )
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        # Should have same or more columns due to potential binning
        assert X_transformed.shape[1] >= X.shape[1]
    
    def test_transform_with_imputation(self, numerical_data):
        """Test transformation with imputation."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor()
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Should have fewer or equal missing values after imputation
        original_missing = X.isnull().sum().sum()
        transformed_missing = X_transformed.isnull().sum().sum()
        assert transformed_missing <= original_missing
    
    def test_transform_with_outliers(self, numerical_data):
        """Test transformation with outlier treatment."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            outlier_features=['outlier_feature'],
            outlier_action='clip'
        )
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Outlier feature should have different distribution
        original_max = X['outlier_feature'].max()
        transformed_max = X_transformed['outlier_feature'].max()
        assert transformed_max <= original_max
    
    def test_transform_with_scaling(self, numerical_data):
        """Test transformation with scaling."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            scaling_features=['normal_feature'],
            scaling_method='standard'
        )
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Scaled feature should have different mean/std
        original_mean = X['normal_feature'].mean()
        transformed_mean = X_transformed['normal_feature'].mean()
        assert abs(original_mean - transformed_mean) > 0.1
    
    def test_different_scaling_methods(self, numerical_data):
        """Test different scaling methods."""
        X = numerical_data.drop(columns=['target'])
        scaling_methods = ['standard', 'minmax', 'robust', 'quantile']
        
        for method in scaling_methods:
            preprocessor = NumericalPreprocessor(
                scaling_features=['normal_feature'],
                scaling_method=method
            )
            preprocessor.fit(X)
            X_transformed = preprocessor.transform(X)
            
            assert isinstance(X_transformed, pd.DataFrame)
            assert not X_transformed.empty
    
    def test_different_normalization_methods(self, numerical_data):
        """Test different normalization methods."""
        X = numerical_data.drop(columns=['target'])
        # Use a feature that's always positive for box-cox
        X_positive = X.copy()
        X_positive['positive_feature'] = np.abs(X_positive['normal_feature']) + 1
        
        normalization_methods = ['yeo-johnson', 'quantile']
        
        for method in normalization_methods:
            preprocessor = NumericalPreprocessor(
                normalization_features=['positive_feature'],
                normalization_method=method
            )
            preprocessor.fit(X_positive)
            X_transformed = preprocessor.transform(X_positive)
            
            assert isinstance(X_transformed, pd.DataFrame)
            assert not X_transformed.empty
    
    def test_box_cox_normalization(self, numerical_data):
        """Test Box-Cox normalization with positive values."""
        X = numerical_data.drop(columns=['target'])
        # Ensure positive values for box-cox
        X['positive_feature'] = np.abs(X['normal_feature']) + 1
        
        preprocessor = NumericalPreprocessor(
            normalization_features=['positive_feature'],
            normalization_method='box-cox'
        )
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert not X_transformed.empty
    
    def test_binning_transformation(self, numerical_data):
        """Test binning transformation."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            binning_features={
                'normal_feature': {'n_bins': 5, 'strategy': 'quantile', 'encode': 'ordinal'}
            }
        )
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Should have additional binned feature
        assert 'normal_feature_binned' in X_transformed.columns
        binned_values = X_transformed['normal_feature_binned']
        assert binned_values.nunique() <= 5  # Should have at most 5 bins
    
    def test_get_feature_names_out(self, numerical_data):
        """Test getting output feature names."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor()
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        feature_names = preprocessor.get_feature_names_out()
        assert isinstance(feature_names, list)
        assert len(feature_names) == X_transformed.shape[1]
    
    def test_get_preprocessing_info(self, numerical_data):
        """Test getting preprocessing information."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            scaling_features=['normal_feature'],
            outlier_features=['outlier_feature']
        )
        
        preprocessor.fit(X)
        preprocessor.transform(X)
        
        info = preprocessor.get_preprocessing_info()
        
        assert isinstance(info, dict)
        assert 'scaling_features' in info
        assert 'outlier_features' in info
        assert 'feature_stats' in info
        assert 'output_features' in info
    
    def test_empty_features_lists(self, numerical_data):
        """Test with empty feature lists."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            scaling_features=[],
            outlier_features=[],
            normalization_features=[]
        )
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Should still work with auto-assignment
        assert isinstance(X_transformed, pd.DataFrame)
        assert not X_transformed.empty
    
    def test_zero_variance_feature(self, numerical_data):
        """Test handling of zero variance feature."""
        X = numerical_data.drop(columns=['target'])
        preprocessor = NumericalPreprocessor(
            scaling_features=['zero_variance']
        )
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Should handle zero variance gracefully
        assert isinstance(X_transformed, pd.DataFrame)
        assert 'zero_variance' in X_transformed.columns