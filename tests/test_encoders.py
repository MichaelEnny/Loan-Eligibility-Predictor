"""
Unit tests for categorical encoders.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from feature_engineering.encoders import TargetEncoder, CategoricalEncoder


class TestTargetEncoder:
    """Test target encoder functionality."""
    
    @pytest.fixture
    def sample_categorical_data(self):
        """Create sample categorical data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'] * 50,
            'target': [1, 0, 1, 0, 1, 0] * 50
        })
    
    def test_initialization(self):
        """Test target encoder initialization."""
        encoder = TargetEncoder()
        
        assert encoder.smoothing == 1.0
        assert encoder.min_samples_leaf == 20
        assert encoder.cv_folds == 5
        assert encoder.random_state == 42
        assert encoder.encodings_ == {}
        assert encoder.global_mean_ is None
    
    def test_custom_initialization(self):
        """Test target encoder with custom parameters."""
        encoder = TargetEncoder(
            smoothing=2.0,
            min_samples_leaf=10,
            cv_folds=3,
            random_state=123
        )
        
        assert encoder.smoothing == 2.0
        assert encoder.min_samples_leaf == 10
        assert encoder.cv_folds == 3
        assert encoder.random_state == 123
    
    def test_fit(self, sample_categorical_data):
        """Test fitting target encoder."""
        X = sample_categorical_data[['category']]
        y = sample_categorical_data['target']
        
        encoder = TargetEncoder()
        encoder.fit(X, y)
        
        assert encoder.global_mean_ is not None
        assert 'category' in encoder.encodings_
        assert len(encoder.encodings_['category']) == 3  # A, B, C
    
    def test_transform(self, sample_categorical_data):
        """Test transforming with target encoder."""
        X = sample_categorical_data[['category']]
        y = sample_categorical_data['target']
        
        encoder = TargetEncoder()
        encoder.fit(X, y)
        X_transformed = encoder.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 1
        assert X_transformed.columns[0] == 'category'
    
    def test_fit_transform_cv(self, sample_categorical_data):
        """Test fit_transform with cross-validation."""
        X = sample_categorical_data[['category']]
        y = sample_categorical_data['target']
        
        encoder = TargetEncoder(cv_folds=3)
        X_transformed = encoder.fit_transform(X, y)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == X.shape
        # Values should be different from simple mean due to CV
        assert not np.array_equal(X_transformed.values, X.values)
    
    def test_transform_without_fit(self):
        """Test transform without fitting should raise error."""
        encoder = TargetEncoder()
        X = pd.DataFrame({'cat': ['A', 'B']})
        
        with pytest.raises(ValueError, match="TargetEncoder must be fitted"):
            encoder.transform(X)
    
    def test_unknown_categories(self, sample_categorical_data):
        """Test handling of unknown categories."""
        X_train = sample_categorical_data[['category']]
        y_train = sample_categorical_data['target']
        
        # Create test data with unknown category
        X_test = pd.DataFrame({'category': ['A', 'B', 'D']})  # D is unknown
        
        encoder = TargetEncoder()
        encoder.fit(X_train, y_train)
        X_transformed = encoder.transform(X_test)
        
        # Unknown category should get global mean
        assert not pd.isna(X_transformed.iloc[2, 0])
        assert X_transformed.iloc[2, 0] == encoder.global_mean_


class TestCategoricalEncoder:
    """Test comprehensive categorical encoder."""
    
    def test_initialization(self):
        """Test categorical encoder initialization."""
        encoder = CategoricalEncoder()
        
        assert encoder.onehot_features == []
        assert encoder.target_features == []
        assert encoder.label_features == []
        assert encoder.ordinal_features == {}
        assert encoder.onehot_max_categories == 50
        assert encoder.random_state == 42
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        encoder = CategoricalEncoder(
            onehot_features=['feat1'],
            target_features=['feat2'],
            label_features=['feat3'],
            onehot_max_categories=10,
            random_state=123
        )
        
        assert encoder.onehot_features == ['feat1']
        assert encoder.target_features == ['feat2']
        assert encoder.label_features == ['feat3']
        assert encoder.onehot_max_categories == 10
        assert encoder.random_state == 123
    
    def test_cardinality_check(self, categorical_data):
        """Test cardinality checking."""
        encoder = CategoricalEncoder()
        cardinality = encoder._check_cardinality(categorical_data)
        
        assert 'low_cardinality' in cardinality
        assert 'high_cardinality' in cardinality
        assert cardinality['low_cardinality'] == 3
        assert cardinality['high_cardinality'] == 100
    
    def test_auto_assign_encoding(self, categorical_data):
        """Test automatic encoding assignment."""
        encoder = CategoricalEncoder()
        encoder._auto_assign_encoding(categorical_data)
        
        # Low cardinality should be assigned to onehot
        assert 'low_cardinality' in encoder.onehot_features
        # High cardinality should be assigned to label
        assert 'high_cardinality' in encoder.label_features
    
    def test_fit_onehot_only(self, categorical_data):
        """Test fitting with only one-hot encoding."""
        X = categorical_data[['low_cardinality', 'binary_feature']]
        y = categorical_data['target']
        
        encoder = CategoricalEncoder(
            onehot_features=['low_cardinality', 'binary_feature']
        )
        encoder.fit(X, y)
        
        assert encoder.onehot_encoder_ is not None
        assert encoder.target_encoder_ is None
        assert len(encoder.label_encoders_) == 0
    
    def test_fit_target_encoding(self, categorical_data):
        """Test fitting with target encoding."""
        X = categorical_data[['medium_cardinality']]
        y = categorical_data['target']
        
        encoder = CategoricalEncoder(
            target_features=['medium_cardinality']
        )
        encoder.fit(X, y)
        
        assert encoder.target_encoder_ is not None
        assert encoder.onehot_encoder_ is None
    
    def test_fit_label_encoding(self, categorical_data):
        """Test fitting with label encoding."""
        X = categorical_data[['high_cardinality']]
        
        encoder = CategoricalEncoder(
            label_features=['high_cardinality']
        )
        encoder.fit(X)
        
        assert 'high_cardinality' in encoder.label_encoders_
        assert encoder.onehot_encoder_ is None
        assert encoder.target_encoder_ is None
    
    def test_fit_ordinal_encoding(self):
        """Test fitting with ordinal encoding."""
        X = pd.DataFrame({
            'size': ['Small', 'Medium', 'Large', 'Small', 'Large']
        })
        
        encoder = CategoricalEncoder(
            ordinal_features={'size': ['Small', 'Medium', 'Large']}
        )
        encoder.fit(X)
        
        assert encoder.ordinal_encoder_ is not None
    
    def test_transform_onehot(self, categorical_data):
        """Test transformation with one-hot encoding."""
        X = categorical_data[['low_cardinality', 'binary_feature']]
        
        encoder = CategoricalEncoder(
            onehot_features=['low_cardinality', 'binary_feature']
        )
        encoder.fit(X)
        X_transformed = encoder.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        # Should have more columns due to one-hot expansion
        assert X_transformed.shape[1] > X.shape[1]
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_transform_target_encoding(self, categorical_data):
        """Test transformation with target encoding."""
        X = categorical_data[['medium_cardinality']]
        y = categorical_data['target']
        
        encoder = CategoricalEncoder(
            target_features=['medium_cardinality']
        )
        encoder.fit(X, y)
        X_transformed = encoder.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert 'medium_cardinality_target_encoded' in X_transformed.columns
    
    def test_transform_label_encoding(self, categorical_data):
        """Test transformation with label encoding."""
        X = categorical_data[['high_cardinality']]
        
        encoder = CategoricalEncoder(
            label_features=['high_cardinality']
        )
        encoder.fit(X)
        X_transformed = encoder.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert 'high_cardinality_label_encoded' in X_transformed.columns
        # Values should be integers
        assert X_transformed['high_cardinality_label_encoded'].dtype in ['int64', 'int32']
    
    def test_mixed_encoding_types(self, categorical_data):
        """Test mixed encoding types."""
        X = categorical_data[['low_cardinality', 'medium_cardinality', 'high_cardinality']]
        y = categorical_data['target']
        
        encoder = CategoricalEncoder(
            onehot_features=['low_cardinality'],
            target_features=['medium_cardinality'],
            label_features=['high_cardinality']
        )
        encoder.fit(X, y)
        X_transformed = encoder.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        
        # Check that all encoding types are present
        onehot_cols = [col for col in X_transformed.columns 
                      if col.startswith('low_cardinality')]
        assert len(onehot_cols) > 1  # Multiple one-hot columns
        
        assert 'medium_cardinality_target_encoded' in X_transformed.columns
        assert 'high_cardinality_label_encoded' in X_transformed.columns
    
    def test_fit_transform_cv(self, categorical_data):
        """Test fit_transform with cross-validation for target encoding."""
        X = categorical_data[['medium_cardinality']]
        y = categorical_data['target']
        
        encoder = CategoricalEncoder(
            target_features=['medium_cardinality'],
            target_cv_folds=3
        )
        X_transformed = encoder.fit_transform(X, y)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert 'medium_cardinality_target_encoded' in X_transformed.columns
    
    def test_get_feature_names_out(self, categorical_data):
        """Test getting output feature names."""
        X = categorical_data[['low_cardinality', 'medium_cardinality']]
        y = categorical_data['target']
        
        encoder = CategoricalEncoder(
            onehot_features=['low_cardinality'],
            target_features=['medium_cardinality']
        )
        encoder.fit(X, y)
        encoder.transform(X)
        
        feature_names = encoder.get_feature_names_out()
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
    
    def test_get_encoding_info(self, categorical_data):
        """Test getting encoding information."""
        X = categorical_data[['low_cardinality', 'high_cardinality']]
        
        encoder = CategoricalEncoder(
            onehot_features=['low_cardinality'],
            label_features=['high_cardinality']
        )
        encoder.fit(X)
        encoder.transform(X)
        
        info = encoder.get_encoding_info()
        
        assert isinstance(info, dict)
        assert 'onehot_features' in info
        assert 'label_features' in info
        assert 'cardinality_check' in info
        assert 'output_features' in info
    
    def test_handle_unknown_categories_label(self):
        """Test handling unknown categories in label encoding."""
        # Train data
        X_train = pd.DataFrame({'cat': ['A', 'B', 'C']})
        # Test data with unknown category
        X_test = pd.DataFrame({'cat': ['A', 'B', 'D']})
        
        encoder = CategoricalEncoder(label_features=['cat'])
        encoder.fit(X_train)
        X_transformed = encoder.transform(X_test)
        
        # Should handle unknown category gracefully
        assert not pd.isna(X_transformed).any().any()
    
    def test_empty_dataframe(self):
        """Test handling empty dataframe."""
        X = pd.DataFrame()
        encoder = CategoricalEncoder()
        encoder.fit(X)
        X_transformed = encoder.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.empty