"""
Unit tests for feature interactions.
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from feature_engineering.interactions import MathematicalTransformer, FeatureInteractionGenerator


class TestMathematicalTransformer:
    """Test mathematical transformation functionality."""
    
    @pytest.fixture
    def math_data(self):
        """Create data for mathematical transformations."""
        np.random.seed(42)
        return pd.DataFrame({
            'positive_feature': np.random.uniform(1, 10, 100),
            'negative_feature': np.random.normal(-2, 1, 100),
            'mixed_feature': np.random.normal(0, 2, 100),
            'zero_feature': np.zeros(100),
            'small_positive': np.random.uniform(0.1, 1, 100)
        })
    
    def test_initialization(self):
        """Test mathematical transformer initialization."""
        transformer = MathematicalTransformer()
        
        assert transformer.features == []
        assert transformer.operations == ['log', 'sqrt', 'square']
        assert transformer.custom_transformations == {}
        assert transformer.handle_negatives == 'shift'
        assert transformer.handle_zeros == 'small_constant'
        assert transformer.small_constant == 1e-6
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        custom_func = lambda x: x ** 0.5
        
        transformer = MathematicalTransformer(
            features=['feat1', 'feat2'],
            operations=['log', 'exp'],
            custom_transformations={'custom': custom_func},
            handle_negatives='abs',
            handle_zeros='skip',
            small_constant=1e-8
        )
        
        assert transformer.features == ['feat1', 'feat2']
        assert transformer.operations == ['log', 'exp']
        assert 'custom' in transformer.custom_transformations
        assert transformer.handle_negatives == 'abs'
        assert transformer.handle_zeros == 'skip'
        assert transformer.small_constant == 1e-8
    
    def test_safe_log_positive(self, math_data):
        """Test safe log transformation on positive values."""
        transformer = MathematicalTransformer()
        
        result = transformer._safe_log(math_data['positive_feature'], 'positive_feature')
        
        assert isinstance(result, pd.Series)
        assert not result.isnull().any()
        assert np.all(result >= 0)  # log1p of positive values
    
    def test_safe_log_negative_shift(self, math_data):
        """Test safe log transformation with negative values (shift strategy)."""
        transformer = MathematicalTransformer(handle_negatives='shift')
        
        # Fit to calculate shift values
        transformer.fit(math_data)
        result = transformer._safe_log(math_data['negative_feature'], 'negative_feature')
        
        assert isinstance(result, pd.Series)
        assert not result.isnull().any()
    
    def test_safe_log_negative_abs(self, math_data):
        """Test safe log transformation with negative values (abs strategy)."""
        transformer = MathematicalTransformer(handle_negatives='abs')
        
        result = transformer._safe_log(math_data['negative_feature'], 'negative_feature')
        
        assert isinstance(result, pd.Series)
        assert not result.isnull().any()
    
    def test_safe_log_zeros(self):
        """Test safe log transformation with zero values."""
        transformer = MathematicalTransformer(handle_zeros='small_constant')
        
        series_with_zeros = pd.Series([0, 1, 2, 0, 3])
        result = transformer._safe_log(series_with_zeros, 'test_feature')
        
        assert isinstance(result, pd.Series)
        assert not result.isnull().any()
    
    def test_safe_sqrt_positive(self, math_data):
        """Test safe sqrt transformation on positive values."""
        transformer = MathematicalTransformer()
        
        result = transformer._safe_sqrt(math_data['positive_feature'], 'positive_feature')
        
        assert isinstance(result, pd.Series)
        assert not result.isnull().any()
        assert np.all(result >= 0)
    
    def test_safe_sqrt_negative(self, math_data):
        """Test safe sqrt transformation with negative values."""
        transformer = MathematicalTransformer(handle_negatives='abs')
        
        result = transformer._safe_sqrt(math_data['negative_feature'], 'negative_feature')
        
        assert isinstance(result, pd.Series)
        assert not result.isnull().any()
        assert np.all(result >= 0)
    
    def test_safe_reciprocal(self, math_data):
        """Test safe reciprocal transformation."""
        transformer = MathematicalTransformer()
        
        result = transformer._safe_reciprocal(math_data['positive_feature'], 'positive_feature')
        
        assert isinstance(result, pd.Series)
        assert not result.isnull().any()
        assert np.all(result > 0)  # Reciprocal of positive values
    
    def test_safe_reciprocal_zeros(self):
        """Test safe reciprocal transformation with zeros."""
        transformer = MathematicalTransformer(handle_zeros='small_constant')
        
        series_with_zeros = pd.Series([0, 1, 2, 0, 3])
        result = transformer._safe_reciprocal(series_with_zeros, 'test_feature')
        
        assert isinstance(result, pd.Series)
        assert not result.isnull().any()
    
    def test_fit(self, math_data):
        """Test fitting mathematical transformer."""
        transformer = MathematicalTransformer(
            features=['negative_feature', 'mixed_feature']
        )
        
        transformer.fit(math_data)
        
        # Should calculate shift values for negative features
        assert 'negative_feature' in transformer.shift_values_
        # Mixed feature might have negatives too
        if (math_data['mixed_feature'] < 0).any():
            assert 'mixed_feature' in transformer.shift_values_
    
    def test_transform_basic(self, math_data):
        """Test basic transformation."""
        transformer = MathematicalTransformer(
            features=['positive_feature'],
            operations=['log', 'sqrt', 'square']
        )
        
        transformer.fit(math_data)
        result = transformer.transform(math_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == math_data.shape[0]
        
        # Should have original columns plus new ones
        expected_new_cols = ['positive_feature_log', 'positive_feature_sqrt', 'positive_feature_square']
        for col in expected_new_cols:
            assert col in result.columns
        
        # Check feature names tracking
        assert len(transformer.feature_names_out_) > len(math_data.columns)
    
    def test_transform_all_operations(self, math_data):
        """Test transformation with all supported operations."""
        transformer = MathematicalTransformer(
            features=['positive_feature'],
            operations=['log', 'sqrt', 'square', 'exp', 'reciprocal']
        )
        
        transformer.fit(math_data)
        result = transformer.transform(math_data)
        
        expected_cols = [
            'positive_feature_log', 'positive_feature_sqrt', 'positive_feature_square',
            'positive_feature_exp', 'positive_feature_reciprocal'
        ]
        
        for col in expected_cols:
            assert col in result.columns
            assert not result[col].isnull().any()
    
    def test_transform_custom_operation(self, math_data):
        """Test transformation with custom operation."""
        custom_func = lambda x: x ** 3
        
        transformer = MathematicalTransformer(
            features=['positive_feature'],
            operations=['cube'],
            custom_transformations={'cube': custom_func}
        )
        
        transformer.fit(math_data)
        result = transformer.transform(math_data)
        
        assert 'positive_feature_cube' in result.columns
        # Check that transformation was applied correctly
        expected_values = math_data['positive_feature'] ** 3
        np.testing.assert_array_almost_equal(
            result['positive_feature_cube'].values, 
            expected_values.values
        )
    
    def test_transform_missing_feature(self, math_data):
        """Test transformation with missing feature."""
        transformer = MathematicalTransformer(
            features=['nonexistent_feature'],
            operations=['log']
        )
        
        transformer.fit(math_data)
        
        with warnings.catch_warnings(record=True) as w:
            result = transformer.transform(math_data)
            assert len(w) > 0
            assert "not found in input data" in str(w[0].message)
        
        # Should still return the original DataFrame
        assert result.shape == math_data.shape
    
    def test_transform_unknown_operation(self, math_data):
        """Test transformation with unknown operation."""
        transformer = MathematicalTransformer(
            features=['positive_feature'],
            operations=['unknown_op']
        )
        
        transformer.fit(math_data)
        
        with warnings.catch_warnings(record=True) as w:
            result = transformer.transform(math_data)
            assert len(w) > 0
            assert "Unknown operation" in str(w[0].message)
        
        # Should not create new columns for unknown operations
        assert result.shape[1] == math_data.shape[1]


class TestFeatureInteractionGenerator:
    """Test feature interaction generation functionality."""
    
    def test_initialization(self):
        """Test interaction generator initialization."""
        generator = FeatureInteractionGenerator()
        
        assert generator.polynomial_degree == 2
        assert generator.polynomial_include_bias is False
        assert generator.polynomial_interaction_only is True
        assert 'log' in generator.math_operations
        assert '+' in generator.arithmetic_operations
        assert generator.correlation_threshold == 0.95
        assert generator.variance_threshold == 0.01
        assert generator.random_state == 42
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        generator = FeatureInteractionGenerator(
            polynomial_features=['feat1', 'feat2'],
            polynomial_degree=3,
            math_features=['feat3'],
            ratio_pairs=[('feat1', 'feat2')],
            arithmetic_pairs=[('feat1', 'feat3')],
            correlation_threshold=0.9,
            variance_threshold=0.05,
            verbose=True
        )
        
        assert generator.polynomial_features == ['feat1', 'feat2']
        assert generator.polynomial_degree == 3
        assert generator.math_features == ['feat3']
        assert generator.ratio_pairs == [('feat1', 'feat2')]
        assert generator.arithmetic_pairs == [('feat1', 'feat3')]
        assert generator.correlation_threshold == 0.9
        assert generator.variance_threshold == 0.05
        assert generator.verbose is True
    
    def test_create_ratio_features(self, interaction_data):
        """Test ratio feature creation."""
        generator = FeatureInteractionGenerator()
        generator.ratio_pairs = [('feature_1', 'feature_2'), ('feature_3', 'feature_1')]
        
        ratio_features = generator._create_ratio_features(interaction_data)
        
        assert isinstance(ratio_features, pd.DataFrame)
        assert 'feature_1_to_feature_2_ratio' in ratio_features.columns
        assert 'feature_3_to_feature_1_ratio' in ratio_features.columns
        assert not ratio_features.isnull().all().any()  # Should handle division by zero
    
    def test_create_arithmetic_features(self, interaction_data):
        """Test arithmetic feature creation."""
        generator = FeatureInteractionGenerator()
        generator.arithmetic_pairs = [('feature_1', 'feature_2')]
        generator.arithmetic_operations = ['+', '-', '*', '/']
        
        arithmetic_features = generator._create_arithmetic_features(interaction_data)
        
        assert isinstance(arithmetic_features, pd.DataFrame)
        
        expected_cols = [
            'feature_1_plus_feature_2',
            'feature_1_minus_feature_2',
            'feature_1_times_feature_2',
            'feature_1_div_feature_2'
        ]
        
        for col in expected_cols:
            assert col in arithmetic_features.columns
    
    def test_create_domain_features_default(self, sample_loan_data):
        """Test domain feature creation with default loan features."""
        generator = FeatureInteractionGenerator()
        
        # Use sample loan data that has the required columns
        domain_features = generator._create_domain_features(sample_loan_data)
        
        assert isinstance(domain_features, pd.DataFrame)
        # Should create some domain features
        assert len(domain_features.columns) > 0
    
    def test_create_domain_features_custom(self, interaction_data):
        """Test domain feature creation with custom features."""
        custom_domain = {
            'custom_feature': {
                'formula': 'feature_1 * feature_2',
                'description': 'Product of feature 1 and 2'
            }
        }
        
        generator = FeatureInteractionGenerator(domain_features=custom_domain)
        
        # Add required columns for the formula
        test_data = interaction_data.copy()
        test_data['feature_1'] = interaction_data['feature_1']
        test_data['feature_2'] = interaction_data['feature_2']
        
        domain_features = generator._create_domain_features(test_data)
        
        assert isinstance(domain_features, pd.DataFrame)
    
    def test_get_default_loan_features(self):
        """Test getting default loan-specific features."""
        generator = FeatureInteractionGenerator()
        default_features = generator._get_default_loan_features()
        
        assert isinstance(default_features, dict)
        assert 'debt_burden_score' in default_features
        assert 'creditworthiness_score' in default_features
        assert 'loan_to_income_ratio' in default_features
        
        # Check structure of feature definitions
        for feature_name, config in default_features.items():
            assert 'formula' in config
            assert 'description' in config
    
    def test_remove_correlated_features(self, interaction_data):
        """Test removal of correlated features."""
        # Create highly correlated features
        test_data = interaction_data.copy()
        test_data['correlated_1'] = test_data['feature_1']
        test_data['correlated_2'] = test_data['feature_1'] + np.random.normal(0, 0.01, len(test_data))
        
        generator = FeatureInteractionGenerator(correlation_threshold=0.9)
        selected_features = generator._remove_correlated_features(test_data)
        
        assert isinstance(selected_features, list)
        # Should have fewer features due to correlation removal
        assert len(selected_features) < len(test_data.columns)
    
    def test_remove_low_variance_features(self):
        """Test removal of low variance features."""
        # Create data with low variance feature
        test_data = pd.DataFrame({
            'high_variance': np.random.normal(0, 1, 100),
            'low_variance': np.ones(100) + np.random.normal(0, 0.001, 100),
            'zero_variance': np.ones(100)
        })
        
        generator = FeatureInteractionGenerator(variance_threshold=0.01)
        selected_features = generator._remove_low_variance_features(test_data)
        
        assert isinstance(selected_features, list)
        assert 'high_variance' in selected_features
        assert 'zero_variance' not in selected_features
    
    def test_fit(self, interaction_data):
        """Test fitting interaction generator."""
        generator = FeatureInteractionGenerator(
            polynomial_features=['feature_1', 'feature_2'],
            math_features=['feature_3']
        )
        
        generator.fit(interaction_data)
        
        # Should initialize polynomial and math transformers
        assert generator.polynomial_transformer_ is not None
        assert generator.math_transformer_ is not None
    
    def test_transform_polynomial(self, interaction_data):
        """Test transformation with polynomial features."""
        generator = FeatureInteractionGenerator(
            polynomial_features=['feature_1', 'feature_2'],
            polynomial_degree=2
        )
        
        generator.fit(interaction_data)
        result = generator.transform(interaction_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == interaction_data.shape[0]
        # Should have more features due to polynomial expansion
        assert result.shape[1] > interaction_data.shape[1]
    
    def test_transform_mathematical(self, interaction_data):
        """Test transformation with mathematical operations."""
        generator = FeatureInteractionGenerator(
            math_features=['feature_3'],  # Use positive feature for log/sqrt
            math_operations=['log', 'sqrt', 'square']
        )
        
        generator.fit(interaction_data)
        result = generator.transform(interaction_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check for mathematical transformation columns
        math_cols = [col for col in result.columns if any(op in col for op in ['log', 'sqrt', 'square'])]
        assert len(math_cols) > 0
    
    def test_transform_ratios(self, interaction_data):
        """Test transformation with ratio features."""
        generator = FeatureInteractionGenerator(
            ratio_pairs=[('feature_1', 'feature_2'), ('feature_3', 'feature_1')]
        )
        
        generator.fit(interaction_data)
        result = generator.transform(interaction_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check for ratio columns
        ratio_cols = [col for col in result.columns if '_to_' in col and 'ratio' in col]
        assert len(ratio_cols) > 0
    
    def test_transform_arithmetic(self, interaction_data):
        """Test transformation with arithmetic combinations."""
        generator = FeatureInteractionGenerator(
            arithmetic_pairs=[('feature_1', 'feature_2')],
            arithmetic_operations=['+', '*']
        )
        
        generator.fit(interaction_data)
        result = generator.transform(interaction_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Check for arithmetic columns
        arithmetic_cols = [col for col in result.columns 
                          if any(op in col for op in ['plus', 'times'])]
        assert len(arithmetic_cols) > 0
    
    def test_transform_all_interactions(self, interaction_data):
        """Test transformation with all interaction types."""
        generator = FeatureInteractionGenerator(
            polynomial_features=['feature_1'],
            math_features=['feature_3'],
            ratio_pairs=[('feature_1', 'feature_2')],
            arithmetic_pairs=[('feature_1', 'feature_2')],
            variance_threshold=0.001  # Low threshold to keep most features
        )
        
        generator.fit(interaction_data)
        result = generator.transform(interaction_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == interaction_data.shape[0]
        # Should have significantly more features
        assert result.shape[1] > interaction_data.shape[1] * 2
    
    def test_max_features_limit(self, interaction_data):
        """Test maximum features limitation."""
        generator = FeatureInteractionGenerator(
            polynomial_features=['feature_1', 'feature_2'],
            math_features=['feature_3'],
            ratio_pairs=[('feature_1', 'feature_2')],
            max_features=10  # Limit features
        )
        
        generator.fit(interaction_data)
        result = generator.transform(interaction_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] <= 10
    
    def test_get_feature_names_out(self, interaction_data):
        """Test getting output feature names."""
        generator = FeatureInteractionGenerator(
            polynomial_features=['feature_1', 'feature_2']
        )
        
        generator.fit(interaction_data)
        result = generator.transform(interaction_data)
        
        feature_names = generator.get_feature_names_out()
        assert isinstance(feature_names, list)
        assert len(feature_names) == result.shape[1]
    
    def test_get_interaction_info(self, interaction_data):
        """Test getting interaction information."""
        generator = FeatureInteractionGenerator(
            polynomial_features=['feature_1'],
            math_features=['feature_3'],
            ratio_pairs=[('feature_1', 'feature_2')]
        )
        
        generator.fit(interaction_data)
        generator.transform(interaction_data)
        
        info = generator.get_interaction_info()
        
        assert isinstance(info, dict)
        assert 'polynomial_features' in info
        assert 'math_features' in info
        assert 'ratio_pairs' in info
        assert 'selected_features' in info
        assert 'total_features' in info
    
    def test_empty_configuration(self, interaction_data):
        """Test with empty configuration (no interactions)."""
        generator = FeatureInteractionGenerator()
        
        generator.fit(interaction_data)
        result = generator.transform(interaction_data)
        
        # Should still work and return at least the domain features
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == interaction_data.shape[0]