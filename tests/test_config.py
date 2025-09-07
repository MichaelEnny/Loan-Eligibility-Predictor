"""
Unit tests for configuration management.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path

from feature_engineering.config import (
    FeaturePipelineConfig,
    CategoricalEncodingConfig,
    NumericalPreprocessingConfig,
    FeatureInteractionConfig,
    DimensionalityReductionConfig,
    create_default_loan_config
)


class TestCategoricalEncodingConfig:
    """Test categorical encoding configuration."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = CategoricalEncodingConfig()
        
        assert config.onehot_features == []
        assert config.target_features == []
        assert config.label_features == []
        assert config.ordinal_features == {}
        assert config.onehot_drop_first is True
        assert config.target_smoothing == 1.0
        assert config.target_min_samples_leaf == 20
        assert config.target_cv_folds == 5
    
    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = CategoricalEncodingConfig(
            onehot_features=['feat1', 'feat2'],
            target_features=['feat3'],
            target_smoothing=2.0,
            target_cv_folds=3
        )
        
        assert config.onehot_features == ['feat1', 'feat2']
        assert config.target_features == ['feat3']
        assert config.target_smoothing == 2.0
        assert config.target_cv_folds == 3
    
    def test_validation_success(self):
        """Test successful validation."""
        config = CategoricalEncodingConfig()
        config.validate()  # Should not raise
    
    def test_validation_errors(self):
        """Test validation error cases."""
        # Negative smoothing
        with pytest.raises(ValueError, match="target_smoothing must be non-negative"):
            config = CategoricalEncodingConfig(target_smoothing=-1.0)
            config.validate()
        
        # Invalid min samples leaf
        with pytest.raises(ValueError, match="target_min_samples_leaf must be positive"):
            config = CategoricalEncodingConfig(target_min_samples_leaf=0)
            config.validate()
        
        # Invalid CV folds
        with pytest.raises(ValueError, match="target_cv_folds must be at least 2"):
            config = CategoricalEncodingConfig(target_cv_folds=1)
            config.validate()
        
        # Invalid max categories
        with pytest.raises(ValueError, match="onehot_max_categories must be positive"):
            config = CategoricalEncodingConfig(onehot_max_categories=0)
            config.validate()


class TestNumericalPreprocessingConfig:
    """Test numerical preprocessing configuration."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = NumericalPreprocessingConfig()
        
        assert config.scaling_method == 'standard'
        assert config.normalization_method == 'yeo-johnson'
        assert config.outlier_method == 'iqr'
        assert config.outlier_threshold == 1.5
        assert config.outlier_action == 'clip'
        assert config.imputation_strategy == 'median'
    
    def test_validation_success(self):
        """Test successful validation."""
        config = NumericalPreprocessingConfig()
        config.validate()  # Should not raise
    
    def test_validation_errors(self):
        """Test validation error cases."""
        # Invalid scaling method
        with pytest.raises(ValueError, match="scaling_method must be one of"):
            config = NumericalPreprocessingConfig(scaling_method='invalid')
            config.validate()
        
        # Invalid normalization method
        with pytest.raises(ValueError, match="normalization_method must be one of"):
            config = NumericalPreprocessingConfig(normalization_method='invalid')
            config.validate()
        
        # Invalid outlier method
        with pytest.raises(ValueError, match="outlier_method must be one of"):
            config = NumericalPreprocessingConfig(outlier_method='invalid')
            config.validate()
        
        # Invalid outlier action
        with pytest.raises(ValueError, match="outlier_action must be one of"):
            config = NumericalPreprocessingConfig(outlier_action='invalid')
            config.validate()
        
        # Invalid outlier threshold
        with pytest.raises(ValueError, match="outlier_threshold must be positive"):
            config = NumericalPreprocessingConfig(outlier_threshold=0)
            config.validate()


class TestFeatureInteractionConfig:
    """Test feature interaction configuration."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = FeatureInteractionConfig()
        
        assert config.polynomial_degree == 2
        assert config.polynomial_include_bias is False
        assert config.polynomial_interaction_only is True
        assert 'log' in config.math_operations
        assert '+' in config.arithmetic_operations
    
    def test_validation_success(self):
        """Test successful validation."""
        config = FeatureInteractionConfig()
        config.validate()  # Should not raise
    
    def test_validation_errors(self):
        """Test validation error cases."""
        # Invalid polynomial degree
        with pytest.raises(ValueError, match="polynomial_degree must be positive"):
            config = FeatureInteractionConfig(polynomial_degree=0)
            config.validate()
        
        # Invalid math operations
        with pytest.raises(ValueError, match="Invalid math operations"):
            config = FeatureInteractionConfig(math_operations=['invalid_op'])
            config.validate()
        
        # Invalid arithmetic operations
        with pytest.raises(ValueError, match="Invalid arithmetic operations"):
            config = FeatureInteractionConfig(arithmetic_operations=['invalid_op'])
            config.validate()


class TestDimensionalityReductionConfig:
    """Test dimensionality reduction configuration."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = DimensionalityReductionConfig()
        
        assert config.pca_enabled is False
        assert config.pca_n_components == 0.95
        assert config.feature_selection_enabled is True
        assert config.selection_method == 'mutual_info'
        assert config.variance_threshold == 0.0
    
    def test_validation_success(self):
        """Test successful validation."""
        config = DimensionalityReductionConfig()
        config.validate()  # Should not raise
    
    def test_validation_errors(self):
        """Test validation error cases."""
        # Invalid PCA components (int)
        with pytest.raises(ValueError, match="pca_n_components must be positive when int"):
            config = DimensionalityReductionConfig(pca_n_components=0)
            config.validate()
        
        # Invalid PCA components (float)
        with pytest.raises(ValueError, match="pca_n_components must be between 0 and 1 when float"):
            config = DimensionalityReductionConfig(pca_n_components=1.5)
            config.validate()
        
        # Invalid selection method
        with pytest.raises(ValueError, match="selection_method must be one of"):
            config = DimensionalityReductionConfig(selection_method='invalid')
            config.validate()
        
        # Invalid variance threshold
        with pytest.raises(ValueError, match="variance_threshold must be non-negative"):
            config = DimensionalityReductionConfig(variance_threshold=-0.1)
            config.validate()


class TestFeaturePipelineConfig:
    """Test main pipeline configuration."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = FeaturePipelineConfig()
        
        assert config.target_column == 'loan_approved'
        assert config.n_jobs == -1
        assert config.random_state == 42
        assert config.verbose is False
        assert isinstance(config.categorical, CategoricalEncodingConfig)
        assert isinstance(config.numerical, NumericalPreprocessingConfig)
        assert isinstance(config.interactions, FeatureInteractionConfig)
        assert isinstance(config.dimensionality, DimensionalityReductionConfig)
    
    def test_validation_success(self):
        """Test successful validation."""
        config = FeaturePipelineConfig()
        config.validate()  # Should not raise
    
    def test_validation_errors(self):
        """Test validation error cases."""
        # Empty target column
        with pytest.raises(ValueError, match="target_column must be specified"):
            config = FeaturePipelineConfig(target_column='')
            config.validate()
        
        # Target in exclude columns
        with pytest.raises(ValueError, match="target_column cannot be in exclude_columns"):
            config = FeaturePipelineConfig(
                target_column='target',
                exclude_columns=['target']
            )
            config.validate()
    
    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'target_column': 'target',
            'categorical': {'onehot_features': ['feat1']},
            'numerical': {'scaling_method': 'minmax'},
            'random_state': 123
        }
        
        config = FeaturePipelineConfig.from_dict(config_dict)
        
        assert config.target_column == 'target'
        assert config.categorical.onehot_features == ['feat1']
        assert config.numerical.scaling_method == 'minmax'
        assert config.random_state == 123
    
    def test_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = FeaturePipelineConfig(
            target_column='test_target',
            random_state=999
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['target_column'] == 'test_target'
        assert config_dict['random_state'] == 999
        assert 'categorical' in config_dict
        assert 'numerical' in config_dict
        assert 'interactions' in config_dict
        assert 'dimensionality' in config_dict
    
    def test_save_and_load_yaml(self, temp_dir):
        """Test saving and loading YAML configuration."""
        config = FeaturePipelineConfig(target_column='test_target')
        yaml_path = temp_dir / 'config.yaml'
        
        # Save configuration
        config.save(yaml_path)
        assert yaml_path.exists()
        
        # Load configuration
        loaded_config = FeaturePipelineConfig.from_file(yaml_path)
        assert loaded_config.target_column == 'test_target'
    
    def test_save_and_load_json(self, temp_dir):
        """Test saving and loading JSON configuration."""
        config = FeaturePipelineConfig(target_column='test_target')
        json_path = temp_dir / 'config.json'
        
        # Save configuration
        config.save(json_path)
        assert json_path.exists()
        
        # Load configuration
        loaded_config = FeaturePipelineConfig.from_file(json_path)
        assert loaded_config.target_column == 'test_target'
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            FeaturePipelineConfig.from_file('nonexistent.yaml')
    
    def test_unsupported_file_format(self, temp_dir):
        """Test unsupported file format."""
        config = FeaturePipelineConfig()
        unsupported_path = temp_dir / 'config.txt'
        
        with pytest.raises(ValueError, match="Unsupported config file format"):
            config.save(unsupported_path)


class TestDefaultLoanConfig:
    """Test default loan configuration creation."""
    
    def test_create_default_config(self):
        """Test default loan configuration creation."""
        config = create_default_loan_config()
        
        assert isinstance(config, FeaturePipelineConfig)
        assert config.target_column == 'loan_approved'
        assert len(config.categorical.onehot_features) > 0
        assert len(config.numerical.scaling_features) > 0
        assert len(config.interactions.ratio_pairs) > 0
        assert config.dimensionality.feature_selection_enabled is True
    
    def test_default_config_validation(self):
        """Test that default configuration is valid."""
        config = create_default_loan_config()
        config.validate()  # Should not raise
    
    def test_default_config_features(self):
        """Test specific features in default configuration."""
        config = create_default_loan_config()
        
        # Check some expected categorical features
        expected_onehot = ['gender', 'marital_status', 'education']
        for feature in expected_onehot:
            assert feature in config.categorical.onehot_features
        
        # Check some expected numerical features
        expected_scaling = ['age', 'annual_income', 'credit_score']
        for feature in expected_scaling:
            assert feature in config.numerical.scaling_features
        
        # Check ratio pairs
        assert len(config.interactions.ratio_pairs) >= 3
        
        # Check domain features
        assert 'debt_burden_score' in config.interactions.domain_features