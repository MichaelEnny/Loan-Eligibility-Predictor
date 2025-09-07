"""
Unit tests for the main feature engineering pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pickle
import json
from pathlib import Path

from feature_engineering.pipeline import FeatureEngineeringPipeline
from feature_engineering.config import FeaturePipelineConfig, create_default_loan_config


class TestFeatureEngineeringPipeline:
    """Test comprehensive feature engineering pipeline."""
    
    def test_initialization_with_config(self, default_config):
        """Test initialization with configuration object."""
        pipeline = FeatureEngineeringPipeline(config=default_config)
        
        assert pipeline.config == default_config
        assert pipeline.config.target_column == 'loan_approved'
        assert pipeline.is_fitted_ is False
    
    def test_initialization_with_kwargs(self):
        """Test initialization with keyword arguments."""
        pipeline = FeatureEngineeringPipeline(
            target_column='target',
            random_state=123,
            verbose=True
        )
        
        assert pipeline.config.target_column == 'target'
        assert pipeline.config.random_state == 123
        assert pipeline.config.verbose is True
    
    def test_initialization_minimal(self, minimal_config):
        """Test initialization with minimal configuration."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        assert pipeline.config.target_column == 'loan_approved'
        assert pipeline.config.n_jobs == 1
        assert pipeline.config.random_state == 42
    
    def test_identify_feature_types(self, sample_features):
        """Test feature type identification."""
        pipeline = FeatureEngineeringPipeline()
        
        categorical, numerical = pipeline._identify_feature_types(sample_features)
        
        assert isinstance(categorical, list)
        assert isinstance(numerical, list)
        
        # Check some expected categorizations
        assert 'gender' in categorical
        assert 'age' in numerical
        assert 'annual_income' in numerical
    
    def test_validate_input(self, sample_loan_data):
        """Test input validation."""
        pipeline = FeatureEngineeringPipeline()
        
        X_cleaned, y_cleaned = pipeline._validate_input(
            sample_loan_data, 
            sample_loan_data['loan_approved']
        )
        
        assert isinstance(X_cleaned, pd.DataFrame)
        assert isinstance(y_cleaned, pd.Series)
        assert 'loan_approved' not in X_cleaned.columns  # Should be removed
        assert len(X_cleaned) == len(y_cleaned)
    
    def test_validate_input_extract_target(self, sample_loan_data):
        """Test input validation with target extraction."""
        pipeline = FeatureEngineeringPipeline()
        
        # Include target in features DataFrame
        X_with_target = sample_loan_data.copy()
        X_cleaned, y_extracted = pipeline._validate_input(X_with_target, None)
        
        assert isinstance(X_cleaned, pd.DataFrame)
        assert isinstance(y_extracted, pd.Series)
        assert 'loan_approved' not in X_cleaned.columns
        assert len(y_extracted) == len(X_cleaned)
    
    def test_validate_input_missing_target(self, sample_features):
        """Test input validation with missing target values."""
        pipeline = FeatureEngineeringPipeline()
        
        # Create target with some missing values
        y_with_missing = pd.Series([1, 0, np.nan, 1, 0] * 200)
        
        X_cleaned, y_cleaned = pipeline._validate_input(sample_features, y_with_missing)
        
        assert len(X_cleaned) == len(y_cleaned)
        assert not y_cleaned.isnull().any()
    
    def test_create_encoders(self, default_config):
        """Test encoder creation from configuration."""
        pipeline = FeatureEngineeringPipeline(config=default_config)
        
        # Test categorical encoder creation
        cat_encoder = pipeline._create_categorical_encoder()
        assert cat_encoder is not None
        assert len(cat_encoder.onehot_features) > 0
        
        # Test numerical preprocessor creation
        num_preprocessor = pipeline._create_numerical_preprocessor()
        assert num_preprocessor is not None
        assert len(num_preprocessor.scaling_features) > 0
        
        # Test interaction generator creation
        interaction_gen = pipeline._create_interaction_generator()
        assert interaction_gen is not None
        assert len(interaction_gen.ratio_pairs) > 0
        
        # Test dimensionality reducer creation
        dim_reducer = pipeline._create_dimensionality_reducer()
        assert dim_reducer is not None
        assert dim_reducer.feature_selection_enabled is True
    
    def test_fit_basic(self, sample_features, sample_target, minimal_config):
        """Test basic pipeline fitting."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        pipeline.fit(sample_features, sample_target)
        
        assert pipeline.is_fitted_ is True
        assert pipeline.fit_time_ is not None
        assert pipeline.fit_time_ > 0
        assert len(pipeline.feature_names_in_) > 0
        assert len(pipeline.feature_names_out_) > 0
        assert len(pipeline.processing_stats_) > 0
    
    def test_fit_with_categorical_features(self, sample_features, sample_target):
        """Test fitting with categorical features."""
        config = FeaturePipelineConfig(
            target_column='loan_approved',
            random_state=42
        )
        config.categorical.onehot_features = ['gender', 'marital_status']
        config.categorical.target_features = ['education']
        
        pipeline = FeatureEngineeringPipeline(config=config)
        pipeline.fit(sample_features, sample_target)
        
        assert pipeline.categorical_encoder_ is not None
        assert pipeline.is_fitted_ is True
        assert 'categorical_encoding' in pipeline.processing_stats_['steps_completed']
    
    def test_fit_with_numerical_features(self, sample_features, sample_target):
        """Test fitting with numerical preprocessing."""
        config = FeaturePipelineConfig(
            target_column='loan_approved',
            random_state=42
        )
        config.numerical.scaling_features = ['age', 'annual_income']
        config.numerical.outlier_features = ['annual_income']
        
        pipeline = FeatureEngineeringPipeline(config=config)
        pipeline.fit(sample_features, sample_target)
        
        assert pipeline.numerical_preprocessor_ is not None
        assert pipeline.is_fitted_ is True
        assert 'numerical_preprocessing' in pipeline.processing_stats_['steps_completed']
    
    def test_fit_with_interactions(self, sample_features, sample_target):
        """Test fitting with feature interactions."""
        config = FeaturePipelineConfig(
            target_column='loan_approved',
            random_state=42
        )
        config.interactions.ratio_pairs = [('loan_amount', 'annual_income')]
        config.interactions.polynomial_features = ['credit_score']
        
        pipeline = FeatureEngineeringPipeline(config=config)
        pipeline.fit(sample_features, sample_target)
        
        assert pipeline.interaction_generator_ is not None
        assert pipeline.is_fitted_ is True
        assert 'feature_interactions' in pipeline.processing_stats_['steps_completed']
    
    def test_fit_with_dimensionality_reduction(self, sample_features, sample_target):
        """Test fitting with dimensionality reduction."""
        config = FeaturePipelineConfig(
            target_column='loan_approved',
            random_state=42
        )
        config.dimensionality.feature_selection_enabled = True
        config.dimensionality.selection_k = 10
        config.dimensionality.variance_threshold = 0.01
        
        pipeline = FeatureEngineeringPipeline(config=config)
        pipeline.fit(sample_features, sample_target)
        
        assert pipeline.dimensionality_reducer_ is not None
        assert pipeline.is_fitted_ is True
        assert 'dimensionality_reduction' in pipeline.processing_stats_['steps_completed']
    
    def test_fit_with_pca(self, sample_features, sample_target):
        """Test fitting with PCA."""
        config = FeaturePipelineConfig(
            target_column='loan_approved',
            random_state=42
        )
        config.dimensionality.pca_enabled = True
        config.dimensionality.pca_n_components = 5
        
        pipeline = FeatureEngineeringPipeline(config=config)
        pipeline.fit(sample_features, sample_target)
        
        assert pipeline.dimensionality_reducer_ is not None
        assert pipeline.is_fitted_ is True
        assert len(pipeline.feature_names_out_) == 5  # PCA components
    
    def test_transform(self, sample_features, sample_target, minimal_config):
        """Test pipeline transformation."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        pipeline.fit(sample_features, sample_target)
        X_transformed = pipeline.transform(sample_features)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == sample_features.shape[0]
        assert pipeline.transform_time_ is not None
        assert pipeline.transform_time_ > 0
    
    def test_transform_without_fit(self, sample_features, minimal_config):
        """Test transform without fitting should raise error."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        with pytest.raises(ValueError, match="Pipeline must be fitted"):
            pipeline.transform(sample_features)
    
    def test_fit_transform(self, sample_features, sample_target, minimal_config):
        """Test fit_transform method."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        X_transformed = pipeline.fit_transform(sample_features, sample_target)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == sample_features.shape[0]
        assert pipeline.is_fitted_ is True
    
    def test_comprehensive_pipeline(self, sample_features, sample_target):
        """Test comprehensive pipeline with all components."""
        config = FeaturePipelineConfig(
            target_column='loan_approved',
            random_state=42,
            verbose=False
        )
        
        # Configure all components
        config.categorical.onehot_features = ['gender', 'marital_status']
        config.categorical.target_features = ['education']
        config.numerical.scaling_features = ['age', 'annual_income', 'credit_score']
        config.numerical.outlier_features = ['annual_income']
        config.interactions.ratio_pairs = [('loan_amount', 'annual_income')]
        config.dimensionality.feature_selection_enabled = True
        config.dimensionality.selection_k = 15
        
        pipeline = FeatureEngineeringPipeline(config=config)
        X_transformed = pipeline.fit_transform(sample_features, sample_target)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == sample_features.shape[0]
        assert pipeline.is_fitted_ is True
        
        # Check that all components were created
        assert pipeline.categorical_encoder_ is not None
        assert pipeline.numerical_preprocessor_ is not None
        assert pipeline.interaction_generator_ is not None
        assert pipeline.dimensionality_reducer_ is not None
        
        # Check all steps completed
        expected_steps = [
            'categorical_encoding',
            'numerical_preprocessing', 
            'feature_interactions',
            'dimensionality_reduction'
        ]
        for step in expected_steps:
            assert step in pipeline.processing_stats_['steps_completed']
    
    def test_get_feature_names_out(self, sample_features, sample_target, minimal_config):
        """Test getting output feature names."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        pipeline.fit(sample_features, sample_target)
        feature_names = pipeline.get_feature_names_out()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert feature_names == pipeline.feature_names_out_
    
    def test_get_feature_info(self, sample_features, sample_target, minimal_config):
        """Test getting comprehensive feature information."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        pipeline.fit(sample_features, sample_target)
        pipeline.transform(sample_features)
        
        info = pipeline.get_feature_info()
        
        assert isinstance(info, dict)
        assert 'pipeline_config' in info
        assert 'input_features' in info
        assert 'output_features' in info
        assert 'processing_stats' in info
        assert 'feature_types' in info
        
        # Check structure
        assert info['input_features'] > 0
        assert info['output_features'] > 0
        assert isinstance(info['feature_types'], dict)
    
    def test_get_feature_info_unfitted(self):
        """Test getting feature info from unfitted pipeline."""
        pipeline = FeatureEngineeringPipeline()
        
        info = pipeline.get_feature_info()
        
        assert isinstance(info, dict)
        assert 'error' in info
        assert info['error'] == "Pipeline not fitted yet"
    
    def test_save_and_load_pipeline(self, sample_features, sample_target, temp_dir, minimal_config):
        """Test saving and loading pipeline."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        pipeline.fit(sample_features, sample_target)
        
        # Save pipeline
        save_path = temp_dir / 'test_pipeline.pkl'
        pipeline.save_pipeline(save_path)
        
        assert save_path.exists()
        assert (temp_dir / 'test_pipeline.json').exists()  # Config saved separately
        
        # Load pipeline
        loaded_pipeline = FeatureEngineeringPipeline.load_pipeline(save_path)
        
        assert loaded_pipeline.is_fitted_ is True
        assert loaded_pipeline.config.target_column == pipeline.config.target_column
        assert loaded_pipeline.feature_names_out_ == pipeline.feature_names_out_
        
        # Test that loaded pipeline works
        X_transformed = loaded_pipeline.transform(sample_features)
        assert isinstance(X_transformed, pd.DataFrame)
    
    def test_save_unfitted_pipeline(self, temp_dir, minimal_config):
        """Test saving unfitted pipeline should raise error."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        save_path = temp_dir / 'unfitted_pipeline.pkl'
        
        with pytest.raises(ValueError, match="Cannot save unfitted pipeline"):
            pipeline.save_pipeline(save_path)
    
    def test_load_nonexistent_pipeline(self):
        """Test loading non-existent pipeline should raise error."""
        with pytest.raises(FileNotFoundError):
            FeatureEngineeringPipeline.load_pipeline('nonexistent.pkl')
    
    def test_create_sample_config(self, sample_features, sample_target, temp_dir):
        """Test creating sample configuration."""
        pipeline = FeatureEngineeringPipeline()
        
        config = pipeline.create_sample_config(
            sample_features, 
            sample_target,
            save_path=temp_dir / 'sample_config.yaml'
        )
        
        assert isinstance(config, FeaturePipelineConfig)
        assert (temp_dir / 'sample_config.yaml').exists()
        
        # Should have detected some features
        assert len(config.categorical.onehot_features) > 0 or len(config.categorical.label_features) > 0
        assert len(config.numerical.scaling_features) > 0
    
    def test_validate_pipeline(self, sample_features, sample_target, minimal_config):
        """Test pipeline validation."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        results = pipeline.validate_pipeline(sample_features, sample_target)
        
        assert isinstance(results, dict)
        assert 'status' in results
        assert 'metrics' in results
        assert 'errors' in results
        assert 'warnings' in results
        
        # Should be successful
        assert results['status'] == 'success'
        
        # Check metrics
        metrics = results['metrics']
        assert 'fit_time' in metrics
        assert 'transform_time' in metrics
        assert 'input_shape' in metrics
        assert 'output_shape' in metrics
    
    def test_validate_pipeline_small_dataset(self):
        """Test pipeline validation with small dataset."""
        # Create very small dataset
        X = pd.DataFrame({
            'feat1': [1, 2, 3, 4, 5],
            'feat2': ['A', 'B', 'A', 'B', 'A']
        })
        y = pd.Series([0, 1, 0, 1, 0])
        
        pipeline = FeatureEngineeringPipeline()
        results = pipeline.validate_pipeline(X, y)
        
        assert isinstance(results, dict)
        assert 'status' in results
        # Should still work with small dataset
    
    def test_empty_dataframe(self, minimal_config):
        """Test pipeline with empty DataFrame."""
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=int)
        
        # Should handle gracefully or raise appropriate error
        try:
            pipeline.fit(empty_X, empty_y)
            X_transformed = pipeline.transform(empty_X)
            assert X_transformed.empty
        except (ValueError, IndexError):
            # Expected for empty data
            pass
    
    def test_single_feature(self, minimal_config):
        """Test pipeline with single feature."""
        X = pd.DataFrame({'single_feature': np.random.normal(0, 1, 100)})
        y = pd.Series(np.random.choice([0, 1], 100))
        
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] >= 1
    
    def test_missing_values_handling(self, minimal_config):
        """Test pipeline with missing values."""
        X = pd.DataFrame({
            'numerical': [1, 2, np.nan, 4, 5, np.nan, 7],
            'categorical': ['A', 'B', None, 'A', 'B', 'A', None]
        })
        y = pd.Series([0, 1, 0, 1, 0, 1, 0])
        
        pipeline = FeatureEngineeringPipeline(config=minimal_config)
        
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        # Should handle missing values appropriately
    
    def test_default_loan_config_integration(self, sample_features, sample_target):
        """Test integration with default loan configuration."""
        config = create_default_loan_config()
        pipeline = FeatureEngineeringPipeline(config=config)
        
        X_transformed = pipeline.fit_transform(sample_features, sample_target)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == sample_features.shape[0]
        assert pipeline.is_fitted_ is True
        
        # Should create comprehensive feature set
        assert X_transformed.shape[1] >= 10  # Should have reasonable number of features
    
    def test_verbose_mode(self, sample_features, sample_target, capfd):
        """Test verbose mode output."""
        config = FeaturePipelineConfig(
            target_column='loan_approved',
            verbose=True
        )
        
        pipeline = FeatureEngineeringPipeline(config=config)
        pipeline.fit(sample_features, sample_target)
        
        # Should have some output in verbose mode
        captured = capfd.readouterr()
        # Note: Actual logging output might not be captured by capfd
        # This test mainly ensures verbose mode doesn't break
        assert pipeline.is_fitted_ is True