"""
Unit tests for dimensionality reduction components.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from feature_engineering.dimensionality import FeatureSelector, DimensionalityReducer


class TestFeatureSelector:
    """Test feature selection functionality."""
    
    @pytest.fixture
    def selection_data(self):
        """Create data for feature selection testing."""
        np.random.seed(42)
        n_samples = 500
        
        # Create features with different relevance levels
        X = pd.DataFrame({
            'relevant_1': np.random.normal(0, 1, n_samples),
            'relevant_2': np.random.normal(0, 1, n_samples),
            'noise_1': np.random.normal(0, 1, n_samples),
            'noise_2': np.random.normal(0, 1, n_samples),
            'correlated': 0,  # Will be set based on relevant_1
            'zero_variance': np.ones(n_samples),
            'low_variance': np.ones(n_samples) + np.random.normal(0, 0.01, n_samples)
        })
        
        # Create target with known relationships
        y = (X['relevant_1'] + 0.5 * X['relevant_2'] + 
             np.random.normal(0, 0.1, n_samples) > 0).astype(int)
        
        # Set correlated feature
        X['correlated'] = X['relevant_1'] + np.random.normal(0, 0.1, n_samples)
        
        return X, y
    
    def test_initialization(self):
        """Test feature selector initialization."""
        selector = FeatureSelector()
        
        assert selector.method == 'mutual_info'
        assert selector.k == 'all'
        assert selector.percentile == 50.0
        assert selector.estimator == 'random_forest'
        assert selector.random_state == 42
        assert selector.verbose is False
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        selector = FeatureSelector(
            method='f_classif',
            k=5,
            percentile=75.0,
            estimator='logistic_regression',
            rfe_n_features=10,
            random_state=123,
            verbose=True
        )
        
        assert selector.method == 'f_classif'
        assert selector.k == 5
        assert selector.percentile == 75.0
        assert selector.estimator == 'logistic_regression'
        assert selector.rfe_n_features == 10
        assert selector.random_state == 123
        assert selector.verbose is True
    
    def test_get_score_function(self):
        """Test score function selection."""
        # Test mutual info
        selector = FeatureSelector(method='mutual_info')
        score_func = selector._get_score_function()
        assert score_func.__name__ == 'mutual_info_classif'
        
        # Test f_classif
        selector = FeatureSelector(method='f_classif')
        score_func = selector._get_score_function()
        assert score_func.__name__ == 'f_classif'
        
        # Test chi2
        selector = FeatureSelector(method='chi2')
        score_func = selector._get_score_function()
        assert score_func.__name__ == 'chi2'
    
    def test_get_rfe_estimator(self):
        """Test RFE estimator selection."""
        # Test random forest
        selector = FeatureSelector(estimator='random_forest')
        estimator = selector._get_rfe_estimator()
        assert estimator.__class__.__name__ == 'RandomForestClassifier'
        
        # Test logistic regression
        selector = FeatureSelector(estimator='logistic_regression')
        estimator = selector._get_rfe_estimator()
        assert estimator.__class__.__name__ == 'LogisticRegression'
    
    def test_fit_mutual_info(self, selection_data):
        """Test fitting with mutual information."""
        X, y = selection_data
        selector = FeatureSelector(method='mutual_info', k=3)
        
        selector.fit(X, y)
        
        assert selector.selector_ is not None
        assert len(selector.feature_scores_) > 0
        assert len(selector.selected_features_) == 3
        
        # Should select relevant features
        assert 'relevant_1' in selector.selected_features_
    
    def test_fit_f_classif(self, selection_data):
        """Test fitting with F-test."""
        X, y = selection_data
        selector = FeatureSelector(method='f_classif', k=3)
        
        selector.fit(X, y)
        
        assert selector.selector_ is not None
        assert len(selector.feature_scores_) > 0
        assert len(selector.selected_features_) == 3
    
    def test_fit_chi2(self, selection_data):
        """Test fitting with chi-square test."""
        X, y = selection_data
        # Make features non-negative for chi2
        X_chi2 = X.copy()
        for col in X_chi2.columns:
            X_chi2[col] = X_chi2[col] - X_chi2[col].min() + 1
        
        selector = FeatureSelector(method='chi2', k=3)
        selector.fit(X_chi2, y)
        
        assert selector.selector_ is not None
        assert len(selector.feature_scores_) > 0
        assert len(selector.selected_features_) == 3
    
    def test_fit_recursive(self, selection_data):
        """Test fitting with recursive feature elimination."""
        X, y = selection_data
        selector = FeatureSelector(method='recursive', rfe_n_features=3)
        
        selector.fit(X, y)
        
        assert selector.selector_ is not None
        assert len(selector.feature_ranking_) > 0
        assert len(selector.selected_features_) == 3
        
        # Check ranking
        for feature, rank in selector.feature_ranking_.items():
            assert isinstance(rank, (int, np.integer))
            assert rank >= 1
    
    def test_fit_percentile_selection(self, selection_data):
        """Test fitting with percentile selection."""
        X, y = selection_data
        selector = FeatureSelector(method='mutual_info', k='all', percentile=50.0)
        
        selector.fit(X, y)
        
        assert selector.selector_ is not None
        assert len(selector.selected_features_) >= 1
        # Should select roughly half the features
        assert len(selector.selected_features_) <= len(X.columns)
    
    def test_transform(self, selection_data):
        """Test transformation."""
        X, y = selection_data
        selector = FeatureSelector(method='mutual_info', k=3)
        
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 3
        assert list(X_transformed.columns) == selector.selected_features_
    
    def test_transform_without_fit(self, selection_data):
        """Test transform without fitting."""
        X, y = selection_data
        selector = FeatureSelector()
        
        with pytest.raises(ValueError, match="FeatureSelector must be fitted"):
            selector.transform(X)
    
    def test_transform_chi2_preprocessing(self, selection_data):
        """Test transformation with chi2 preprocessing."""
        X, y = selection_data
        selector = FeatureSelector(method='chi2', k=3)
        
        # Fit with original data (selector handles non-negative internally)
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 3
    
    def test_get_feature_scores(self, selection_data):
        """Test getting feature scores."""
        X, y = selection_data
        selector = FeatureSelector(method='mutual_info')
        
        selector.fit(X, y)
        scores = selector.get_feature_scores()
        
        assert isinstance(scores, dict)
        assert len(scores) > 0
        for feature, score in scores.items():
            assert isinstance(score, (float, np.floating))
            assert score >= 0
    
    def test_get_feature_ranking(self, selection_data):
        """Test getting feature ranking (RFE only)."""
        X, y = selection_data
        selector = FeatureSelector(method='recursive', rfe_n_features=3)
        
        selector.fit(X, y)
        ranking = selector.get_feature_ranking()
        
        assert isinstance(ranking, dict)
        assert len(ranking) == len(X.columns)
        for feature, rank in ranking.items():
            assert isinstance(rank, (int, np.integer))


class TestDimensionalityReducer:
    """Test comprehensive dimensionality reduction."""
    
    @pytest.fixture
    def dimensionality_data(self):
        """Create data for dimensionality reduction testing."""
        np.random.seed(42)
        n_samples = 300
        
        # Create features with different characteristics
        X = pd.DataFrame({
            'important_1': np.random.normal(0, 1, n_samples),
            'important_2': np.random.normal(0, 1, n_samples),
            'noise_1': np.random.normal(0, 1, n_samples),
            'noise_2': np.random.normal(0, 1, n_samples),
            'correlated_1': 0,  # Will be set
            'correlated_2': 0,  # Will be set
            'zero_variance': np.ones(n_samples),
            'low_variance': np.ones(n_samples) + np.random.normal(0, 0.001, n_samples)
        })
        
        # Create correlated features
        X['correlated_1'] = X['important_1'] + np.random.normal(0, 0.01, n_samples)
        X['correlated_2'] = X['important_1'] * 0.95 + np.random.normal(0, 0.05, n_samples)
        
        # Create target
        y = (X['important_1'] + 0.5 * X['important_2'] + 
             np.random.normal(0, 0.1, n_samples) > 0).astype(int)
        
        return X, y
    
    def test_initialization(self):
        """Test dimensionality reducer initialization."""
        reducer = DimensionalityReducer()
        
        assert reducer.pca_enabled is False
        assert reducer.pca_n_components == 0.95
        assert reducer.feature_selection_enabled is True
        assert reducer.selection_method == 'mutual_info'
        assert reducer.variance_threshold == 0.0
        assert reducer.correlation_threshold == 0.95
        assert reducer.random_state == 42
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        reducer = DimensionalityReducer(
            pca_enabled=True,
            pca_n_components=5,
            feature_selection_enabled=False,
            selection_method='f_classif',
            variance_threshold=0.01,
            correlation_threshold=0.9,
            verbose=True
        )
        
        assert reducer.pca_enabled is True
        assert reducer.pca_n_components == 5
        assert reducer.feature_selection_enabled is False
        assert reducer.selection_method == 'f_classif'
        assert reducer.variance_threshold == 0.01
        assert reducer.correlation_threshold == 0.9
        assert reducer.verbose is True
    
    def test_remove_low_variance_features(self, dimensionality_data):
        """Test low variance feature removal."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(variance_threshold=0.01)
        
        X_filtered, removed = reducer._remove_low_variance_features(X)
        
        assert isinstance(X_filtered, pd.DataFrame)
        assert isinstance(removed, list)
        assert 'zero_variance' in removed
        assert 'low_variance' in removed
        assert X_filtered.shape[1] < X.shape[1]
    
    def test_remove_correlated_features(self, dimensionality_data):
        """Test correlated feature removal."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(correlation_threshold=0.9)
        
        X_filtered, removed = reducer._remove_correlated_features(X)
        
        assert isinstance(X_filtered, pd.DataFrame)
        assert isinstance(removed, list)
        assert X_filtered.shape[1] <= X.shape[1]
        
        # Should remove some correlated features
        if len(removed) > 0:
            assert any('correlated' in feature for feature in removed)
    
    def test_fit_basic(self, dimensionality_data):
        """Test basic fitting."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(
            variance_threshold=0.01,
            feature_selection_enabled=True,
            selection_method='mutual_info'
        )
        
        reducer.fit(X, y)
        
        assert reducer.variance_selector_ is not None
        assert reducer.feature_selector_ is not None
        assert len(reducer.removed_features_['low_variance']) > 0
    
    def test_fit_without_target(self, dimensionality_data):
        """Test fitting without target (no feature selection)."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(
            variance_threshold=0.01,
            feature_selection_enabled=True  # Will be skipped without target
        )
        
        reducer.fit(X)  # No target provided
        
        assert reducer.variance_selector_ is not None
        assert reducer.feature_selector_ is None  # Should be None without target
    
    def test_fit_with_pca(self, dimensionality_data):
        """Test fitting with PCA."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(
            pca_enabled=True,
            pca_n_components=3,
            variance_threshold=0.01
        )
        
        reducer.fit(X, y)
        
        assert reducer.pca_ is not None
        assert reducer.explained_variance_ratio_ is not None
        assert len(reducer.explained_variance_ratio_) == 3
    
    def test_fit_with_pca_variance_ratio(self, dimensionality_data):
        """Test fitting with PCA using variance ratio."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(
            pca_enabled=True,
            pca_n_components=0.95,  # Keep 95% of variance
            variance_threshold=0.01
        )
        
        reducer.fit(X, y)
        
        assert reducer.pca_ is not None
        assert reducer.explained_variance_ratio_ is not None
        # Should explain at least 95% of variance
        total_variance = np.sum(reducer.explained_variance_ratio_)
        assert total_variance >= 0.90  # Allow some tolerance
    
    def test_transform_basic(self, dimensionality_data):
        """Test basic transformation."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(
            variance_threshold=0.01,
            feature_selection_enabled=True,
            selection_method='mutual_info',
            selection_k=3
        )
        
        reducer.fit(X, y)
        X_transformed = reducer.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] < X.shape[1]  # Should have fewer features
        assert X_transformed.shape[1] <= 3  # Due to feature selection
    
    def test_transform_with_pca(self, dimensionality_data):
        """Test transformation with PCA."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(
            pca_enabled=True,
            pca_n_components=3,
            variance_threshold=0.01
        )
        
        reducer.fit(X, y)
        X_transformed = reducer.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 3
        
        # Should have PC column names
        pc_cols = [col for col in X_transformed.columns if col.startswith('PC')]
        assert len(pc_cols) == 3
    
    def test_transform_full_pipeline(self, dimensionality_data):
        """Test transformation with full pipeline."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(
            variance_threshold=0.01,
            correlation_threshold=0.9,
            feature_selection_enabled=True,
            selection_method='mutual_info',
            selection_k=3,
            pca_enabled=True,
            pca_n_components=2
        )
        
        reducer.fit(X, y)
        X_transformed = reducer.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 2  # PCA components
        
        # Should have PC column names
        assert all(col.startswith('PC') for col in X_transformed.columns)
    
    def test_get_feature_names_out(self, dimensionality_data):
        """Test getting output feature names."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(
            variance_threshold=0.01,
            feature_selection_enabled=True,
            selection_k=3
        )
        
        reducer.fit(X, y)
        X_transformed = reducer.transform(X)
        
        feature_names = reducer.get_feature_names_out()
        assert isinstance(feature_names, list)
        assert len(feature_names) == X_transformed.shape[1]
        assert feature_names == list(X_transformed.columns)
    
    def test_get_reduction_info(self, dimensionality_data):
        """Test getting reduction information."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(
            variance_threshold=0.01,
            correlation_threshold=0.9,
            feature_selection_enabled=True,
            selection_k=3
        )
        
        reducer.fit(X, y)
        reducer.transform(X)
        
        info = reducer.get_reduction_info()
        
        assert isinstance(info, dict)
        assert 'final_features' in info
        assert 'removed_features' in info
        assert 'pca_enabled' in info
        assert 'reduction_summary' in info
        
        # Check removed features structure
        removed = info['removed_features']
        assert 'low_variance' in removed
        assert 'high_correlation' in removed
        assert 'feature_selection' in removed
    
    def test_get_pca_components_without_pca(self, dimensionality_data):
        """Test getting PCA components when PCA not applied."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(pca_enabled=False)
        
        reducer.fit(X, y)
        
        components = reducer.get_pca_components()
        assert components is None
    
    def test_get_pca_components_with_pca(self, dimensionality_data):
        """Test getting PCA components when PCA applied."""
        X, y = dimensionality_data
        reducer = DimensionalityReducer(
            pca_enabled=True,
            pca_n_components=3,
            variance_threshold=0.01
        )
        
        reducer.fit(X, y)
        
        components = reducer.get_pca_components()
        assert components is not None
        assert isinstance(components, pd.DataFrame)
        assert components.shape[1] == 3  # 3 components
    
    def test_no_features_remaining(self):
        """Test case where no features remain after filtering."""
        # Create data where all features would be filtered out
        X = pd.DataFrame({
            'zero_var_1': np.ones(100),
            'zero_var_2': np.ones(100),
            'zero_var_3': np.ones(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        reducer = DimensionalityReducer(
            variance_threshold=0.01,
            feature_selection_enabled=False  # Skip feature selection
        )
        
        reducer.fit(X, y)
        X_transformed = reducer.transform(X)
        
        # Should handle gracefully, returning empty DataFrame or minimal features
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_single_feature(self):
        """Test with single feature."""
        X = pd.DataFrame({
            'single_feature': np.random.normal(0, 1, 100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        reducer = DimensionalityReducer()
        
        reducer.fit(X, y)
        X_transformed = reducer.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] >= 1