"""
Dimensionality Reduction Components for Feature Engineering Pipeline
Handles PCA, feature selection, and variance filtering with intelligent optimization.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, VarianceThreshold,
    mutual_info_classif, f_classif, chi2, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
import logging

logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature selection component.
    
    Supports multiple selection methods including mutual information, F-test,
    chi-square, and recursive feature elimination with intelligent feature ranking.
    """
    
    def __init__(self,
                 method: str = 'mutual_info',
                 k: Union[int, str] = 'all',
                 percentile: float = 50.0,
                 score_func: callable = None,
                 estimator: str = 'random_forest',
                 rfe_n_features: Union[int, float] = 0.5,
                 rfe_step: Union[int, float] = 1,
                 random_state: int = 42,
                 verbose: bool = False):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('mutual_info', 'f_classif', 'chi2', 'recursive')
            k: Number of features to select ('all' or integer)
            percentile: Percentile of features to select (for percentile-based methods)
            score_func: Custom scoring function
            estimator: Estimator for RFE ('random_forest', 'logistic_regression')
            rfe_n_features: Number of features for RFE
            rfe_step: Step size for RFE
            random_state: Random state for reproducibility
            verbose: Whether to print selection information
        """
        self.method = method
        self.k = k
        self.percentile = percentile
        self.score_func = score_func
        self.estimator = estimator
        self.rfe_n_features = rfe_n_features
        self.rfe_step = rfe_step
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted components
        self.selector_ = None
        self.feature_scores_ = {}
        self.selected_features_ = []
        self.feature_ranking_ = {}
        
    def _get_score_function(self):
        """Get the appropriate scoring function."""
        if self.score_func is not None:
            return self.score_func
        elif self.method == 'mutual_info':
            return mutual_info_classif
        elif self.method == 'f_classif':
            return f_classif
        elif self.method == 'chi2':
            return chi2
        else:
            return mutual_info_classif
    
    def _get_rfe_estimator(self):
        """Get the appropriate estimator for RFE."""
        if self.estimator == 'random_forest':
            return RandomForestClassifier(
                n_estimators=50, 
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.estimator == 'logistic_regression':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
        else:
            return RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit feature selector.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        # Ensure non-negative values for chi2
        X_non_negative = X.copy()
        if self.method == 'chi2':
            # Make features non-negative for chi2 test
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    if (X[col] < 0).any():
                        X_non_negative[col] = X[col] - X[col].min() + 1e-6
        
        if self.method == 'recursive':
            # Recursive Feature Elimination
            estimator = self._get_rfe_estimator()
            
            if isinstance(self.rfe_n_features, float):
                n_features = int(len(X.columns) * self.rfe_n_features)
            else:
                n_features = self.rfe_n_features
            
            self.selector_ = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=self.rfe_step,
                verbose=self.verbose
            )
            self.selector_.fit(X, y)
            
            # Get feature ranking
            self.feature_ranking_ = dict(zip(X.columns, self.selector_.ranking_))
            self.selected_features_ = [col for col, selected in 
                                     zip(X.columns, self.selector_.support_) if selected]
            
        else:
            # Statistical feature selection
            score_func = self._get_score_function()
            
            if self.k == 'all':
                # Use percentile selection
                self.selector_ = SelectPercentile(
                    score_func=score_func,
                    percentile=self.percentile
                )
            else:
                # Use k-best selection
                k_value = min(self.k, len(X.columns)) if isinstance(self.k, int) else len(X.columns)
                self.selector_ = SelectKBest(
                    score_func=score_func,
                    k=k_value
                )
            
            # Fit selector
            X_to_fit = X_non_negative if self.method == 'chi2' else X
            self.selector_.fit(X_to_fit, y)
            
            # Get feature scores and selected features
            if hasattr(self.selector_, 'scores_'):
                self.feature_scores_ = dict(zip(X.columns, self.selector_.scores_))
            
            self.selected_features_ = [col for col, selected in 
                                     zip(X.columns, self.selector_.get_support()) if selected]
        
        if self.verbose:
            logger.info(f"Selected {len(self.selected_features_)} features using {self.method}")
            if self.feature_scores_:
                top_features = sorted(self.feature_scores_.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"Top 5 features: {[f'{name}: {score:.3f}' for name, score in top_features]}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted selector.
        
        Args:
            X: Input features
            
        Returns:
            Selected features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if self.selector_ is None:
            raise ValueError("FeatureSelector must be fitted before transform")
        
        # Ensure non-negative values for chi2
        X_transform = X.copy()
        if self.method == 'chi2':
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    if (X[col] < 0).any():
                        X_transform[col] = X[col] - X[col].min() + 1e-6
        
        # Transform using selector
        if self.method == 'recursive':
            selected_data = self.selector_.transform(X_transform)
            result = pd.DataFrame(selected_data, columns=self.selected_features_, index=X.index)
        else:
            X_to_transform = X_transform if self.method == 'chi2' else X
            selected_data = self.selector_.transform(X_to_transform)
            result = pd.DataFrame(selected_data, columns=self.selected_features_, index=X.index)
        
        return result
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature scores if available."""
        return self.feature_scores_
    
    def get_feature_ranking(self) -> Dict[str, int]:
        """Get feature ranking if available."""
        return self.feature_ranking_


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Comprehensive dimensionality reduction component.
    
    Combines variance thresholding, feature selection, and PCA with intelligent
    preprocessing and optimization for production ML pipelines.
    """
    
    def __init__(self,
                 pca_enabled: bool = False,
                 pca_n_components: Union[int, float, str] = 0.95,
                 pca_whiten: bool = False,
                 pca_svd_solver: str = 'auto',
                 feature_selection_enabled: bool = True,
                 selection_method: str = 'mutual_info',
                 selection_k: Union[int, str] = 'all',
                 selection_percentile: float = 50.0,
                 rfe_estimator: str = 'random_forest',
                 rfe_n_features: Union[int, float] = 0.5,
                 rfe_step: Union[int, float] = 1,
                 variance_threshold: float = 0.0,
                 correlation_threshold: float = 0.95,
                 random_state: int = 42,
                 verbose: bool = False):
        """
        Initialize dimensionality reducer.
        
        Args:
            pca_enabled: Whether to apply PCA
            pca_n_components: Number of PCA components
            pca_whiten: Whether to whiten PCA components
            pca_svd_solver: SVD solver for PCA
            feature_selection_enabled: Whether to apply feature selection
            selection_method: Feature selection method
            selection_k: Number of features to select
            selection_percentile: Percentile of features to select
            rfe_estimator: Estimator for RFE
            rfe_n_features: Number of features for RFE
            rfe_step: Step size for RFE
            variance_threshold: Minimum variance threshold
            correlation_threshold: Correlation threshold for feature removal
            random_state: Random state for reproducibility
            verbose: Whether to print processing information
        """
        self.pca_enabled = pca_enabled
        self.pca_n_components = pca_n_components
        self.pca_whiten = pca_whiten
        self.pca_svd_solver = pca_svd_solver
        self.feature_selection_enabled = feature_selection_enabled
        self.selection_method = selection_method
        self.selection_k = selection_k
        self.selection_percentile = selection_percentile
        self.rfe_estimator = rfe_estimator
        self.rfe_n_features = rfe_n_features
        self.rfe_step = rfe_step
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted components
        self.variance_selector_ = None
        self.feature_selector_ = None
        self.pca_ = None
        self.feature_names_out_ = []
        self.removed_features_ = {
            'low_variance': [],
            'high_correlation': [],
            'feature_selection': []
        }
        self.explained_variance_ratio_ = None
        
    def _remove_low_variance_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with low variance."""
        if self.variance_threshold <= 0:
            return X, []
        
        self.variance_selector_ = VarianceThreshold(threshold=self.variance_threshold)
        self.variance_selector_.fit(X)
        
        # Get selected features
        selected_mask = self.variance_selector_.get_support()
        selected_features = X.columns[selected_mask].tolist()
        removed_features = X.columns[~selected_mask].tolist()
        
        if self.verbose and removed_features:
            logger.info(f"Removed {len(removed_features)} low variance features")
        
        return X[selected_features], removed_features
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features."""
        if len(X.columns) < 2 or self.correlation_threshold >= 1.0:
            return X, []
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove features with high correlation (keep first one in each pair)
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # Remove the feature with lower variance
            if X[feat1].var() < X[feat2].var():
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        
        selected_features = [col for col in X.columns if col not in features_to_remove]
        removed_features = list(features_to_remove)
        
        if self.verbose and removed_features:
            logger.info(f"Removed {len(removed_features)} highly correlated features")
        
        return X[selected_features], removed_features
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DimensionalityReducer':
        """
        Fit dimensionality reduction components.
        
        Args:
            X: Input features
            y: Target variable (required for feature selection)
            
        Returns:
            Self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_processed = X.copy()
        
        # Step 1: Remove low variance features
        X_processed, low_var_removed = self._remove_low_variance_features(X_processed)
        self.removed_features_['low_variance'] = low_var_removed
        
        # Step 2: Remove highly correlated features
        X_processed, corr_removed = self._remove_correlated_features(X_processed)
        self.removed_features_['high_correlation'] = corr_removed
        
        # Step 3: Feature selection (if enabled and target provided)
        if self.feature_selection_enabled and y is not None:
            self.feature_selector_ = FeatureSelector(
                method=self.selection_method,
                k=self.selection_k,
                percentile=self.selection_percentile,
                estimator=self.rfe_estimator,
                rfe_n_features=self.rfe_n_features,
                rfe_step=self.rfe_step,
                random_state=self.random_state,
                verbose=self.verbose
            )
            
            self.feature_selector_.fit(X_processed, y)
            X_selected = self.feature_selector_.transform(X_processed)
            
            # Track removed features
            all_features = set(X_processed.columns)
            selected_features = set(X_selected.columns)
            selection_removed = list(all_features - selected_features)
            self.removed_features_['feature_selection'] = selection_removed
            
            X_processed = X_selected
        
        # Step 4: PCA (if enabled)
        if self.pca_enabled:
            # Determine number of components
            if isinstance(self.pca_n_components, str) and self.pca_n_components == 'mle':
                n_components = 'mle'
            elif isinstance(self.pca_n_components, float) and 0 < self.pca_n_components < 1:
                n_components = self.pca_n_components
            elif isinstance(self.pca_n_components, int):
                n_components = min(self.pca_n_components, len(X_processed.columns))
            else:
                n_components = 0.95
                
            self.pca_ = PCA(
                n_components=n_components,
                whiten=self.pca_whiten,
                svd_solver=self.pca_svd_solver,
                random_state=self.random_state
            )
            
            # Fit PCA on processed features
            self.pca_.fit(X_processed)
            
            # Store explained variance ratio
            self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
            
            if self.verbose:
                logger.info(f"PCA reduced {len(X_processed.columns)} features to {self.pca_.n_components_} components")
                logger.info(f"Explained variance ratio: {self.explained_variance_ratio_[:5].sum():.3f} (first 5 components)")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted dimensionality reducers.
        
        Args:
            X: Input features
            
        Returns:
            Dimensionality reduced features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_processed = X.copy()
        
        # Step 1: Remove low variance features
        if self.variance_selector_ is not None:
            selected_columns = X.columns[self.variance_selector_.get_support()]
            X_processed = X_processed[selected_columns]
        
        # Step 2: Remove highly correlated features
        corr_removed = self.removed_features_['high_correlation']
        if corr_removed:
            remaining_columns = [col for col in X_processed.columns if col not in corr_removed]
            X_processed = X_processed[remaining_columns]
        
        # Step 3: Apply feature selection
        if self.feature_selector_ is not None:
            X_processed = self.feature_selector_.transform(X_processed)
        
        # Step 4: Apply PCA
        if self.pca_ is not None:
            X_pca = self.pca_.transform(X_processed)
            
            # Create component names
            component_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
            X_processed = pd.DataFrame(X_pca, columns=component_names, index=X.index)
        
        self.feature_names_out_ = list(X_processed.columns)
        return X_processed
    
    def get_feature_names_out(self, input_features: List[str] = None) -> List[str]:
        """Get output feature names."""
        return self.feature_names_out_
    
    def get_reduction_info(self) -> Dict[str, Any]:
        """Get information about dimensionality reduction."""
        info = {
            'original_features': None,  # Set during pipeline
            'final_features': len(self.feature_names_out_),
            'removed_features': {
                'low_variance': len(self.removed_features_['low_variance']),
                'high_correlation': len(self.removed_features_['high_correlation']),
                'feature_selection': len(self.removed_features_['feature_selection'])
            },
            'pca_enabled': self.pca_enabled,
            'pca_components': self.pca_.n_components_ if self.pca_ else None,
            'explained_variance_ratio': self.explained_variance_ratio_[:10].tolist() if self.explained_variance_ratio_ is not None else None
        }
        
        # Calculate total reduction
        total_removed = sum(info['removed_features'].values())
        if self.pca_enabled and self.pca_:
            # PCA creates new features, so we need to account for that differently
            info['reduction_summary'] = f"Reduced from original to {info['final_features']} features via PCA"
        else:
            info['reduction_summary'] = f"Removed {total_removed} features, keeping {info['final_features']}"
        
        return info
    
    def get_pca_components(self) -> Optional[pd.DataFrame]:
        """Get PCA component loadings if PCA was applied."""
        if self.pca_ is None:
            return None
        
        # Get feature names before PCA
        if hasattr(self, '_pre_pca_features'):
            feature_names = self._pre_pca_features
        else:
            feature_names = [f'feature_{i}' for i in range(self.pca_.components_.shape[1])]
        
        component_names = [f'PC{i+1}' for i in range(self.pca_.n_components_)]
        
        return pd.DataFrame(
            self.pca_.components_.T,
            columns=component_names,
            index=feature_names
        )
    
    def plot_explained_variance(self, ax=None):
        """Plot explained variance ratio (requires matplotlib)."""
        if self.explained_variance_ratio_ is None:
            warnings.warn("PCA was not applied, cannot plot explained variance")
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot individual explained variance
            ax.bar(range(1, len(self.explained_variance_ratio_) + 1), 
                   self.explained_variance_ratio_, 
                   alpha=0.7, label='Individual')
            
            # Plot cumulative explained variance
            cumsum = np.cumsum(self.explained_variance_ratio_)
            ax.plot(range(1, len(cumsum) + 1), cumsum, 
                   'ro-', alpha=0.7, label='Cumulative')
            
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA Explained Variance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return ax
            
        except ImportError:
            warnings.warn("matplotlib not available for plotting")
            return None