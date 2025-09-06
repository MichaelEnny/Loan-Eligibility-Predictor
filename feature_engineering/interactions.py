"""
Feature Interaction Generation Components
Creates polynomial features, mathematical transformations, ratios, and domain-specific interactions.
"""

from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
import warnings
import logging

logger = logging.getLogger(__name__)


class MathematicalTransformer(BaseEstimator, TransformerMixin):
    """
    Apply mathematical transformations to numerical features.
    
    Supports log, sqrt, square, exponential, reciprocal, and custom transformations
    with intelligent handling of edge cases and invalid operations.
    """
    
    def __init__(self,
                 features: List[str] = None,
                 operations: List[str] = None,
                 custom_transformations: Dict[str, Callable] = None,
                 handle_negatives: str = 'shift',
                 handle_zeros: str = 'small_constant',
                 small_constant: float = 1e-6):
        """
        Initialize mathematical transformer.
        
        Args:
            features: Features to transform
            operations: Mathematical operations to apply
            custom_transformations: Custom transformation functions
            handle_negatives: How to handle negative values ('shift', 'abs', 'skip')
            handle_zeros: How to handle zero values ('small_constant', 'skip')
            small_constant: Small constant for numerical stability
        """
        self.features = features or []
        self.operations = operations or ['log', 'sqrt', 'square']
        self.custom_transformations = custom_transformations or {}
        self.handle_negatives = handle_negatives
        self.handle_zeros = handle_zeros
        self.small_constant = small_constant
        
        # Fitted parameters
        self.shift_values_ = {}
        self.feature_names_out_ = []
        
    def _safe_log(self, x: pd.Series, feature_name: str) -> pd.Series:
        """Apply safe logarithm transformation."""
        x_safe = x.copy()
        
        # Handle zeros
        if self.handle_zeros == 'small_constant':
            x_safe = x_safe.replace(0, self.small_constant)
        elif self.handle_zeros == 'skip':
            if (x_safe == 0).any():
                warnings.warn(f"Skipping log transformation for {feature_name} due to zero values")
                return x_safe
        
        # Handle negatives
        if (x_safe < 0).any():
            if self.handle_negatives == 'shift':
                shift_val = self.shift_values_.get(feature_name, abs(x_safe.min()) + 1)
                x_safe = x_safe + shift_val
            elif self.handle_negatives == 'abs':
                x_safe = x_safe.abs()
            elif self.handle_negatives == 'skip':
                warnings.warn(f"Skipping log transformation for {feature_name} due to negative values")
                return x
        
        return np.log1p(x_safe)
    
    def _safe_sqrt(self, x: pd.Series, feature_name: str) -> pd.Series:
        """Apply safe square root transformation."""
        if (x < 0).any():
            if self.handle_negatives == 'shift':
                shift_val = self.shift_values_.get(feature_name, abs(x.min()) + 1)
                x_shifted = x + shift_val
                return np.sqrt(x_shifted)
            elif self.handle_negatives == 'abs':
                return np.sqrt(x.abs())
            elif self.handle_negatives == 'skip':
                warnings.warn(f"Skipping sqrt transformation for {feature_name} due to negative values")
                return x
        
        return np.sqrt(x)
    
    def _safe_reciprocal(self, x: pd.Series, feature_name: str) -> pd.Series:
        """Apply safe reciprocal transformation."""
        x_safe = x.copy()
        
        # Handle zeros
        if (x_safe == 0).any():
            if self.handle_zeros == 'small_constant':
                x_safe = x_safe.replace(0, self.small_constant)
            elif self.handle_zeros == 'skip':
                warnings.warn(f"Skipping reciprocal transformation for {feature_name} due to zero values")
                return x_safe
        
        return 1.0 / x_safe
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'MathematicalTransformer':
        """
        Fit transformer by calculating shift values for handling negatives.
        
        Args:
            X: Input features
            y: Target variable (not used)
            
        Returns:
            Self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Calculate shift values for features with negative values
        for feature in self.features:
            if feature in X.columns:
                if (X[feature] < 0).any():
                    self.shift_values_[feature] = abs(X[feature].min()) + 1
                    
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mathematical transformations to features.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features with new mathematical features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        self.feature_names_out_ = list(X.columns)
        
        for feature in self.features:
            if feature not in X.columns:
                warnings.warn(f"Feature {feature} not found in input data")
                continue
                
            for operation in self.operations:
                try:
                    if operation == 'log':
                        transformed = self._safe_log(X[feature], feature)
                        new_feature_name = f"{feature}_log"
                    elif operation == 'sqrt':
                        transformed = self._safe_sqrt(X[feature], feature)
                        new_feature_name = f"{feature}_sqrt"
                    elif operation == 'square':
                        transformed = X[feature] ** 2
                        new_feature_name = f"{feature}_square"
                    elif operation == 'exp':
                        # Clip extreme values to prevent overflow
                        clipped_values = X[feature].clip(-50, 50)
                        transformed = np.exp(clipped_values)
                        new_feature_name = f"{feature}_exp"
                    elif operation == 'reciprocal':
                        transformed = self._safe_reciprocal(X[feature], feature)
                        new_feature_name = f"{feature}_reciprocal"
                    elif operation in self.custom_transformations:
                        transformed = self.custom_transformations[operation](X[feature])
                        new_feature_name = f"{feature}_{operation}"
                    else:
                        warnings.warn(f"Unknown operation: {operation}")
                        continue
                    
                    X_transformed[new_feature_name] = transformed
                    self.feature_names_out_.append(new_feature_name)
                    
                except Exception as e:
                    warnings.warn(f"Error applying {operation} to {feature}: {str(e)}")
                    
        return X_transformed


class FeatureInteractionGenerator(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature interaction generator.
    
    Creates polynomial features, ratios, arithmetic combinations, and domain-specific
    interactions with intelligent feature selection and validation.
    """
    
    def __init__(self,
                 polynomial_features: List[str] = None,
                 polynomial_degree: int = 2,
                 polynomial_include_bias: bool = False,
                 polynomial_interaction_only: bool = True,
                 math_features: List[str] = None,
                 math_operations: List[str] = None,
                 ratio_pairs: List[tuple] = None,
                 arithmetic_pairs: List[tuple] = None,
                 arithmetic_operations: List[str] = None,
                 domain_features: Dict[str, Any] = None,
                 max_features: int = None,
                 correlation_threshold: float = 0.95,
                 variance_threshold: float = 0.01,
                 random_state: int = 42,
                 verbose: bool = False):
        """
        Initialize feature interaction generator.
        
        Args:
            polynomial_features: Features for polynomial expansion
            polynomial_degree: Degree of polynomial features
            polynomial_include_bias: Include bias term in polynomial features
            polynomial_interaction_only: Only create interaction terms
            math_features: Features for mathematical transformations
            math_operations: Mathematical operations to apply
            ratio_pairs: Pairs of features for ratio creation
            arithmetic_pairs: Pairs of features for arithmetic combinations
            arithmetic_operations: Arithmetic operations to apply
            domain_features: Domain-specific feature definitions
            max_features: Maximum number of features to create
            correlation_threshold: Threshold for removing highly correlated features
            variance_threshold: Threshold for removing low variance features
            random_state: Random state for reproducibility
            verbose: Whether to print processing information
        """
        self.polynomial_features = polynomial_features or []
        self.polynomial_degree = polynomial_degree
        self.polynomial_include_bias = polynomial_include_bias
        self.polynomial_interaction_only = polynomial_interaction_only
        self.math_features = math_features or []
        self.math_operations = math_operations or ['log', 'sqrt', 'square']
        self.ratio_pairs = ratio_pairs or []
        self.arithmetic_pairs = arithmetic_pairs or []
        self.arithmetic_operations = arithmetic_operations or ['+', '-', '*', '/']
        self.domain_features = domain_features or {}
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted components
        self.polynomial_transformer_ = None
        self.math_transformer_ = None
        self.feature_names_out_ = []
        self.selected_features_ = []
        self.feature_importance_ = {}
        
    def _create_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features from specified pairs."""
        ratio_features = pd.DataFrame(index=X.index)
        
        for numerator, denominator in self.ratio_pairs:
            if numerator in X.columns and denominator in X.columns:
                # Handle division by zero
                denom_safe = X[denominator].replace(0, np.nan)
                ratio = X[numerator] / denom_safe
                
                # Fill NaN with median ratio
                if ratio.isna().any():
                    ratio_median = ratio.median()
                    ratio = ratio.fillna(ratio_median)
                
                feature_name = f"{numerator}_to_{denominator}_ratio"
                ratio_features[feature_name] = ratio
                
                if self.verbose:
                    logger.info(f"Created ratio feature: {feature_name}")
        
        return ratio_features
    
    def _create_arithmetic_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create arithmetic combination features."""
        arithmetic_features = pd.DataFrame(index=X.index)
        
        for feature1, feature2 in self.arithmetic_pairs:
            if feature1 in X.columns and feature2 in X.columns:
                for operation in self.arithmetic_operations:
                    try:
                        if operation == '+':
                            result = X[feature1] + X[feature2]
                            op_name = 'plus'
                        elif operation == '-':
                            result = X[feature1] - X[feature2]
                            op_name = 'minus'
                        elif operation == '*':
                            result = X[feature1] * X[feature2]
                            op_name = 'times'
                        elif operation == '/':
                            # Handle division by zero
                            denom_safe = X[feature2].replace(0, np.nan)
                            result = X[feature1] / denom_safe
                            result = result.fillna(result.median())
                            op_name = 'div'
                        else:
                            continue
                        
                        feature_name = f"{feature1}_{op_name}_{feature2}"
                        arithmetic_features[feature_name] = result
                        
                        if self.verbose:
                            logger.info(f"Created arithmetic feature: {feature_name}")
                            
                    except Exception as e:
                        warnings.warn(f"Error creating {operation} feature for {feature1}, {feature2}: {str(e)}")
        
        return arithmetic_features
    
    def _create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for loan eligibility."""
        domain_features = pd.DataFrame(index=X.index)
        
        # Default loan-specific domain features if none specified
        if not self.domain_features:
            self.domain_features = self._get_default_loan_features()
        
        for feature_name, config in self.domain_features.items():
            try:
                if isinstance(config, dict) and 'formula' in config:
                    # Parse and evaluate formula
                    formula = config['formula']
                    
                    # Simple formula evaluation (extend as needed)
                    if 'debt_to_income_ratio * loan_amount / annual_income' in formula:
                        if all(col in X.columns for col in ['debt_to_income_ratio', 'loan_amount', 'annual_income']):
                            result = (X['debt_to_income_ratio'] * X['loan_amount'] / 
                                    X['annual_income'].replace(0, X['annual_income'].median()))
                            domain_features[feature_name] = result
                    
                    elif 'credit_score * credit_history_length / 100' in formula:
                        if all(col in X.columns for col in ['credit_score', 'credit_history_length']):
                            result = (X['credit_score'] * X['credit_history_length'] / 100)
                            domain_features[feature_name] = result
                    
                    elif 'loan_amount / total_household_income' in formula:
                        if all(col in X.columns for col in ['loan_amount', 'total_household_income']):
                            result = (X['loan_amount'] / 
                                    X['total_household_income'].replace(0, X['total_household_income'].median()))
                            domain_features[feature_name] = result
                    
                    elif 'existing_debt + loan_amount' in formula:
                        if all(col in X.columns for col in ['existing_debt', 'loan_amount']):
                            result = X['existing_debt'] + X['loan_amount']
                            domain_features[feature_name] = result
                    
                    elif 'property_value / loan_amount' in formula:
                        if all(col in X.columns for col in ['property_value', 'loan_amount']):
                            result = (X['property_value'] / 
                                    X['loan_amount'].replace(0, X['loan_amount'].median()))
                            domain_features[feature_name] = result
                    
                    if self.verbose and feature_name in domain_features.columns:
                        logger.info(f"Created domain feature: {feature_name}")
                        
            except Exception as e:
                warnings.warn(f"Error creating domain feature {feature_name}: {str(e)}")
        
        return domain_features
    
    def _get_default_loan_features(self) -> Dict[str, Any]:
        """Get default domain-specific features for loan eligibility."""
        return {
            'debt_burden_score': {
                'formula': 'debt_to_income_ratio * loan_amount / annual_income',
                'description': 'Combined debt burden indicator'
            },
            'creditworthiness_score': {
                'formula': 'credit_score * credit_history_length / 100',
                'description': 'Overall creditworthiness indicator'
            },
            'loan_to_income_ratio': {
                'formula': 'loan_amount / total_household_income',
                'description': 'Loan amount relative to household income'
            },
            'total_debt_after_loan': {
                'formula': 'existing_debt + loan_amount',
                'description': 'Total debt if loan is approved'
            },
            'collateral_ratio': {
                'formula': 'property_value / loan_amount',
                'description': 'Property value as collateral ratio'
            }
        }
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> List[str]:
        """Remove highly correlated features."""
        if len(X.columns) < 2:
            return list(X.columns)
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs with high correlation
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove features with high correlation (keep first one in each pair)
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            features_to_remove.add(feat2)
        
        selected_features = [col for col in X.columns if col not in features_to_remove]
        
        if self.verbose and features_to_remove:
            logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        
        return selected_features
    
    def _remove_low_variance_features(self, X: pd.DataFrame) -> List[str]:
        """Remove low variance features."""
        feature_variances = X.var()
        high_variance_features = feature_variances[feature_variances > self.variance_threshold].index.tolist()
        
        removed_count = len(X.columns) - len(high_variance_features)
        if self.verbose and removed_count > 0:
            logger.info(f"Removed {removed_count} low variance features")
        
        return high_variance_features
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeatureInteractionGenerator':
        """
        Fit interaction generators.
        
        Args:
            X: Input features
            y: Target variable (not used directly)
            
        Returns:
            Self for method chaining
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Fit polynomial transformer
        if self.polynomial_features:
            poly_cols = [col for col in self.polynomial_features if col in X.columns]
            if poly_cols:
                self.polynomial_transformer_ = PolynomialFeatures(
                    degree=self.polynomial_degree,
                    include_bias=self.polynomial_include_bias,
                    interaction_only=self.polynomial_interaction_only
                )
                self.polynomial_transformer_.fit(X[poly_cols])
        
        # Fit mathematical transformer
        if self.math_features:
            math_cols = [col for col in self.math_features if col in X.columns]
            if math_cols:
                self.math_transformer_ = MathematicalTransformer(
                    features=math_cols,
                    operations=self.math_operations
                )
                self.math_transformer_.fit(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate feature interactions.
        
        Args:
            X: Input features
            
        Returns:
            Features with generated interactions
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Start with original features
        result_dfs = [X.copy()]
        
        # Add polynomial features
        if self.polynomial_transformer_ is not None:
            poly_cols = [col for col in self.polynomial_features if col in X.columns]
            if poly_cols:
                poly_features = self.polynomial_transformer_.transform(X[poly_cols])
                poly_names = self.polynomial_transformer_.get_feature_names_out(poly_cols)
                poly_df = pd.DataFrame(poly_features, columns=poly_names, index=X.index)
                
                # Remove original features to avoid duplication
                original_cols = set(poly_cols)
                poly_df = poly_df[[col for col in poly_df.columns if col not in original_cols]]
                
                if not poly_df.empty:
                    result_dfs.append(poly_df)
        
        # Add mathematical transformations
        if self.math_transformer_ is not None:
            math_features = self.math_transformer_.transform(X)
            # Only keep the new mathematical features
            original_cols = set(X.columns)
            math_only = math_features[[col for col in math_features.columns if col not in original_cols]]
            if not math_only.empty:
                result_dfs.append(math_only)
        
        # Add ratio features
        if self.ratio_pairs:
            ratio_features = self._create_ratio_features(X)
            if not ratio_features.empty:
                result_dfs.append(ratio_features)
        
        # Add arithmetic features
        if self.arithmetic_pairs:
            arithmetic_features = self._create_arithmetic_features(X)
            if not arithmetic_features.empty:
                result_dfs.append(arithmetic_features)
        
        # Add domain-specific features
        domain_features = self._create_domain_features(X)
        if not domain_features.empty:
            result_dfs.append(domain_features)
        
        # Combine all features
        X_with_interactions = pd.concat(result_dfs, axis=1)
        
        # Feature selection
        selected_features = list(X_with_interactions.columns)
        
        # Remove low variance features
        selected_features = [col for col in selected_features 
                           if col in self._remove_low_variance_features(X_with_interactions[selected_features])]
        
        # Remove highly correlated features
        if len(selected_features) > 1:
            selected_features = self._remove_correlated_features(X_with_interactions[selected_features])
        
        # Limit number of features if specified
        if self.max_features and len(selected_features) > self.max_features:
            # Keep original features plus top interaction features by variance
            original_features = [col for col in selected_features if col in X.columns]
            interaction_features = [col for col in selected_features if col not in X.columns]
            
            # Sort interaction features by variance
            if interaction_features:
                feature_variances = X_with_interactions[interaction_features].var().sort_values(ascending=False)
                top_interactions = feature_variances.head(self.max_features - len(original_features)).index.tolist()
                selected_features = original_features + top_interactions
        
        self.selected_features_ = selected_features
        self.feature_names_out_ = selected_features
        
        if self.verbose:
            logger.info(f"Generated {len(selected_features) - len(X.columns)} interaction features")
            logger.info(f"Total features: {len(selected_features)}")
        
        return X_with_interactions[selected_features]
    
    def get_feature_names_out(self, input_features: List[str] = None) -> List[str]:
        """Get output feature names."""
        return self.feature_names_out_
    
    def get_interaction_info(self) -> Dict[str, Any]:
        """Get information about generated interactions."""
        info = {
            'polynomial_features': self.polynomial_features,
            'math_features': self.math_features,
            'ratio_pairs': self.ratio_pairs,
            'arithmetic_pairs': self.arithmetic_pairs,
            'domain_features': list(self.domain_features.keys()),
            'selected_features': len(self.selected_features_),
            'total_features': len(self.feature_names_out_)
        }
        return info