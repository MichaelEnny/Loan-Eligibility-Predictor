"""
Configuration Management for Feature Engineering Pipeline
Provides flexible configuration system for all pipeline components.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import yaml
import json
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


@dataclass
class CategoricalEncodingConfig:
    """Configuration for categorical encoding operations."""
    
    # One-hot encoding settings
    onehot_features: List[str] = field(default_factory=list)
    onehot_drop_first: bool = True
    onehot_handle_unknown: str = "ignore"
    onehot_max_categories: Optional[int] = 50
    
    # Target encoding settings  
    target_features: List[str] = field(default_factory=list)
    target_smoothing: float = 1.0
    target_min_samples_leaf: int = 20
    target_cv_folds: int = 5
    
    # Label encoding settings
    label_features: List[str] = field(default_factory=list)
    label_handle_unknown: str = "use_encoded_value"
    label_unknown_value: int = -1
    
    # Ordinal encoding settings
    ordinal_features: Dict[str, List[str]] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.target_smoothing < 0:
            raise ValueError("target_smoothing must be non-negative")
        if self.target_min_samples_leaf < 1:
            raise ValueError("target_min_samples_leaf must be positive")
        if self.target_cv_folds < 2:
            raise ValueError("target_cv_folds must be at least 2")
        if self.onehot_max_categories is not None and self.onehot_max_categories < 1:
            raise ValueError("onehot_max_categories must be positive or None")


@dataclass
class NumericalPreprocessingConfig:
    """Configuration for numerical preprocessing operations."""
    
    # Scaling settings
    scaling_features: List[str] = field(default_factory=list)
    scaling_method: str = "standard"  # standard, minmax, robust, quantile
    quantile_range: tuple = (25.0, 75.0)
    quantile_n_quantiles: int = 1000
    
    # Normalization settings
    normalization_features: List[str] = field(default_factory=list)
    normalization_method: str = "yeo-johnson"  # yeo-johnson, box-cox, quantile
    
    # Binning settings
    binning_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Outlier handling
    outlier_features: List[str] = field(default_factory=list)
    outlier_method: str = "iqr"  # iqr, isolation_forest, local_outlier_factor
    outlier_threshold: float = 1.5
    outlier_action: str = "clip"  # clip, remove, transform
    
    # Missing value handling
    imputation_strategy: str = "median"  # mean, median, mode, constant
    imputation_constant: float = 0.0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        valid_scaling_methods = {"standard", "minmax", "robust", "quantile"}
        if self.scaling_method not in valid_scaling_methods:
            raise ValueError(f"scaling_method must be one of {valid_scaling_methods}")
            
        valid_normalization_methods = {"yeo-johnson", "box-cox", "quantile"}
        if self.normalization_method not in valid_normalization_methods:
            raise ValueError(f"normalization_method must be one of {valid_normalization_methods}")
            
        valid_outlier_methods = {"iqr", "isolation_forest", "local_outlier_factor"}
        if self.outlier_method not in valid_outlier_methods:
            raise ValueError(f"outlier_method must be one of {valid_outlier_methods}")
            
        valid_outlier_actions = {"clip", "remove", "transform"}
        if self.outlier_action not in valid_outlier_actions:
            raise ValueError(f"outlier_action must be one of {valid_outlier_actions}")
            
        if self.outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be positive")


@dataclass
class FeatureInteractionConfig:
    """Configuration for feature interaction generation."""
    
    # Polynomial features
    polynomial_features: List[str] = field(default_factory=list)
    polynomial_degree: int = 2
    polynomial_include_bias: bool = False
    polynomial_interaction_only: bool = True
    
    # Mathematical transformations
    math_features: List[str] = field(default_factory=list)
    math_operations: List[str] = field(default_factory=lambda: ["log", "sqrt", "square"])
    
    # Ratio features
    ratio_pairs: List[tuple] = field(default_factory=list)
    
    # Arithmetic combinations
    arithmetic_pairs: List[tuple] = field(default_factory=list)
    arithmetic_operations: List[str] = field(default_factory=lambda: ["+", "-", "*", "/"])
    
    # Domain-specific features
    domain_features: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.polynomial_degree < 1:
            raise ValueError("polynomial_degree must be positive")
        
        valid_math_ops = {"log", "sqrt", "square", "exp", "reciprocal"}
        invalid_ops = set(self.math_operations) - valid_math_ops
        if invalid_ops:
            raise ValueError(f"Invalid math operations: {invalid_ops}")
            
        valid_arithmetic_ops = {"+", "-", "*", "/"}
        invalid_arithmetic = set(self.arithmetic_operations) - valid_arithmetic_ops
        if invalid_arithmetic:
            raise ValueError(f"Invalid arithmetic operations: {invalid_arithmetic}")


@dataclass
class DimensionalityReductionConfig:
    """Configuration for dimensionality reduction techniques."""
    
    # PCA settings
    pca_enabled: bool = False
    pca_n_components: Union[int, float, str] = 0.95
    pca_whiten: bool = False
    pca_svd_solver: str = "auto"
    
    # Feature selection settings
    feature_selection_enabled: bool = True
    selection_method: str = "mutual_info"  # mutual_info, f_classif, chi2, recursive
    selection_k: Union[int, str] = "all"
    selection_percentile: float = 50.0
    
    # Recursive feature elimination
    rfe_estimator: str = "random_forest"
    rfe_n_features: Union[int, float] = 0.5
    rfe_step: Union[int, float] = 1
    
    # Variance threshold
    variance_threshold: float = 0.0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if isinstance(self.pca_n_components, int) and self.pca_n_components < 1:
            raise ValueError("pca_n_components must be positive when int")
        if isinstance(self.pca_n_components, float) and not (0 < self.pca_n_components <= 1):
            raise ValueError("pca_n_components must be between 0 and 1 when float")
            
        valid_selection_methods = {"mutual_info", "f_classif", "chi2", "recursive"}
        if self.selection_method not in valid_selection_methods:
            raise ValueError(f"selection_method must be one of {valid_selection_methods}")
            
        if isinstance(self.selection_k, int) and self.selection_k < 1:
            raise ValueError("selection_k must be positive when int")
            
        if not (0 < self.selection_percentile <= 100):
            raise ValueError("selection_percentile must be between 0 and 100")
            
        if self.variance_threshold < 0:
            raise ValueError("variance_threshold must be non-negative")


@dataclass
class FeaturePipelineConfig:
    """Main configuration class for the entire feature engineering pipeline."""
    
    # Sub-configurations
    categorical: CategoricalEncodingConfig = field(default_factory=CategoricalEncodingConfig)
    numerical: NumericalPreprocessingConfig = field(default_factory=NumericalPreprocessingConfig)
    interactions: FeatureInteractionConfig = field(default_factory=FeatureInteractionConfig)
    dimensionality: DimensionalityReductionConfig = field(default_factory=DimensionalityReductionConfig)
    
    # Global pipeline settings
    target_column: str = "loan_approved"
    feature_columns: List[str] = field(default_factory=list)
    exclude_columns: List[str] = field(default_factory=lambda: ["application_date"])
    
    # Pipeline execution settings
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = False
    
    # Data validation settings
    validate_input: bool = True
    handle_unknown_categories: bool = True
    
    def validate(self) -> None:
        """Validate the entire configuration."""
        # Validate sub-configurations
        self.categorical.validate()
        self.numerical.validate()
        self.interactions.validate()
        self.dimensionality.validate()
        
        # Validate global settings
        if not self.target_column:
            raise ValueError("target_column must be specified")
        
        if self.target_column in self.exclude_columns:
            raise ValueError("target_column cannot be in exclude_columns")
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'FeaturePipelineConfig':
        """Load configuration from file (YAML or JSON)."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            return cls.from_dict(config_dict)
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeaturePipelineConfig':
        """Create configuration from dictionary."""
        # Extract sub-configurations
        categorical_config = CategoricalEncodingConfig(**config_dict.get('categorical', {}))
        numerical_config = NumericalPreprocessingConfig(**config_dict.get('numerical', {}))
        interactions_config = FeatureInteractionConfig(**config_dict.get('interactions', {}))
        dimensionality_config = DimensionalityReductionConfig(**config_dict.get('dimensionality', {}))
        
        # Extract global settings
        global_config = {k: v for k, v in config_dict.items() 
                        if k not in ['categorical', 'numerical', 'interactions', 'dimensionality']}
        
        return cls(
            categorical=categorical_config,
            numerical=numerical_config,
            interactions=interactions_config,
            dimensionality=dimensionality_config,
            **global_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'categorical': self.categorical.__dict__,
            'numerical': self.numerical.__dict__,
            'interactions': self.interactions.__dict__,
            'dimensionality': self.dimensionality.__dict__,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'exclude_columns': self.exclude_columns,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'validate_input': self.validate_input,
            'handle_unknown_categories': self.handle_unknown_categories
        }
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        
        try:
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.to_dict(), f, default_flow_style=False)
                elif config_path.suffix.lower() == '.json':
                    json.dump(self.to_dict(), f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                    
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise


def create_default_loan_config() -> FeaturePipelineConfig:
    """Create default configuration optimized for loan eligibility prediction."""
    
    # Define categorical features based on the loan dataset
    categorical_config = CategoricalEncodingConfig(
        onehot_features=[
            'gender', 'marital_status', 'education', 'employment_status',
            'loan_purpose', 'state', 'area_type'
        ],
        target_features=['education', 'employment_status', 'loan_purpose'],
        label_features=['gender', 'marital_status', 'state', 'area_type'],
        ordinal_features={
            'education': ['High School', 'Some College', "Bachelor's", 'Advanced']
        }
    )
    
    # Define numerical preprocessing
    numerical_config = NumericalPreprocessingConfig(
        scaling_features=[
            'age', 'years_employed', 'annual_income', 'monthly_income',
            'credit_score', 'credit_history_length', 'num_credit_accounts',
            'existing_debt', 'monthly_debt_payments', 'loan_amount',
            'property_value', 'years_with_bank', 'coapplicant_income',
            'total_household_income'
        ],
        outlier_features=[
            'annual_income', 'existing_debt', 'loan_amount',
            'property_value', 'coapplicant_income'
        ],
        scaling_method='robust',
        outlier_method='iqr',
        outlier_threshold=2.0,
        outlier_action='clip'
    )
    
    # Define feature interactions
    interactions_config = FeatureInteractionConfig(
        polynomial_features=['debt_to_income_ratio', 'credit_score'],
        ratio_pairs=[
            ('loan_amount', 'annual_income'),
            ('monthly_debt_payments', 'monthly_income'),
            ('existing_debt', 'annual_income'),
            ('property_value', 'loan_amount')
        ],
        arithmetic_pairs=[
            ('annual_income', 'coapplicant_income'),
            ('credit_score', 'credit_history_length')
        ],
        domain_features={
            'debt_burden_score': {
                'formula': 'debt_to_income_ratio * loan_amount / annual_income',
                'description': 'Combined debt burden indicator'
            },
            'creditworthiness_score': {
                'formula': 'credit_score * credit_history_length / 100',
                'description': 'Overall creditworthiness indicator'
            }
        }
    )
    
    # Define dimensionality reduction
    dimensionality_config = DimensionalityReductionConfig(
        feature_selection_enabled=True,
        selection_method='mutual_info',
        selection_k=25,
        variance_threshold=0.01,
        pca_enabled=False  # Keep interpretability for loan decisions
    )
    
    return FeaturePipelineConfig(
        categorical=categorical_config,
        numerical=numerical_config,
        interactions=interactions_config,
        dimensionality=dimensionality_config,
        target_column='loan_approved',
        exclude_columns=['application_date'],
        n_jobs=-1,
        random_state=42,
        verbose=True
    )