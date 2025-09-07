"""
Feature Engineering Pipeline for Loan Eligibility Predictor
Task ID: ML-002 - Feature Engineering Pipeline

This package contains comprehensive feature engineering components for production-ready
ML pipelines including categorical encoding, numerical preprocessing, feature interactions,
and dimensionality reduction.
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"

from .pipeline import FeatureEngineeringPipeline
from .encoders import CategoricalEncoder, TargetEncoder
from .preprocessors import NumericalPreprocessor, OutlierDetector
from .interactions import FeatureInteractionGenerator, MathematicalTransformer
from .dimensionality import DimensionalityReducer, FeatureSelector
from .config import (
    FeaturePipelineConfig,
    CategoricalEncodingConfig,
    NumericalPreprocessingConfig,
    FeatureInteractionConfig,
    DimensionalityReductionConfig,
    create_default_loan_config
)

__all__ = [
    # Main Pipeline
    "FeatureEngineeringPipeline",
    
    # Encoders
    "CategoricalEncoder", 
    "TargetEncoder",
    
    # Preprocessors
    "NumericalPreprocessor",
    "OutlierDetector",
    
    # Interactions
    "FeatureInteractionGenerator",
    "MathematicalTransformer",
    
    # Dimensionality
    "DimensionalityReducer",
    "FeatureSelector",
    
    # Configuration
    "FeaturePipelineConfig",
    "CategoricalEncodingConfig",
    "NumericalPreprocessingConfig", 
    "FeatureInteractionConfig",
    "DimensionalityReductionConfig",
    "create_default_loan_config"
]