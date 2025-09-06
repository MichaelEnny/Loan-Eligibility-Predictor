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
from .encoders import CategoricalEncoder
from .preprocessors import NumericalPreprocessor
from .interactions import FeatureInteractionGenerator
from .dimensionality import DimensionalityReducer
from .config import FeaturePipelineConfig

__all__ = [
    "FeatureEngineeringPipeline",
    "CategoricalEncoder", 
    "NumericalPreprocessor",
    "FeatureInteractionGenerator",
    "DimensionalityReducer",
    "FeaturePipelineConfig"
]