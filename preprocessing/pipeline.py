"""
Configurable Data Preprocessing Pipeline
Orchestrates all preprocessing steps in a cohesive, configurable pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
import json
import logging
from datetime import datetime

from .imputation import MissingValueImputer
from .outliers import OutlierDetector
from .normalization import DataNormalizer
from .duplicates import DuplicateHandler
from .quality import DataQualityScorer

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Configurable data preprocessing pipeline that orchestrates all preprocessing steps
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize preprocessing pipeline
        
        Args:
            config: Configuration dictionary for all preprocessing components
        """
        self.config = config or self._get_default_config()
        self._init_components()
        self.fitted = False
        self.processing_log = []
        self.quality_reports = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the pipeline"""
        return {
            'steps': ['quality_check', 'duplicates', 'imputation', 'outliers', 'normalization'],
            'imputation': {
                'strategies': {},  # Will be auto-detected
                'default_numeric': 'median',
                'default_categorical': 'mode'
            },
            'outliers': {
                'methods': {},  # Will be auto-detected  
                'handle_strategy': 'clip',
                'thresholds': {
                    'iqr': {'multiplier': 1.5},
                    'zscore': {'threshold': 3},
                    'isolation_forest': {'contamination': 0.1}
                }
            },
            'normalization': {
                'scaling_methods': {},  # Will be auto-detected
                'categorical_encoding': 'onehot'
            },
            'duplicates': {
                'strategy': 'exact',
                'threshold': 0.9,
                'keep': 'first'
            },
            'quality': {
                'min_quality_score': 0.7,
                'generate_report': True
            }
        }
    
    def _init_components(self):
        """Initialize preprocessing components based on config"""
        # Initialize imputation component
        imputation_config = self.config.get('imputation', {})
        self.imputer = MissingValueImputer(
            strategies=imputation_config.get('strategies', {})
        )
        
        # Initialize outlier detection component
        outliers_config = self.config.get('outliers', {})
        self.outlier_detector = OutlierDetector(
            methods=outliers_config.get('methods', {}),
            thresholds=outliers_config.get('thresholds', {}),
            handle_strategy=outliers_config.get('handle_strategy', 'clip')
        )
        
        # Initialize normalization component
        normalization_config = self.config.get('normalization', {})
        self.normalizer = DataNormalizer(
            scaling_methods=normalization_config.get('scaling_methods', {}),
            categorical_encoding=normalization_config.get('categorical_encoding', 'onehot')
        )
        
        # Initialize duplicate handler
        duplicates_config = self.config.get('duplicates', {})
        self.duplicate_handler = DuplicateHandler(
            strategy=duplicates_config.get('strategy', 'exact'),
            threshold=duplicates_config.get('threshold', 0.9),
            keep=duplicates_config.get('keep', 'first')
        )
        
        # Initialize quality scorer
        quality_config = self.config.get('quality', {})
        self.quality_scorer = DataQualityScorer()
    
    def fit(self, df: pd.DataFrame, target_column: str = None) -> 'PreprocessingPipeline':
        """
        Fit the preprocessing pipeline on the data
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            
        Returns:
            Self for method chaining
        """
        self.target_column = target_column
        self.original_columns = df.columns.tolist()
        self.processing_log = []
        
        # Log initial state
        self._log_step('fit_start', {
            'original_shape': df.shape,
            'original_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum()
        })
        
        # Fit components based on configured steps
        steps = self.config.get('steps', [])
        
        current_df = df.copy()
        
        if 'imputation' in steps:
            self.imputer.fit(current_df)
            self._log_step('imputation_fit', {'component': 'MissingValueImputer'})
        
        if 'outliers' in steps:
            self.outlier_detector.fit(current_df)
            self._log_step('outliers_fit', {'component': 'OutlierDetector'})
        
        if 'normalization' in steps:
            self.normalizer.fit(current_df, target_column)
            self._log_step('normalization_fit', {'component': 'DataNormalizer'})
        
        # Duplicate detection doesn't need fitting
        
        self.fitted = True
        self._log_step('fit_complete', {'fitted_components': len(steps)})
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations to the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        result_df = df.copy()
        steps = self.config.get('steps', [])
        
        self._log_step('transform_start', {
            'input_shape': result_df.shape,
            'steps_to_apply': steps
        })
        
        # Apply preprocessing steps in order
        for step in steps:
            if step == 'quality_check':
                result_df = self._apply_quality_check(result_df)
            elif step == 'duplicates':
                result_df = self._apply_duplicate_removal(result_df)
            elif step == 'imputation':
                result_df = self._apply_imputation(result_df)
            elif step == 'outliers':
                result_df = self._apply_outlier_handling(result_df)
            elif step == 'normalization':
                result_df = self._apply_normalization(result_df)
        
        self._log_step('transform_complete', {
            'output_shape': result_df.shape,
            'transformation_steps': len(steps)
        })
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """
        Fit and transform in one step
        
        Args:
            df: Input DataFrame  
            target_column: Target column name
            
        Returns:
            Preprocessed DataFrame
        """
        return self.fit(df, target_column).transform(df)
    
    def _apply_quality_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality assessment"""
        quality_config = self.config.get('quality', {})
        min_score = quality_config.get('min_quality_score', 0.7)
        
        # Assess quality
        quality_assessment = self.quality_scorer.assess_quality(df, self.target_column)
        overall_score = quality_assessment['overall_score']
        
        self._log_step('quality_check', {
            'quality_score': overall_score,
            'min_required': min_score,
            'passed': overall_score >= min_score
        })
        
        # Store quality report
        if quality_config.get('generate_report', True):
            report = self.quality_scorer.generate_quality_report(df, self.target_column)
            self.quality_reports.append({
                'timestamp': datetime.now().isoformat(),
                'score': overall_score,
                'report': report
            })
        
        # Warning if quality is below threshold
        if overall_score < min_score:
            logger.warning(f"Data quality score ({overall_score:.3f}) is below minimum threshold ({min_score})")
        
        return df
    
    def _apply_duplicate_removal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply duplicate detection and removal"""
        original_count = len(df)
        result_df, duplicate_info = self.duplicate_handler.detect_and_remove(df)
        removed_count = original_count - len(result_df)
        
        self._log_step('duplicates', {
            'original_count': original_count,
            'duplicates_found': duplicate_info['duplicate_count'],
            'duplicates_removed': removed_count,
            'final_count': len(result_df)
        })
        
        return result_df
    
    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value imputation"""
        missing_before = df.isnull().sum().sum()
        result_df = self.imputer.transform(df)
        missing_after = result_df.isnull().sum().sum()
        
        self._log_step('imputation', {
            'missing_before': missing_before,
            'missing_after': missing_after,
            'imputed_values': missing_before - missing_after
        })
        
        return result_df
    
    def _apply_outlier_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier detection and handling"""
        outliers = self.outlier_detector.detect(df)
        total_outliers = sum(mask.sum() for mask in outliers.values())
        result_df = self.outlier_detector.handle_outliers(df, outliers)
        
        self._log_step('outliers', {
            'outliers_detected': total_outliers,
            'handling_strategy': self.outlier_detector.handle_strategy,
            'affected_columns': len(outliers)
        })
        
        return result_df
    
    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data normalization"""
        original_columns = len(df.columns)
        result_df = self.normalizer.transform(df)
        final_columns = len(result_df.columns)
        
        self._log_step('normalization', {
            'original_columns': original_columns,
            'final_columns': final_columns,
            'encoding_applied': final_columns > original_columns
        })
        
        return result_df
    
    def _log_step(self, step_name: str, details: Dict[str, Any]):
        """Log a preprocessing step"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'details': details
        }
        self.processing_log.append(log_entry)
        logger.info(f"Pipeline step {step_name}: {details}")
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of preprocessing applied
        
        Returns:
            Dict with preprocessing summary
        """
        if not self.processing_log:
            return {'status': 'No preprocessing applied'}
        
        # Extract key metrics from log
        start_log = next((log for log in self.processing_log if log['step'] == 'transform_start'), None)
        end_log = next((log for log in self.processing_log if log['step'] == 'transform_complete'), None)
        
        summary = {
            'pipeline_config': self.config,
            'steps_applied': self.config.get('steps', []),
            'fitted': self.fitted,
            'processing_steps': len(self.processing_log),
        }
        
        if start_log and end_log:
            summary.update({
                'input_shape': start_log['details']['input_shape'],
                'output_shape': end_log['details']['output_shape'],
                'shape_change': {
                    'rows': end_log['details']['output_shape'][0] - start_log['details']['input_shape'][0],
                    'columns': end_log['details']['output_shape'][1] - start_log['details']['input_shape'][1]
                }
            })
        
        # Add step-specific summaries
        step_summaries = {}
        for log_entry in self.processing_log:
            step = log_entry['step']
            if step in ['quality_check', 'duplicates', 'imputation', 'outliers', 'normalization']:
                step_summaries[step] = log_entry['details']
        
        summary['step_summaries'] = step_summaries
        
        return summary
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after preprocessing
        
        Returns:
            List of feature names
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted to get feature names")
        
        # Get feature names from normalizer if normalization was applied
        if 'normalization' in self.config.get('steps', []):
            return self.normalizer.get_feature_names()
        else:
            return self.original_columns
    
    def save_config(self, filepath: str):
        """
        Save pipeline configuration to file
        
        Args:
            filepath: Path to save config file
        """
        config_to_save = {
            'pipeline_config': self.config,
            'fitted': self.fitted,
            'original_columns': getattr(self, 'original_columns', []),
            'target_column': getattr(self, 'target_column', None),
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        logger.info(f"Pipeline configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'PreprocessingPipeline':
        """
        Load pipeline from saved configuration
        
        Args:
            filepath: Path to config file
            
        Returns:
            PreprocessingPipeline instance
        """
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        pipeline = cls(config=config_data['pipeline_config'])
        pipeline.fitted = config_data.get('fitted', False)
        pipeline.original_columns = config_data.get('original_columns', [])
        pipeline.target_column = config_data.get('target_column', None)
        
        logger.info(f"Pipeline configuration loaded from {filepath}")
        
        return pipeline
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data against pipeline requirements
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check if expected columns are present
        if hasattr(self, 'original_columns'):
            missing_columns = set(self.original_columns) - set(df.columns)
            extra_columns = set(df.columns) - set(self.original_columns)
            
            if missing_columns:
                validation_results['valid'] = False
                validation_results['issues'].append(f"Missing columns: {missing_columns}")
            
            if extra_columns:
                validation_results['warnings'].append(f"Extra columns: {extra_columns}")
        
        # Check data types
        # This would be more comprehensive in a production system
        
        # Check for completely empty dataset
        if len(df) == 0:
            validation_results['valid'] = False
            validation_results['issues'].append("Dataset is empty")
        
        # Check for excessive missing values
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        
        if missing_ratio > 0.8:
            validation_results['warnings'].append(f"High missing value ratio: {missing_ratio:.2f}")
        
        return validation_results
    
    def get_processing_report(self) -> str:
        """
        Generate comprehensive processing report
        
        Returns:
            Formatted processing report string
        """
        if not self.processing_log:
            return "No preprocessing has been performed."
        
        report = []
        report.append("=" * 60)
        report.append("PREPROCESSING PIPELINE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Pipeline configuration
        report.append("Pipeline Configuration:")
        report.append(f"Steps: {', '.join(self.config.get('steps', []))}")
        report.append("")
        
        # Processing summary
        summary = self.get_preprocessing_summary()
        if 'input_shape' in summary and 'output_shape' in summary:
            report.append(f"Data Transformation:")
            report.append(f"Input Shape:  {summary['input_shape'][0]:,} rows × {summary['input_shape'][1]} columns")
            report.append(f"Output Shape: {summary['output_shape'][0]:,} rows × {summary['output_shape'][1]} columns")
            
            shape_change = summary['shape_change']
            if shape_change['rows'] != 0:
                report.append(f"Row Change:   {shape_change['rows']:+,}")
            if shape_change['columns'] != 0:
                report.append(f"Column Change: {shape_change['columns']:+}")
            report.append("")
        
        # Step-by-step details
        step_summaries = summary.get('step_summaries', {})
        if step_summaries:
            report.append("Step Details:")
            report.append("-" * 30)
            
            for step_name, details in step_summaries.items():
                report.append(f"\n{step_name.upper()}:")
                for key, value in details.items():
                    report.append(f"  {key}: {value}")
        
        # Quality reports
        if self.quality_reports:
            latest_quality = self.quality_reports[-1]
            report.append(f"\nData Quality Score: {latest_quality['score']:.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def reset(self):
        """Reset pipeline to unfitted state"""
        self.fitted = False
        self.processing_log = []
        self.quality_reports = []
        if hasattr(self, 'original_columns'):
            delattr(self, 'original_columns')
        if hasattr(self, 'target_column'):
            delattr(self, 'target_column')
        
        logger.info("Pipeline reset to unfitted state")