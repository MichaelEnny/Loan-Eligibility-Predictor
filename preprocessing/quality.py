"""
Data Quality Scoring and Assessment
Implements comprehensive data quality metrics and scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

class DataQualityScorer:
    """
    Comprehensive data quality assessment and scoring
    """
    
    def __init__(self, 
                 weights: Dict[str, float] = None,
                 thresholds: Dict[str, Dict[str, float]] = None):
        """
        Initialize data quality scorer
        
        Args:
            weights: Weights for different quality dimensions
            thresholds: Thresholds for quality scoring
        """
        self.weights = weights or {
            'completeness': 0.25,
            'validity': 0.20,
            'consistency': 0.20,
            'uniqueness': 0.15,
            'accuracy': 0.10,
            'timeliness': 0.10
        }
        
        self.thresholds = thresholds or {
            'completeness': {'excellent': 0.95, 'good': 0.85, 'poor': 0.70},
            'validity': {'excellent': 0.95, 'good': 0.85, 'poor': 0.70},
            'consistency': {'excellent': 0.95, 'good': 0.85, 'poor': 0.70},
            'uniqueness': {'excellent': 0.95, 'good': 0.85, 'poor': 0.70}
        }
        
        self.quality_dimensions = [
            'completeness', 'validity', 'consistency', 
            'uniqueness', 'accuracy', 'timeliness'
        ]
    
    def assess_quality(self, df: pd.DataFrame, 
                      target_column: str = None,
                      business_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment
        
        Args:
            df: Input DataFrame
            target_column: Target column for analysis
            business_rules: Dict of business rules for validation
            
        Returns:
            Dict with quality assessment results
        """
        assessment = {}
        
        # Calculate individual quality dimensions
        assessment['completeness'] = self._assess_completeness(df)
        assessment['validity'] = self._assess_validity(df, business_rules)
        assessment['consistency'] = self._assess_consistency(df)
        assessment['uniqueness'] = self._assess_uniqueness(df)
        assessment['accuracy'] = self._assess_accuracy(df, target_column)
        assessment['timeliness'] = self._assess_timeliness(df)
        
        # Calculate overall quality score
        assessment['overall_score'] = self._calculate_overall_score(assessment)
        
        # Generate quality summary
        assessment['summary'] = self._generate_quality_summary(assessment)
        
        # Quality recommendations
        assessment['recommendations'] = self._generate_recommendations(assessment)
        
        return assessment
    
    def get_quality_score(self, df: pd.DataFrame, 
                         target_column: str = None,
                         business_rules: Dict[str, Any] = None) -> float:
        """
        Get overall quality score (0-1)
        
        Args:
            df: Input DataFrame
            target_column: Target column for analysis
            business_rules: Dict of business rules for validation
            
        Returns:
            Overall quality score
        """
        assessment = self.assess_quality(df, target_column, business_rules)
        return assessment['overall_score']
    
    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = (total_cells - missing_cells) / total_cells
        
        # Column-level completeness
        column_completeness = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            completeness = (len(df) - missing_count) / len(df)
            column_completeness[column] = {
                'completeness_ratio': completeness,
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(df)) * 100
            }
        
        # Identify critical gaps
        critical_gaps = []
        for column, stats in column_completeness.items():
            if stats['completeness_ratio'] < self.thresholds['completeness']['poor']:
                critical_gaps.append({
                    'column': column,
                    'completeness': stats['completeness_ratio'],
                    'missing_percentage': stats['missing_percentage']
                })
        
        return {
            'score': completeness_ratio,
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'column_completeness': column_completeness,
            'critical_gaps': critical_gaps,
            'quality_level': self._get_quality_level(completeness_ratio, 'completeness')
        }
    
    def _assess_validity(self, df: pd.DataFrame, 
                        business_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess data validity"""
        validity_issues = []
        column_validity = {}
        
        for column in df.columns:
            valid_count = 0
            total_count = df[column].notna().sum()
            
            if total_count == 0:
                column_validity[column] = {
                    'validity_ratio': 0.0,
                    'issues': ['All values missing']
                }
                continue
            
            issues = []
            
            # Check data types
            if df[column].dtype == 'object':
                # Check for obviously invalid string patterns
                if column.lower() in ['email']:
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    invalid_emails = ~df[column].str.match(email_pattern, na=False)
                    invalid_count = invalid_emails.sum()
                    if invalid_count > 0:
                        issues.append(f'{invalid_count} invalid email formats')
                        valid_count = total_count - invalid_count
                    else:
                        valid_count = total_count
                
                elif column.lower() in ['phone', 'phone_number']:
                    # Basic phone validation
                    phone_pattern = r'^\+?[\d\s\-\(\)]{7,}$'
                    invalid_phones = ~df[column].str.match(phone_pattern, na=False)
                    invalid_count = invalid_phones.sum()
                    if invalid_count > 0:
                        issues.append(f'{invalid_count} invalid phone formats')
                        valid_count = total_count - invalid_count
                    else:
                        valid_count = total_count
                else:
                    valid_count = total_count
            
            elif df[column].dtype in ['int64', 'float64']:
                # Check for impossible values
                if column.lower() in ['age']:
                    invalid_age = (df[column] < 0) | (df[column] > 150)
                    invalid_count = invalid_age.sum()
                    if invalid_count > 0:
                        issues.append(f'{invalid_count} impossible age values')
                        valid_count = total_count - invalid_count
                    else:
                        valid_count = total_count
                
                elif column.lower() in ['salary', 'income', 'amount']:
                    invalid_amount = df[column] < 0
                    invalid_count = invalid_amount.sum()
                    if invalid_count > 0:
                        issues.append(f'{invalid_count} negative monetary values')
                        valid_count = total_count - invalid_count
                    else:
                        valid_count = total_count
                else:
                    valid_count = total_count
            else:
                valid_count = total_count
            
            # Apply business rules if provided
            if business_rules and column in business_rules:
                rule = business_rules[column]
                if 'min' in rule:
                    invalid_min = df[column] < rule['min']
                    invalid_count = invalid_min.sum()
                    if invalid_count > 0:
                        issues.append(f'{invalid_count} values below minimum ({rule["min"]})')
                        valid_count -= invalid_count
                
                if 'max' in rule:
                    invalid_max = df[column] > rule['max']
                    invalid_count = invalid_max.sum()
                    if invalid_count > 0:
                        issues.append(f'{invalid_count} values above maximum ({rule["max"]})')
                        valid_count -= invalid_count
            
            validity_ratio = valid_count / total_count if total_count > 0 else 0
            column_validity[column] = {
                'validity_ratio': validity_ratio,
                'valid_count': valid_count,
                'total_count': total_count,
                'issues': issues
            }
            
            if validity_ratio < self.thresholds['validity']['poor']:
                validity_issues.extend([{
                    'column': column,
                    'issue': issue,
                    'validity_ratio': validity_ratio
                } for issue in issues])
        
        # Overall validity score
        total_valid = sum(stats['valid_count'] for stats in column_validity.values())
        total_values = sum(stats['total_count'] for stats in column_validity.values())
        overall_validity = total_valid / total_values if total_values > 0 else 0
        
        return {
            'score': overall_validity,
            'column_validity': column_validity,
            'validity_issues': validity_issues,
            'quality_level': self._get_quality_level(overall_validity, 'validity')
        }
    
    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency"""
        consistency_issues = []
        column_consistency = {}
        
        for column in df.columns:
            issues = []
            consistency_score = 1.0
            
            if df[column].dtype == 'object':
                # Check for inconsistent formatting
                non_null_values = df[column].dropna()
                if len(non_null_values) > 0:
                    # Check case consistency
                    if len(non_null_values.str.lower().unique()) < len(non_null_values.unique()):
                        case_issues = len(non_null_values.unique()) - len(non_null_values.str.lower().unique())
                        issues.append(f'{case_issues} case inconsistencies')
                        consistency_score -= 0.1
                    
                    # Check whitespace consistency
                    stripped_values = non_null_values.str.strip()
                    if not non_null_values.equals(stripped_values):
                        whitespace_issues = (non_null_values != stripped_values).sum()
                        issues.append(f'{whitespace_issues} whitespace inconsistencies')
                        consistency_score -= 0.1
            
            elif df[column].dtype in ['int64', 'float64']:
                # Check for precision consistency
                non_null_values = df[column].dropna()
                if len(non_null_values) > 0 and df[column].dtype == 'float64':
                    decimal_places = non_null_values.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0)
                    if decimal_places.nunique() > 3:  # Too many different precisions
                        issues.append(f'Inconsistent decimal precision ({decimal_places.nunique()} different precisions)')
                        consistency_score -= 0.1
            
            column_consistency[column] = {
                'consistency_score': max(0, consistency_score),
                'issues': issues
            }
            
            if consistency_score < self.thresholds['consistency']['poor']:
                consistency_issues.extend([{
                    'column': column,
                    'issue': issue,
                    'consistency_score': consistency_score
                } for issue in issues])
        
        # Overall consistency score
        overall_consistency = np.mean([stats['consistency_score'] for stats in column_consistency.values()])
        
        return {
            'score': overall_consistency,
            'column_consistency': column_consistency,
            'consistency_issues': consistency_issues,
            'quality_level': self._get_quality_level(overall_consistency, 'consistency')
        }
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness"""
        total_records = len(df)
        unique_records = len(df.drop_duplicates())
        uniqueness_ratio = unique_records / total_records if total_records > 0 else 1.0
        
        # Column-level uniqueness
        column_uniqueness = {}
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            unique_count = df[column].nunique()
            uniqueness = unique_count / non_null_count if non_null_count > 0 else 0
            
            column_uniqueness[column] = {
                'uniqueness_ratio': uniqueness,
                'unique_count': unique_count,
                'total_count': non_null_count,
                'duplicate_count': non_null_count - unique_count
            }
        
        # Identify potential key columns (high uniqueness)
        potential_keys = []
        for column, stats in column_uniqueness.items():
            if stats['uniqueness_ratio'] > 0.95 and stats['total_count'] > 10:
                potential_keys.append({
                    'column': column,
                    'uniqueness': stats['uniqueness_ratio']
                })
        
        return {
            'score': uniqueness_ratio,
            'total_records': total_records,
            'unique_records': unique_records,
            'duplicate_records': total_records - unique_records,
            'column_uniqueness': column_uniqueness,
            'potential_keys': potential_keys,
            'quality_level': self._get_quality_level(uniqueness_ratio, 'uniqueness')
        }
    
    def _assess_accuracy(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Assess data accuracy (limited without external reference)"""
        accuracy_indicators = []
        accuracy_score = 0.8  # Default assumption
        
        # Check for obvious accuracy issues
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # Check for statistically impossible values
                if len(df[column].dropna()) > 10:
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((df[column] < q1 - 3 * iqr) | (df[column] > q3 + 3 * iqr)).sum()
                    outlier_ratio = outliers / len(df[column].dropna())
                    
                    if outlier_ratio > 0.1:  # More than 10% extreme outliers
                        accuracy_indicators.append({
                            'column': column,
                            'issue': 'High number of extreme outliers',
                            'outlier_ratio': outlier_ratio
                        })
                        accuracy_score -= 0.1
        
        # Check for logical inconsistencies between columns
        if target_column and target_column in df.columns:
            # This would be domain-specific logic
            # For loan data, we might check if loan amount is consistent with income
            pass
        
        return {
            'score': max(0, accuracy_score),
            'accuracy_indicators': accuracy_indicators,
            'quality_level': self._get_quality_level(accuracy_score, 'accuracy')
        }
    
    def _assess_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data timeliness"""
        timeliness_score = 0.8  # Default assumption
        timeliness_indicators = []
        
        # Look for date columns
        date_columns = []
        for column in df.columns:
            if 'date' in column.lower() or 'time' in column.lower():
                try:
                    pd.to_datetime(df[column].dropna().head(100))
                    date_columns.append(column)
                except:
                    continue
        
        if date_columns:
            for column in date_columns:
                try:
                    dates = pd.to_datetime(df[column].dropna())
                    if len(dates) > 0:
                        latest_date = dates.max()
                        oldest_date = dates.min()
                        
                        # Check if data is recent (within last year)
                        days_old = (pd.Timestamp.now() - latest_date).days
                        if days_old > 365:
                            timeliness_score -= 0.2
                            timeliness_indicators.append({
                                'column': column,
                                'issue': f'Data is {days_old} days old',
                                'latest_date': latest_date.strftime('%Y-%m-%d')
                            })
                except:
                    continue
        
        return {
            'score': max(0, timeliness_score),
            'date_columns': date_columns,
            'timeliness_indicators': timeliness_indicators,
            'quality_level': self._get_quality_level(timeliness_score, 'timeliness')
        }
    
    def _calculate_overall_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate weighted overall quality score"""
        score = 0
        for dimension in self.quality_dimensions:
            if dimension in assessment:
                weight = self.weights.get(dimension, 0)
                dimension_score = assessment[dimension].get('score', 0)
                score += weight * dimension_score
        
        return score
    
    def _get_quality_level(self, score: float, dimension: str) -> str:
        """Get quality level label for a score"""
        thresholds = self.thresholds.get(dimension, self.thresholds['completeness'])
        
        if score >= thresholds['excellent']:
            return 'Excellent'
        elif score >= thresholds['good']:
            return 'Good'
        elif score >= thresholds['poor']:
            return 'Fair'
        else:
            return 'Poor'
    
    def _generate_quality_summary(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality summary"""
        overall_score = assessment['overall_score']
        overall_level = self._get_quality_level(overall_score, 'completeness')
        
        # Count issues by severity
        critical_issues = 0
        moderate_issues = 0
        minor_issues = 0
        
        for dimension in self.quality_dimensions:
            if dimension in assessment:
                level = assessment[dimension].get('quality_level', 'Good')
                if level == 'Poor':
                    critical_issues += 1
                elif level == 'Fair':
                    moderate_issues += 1
                elif level == 'Good':
                    minor_issues += 1
        
        return {
            'overall_score': overall_score,
            'overall_level': overall_level,
            'critical_issues': critical_issues,
            'moderate_issues': moderate_issues,
            'minor_issues': minor_issues,
            'dimension_scores': {dim: assessment[dim]['score'] 
                               for dim in self.quality_dimensions if dim in assessment}
        }
    
    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Completeness recommendations
        if assessment['completeness']['score'] < 0.9:
            recommendations.append("Improve data completeness by implementing missing value imputation")
        
        # Validity recommendations  
        if assessment['validity']['score'] < 0.9:
            recommendations.append("Implement data validation rules to improve validity")
        
        # Consistency recommendations
        if assessment['consistency']['score'] < 0.9:
            recommendations.append("Standardize data formats and implement consistency checks")
        
        # Uniqueness recommendations
        if assessment['uniqueness']['score'] < 0.9:
            recommendations.append("Implement duplicate detection and removal processes")
        
        # General recommendations based on overall score
        overall_score = assessment['overall_score']
        if overall_score < 0.7:
            recommendations.append("Consider comprehensive data quality improvement initiative")
        elif overall_score < 0.85:
            recommendations.append("Focus on specific quality dimensions with lowest scores")
        
        return recommendations
    
    def generate_quality_report(self, df: pd.DataFrame,
                              target_column: str = None,
                              business_rules: Dict[str, Any] = None) -> str:
        """
        Generate comprehensive data quality report
        
        Args:
            df: Input DataFrame
            target_column: Target column for analysis
            business_rules: Dict of business rules for validation
            
        Returns:
            Formatted quality report string
        """
        assessment = self.assess_quality(df, target_column, business_rules)
        
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall summary
        summary = assessment['summary']
        report.append(f"Overall Quality Score: {summary['overall_score']:.2f} ({summary['overall_level']})")
        report.append(f"Dataset Size: {len(df):,} records, {len(df.columns)} columns")
        report.append("")
        
        # Dimension scores
        report.append("Quality Dimensions:")
        report.append("-" * 30)
        for dimension in self.quality_dimensions:
            if dimension in assessment:
                score = assessment[dimension]['score']
                level = assessment[dimension]['quality_level']
                report.append(f"{dimension.capitalize():15} {score:.3f} ({level})")
        report.append("")
        
        # Issues summary
        report.append(f"Issues Summary:")
        report.append(f"Critical Issues: {summary['critical_issues']}")
        report.append(f"Moderate Issues: {summary['moderate_issues']}")
        report.append(f"Minor Issues: {summary['minor_issues']}")
        report.append("")
        
        # Recommendations
        recommendations = assessment['recommendations']
        if recommendations:
            report.append("Recommendations:")
            report.append("-" * 20)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
        
        report.append("=" * 60)
        
        return "\n".join(report)