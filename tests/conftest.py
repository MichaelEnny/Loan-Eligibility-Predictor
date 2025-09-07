"""
Pytest configuration and shared fixtures for feature engineering tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from feature_engineering.config import FeaturePipelineConfig, create_default_loan_config


@pytest.fixture
def sample_loan_data():
    """Create sample loan dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        # Categorical features
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'education': np.random.choice(['High School', 'Some College', "Bachelor's", 'Advanced'], n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Retired'], n_samples),
        'loan_purpose': np.random.choice(['Auto', 'Home Improvement', 'Debt Consolidation', 'Personal'], n_samples),
        'state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'Other'], n_samples),
        'area_type': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        
        # Numerical features
        'age': np.random.randint(18, 80, n_samples),
        'years_employed': np.random.uniform(0, 40, n_samples),
        'annual_income': np.random.lognormal(10, 0.5, n_samples),
        'monthly_income': lambda x: x / 12,  # Will be calculated
        'credit_score': np.random.randint(300, 850, n_samples),
        'credit_history_length': np.random.uniform(0, 30, n_samples),
        'num_credit_accounts': np.random.randint(0, 20, n_samples),
        'existing_debt': np.random.lognormal(8, 1, n_samples),
        'monthly_debt_payments': lambda x: x * 0.05,  # Will be calculated
        'debt_to_income_ratio': lambda x, y: x / y,  # Will be calculated
        'loan_amount': np.random.lognormal(9, 0.8, n_samples),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60, 72, 84, 96], n_samples),
        'property_value': np.random.lognormal(11, 0.7, n_samples),
        'years_with_bank': np.random.uniform(0, 30, n_samples),
        'coapplicant_income': np.random.lognormal(9, 1, n_samples) * np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'total_household_income': lambda annual, coapp: annual + coapp,  # Will be calculated
        'has_coapplicant': lambda coapp: (coapp > 0).astype(int),
        'owns_property': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'has_bank_account': np.random.choice([0, 1], n_samples, p=[0.05, 0.95]),
        'previous_loans': np.random.poisson(1, n_samples),
        'previous_loan_defaults': np.random.poisson(0.1, n_samples),
        
        # Target variable
        'loan_approved': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    # Create DataFrame
    df = pd.DataFrame()
    for key, value in data.items():
        if callable(value):
            continue
        df[key] = value
    
    # Calculate derived features
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_debt_payments'] = df['existing_debt'] * 0.05
    df['debt_to_income_ratio'] = df['existing_debt'] / df['annual_income']
    df['total_household_income'] = df['annual_income'] + df['coapplicant_income']
    df['has_coapplicant'] = (df['coapplicant_income'] > 0).astype(int)
    
    # Add some missing values for testing
    missing_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[missing_indices, 'credit_score'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[missing_indices, 'years_employed'] = np.nan
    
    return df


@pytest.fixture
def sample_target(sample_loan_data):
    """Extract target variable from sample data."""
    return sample_loan_data['loan_approved']


@pytest.fixture
def sample_features(sample_loan_data):
    """Extract features from sample data."""
    return sample_loan_data.drop(columns=['loan_approved'])


@pytest.fixture
def default_config():
    """Create default configuration for testing."""
    return create_default_loan_config()


@pytest.fixture
def minimal_config():
    """Create minimal configuration for testing."""
    return FeaturePipelineConfig(
        target_column='loan_approved',
        exclude_columns=[],
        n_jobs=1,
        random_state=42,
        verbose=False
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def categorical_data():
    """Create simple categorical dataset for testing."""
    np.random.seed(42)
    n_samples = 500
    
    return pd.DataFrame({
        'low_cardinality': np.random.choice(['A', 'B', 'C'], n_samples),
        'medium_cardinality': np.random.choice([f'Cat_{i}' for i in range(10)], n_samples),
        'high_cardinality': np.random.choice([f'Item_{i}' for i in range(100)], n_samples),
        'ordinal_feature': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'binary_feature': np.random.choice(['Yes', 'No'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })


@pytest.fixture
def numerical_data():
    """Create simple numerical dataset for testing."""
    np.random.seed(42)
    n_samples = 500
    
    return pd.DataFrame({
        'normal_feature': np.random.normal(0, 1, n_samples),
        'skewed_feature': np.random.lognormal(0, 1, n_samples),
        'uniform_feature': np.random.uniform(-1, 1, n_samples),
        'outlier_feature': np.concatenate([
            np.random.normal(0, 1, n_samples - 50),
            np.random.normal(10, 1, 50)  # Outliers
        ]),
        'zero_variance': np.ones(n_samples),
        'missing_values': np.where(
            np.random.random(n_samples) < 0.1, 
            np.nan, 
            np.random.normal(0, 1, n_samples)
        ),
        'target': np.random.choice([0, 1], n_samples)
    })


@pytest.fixture
def interaction_data():
    """Create dataset suitable for interaction testing."""
    np.random.seed(42)
    n_samples = 300
    
    # Create features with known interactions
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.uniform(1, 10, n_samples)  # Always positive for log transform
    
    return pd.DataFrame({
        'feature_1': x1,
        'feature_2': x2,
        'feature_3': x3,
        'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': (x1 * x2 + 0.5 * x3 + np.random.normal(0, 0.1, n_samples) > 0).astype(int)
    })