#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis for Loan Eligibility Dataset
Task ID: ML-001 - Data Analysis & Exploration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import scipy.stats as stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('default')

def load_and_explore_data():
    """Load data and perform comprehensive EDA"""
    print("="*80)
    print("LOAN ELIGIBILITY PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("Task ID: ML-001")
    print("="*80)
    
    # Load data
    print("\nLoading loan dataset...")
    df = pd.read_csv('loan_dataset.csv')
    df['application_date'] = pd.to_datetime(df['application_date'])
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Basic Information
    print("\n" + "="*50)
    print("1. BASIC DATASET INFORMATION")
    print("="*50)
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percentage': missing_pct
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    
    if len(missing_summary) > 0:
        print(f"\nMissing Data:")
        print(missing_summary)
    else:
        print("\nNo missing values found!")
    
    # Target Variable Analysis
    print("\n" + "="*50)
    print("2. TARGET VARIABLE ANALYSIS")
    print("="*50)
    
    target_counts = df['loan_approved'].value_counts()
    target_pcts = df['loan_approved'].value_counts(normalize=True) * 100
    
    print("Loan Approval Distribution:")
    print(f"Approved (1): {target_counts[1]:,} ({target_pcts[1]:.1f}%)")
    print(f"Rejected (0): {target_counts[0]:,} ({target_pcts[0]:.1f}%)")
    print(f"Approval Rate: {target_pcts[1]:.1f}%")
    
    # Descriptive Statistics
    print("\n" + "="*50)
    print("3. DESCRIPTIVE STATISTICS")
    print("="*50)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'loan_approved' in numerical_cols:
        numerical_cols.remove('loan_approved')
    
    print("\nNumerical Features Summary:")
    print(df[numerical_cols].describe().round(2))
    
    # Categorical Features
    print("\nCategorical Features Summary:")
    categorical_cols = ['gender', 'education', 'employment_status', 'loan_purpose']
    
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col.upper()}:")
            value_counts = df[col].value_counts()
            percentages = (df[col].value_counts(normalize=True) * 100).round(1)
            
            for value, count in value_counts.items():
                print(f"  {value}: {count} ({percentages[value]:.1f}%)")
    
    # Correlation Analysis
    print("\n" + "="*50)
    print("4. CORRELATION ANALYSIS")
    print("="*50)
    
    corr_matrix = df[numerical_cols + ['loan_approved']].corr()
    target_corr = corr_matrix['loan_approved'].abs().sort_values(ascending=False)
    
    print("Features most correlated with Loan Approval:")
    for feature, correlation in target_corr.head(10).items():
        if feature != 'loan_approved':
            direction = "positive" if corr_matrix['loan_approved'][feature] > 0 else "negative"
            print(f"  {feature:<25} {correlation:.3f} ({direction})")
    
    # Outlier Analysis
    print("\n" + "="*50)
    print("5. OUTLIER ANALYSIS")
    print("="*50)
    
    outlier_summary = []
    for col in ['annual_income', 'credit_score', 'loan_amount', 'existing_debt']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            print(f"{col}:")
            print(f"  Outliers: {outlier_count} ({outlier_percentage:.1f}%)")
            print(f"  Normal range: {lower_bound:.2f} to {upper_bound:.2f}")
    
    # Create Visualizations
    print("\n" + "="*50)
    print("6. GENERATING VISUALIZATIONS")
    print("="*50)
    
    # Target distribution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    target_counts.plot(kind='bar', color=['orange', 'blue'])
    plt.title('Loan Approval Distribution')
    plt.xlabel('Loan Status (0=Rejected, 1=Approved)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    plt.pie(target_counts.values, labels=['Rejected', 'Approved'], 
            autopct='%1.1f%%', startangle=90, colors=['orange', 'blue'])
    plt.title('Loan Approval Percentage')
    
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    key_features = ['annual_income', 'credit_score', 'debt_to_income_ratio', 
                   'loan_amount', 'age', 'existing_debt', 'loan_approved']
    
    if all(col in df.columns for col in key_features):
        sns.heatmap(df[key_features].corr(), annot=True, cmap='RdBu_r', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    key_numerical = ['annual_income', 'credit_score', 'debt_to_income_ratio', 
                    'loan_amount', 'existing_debt', 'age']
    
    for i, col in enumerate(key_numerical):
        if col in df.columns and i < 6:
            # Remove extreme outliers for visualization
            Q1 = df[col].quantile(0.05)
            Q3 = df[col].quantile(0.95)
            filtered_data = df[(df[col] >= Q1) & (df[col] <= Q3)][col]
            
            axes[i].hist(filtered_data, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Bivariate analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    key_features_bivar = ['annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount']
    
    for i, feature in enumerate(key_features_bivar):
        if feature in df.columns:
            approved = df[df['loan_approved'] == 1][feature].dropna()
            rejected = df[df['loan_approved'] == 0][feature].dropna()
            
            axes[i].boxplot([rejected, approved], labels=['Rejected', 'Approved'])
            axes[i].set_title(f'{feature} by Loan Status')
            axes[i].set_ylabel(feature)
            
            # Statistical test
            if len(approved) > 0 and len(rejected) > 0:
                _, p_value = stats.ttest_ind(approved, rejected)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                axes[i].text(0.5, 0.95, f'p={p_value:.4f} {significance}', 
                           transform=axes[i].transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig('bivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Data Quality Report
    print("\n" + "="*50)
    print("7. DATA QUALITY ASSESSMENT")
    print("="*50)
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness_score = ((total_cells - missing_cells) / total_cells) * 100
    
    print(f"Overall Data Quality Metrics:")
    print(f"  Data Completeness: {completeness_score:.2f}%")
    print(f"  Total Records: {len(df):,}")
    print(f"  Total Features: {len(df.columns)}")
    print(f"  Missing Values: {missing_cells:,}")
    
    # Key Insights
    print("\n" + "="*50)
    print("8. KEY INSIGHTS & FINDINGS")
    print("="*50)
    
    print(f"Dataset Overview:")
    print(f"  • Dataset contains {len(df):,} loan applications with {len(df.columns)} features")
    print(f"  • Overall approval rate: {target_pcts[1]:.1f}%")
    print(f"  • Data completeness: {completeness_score:.1f}%")
    
    print(f"\nTop predictive features (by correlation):")
    top_features = target_corr.head(6)[1:6]  # Exclude target itself
    for feature, corr in top_features.items():
        direction = "positively" if corr_matrix['loan_approved'][feature] > 0 else "negatively" 
        print(f"  • {feature} ({direction} correlated, r={corr:.3f})")
    
    print(f"\nData Quality:")
    if missing_cells == 0:
        print(f"  • No missing values detected")
    else:
        print(f"  • {missing_cells:,} missing values across {len(missing_summary)} features")
    
    print(f"\nEDA Completed Successfully!")
    print(f"Generated visualizations:")
    print(f"  • target_distribution.png")
    print(f"  • correlation_matrix.png") 
    print(f"  • feature_distributions.png")
    print(f"  • bivariate_analysis.png")
    
    return df, target_corr, missing_summary

if __name__ == "__main__":
    try:
        df, correlations, missing_data = load_and_explore_data()
        print(f"\nTask ML-001 completed successfully!")
    except Exception as e:
        print(f"Error during EDA: {str(e)}")
        raise