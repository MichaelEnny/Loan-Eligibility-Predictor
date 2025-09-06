#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis for Loan Eligibility Dataset
Task ID: ML-001 - Data Analysis & Exploration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import scipy.stats as stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class LoanEDA:
    def __init__(self, data_path='loan_dataset.csv'):
        """Initialize EDA class and load data"""
        print("Loading loan dataset...")
        self.df = pd.read_csv(data_path)
        self.df['application_date'] = pd.to_datetime(self.df['application_date'])
        print(f"Dataset loaded successfully: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        
    def basic_info(self):
        """Generate basic dataset information"""
        print("\n" + "="*80)
        print("📊 BASIC DATASET INFORMATION")
        print("="*80)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nColumn Information:")
        print("-" * 50)
        info_df = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes,
            'Non-Null Count': self.df.count(),
            'Null Count': self.df.isnull().sum(),
            'Null %': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        print(info_df.to_string(index=False))
        
        return info_df
    
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        print("\n" + "="*80)
        print("📈 DESCRIPTIVE STATISTICS")
        print("="*80)
        
        # Numerical features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'loan_approved' in numerical_cols:
            numerical_cols.remove('loan_approved')
            
        print("\n🔢 NUMERICAL FEATURES SUMMARY:")
        print("-" * 70)
        num_stats = self.df[numerical_cols].describe().round(2)
        print(num_stats)
        
        # Categorical features
        categorical_cols = self.df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        print(f"\n🏷️  CATEGORICAL FEATURES SUMMARY ({len(categorical_cols)} features):")
        print("-" * 70)
        for col in categorical_cols:
            if col != 'application_date':
                print(f"\n{col.upper()}:")
                value_counts = self.df[col].value_counts()
                percentages = (self.df[col].value_counts(normalize=True) * 100).round(1)
                
                summary = pd.DataFrame({
                    'Count': value_counts,
                    'Percentage': percentages.map(lambda x: f"{x}%")
                })
                print(summary)
                
        return num_stats
    
    def missing_data_analysis(self):
        """Analyze missing data patterns"""
        print("\n" + "="*80)
        print("❓ MISSING DATA ANALYSIS")
        print("="*80)
        
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        if len(missing_data) == 0:
            print("✅ No missing values found in the dataset!")
        else:
            print("📋 Missing Data Summary:")
            print("-" * 50)
            print(missing_data.to_string(index=False))
            
            # Create missing data heatmap
            plt.figure(figsize=(12, 8))
            missing_cols = missing_data['Column'].tolist()
            sns.heatmap(self.df[missing_cols].isnull(), cmap='viridis', cbar=True, yticklabels=False)
            plt.title('Missing Data Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('missing_data_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        return missing_data
    
    def target_analysis(self):
        """Analyze target variable distribution"""
        print("\n" + "="*80)
        print("🎯 TARGET VARIABLE ANALYSIS")
        print("="*80)
        
        target_counts = self.df['loan_approved'].value_counts()
        target_pcts = self.df['loan_approved'].value_counts(normalize=True) * 100
        
        print("Loan Approval Distribution:")
        print("-" * 30)
        print(f"Approved (1): {target_counts[1]:,} ({target_pcts[1]:.1f}%)")
        print(f"Rejected (0): {target_counts[0]:,} ({target_pcts[0]:.1f}%)")
        print(f"Approval Rate: {target_pcts[1]:.1f}%")
        
        # Create target distribution visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot
        target_counts.plot(kind='bar', ax=ax1, color=['#ff7f0e', '#1f77b4'])
        ax1.set_title('Loan Approval Distribution (Count)', fontweight='bold')
        ax1.set_xlabel('Loan Status (0=Rejected, 1=Approved)')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # Pie chart
        ax2.pie(target_counts.values, labels=['Rejected', 'Approved'], 
                autopct='%1.1f%%', startangle=90, colors=['#ff7f0e', '#1f77b4'])
        ax2.set_title('Loan Approval Distribution (Percentage)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return target_counts, target_pcts
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("\n" + "="*80)
        print("🔗 CORRELATION ANALYSIS")
        print("="*80)
        
        # Select numerical columns for correlation
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = self.df[numerical_cols].corr()
        
        # Find correlations with target variable
        target_corr = corr_matrix['loan_approved'].abs().sort_values(ascending=False)
        
        print("🎯 Features most correlated with Loan Approval:")
        print("-" * 50)
        for feature, correlation in target_corr.head(10).items():
            if feature != 'loan_approved':
                direction = "positive" if corr_matrix['loan_approved'][feature] > 0 else "negative"
                print(f"{feature:.<30} {correlation:.3f} ({direction})")
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix, target_corr
    
    def univariate_analysis(self):
        """Perform univariate analysis of key features"""
        print("\n" + "="*80)
        print("📊 UNIVARIATE ANALYSIS")
        print("="*80)
        
        # Key numerical features to analyze
        key_numerical = ['annual_income', 'credit_score', 'debt_to_income_ratio', 
                        'loan_amount', 'existing_debt', 'age']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(key_numerical):
            if col in self.df.columns:
                # Remove outliers for better visualization
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                filtered_data = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)][col]
                
                axes[i].hist(filtered_data, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}', fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                
                # Add statistics to plot
                mean_val = filtered_data.mean()
                median_val = filtered_data.median()
                axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('univariate_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def bivariate_analysis(self):
        """Perform bivariate analysis with target variable"""
        print("\n" + "="*80)
        print("🔄 BIVARIATE ANALYSIS")
        print("="*80)
        
        # Key features for bivariate analysis
        key_features = ['annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(key_features):
            if feature in self.df.columns:
                # Box plot for each feature by loan approval status
                approved = self.df[self.df['loan_approved'] == 1][feature]
                rejected = self.df[self.df['loan_approved'] == 0][feature]
                
                axes[i].boxplot([rejected.dropna(), approved.dropna()], 
                               labels=['Rejected', 'Approved'])
                axes[i].set_title(f'{feature} by Loan Approval Status', fontweight='bold')
                axes[i].set_ylabel(feature)
                
                # Perform t-test
                if len(approved.dropna()) > 0 and len(rejected.dropna()) > 0:
                    statistic, p_value = stats.ttest_ind(approved.dropna(), rejected.dropna())
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    axes[i].text(0.5, 0.95, f'p-value: {p_value:.4f} {significance}', 
                               transform=axes[i].transAxes, ha='center', va='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('bivariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Categorical features analysis
        categorical_features = ['gender', 'education', 'employment_status', 'loan_purpose']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(categorical_features):
            if feature in self.df.columns:
                # Create contingency table
                contingency = pd.crosstab(self.df[feature], self.df['loan_approved'], normalize='index') * 100
                
                contingency.plot(kind='bar', ax=axes[i], color=['#ff7f0e', '#1f77b4'])
                axes[i].set_title(f'Loan Approval Rate by {feature}', fontweight='bold')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Approval Rate (%)')
                axes[i].legend(['Rejected', 'Approved'], title='Loan Status')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('categorical_bivariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def outlier_analysis(self):
        """Identify and analyze outliers"""
        print("\n" + "="*80)
        print("🚨 OUTLIER ANALYSIS")
        print("="*80)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'loan_approved' in numerical_cols:
            numerical_cols.remove('loan_approved')
        
        outlier_summary = []
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.df)) * 100
            
            outlier_summary.append({
                'Feature': col,
                'Outlier_Count': outlier_count,
                'Outlier_Percentage': round(outlier_percentage, 2),
                'Q1': round(Q1, 2),
                'Q3': round(Q3, 2),
                'IQR': round(IQR, 2),
                'Lower_Bound': round(lower_bound, 2),
                'Upper_Bound': round(upper_bound, 2)
            })
        
        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df = outlier_df[outlier_df['Outlier_Count'] > 0].sort_values('Outlier_Percentage', ascending=False)
        
        if len(outlier_df) > 0:
            print("📊 Outlier Summary:")
            print("-" * 80)
            print(outlier_df.to_string(index=False))
        else:
            print("✅ No outliers detected using IQR method")
            
        return outlier_df
    
    def data_quality_report(self):
        """Generate comprehensive data quality assessment report"""
        print("\n" + "="*80)
        print("🔍 DATA QUALITY ASSESSMENT REPORT")
        print("="*80)
        
        # Overall data quality score
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100
        
        print(f"📊 OVERALL DATA QUALITY METRICS:")
        print("-" * 50)
        print(f"Data Completeness: {completeness_score:.2f}%")
        print(f"Total Records: {len(self.df):,}")
        print(f"Total Features: {len(self.df.columns)}")
        print(f"Missing Values: {missing_cells:,}")
        
        # Feature quality assessment
        quality_assessment = []
        for col in self.df.columns:
            if col != 'application_date':  # Skip date column
                missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
                unique_values = self.df[col].nunique()
                unique_ratio = unique_values / len(self.df)
                
                # Data type consistency
                dtype_consistent = True  # Assuming consistent based on pandas loading
                
                # Quality score calculation
                quality_score = 100 - missing_pct - (10 if unique_ratio > 0.95 and self.df[col].dtype == 'object' else 0)
                
                quality_assessment.append({
                    'Feature': col,
                    'Missing_%': round(missing_pct, 2),
                    'Unique_Values': unique_values,
                    'Unique_Ratio': round(unique_ratio, 4),
                    'Data_Type': str(self.df[col].dtype),
                    'Quality_Score': round(quality_score, 1)
                })
        
        quality_df = pd.DataFrame(quality_assessment)
        quality_df = quality_df.sort_values('Quality_Score', ascending=False)
        
        print(f"\n📋 FEATURE QUALITY ASSESSMENT:")
        print("-" * 80)
        print(quality_df.to_string(index=False))
        
        # Data quality recommendations
        print(f"\n💡 DATA QUALITY RECOMMENDATIONS:")
        print("-" * 50)
        
        low_quality_features = quality_df[quality_df['Quality_Score'] < 90]['Feature'].tolist()
        high_missing_features = quality_df[quality_df['Missing_%'] > 5]['Feature'].tolist()
        high_cardinality_features = quality_df[(quality_df['Unique_Ratio'] > 0.95) & 
                                              (quality_df['Data_Type'] == 'object')]['Feature'].tolist()
        
        if low_quality_features:
            print(f"⚠️  Features requiring attention: {', '.join(low_quality_features)}")
        
        if high_missing_features:
            print(f"❓ Features with high missing values: {', '.join(high_missing_features)}")
            
        if high_cardinality_features:
            print(f"🔢 High cardinality features: {', '.join(high_cardinality_features)}")
            
        if not low_quality_features and not high_missing_features:
            print("✅ Overall data quality is excellent!")
            
        return quality_df
    
    def generate_full_report(self):
        """Generate complete EDA report"""
        print("\n" + "🔍" * 40)
        print("COMPREHENSIVE LOAN ELIGIBILITY EDA REPORT")
        print("🔍" * 40 + "\n")
        
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Dataset: Loan Eligibility Prediction")
        print(f"Task ID: ML-001")
        
        # Run all analyses
        info_df = self.basic_info()
        stats_df = self.descriptive_statistics()
        missing_df = self.missing_data_analysis()
        target_counts, target_pcts = self.target_analysis()
        corr_matrix, target_corr = self.correlation_analysis()
        self.univariate_analysis()
        self.bivariate_analysis()
        outlier_df = self.outlier_analysis()
        quality_df = self.data_quality_report()
        
        # Summary insights
        print(f"\n" + "🎯" * 40)
        print("KEY INSIGHTS & FINDINGS")
        print("🎯" * 40 + "\n")
        
        print("📊 Dataset Overview:")
        print(f"   • Dataset contains {len(self.df):,} loan applications with {len(self.df.columns)} features")
        print(f"   • Overall approval rate: {target_pcts[1]:.1f}%")
        print(f"   • Data completeness: {((self.df.shape[0] * self.df.shape[1] - self.df.isnull().sum().sum()) / (self.df.shape[0] * self.df.shape[1]) * 100):.1f}%")
        
        print(f"\n🔗 Feature Relationships:")
        top_corr_features = target_corr.head(6)[1:6]  # Exclude target itself
        for feature, corr in top_corr_features.items():
            direction = "positively" if corr_matrix['loan_approved'][feature] > 0 else "negatively"
            print(f"   • {feature} is {direction} correlated with loan approval (r={corr:.3f})")
        
        if len(outlier_df) > 0:
            print(f"\n🚨 Data Quality Issues:")
            for _, row in outlier_df.head(3).iterrows():
                print(f"   • {row['Feature']}: {row['Outlier_Count']} outliers ({row['Outlier_Percentage']:.1f}%)")
        
        print(f"\n✅ EDA Report Complete!")
        print(f"   • Generated visualizations: 6 charts")
        print(f"   • Analysis completed for all {len(self.df.columns)} features")
        print(f"   • Ready for feature engineering and model development")
        
        return {
            'basic_info': info_df,
            'statistics': stats_df,
            'missing_data': missing_df,
            'target_analysis': (target_counts, target_pcts),
            'correlations': (corr_matrix, target_corr),
            'outliers': outlier_df,
            'quality_assessment': quality_df
        }

def main():
    """Main function to run the EDA"""
    try:
        # Initialize EDA
        eda = LoanEDA('loan_dataset.csv')
        
        # Generate complete report
        results = eda.generate_full_report()
        
        print(f"\n🎉 EDA Task ML-001 completed successfully!")
        print(f"📁 Generated files:")
        print(f"   • missing_data_heatmap.png")
        print(f"   • target_distribution.png") 
        print(f"   • correlation_heatmap.png")
        print(f"   • univariate_distributions.png")
        print(f"   • bivariate_analysis.png")
        print(f"   • categorical_bivariate_analysis.png")
        
    except Exception as e:
        print(f"❌ Error during EDA: {str(e)}")
        raise

if __name__ == "__main__":
    main()