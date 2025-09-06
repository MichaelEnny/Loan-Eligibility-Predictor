#!/usr/bin/env python3
"""
Sample Loan Dataset Generator for EDA
Creates a realistic synthetic loan dataset based on PRD requirements
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_loan_dataset(n_samples=5000):
    """Generate synthetic loan dataset with realistic distributions"""
    
    data = {}
    
    # Basic Demographics
    ages = np.random.normal(35, 12, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)  # Age between 18-80
    data['age'] = ages
    
    # Gender (protected attribute)
    data['gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
    
    # Marital Status
    marital_probs = [0.45, 0.35, 0.15, 0.05]  # Single, Married, Divorced, Widowed
    data['marital_status'] = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 
                                            n_samples, p=marital_probs)
    
    # Education Level
    education_probs = [0.15, 0.25, 0.35, 0.25]  # High School, Some College, Bachelor's, Advanced
    data['education'] = np.random.choice(['High School', 'Some College', 'Bachelor\'s', 'Advanced'], 
                                       n_samples, p=education_probs)
    
    # Employment Information
    employment_probs = [0.75, 0.15, 0.05, 0.05]  # Employed, Self-employed, Unemployed, Retired
    data['employment_status'] = np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Retired'], 
                                                n_samples, p=employment_probs)
    
    # Years at current job
    data['years_employed'] = np.random.exponential(5, n_samples)
    data['years_employed'] = np.clip(data['years_employed'], 0, 40)
    
    # Annual Income (correlated with education and employment)
    base_income = np.random.normal(50000, 20000, n_samples)
    
    # Adjust income based on education
    education_multiplier = np.where(data['education'] == 'High School', 0.8,
                          np.where(data['education'] == 'Some College', 0.9,
                          np.where(data['education'] == 'Bachelor\'s', 1.1, 1.3)))
    
    # Adjust income based on employment status
    employment_multiplier = np.where(data['employment_status'] == 'Unemployed', 0.1,
                           np.where(data['employment_status'] == 'Retired', 0.4,
                           np.where(data['employment_status'] == 'Self-employed', 1.2, 1.0)))
    
    data['annual_income'] = base_income * education_multiplier * employment_multiplier
    data['annual_income'] = np.clip(data['annual_income'], 0, 500000)
    
    # Monthly Income
    data['monthly_income'] = data['annual_income'] / 12
    
    # Credit History
    data['credit_score'] = np.random.normal(650, 100, n_samples).astype(int)
    data['credit_score'] = np.clip(data['credit_score'], 300, 850)
    
    # Credit History Length (years)
    data['credit_history_length'] = np.random.exponential(8, n_samples)
    data['credit_history_length'] = np.clip(data['credit_history_length'], 0, 50)
    
    # Number of credit accounts
    data['num_credit_accounts'] = np.random.poisson(3, n_samples)
    data['num_credit_accounts'] = np.clip(data['num_credit_accounts'], 0, 20)
    
    # Current Debt Information
    data['existing_debt'] = np.random.exponential(25000, n_samples)
    data['existing_debt'] = np.clip(data['existing_debt'], 0, 200000)
    
    # Monthly debt payments
    data['monthly_debt_payments'] = data['existing_debt'] * 0.03  # 3% monthly payment
    
    # Debt-to-Income Ratio
    # Avoid division by zero
    data['debt_to_income_ratio'] = np.where(data['monthly_income'] > 0, 
                                          data['monthly_debt_payments'] / data['monthly_income'], 0)
    data['debt_to_income_ratio'] = np.clip(data['debt_to_income_ratio'], 0, 2)
    
    # Loan Request Information
    # Loan amount requested
    data['loan_amount'] = np.random.normal(25000, 15000, n_samples)
    data['loan_amount'] = np.clip(data['loan_amount'], 1000, 100000)
    
    # Loan term (months)
    loan_terms = [12, 24, 36, 48, 60, 72, 84, 96]
    data['loan_term_months'] = np.random.choice(loan_terms, n_samples)
    
    # Loan purpose
    purposes = ['Auto', 'Home Improvement', 'Debt Consolidation', 'Medical', 'Education', 
               'Business', 'Personal', 'Other']
    purpose_probs = [0.25, 0.15, 0.20, 0.08, 0.10, 0.12, 0.05, 0.05]
    data['loan_purpose'] = np.random.choice(purposes, n_samples, p=purpose_probs)
    
    # Property ownership
    data['owns_property'] = np.random.choice([True, False], n_samples, p=[0.65, 0.35])
    
    # Property value (for those who own property)
    property_values = np.random.normal(200000, 80000, n_samples)
    data['property_value'] = np.where(data['owns_property'], 
                                    np.clip(property_values, 50000, 1000000), 0)
    
    # Banking relationship
    data['has_bank_account'] = np.random.choice([True, False], n_samples, p=[0.95, 0.05])
    data['years_with_bank'] = np.random.exponential(7, n_samples)
    data['years_with_bank'] = np.where(data['has_bank_account'], 
                                     np.clip(data['years_with_bank'], 0, 50), 0)
    
    # Previous loan history
    data['previous_loans'] = np.random.poisson(1, n_samples)
    data['previous_loan_defaults'] = np.random.binomial(data['previous_loans'], 0.05)
    
    # Geographic information
    states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 'Other']
    state_probs = [0.12, 0.09, 0.06, 0.06, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.46]
    data['state'] = np.random.choice(states, n_samples, p=state_probs)
    
    # Urban vs Rural
    data['area_type'] = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.4, 0.45, 0.15])
    
    # Co-applicant information (30% have co-applicants)
    has_coapplicant = np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    data['has_coapplicant'] = has_coapplicant
    
    coapplicant_income = np.random.normal(40000, 15000, n_samples)
    data['coapplicant_income'] = np.where(has_coapplicant, 
                                        np.clip(coapplicant_income, 0, 200000), 0)
    
    # Total household income
    data['total_household_income'] = data['annual_income'] + data['coapplicant_income']
    
    # Generate target variable (loan approval) based on realistic factors
    # Higher probability of approval for:
    # - Higher credit score
    # - Lower debt-to-income ratio
    # - Higher income
    # - Property ownership
    # - Longer employment history
    # - Bank relationship
    
    approval_score = (
        (data['credit_score'] - 300) / 550 * 0.3 +  # Credit score contribution
        (1 - np.clip(data['debt_to_income_ratio'], 0, 1)) * 0.25 +  # DTI contribution  
        np.log1p(data['annual_income']) / np.log1p(100000) * 0.2 +  # Income contribution
        data['owns_property'].astype(float) * 0.1 +  # Property ownership
        np.clip(data['years_employed'], 0, 10) / 10 * 0.1 +  # Employment history
        (data['years_with_bank'] > 0).astype(float) * 0.05  # Banking relationship
    )
    
    # Add some noise and convert to probability
    approval_prob = 1 / (1 + np.exp(-(approval_score * 6 - 2)))  # Sigmoid transformation
    
    # Generate actual approvals
    data['loan_approved'] = np.random.binomial(1, approval_prob, n_samples)
    
    # Convert to DataFrame first
    df = pd.DataFrame(data)
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(n_samples, int(n_samples * 0.02), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], 'credit_score'] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], 'property_value'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, int(n_samples * 0.01), replace=False)
    df.loc[outlier_indices, 'annual_income'] *= np.random.uniform(5, 10, len(outlier_indices))
    
    # Application date (last 2 years)
    start_date = datetime.now() - timedelta(days=730)
    application_dates = [start_date + timedelta(days=random.randint(0, 730)) 
                        for _ in range(n_samples)]
    df['application_date'] = application_dates
    
    # Round numerical columns appropriately
    df['age'] = df['age'].round(0).astype('Int64')
    df['annual_income'] = df['annual_income'].round(2)
    df['monthly_income'] = df['monthly_income'].round(2)
    df['existing_debt'] = df['existing_debt'].round(2)
    df['loan_amount'] = df['loan_amount'].round(2)
    df['property_value'] = df['property_value'].round(2)
    df['years_employed'] = df['years_employed'].round(1)
    df['credit_history_length'] = df['credit_history_length'].round(1)
    df['years_with_bank'] = df['years_with_bank'].round(1)
    df['debt_to_income_ratio'] = df['debt_to_income_ratio'].round(3)
    
    return df

if __name__ == "__main__":
    # Generate the dataset
    print("Generating synthetic loan dataset...")
    loan_data = generate_loan_dataset(5000)
    
    # Save to CSV
    loan_data.to_csv('loan_dataset.csv', index=False)
    print(f"Dataset saved to loan_dataset.csv")
    print(f"Dataset shape: {loan_data.shape}")
    print(f"Columns: {list(loan_data.columns)}")
    print(f"Target variable distribution:")
    print(loan_data['loan_approved'].value_counts())