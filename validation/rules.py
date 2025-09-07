"""
Business rule validation for loan eligibility prediction system.

Implements domain-specific validation rules that ensure data meets
business requirements and logical constraints for loan processing.
"""

from typing import Any, Dict, List, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .exceptions import BusinessRuleValidationError


logger = logging.getLogger(__name__)


class LoanBusinessRules:
    """Business rules specific to loan eligibility validation."""
    
    def __init__(self):
        """Initialize with default business rules."""
        self.rules = {}
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register all default business rules."""
        # Financial consistency rules
        self.add_rule(
            "income_consistency",
            self.validate_income_consistency,
            "Monthly income should be approximately annual income / 12"
        )
        
        self.add_rule(
            "debt_income_consistency", 
            self.validate_debt_income_consistency,
            "Debt-to-income ratio should match calculated ratio"
        )
        
        self.add_rule(
            "household_income_consistency",
            self.validate_household_income_consistency,
            "Total household income should include applicant and co-applicant income"
        )
        
        # Loan feasibility rules
        self.add_rule(
            "loan_to_income_ratio",
            self.validate_loan_to_income_ratio,
            "Loan amount should not exceed reasonable multiple of annual income"
        )
        
        self.add_rule(
            "debt_service_capacity",
            self.validate_debt_service_capacity,
            "Combined debt payments should not exceed safe percentage of income"
        )
        
        self.add_rule(
            "loan_term_reasonableness",
            self.validate_loan_term_reasonableness,
            "Loan term should be reasonable for loan purpose and amount"
        )
        
        # Credit profile rules
        self.add_rule(
            "credit_history_consistency",
            self.validate_credit_history_consistency,
            "Credit history length should align with age and credit accounts"
        )
        
        self.add_rule(
            "defaults_vs_credit_score",
            self.validate_defaults_vs_credit_score,
            "Credit score should reflect loan default history"
        )
        
        # Property and collateral rules
        self.add_rule(
            "property_loan_alignment",
            self.validate_property_loan_alignment,
            "Property ownership should align with loan purpose and collateral"
        )
        
        self.add_rule(
            "collateral_adequacy",
            self.validate_collateral_adequacy,
            "Property value should provide adequate collateral for secured loans"
        )
        
        # Employment and stability rules
        self.add_rule(
            "employment_income_alignment",
            self.validate_employment_income_alignment,
            "Employment status should support reported income level"
        )
        
        self.add_rule(
            "employment_stability",
            self.validate_employment_stability,
            "Employment tenure should indicate income stability"
        )
        
        # Banking relationship rules
        self.add_rule(
            "banking_relationship_consistency",
            self.validate_banking_relationship_consistency,
            "Banking relationship should support creditworthiness assessment"
        )
        
        # Age and life stage rules
        self.add_rule(
            "age_loan_term_alignment",
            self.validate_age_loan_term_alignment,
            "Loan term should be reasonable given applicant's age"
        )
        
        # Co-applicant rules
        self.add_rule(
            "coapplicant_data_consistency",
            self.validate_coapplicant_data_consistency,
            "Co-applicant data should be consistent with application flags"
        )
    
    def add_rule(self, name: str, rule_func, description: str = ""):
        """Add a business rule."""
        self.rules[name] = {
            'function': rule_func,
            'description': description,
            'active': True
        }
    
    def disable_rule(self, name: str):
        """Disable a specific rule."""
        if name in self.rules:
            self.rules[name]['active'] = False
    
    def enable_rule(self, name: str):
        """Enable a specific rule."""
        if name in self.rules:
            self.rules[name]['active'] = True
    
    def validate_all_rules(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate all active business rules.
        
        Returns:
            Dictionary with rule results and violations
        """
        results = {
            'passed_rules': [],
            'failed_rules': [],
            'rule_violations': [],
            'overall_valid': True
        }
        
        for rule_name, rule_info in self.rules.items():
            if not rule_info['active']:
                continue
            
            try:
                is_valid, violations = rule_info['function'](data)
                
                if is_valid:
                    results['passed_rules'].append(rule_name)
                else:
                    results['failed_rules'].append(rule_name)
                    results['overall_valid'] = False
                    
                    for violation in violations:
                        violation['rule_name'] = rule_name
                        violation['rule_description'] = rule_info['description']
                        results['rule_violations'].append(violation)
                        
            except Exception as e:
                logger.error(f"Error validating rule '{rule_name}': {str(e)}")
                results['failed_rules'].append(rule_name)
                results['overall_valid'] = False
                results['rule_violations'].append({
                    'rule_name': rule_name,
                    'rule_description': rule_info['description'],
                    'error': f"Rule execution failed: {str(e)}",
                    'violation_type': 'execution_error'
                })
        
        return results
    
    # Business rule implementations
    
    def validate_income_consistency(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate that monthly income is consistent with annual income."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            annual = row.get('annual_income')
            monthly = row.get('monthly_income')
            
            if pd.isna(annual) or pd.isna(monthly):
                continue
            
            expected_monthly = annual / 12
            tolerance = 0.15  # 15% tolerance
            
            if abs(monthly - expected_monthly) > (expected_monthly * tolerance):
                violations.append({
                    'row_index': idx,
                    'violation_type': 'income_inconsistency',
                    'message': f"Monthly income ({monthly:.2f}) inconsistent with annual income ({annual:.2f})",
                    'expected_monthly': expected_monthly,
                    'actual_monthly': monthly,
                    'tolerance_exceeded': True
                })
        
        return len(violations) == 0, violations
    
    def validate_debt_income_consistency(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate debt-to-income ratio calculation."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            monthly_debt = row.get('monthly_debt_payments')
            monthly_income = row.get('monthly_income')
            reported_ratio = row.get('debt_to_income_ratio')
            
            if pd.isna(monthly_debt) or pd.isna(monthly_income) or pd.isna(reported_ratio):
                continue
            
            if monthly_income <= 0:
                continue
            
            calculated_ratio = monthly_debt / monthly_income
            tolerance = 0.02  # 2% tolerance
            
            if abs(calculated_ratio - reported_ratio) > tolerance:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'debt_ratio_inconsistency',
                    'message': f"Debt-to-income ratio ({reported_ratio:.3f}) doesn't match calculation ({calculated_ratio:.3f})",
                    'calculated_ratio': calculated_ratio,
                    'reported_ratio': reported_ratio,
                    'tolerance_exceeded': True
                })
        
        return len(violations) == 0, violations
    
    def validate_household_income_consistency(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate total household income calculation."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            applicant_income = row.get('annual_income', 0)
            coapplicant_income = row.get('coapplicant_income', 0)
            total_household = row.get('total_household_income')
            has_coapplicant = row.get('has_coapplicant', False)
            
            if pd.isna(total_household):
                continue
            
            # Handle missing coapplicant income
            if pd.isna(coapplicant_income):
                coapplicant_income = 0
            
            # Calculate expected total
            if has_coapplicant:
                expected_total = applicant_income + coapplicant_income
            else:
                expected_total = applicant_income
                # Coapplicant income should be 0 if no coapplicant
                if coapplicant_income > 0:
                    violations.append({
                        'row_index': idx,
                        'violation_type': 'coapplicant_income_without_coapplicant',
                        'message': f"Co-applicant income ({coapplicant_income}) reported without co-applicant"
                    })
            
            tolerance = 0.01  # 1% tolerance
            if abs(total_household - expected_total) > (expected_total * tolerance):
                violations.append({
                    'row_index': idx,
                    'violation_type': 'household_income_inconsistency',
                    'message': f"Total household income ({total_household:.2f}) doesn't match sum of incomes ({expected_total:.2f})",
                    'expected_total': expected_total,
                    'reported_total': total_household
                })
        
        return len(violations) == 0, violations
    
    def validate_loan_to_income_ratio(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate loan amount relative to income."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            loan_amount = row.get('loan_amount')
            annual_income = row.get('annual_income')
            loan_purpose = row.get('loan_purpose')
            
            if pd.isna(loan_amount) or pd.isna(annual_income) or annual_income <= 0:
                continue
            
            ratio = loan_amount / annual_income
            
            # Different thresholds by loan purpose
            thresholds = {
                'Auto': 1.0,
                'Personal': 0.5,
                'Debt Consolidation': 1.0,
                'Home Improvement': 2.0,
                'Business': 3.0,
                'Medical': 0.5,
                'Education': 2.0,
                'Other': 1.0
            }
            
            threshold = thresholds.get(loan_purpose, 1.0)
            
            if ratio > threshold:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'excessive_loan_to_income',
                    'message': f"Loan-to-income ratio ({ratio:.2f}) exceeds threshold ({threshold}) for {loan_purpose} loans",
                    'loan_to_income_ratio': ratio,
                    'threshold': threshold,
                    'loan_purpose': loan_purpose
                })
        
        return len(violations) == 0, violations
    
    def validate_debt_service_capacity(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate total debt service capacity."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            monthly_income = row.get('monthly_income')
            current_debt_payments = row.get('monthly_debt_payments', 0)
            loan_amount = row.get('loan_amount')
            loan_term = row.get('loan_term_months')
            
            if pd.isna(monthly_income) or pd.isna(loan_amount) or pd.isna(loan_term):
                continue
            
            if monthly_income <= 0 or loan_term <= 0:
                continue
            
            # Estimate new loan payment (simplified calculation)
            # Using 6% annual rate as estimate
            monthly_rate = 0.06 / 12
            if monthly_rate > 0:
                new_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**loan_term) / ((1 + monthly_rate)**loan_term - 1)
            else:
                new_payment = loan_amount / loan_term
            
            total_debt_payments = current_debt_payments + new_payment
            debt_to_income = total_debt_payments / monthly_income
            
            # Maximum recommended debt-to-income ratio
            max_ratio = 0.43  # 43% is common threshold
            
            if debt_to_income > max_ratio:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'excessive_debt_service',
                    'message': f"Total debt service ratio ({debt_to_income:.1%}) exceeds safe threshold ({max_ratio:.1%})",
                    'debt_to_income_ratio': debt_to_income,
                    'max_ratio': max_ratio,
                    'estimated_new_payment': new_payment
                })
        
        return len(violations) == 0, violations
    
    def validate_loan_term_reasonableness(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate loan term reasonableness."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Reasonable term ranges by loan purpose (in months)
        term_ranges = {
            'Auto': (12, 84),
            'Personal': (12, 60),
            'Debt Consolidation': (24, 84),
            'Home Improvement': (12, 240),
            'Business': (12, 120),
            'Medical': (6, 60),
            'Education': (60, 240),
            'Other': (12, 60)
        }
        
        for idx, row in data.iterrows():
            loan_term = row.get('loan_term_months')
            loan_purpose = row.get('loan_purpose')
            
            if pd.isna(loan_term) or loan_purpose is None:
                continue
            
            min_term, max_term = term_ranges.get(loan_purpose, (6, 480))
            
            if loan_term < min_term or loan_term > max_term:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'unreasonable_loan_term',
                    'message': f"Loan term ({loan_term} months) outside reasonable range ({min_term}-{max_term}) for {loan_purpose} loans",
                    'loan_term': loan_term,
                    'min_term': min_term,
                    'max_term': max_term,
                    'loan_purpose': loan_purpose
                })
        
        return len(violations) == 0, violations
    
    def validate_credit_history_consistency(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate credit history length consistency."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            age = row.get('age')
            credit_history_length = row.get('credit_history_length')
            num_credit_accounts = row.get('num_credit_accounts')
            
            if pd.isna(age) or pd.isna(credit_history_length):
                continue
            
            # Credit history can't be longer than age - 18
            max_possible_history = max(0, age - 18)
            
            if credit_history_length > max_possible_history:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'impossible_credit_history',
                    'message': f"Credit history length ({credit_history_length} years) exceeds possible maximum ({max_possible_history} years) for age {age}",
                    'credit_history_length': credit_history_length,
                    'max_possible': max_possible_history,
                    'age': age
                })
            
            # If credit history is substantial but no credit accounts, flag inconsistency
            if not pd.isna(num_credit_accounts) and credit_history_length > 2 and num_credit_accounts == 0:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'credit_history_account_mismatch',
                    'message': f"Long credit history ({credit_history_length} years) but no active credit accounts",
                    'credit_history_length': credit_history_length,
                    'num_credit_accounts': num_credit_accounts
                })
        
        return len(violations) == 0, violations
    
    def validate_defaults_vs_credit_score(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate credit score relative to default history."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            credit_score = row.get('credit_score')
            defaults = row.get('previous_loan_defaults', 0)
            
            if pd.isna(credit_score) or pd.isna(defaults):
                continue
            
            # High credit score with many defaults is suspicious
            if credit_score >= 700 and defaults >= 2:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'credit_score_default_mismatch',
                    'message': f"High credit score ({credit_score}) inconsistent with {defaults} previous defaults",
                    'credit_score': credit_score,
                    'previous_defaults': defaults
                })
            
            # Very low credit score with no defaults might indicate other issues
            if credit_score < 500 and defaults == 0:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'low_score_no_defaults',
                    'message': f"Very low credit score ({credit_score}) with no recorded defaults may indicate incomplete credit history",
                    'credit_score': credit_score,
                    'previous_defaults': defaults
                })
        
        return len(violations) == 0, violations
    
    def validate_property_loan_alignment(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate property ownership alignment with loan purpose."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            owns_property = row.get('owns_property')
            property_value = row.get('property_value')
            loan_purpose = row.get('loan_purpose')
            
            if pd.isna(owns_property):
                continue
            
            # If owns property, should have property value
            if owns_property and (pd.isna(property_value) or property_value <= 0):
                violations.append({
                    'row_index': idx,
                    'violation_type': 'missing_property_value',
                    'message': "Property owner should have valid property value",
                    'owns_property': owns_property,
                    'property_value': property_value
                })
            
            # Home improvement loans make more sense for property owners
            if loan_purpose == 'Home Improvement' and not owns_property:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'home_improvement_non_owner',
                    'message': "Home improvement loan for non-property owner",
                    'loan_purpose': loan_purpose,
                    'owns_property': owns_property
                })
        
        return len(violations) == 0, violations
    
    def validate_collateral_adequacy(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate collateral adequacy for secured loans."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            loan_amount = row.get('loan_amount')
            property_value = row.get('property_value')
            owns_property = row.get('owns_property')
            loan_purpose = row.get('loan_purpose')
            
            if not owns_property or pd.isna(property_value) or pd.isna(loan_amount):
                continue
            
            # For property-secured loans, check loan-to-value ratio
            secured_purposes = ['Home Improvement']  # Could expand this list
            
            if loan_purpose in secured_purposes:
                ltv_ratio = loan_amount / property_value
                max_ltv = 0.80  # 80% max loan-to-value
                
                if ltv_ratio > max_ltv:
                    violations.append({
                        'row_index': idx,
                        'violation_type': 'excessive_loan_to_value',
                        'message': f"Loan-to-value ratio ({ltv_ratio:.1%}) exceeds maximum ({max_ltv:.1%})",
                        'ltv_ratio': ltv_ratio,
                        'max_ltv': max_ltv,
                        'loan_amount': loan_amount,
                        'property_value': property_value
                    })
        
        return len(violations) == 0, violations
    
    def validate_employment_income_alignment(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate employment status supports income level."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            employment_status = row.get('employment_status')
            annual_income = row.get('annual_income')
            
            if employment_status is None or pd.isna(annual_income):
                continue
            
            # Unemployed should have very low or zero income
            if employment_status == 'Unemployed' and annual_income > 10000:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'unemployed_high_income',
                    'message': f"Unemployed status with high annual income ({annual_income:.2f})",
                    'employment_status': employment_status,
                    'annual_income': annual_income
                })
            
            # Students typically have lower incomes
            if employment_status == 'Student' and annual_income > 50000:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'student_high_income',
                    'message': f"Student status with high annual income ({annual_income:.2f})",
                    'employment_status': employment_status,
                    'annual_income': annual_income
                })
        
        return len(violations) == 0, violations
    
    def validate_employment_stability(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate employment stability indicators."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            employment_status = row.get('employment_status')
            years_employed = row.get('years_employed')
            age = row.get('age')
            
            if employment_status is None or pd.isna(years_employed) or pd.isna(age):
                continue
            
            # Years employed can't exceed working age
            max_employment_years = max(0, age - 16)  # Assuming work starts at 16
            
            if years_employed > max_employment_years:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'impossible_employment_duration',
                    'message': f"Employment duration ({years_employed} years) exceeds possible maximum ({max_employment_years} years) for age {age}",
                    'years_employed': years_employed,
                    'max_possible': max_employment_years,
                    'age': age
                })
        
        return len(violations) == 0, violations
    
    def validate_banking_relationship_consistency(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate banking relationship consistency."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            has_bank_account = row.get('has_bank_account')
            years_with_bank = row.get('years_with_bank')
            age = row.get('age')
            
            if pd.isna(has_bank_account):
                continue
            
            # If has bank account, should have years with bank
            if has_bank_account and (pd.isna(years_with_bank) or years_with_bank < 0):
                violations.append({
                    'row_index': idx,
                    'violation_type': 'missing_banking_tenure',
                    'message': "Has bank account but missing/invalid years with bank",
                    'has_bank_account': has_bank_account,
                    'years_with_bank': years_with_bank
                })
            
            # Years with bank can't exceed age
            if not pd.isna(years_with_bank) and not pd.isna(age) and years_with_bank > age:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'impossible_banking_tenure',
                    'message': f"Years with bank ({years_with_bank}) exceeds age ({age})",
                    'years_with_bank': years_with_bank,
                    'age': age
                })
        
        return len(violations) == 0, violations
    
    def validate_age_loan_term_alignment(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate loan term is reasonable for applicant age."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            age = row.get('age')
            loan_term = row.get('loan_term_months')
            
            if pd.isna(age) or pd.isna(loan_term):
                continue
            
            # Very long loans for older applicants may be problematic
            loan_term_years = loan_term / 12
            age_at_maturity = age + loan_term_years
            
            if age_at_maturity > 75:  # Reasonable retirement age
                violations.append({
                    'row_index': idx,
                    'violation_type': 'loan_extends_past_retirement',
                    'message': f"Loan extends to age {age_at_maturity:.1f}, past typical retirement age",
                    'current_age': age,
                    'loan_term_years': loan_term_years,
                    'age_at_maturity': age_at_maturity
                })
        
        return len(violations) == 0, violations
    
    def validate_coapplicant_data_consistency(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[bool, List[Dict]]:
        """Validate co-applicant data consistency."""
        violations = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        for idx, row in data.iterrows():
            has_coapplicant = row.get('has_coapplicant')
            coapplicant_income = row.get('coapplicant_income')
            
            if pd.isna(has_coapplicant):
                continue
            
            # If no co-applicant, should have zero or null co-applicant income
            if not has_coapplicant and not pd.isna(coapplicant_income) and coapplicant_income > 0:
                violations.append({
                    'row_index': idx,
                    'violation_type': 'coapplicant_income_without_coapplicant',
                    'message': f"Co-applicant income ({coapplicant_income}) reported without co-applicant",
                    'has_coapplicant': has_coapplicant,
                    'coapplicant_income': coapplicant_income
                })
            
            # If has co-applicant, should have positive co-applicant income
            if has_coapplicant and (pd.isna(coapplicant_income) or coapplicant_income <= 0):
                violations.append({
                    'row_index': idx,
                    'violation_type': 'missing_coapplicant_income',
                    'message': "Has co-applicant but missing/zero co-applicant income",
                    'has_coapplicant': has_coapplicant,
                    'coapplicant_income': coapplicant_income
                })
        
        return len(violations) == 0, violations