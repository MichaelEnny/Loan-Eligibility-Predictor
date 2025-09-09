"""
FastAPI Backend for Loan Eligibility Prediction
Serves trained ML models and handles prediction requests from the React frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add models directory to path
sys.path.append('models')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Loan Eligibility Predictor API",
    description="AI-powered loan eligibility prediction system",
    version="1.0.0"
)

# Add CORS middleware to allow React frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None
    probability: Optional[float] = None
    message: str

# Global variables to store loaded models
loaded_model = None
feature_pipeline = None

def load_models():
    """Load the trained ML model and feature pipeline"""
    global loaded_model, feature_pipeline
    
    try:
        # Try to load the latest model from model registry
        model_registry_path = Path("test_registry")
        
        if model_registry_path.exists():
            # Look for the latest model
            model_paths = list(model_registry_path.glob("models/*/model.pkl"))
            if model_paths:
                latest_model_path = max(model_paths, key=lambda p: p.parent.stat().st_mtime)
                logger.info(f"Loading model from: {latest_model_path}")
                
                # Load and inspect the model
                model_data = joblib.load(latest_model_path)
                logger.info(f"Loaded model type: {type(model_data)}")
                
                # Handle different model storage formats
                if hasattr(model_data, 'predict'):
                    # Direct model object
                    loaded_model = model_data
                elif isinstance(model_data, dict):
                    # Model stored in dictionary format
                    if 'model' in model_data:
                        loaded_model = model_data['model']
                    elif 'trained_model' in model_data:
                        loaded_model = model_data['trained_model']
                    else:
                        # Try to find any sklearn-like object in the dict
                        for key, value in model_data.items():
                            if hasattr(value, 'predict'):
                                loaded_model = value
                                logger.info(f"Found model in key: {key}")
                                break
                        else:
                            logger.error(f"No model found in dictionary keys: {list(model_data.keys())}")
                            return False
                else:
                    loaded_model = model_data
                
                # Verify the loaded model has predict method
                if not hasattr(loaded_model, 'predict'):
                    logger.error(f"Loaded object doesn't have predict method: {type(loaded_model)}")
                    return False
                
                # Try to load scaler if available
                scaler_path = latest_model_path.parent / "model_scaler.pkl"
                if scaler_path.exists():
                    feature_pipeline = joblib.load(scaler_path)
                    logger.info("Loaded feature scaler")
                
                logger.info(f"Models loaded successfully. Model type: {type(loaded_model)}")
                return True
        
        logger.warning("No trained models found in registry")
        return False
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def preprocess_input(data: LoanApplication) -> pd.DataFrame:
    """Preprocess input data to match model training format"""
    
    # Convert input data to the model's expected feature format
    # Map frontend fields to model features
    
    # Extract values from the application
    application_data = data.dict()
    
    # Create feature mappings based on the model's expected features
    feature_data = {}
    
    # Map categorical variables to scores
    gender_score = 1 if application_data['Gender'] == 'Male' else 0
    married_score = 1 if application_data['Married'] == 'Yes' else 0
    education_score = 1 if application_data['Education'] == 'Graduate' else 0
    employment_score = 1 if application_data['Self_Employed'] == 'Yes' else 0
    
    # Handle dependents
    dependents = application_data['Dependents']
    if dependents == '3+':
        dependents = 3
    else:
        dependents = int(dependents)
    
    # Create synthetic credit score based on inputs
    # Higher income + good credit history + education = better score
    base_score = 650
    income_factor = min((application_data['ApplicantIncome'] + application_data['CoapplicantIncome']) / 100000, 2) * 100
    credit_factor = application_data['Credit_History'] * 50
    education_factor = education_score * 25
    feature_data['credit_score'] = base_score + income_factor + credit_factor + education_factor
    
    # Map other features
    feature_data['annual_income'] = application_data['ApplicantIncome'] + application_data['CoapplicantIncome']
    feature_data['loan_amount'] = application_data['LoanAmount']
    
    # Calculate debt-to-income ratio
    total_income = feature_data['annual_income']
    if total_income > 0:
        feature_data['debt_to_income_ratio'] = (application_data['LoanAmount'] * 12) / total_income
    else:
        feature_data['debt_to_income_ratio'] = 0
    
    # Estimate years employed (synthetic based on age estimation)
    estimated_age = 30 + (feature_data['annual_income'] - 40000) / 10000  # Rough age estimation
    feature_data['years_employed'] = max(1, estimated_age - 22)  # Assume work started at 22
    feature_data['age'] = estimated_age
    
    # Calculate monthly debt payments
    loan_term_years = application_data['Loan_Amount_Term'] / 12
    if loan_term_years > 0:
        feature_data['monthly_debt_payments'] = application_data['LoanAmount'] / application_data['Loan_Amount_Term']
    else:
        feature_data['monthly_debt_payments'] = 0
        
    feature_data['number_of_dependents'] = dependents
    
    # Property value estimation based on area and loan amount
    area_multipliers = {'Urban': 1.5, 'Semiurban': 1.2, 'Rural': 1.0}
    area_multiplier = area_multipliers.get(application_data['Property_Area'], 1.0)
    feature_data['property_value'] = application_data['LoanAmount'] * 1.5 * area_multiplier
    
    feature_data['loan_term'] = application_data['Loan_Amount_Term']
    
    # Score categorical variables
    feature_data['employment_status_score'] = employment_score
    feature_data['education_level_score'] = education_score  
    feature_data['marital_status_score'] = married_score
    feature_data['payment_history_score'] = application_data['Credit_History']
    
    # Account balance estimation
    feature_data['account_balance_score'] = min(feature_data['annual_income'] / 50000, 2.0)
    
    # Create the expected feature order
    expected_features = [
        'credit_score', 'annual_income', 'loan_amount', 'debt_to_income_ratio',
        'years_employed', 'age', 'monthly_debt_payments', 'number_of_dependents',
        'property_value', 'loan_term', 'employment_status_score', 
        'education_level_score', 'marital_status_score', 'payment_history_score',
        'account_balance_score'
    ]
    
    # Create DataFrame with the expected features in the correct order
    df_data = []
    for feature in expected_features:
        df_data.append(feature_data[feature])
    
    df = pd.DataFrame([df_data], columns=expected_features)
    
    # Ensure all values are numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)  # Handle any NaN values
    
    logger.info(f"Created feature vector: {df.iloc[0].to_dict()}")
    
    return df

@app.on_event("startup")
async def startup_event():
    """Load models when the API starts"""
    success = load_models()
    if not success:
        logger.warning("API started without trained models - predictions will not work")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Loan Eligibility Predictor API",
        "status": "running",
        "model_loaded": loaded_model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api_status": "healthy",
        "model_loaded": loaded_model is not None,
        "feature_pipeline_loaded": feature_pipeline is not None
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_loan_eligibility(application: LoanApplication):
    """
    Predict loan eligibility based on applicant information
    """
    
    if loaded_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        # Preprocess the input data
        processed_data = preprocess_input(application)
        logger.info(f"Processed input shape: {processed_data.shape}")
        
        # Make prediction
        prediction = loaded_model.predict(processed_data)[0]
        
        # Get prediction probability if available
        probability = None
        confidence = None
        
        if hasattr(loaded_model, 'predict_proba'):
            probabilities = loaded_model.predict_proba(processed_data)[0]
            probability = float(probabilities[1])  # Probability of approval
            confidence = float(max(probabilities))  # Confidence is max probability
        
        # Convert prediction to human readable format
        prediction_text = "Y" if prediction == 1 else "N"
        
        # Create response message
        if prediction_text == "Y":
            message = "Congratulations! Your loan application is likely to be approved."
        else:
            message = "Unfortunately, your loan application may not be approved based on the provided information."
        
        return PredictionResponse(
            prediction=prediction_text,
            confidence=confidence,
            probability=probability,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.post("/api/batch_predict")
async def batch_predict(applications: list[LoanApplication]):
    """
    Predict loan eligibility for multiple applications
    """
    
    if loaded_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        results = []
        for app in applications:
            processed_data = preprocess_input(app)
            prediction = loaded_model.predict(processed_data)[0]
            prediction_text = "Y" if prediction == 1 else "N"
            
            probability = None
            if hasattr(loaded_model, 'predict_proba'):
                probabilities = loaded_model.predict_proba(processed_data)[0]
                probability = float(probabilities[1])
            
            results.append({
                "prediction": prediction_text,
                "probability": probability
            })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making batch predictions: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI app
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )