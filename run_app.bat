@echo off
echo Starting Loan Eligibility Prediction System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

echo Python detected. Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo Starting Streamlit application...
echo.
echo The application will open in your default web browser
echo Press Ctrl+C to stop the application
echo.

streamlit run loan_prediction_app.py

pause