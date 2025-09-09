🚀 Manual Setup Guide: Step-by-Step Instructions

📋 Prerequisites:

- Python 3.8+ installed
- Node.js and npm installed
- All project files in the correct directory

  ---
🔧 STEP 1: Set Up the Backend (Python API)

1.1 Open Terminal/Command Prompt

# Navigate to your project directory
cd "C:\Users\wisdo\OneDrive\Desktop\Data-Science-ML\Loan-Eligibility-Predictor"

1.2 Install Python Dependencies

# Install required Python packages
pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib

1.3 Start the Backend Server

# Start the FastAPI backend on port 8000
python app.py

Expected Output:
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:app:Loading model from: test_registry\models\test_xgboost_v...
INFO:app:Models loaded successfully. Model type: <class 'xgboost.sklearn.XGBClassifier'>
INFO:     Application startup complete.

✅ Backend Ready: Keep this terminal open - your API is now running on http://localhost:8000

  ---
🌐 STEP 2: Set Up the Frontend (React App)

2.1 Open New Terminal/Command Prompt

# Navigate to the frontend directory
cd "C:\Users\wisdo\OneDrive\Desktop\Data-Science-ML\Loan-Eligibility-Predictor\frontend"

2.2 Install Node.js Dependencies (if needed)

# Install React dependencies
npm install

2.3 Start the Frontend Server

# Start the React development server
npm start

Expected Output:
> loan-eligibility-frontend@0.1.0 start
> react-scripts start

Compiled successfully!

You can now view loan-eligibility-frontend in the browser.

    Local:            http://localhost:3003
    On Your Network:  http://192.168.x.x:3003

✅ Frontend Ready: Your React app is now running on http://localhost:3003

  ---
🎯 STEP 3: Access and Use the Application

3.1 Open Your Web Browser

Navigate to: http://localhost:3003

3.2 Fill Out the Loan Application Form

Enter realistic data:

Personal Information:
- Gender: Male/Female
- Married: Yes/No
- Dependents: 0, 1, 2, or 3+
- Education: Graduate/Not Graduate
- Self Employed: Yes/No

Financial Information:
- Applicant Income: e.g., 50000
- Coapplicant Income: e.g., 20000 (can be 0)
- Loan Amount: e.g., 300000
- Loan Term: e.g., 360 (months)
- Credit History: Good/Poor

Property Information:
- Property Area: Urban/Semiurban/Rural

3.3 Get Your Prediction

1. Click "Predict Eligibility"
2. Wait for AI processing (1-2 seconds)
3. See results with confidence score!

  ---
📝 Alternative: Create Batch Files for Easy Startup

Let me help you create convenient batch files:

Backend Startup Script (start_backend.bat):

@echo off
echo Starting Loan Eligibility Predictor Backend...
cd /d "C:\Users\wisdo\OneDrive\Desktop\Data-Science-ML\Loan-Eligibility-Predictor"
echo Installing dependencies...
pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib
echo Starting FastAPI server on port 8000...
python app.py
pause

Frontend Startup Script (start_frontend.bat):

@echo off
echo Starting Loan Eligibility Predictor Frontend...
cd /d "C:\Users\wisdo\OneDrive\Desktop\Data-Science-ML\Loan-Eligibility-Predictor\frontend"
echo Installing dependencies...
npm install
echo Starting React development server...
npm start
pause

Complete App Startup Script (run_app.bat):

@echo off
title Loan Eligibility Predictor
echo ========================================
echo   LOAN ELIGIBILITY PREDICTOR STARTUP
echo ========================================
echo.

echo 1. Starting Backend (Python API)...
cd /d "C:\Users\wisdo\OneDrive\Desktop\Data-Science-ML\Loan-Eligibility-Predictor"
start "Backend Server" cmd /k "python app.py"

echo 2. Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo 3. Starting Frontend (React App)...
cd /d "C:\Users\wisdo\OneDrive\Desktop\Data-Science-ML\Loan-Eligibility-Predictor\frontend"
start "Frontend Server" cmd /k "npm start"

echo 4. Waiting 10 seconds for frontend to start...
timeout /t 10 /nobreak >nul

echo 5. Opening browser...
start http://localhost:3003

echo.
echo ========================================
echo   APPLICATION STARTUP COMPLETE!
echo ========================================
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:3003
echo.
echo Press any key to exit...
pause >nul

  ---
🔍 Troubleshooting Guide:

If Backend Fails to Start:

# Check Python version
python --version

# Reinstall dependencies
pip install --upgrade fastapi uvicorn pandas numpy scikit-learn xgboost joblib

# Check if port 8000 is free
netstat -an | findstr :8000

If Frontend Fails to Start:

# Check Node.js version
node --version
npm --version

# Clear npm cache and reinstall
npm cache clean --force
rm -rf node_modules
npm install

# Check if port 3003 is free
netstat -an | findstr :3003

If Prediction Fails:

1. Check that both services are running
2. Check browser developer console for errors
3. Test backend directly: curl http://localhost:8000/health

  ---
🎯 Quick Start Summary:

1. Terminal 1: cd project && python app.py
2. Terminal 2: cd project/frontend && npm start
3. Browser: Go to http://localhost:3003
4. Use: Fill form → Click "Predict" → Get results!

That's it! Your AI-powered loan predictor is ready to use! 🚀✨
