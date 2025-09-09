# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Python Backend/API
- **Start API Server**: `python start_api.py` or `python -m uvicorn api.app:app --host 0.0.0.0 --port 8000`
- **Run Tests**: `pytest` or `python run_tests.py`
- **Install Dependencies**: `pip install -r requirements.txt`
- **Docker Build**: `docker build -t loan-api .`
- **Docker Compose**: `docker-compose up -d`

### React Frontend
- **Start Development**: `cd frontend && npm start`
- **Build Production**: `cd frontend && npm run build`
- **Run Tests**: `cd frontend && npm test`
- **Install Dependencies**: `cd frontend && npm install`

### ML Model Training
- **Train Ensemble Models**: `python train_ensemble_models.py`
- **Advanced Training**: `python advanced_ensemble_training.py`
- **Run ML Tests**: `python test_ml_infrastructure.py`

## Architecture Overview

This is a **loan eligibility prediction system** with a comprehensive ML pipeline and production-ready infrastructure.

### High-Level Architecture
```
Frontend (React) → API (FastAPI) → ML Models → Data Pipeline
     ↓               ↓              ↓           ↓
   Dashboard    REST Endpoints   Ensemble     Validation
   Analytics    Authentication    Models      Monitoring
```

### Key Components

**1. ML Infrastructure** (`models/`, `validation/`, `preprocessing/`)
- Ensemble models combining Random Forest, XGBoost, Neural Networks
- Model registry with versioning and evaluation metrics
- Validation framework with schema, business rules, and data quality checks
- Feature engineering pipeline with automated preprocessing

**2. API Layer** (`api/`)
- FastAPI with structured logging, rate limiting, authentication
- Comprehensive health checks, metrics collection (Prometheus)
- Async prediction endpoints with batch processing support
- Model management and monitoring endpoints

**3. Frontend** (`frontend/`)
- React with TypeScript, Material-UI components
- Redux Toolkit for state management
- Prediction dashboard with confidence visualization
- Settings, reports, and model management interfaces

**4. Production Infrastructure**
- Multi-stage Docker builds with security best practices
- Docker Compose with Redis, PostgreSQL, Prometheus, Grafana
- Structured logging with optional ELK stack integration
- Rate limiting, CORS, and security middleware

### Data Flow
1. **Input Validation**: Schema validation → Business rules → Data quality checks
2. **Feature Engineering**: Automated preprocessing with configurable pipelines
3. **Prediction**: Ensemble voting across multiple ML models
4. **Output**: Risk score + recommendation + confidence + explanation
5. **Monitoring**: Performance tracking, bias detection, drift monitoring

### Key Directories
- `api/`: FastAPI application with services, middleware, models
- `frontend/`: React TypeScript application
- `models/`: ML model implementations and ensemble logic
- `validation/`: Input validation and data quality framework
- `preprocessing/`: Feature engineering and data transformation
- `trained_models/`: Serialized model artifacts
- `bias_detection_output/`: Fairness monitoring outputs
- `data_quality_monitoring/`: Data drift and quality reports

### Configuration
- **Environment**: `.env` file for API configuration
- **Frontend Config**: `frontend/package.json` for React app settings  
- **Docker**: `docker-compose.yml` for full stack deployment
- **Requirements**: `requirements.txt` for Python dependencies

### Testing Strategy
- Unit tests for ML models and API endpoints
- Integration tests for end-to-end workflows
- Performance benchmarks for prediction latency
- Data validation tests for input quality

The system is designed for production use with comprehensive monitoring, security, and scalability features built-in.