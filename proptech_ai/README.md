# PropTech AI Platform

A PropTech AI showcase for **創未來智慧生活股份有限公司** demonstrating end-to-end ML/NLP pipelines, time-series forecasting, recommendation systems, and LLM-powered services for rental/property management.

## Core Features

- **Rental Demand Forecasting (Time Series / PyTorch)**: LSTM forecaster for per-property rental demand and pricing trends.
- **Property Recommendation Engine**: Matrix factorization to match tenants with properties based on preferences and transaction history.
- **NLP & LLM Services**: Hugging Face summarization of tenant reviews and a lightweight chatbot for property Q&A.
- **Data Engineering**: Synthetic property listings, rental transactions, and review datasets with configurable quality checks.
- **FastAPI Microservice**: REST endpoints for forecasting, recommendations, review summarization, and chatbot interactions.
- **Testing & Observability**: Pytest coverage, structured logging, and Docker/CI setup.

## Architecture

```
proptech_ai/
├── README.md
├── requirements.txt
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/                     # Synthetic datasets
├── artifacts/                # Model weights & reports
└── src/
    └── proptech_ai/
        ├── __init__.py
        ├── config.py
        ├── data_loader.py
        ├── feature_engineering.py
        ├── models/
        │   ├── __init__.py
        │   ├── time_series.py
        │   ├── recommender.py
        │   └── nlp.py
        ├── pipeline.py
        └── api/
            ├── __init__.py
            └── main.py
└── tests/
    ├── __init__.py
    ├── test_feature_engineering.py
    ├── test_time_series.py
    └── test_recommender.py
```

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r proptech_ai/requirements.txt
pip install -e proptech_ai
python -m proptech_ai.pipeline --epochs 5 --forecast-horizon 7
uvicorn proptech_ai.api.main:app --reload
```

### API Endpoints

- `GET /health` – dataset sizes + configuration
- `POST /forecast` – property rental demand forecast
- `POST /recommend` – tenant-to-property recommendations
- `POST /summarize` – tenant review summarization
- `POST /chatbot` – LLM-powered property Q&A

## Why This Matches the Job

| JD Requirement | Implementation |
| --- | --- |
| 2+ years ML/AI, Python, TensorFlow/PyTorch/Scikit-learn | PyTorch LSTM forecaster + scikit-learn pipelines |
| Recommendation systems / Time Series | ALS recommender + LSTM demand forecasting |
| NLP / LLM deployment | Hugging Face summarizer + chatbot |
| Data cleaning & feature engineering | Synthetic data generator + feature pipelines |
| RESTful APIs / Microservices | FastAPI with typed schemas |
| Llama/MiniCPM familiarity | Configurable model loading for HF models |
| LLM/Chatbot development | Simple chatbot endpoint with prompt handling |

## Next Steps

- Add OAuth/JWT for production auth
- Containerize with Docker + CI
- Connect to real PropTech data sources
- Deploy to cloud (Azure/AWS/ECS)
