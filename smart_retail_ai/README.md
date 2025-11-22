# Smart Retail AI Platform

A full-stack AI engineering showcase tailored to the requirements of **創未來智慧生活股份有限公司** for the AI Engineer role on 104. It demonstrates hands-on mastery of ML/NLP model development, deployment-ready APIs, and data engineering best practices.

## Highlights

- **Demand Forecasting (Time Series / PyTorch)**: LSTM forecaster with configurable lookback, horizon, and metrics (MAE/MAPE) for per-product demand tracking.
- **Personalized Recommendations**: Matrix factorization recommender trained from user-product transaction logs with explainable scores.
- **NLP & LLM Integration**: Hugging Face summarization pipeline to distill unstructured product reviews into actionable insights.
- **Data Engineering**: Synthetic-yet-realistic data generator, data quality checks, feature pipelines, and automated artifact storage.
- **FastAPI Microservice**: REST endpoints for forecast, recommendation, review summarization, and system health—deployable behind any ASGI server.
- **Testing & Observability**: Pytest coverage for core utilities plus structured logging and configuration management.

## Project Layout

```
smart_retail_ai/
├── README.md
├── requirements.txt
├── data/                  # Auto-populated synthetic datasets + docs
├── artifacts/             # Saved model weights & generated reports
├── src/
│   └── smart_retail_ai/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── feature_engineering.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── time_series.py
│       │   ├── recommender.py
│       │   └── nlp.py
│       ├── pipeline.py
│       └── api/
│           ├── __init__.py
│           └── main.py
└── tests/
    ├── __init__.py
    ├── test_feature_engineering.py
    ├── test_time_series.py
    └── test_recommender.py
```

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r smart_retail_ai/requirements.txt
pip install -e smart_retail_ai          # uses pyproject (editable install)
python -m smart_retail_ai.pipeline --epochs 5 --forecast-horizon 7
uvicorn smart_retail_ai.api.main:app --reload
```

Then interact with the API:

```bash
curl -X POST http://127.0.0.1:8000/forecast -H "Content-Type: application/json" \
  -d '{"history": [120, 118, 130, 140, 150], "horizon": 3}'
```

### Testing & Tooling

```bash
pytest smart_retail_ai/tests
```

Key FastAPI endpoints once the server is live:

1. `GET /health` – dataset sizes + configuration snapshot.
2. `POST /forecast` – train+serve a product-level demand forecast.
3. `POST /recommend` – ALS recommender results for a given user.
4. `POST /summarize` – distill raw review text via HF summarization.

## Why This Matches the Job Description

| JD Requirement | How the project addresses it |
| --- | --- |
| 2+ years ML/AI, Python, TensorFlow/PyTorch/Scikit-learn | PyTorch LSTM forecaster + scikit-learn-based preprocessing & metrics. |
| Recommendation systems / Time Series Analysis | Dedicated recommender + LSTM demand forecasting modules. |
| NLP / LLM deployment | Hugging Face summarizer with REST exposure for downstream use. |
| Data cleaning & feature engineering, unstructured data | `data_loader` + `feature_engineering` handle structured (transactions) & unstructured (reviews). |
| RESTful APIs / Microservices | FastAPI app with typed schemas and health/ops endpoints. |
| SQL design experience (data modeling) | Feature pipeline mimics dimensional modeling via aggregated fact tables. |

## Next Steps

1. Extend the FastAPI layer with OAuth/JWT for production-grade auth.
2. Containerize with Docker + CI to run lint/tests and push images.
3. Connect to real transactional data sources (Snowflake, BigQuery, etc.) via the `RetailDataLoader` abstraction.
4. Deploy to Azure Container Apps or AWS ECS with autoscaling.
