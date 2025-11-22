from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import PropTechConfig
from ..data_loader import PropTechDataLoader
from ..feature_engineering import build_review_corpus
from ..models import RentalDemandForecaster, PropertyRecommender, ReviewSummarizer, PropertyChatbot

app = FastAPI(title="PropTech AI", description="AI services for rental/property management", version="0.1.0")

_BASE_CONFIG = PropTechConfig()
_loader = PropTechDataLoader(_BASE_CONFIG)


class ForecastRequest(BaseModel):
    history: List[float] = Field(..., min_length=3, description="Recent rent history for the property")
    horizon: int = Field(7, ge=1, le=30)
    property_id: Optional[int] = Field(None, description="Property to forecast; defaults to top revenue")
    epochs: Optional[int] = Field(None, ge=1, le=25)


class ForecastResponse(BaseModel):
    property_id: int
    horizon: int
    predictions: List[float]
    artifact_path: str


class RecommendRequest(BaseModel):
    tenant_id: int
    top_k: int = Field(5, ge=1, le=20)


class RecommendationItem(BaseModel):
    property_id: int
    score: float


class RecommendResponse(BaseModel):
    tenant_id: int
    recommendations: List[RecommendationItem]
    artifact_path: str


class SummarizeRequest(BaseModel):
    reviews: Optional[List[str]] = None
    max_chars: int = Field(512, ge=64, le=2048)


class SummarizeResponse(BaseModel):
    summary: str
    used_reviews: int


class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    used_context: bool


class HealthResponse(BaseModel):
    transactions: int
    properties: int
    reviews: int
    config: dict


@app.on_event("startup")
def _startup() -> None:
    _loader.ensure_data()


def _clone_config(**overrides) -> PropTechConfig:
    cfg = replace(_BASE_CONFIG)
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    tx = _loader.load_transactions()
    props = _loader.load_properties()
    reviews = _loader.load_reviews()
    return HealthResponse(
        transactions=len(tx),
        properties=len(props),
        reviews=len(reviews),
        config=_BASE_CONFIG.to_dict(),
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    tx = _loader.load_transactions()
    if request.property_id is not None and request.property_id not in tx["property_id"].unique():
        raise HTTPException(status_code=404, detail="Unknown property_id")
    property_id = request.property_id or int(tx.groupby("property_id")["rent_amount"].sum().idxmax())

    cfg = _clone_config(forecast_horizon=request.horizon, forecast_epochs=request.epochs)
    forecaster = RentalDemandForecaster(cfg)
    artifact_path = forecaster.fit(tx, property_id=property_id, epochs=cfg.forecast_epochs)

    history = request.history
    required = cfg.sequence_lookback
    if len(history) < required:
        fallback = (
            tx.loc[tx["property_id"] == property_id]
            .sort_values("timestamp")
            ["rent_amount"]
            .tail(required)
            .tolist()
        )
        history = fallback
    preds = forecaster.forecast(history[-required:], horizon=cfg.forecast_horizon)
    return ForecastResponse(
        property_id=property_id,
        horizon=cfg.forecast_horizon,
        predictions=np.round(preds, 2).tolist(),
        artifact_path=str(artifact_path),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    tx = _loader.load_transactions()
    if request.tenant_id not in tx["tenant_id"].unique():
        raise HTTPException(status_code=404, detail="Unknown tenant_id")
    cfg = _clone_config()
    recommender = PropertyRecommender(cfg)
    artifact_path = recommender.fit(tx)
    recs = recommender.recommend(request.tenant_id, top_k=request.top_k)
    payload = [RecommendationItem(property_id=r.property_id, score=round(r.score, 4)) for r in recs]
    return RecommendResponse(tenant_id=request.tenant_id, recommendations=payload, artifact_path=str(artifact_path))


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    reviews_df = _loader.load_reviews()
    texts = request.reviews or build_review_corpus(reviews_df.head(request.max_chars))
    summarizer = ReviewSummarizer()
    summary = summarizer.summarize(texts, max_chars=request.max_chars)
    return SummarizeResponse(summary=summary.summary_text, used_reviews=summary.num_inputs)


@app.post("/chatbot", response_model=ChatResponse)
def chatbot(request: ChatRequest) -> ChatResponse:
    chatbot = PropertyChatbot()
    resp = chatbot.answer(request.question, context=request.context)
    return ChatResponse(answer=resp.answer, used_context=resp.used_context)
