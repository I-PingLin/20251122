from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .config import PropTechConfig
from .data_loader import PropTechDataLoader
from .feature_engineering import build_review_corpus
from .models import RentalDemandForecaster, PropertyRecommender, ReviewSummarizer, PropertyChatbot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _configure(args: argparse.Namespace) -> PropTechConfig:
    config = PropTechConfig()
    if args.lookback:
        config.sequence_lookback = args.lookback
    if args.horizon:
        config.forecast_horizon = args.horizon
    if args.epochs:
        config.forecast_epochs = args.epochs
    if args.batch_size:
        config.forecast_batch_size = args.batch_size
    if args.components:
        config.recommender_components = args.components
    return config


def run_pipeline(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="PropTech AI orchestration")
    parser.add_argument("--property-id", type=int, default=None, help="Property ID to forecast")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--lookback", type=int, default=None, help="Sequence length for LSTM")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon")
    parser.add_argument("--batch-size", type=int, default=None, help="Forecast batch size")
    parser.add_argument("--components", type=int, default=None, help="Latent factors for ALS recommender")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations to preview")
    parser.add_argument(
        "--max-reviews", type=int, default=50, help="How many reviews to summarize for sample artifact"
    )
    if args is None:
        parsed = parser.parse_args()
    else:
        parsed = args
    config = _configure(parsed)

    loader = PropTechDataLoader(config)
    loader.ensure_data()
    tx = loader.load_transactions()
    reviews = loader.load_reviews()

    # --------------------- Rental Demand Forecasting ---------------------
    if parsed.property_id is not None:
        property_id = parsed.property_id
    else:
        property_id = int(tx.groupby("property_id")["rent_amount"].sum().idxmax())
    logger.info("Training LSTM forecaster for property_id=%s", property_id)
    forecaster = RentalDemandForecaster(config)
    forecast_artifact = forecaster.fit(tx, property_id=property_id, epochs=config.forecast_epochs)
    last_values = tx.loc[tx["property_id"] == property_id].sort_values("timestamp")["rent_amount"].tail(
        config.sequence_lookback
    )
    preview = forecaster.forecast(last_values, horizon=config.forecast_horizon)
    logger.info("Sample forecast: %s", np.round(preview, 2).tolist())

    # --------------------- Property Recommendations ------------------------
    logger.info("Training matrix factorization recommender")
    recommender = PropertyRecommender(config)
    rec_artifact = recommender.fit(tx)
    sample_tenant = int(tx["tenant_id"].mode()[0])
    recs = recommender.recommend(sample_tenant, top_k=parsed.top_k)
    logger.info("Top-%s recommendations for tenant %s: %s", parsed.top_k, sample_tenant, recs)

    # --------------------- NLP Summaries --------------------------
    logger.info("Summarizing %s tenant reviews", parsed.max_reviews)
    summarizer = ReviewSummarizer()
    corpus = build_review_corpus(reviews.head(parsed.max_reviews))
    summary = summarizer.summarize(corpus)
    summary_path = config.artifact_dir / "review_summary.txt"
    summary_path.write_text(summary.summary_text, encoding="utf-8")
    logger.info("Saved review summary to %s", summary_path)

    # --------------------- Chatbot Demo --------------------------
    logger.info("Running chatbot demo")
    chatbot = PropertyChatbot()
    demo_q = "這地區交通方便嗎？"
    resp = chatbot.answer(demo_q)
    logger.info("Q: %s | A: %s", demo_q, resp.answer)

    logger.info("Artifacts saved: %s, %s, %s", forecast_artifact, rec_artifact, summary_path)


if __name__ == "__main__":
    run_pipeline()
