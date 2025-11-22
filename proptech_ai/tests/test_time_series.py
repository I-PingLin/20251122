import numpy as np
import pandas as pd
import pytest

from proptech_ai.models.time_series import RentalDemandForecaster


def test_forecaster_fit_and_predict():
    # Create dummy transaction data
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    tx = pd.DataFrame(
        {
            "timestamp": dates,
            "property_id": 1,
            "rent_amount": np.random.uniform(8000, 12000, size=30),
        }
    )
    forecaster = RentalDemandForecaster()
    artifact_path = forecaster.fit(tx, property_id=1, epochs=1)
    assert artifact_path.exists()
    # Predict using the last lookback values
    history = tx["rent_amount"].tail(14).tolist()
    preds = forecaster.forecast(history, horizon=3)
    assert preds.shape == (3,)
    assert np.all(preds >= 0)
