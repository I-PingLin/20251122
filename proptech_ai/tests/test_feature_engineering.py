import numpy as np
import pandas as pd
import pytest

from proptech_ai.feature_engineering import build_interaction_matrix, build_review_corpus, build_rental_timeseries, sliding_window


def test_sliding_window():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X, y = sliding_window(series, lookback=3, horizon=2)
    assert X.shape == (6, 3)
    assert y.shape == (6, 2)
    np.testing.assert_array_equal(X[0], [1, 2, 3])
    np.testing.assert_array_equal(y[0], [4, 5])


def test_build_rental_timeseries():
    tx = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "property_id": [1, 2, 1],
            "rent_amount": [10000, 12000, 11000],
        }
    )
    df = build_rental_timeseries(tx)
    assert 1 in df.columns
    assert 2 in df.columns
    assert df.loc[pd.Timestamp("2024-01-01"), 1] == 10000.0


def test_build_interaction_matrix():
    tx = pd.DataFrame(
        {
            "tenant_id": [1, 1, 2, 2],
            "property_id": [1, 2, 1, 2],
            "rent_amount": [10000, 12000, 11000, 13000],
        }
    )
    matrix, artifacts = build_interaction_matrix(tx)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 10000.0
    assert matrix[0, 1] == 12000.0
    assert artifacts.tenant_mapping == {1: 0, 2: 1}
    assert artifacts.property_mapping == {1: 0, 2: 1}


def test_build_review_corpus():
    reviews = pd.DataFrame(
        {
            "review_text": ["好", "交通便利", None, ""],
        }
    )
    corpus = build_review_corpus(reviews)
    assert corpus == ["好", "交通便利"]
