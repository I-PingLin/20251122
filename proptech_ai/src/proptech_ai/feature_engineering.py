from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PropTechConfig


@dataclass
class FeatureArtifacts:
    property_mapping: dict[int, int]
    tenant_mapping: dict[int, int]
    inverse_property_mapping: dict[int, int]
    inverse_tenant_mapping: dict[int, int]


def build_rental_timeseries(
    transactions: pd.DataFrame,
    config: PropTechConfig | None = None,
) -> pd.DataFrame:
    """Aggregate daily rent per property."""
    _ = config or DEFAULT_CONFIG
    frame = transactions.copy()
    frame["date"] = frame["timestamp"].dt.floor("D")
    daily = (
        frame.groupby(["date", "property_id"], as_index=False)["rent_amount"]
        .sum()
        .sort_values("date")
    )
    pivot = daily.pivot(index="date", columns="property_id", values="rent_amount").fillna(0.0)
    return pivot


def sliding_window(
    series: pd.Series,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    values = series.to_numpy(dtype=np.float32)
    if len(values) <= lookback + horizon:
        raise ValueError("Time series is shorter than lookback + horizon window")
    X, y = [], []
    for idx in range(lookback, len(values) - horizon + 1):
        X.append(values[idx - lookback : idx])
        y.append(values[idx : idx + horizon])
    return np.stack(X), np.stack(y)


def build_interaction_matrix(
    transactions: pd.DataFrame,
    min_rating: float = 1.0,
) -> Tuple[np.ndarray, FeatureArtifacts]:
    frame = transactions.copy()
    frame["rating"] = frame.groupby(["tenant_id", "property_id"])["rent_amount"].transform("sum")
    frame = frame[["tenant_id", "property_id", "rating"]].drop_duplicates()
    frame["rating"] = frame["rating"].clip(lower=min_rating)

    tenants = sorted(frame["tenant_id"].unique())
    properties = sorted(frame["property_id"].unique())
    tenant_map = {tid: idx for idx, tid in enumerate(tenants)}
    prop_map = {pid: idx for idx, pid in enumerate(properties)}
    inv_tenant = {idx: tid for tid, idx in tenant_map.items()}
    inv_prop = {idx: pid for pid, idx in prop_map.items()}

    matrix = np.zeros((len(tenants), len(properties)), dtype=np.float32)
    for row in frame.itertuples():
        matrix[tenant_map[row.tenant_id], prop_map[row.property_id]] = float(row.rating)

    return matrix, FeatureArtifacts(prop_map, tenant_map, inv_prop, inv_tenant)


def build_review_corpus(reviews: pd.DataFrame) -> list[str]:
    return [str(text) for text in reviews["review_text"].fillna("") if text]
