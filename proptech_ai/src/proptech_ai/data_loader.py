from __future__ import annotations

import json
from dataclasses import asdict
from typing import Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PropTechConfig


class PropTechDataLoader:
    """Generate and load synthetic PropTech datasets."""

    def __init__(self, config: PropTechConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.rng = np.random.default_rng(self.config.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_data(self) -> None:
        """Create datasets on first run so the project is self-contained."""
        tx_path = self.config.data_dir / "transactions.parquet"
        prop_path = self.config.data_dir / "properties.parquet"
        review_path = self.config.data_dir / "reviews.jsonl"

        if not tx_path.exists():
            transactions = self._generate_transactions(num_tenants=300, num_properties=60, days=150)
            transactions.to_parquet(tx_path, index=False)
        if not prop_path.exists():
            properties = self._generate_properties(num_properties=60)
            properties.to_parquet(prop_path, index=False)
        if not review_path.exists():
            reviews = self._generate_reviews(num_tenants=200, num_properties=60, samples=600)
            with review_path.open("w", encoding="utf-8") as f:
                for row in reviews:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load_transactions(self) -> pd.DataFrame:
        self.ensure_data()
        tx_path = self.config.data_dir / "transactions.parquet"
        frame = pd.read_parquet(tx_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def load_properties(self) -> pd.DataFrame:
        self.ensure_data()
        prop_path = self.config.data_dir / "properties.parquet"
        return pd.read_parquet(prop_path)

    def load_reviews(self) -> pd.DataFrame:
        self.ensure_data()
        review_path = self.config.data_dir / "reviews.jsonl"
        rows = []
        with review_path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Synthetic data generation helpers
    # ------------------------------------------------------------------
    def _generate_properties(self, num_properties: int) -> pd.DataFrame:
        districts = ["西屯區", "北屯區", "南屯區", "西區", "北區", "中區"]
        types_ = ["公寓", "電梯大樓", "透天", "套房"]
        property_ids = np.arange(1, num_properties + 1)
        props = pd.DataFrame(
            {
                "property_id": property_ids,
                "district": self.rng.choice(districts, size=num_properties),
                "type": self.rng.choice(types_, size=num_properties),
                "bedrooms": self.rng.integers(1, 5, size=num_properties),
                "area_sqm": self.rng.uniform(15, 120, size=num_properties).round(2),
                "base_rent": self.rng.uniform(8000, 45000, size=num_properties).round(2),
            }
        )
        props["parking"] = self.rng.choice([0, 1], size=num_properties, p=[0.6, 0.4])
        props["floor"] = self.rng.integers(1, 15, size=num_properties)
        return props

    def _generate_transactions(self, num_tenants: int, num_properties: int, days: int) -> pd.DataFrame:
        start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days)
        timestamps = pd.date_range(start=start, periods=days, freq="D")
        records = []
        for ts in timestamps:
            daily_tenants = self.rng.poisson(lam=num_tenants // 5)
            sampled_tenants = self.rng.choice(np.arange(1, num_tenants + 1), size=max(1, daily_tenants), replace=False)
            for tenant_id in sampled_tenants:
                prop_id = self.rng.integers(1, num_properties + 1)
                rent = float(self.rng.uniform(8000, 45000))
                records.append(
                    {
                        "transaction_id": f"TX-{ts.strftime('%Y%m%d')}-{tenant_id}-{prop_id}",
                        "tenant_id": int(tenant_id),
                        "property_id": int(prop_id),
                        "rent_amount": round(rent, 2),
                        "timestamp": ts + pd.to_timedelta(int(self.rng.integers(0, 24)), unit="h"),
                    }
                )
        df = pd.DataFrame.from_records(records)
        return df

    def _generate_reviews(self, num_tenants: int, num_properties: int, samples: int) -> list[dict]:
        phrases = [
            "交通便利，附近有公車站",
            "房東很好，維修回應迅速",
            "周邊安靜，適合居住",
            "租金合理，性價比高",
            "採光良好，通風不錯",
            "離捷運站很近，通勤方便",
            "社區安全，鄰居友善",
            "裝潢簡潔，設備齊全",
        ]
        reviews = []
        for idx in range(samples):
            reviews.append(
                {
                    "review_id": f"RV-{idx:05d}",
                    "tenant_id": int(self.rng.integers(1, num_tenants + 1)),
                    "property_id": int(self.rng.integers(1, num_properties + 1)),
                    "rating": int(self.rng.integers(1, 6)),
                    "review_text": str(self.rng.choice(phrases)),
                }
            )
        return reviews


def summarize_dataset_shapes(config: PropTechConfig | None = None) -> Tuple[int, int, int]:
    loader = PropTechDataLoader(config)
    tx = loader.load_transactions()
    props = loader.load_properties()
    reviews = loader.load_reviews()
    return len(tx), len(props), len(reviews)
