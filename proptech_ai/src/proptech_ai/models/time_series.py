from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..config import DEFAULT_CONFIG, PropTechConfig
from ..feature_engineering import build_rental_timeseries, sliding_window


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))


@dataclass
class ForecastBatch:
    inputs: torch.Tensor
    targets: torch.Tensor


class _LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


class RentalDemandForecaster:
    """PyTorch LSTM forecaster for per-property rental demand."""

    def __init__(self, config: PropTechConfig | None = None, device: str | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: _LSTMForecaster | None = None

    # ------------------------------------------------------------------
    def _prepare_dataset(
        self,
        property_series: np.ndarray,
        lookback: int,
        horizon: int,
    ) -> Dataset:
        X, y = sliding_window(
            series=property_series,
            lookback=lookback,
            horizon=horizon,
        )
        X = X.reshape(X.shape[0], lookback, 1)
        return TensorDataset(_to_tensor(X), _to_tensor(y))

    def _init_model(self, horizon: int) -> None:
        self.model = _LSTMForecaster(
            input_size=1,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            horizon=horizon,
        ).to(self.device)

    # ------------------------------------------------------------------
    def fit(
        self,
        transactions_frame,
        property_id: int,
        epochs: int | None = None,
    ) -> Path:
        df = build_rental_timeseries(transactions_frame, self.config)
        if property_id not in df.columns:
            raise ValueError(f"Property {property_id} not found in demand table")
        series = df[property_id]
        dataset = self._prepare_dataset(
            series,
            self.config.sequence_lookback,
            self.config.forecast_horizon,
        )
        loader = DataLoader(dataset, batch_size=self.config.forecast_batch_size, shuffle=True)
        self._init_model(self.config.forecast_horizon)
        assert self.model is not None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.L1Loss()

        self.model.train()
        for epoch in range(epochs or self.config.forecast_epochs):
            epoch_loss = 0.0
            for batch in loader:
                X, y = [tensor.to(self.device) for tensor in batch]
                pred = self.model(X)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
            if (epoch + 1) % 1 == 0:
                print(f"[RentalDemandForecaster] epoch={epoch+1} loss={epoch_loss/len(loader):.4f}")

        artifact_path = self.config.artifact_dir / f"lstm_property_{property_id}.pt"
        torch.save(
            {
                "property_id": property_id,
                "config": self.config.to_dict(),
                "state_dict": self.model.state_dict(),
            },
            artifact_path,
        )
        return artifact_path

    # ------------------------------------------------------------------
    def forecast(self, recent_history: Iterable[float], horizon: int | None = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first or load weights.")
        lookback = self.config.sequence_lookback
        history = np.array(list(recent_history), dtype=np.float32)
        if len(history) < lookback:
            raise ValueError("Insufficient history for forecasting")
        window = history[-lookback:].reshape(1, lookback, 1)
        with torch.no_grad():
            pred = self.model(_to_tensor(window).to(self.device))
        horizon = horizon or self.config.forecast_horizon
        return pred.cpu().numpy().reshape(-1)[:horizon]

    def load_weights(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.config = PropTechConfig(**{**self.config.to_dict(), **checkpoint.get("config", {})})
        self._init_model(self.config.forecast_horizon)
        assert self.model is not None
        self.model.load_state_dict(checkpoint["state_dict"])
