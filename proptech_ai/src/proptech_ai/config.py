from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class PropTechConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    artifact_dir: Path = field(init=False)
    random_seed: int = 42
    sequence_lookback: int = 14
    forecast_horizon: int = 7
    forecast_epochs: int = 5
    forecast_batch_size: int = 32
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    recommender_components: int = 12
    recommender_iter: int = 200

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.artifact_dir = self.project_root / "artifacts"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        serializable = asdict(self)
        serializable["project_root"] = str(self.project_root)
        serializable["data_dir"] = str(self.data_dir)
        serializable["artifact_dir"] = str(self.artifact_dir)
        return serializable


DEFAULT_CONFIG = PropTechConfig()
