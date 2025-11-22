from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from ..config import DEFAULT_CONFIG, PropTechConfig
from ..feature_engineering import FeatureArtifacts, build_interaction_matrix


@dataclass
class Recommendation:
    property_id: int
    score: float


class PropertyRecommender:
    """ALS-based property-to-tenant recommender."""

    def __init__(self, config: PropTechConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.tenant_factors: np.ndarray | None = None
        self.property_factors: np.ndarray | None = None
        self.artifacts: FeatureArtifacts | None = None

    def fit(self, transactions_frame) -> Path:
        matrix, artifacts = build_interaction_matrix(transactions_frame)
        self.artifacts = artifacts
        num_tenants, num_properties = matrix.shape
        components = min(self.config.recommender_components, num_tenants, num_properties)
        tenant_factors = np.random.normal(0, 0.1, size=(num_tenants, components))
        property_factors = np.random.normal(0, 0.1, size=(num_properties, components))

        reg = 0.1
        for iteration in range(self.config.recommender_iter):
            # Update tenant factors
            for t in range(num_tenants):
                prop_idx = matrix[t] > 0
                if not np.any(prop_idx):
                    continue
                V = property_factors[prop_idx]
                ratings = matrix[t, prop_idx]
                A = V.T @ V + reg * np.eye(components)
                b = V.T @ ratings
                tenant_factors[t] = np.linalg.solve(A, b)
            # Update property factors
            for p in range(num_properties):
                tenant_idx = matrix[:, p] > 0
                if not np.any(tenant_idx):
                    continue
                U = tenant_factors[tenant_idx]
                ratings = matrix[tenant_idx, p]
                A = U.T @ U + reg * np.eye(components)
                b = U.T @ ratings
                property_factors[p] = np.linalg.solve(A, b)
            if (iteration + 1) % 50 == 0:
                recon = tenant_factors @ property_factors.T
                mask = matrix > 0
                rmse = np.sqrt(np.mean((recon[mask] - matrix[mask]) ** 2))
                print(f"[ALS] iter={iteration+1} rmse={rmse:.4f}")

        self.tenant_factors = tenant_factors
        self.property_factors = property_factors
        artifact_path = self.config.artifact_dir / "recommender_weights.npz"
        np.savez(
            artifact_path,
            tenant_factors=tenant_factors,
            property_factors=property_factors,
            property_mapping=self.artifacts.property_mapping,
            tenant_mapping=self.artifacts.tenant_mapping,
        )
        return artifact_path

    def recommend(self, tenant_id: int, top_k: int = 5) -> List[Recommendation]:
        if self.tenant_factors is None or self.property_factors is None or self.artifacts is None:
            raise RuntimeError("Model not trained")
        if tenant_id not in self.artifacts.tenant_mapping:
            raise ValueError(f"Tenant {tenant_id} unknown")
        t_idx = self.artifacts.tenant_mapping[tenant_id]
        scores = self.property_factors @ self.tenant_factors[t_idx]
        ranked = np.argsort(scores)[::-1][:top_k]
        recommendations = []
        for idx in ranked:
            property_id = self.artifacts.inverse_property_mapping[idx]
            recommendations.append(Recommendation(property_id=property_id, score=float(scores[idx])))
        return recommendations
