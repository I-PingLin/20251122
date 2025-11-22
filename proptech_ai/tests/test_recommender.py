import pandas as pd
import pytest

from proptech_ai.models.recommender import PropertyRecommender


def test_recommender_fit_and_recommend():
    tx = pd.DataFrame(
        {
            "tenant_id": [1, 1, 2, 2, 3],
            "property_id": [1, 2, 1, 2, 1],
            "rent_amount": [10000, 12000, 11000, 13000, 9000],
        }
    )
    recommender = PropertyRecommender()
    artifact_path = recommender.fit(tx)
    assert artifact_path.exists()
    recs = recommender.recommend(tenant_id=1, top_k=2)
    assert len(recs) == 2
    assert all(isinstance(r.property_id, int) for r in recs)
    assert all(isinstance(r.score, float) for r in recs)
