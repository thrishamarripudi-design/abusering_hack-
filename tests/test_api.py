from fastapi.testclient import TestClient

from abusering.api.main import app


def test_health_and_predict_end_to_end():
    with TestClient(app) as client:
        health = client.get("/health").json()
        if health["status"] != "ok":
            import pytest

            pytest.skip(
                "artifacts/experiments.json not present — run the experiment "
                "pipeline before running API tests"
            )

        info = client.get("/model/info").json()
        assert "model_version" in info
        assert "selected_threshold" in info

        bad = client.post("/predict", json={"transaction_id": "DOES_NOT_EXIST"})
        assert bad.status_code == 404

        exps = client.get("/experiments").json()
        assert "baseline_rules" in exps

        bench = client.get("/metrics/benchmark").json()
        assert "winner" in bench
