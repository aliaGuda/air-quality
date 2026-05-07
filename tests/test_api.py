from fastapi.testclient import TestClient
from app.main import app


def build_sample_features(expected_features):
    default_values = {
        "PT08.S1(CO)": 1000.0,
        "NMHC(GT)": 150.0,
        "C6H6(GT)": 10.0,
        "PT08.S2(NMHC)": 900.0,
        "NOx(GT)": 120.0,
        "PT08.S3(NOx)": 800.0,
        "NO2(GT)": 80.0,
        "PT08.S4(NO2)": 1600.0,
        "PT08.S5(O3)": 1000.0,
        "T": 20.0,
        "RH": 50.0,
        "AH": 1.0,
        "month": 5.0,
        "day": 7.0,
        "day_of_week": 3.0,
        "hour": 12.0,
    }

    return {
        feature: float(default_values.get(feature, 0.0))
        for feature in expected_features
    }


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"
        assert "model_name" in data
        assert "model_version" in data
        assert "expected_feature_count" in data


def test_schema():
    with TestClient(app) as client:
        response = client.get("/schema")

        assert response.status_code == 200

        data = response.json()

        assert "expected_features" in data
        assert isinstance(data["expected_features"], list)
        assert len(data["expected_features"]) > 0


def test_predict():
    with TestClient(app) as client:
        schema_response = client.get("/schema")
        expected_features = schema_response.json()["expected_features"]

        sample_features = build_sample_features(expected_features)

        response = client.post(
            "/predict",
            json={"features": sample_features}
        )

        assert response.status_code == 200, response.text

        data = response.json()

        assert "request_id" in data
        assert "model_version" in data
        assert "confidence" in data
        assert "latency_ms" in data
        assert "feature_hash" in data
        assert "prediction" in data

        assert "features" not in data

        assert isinstance(data["prediction"], float)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["latency_ms"], float)


def test_batch_predict():
    with TestClient(app) as client:
        schema_response = client.get("/schema")
        expected_features = schema_response.json()["expected_features"]

        sample_features = build_sample_features(expected_features)

        response = client.post(
            "/predict/batch",
            json={
                "records": [
                    {"features": sample_features},
                    {"features": sample_features},
                ]
            }
        )

        assert response.status_code == 200, response.text

        data = response.json()

        assert data["count"] == 2
        assert "predictions" in data
        assert len(data["predictions"]) == 2

        for prediction in data["predictions"]:
            assert "request_id" in prediction
            assert "model_version" in prediction
            assert "confidence" in prediction
            assert "latency_ms" in prediction
            assert "feature_hash" in prediction
            assert "prediction" in prediction
            assert "features" not in prediction


def test_missing_features_validation():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "features": {
                    "T": 20.0,
                    "RH": 50.0
                }
            }
        )

        assert response.status_code == 400
        assert "Missing required features" in response.json()["detail"]


def test_metrics():
    with TestClient(app) as client:
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "prediction_requests_total" in response.text