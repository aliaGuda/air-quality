import time
import uuid
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
)
from app.model_loader import ModelService
from app.logging_config import log_prediction
from app.metrics import (
    PREDICTION_REQUESTS_TOTAL,
    PREDICTION_LATENCY_SECONDS,
    LATEST_PREDICTION_VALUE,
    PREDICTION_ERRORS_TOTAL,
)


model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="Air Quality UCI Model Serving API",
    description="FastAPI model serving app with structured JSON logging and Prometheus metrics",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Air Quality Model API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: #f4f7fb;
                    color: #1f2937;
                }
                .card {
                    max-width: 900px;
                    margin: auto;
                    background: white;
                    padding: 30px;
                    border-radius: 16px;
                    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
                }
                h1 { color: #2563eb; }
                a {
                    display: inline-block;
                    margin: 8px 8px 8px 0;
                    padding: 10px 14px;
                    background: #2563eb;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                }
                code {
                    background: #eef2ff;
                    padding: 4px 8px;
                    border-radius: 6px;
                }
                li { margin-bottom: 8px; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🌫️ Air Quality UCI Model Serving API</h1>
                <p>This service loads a trained ML model and serves predictions through FastAPI.</p>

                <h2>Endpoints</h2>
                <ul>
                    <li><code>GET /health</code> — model and API status</li>
                    <li><code>POST /predict</code> — single prediction</li>
                    <li><code>POST /predict/batch</code> — batch predictions</li>
                    <li><code>GET /metrics</code> — Prometheus metrics</li>
                    <li><code>GET /schema</code> — expected model features</li>
                </ul>

                <a href="/health">Open Health</a>
                <a href="/docs">Open Swagger Docs</a>
                <a href="/metrics">Open Metrics</a>
                <a href="/schema">Open Schema</a>

                <h2>Structured Logging</h2>
                <p>Every prediction is logged to <code>logs/predictions.jsonl</code> with:</p>
                <ul>
                    <li>request_id</li>
                    <li>model_version</li>
                    <li>confidence</li>
                    <li>latency_ms</li>
                    <li>feature_hash</li>
                    <li>prediction</li>
                </ul>

                <p>No raw feature values are logged.</p>
            </div>
        </body>
    </html>
    """


def validate_and_order_features(features: dict) -> dict:
    expected = model_service.expected_features

    missing = set(expected) - set(features.keys())

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {sorted(missing)}"
        )

    return {feature: float(features[feature]) for feature in expected}


def hash_features(features: dict) -> str:
    raw = str(sorted(features.items()))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def calculate_confidence(prediction: float) -> float:
    confidence = 1 / (1 + abs(prediction))
    return round(float(confidence), 4)


def make_prediction(request: PredictionRequest) -> dict:
    start_time = time.time()
    request_id = str(uuid.uuid4())

    ordered_features = validate_and_order_features(request.features)

    with PREDICTION_LATENCY_SECONDS.time():
        prediction = model_service.predict_one(ordered_features)

    latency_ms = round((time.time() - start_time) * 1000, 3)
    confidence = calculate_confidence(prediction)

    PREDICTION_REQUESTS_TOTAL.inc()
    LATEST_PREDICTION_VALUE.set(prediction)

    response_payload = {
        "request_id": request_id,
        "model_version": model_service.model_version,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "feature_hash": hash_features(ordered_features),
        "prediction": prediction,
    }

    log_prediction(response_payload)

    return response_payload


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": model_service.model_name,
        "model_version": model_service.model_version,
        "target_variable": model_service.target_variable,
        "expected_feature_count": len(model_service.expected_features),
    }


@app.get("/schema")
def schema():
    return {
        "target_variable": model_service.target_variable,
        "expected_features": model_service.expected_features,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        return make_prediction(request)

    except HTTPException:
        raise

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    try:
        predictions = [make_prediction(record) for record in request.records]

        return {
            "count": len(predictions),
            "predictions": predictions,
        }

    except HTTPException:
        raise

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )