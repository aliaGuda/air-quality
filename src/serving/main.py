import hashlib
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.serving.logging_config import log_prediction
from src.serving.metrics import (
    CONFIDENCE_HISTOGRAM,
    CURRENT_MODEL_VERSION,
    FEATURE_C6H6_GT_HISTOGRAM,
    FEATURE_PT08_S1_CO_HISTOGRAM,
    INFERENCE_COUNT_BY_CLASS,
    LATEST_PREDICTION_VALUE,
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_LATENCY_SECONDS,
    PREDICTION_REQUESTS_TOTAL,
    bucket_prediction,
)
from src.serving.model_loader import ModelService
from src.serving.schemas import (
    BatchPredictionRequest,
    PredictionRequest,
    PredictionResponse,
)


model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()

    try:
        CURRENT_MODEL_VERSION.set(float(model_service.model_version))
    except (ValueError, TypeError):
        CURRENT_MODEL_VERSION.set(1)

    yield


app = FastAPI(
    title="Air Quality UCI Model Serving API",
    description="FastAPI model serving app with structured JSON logging and Prometheus metrics",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Air Quality Model API</title>
        </head>
        <body>
            <h1>Air Quality UCI Model Serving API</h1>
            <p>This service serves CO(GT) predictions using FastAPI.</p>
            <ul>
                <li><a href="/health">Health</a></li>
                <li><a href="/docs">Swagger Docs</a></li>
                <li><a href="/metrics">Prometheus Metrics</a></li>
                <li><a href="/schema">Schema</a></li>
            </ul>
        </body>
    </html>
    """


def validate_and_order_features(features: dict) -> dict:
    expected = model_service.expected_features

    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Expected feature list is not loaded.",
        )

    missing = set(expected) - set(features.keys())

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {sorted(missing)}",
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

    CONFIDENCE_HISTOGRAM.observe(float(confidence))

    if "PT08.S1(CO)" in request.features:
        FEATURE_PT08_S1_CO_HISTOGRAM.observe(
            float(request.features["PT08.S1(CO)"])
        )

    if "C6H6(GT)" in request.features:
        FEATURE_C6H6_GT_HISTOGRAM.observe(float(request.features["C6H6(GT)"]))

    try:
        CURRENT_MODEL_VERSION.set(float(model_service.model_version))
    except (ValueError, TypeError):
        CURRENT_MODEL_VERSION.set(1)

    prediction_class = bucket_prediction(float(prediction))
    INFERENCE_COUNT_BY_CLASS.labels(prediction_class=prediction_class).inc()

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

    except Exception as error:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(
            status_code=500,
            detail=str(error),
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

    except Exception as error:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )