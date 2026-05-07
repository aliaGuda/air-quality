import json
import logging
from pathlib import Path
from typing import Dict

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

PREDICTION_LOG_PATH = LOG_DIR / "predictions.jsonl"

REQUIRED_LOG_FIELDS = {
    "request_id",
    "model_version",
    "confidence",
    "latency_ms",
    "feature_hash",
    "prediction",
}


logger = logging.getLogger("prediction_logger")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(PREDICTION_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)


def log_prediction(payload: Dict) -> None:
    safe_payload = {
        "request_id": payload["request_id"],
        "model_version": payload["model_version"],
        "confidence": payload["confidence"],
        "latency_ms": payload["latency_ms"],
        "feature_hash": payload["feature_hash"],
        "prediction": payload["prediction"],
    }

    missing = REQUIRED_LOG_FIELDS - set(safe_payload.keys())

    if missing:
        raise ValueError(f"Missing required log fields: {missing}")

    logger.info(json.dumps(safe_payload))