import json
import logging
import os
from pathlib import Path


LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "predictions.jsonl"

LOG_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


def log_prediction(payload: dict) -> None:
    """
    Write PII-safe structured prediction logs.
    Raw feature values are not logged.
    """
    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + os.linesep)

    logging.info(json.dumps(payload))