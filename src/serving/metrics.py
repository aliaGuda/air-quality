from prometheus_client import Counter, Histogram, Gauge


PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

LATEST_PREDICTION_VALUE = Gauge(
    "latest_prediction_value",
    "Most recent model prediction value"
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)