from prometheus_client import Counter, Gauge, Histogram


PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
)

LATEST_PREDICTION_VALUE = Gauge(
    "latest_prediction_value",
    "Latest prediction value",
)

CONFIDENCE_HISTOGRAM = Histogram(
    "prediction_confidence_histogram",
    "Histogram of prediction confidence scores",
)

FEATURE_PT08_S1_CO_HISTOGRAM = Histogram(
    "feature_pt08_s1_co_histogram",
    "Histogram of PT08.S1(CO) input feature values",
)

FEATURE_C6H6_GT_HISTOGRAM = Histogram(
    "feature_c6h6_gt_histogram",
    "Histogram of C6H6(GT) input feature values",
)

CURRENT_MODEL_VERSION = Gauge(
    "current_model_version",
    "Current deployed model version",
)

INFERENCE_COUNT_BY_CLASS = Counter(
    "inference_count_by_class",
    "Inference count grouped by prediction bucket",
    ["prediction_class"],
)


def bucket_prediction(prediction: float) -> str:
    if prediction < 1.0:
        return "low_co"
    if prediction < 3.0:
        return "medium_co"
    return "high_co"