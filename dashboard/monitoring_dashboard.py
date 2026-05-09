import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


PREDICTION_LOG_PATH = Path("logs/predictions.jsonl")
DRIFT_LOG_PATH = Path("src/monitoring/drift_warnings.log")
BASELINE_REPORT_PATH = Path("src/monitoring/evidently_reports/baseline_report.html")
DRIFT_REPORT_PATH = Path("src/monitoring/evidently_reports/drift_report.html")


st.set_page_config(
    page_title="Air Quality Monitoring Dashboard",
    page_icon="🌫️",
    layout="wide",
)


st.title("🌫️ Air Quality Model Monitoring Dashboard")
st.caption(
    "Production monitoring dashboard for FastAPI inference logs, "
    "Prometheus-style metrics, drift detection, and Evidently reports."
)


def load_prediction_logs() -> pd.DataFrame:
    if not PREDICTION_LOG_PATH.exists():
        return pd.DataFrame()

    records = []

    with open(PREDICTION_LOG_PATH, "r", encoding="utf-8") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)

    if not df.empty:
        df["request_number"] = range(1, len(df) + 1)

    return df


def load_drift_logs() -> list:
    if not DRIFT_LOG_PATH.exists():
        return []

    logs = []

    with open(DRIFT_LOG_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if "DRIFT_CHECK_RESULT" in line:
                try:
                    json_part = line.split("|")[-1].strip()
                    logs.append(json.loads(json_part))
                except Exception:
                    continue

    return logs


def prediction_bucket(value: float) -> str:
    if value < 1.0:
        return "low_co"
    if value < 3.0:
        return "medium_co"
    return "high_co"


def show_evidently_report(path: Path, height: int = 900) -> None:
    if not path.exists():
        st.error(f"Report not found: {path}")
        return

    html = path.read_text(encoding="utf-8")
    components.html(html, height=height, scrolling=True)


predictions_df = load_prediction_logs()
drift_logs = load_drift_logs()


if not predictions_df.empty and "prediction" in predictions_df.columns:
    predictions_df["prediction_bucket"] = predictions_df["prediction"].apply(
        prediction_bucket
    )

if not predictions_df.empty and "latency_ms" in predictions_df.columns:
    predictions_df["rolling_latency_ms"] = (
        predictions_df["latency_ms"].rolling(window=5, min_periods=1).mean()
    )


st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Choose section",
    [
        "Overview",
        "Prediction Analytics",
        "Latency Monitoring",
        "Confidence Monitoring",
        "Drift Detection",
        "Evidently Reports",
        "Raw Logs",
    ],
)


if section == "Overview":
    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    total_requests = len(predictions_df)

    if not predictions_df.empty:
        avg_latency = predictions_df["latency_ms"].mean()
        latest_prediction = predictions_df["prediction"].iloc[-1]
        latest_model_version = predictions_df["model_version"].iloc[-1]
    else:
        avg_latency = 0
        latest_prediction = 0
        latest_model_version = "N/A"

    col1.metric("Total Predictions", total_requests)
    col2.metric("Average Latency", f"{avg_latency:.2f} ms")
    col3.metric("Latest Prediction", f"{latest_prediction:.4f}")
    col4.metric("Model Version", latest_model_version)

    st.divider()

    if drift_logs:
        latest_drift = drift_logs[-1]
        drift_detected = latest_drift.get("drift_detected", False)
        drift_share = latest_drift.get("drift_share", 0)

        if drift_detected:
            st.error(f"⚠️ Drift detected: {drift_share:.2%} of features drifted.")
        else:
            st.success(f"✅ No major drift detected: {drift_share:.2%} drift share.")
    else:
        st.warning("No drift log found yet. Run the Evidently monitoring script first.")

    st.subheader("Monitoring Architecture")
    st.code(
        """
User → FastAPI Serving API → MLflow / Local Model
     → Structured JSON Logs
     → Prometheus Metrics
     → Grafana Live Dashboard
     → Evidently Drift Reports
     → Streamlit Monitoring Dashboard
        """,
        language="text",
    )


elif section == "Prediction Analytics":
    st.subheader("Prediction Analytics")

    if predictions_df.empty:
        st.warning("No prediction logs found yet.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("Prediction Trend")
            st.line_chart(
                predictions_df.set_index("request_number")["prediction"]
            )

        with col2:
            st.write("Prediction Distribution")
            st.bar_chart(predictions_df["prediction"].value_counts(bins=10).sort_index())

        st.divider()

        col3, col4 = st.columns(2)

        with col3:
            st.write("Prediction Buckets")
            bucket_counts = predictions_df["prediction_bucket"].value_counts()
            st.bar_chart(bucket_counts)

        with col4:
            st.write("Model Version Usage")
            model_counts = predictions_df["model_version"].value_counts()
            st.bar_chart(model_counts)


elif section == "Latency Monitoring":
    st.subheader("Latency Monitoring")

    if predictions_df.empty:
        st.warning("No prediction logs found yet.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("Latency Over Requests")
            st.line_chart(
                predictions_df.set_index("request_number")["latency_ms"]
            )

        with col2:
            st.write("Rolling Average Latency")
            st.line_chart(
                predictions_df.set_index("request_number")["rolling_latency_ms"]
            )

        st.divider()

        st.write("Latency Distribution")
        st.bar_chart(predictions_df["latency_ms"].value_counts(bins=10).sort_index())

        st.divider()

        st.write("Latency Summary")
        st.dataframe(
            predictions_df["latency_ms"].describe().to_frame("latency_ms"),
            use_container_width=True,
        )


elif section == "Confidence Monitoring":
    st.subheader("Confidence Monitoring")

    if predictions_df.empty:
        st.warning("No prediction logs found yet.")
    elif "confidence" not in predictions_df.columns:
        st.warning("No confidence column found in prediction logs.")
    else:
        st.info(
            "Confidence is kept because Component 4 requires confidence scores "
            "and a confidence histogram."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("Confidence Trend")
            st.line_chart(
                predictions_df.set_index("request_number")["confidence"]
            )

        with col2:
            st.write("Confidence Distribution")
            st.bar_chart(
                predictions_df["confidence"].value_counts(bins=10).sort_index()
            )

        st.divider()

        st.write("Confidence Summary")
        st.dataframe(
            predictions_df["confidence"].describe().to_frame("confidence"),
            use_container_width=True,
        )


elif section == "Drift Detection":
    st.subheader("Drift Detection Results")

    if not drift_logs:
        st.warning("No drift logs found yet.")
    else:
        drift_df = pd.DataFrame(drift_logs)
        drift_df["run_number"] = range(1, len(drift_df) + 1)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Drift Share Over Runs")
            st.line_chart(drift_df.set_index("run_number")["drift_share"])

        with col2:
            st.write("Drift Status Counts")
            st.bar_chart(drift_df["drift_detected"].value_counts())

        st.divider()

        all_drifted_features = []

        for log in drift_logs:
            report_name = log.get("report", "unknown")
            for feature in log.get("drifted_features", []):
                all_drifted_features.append(
                    {
                        "report": report_name,
                        "feature": feature.get("feature"),
                        "score": feature.get("score"),
                        "method": feature.get("method"),
                    }
                )

        if all_drifted_features:
            feature_df = pd.DataFrame(all_drifted_features)

            st.write("Top Drifted Features")
            st.bar_chart(feature_df["feature"].value_counts())

            st.write("Drifted Feature Details")
            st.dataframe(feature_df, use_container_width=True)
        else:
            st.success("No drifted features found in the latest logs.")

        st.divider()

        st.write("Raw Drift Results")
        for log in drift_logs:
            report_name = log.get("report", "unknown")
            drift_share = log.get("drift_share", 0)
            drift_detected = log.get("drift_detected", False)

            with st.expander(f"{report_name} | Drift Share: {drift_share:.2%}"):
                if drift_detected:
                    st.error("Drift detected above threshold.")
                else:
                    st.success("No major drift detected.")

                st.json(log)


elif section == "Evidently Reports":
    st.subheader("Evidently HTML Reports")

    report_choice = st.radio(
        "Choose report",
        ["Baseline Report", "Drift Report"],
        horizontal=True,
    )

    if report_choice == "Baseline Report":
        if BASELINE_REPORT_PATH.exists():
            st.success("Baseline report loaded.")
            show_evidently_report(BASELINE_REPORT_PATH)
        else:
            st.error("Baseline report not found.")

    if report_choice == "Drift Report":
        if DRIFT_REPORT_PATH.exists():
            st.success("Drift report loaded.")
            show_evidently_report(DRIFT_REPORT_PATH)
        else:
            st.error("Drift report not found.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if BASELINE_REPORT_PATH.exists():
            with open(BASELINE_REPORT_PATH, "rb") as file:
                st.download_button(
                    label="Download Baseline Report",
                    data=file,
                    file_name="baseline_report.html",
                    mime="text/html",
                )

    with col2:
        if DRIFT_REPORT_PATH.exists():
            with open(DRIFT_REPORT_PATH, "rb") as file:
                st.download_button(
                    label="Download Drift Report",
                    data=file,
                    file_name="drift_report.html",
                    mime="text/html",
                )


elif section == "Raw Logs":
    st.subheader("Raw Prediction Logs")

    if predictions_df.empty:
        st.warning("No prediction logs found yet.")
    else:
        st.dataframe(predictions_df, use_container_width=True)

        st.download_button(
            label="Download Prediction Logs as CSV",
            data=predictions_df.to_csv(index=False),
            file_name="prediction_logs.csv",
            mime="text/csv",
        )