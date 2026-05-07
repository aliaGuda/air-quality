import json
from pathlib import Path

import pandas as pd
import streamlit as st


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
st.caption("Production observability dashboard for inference logs, drift detection, and model health.")


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

    return pd.DataFrame(records)


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


predictions_df = load_prediction_logs()
drift_logs = load_drift_logs()


st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Choose section",
    [
        "Overview",
        "Prediction Logs",
        "Latency & Confidence",
        "Drift Detection",
        "Evidently Reports",
    ],
)


if section == "Overview":
    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    total_requests = len(predictions_df)

    if not predictions_df.empty:
        avg_latency = predictions_df["latency_ms"].mean()
        avg_confidence = predictions_df["confidence"].mean()
        latest_model_version = predictions_df["model_version"].iloc[-1]
    else:
        avg_latency = 0
        avg_confidence = 0
        latest_model_version = "N/A"

    col1.metric("Total Predictions", total_requests)
    col2.metric("Average Latency", f"{avg_latency:.2f} ms")
    col3.metric("Average Confidence", f"{avg_confidence:.3f}")
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
        st.warning("No drift log found yet. Run src/monitoring/run_monitoring.py first.")

    st.subheader("Architecture")
    st.code(
        """
User → FastAPI Serving API → MLflow/Local Model
     → Structured JSON Logs
     → Prometheus Metrics
     → Evidently Drift Reports
     → Monitoring Dashboard
        """,
        language="text",
    )


elif section == "Prediction Logs":
    st.subheader("Prediction Logs")

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


elif section == "Latency & Confidence":
    st.subheader("Latency & Confidence Monitoring")

    if predictions_df.empty:
        st.warning("No prediction logs found yet.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("Latency Distribution")
            st.bar_chart(predictions_df["latency_ms"])

        with col2:
            st.write("Confidence Distribution")
            st.bar_chart(predictions_df["confidence"])

        st.divider()

        st.write("Prediction Values Over Time")
        st.line_chart(predictions_df["prediction"])


elif section == "Drift Detection":
    st.subheader("Drift Detection Results")

    if not drift_logs:
        st.warning("No drift logs found yet.")
    else:
        for log in drift_logs:
            report_name = log.get("report", "unknown")
            drift_share = log.get("drift_share", 0)
            drift_detected = log.get("drift_detected", False)
            drifted_features = log.get("drifted_features", [])

            with st.expander(f"{report_name} | Drift Share: {drift_share:.2%}"):
                if drift_detected:
                    st.error("Drift detected above threshold.")
                else:
                    st.success("No major drift detected.")

                st.json(log)

                if drifted_features:
                    st.write("Drifted Features")
                    st.dataframe(pd.DataFrame(drifted_features), use_container_width=True)


elif section == "Evidently Reports":
    st.subheader("Evidently HTML Reports")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Baseline Report")
        if BASELINE_REPORT_PATH.exists():
            st.success("Baseline report exists.")
            with open(BASELINE_REPORT_PATH, "rb") as file:
                st.download_button(
                    label="Download Baseline Report",
                    data=file,
                    file_name="baseline_report.html",
                    mime="text/html",
                )
        else:
            st.error("Baseline report not found.")

    with col2:
        st.write("Drift Report")
        if DRIFT_REPORT_PATH.exists():
            st.success("Drift report exists.")
            with open(DRIFT_REPORT_PATH, "rb") as file:
                st.download_button(
                    label="Download Drift Report",
                    data=file,
                    file_name="drift_report.html",
                    mime="text/html",
                )
        else:
            st.error("Drift report not found.")

    st.info("Open the HTML reports directly from src/monitoring/evidently_reports for full interactive Evidently views.")