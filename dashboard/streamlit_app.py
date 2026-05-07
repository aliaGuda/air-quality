import requests
import pandas as pd
import streamlit as st
from datetime import datetime


API_BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{API_BASE_URL}/health"
SCHEMA_URL = f"{API_BASE_URL}/schema"
PREDICT_URL = f"{API_BASE_URL}/predict"


st.set_page_config(
    page_title="Air Quality Model Inference",
    page_icon="🌫️",
    layout="wide"
)


st.title("🌫️ Air Quality UCI Model Inference UI")
st.caption("User-friendly inference dashboard connected to the FastAPI model serving backend.")


def get_api_health():
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_model_schema():
    try:
        response = requests.get(SCHEMA_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


health = get_api_health()
schema = get_model_schema()

with st.sidebar:
    st.header("System Status")

    if health:
        st.success("FastAPI backend is running")
        st.write(f"Model: `{health.get('model_name')}`")
        st.write(f"Version: `{health.get('model_version')}`")
        st.write(f"Target: `{health.get('target_variable')}`")
        st.write(f"Features: `{health.get('expected_feature_count')}`")
    else:
        st.error("FastAPI backend is not reachable")
        st.info("Run: uvicorn app.main:app --reload")

    st.divider()

    st.header("How it works")
    st.write(
        "1. Enter air-quality sensor values\n"
        "2. Select measurement timestamp\n"
        "3. Click Run Prediction\n"
        "4. Streamlit sends data to FastAPI\n"
        "5. FastAPI returns prediction and logs it safely"
    )


if not schema:
    st.error("Could not load model schema from FastAPI.")
    st.stop()


expected_features = schema["expected_features"]
target_variable = schema["target_variable"]

st.info(f"This model predicts: **{target_variable}**")


preset = st.selectbox(
    "Choose an input scenario",
    ["Normal air quality", "High pollution", "Low pollution", "Custom"]
)


preset_values = {
    "Normal air quality": {
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
    },
    "High pollution": {
        "PT08.S1(CO)": 2500.0,
        "NMHC(GT)": 800.0,
        "C6H6(GT)": 35.0,
        "PT08.S2(NMHC)": 2200.0,
        "NOx(GT)": 500.0,
        "PT08.S3(NOx)": 300.0,
        "NO2(GT)": 240.0,
        "PT08.S4(NO2)": 2600.0,
        "PT08.S5(O3)": 2300.0,
        "T": 35.0,
        "RH": 80.0,
        "AH": 3.0,
    },
    "Low pollution": {
        "PT08.S1(CO)": 600.0,
        "NMHC(GT)": 50.0,
        "C6H6(GT)": 2.0,
        "PT08.S2(NMHC)": 500.0,
        "NOx(GT)": 30.0,
        "PT08.S3(NOx)": 1200.0,
        "NO2(GT)": 20.0,
        "PT08.S4(NO2)": 900.0,
        "PT08.S5(O3)": 500.0,
        "T": 15.0,
        "RH": 40.0,
        "AH": 0.5,
    },
    "Custom": {},
}


defaults = preset_values[preset]
features = {}


tab1, tab2, tab3, tab4 = st.tabs(
    ["Gas Sensors", "Pollutants", "Weather", "Measurement Timestamp"]
)


with tab1:
    gas_sensor_features = [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
    ]

    cols = st.columns(2)

    for index, feature in enumerate(gas_sensor_features):
        if feature in expected_features:
            with cols[index % 2]:
                features[feature] = st.slider(
                    feature,
                    min_value=0.0,
                    max_value=4000.0,
                    value=float(defaults.get(feature, 1000.0)),
                )


with tab2:
    pollutant_features = [
        "NMHC(GT)",
        "C6H6(GT)",
        "NOx(GT)",
        "NO2(GT)",
    ]

    cols = st.columns(2)

    for index, feature in enumerate(pollutant_features):
        if feature in expected_features:
            with cols[index % 2]:
                features[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    value=float(defaults.get(feature, 10.0)),
                )


with tab3:
    weather_features = ["T", "RH", "AH"]

    cols = st.columns(3)

    for index, feature in enumerate(weather_features):
        if feature in expected_features:
            with cols[index]:
                if feature == "T":
                    features[feature] = st.slider(
                        "Temperature (T)",
                        min_value=-10.0,
                        max_value=50.0,
                        value=float(defaults.get(feature, 20.0)),
                    )
                elif feature == "RH":
                    features[feature] = st.slider(
                        "Relative Humidity (RH)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(defaults.get(feature, 50.0)),
                    )
                else:
                    features[feature] = st.slider(
                        "Absolute Humidity (AH)",
                        min_value=0.0,
                        max_value=5.0,
                        value=float(defaults.get(feature, 1.0)),
                    )


with tab4:
    st.info(
        "For time-series data, the model should use the timestamp of the measurement, "
        "not the current system time."
    )

    measurement_date = st.date_input(
        "Measurement Date",
        value=datetime.now().date()
    )

    measurement_time = st.time_input(
        "Measurement Time",
        value=datetime.now().time()
    )

    measurement_datetime = datetime.combine(
        measurement_date,
        measurement_time
    )

    time_values = {
        "month": float(measurement_datetime.month),
        "day": float(measurement_datetime.day),
        "day_of_week": float(measurement_datetime.weekday()),
        "hour": float(measurement_datetime.hour),
    }

    for feature, value in time_values.items():
        if feature in expected_features:
            features[feature] = value

    st.json({
        "measurement_timestamp": str(measurement_datetime),
        **time_values
    })


remaining_features = [
    feature for feature in expected_features
    if feature not in features
]

if remaining_features:
    with st.expander("Other model-required features"):
        for feature in remaining_features:
            features[feature] = st.number_input(
                feature,
                value=float(defaults.get(feature, 0.0)),
            )


ordered_features = {
    feature: float(features[feature])
    for feature in expected_features
}


st.divider()

left, right = st.columns([1.4, 1])


with left:
    st.subheader("Input Sent to Model")
    st.dataframe(
        pd.DataFrame([ordered_features]),
        use_container_width=True
    )


with right:
    st.subheader("Prediction")

    if st.button("Run Prediction", type="primary", use_container_width=True):
        payload = {
            "features": ordered_features
        }

        try:
            response = requests.post(
                PREDICT_URL,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction completed successfully")

                st.metric("Predicted Value", round(result["prediction"], 4))
                st.metric("Confidence", result["confidence"])
                st.metric("Latency", f"{result['latency_ms']} ms")

                with st.expander("Full API Response"):
                    st.json(result)

                with st.expander("Submitted Payload"):
                    st.json(payload)

            else:
                st.error("Prediction failed")
                try:
                    st.json(response.json())
                except Exception:
                    st.write(response.text)

        except Exception as e:
            st.error("Could not connect to FastAPI")
            st.write(str(e))