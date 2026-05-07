import os
from pathlib import Path
from typing import List

import yaml
import joblib
import mlflow
import pandas as pd


MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessing_pipeline.joblib")

REGISTERED_MODEL_NAME = "air-quality-regressor"
MLFLOW_MODEL_STAGE = "Production"


DEFAULT_FEATURES = [
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
    "month",
    "day",
    "day_of_week",
    "hour",
]

def load_target_variable(path: str = "configs/params.yaml") -> str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        return params["data"]["target_column"]

    except Exception:
        return "unknown_target"

class ModelService:
    def __init__(self):
        self.model = None
        self.preprocessor = None

        self.model_name = REGISTERED_MODEL_NAME
        self.model_version = "unknown"
        self.expected_features: List[str] = DEFAULT_FEATURES
        self.target_variable = load_target_variable()
    def load(self) -> None:
        """
        Load model from MLflow Production Registry first.
        If MLflow loading fails, fall back to local models/model.joblib.
        """

        try:
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(mlflow_uri)

            model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"

            self.model = mlflow.pyfunc.load_model(model_uri)

            client = mlflow.tracking.MlflowClient()

            latest_versions = client.get_latest_versions(
                REGISTERED_MODEL_NAME,
                stages=[MLFLOW_MODEL_STAGE],
            )

            if latest_versions:
                self.model_version = latest_versions[0].version
            else:
                self.model_version = MLFLOW_MODEL_STAGE

            self.model_name = REGISTERED_MODEL_NAME

            print(f"Loaded model from MLflow Registry: {model_uri}")
            print(f"MLflow tracking URI: {mlflow_uri}")
            print(f"Model version: {self.model_version}")

        except Exception as e:
            print("Could not load model from MLflow Registry.")
            print(f"MLflow error: {e}")
            print("Falling back to local saved artifact.")

            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Local model artifact not found: {MODEL_PATH}")

            self.model = joblib.load(MODEL_PATH)

            self.model_name = REGISTERED_MODEL_NAME
            self.model_version = "local-artifact-v1"

            print(f"Loaded local model artifact from: {MODEL_PATH}")

        if PREPROCESSOR_PATH.exists():
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"Loaded preprocessing pipeline from: {PREPROCESSOR_PATH}")
        else:
            self.preprocessor = None
            print("No preprocessing pipeline found. Using raw features.")

        self.expected_features = self._detect_expected_features()

        print(f"Expected features: {self.expected_features}")

    def _detect_expected_features(self) -> List[str]:
        """
        Detect original feature names from preprocessing pipeline if available.
        """

        if self.preprocessor is not None and hasattr(self.preprocessor, "feature_names_in_"):
            return list(self.preprocessor.feature_names_in_)

        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)

        return DEFAULT_FEATURES

    def predict_one(self, features: dict) -> float:
        """
        Run one prediction using the trained preprocessing pipeline and model.
        """

        df = pd.DataFrame(
            [features],
            columns=self.expected_features
        )

        if self.preprocessor is not None:
            X = self.preprocessor.transform(df)
        else:
            X = df

        prediction = self.model.predict(X)

        return float(prediction[0])