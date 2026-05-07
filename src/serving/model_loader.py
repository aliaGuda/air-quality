import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml


REGISTERED_MODEL_NAME = "air-quality-regressor"
MLFLOW_MODEL_STAGE = "Production"

MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessing_pipeline.joblib")
PARAMS_PATH = Path("configs/params.yaml")


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.preprocessor = None
        self.model_name = REGISTERED_MODEL_NAME
        self.model_version = "local"
        self.target_variable = "CO(GT)"
        self.expected_features = []

    def load_params(self) -> dict:
        with open(PARAMS_PATH, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def load_expected_features(self) -> None:
        params = self.load_params()
        self.target_variable = params["data"]["target_column"]

        train_path = Path(params["data"]["train_path"])

        if train_path.exists():
            train_df = pd.read_csv(train_path)
            self.expected_features = [
                col for col in train_df.columns if col != self.target_variable
            ]
        else:
            self.expected_features = []

    def load_local_model(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Local model not found: {MODEL_PATH}")

        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline not found: {PREPROCESSOR_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        self.model_name = "local-model"
        self.model_version = "1"

        self.load_expected_features()

        print("Loaded local model artifact.")
        print(f"Model path: {MODEL_PATH}")
        print(f"Preprocessor path: {PREPROCESSOR_PATH}")
        print(f"Expected features: {len(self.expected_features)}")

    def load(self) -> None:
        """
        Load model from MLflow Production Registry first.
        If MLflow Registry is unavailable, fall back to local model artifact.
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

            if PREPROCESSOR_PATH.exists():
                self.preprocessor = joblib.load(PREPROCESSOR_PATH)

            self.load_expected_features()

            print(f"Loaded model from MLflow Registry: {model_uri}")
            print(f"MLflow tracking URI: {mlflow_uri}")
            print(f"Model version: {self.model_version}")

        except Exception as error:
            print(f"MLflow model loading failed: {error}")
            print("Falling back to local model artifact.")
            self.load_local_model()

    def predict_one(self, features: dict) -> float:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        df = pd.DataFrame([features])

        if self.preprocessor is not None:
            transformed = self.preprocessor.transform(df)
            prediction = self.model.predict(transformed)
        else:
            prediction = self.model.predict(df)

        return float(prediction[0])