import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml


REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "air-quality-regressor")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessing_pipeline.joblib")
PARAMS_PATH = Path("configs/params.yaml")


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.preprocessor = None

        self.model_source = "not-loaded"
        self.model_name = "not-loaded"
        self.model_version = "not-loaded"

        self.target_variable = "CO(GT)"
        self.expected_features = []

    def load_params(self) -> dict:
        if not PARAMS_PATH.exists():
            raise FileNotFoundError(f"Params file not found: {PARAMS_PATH}")

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

        self.model_source = "local"
        self.model_name = "local-model"
        self.model_version = "1"

        self.load_expected_features()

        print("Loaded local model artifact.")
        print(f"Model path: {MODEL_PATH}")
        print(f"Preprocessor path: {PREPROCESSOR_PATH}")
        print(f"Expected features: {len(self.expected_features)}")

    def load_mlflow_model(self) -> None:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"

        self.model = mlflow.pyfunc.load_model(model_uri)

        self.model_source = "mlflow_registry"
        self.model_name = REGISTERED_MODEL_NAME
        self.model_version = MLFLOW_MODEL_STAGE

        if PREPROCESSOR_PATH.exists():
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)

        self.load_expected_features()

        print(f"Loaded model from MLflow Registry: {model_uri}")
        print(f"MLflow tracking URI: {mlflow_uri}")

    def load(self) -> None:
        use_mlflow = os.getenv("USE_MLFLOW", "false").lower() == "true"

        if not use_mlflow:
            print("USE_MLFLOW is false. Loading local model artifact.")
            self.load_local_model()
            return

        try:
            self.load_mlflow_model()
        except Exception as error:
            print(f"MLflow model loading failed: {error}")
            print("Falling back to local model artifact.")
            self.load_local_model()

    def predict_one(self, features: dict) -> float:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        df = pd.DataFrame([features])

        if self.expected_features:
            df = df.reindex(columns=self.expected_features)

        if self.preprocessor is not None:
            transformed = self.preprocessor.transform(df)
            prediction = self.model.predict(transformed)
        else:
            prediction = self.model.predict(df)

        return float(prediction[0])