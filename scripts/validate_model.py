from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessing_pipeline.joblib")
TEST_PATH = Path("data/splits/test.csv")
PARAMS_PATH = Path("configs/params.yaml")


def load_params() -> dict:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing params file: {PARAMS_PATH}")

    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    params = load_params()

    target_column = params["data"]["target_column"]
    min_r2 = params.get("model_validation", {}).get("min_r2", 0.30)
    max_mae = params.get("model_validation", {}).get("max_mae", 2.0)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessing pipeline: {PREPROCESSOR_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test dataset: {TEST_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    test_df = pd.read_csv(TEST_PATH)

    if target_column not in test_df.columns:
        raise ValueError(f"Target column missing from test set: {target_column}")

    test_df = test_df.dropna(subset=[target_column])

    if test_df.empty:
        raise ValueError("Test dataset is empty after dropping missing target rows.")

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    X_test_transformed = preprocessor.transform(X_test)
    predictions = model.predict(X_test_transformed)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print("Model validation results:")
    print(f"R2:  {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

    if r2 < min_r2:
        raise ValueError(f"Model R2 too low: {r2:.4f} < {min_r2}")

    if mae > max_mae:
        raise ValueError(f"Model MAE too high: {mae:.4f} > {max_mae}")

    print("Model validation passed successfully.")


if __name__ == "__main__":
    main()