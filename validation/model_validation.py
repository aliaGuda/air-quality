from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_model(
    model_path: str,
    test_data_path: str,
    target_column: str,
    min_r2: float,
    max_mae: float,
) -> None:
    model_file = Path(model_path)
    test_file = Path(test_data_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    model = joblib.load(model_file)
    test_df = pd.read_csv(test_file)

    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in test data.")

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

    if r2 < min_r2:
        raise ValueError(f"Model R2 is too low: {r2:.4f} < {min_r2}")

    if mae > max_mae:
        raise ValueError(f"Model MAE is too high: {mae:.4f} > {max_mae}")

    print("Model validation passed successfully.")


if __name__ == "__main__":
    params = load_params()

    validate_model(
        model_path=params["model"]["model_path"],
        test_data_path=params["data"]["test_path"],
        target_column=params["data"]["target_column"],
        min_r2=params["validation"]["min_r2"],
        max_mae=params["validation"]["max_mae"],
    )