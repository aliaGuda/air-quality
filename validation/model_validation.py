from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import r2_score


MODEL_PATH = Path("models/preprocessing_pipeline.joblib")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")

MIN_R2_SCORE = 0.30


def validate_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    if not X_TEST_PATH.exists():
        raise FileNotFoundError(f"Missing X_test file: {X_TEST_PATH}")

    if not Y_TEST_PATH.exists():
        raise FileNotFoundError(f"Missing y_test file: {Y_TEST_PATH}")

    model = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)

    print(f"Model R2 score: {score:.4f}")

    if score < MIN_R2_SCORE:
        raise AssertionError(
            f"Model validation failed. R2={score:.4f}, "
            f"minimum required={MIN_R2_SCORE}"
        )

    print("Model validation passed.")


if __name__ == "__main__":
    validate_model()