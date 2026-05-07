from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import r2_score


MODEL_PATH = Path("models/model.pkl")
TEST_PATH = Path("data/splits/test.csv")
TARGET_COLUMN = "CO(GT)"

MIN_R2_SCORE = 0.30


def validate_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_PATH}")

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from test data")

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

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