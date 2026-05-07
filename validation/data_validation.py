from pathlib import Path

import pandas as pd
from pandera import Check, Column, DataFrameSchema


X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")


def validate_features(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{path} is empty")

    if df.isna().sum().sum() > 0:
        raise ValueError(f"{path} contains missing values")

    schema = DataFrameSchema(
        {
            column: Column(
                float,
                nullable=False,
                coerce=True,
                checks=Check(lambda series: series.notna().all()),
            )
            for column in df.columns
        }
    )

    schema.validate(df, lazy=True)
    print(f"Feature validation passed for {path}")


def validate_target(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    y = pd.read_csv(path)

    if y.empty:
        raise ValueError(f"{path} is empty")

    if y.isna().sum().sum() > 0:
        raise ValueError(f"{path} contains missing values")

    print(f"Target validation passed for {path}")


def validate_data() -> None:
    validate_features(X_TRAIN_PATH)
    validate_features(X_TEST_PATH)
    validate_target(Y_TRAIN_PATH)
    validate_target(Y_TEST_PATH)

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train row counts do not match")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test row counts do not match")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test columns do not match")

    print("All data validation checks passed.")


if __name__ == "__main__":
    validate_data()