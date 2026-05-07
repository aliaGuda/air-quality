from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema


TARGET_COLUMN = "CO(GT)"

TRAIN_PATH = Path("data/splits/train.csv")
TEST_PATH = Path("data/splits/test.csv")


def validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{path} is empty")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from {path}")

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
    print(f"Data validation passed for {path}")


def validate_data() -> None:
    validate_file(TRAIN_PATH)
    validate_file(TEST_PATH)
    print("All data validation checks passed.")


if __name__ == "__main__":
    validate_data()