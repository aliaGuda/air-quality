from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check


DATA_PATH = Path("data/processed/cleaned.csv")
TARGET_COLUMN = "CO(GT)"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")

    schema = pa.DataFrameSchema(
        {
            TARGET_COLUMN: Column(
                float,
                checks=[
                    Check(lambda s: s.notna().all(), error="Target column contains missing values."),
                    Check.greater_than_or_equal_to(0),
                ],
                coerce=True,
            )
        },
        strict=False,
        coerce=True,
    )

    schema.validate(df)

    missing_ratio = df.isna().mean().mean()
    if missing_ratio > 0.30:
        raise ValueError(f"Too many missing values: {missing_ratio:.2%}")

    duplicate_ratio = df.duplicated().mean()
    if duplicate_ratio > 0.20:
        raise ValueError(f"Too many duplicate rows: {duplicate_ratio:.2%}")

    print("Data validation passed successfully.")


if __name__ == "__main__":
    main()