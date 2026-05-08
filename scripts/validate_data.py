from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column


DATA_PATHS = [
    Path("data/splits/train.csv"),
    Path("data/splits/test.csv"),
    Path("data/splits/reference.csv"),
    Path("data/splits/production.csv"),
]

TARGET_COLUMN = "CO(GT)"


def validate_single_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing in {path}: {TARGET_COLUMN}")

    schema = pa.DataFrameSchema(
        {
            TARGET_COLUMN: Column(
                float,
                checks=[
                    Check.greater_than_or_equal_to(
                        0,
                        error="CO(GT) must be >= 0.",
                    ),
                ],
                nullable=False,
                coerce=True,
            )
        },
        strict=False,
        coerce=True,
    )

    schema.validate(df)

    feature_cols = [column for column in df.columns if column != TARGET_COLUMN]

    missing_ratio = df[feature_cols].isna().mean().mean()
    if missing_ratio > 0.30:
        raise ValueError(
            f"Too many missing values in feature columns for {path}: "
            f"{missing_ratio:.2%}"
        )

    duplicate_ratio = df.duplicated().mean()
    if duplicate_ratio > 0.20:
        raise ValueError(f"Too many duplicate rows in {path}: {duplicate_ratio:.2%}")

    print(f"Data validation passed: {path}")


def main() -> None:
    for path in DATA_PATHS:
        validate_single_file(path)

    print("All data validation checks passed successfully.")


if __name__ == "__main__":
    main()