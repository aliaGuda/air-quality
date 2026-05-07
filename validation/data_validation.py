from pathlib import Path

import pandas as pd
import pandera as pa
import yaml
from pandera import Column, DataFrameSchema, Check


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_dataset(data_path: str, target_column: str) -> None:
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    schema_columns = {}

    for column in df.columns:
        if column == target_column:
            schema_columns[column] = Column(
                float,
                nullable=False,
                checks=Check(lambda s: s.notna().all()),
                coerce=True,
            )
        else:
            schema_columns[column] = Column(
                pa.Float,
                nullable=True,
                coerce=True,
                required=False,
            )

    schema = DataFrameSchema(
        schema_columns,
        checks=[
            Check(lambda dataframe: dataframe.shape[0] > 0, error="No rows found."),
            Check(lambda dataframe: dataframe.shape[1] > 1, error="Too few columns."),
        ],
        coerce=True,
    )

    schema.validate(df)

    missing_ratio = df.isna().mean().mean()

    if missing_ratio > 0.5:
        raise ValueError(f"Dataset has too many missing values: {missing_ratio:.2%}")

    print("Data validation passed successfully.")


if __name__ == "__main__":
    params = load_params()

    data_path = params["data"]["test_path"]
    target_column = params["data"]["target_column"]

    validate_dataset(data_path, target_column)