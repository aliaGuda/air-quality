from pathlib import Path

import pandas as pd
import yaml


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
        raise ValueError(f"Target column '{target_column}' not found.")

    missing_ratio = df.isna().mean().mean()

    if missing_ratio > 0.5:
        raise ValueError(f"Too many missing values: {missing_ratio:.2%}")

    print("Data validation passed.")


if __name__ == "__main__":
    params = load_params()

    validate_dataset(
        data_path=params["data"]["test_path"],
        target_column=params["data"]["target_column"],
    )