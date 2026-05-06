import os
import yaml
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures


def load_params(path="configs/params.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def build_preprocessing_pipeline(X, params):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy=params["preprocessing"]["imputation_strategy"])),
        ("scaler", StandardScaler()),
    ]

    if params["preprocessing"]["polynomial_features"]["enabled"]:
        numeric_steps.append(
            (
                "poly",
                PolynomialFeatures(
                    degree=params["preprocessing"]["polynomial_features"]["degree"],
                    include_bias=params["preprocessing"]["polynomial_features"]["include_bias"],
                ),
            )
        )

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=params["preprocessing"]["categorical_imputation_strategy"]
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def main():
    params = load_params()

    train_path = params["data"]["train_path"]
    test_path = params["data"]["test_path"]
    target_column = params["data"]["target_column"]
    pipeline_path = params["serving"]["preprocessing_pipeline_path"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    preprocessor = build_preprocessing_pipeline(X_train, params)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    pd.DataFrame(X_train_processed).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test_processed).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

    joblib.dump(preprocessor, pipeline_path)

    print(f"Preprocessing pipeline saved to {pipeline_path}")
    print(f"X_train shape: {X_train_processed.shape}")
    print(f"X_test shape: {X_test_processed.shape}")


if __name__ == "__main__":
    main()