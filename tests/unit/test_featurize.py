import pandas as pd

from src.features.featurize import (
    build_preprocessing_pipeline,
    get_feature_types,
    split_features_target,
)


def sample_params():
    return {
        "preprocessing": {
            "imputation_strategy": "median",
            "categorical_imputation_strategy": "most_frequent",
            "scaling": "standard",
            "encoding": "onehot",
            "polynomial_features": {
                "enabled": True,
                "degree": 2,
                "include_bias": False,
            },
            "feature_selection": {
                "enabled": False,
                "k_best": 3,
            },
        }
    }


def test_split_features_target():
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [10, 20, 30],
        }
    )

    X, y = split_features_target(df, "target")

    assert "target" not in X.columns
    assert y.tolist() == [10, 20, 30]


def test_get_feature_types():
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0],
            "cat1": ["a", "b", "a"],
        }
    )

    numeric_features, categorical_features = get_feature_types(df)

    assert "num1" in numeric_features
    assert "cat1" in categorical_features


def test_pipeline_handles_missing_values_and_categoricals():
    X = pd.DataFrame(
        {
            "num1": [1.0, None, 3.0, 4.0],
            "num2": [10.0, 20.0, None, 40.0],
            "cat1": ["a", "b", None, "a"],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    pipeline = build_preprocessing_pipeline(X, sample_params())
    transformed = pipeline.fit_transform(X, y)

    assert transformed.shape[0] == 4
    assert pd.DataFrame(transformed).isna().sum().sum() == 0