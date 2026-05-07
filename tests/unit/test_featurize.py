from pathlib import Path

import pandas as pd

from src.features import featurize


X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")


def test_featurize_module_imports():
    assert featurize is not None


def test_processed_files_exist():
    assert X_TRAIN_PATH.exists()
    assert X_TEST_PATH.exists()
    assert Y_TRAIN_PATH.exists()
    assert Y_TEST_PATH.exists()


def test_processed_files_not_empty():
    assert not pd.read_csv(X_TRAIN_PATH).empty
    assert not pd.read_csv(X_TEST_PATH).empty
    assert not pd.read_csv(Y_TRAIN_PATH).empty
    assert not pd.read_csv(Y_TEST_PATH).empty


def test_x_train_has_no_missing_values():
    df = pd.read_csv(X_TRAIN_PATH)
    assert df.isna().sum().sum() == 0


def test_x_test_has_no_missing_values():
    df = pd.read_csv(X_TEST_PATH)
    assert df.isna().sum().sum() == 0


def test_y_train_has_no_missing_values():
    y = pd.read_csv(Y_TRAIN_PATH)
    assert y.isna().sum().sum() == 0


def test_y_test_has_no_missing_values():
    y = pd.read_csv(Y_TEST_PATH)
    assert y.isna().sum().sum() == 0


def test_train_rows_match():
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)

    assert len(X_train) == len(y_train)


def test_test_rows_match():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    assert len(X_test) == len(y_test)


def test_train_and_test_have_same_columns():
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    assert list(X_train.columns) == list(X_test.columns)


def test_processed_features_are_numeric():
    X_train = pd.read_csv(X_TRAIN_PATH)

    non_numeric_cols = X_train.select_dtypes(exclude=["number"]).columns

    assert len(non_numeric_cols) == 0