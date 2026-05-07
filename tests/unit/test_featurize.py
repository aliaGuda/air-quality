from pathlib import Path

import pandas as pd


TRAIN_PATH = Path("data/splits/train.csv")
TEST_PATH = Path("data/splits/test.csv")
MODEL_PATH = Path("models/model.pkl")
TARGET_COLUMN = "CO(GT)"


def test_train_split_exists():
    assert TRAIN_PATH.exists()


def test_test_split_exists():
    assert TEST_PATH.exists()


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_train_split_not_empty():
    df = pd.read_csv(TRAIN_PATH)
    assert not df.empty


def test_test_split_not_empty():
    df = pd.read_csv(TEST_PATH)
    assert not df.empty


def test_target_column_exists_in_train():
    df = pd.read_csv(TRAIN_PATH)
    assert TARGET_COLUMN in df.columns


def test_target_column_exists_in_test():
    df = pd.read_csv(TEST_PATH)
    assert TARGET_COLUMN in df.columns


def test_train_has_no_missing_values():
    df = pd.read_csv(TRAIN_PATH)
    assert df.isna().sum().sum() == 0


def test_test_has_no_missing_values():
    df = pd.read_csv(TEST_PATH)
    assert df.isna().sum().sum() == 0


def test_train_and_test_have_same_columns():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    assert list(train_df.columns) == list(test_df.columns)