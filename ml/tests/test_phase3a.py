import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
SOURCE_FILE = BASE_DIR / "ml" / "data" / "processed" / "feature_dataset.csv"
FROZEN_FILE = MODELING_DIR / "feature_dataset_frozen.csv"
TRAIN_FILE = MODELING_DIR / "train.csv"
VAL_FILE = MODELING_DIR / "validation.csv"
TEST_FILE = MODELING_DIR / "test.csv"
SPLIT_MANIFEST_FILE = MODELING_DIR / "split_manifest.json"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_frozen_dataset_existence_and_hash():
    assert FROZEN_FILE.exists(), "Frozen dataset does not exist!"
    source_hash = calculate_sha256(SOURCE_FILE)
    frozen_hash = calculate_sha256(FROZEN_FILE)
    assert source_hash == frozen_hash, "Frozen dataset hash does not match source dataset hash!"


def test_dataset_rows_and_date_range():
    df = pd.read_csv(FROZEN_FILE)
    assert len(df) == 731, f"Expected 731 rows, got {len(df)}"
    assert df["date"].min() == "2023-01-01"
    assert df["date"].max() == "2024-12-31"


def test_date_uniqueness_and_ordering():
    df = pd.read_csv(FROZEN_FILE)
    assert not df["date"].duplicated().any(), "Duplicate dates found!"
    dates = pd.to_datetime(df["date"])
    assert dates.is_monotonic_increasing, "Dates are not strictly chronological!"


def test_target_and_no_nans_or_infs():
    df = pd.read_csv(FROZEN_FILE)
    assert "pm25" in df.columns, "Target column pm25 missing!"
    assert df.isnull().sum().sum() == 0, "Dataset contains NaN values!"
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert np.isinf(df[numeric_cols].values).sum() == 0, "Dataset contains infinite values!"


def test_temporal_split_disjointness_and_ordering():
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Row counts
    assert len(train_df) == 365, f"Expected 365 train rows, got {len(train_df)}"
    assert len(val_df) == 182, f"Expected 182 val rows, got {len(val_df)}"
    assert len(test_df) == 184, f"Expected 184 test rows, got {len(test_df)}"

    # Date ranges
    assert train_df["date"].min() == "2023-01-01"
    assert train_df["date"].max() == "2023-12-31"

    assert val_df["date"].min() == "2024-01-01"
    assert val_df["date"].max() == "2024-06-30"

    assert test_df["date"].min() == "2024-07-01"
    assert test_df["date"].max() == "2024-12-31"

    # Temporal Ordering
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()

    # Disjointness
    set_train = set(train_df["date"])
    set_val = set(val_df["date"])
    set_test = set(test_df["date"])

    assert len(set_train.intersection(set_val)) == 0
    assert len(set_train.intersection(set_test)) == 0
    assert len(set_val.intersection(set_test)) == 0


def test_split_manifest_and_hashes():
    assert SPLIT_MANIFEST_FILE.exists()
    with open(SPLIT_MANIFEST_FILE, "r") as f:
        manifest = json.load(f)

    assert manifest["random_shuffle"] is False
    assert manifest["train_rows"] == 365
    assert manifest["validation_rows"] == 182
    assert manifest["test_rows"] == 184

    assert manifest["sha256_hashes"]["train_csv"] == calculate_sha256(TRAIN_FILE)
    assert manifest["sha256_hashes"]["validation_csv"] == calculate_sha256(VAL_FILE)
    assert manifest["sha256_hashes"]["test_csv"] == calculate_sha256(TEST_FILE)


def test_no_target_leakage_in_predictors():
    df = pd.read_csv(FROZEN_FILE)
    predictor_cols = [c for c in df.columns if c not in ["date", "pm25"]]
    assert "pm25" not in predictor_cols, "Raw current-day target pm25 accidentally in predictor columns!"
