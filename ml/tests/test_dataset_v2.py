import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_V1_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
MODELING_V2_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v2"
KAGGLE_DIR = BASE_DIR / "kaggle"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_dataset_v1_remains_immutable():
    v1_frozen = MODELING_V1_DIR / "feature_dataset_frozen.csv"
    expected_v1_hash = "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    actual_hash = calculate_sha256(v1_frozen)
    assert actual_hash == expected_v1_hash, "Dataset v1 modified during Phase 3E!"


def test_dataset_v2_total_row_count_and_dates():
    v2_frozen = MODELING_V2_DIR / "feature_dataset_frozen.csv"
    assert v2_frozen.exists(), f"Dataset v2 frozen copy missing: {v2_frozen}"

    df = pd.read_csv(v2_frozen)
    assert len(df) == 1827, f"Expected 1,827 rows for 2020-2024, got {len(df)}"

    df["date_dt"] = pd.to_datetime(df["date"])
    assert df["date_dt"].is_monotonic_increasing, "Dataset v2 is not chronologically sorted!"
    assert df["date"].duplicated().sum() == 0, "Duplicate dates found in Dataset v2!"

    assert df["date"].iloc[0] == "2020-01-01"
    assert df["date"].iloc[-1] == "2024-12-31"


def test_dataset_v2_target_validity_and_no_nans():
    df = pd.read_csv(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    assert "pm25" in df.columns, "Target 'pm25' missing from Dataset v2!"
    assert (df["pm25"] >= 0).all(), "Negative PM2.5 target values found!"

    num_cols = df.select_dtypes(include=[np.number]).columns
    assert df[num_cols].isnull().sum().sum() == 0, "NaNs found in Dataset v2 numerical columns!"
    assert np.isinf(df[num_cols].values).sum() == 0, "Infinite values found in Dataset v2!"


def test_dataset_v2_splits_integrity():
    tr = pd.read_csv(MODELING_V2_DIR / "train.csv")
    val = pd.read_csv(MODELING_V2_DIR / "validation.csv")
    te = pd.read_csv(MODELING_V2_DIR / "test.csv")

    assert len(tr) == 1096, f"Expected 1,096 train rows, got {len(tr)}"
    assert len(val) == 365, f"Expected 365 validation rows, got {len(val)}"
    assert len(te) == 366, f"Expected 366 test rows, got {len(te)}"

    assert tr["date"].iloc[0] == "2020-01-01"
    assert tr["date"].iloc[-1] == "2022-12-31"

    assert val["date"].iloc[0] == "2023-01-01"
    assert val["date"].iloc[-1] == "2023-12-31"

    assert te["date"].iloc[0] == "2024-01-01"
    assert te["date"].iloc[-1] == "2024-12-31"


def test_dataset_v2_manifest_and_sha256_hash():
    manifest_file = MODELING_V2_DIR / "dataset_manifest.json"
    hash_file = MODELING_V2_DIR / "dataset_hash.txt"

    assert manifest_file.exists()
    assert hash_file.exists()

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    with open(hash_file, "r") as f:
        stored_hash = f.read().strip()

    actual_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    assert actual_hash == stored_hash
    assert manifest["sha256"] == stored_hash
    assert manifest["total_rows"] == 1827


def test_kaggle_public_release_artifacts():
    kaggle_csv = KAGGLE_DIR / "atmosiq_delhi_pm25.csv"
    dict_csv = KAGGLE_DIR / "atmosiq_data_dictionary.csv"
    readme = KAGGLE_DIR / "README.md"
    license_file = KAGGLE_DIR / "LICENSE"

    assert kaggle_csv.exists()
    assert dict_csv.exists()
    assert readme.exists()
    assert license_file.exists()

    df_k = pd.read_csv(kaggle_csv)
    assert len(df_k) == 1827
