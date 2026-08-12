import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.src.modeling.feature_audit import FeatureAuditEngine
from ml.src.modeling.feature_selection import FeatureSelectionEngine

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
EXP_DIR = BASE_DIR / "ml" / "experiments" / "phase3c"
PHASE2_FILE = BASE_DIR / "ml" / "data" / "processed" / "feature_dataset.csv"
FROZEN_FILE = MODELING_DIR / "feature_dataset_frozen.csv"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase3c_feature_discovery_and_exclusions():
    df_avail, safe_features = FeatureAuditEngine.load_safe_features()
    assert len(safe_features) == 201
    assert "date" not in safe_features
    assert "pm25" not in safe_features

    same_day = df_avail[df_avail["availability_class"] == "SAME_DAY_FEATURE"]["feature_name"].tolist()
    for s in same_day:
        assert s not in safe_features


def test_phase3c_frozen_dataset_hash_unmodified():
    expected_hash = "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    actual_hash = calculate_sha256(FROZEN_FILE)
    assert actual_hash == expected_hash, "Phase 3A frozen dataset modified during Phase 3C!"


def test_phase3c_split_dimensions():
    tr = pd.read_csv(MODELING_DIR / "train.csv")
    val = pd.read_csv(MODELING_DIR / "validation.csv")
    te = pd.read_csv(MODELING_DIR / "test.csv")

    assert len(tr) == 365
    assert len(val) == 182
    assert len(te) == 184


def test_phase3c_selected_feature_sets_validity():
    registry_file = EXP_DIR / "feature_set_registry.json"
    assert registry_file.exists()

    with open(registry_file, "r") as f:
        reg_data = json.load(f)

    _, safe_features = FeatureAuditEngine.load_safe_features()
    safe_set = set(safe_features)

    for set_name, set_info in reg_data.items():
        f_list = set_info["features"]

        # 1. No duplicates within set
        assert len(f_list) == len(set(f_list)), f"Duplicate features in set '{set_name}'!"

        # 2. All features exist in safe whitelist
        for feat in f_list:
            assert feat in safe_set, f"Unsafe or unknown feature '{feat}' in set '{set_name}'!"
            assert feat != "date"
            assert feat != "pm25"


def test_phase3c_model_metrics_integrity():
    metrics_file = EXP_DIR / "model_metrics.csv"
    assert metrics_file.exists()

    df_m = pd.read_csv(metrics_file)
    assert len(df_m) > 0

    for col in ["MAE", "RMSE", "R2", "Median_AE"]:
        assert df_m[col].isnull().sum() == 0
        assert np.isinf(df_m[col].values).sum() == 0
