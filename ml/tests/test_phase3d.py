import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.src.modeling.feature_sets import FeatureSetManager
from ml.src.modeling.tuned_models import TunedModelEvaluator

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
EXP_DIR = BASE_DIR / "ml" / "experiments" / "phase3d"
FROZEN_FILE = MODELING_DIR / "feature_dataset_frozen.csv"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase3d_frozen_dataset_hash_unmodified():
    expected_hash = "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    actual_hash = calculate_sha256(FROZEN_FILE)
    assert actual_hash == expected_hash, "Phase 3A frozen dataset modified during Phase 3D!"


def test_phase3d_split_row_counts_and_dates():
    tr = pd.read_csv(MODELING_DIR / "train.csv")
    val = pd.read_csv(MODELING_DIR / "validation.csv")
    te = pd.read_csv(MODELING_DIR / "test.csv")

    assert len(tr) == 365
    assert len(val) == 182
    assert len(te) == 184

    assert tr["date"].iloc[0] == "2023-01-01"
    assert tr["date"].iloc[-1] == "2023-12-31"

    assert val["date"].iloc[0] == "2024-01-01"
    assert val["date"].iloc[-1] == "2024-06-30"

    assert te["date"].iloc[0] == "2024-07-01"
    assert te["date"].iloc[-1] == "2024-12-31"


def test_phase3d_feature_sets_integrity():
    mgr = FeatureSetManager()
    fsets = mgr.get_phase3d_feature_sets()

    assert len(fsets["set_b_pm25_history"]) == 29
    assert len(fsets["domain_reduced"]) == 15
    assert len(fsets["set_b_plus_core_environment"]) == 34

    avail_df = pd.read_csv(MODELING_DIR / "feature_availability.csv")
    same_day_cols = avail_df[avail_df["availability_class"] == "SAME_DAY_FEATURE"]["feature_name"].tolist()

    for s_name, f_cols in fsets.items():
        assert "date" not in f_cols
        assert "pm25" not in f_cols
        for feat in f_cols:
            assert feat not in same_day_cols, f"Same-day feature '{feat}' in set '{s_name}'!"


def test_phase3d_best_params_and_optimization_results():
    params_file = EXP_DIR / "best_parameters" / "best_params.json"
    results_file = EXP_DIR / "metrics" / "optimization_results.csv"

    assert params_file.exists()
    assert results_file.exists()

    with open(params_file, "r") as f:
        p_map = json.load(f)

    assert len(p_map) == 12  # 4 models x 3 feature sets

    df_trials = pd.read_csv(results_file)
    assert len(df_trials) == 600  # 12 studies x 50 trials
    assert df_trials["val_mae"].isnull().sum() == 0
    assert np.isinf(df_trials["val_mae"].values).sum() == 0


def test_phase3d_model_metrics_non_null():
    val_m = pd.read_csv(EXP_DIR / "metrics" / "validation_metrics.csv")
    test_m = pd.read_csv(EXP_DIR / "metrics" / "test_metrics.csv")

    assert len(val_m) == 12
    assert len(test_m) == 12

    for col in ["MAE", "RMSE", "R2", "Median_AE"]:
        assert val_m[col].isnull().sum() == 0
        assert np.isinf(val_m[col].values).sum() == 0
        assert test_m[col].isnull().sum() == 0
        assert np.isinf(test_m[col].values).sum() == 0
