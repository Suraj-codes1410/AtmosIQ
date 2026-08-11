import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.src.modeling.baselines import BaselineEvaluator

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
EXP_DIR = BASE_DIR / "ml" / "experiments" / "phase3b1"
PRED_DIR = EXP_DIR / "predictions"
PHASE2_FILE = BASE_DIR / "ml" / "data" / "processed" / "feature_dataset.csv"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_feature_whitelist_and_exclusions():
    evaluator = BaselineEvaluator()
    safe_features = evaluator.load_feature_whitelist()

    assert len(safe_features) > 0
    assert "date" not in safe_features, "Date column found in prediction whitelist!"
    assert "pm25" not in safe_features, "Target pm25 found in prediction whitelist!"

    df_avail = pd.read_csv(MODELING_DIR / "feature_availability.csv")
    same_day_features = df_avail[df_avail["availability_class"] == "SAME_DAY_FEATURE"]["feature_name"].tolist()

    for s_feat in same_day_features:
        assert s_feat not in safe_features, f"Same-day feature '{s_feat}' accidentally in safe feature whitelist!"


def test_split_row_counts_and_dates():
    evaluator = BaselineEvaluator()
    safe_features = evaluator.load_feature_whitelist()
    (
        train_raw, val_raw, test_raw,
        X_train, y_train, dates_train,
        X_val, y_val, dates_val,
        X_test, y_test, dates_test
    ) = evaluator.load_and_validate_splits(safe_features)

    assert len(X_train) == 365
    assert len(X_val) == 182
    assert len(X_test) == 184

    assert dates_train.iloc[0] == "2023-01-01"
    assert dates_train.iloc[-1] == "2023-12-31"

    assert dates_val.iloc[0] == "2024-01-01"
    assert dates_val.iloc[-1] == "2024-06-30"

    assert dates_test.iloc[0] == "2024-07-01"
    assert dates_test.iloc[-1] == "2024-12-31"


def test_persistence_baseline_boundary_values():
    evaluator = BaselineEvaluator()
    safe_features = evaluator.load_feature_whitelist()
    (
        train_raw, val_raw, test_raw,
        X_train, y_train, dates_train,
        X_val, y_val, dates_val,
        X_test, y_test, dates_test
    ) = evaluator.load_and_validate_splits(safe_features)

    preds = evaluator.generate_persistence_predictions(train_raw, val_raw, test_raw)

    p_val = preds["validation"]
    p_test = preds["test"]

    # Val day 1 (2024-01-01) persistence prediction must equal Train final day (2023-12-31) actual PM2.5
    last_train_actual = y_train.iloc[-1]
    assert p_val.iloc[0] == last_train_actual, f"Val day 1 persistence error! Expected {last_train_actual}, got {p_val.iloc[0]}"

    # Test day 1 (2024-07-01) persistence prediction must equal Val final day (2024-06-30) actual PM2.5
    last_val_actual = y_val.iloc[-1]
    assert p_test.iloc[0] == last_val_actual, f"Test day 1 persistence error! Expected {last_val_actual}, got {p_test.iloc[0]}"


def test_linear_and_ridge_model_outputs():
    evaluator = BaselineEvaluator()
    safe_features = evaluator.load_feature_whitelist()
    (
        train_raw, val_raw, test_raw,
        X_train, y_train, dates_train,
        X_val, y_val, dates_val,
        X_test, y_test, dates_test
    ) = evaluator.load_and_validate_splits(safe_features)

    # Linear Regression
    lr_pipeline, lr_preds = evaluator.train_linear_regression(X_train, y_train, X_val, X_test)
    assert len(lr_preds["train"]) == 365
    assert len(lr_preds["validation"]) == 182
    assert len(lr_preds["test"]) == 184

    # Ridge Regression
    ridge_pipeline, ridge_preds = evaluator.train_ridge_regression(X_train, y_train, X_val, X_test, alpha=1.0)
    assert len(ridge_preds["train"]) == 365
    assert len(ridge_preds["validation"]) == 182
    assert len(ridge_preds["test"]) == 184

    for model_name, preds_dict in [("Linear", lr_preds), ("Ridge", ridge_preds)]:
        for split_name in ["train", "validation", "test"]:
            p = preds_dict[split_name]
            assert np.isnan(p).sum() == 0, f"{model_name} {split_name} predictions contain NaNs!"
            assert np.isinf(p).sum() == 0, f"{model_name} {split_name} predictions contain Infs!"


def test_metrics_integrity():
    metrics_file = EXP_DIR / "metrics.csv"
    assert metrics_file.exists(), "metrics.csv does not exist!"

    df_metrics = pd.read_csv(metrics_file)
    assert len(df_metrics) == 9  # 3 models x 3 splits

    for col in ["MAE", "RMSE", "R2", "Median_AE"]:
        assert df_metrics[col].isnull().sum() == 0
        assert np.isinf(df_metrics[col].values).sum() == 0


def test_phase2_dataset_untouched():
    h = calculate_sha256(PHASE2_FILE)
    frozen_h = calculate_sha256(MODELING_DIR / "feature_dataset_frozen.csv")
    assert h == frozen_h, "Original Phase 2 dataset or Phase 3A frozen dataset modified!"
