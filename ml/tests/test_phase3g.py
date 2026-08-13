import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_V1_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
MODELING_V2_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v2"
EXP_DIR = BASE_DIR / "ml" / "experiments" / "phase3g"
MODEL_DIR = BASE_DIR / "ml" / "models" / "phase3g"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase3g_dataset_hashes_unmodified():
    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"


def test_phase3g_dataset_properties_and_continuity():
    df = pd.read_csv(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    assert len(df) == 1827
    assert "pm25" in df.columns
    assert "date" in df.columns

    df["date_dt"] = pd.to_datetime(df["date"])
    diffs = df["date_dt"].diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all(), "Dates are not continuous daily!"

    # Verify no NaN / Inf
    assert df["pm25"].isnull().sum() == 0
    assert np.isinf(df["pm25"].values).sum() == 0


def test_phase3g_feature_safety_and_no_same_day_leakage():
    avail_df = pd.read_csv(MODELING_V2_DIR / "feature_availability.csv")
    same_day_cols = set(avail_df[avail_df["availability_class"] == "SAME_DAY_FEATURE"]["feature_name"])

    with open(MODEL_DIR / "feature_list.json", "r") as f:
        used_feats = json.load(f)["features"]

    assert "pm25" not in used_feats
    assert "date" not in used_feats

    for f in used_feats:
        assert f not in same_day_cols, f"Same-day feature '{f}' leaked into production model!"


def test_phase3g_optuna_trials_and_reproducibility():
    trials_file = EXP_DIR / "optuna" / "trials.csv"
    best_p_file = EXP_DIR / "optuna" / "best_params.json"

    assert trials_file.exists()
    assert best_p_file.exists()

    trials_df = pd.read_csv(trials_file)
    assert len(trials_df) > 0
    assert trials_df["val_mae"].isnull().sum() == 0


def test_phase3g_final_model_export_and_metadata():
    model_pkl = MODEL_DIR / "model.pkl"
    feature_json = MODEL_DIR / "feature_list.json"
    config_json = MODEL_DIR / "model_config.json"
    metadata_json = MODEL_DIR / "training_metadata.json"
    manifest_json = MODEL_DIR / "dataset_manifest.json"
    metrics_json = MODEL_DIR / "metrics.json"

    assert model_pkl.exists()
    assert feature_json.exists()
    assert config_json.exists()
    assert metadata_json.exists()
    assert manifest_json.exists()
    assert metrics_json.exists()

    with open(metrics_json, "r") as f:
        m = json.load(f)
    assert "dev_walk_forward_mean_mae" in m
    assert "final_test_2024_mae" in m

    final_preds_file = EXP_DIR / "predictions" / "final_test_predictions.csv"
    assert final_preds_file.exists()
    preds_df = pd.read_csv(final_preds_file)
    assert len(preds_df) == 366  # Locked 2024 leap year test set count
    assert preds_df["predicted_pm25"].isnull().sum() == 0
    assert np.isinf(preds_df["predicted_pm25"].values).sum() == 0


def test_phase3g_post_execution_hash_check():
    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
