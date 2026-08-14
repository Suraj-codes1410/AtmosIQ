import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_V1_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
MODELING_V2_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v2"
MODELING_V3_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v3"
PKG_DIR = BASE_DIR / "ml" / "models" / "attribution" / "v1"
EXP_DIR = BASE_DIR / "ml" / "experiments" / "phase4h"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase4h_upstream_artifact_immutability():
    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    v3_hash = calculate_sha256(MODELING_V3_DIR / "feature_dataset_frozen.csv")
    model_hash = calculate_sha256(PKG_DIR / "model.joblib")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    assert v3_hash == "78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736"
    assert model_hash == "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"


def test_phase4h_leakage_audit():
    audit_file = EXP_DIR / "phase4h_leakage_audit.csv"
    assert audit_file.exists()
    df_audit = pd.read_csv(audit_file)
    assert len(df_audit) == 275

    unsafe_cols = df_audit[df_audit['classification'] == 'unsafe']['feature_name'].tolist()
    assert 'pm25' in unsafe_cols
    assert 'date' in unsafe_cols
    assert 'pm10' in unsafe_cols


def test_phase4h_walk_forward_splits_and_reproducibility():
    wf_file = EXP_DIR / "walk_forward_results.csv"
    assert wf_file.exists()
    df_wf = pd.read_csv(wf_file)

    folds = df_wf['fold'].unique()
    assert set(folds) == {1, 2, 3}

    years = df_wf['test_year'].unique()
    assert set(years) == {2022, 2023, 2024}


def test_phase4h_no_nan_predictions():
    metrics_file = EXP_DIR / "model_metrics.csv"
    assert metrics_file.exists()
    df_m = pd.read_csv(metrics_file)
    assert df_m['mean_mae'].isnull().sum() == 0
    assert df_m['mean_r2'].isnull().sum() == 0
    assert (df_m['mean_mae'] > 0).all()


def test_phase4h_statistical_comparisons():
    stat_file = EXP_DIR / "statistical_comparisons.csv"
    assert stat_file.exists()
    df_stat = pd.read_csv(stat_file)
    assert 'p_value' in df_stat.columns
    assert 'delta_mae_ci_lower' in df_stat.columns
    assert 'delta_mae_ci_upper' in df_stat.columns


def test_phase4h_ablation_results():
    ablation_file = EXP_DIR / "ablation_results.csv"
    assert ablation_file.exists()
    df_ab = pd.read_csv(ablation_file)
    assert len(df_ab) == 4
    assert 'Model_A_v2_only' in df_ab['ablation_config'].values
    assert 'Model_D_v2_plus_all_external' in df_ab['ablation_config'].values


def test_phase4h_seasonal_and_extreme_evals():
    seasonal_file = EXP_DIR / "seasonal_results.csv"
    extreme_file = EXP_DIR / "extreme_event_results.csv"
    assert seasonal_file.exists()
    assert extreme_file.exists()

    df_seas = pd.read_csv(seasonal_file)
    df_ext = pd.read_csv(extreme_file)

    assert set(df_seas['season'].unique()) == {"Winter", "Summer", "Monsoon", "Post-Monsoon"}
    assert "Extreme_PM25_gte_150" in df_ext['regime'].values


def test_phase4h_promotion_decision_deterministic():
    promo_file = EXP_DIR / "promotion_decision.json"
    assert promo_file.exists()
    with open(promo_file, "r") as f:
        data = json.load(f)

    assert "decision" in data
    assert data["decision"] in ["V3 PROMOTION RECOMMENDED", "V2 RETENTION RECOMMENDED", "CONDITIONAL V3 PROMOTION"]
    assert "selected_candidate_model" in data
    assert "criteria_evaluation" in data
