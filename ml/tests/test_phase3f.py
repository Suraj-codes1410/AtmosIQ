import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.src.modeling.phase3f.feature_groups import FeatureGroupManagerPhase3F
from ml.src.modeling.phase3f.models import ModelFactoryPhase3F
from ml.src.modeling.phase3f.evaluation import MetricsEvaluatorPhase3F

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_V1_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
MODELING_V2_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v2"
EXP_DIR = BASE_DIR / "ml" / "experiments" / "phase3f"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase3f_datasets_hashes_unmodified():
    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"


def test_phase3f_feature_safety_and_group_definitions():
    mgr = FeatureGroupManagerPhase3F()
    groups = mgr.build_feature_groups()

    avail_df = pd.read_csv(MODELING_V2_DIR / "feature_availability.csv")
    same_day_cols = set(avail_df[avail_df["availability_class"] == "SAME_DAY_FEATURE"]["feature_name"].tolist())

    assert len(groups["group_b_pm25_history"]) == 29

    for g_name, f_cols in groups.items():
        assert "pm25" not in f_cols
        assert "date" not in f_cols
        for feat in f_cols:
            assert feat not in same_day_cols, f"Same-day feature '{feat}' in group '{g_name}'!"


def test_phase3f_temporal_folds_integrity():
    df = pd.read_csv(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    df["date_dt"] = pd.to_datetime(df["date"])

    folds = [
        {"train_end": "2021-12-31", "eval_start": "2022-01-01", "eval_end": "2022-12-31"},
        {"train_end": "2022-12-31", "eval_start": "2023-01-01", "eval_end": "2023-12-31"},
        {"train_end": "2023-12-31", "eval_start": "2024-01-01", "eval_end": "2024-12-31"}
    ]

    for f in folds:
        tr = df[df["date_dt"] <= f["train_end"]]
        ev = df[(df["date_dt"] >= f["eval_start"]) & (df["date_dt"] <= f["eval_end"])]

        assert len(tr) > 0
        assert len(ev) > 0
        assert tr["date_dt"].max() < ev["date_dt"].min(), "Temporal overlap in fold!"


def test_phase3f_persisted_experiment_artifacts():
    metrics_file = EXP_DIR / "feature_group_metrics.csv"
    stab_file = EXP_DIR / "cross_fold_summary.csv"
    inc_file = EXP_DIR / "incremental_information.csv"
    proc_file = EXP_DIR / "process_contribution_summary.csv"

    assert metrics_file.exists()
    assert stab_file.exists()
    assert inc_file.exists()
    assert proc_file.exists()

    df_m = pd.read_csv(metrics_file)
    assert len(df_m) == 180  # 3 folds x 5 models x 12 feature groups
    assert df_m["MAE"].isnull().sum() == 0
    assert np.isinf(df_m["MAE"].values).sum() == 0
