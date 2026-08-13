import json
import joblib
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_V1_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v1"
MODELING_V2_DIR = BASE_DIR / "ml" / "data" / "modeling" / "v2"
PKG_DIR = BASE_DIR / "ml" / "models" / "attribution" / "v1"
EXP_DIR = BASE_DIR / "ml" / "experiments" / "phase4b"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase4b_frozen_package_and_dataset_hashes_unchanged():
    model_joblib = PKG_DIR / "model.joblib"
    assert model_joblib.exists()

    model_hash = calculate_sha256(model_joblib)
    manifest_json = PKG_DIR / "model_manifest.json"
    with open(manifest_json, "r") as f:
        manifest = json.load(f)

    assert model_hash == manifest["model_sha256"], "model.joblib SHA256 mismatch!"

    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"


def test_phase4b_shap_output_shape_and_feature_coverage():
    wide_shap_file = EXP_DIR / "shap_values" / "shap_values_all.csv"
    assert wide_shap_file.exists(), f"Wide SHAP output missing: {wide_shap_file}"

    wide_df = pd.read_csv(wide_shap_file)
    assert len(wide_df) == 1827
    assert len(wide_df["date"].unique()) == 1827

    feat_reg = pd.read_csv(PKG_DIR / "feature_registry.csv").sort_values("model_feature_order")
    f_cols = feat_reg["feature_name"].tolist()

    assert len(f_cols) == 147
    assert "pm25" not in f_cols
    assert "date" not in f_cols

    shap_cols = [c for c in wide_df.columns if c.startswith("shap_")]
    assert len(shap_cols) == 147

    # Check no NaN / Inf
    assert wide_df[shap_cols].isnull().sum().sum() == 0
    assert np.isinf(wide_df[shap_cols].values).sum() == 0


def test_phase4b_additivity_and_reconstruction_tolerance():
    summary_file = EXP_DIR / "summaries" / "reconstruction_summary.csv"
    assert summary_file.exists()

    sum_df = pd.read_csv(summary_file).set_index("metric")
    max_err = float(sum_df.loc["max_absolute_error", "value"])

    assert max_err <= 1e-4, f"SHAP reconstruction max error {max_err} exceeds 1e-4!"
    assert sum_df.loc["max_absolute_error", "status"] == "PASS"


def test_phase4b_group_attributions_and_reconstruction():
    group_file = EXP_DIR / "group_attributions" / "group_attributions_all.csv"
    assert group_file.exists()

    grp_df = pd.read_csv(group_file)
    assert len(grp_df) == 1827

    expected_groups = ["pm25_persistence_shap", "meteorology_shap", "wind_ventilation_shap", "biomass_burning_shap", "calendar_seasonal_shap"]
    for col in expected_groups:
        assert col in grp_df.columns, f"Missing group SHAP column: {col}"

    max_grp_err = grp_df["reconstruction_error"].abs().max()
    assert max_grp_err <= 1e-4, f"Group reconstruction max error {max_grp_err} exceeds 1e-4!"


def test_phase4b_global_and_temporal_summaries_exist():
    assert (EXP_DIR / "summaries" / "global_feature_importance.csv").exists()
    assert (EXP_DIR / "summaries" / "global_group_importance.csv").exists()
    assert (EXP_DIR / "summaries" / "high_pollution_analysis.csv").exists()
    assert (EXP_DIR / "summaries" / "temporal_stability.csv").exists()
    assert (EXP_DIR / "summaries" / "extreme_caution_cases.csv").exists()
    assert (EXP_DIR / "metadata.json").exists()


def test_phase4b_plots_and_reports_generated():
    plots_dir = EXP_DIR / "plots"
    assert (plots_dir / "global_feature_importance.png").exists()
    assert (plots_dir / "shap_summary_beeswarm.png").exists()
    assert (plots_dir / "global_group_importance.png").exists()
    assert (plots_dir / "high_vs_normal_pollution_shap.png").exists()
    assert (plots_dir / "waterfall_low_pm25.png").exists()
    assert (plots_dir / "waterfall_episode_post_monsoon.png").exists()
    assert (BASE_DIR / "docs" / "phase4" / "phase4b_treeshap.md").exists()
