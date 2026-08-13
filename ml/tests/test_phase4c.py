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
EXP_DIR_4B = BASE_DIR / "ml" / "experiments" / "phase4b"
EXP_DIR_4C = BASE_DIR / "ml" / "experiments" / "phase4c"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase4c_dataset_and_model_hashes_unchanged():
    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    model_hash = calculate_sha256(PKG_DIR / "model.joblib")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    assert model_hash == "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"


def test_phase4c_shap_and_group_artifacts_exist():
    assert (EXP_DIR_4B / "shap_values" / "shap_values_all.csv").exists()
    assert (EXP_DIR_4B / "group_attributions" / "group_attributions_all.csv").exists()

    attr_groups_file = PKG_DIR / "attribution_groups.csv"
    assert attr_groups_file.exists()
    attr_df = pd.read_csv(attr_groups_file)
    assert (attr_df["attribution_group"] != "unmapped").all()


def test_phase4c_output_files_exist():
    expected_files = [
        "attribution_validation_summary.csv",
        "biomass_validation.csv",
        "wind_validation.csv",
        "meteorology_validation.csv",
        "seasonal_validation.csv",
        "temporal_validation.csv",
        "event_catalog.csv",
        "event_attributions.csv",
        "attribution_conflicts.csv",
        "confidence_scores.csv",
        "statistical_tests.csv",
        "metadata.json",
        "phase4c_report.md"
    ]
    for fname in expected_files:
        assert (EXP_DIR_4C / fname).exists(), f"Phase 4C output missing: {fname}"

    assert (BASE_DIR / "docs" / "phase4" / "phase4c_attribution_validation.md").exists()


def test_phase4c_confidence_scores_and_conflicts_validity():
    conf_df = pd.read_csv(EXP_DIR_4C / "confidence_scores.csv")
    assert len(conf_df) == 1827
    assert conf_df["evidence_score"].isin([0, 1, 2, 3]).all()
    assert conf_df["confidence_level"].isin(["Low", "Moderate", "High"]).all()
    assert conf_df["evidence_score"].isnull().sum() == 0

    conflict_df = pd.read_csv(EXP_DIR_4C / "attribution_conflicts.csv")
    assert "conflict_type" in conflict_df.columns


def test_phase4c_event_detection_validity():
    catalog_df = pd.read_csv(EXP_DIR_4C / "event_catalog.csv")
    assert len(catalog_df) > 0
    assert (catalog_df["duration_days"] >= 1).all()
    assert (catalog_df["peak_pm25"] >= 300.0).all()


def test_phase4c_all_10_plots_exist():
    plot_dir = EXP_DIR_4C / "plots"
    expected_plots = [
        "biomass_shap_vs_fire_activity.png",
        "biomass_shap_by_fire_quantile.png",
        "wind_shap_vs_wind_speed.png",
        "wind_shap_vs_ventilation.png",
        "seasonal_group_attribution_heatmap.png",
        "yearly_attribution_stability.png",
        "pollution_event_attribution_timeline.png",
        "fire_activity_and_biomass_shap_timeline.png",
        "attribution_conflict_cases.png",
        "attribution_confidence_distribution.png"
    ]
    for pname in expected_plots:
        assert (plot_dir / pname).exists(), f"Phase 4C plot missing: {pname}"
