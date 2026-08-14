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
EXP_DIR_4D = BASE_DIR / "ml" / "experiments" / "phase4d"


def calculate_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_phase4d_frozen_dataset_and_model_hashes_unchanged():
    v1_hash = calculate_sha256(MODELING_V1_DIR / "feature_dataset_frozen.csv")
    v2_hash = calculate_sha256(MODELING_V2_DIR / "feature_dataset_frozen.csv")
    model_hash = calculate_sha256(PKG_DIR / "model.joblib")

    assert v1_hash == "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    assert v2_hash == "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    assert model_hash == "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"


def test_phase4d_feature_mapping_and_registry():
    feat_reg = pd.read_csv(PKG_DIR / "feature_registry.csv")
    attr_df = pd.read_csv(PKG_DIR / "attribution_groups.csv")

    assert len(feat_reg) == 147
    assert len(attr_df) == 147
    assert (attr_df["attribution_group"] != "unmapped").all()


def test_phase4d_scenario_registry_and_outputs():
    registry_json = EXP_DIR_4D / "scenario_registry.json"
    assert registry_json.exists()

    with open(registry_json, "r") as f:
        scenarios = json.load(f)

    assert "biomass_low" in scenarios
    assert "wind_dispersion" in scenarios
    assert "combined_all_favorable" in scenarios

    expected_csvs = [
        "counterfactual_results.csv",
        "group_counterfactual_summary.csv",
        "interaction_analysis.csv",
        "event_counterfactuals.csv",
        "plausibility_checks.csv",
        "ood_analysis.csv",
        "confidence_scores.csv",
        "shap_counterfactual_consistency.csv",
        "metadata.json",
        "phase4d_report.md"
    ]
    for fname in expected_csvs:
        assert (EXP_DIR_4D / fname).exists(), f"Phase 4D output missing: {fname}"

    assert (BASE_DIR / "docs" / "phase4" / "phase4d_counterfactuals.md").exists()


def test_phase4d_feature_isolation_and_delta_calculation():
    results_df = pd.read_csv(EXP_DIR_4D / "counterfactual_results.csv")
    assert len(results_df) > 0
    assert results_df["delta_prediction"].isnull().sum() == 0

    # Delta definition check: delta = counterfactual - observed
    calc_delta = results_df["prediction_counterfactual"] - results_df["prediction_observed"]
    assert np.allclose(results_df["delta_prediction"], calc_delta, atol=1e-5)


def test_phase4d_all_10_plots_exist():
    plot_dir = EXP_DIR_4D / "plots"
    expected_plots = [
        "counterfactual_effect_distribution.png",
        "biomass_counterfactual_effect.png",
        "wind_counterfactual_effect.png",
        "meteorology_counterfactual_effect.png",
        "group_counterfactual_comparison.png",
        "interaction_effects.png",
        "event_counterfactual_effects.png",
        "observed_vs_counterfactual.png",
        "ood_counterfactuals.png",
        "confidence_distribution.png"
    ]
    for pname in expected_plots:
        assert (plot_dir / pname).exists(), f"Phase 4D plot missing: {pname}"


def test_phase4d_scientific_disclaimer_present():
    report_file = BASE_DIR / "docs" / "phase4" / "phase4d_counterfactuals.md"
    content = open(report_file, "r", encoding="utf-8").read()

    assert "Predictive Importance != SHAP Attribution != Counterfactual Model Response != Causal Effect != Actual Emission Contribution" in content
