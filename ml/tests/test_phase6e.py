import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestPhase6E:

    v1_path = ROOT_DIR / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
    v2_path = ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
    v3_path = ROOT_DIR / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
    ctrl_model_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "model.joblib"
    v3_prod_model_path = ROOT_DIR / "ml" / "models" / "production" / "v3" / "model.joblib"
    feat_reg_path = ROOT_DIR / "ml" / "models" / "production" / "v3" / "feature_registry.csv"
    prod_unc_path = ROOT_DIR / "ml" / "uncertainty" / "production" / "v1" / "uncertainty_method.json"
    exp_dir = ROOT_DIR / "ml" / "experiments" / "phase6e"

    v1_expected_hash = "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    v2_expected_hash = "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    v3_expected_hash = "78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736"
    ctrl_expected_hash = "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"
    v3_model_expected_hash = "9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210"

    def calculate_sha256(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def test_upstream_lineage_hashes(self):
        assert self.calculate_sha256(self.v1_path) == self.v1_expected_hash
        assert self.calculate_sha256(self.v2_path) == self.v2_expected_hash
        assert self.calculate_sha256(self.v3_path) == self.v3_expected_hash
        assert self.calculate_sha256(self.ctrl_model_path) == self.ctrl_expected_hash
        assert self.calculate_sha256(self.v3_prod_model_path) == self.v3_model_expected_hash

    def test_feature_registry_and_uncertainty_layer_immutability(self):
        df_feat = pd.read_csv(self.feat_reg_path)
        assert len(df_feat) == 35, f"Expected 35 features, found {len(df_feat)}"
        assert self.prod_unc_path.exists(), "Phase 6D production uncertainty layer missing!"

    def test_shap_uncertainty_output_validity(self):
        shap_csv = self.exp_dir / "shap_uncertainty.csv"
        feat_sum_csv = self.exp_dir / "shap_feature_summary.csv"
        if shap_csv.exists() and feat_sum_csv.exists():
            df_feat = pd.read_csv(feat_sum_csv)
            assert len(df_feat) == 35
            assert (df_feat["mean_sign_stability"] >= 0.0).all()
            assert (df_feat["mean_sign_stability"] <= 1.0).all()
            assert (df_feat["mean_absolute_shap"] >= 0.0).all()

    def test_group_attribution_uncertainty(self):
        grp_csv = self.exp_dir / "group_attribution_uncertainty.csv"
        if grp_csv.exists():
            df_grp = pd.read_csv(grp_csv)
            groups = set(df_grp["feature_group"].unique())
            assert "pm25_persistence" in groups
            assert "biomass_burning" in groups
            assert "wind_ventilation" in groups
            assert "meteorology" in groups
            assert "external_environmental" in groups

    def test_counterfactual_uncertainty_and_directional_stability(self):
        cf_csv = self.exp_dir / "counterfactual_uncertainty.csv"
        cf_cases_csv = self.exp_dir / "counterfactual_cases.csv"
        if cf_csv.exists() and cf_cases_csv.exists():
            df_cf = pd.read_csv(cf_csv)
            assert (df_cf["cf_ensemble_mean"] >= 0.0).all()
            assert (df_cf["directional_stability"] >= 0.0).all()
            assert (df_cf["directional_stability"] <= 1.0).all()

    def test_ood_analysis_validity(self):
        ood_csv = self.exp_dir / "ood_uncertainty.csv"
        if ood_csv.exists():
            df_ood = pd.read_csv(ood_csv)
            assert (df_ood["ood_score"] >= 0.0).all()
            assert len(df_ood) >= 1000

    def test_leakage_audit_and_reproducibility(self):
        leak_csv = self.exp_dir / "phase6e_leakage_audit.csv"
        repro_csv = self.exp_dir / "phase6e_reproducibility.csv"
        if leak_csv.exists() and repro_csv.exists():
            df_leak = pd.read_csv(leak_csv)
            df_repro = pd.read_csv(repro_csv)
            assert (df_leak["status"] == "PASS").all()
            assert df_leak["violations_detected"].sum() == 0
            assert (df_repro["status"] == "PASS").all()

    def test_publication_plots_generated(self):
        plot_dir = self.exp_dir / "plots"
        if plot_dir.exists():
            expected = [
                "1_shap_attribution_uncertainty_ranking.png",
                "2_mean_shap_with_90pct_intervals.png",
                "3_shap_sign_stability_by_feature.png",
                "4_group_attribution_uncertainty.png",
                "5_counterfactual_delta_distributions.png",
                "6_counterfactual_uncertainty_intervals.png",
                "7_ood_score_vs_prediction_uncertainty.png",
                "8_ood_score_vs_counterfactual_uncertainty.png",
                "9_attribution_uncertainty_across_regimes.png",
                "10_attribution_uncertainty_across_seasons.png",
                "11_counterfactual_directional_stability.png",
                "12_stability_vs_magnitude.png"
            ]
            for p in expected:
                assert (plot_dir / p).exists(), f"Missing plot: {p}"
