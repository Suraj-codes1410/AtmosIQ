import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestPhase6D:

    v1_path = ROOT_DIR / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
    v2_path = ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
    v3_path = ROOT_DIR / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
    ctrl_model_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "model.joblib"
    v3_prod_model_path = ROOT_DIR / "ml" / "models" / "production" / "v3" / "model.joblib"
    feat_reg_path = ROOT_DIR / "ml" / "models" / "production" / "v3" / "feature_registry.csv"
    exp_dir = ROOT_DIR / "ml" / "experiments" / "phase6d"
    prod_unc_dir = ROOT_DIR / "ml" / "uncertainty" / "production" / "v1"

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

    def test_feature_registry_count(self):
        df_feat = pd.read_csv(self.feat_reg_path)
        assert len(df_feat) == 35, f"Expected 35 features, found {len(df_feat)}"

    def test_phase6d_revalidation_metrics(self):
        reval_csv = self.exp_dir / "phase6d_revalidation.csv"
        if reval_csv.exists():
            df_rev = pd.read_csv(reval_csv)
            row_90 = df_rev[df_rev["nominal_coverage"] == 0.90].iloc[0]
            assert 0.88 <= row_90["empirical_coverage"] <= 0.92
            assert row_90["extreme_150_coverage"] >= 0.85
            assert row_90["extreme_250_coverage"] >= 0.85
            assert row_90["mean_width_ugm3"] < 75.0

    def test_temporal_stability_and_extreme_stress_test(self):
        temp_csv = self.exp_dir / "temporal_stability.csv"
        ext_csv = self.exp_dir / "extreme_threshold_stress_test.csv"
        if temp_csv.exists() and ext_csv.exists():
            df_temp = pd.read_csv(temp_csv)
            df_ext = pd.read_csv(ext_csv)
            assert len(df_temp) >= 3
            assert len(df_ext) >= 4
            assert (df_ext["empirical_coverage_90pct"] >= 0.80).all()

    def test_leakage_and_physical_validity_audits(self):
        leak_csv = self.exp_dir / "phase6d_leakage_audit.csv"
        phys_csv = self.exp_dir / "phase6d_physical_validity.csv"
        if leak_csv.exists() and phys_csv.exists():
            df_leak = pd.read_csv(leak_csv)
            df_phys = pd.read_csv(phys_csv)
            assert (df_leak["status"] == "PASS").all()
            assert df_leak["violations_detected"].sum() == 0
            assert (df_phys["status"] == "PASS").all()
            assert df_phys["violations_detected"].sum() == 0

    def test_production_uncertainty_layer_packaging(self):
        unc_json = self.prod_unc_dir / "uncertainty_method.json"
        cal_art_json = self.prod_unc_dir / "calibration_artifacts.json"
        if unc_json.exists() and cal_art_json.exists():
            with open(unc_json, "r") as f:
                data = json.load(f)
            assert data["uncertainty_method_name"] == "normalized_conformal"
            assert data["promotion_decision"] == "PROMOTE"
            assert data["empirical_coverage_90pct"] >= 0.88

    def test_phase6d_diagnostic_visualizations(self):
        plot_dir = self.exp_dir / "plots"
        if plot_dir.exists():
            expected = [
                "final_coverage_comparison.png",
                "final_calibration_curve.png",
                "final_interval_width_comparison.png",
                "final_coverage_by_regime.png",
                "final_coverage_by_season.png",
                "final_coverage_by_year.png",
                "final_extreme_threshold_stress_test.png",
                "final_temporal_rolling_coverage.png",
                "final_coverage_vs_width.png",
                "final_worst_case_miscoverage.png",
                "final_prediction_intervals_extreme_events.png",
                "final_uncertainty_evolution.png"
            ]
            for p in expected:
                assert (plot_dir / p).exists(), f"Missing plot: {p}"
