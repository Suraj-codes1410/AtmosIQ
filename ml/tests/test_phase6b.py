import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestPhase6B:

    v1_path = ROOT_DIR / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
    v2_path = ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
    v3_path = ROOT_DIR / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
    ctrl_model_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "model.joblib"
    v3_prod_model_path = ROOT_DIR / "ml" / "models" / "production" / "v3" / "model.joblib"
    exp_dir = ROOT_DIR / "ml" / "experiments" / "phase6b"

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

    def test_lineage_and_production_hashes(self):
        assert self.calculate_sha256(self.v1_path) == self.v1_expected_hash
        assert self.calculate_sha256(self.v2_path) == self.v2_expected_hash
        assert self.calculate_sha256(self.v3_path) == self.v3_expected_hash
        assert self.calculate_sha256(self.ctrl_model_path) == self.ctrl_expected_hash
        assert self.calculate_sha256(self.v3_prod_model_path) == self.v3_model_expected_hash

    def test_ensemble_predictions_shape_and_quantiles(self):
        preds_csv = self.exp_dir / "ensemble_predictions.csv"
        if preds_csv.exists():
            df_preds = pd.read_csv(preds_csv)
            assert len(df_preds) >= 1096
            # Test quantile ordering
            assert (df_preds["q02_5"] <= df_preds["q05"]).all()
            assert (df_preds["q05"] <= df_preds["q10"]).all()
            assert (df_preds["q10"] <= df_preds["q90"]).all()
            assert (df_preds["q90"] <= df_preds["q95"]).all()
            assert (df_preds["q95"] <= df_preds["q97_5"]).all()
            assert (df_preds["ensemble_std"] >= 0.0).all()

    def test_intervals_bounds_consistency(self):
        int_csv = self.exp_dir / "ensemble_intervals.csv"
        if int_csv.exists():
            df_int = pd.read_csv(int_csv)
            assert (df_int["lower_bound"] <= df_int["upper_bound"]).all()
            assert (df_int["interval_width"] >= 0.0).all()
            # Test clipped bounds non-negative
            sub_clip = df_int[df_int["is_clipped"]]
            assert (sub_clip["lower_bound"] >= 0.0).all()

    def test_spread_error_analysis(self):
        spread_csv = self.exp_dir / "spread_error_analysis.csv"
        if spread_csv.exists():
            df_sp = pd.read_csv(spread_csv)
            assert len(df_sp) == 5  # 5 quintiles
            assert (df_sp["mae_ugm3"] > 0).all()

    def test_uncertainty_discrimination(self):
        disc_csv = self.exp_dir / "uncertainty_discrimination.csv"
        if disc_csv.exists():
            df_disc = pd.read_csv(disc_csv)
            assert len(df_disc) >= 2
            assert (df_disc["roc_auc"] > 0.50).all()

    def test_leakage_audit_clean(self):
        leak_csv = self.exp_dir / "ensemble_leakage_audit.csv"
        if leak_csv.exists():
            df_leak = pd.read_csv(leak_csv)
            assert (df_leak["status"] == "PASS").all()
            assert df_leak["violations_detected"].sum() == 0

    def test_reproducibility_check(self):
        repro_csv = self.exp_dir / "reproducibility_check.csv"
        if repro_csv.exists():
            df_repro = pd.read_csv(repro_csv)
            assert df_repro["status"].iloc[0] == "PASS"

    def test_plots_generation(self):
        plot_dir = self.exp_dir / "plots"
        if plot_dir.exists():
            expected_plots = [
                "ensemble_prediction_spread.png",
                "ensemble_spread_vs_absolute_error.png",
                "spread_error_quantiles.png",
                "calibration_curve.png",
                "coverage_by_regime.png",
                "interval_width_by_regime.png",
                "uncertainty_by_season.png",
                "uncertainty_by_year.png",
                "extreme_pollution_uncertainty.png",
                "bootstrap_vs_seed_spread.png",
                "ensemble_size_stability.png",
                "prediction_intervals_representative_cases.png",
                "ensemble_error_distribution.png",
                "uncertainty_discrimination.png"
            ]
            for p in expected_plots:
                assert (plot_dir / p).exists(), f"Plot {p} missing!"
