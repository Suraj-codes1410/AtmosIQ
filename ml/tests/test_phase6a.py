import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestPhase6A:

    v1_path = ROOT_DIR / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
    v2_path = ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
    v3_path = ROOT_DIR / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
    ctrl_model_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "model.joblib"
    v3_prod_model_path = ROOT_DIR / "ml" / "models" / "production" / "v3" / "model.joblib"
    exp_dir = ROOT_DIR / "ml" / "experiments" / "phase6a"

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

    def test_configuration_validation(self):
        from ml.src.modeling.phase6a.config import UncertaintyConfigPhase6A
        cfg = UncertaintyConfigPhase6A()
        assert cfg.dataset_sha256 == self.v3_expected_hash
        assert cfg.production_model_sha256 == self.v3_model_expected_hash
        assert len(cfg.nominal_coverage_levels) == 3
        assert 0.90 in cfg.nominal_coverage_levels
        assert len(cfg.walk_forward_folds) == 3

    def test_coverage_metrics_calculation(self):
        from ml.src.modeling.phase6a.coverage_metrics import IntervalEvaluationMetricsPhase6A
        y_true = np.array([100.0, 150.0, 200.0, 50.0])
        lower = np.array([90.0, 140.0, 190.0, 60.0])  # 4th point is below lower
        upper = np.array([110.0, 160.0, 210.0, 70.0])
        res = IntervalEvaluationMetricsPhase6A.evaluate_interval(y_true, lower, upper, 0.90)
        
        assert res["count"] == 4
        assert abs(res["empirical_coverage"] - 0.75) < 1e-6
        assert abs(res["coverage_error"] - (-0.15)) < 1e-6
        assert abs(res["mean_width_ugm3"] - 17.5) < 1e-6
        assert res["under_coverage_count"] == 1
        assert res["over_coverage_count"] == 0
        assert res["winkler_interval_score"] > 20.0  # Includes penalty

    def test_temporal_splits_results(self):
        splits_csv = self.exp_dir / "temporal_splits.csv"
        if splits_csv.exists():
            df_splits = pd.read_csv(splits_csv)
            assert len(df_splits) == 3
            assert (df_splits["eval_year"] == [2022, 2023, 2024]).all()
            assert (df_splits["eval_mae_ugm3"] < 25.0).all()

    def test_residual_predictions_integrity(self):
        preds_csv = self.exp_dir / "residual_predictions.csv"
        if preds_csv.exists():
            df_preds = pd.read_csv(preds_csv)
            assert len(df_preds) == 1096  # 365 + 365 + 366
            assert not df_preds["residual"].isnull().any()
            # Verify residual = observed - predicted
            diff = np.abs(df_preds["residual"] - (df_preds["observed_pm25"] - df_preds["predicted_pm25"]))
            assert (diff < 1e-6).all()

    def test_baseline_intervals_bounds_consistency(self):
        intervals_csv = self.exp_dir / "baseline_intervals.csv"
        if intervals_csv.exists():
            df_intervals = pd.read_csv(intervals_csv)
            assert (df_intervals["lower_bound"] >= 0.0).all()
            assert (df_intervals["lower_bound"] <= df_intervals["upper_bound"]).all()
            assert (df_intervals["interval_width"] >= 0.0).all()

    def test_leakage_audit_clean(self):
        leakage_csv = self.exp_dir / "leakage_audit.csv"
        if leakage_csv.exists():
            df_leakage = pd.read_csv(leakage_csv)
            assert (df_leakage["status"] == "PASS").all()
            assert df_leakage["violations_detected"].sum() == 0

    def test_reproducibility_check(self):
        repro_csv = self.exp_dir / "reproducibility_check.csv"
        if repro_csv.exists():
            df_repro = pd.read_csv(repro_csv)
            assert df_repro["status"].iloc[0] == "PASS"
            assert float(df_repro["metric_max_absolute_difference"].iloc[0]) <= 1e-10

    def test_plots_generation(self):
        plot_dir = self.exp_dir / "plots"
        if plot_dir.exists():
            expected_plots = [
                "residual_distribution.png",
                "residual_quantiles.png",
                "residuals_over_time.png",
                "residuals_vs_prediction.png",
                "residual_distribution_by_season.png",
                "residual_distribution_by_year.png",
                "residual_distribution_by_pollution_regime.png",
                "coverage_comparison.png",
                "interval_width_comparison.png",
                "interval_coverage_by_regime.png",
                "extreme_pollution_interval_performance.png",
                "representative_prediction_intervals.png"
            ]
            for plot_name in expected_plots:
                assert (plot_dir / plot_name).exists(), f"Plot {plot_name} missing!"
