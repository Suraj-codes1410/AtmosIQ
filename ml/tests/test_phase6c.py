import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestPhase6C:

    v1_path = ROOT_DIR / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
    v2_path = ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
    v3_path = ROOT_DIR / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
    ctrl_model_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "model.joblib"
    v3_prod_model_path = ROOT_DIR / "ml" / "models" / "production" / "v3" / "model.joblib"
    exp_dir = ROOT_DIR / "ml" / "experiments" / "phase6c"

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

    def test_conformal_quantile_formula(self):
        from ml.src.modeling.phase6c.conformal_engine import ConformalPredictionEnginePhase6C
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        # n = 10, alpha = 0.1 (90% coverage) -> ceil((11)*0.9)/10 = ceil(9.9)/10 = 10/10 = 1.0 -> 10.0
        q = ConformalPredictionEnginePhase6C.compute_conformal_quantile(scores, 0.10)
        assert q == 10.0

    def test_conformal_intervals_bounds_and_validity(self):
        int_csv = self.exp_dir / "conformal_intervals.csv"
        if int_csv.exists():
            df_int = pd.read_csv(int_csv)
            assert len(df_int) >= 1096
            assert (df_int["lower_bound"] <= df_int["upper_bound"]).all()
            assert (df_int["interval_width"] >= 0.0).all()
            assert (df_int["lower_bound"] >= 0.0).all()

    def test_conformal_benchmark_comparison(self):
        bench_csv = self.exp_dir / "conformal_comparison.csv"
        if bench_csv.exists():
            df_b = pd.read_csv(bench_csv)
            assert len(df_b) >= 6
            assert (df_b["empirical_coverage"] >= 0.0).all()
            assert (df_b["mean_width_ugm3"] > 0.0).all()

    def test_conformal_leakage_audit(self):
        leak_csv = self.exp_dir / "conformal_leakage_audit.csv"
        if leak_csv.exists():
            df_leak = pd.read_csv(leak_csv)
            assert (df_leak["status"] == "PASS").all()
            assert df_leak["violations_detected"].sum() == 0

    def test_method_selection_metadata(self):
        sel_json = self.exp_dir / "method_selection.json"
        if sel_json.exists():
            with open(sel_json, "r") as f:
                data = json.load(f)
            assert "best_method" in data
            assert data["coverage_90pct"] >= 0.85
            assert data["promotion_decision"] in [
                "CONFORMAL METHOD PROMOTION RECOMMENDED",
                "PROMOTION NOT RECOMMENDED"
            ]

    def test_publication_plots_generated(self):
        plot_dir = self.exp_dir / "plots"
        if plot_dir.exists():
            expected = [
                "1_conformal_coverage_comparison.png",
                "2_interval_width_comparison.png",
                "3_calibration_curve.png",
                "4_coverage_by_pollution_regime.png",
                "5_coverage_by_season.png",
                "6_coverage_by_year.png",
                "7_extreme_event_coverage.png",
                "8_winkler_score_comparison.png",
                "9_interval_width_vs_coverage.png",
                "10_ensemble_vs_conformal_intervals.png",
                "11_representative_conformal_cases.png",
                "12_temporal_coverage_stability.png"
            ]
            for p in expected:
                assert (plot_dir / p).exists(), f"Plot {p} missing!"
