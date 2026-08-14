import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestPhase4J:

    v1_path = ROOT_DIR / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
    v2_path = ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
    v3_path = ROOT_DIR / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
    ctrl_model_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "model.joblib"
    v3_prod_model_path = ROOT_DIR / "ml" / "models" / "production" / "v3" / "model.joblib"
    release_manifest_path = ROOT_DIR / "ml" / "releases" / "v1" / "release_manifest.json"
    exp_dir = ROOT_DIR / "ml" / "experiments" / "phase4j"
    kaggle_dir = ROOT_DIR / "kaggle" / "v3"

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
        if self.v3_prod_model_path.exists():
            assert self.calculate_sha256(self.v3_prod_model_path) == self.v3_model_expected_hash

    def test_35_feature_production_registry(self):
        from ml.src.modeling.phase4j.integrity_audit import ReleaseIntegrityAuditPhase4J
        feats = ReleaseIntegrityAuditPhase4J.PRODUCTION_35_FEATURES
        assert len(feats) == 35
        df_v3 = pd.read_csv(self.v3_path)
        assert all(f in df_v3.columns for f in feats)

    def test_zero_leakage_in_model_features(self):
        from ml.src.modeling.phase4j.integrity_audit import ReleaseIntegrityAuditPhase4J
        feats = ReleaseIntegrityAuditPhase4J.PRODUCTION_35_FEATURES
        unsafe = {'pm25', 'pm10', 'no2', 'so2', 'co', 'o3'}
        leaked = [f for f in feats if f in unsafe or f.startswith('pm25_same_day')]
        assert len(leaked) == 0

    def test_prediction_reproducibility_benchmark(self):
        pred_csv = self.exp_dir / "prediction_reproducibility.csv"
        if pred_csv.exists():
            df_pred = pd.read_csv(pred_csv)
            assert (df_pred["status"] == "PASS").all()
            assert (df_pred["absolute_difference"] <= 1e-10).all()

    def test_attribution_reconstruction_error(self):
        recon_csv = self.exp_dir / "attribution_reproducibility.csv"
        if recon_csv.exists():
            df_recon = pd.read_csv(recon_csv)
            assert df_recon["status"].iloc[0] == "PASS"
            assert float(df_recon["max_reconstruction_error"].iloc[0]) <= 1e-4

    def test_authoritative_counterfactual_baseline(self):
        cf_csv = self.exp_dir / "counterfactual_audit.csv"
        if cf_csv.exists():
            df_cf = pd.read_csv(cf_csv)
            cbw = df_cf[df_cf["scenario"] == "combined_biomass_wind"].iloc[0]
            assert abs(cbw["baseline_mean_pred_ugm3"] - 143.0217) < 0.01
            assert abs(cbw["mean_delta_pm25_ugm3"] - (-5.0990)) < 0.01

    def test_active_driver_directional_consistency_audit(self):
        cons_csv = self.exp_dir / "counterfactual_consistency_audit.csv"
        if cons_csv.exists():
            df_cons = pd.read_csv(cons_csv)
            comb = df_cons[df_cons["category"] == "Combined Active Environmental Driver Population"].iloc[0]
            assert abs(comb["directional_consistency_pct"] - 94.73) < 0.1
            assert int(comb["active_days_count"]) == 816
            assert int(comb["correct_directional_responses"]) == 773

    def test_security_scan_clean(self):
        sec_csv = self.exp_dir / "security_scan_results.csv"
        if sec_csv.exists():
            df_sec = pd.read_csv(sec_csv)
            assert (df_sec["status"] == "PASS").all()

    def test_release_manifest_validity(self):
        if self.release_manifest_path.exists():
            with open(self.release_manifest_path, "r") as f:
                data = json.load(f)
            assert data["production_model"]["name"] == "MODEL_V3_PRODUCTION"
            assert data["production_model"]["sha256"] == self.v3_model_expected_hash
            assert data["production_model"]["feature_count"] == 35

    def test_kaggle_v3_release_candidate_package(self):
        if self.kaggle_dir.exists():
            assert (self.kaggle_dir / "dataset.csv").exists()
            assert (self.kaggle_dir / "data_dictionary.csv").exists()
            assert (self.kaggle_dir / "feature_registry.csv").exists()
            assert (self.kaggle_dir / "sources.md").exists()
            assert (self.kaggle_dir / "checksums.txt").exists()
            df_dict = pd.read_csv(self.kaggle_dir / "data_dictionary.csv")
            assert len(df_dict) >= 270
            assert df_dict["is_production_model_input"].sum() == 35
