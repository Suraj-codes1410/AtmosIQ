import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestPhase4I:

    v1_path = ROOT_DIR / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
    v2_path = ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
    v3_path = ROOT_DIR / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
    ctrl_model_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "model.joblib"
    v3_model_path = ROOT_DIR / "ml" / "models" / "attribution" / "v3" / "model.joblib"
    exp_dir = ROOT_DIR / "ml" / "experiments" / "phase4i"

    v1_expected_hash = "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    v2_expected_hash = "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    v3_expected_hash = "78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736"
    ctrl_expected_hash = "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"

    def calculate_sha256(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def test_upstream_hash_integrity(self):
        assert self.calculate_sha256(self.v1_path) == self.v1_expected_hash
        assert self.calculate_sha256(self.v2_path) == self.v2_expected_hash
        assert self.calculate_sha256(self.v3_path) == self.v3_expected_hash
        assert self.calculate_sha256(self.ctrl_model_path) == self.ctrl_expected_hash

    def test_feature_registry_35_features(self):
        from ml.src.modeling.phase4i.feature_registry import FeatureRegistryValidatorPhase4I
        df_v3 = pd.read_csv(self.v3_path)
        validator = FeatureRegistryValidatorPhase4I(df_v3)
        assert len(validator.COMPACT_35_FEATURES) == 35

    def test_leakage_audit_clean(self):
        from ml.src.modeling.phase4i.feature_registry import FeatureRegistryValidatorPhase4I
        df_v3 = pd.read_csv(self.v3_path)
        validator = FeatureRegistryValidatorPhase4I(df_v3)
        unsafe_set = {'pm25', 'pm10', 'no2', 'so2', 'co', 'o3'}
        leaked = [f for f in validator.COMPACT_35_FEATURES if f in unsafe_set]
        assert len(leaked) == 0, f"Leaked features detected: {leaked}"

    def test_group_attribution_mapping_complete(self):
        from ml.src.modeling.phase4i.feature_registry import FeatureRegistryValidatorPhase4I
        from ml.src.modeling.phase4i.attribution_groups import EnvironmentalAttributionGroupsPhase4I
        df_v3 = pd.read_csv(self.v3_path)
        features_35 = FeatureRegistryValidatorPhase4I(df_v3).COMPACT_35_FEATURES
        mapper = EnvironmentalAttributionGroupsPhase4I()
        df_grp = mapper.generate_mapping(features_35, self.exp_dir / "test_v3_attribution_groups.csv")
        assert len(df_grp) == 35
        assert not df_grp['group'].isnull().any()

    def test_shap_reconstruction_error(self):
        rec_path = self.exp_dir / "v3_shap_reconstruction.csv"
        if rec_path.exists():
            df_rec = pd.read_csv(rec_path)
            max_err = float(df_rec.loc[0, "max_reconstruction_error"])
            assert max_err <= 1e-4, f"Reconstruction error {max_err} exceeds 1e-4"

    def test_counterfactual_plausibility_pass_rate(self):
        plaus_path = self.exp_dir / "v3_counterfactual_plausibility.csv"
        if plaus_path.exists():
            df_plaus = pd.read_csv(plaus_path)
            assert (df_plaus["status"] == "PASS").all()

    def test_api_validation_pass(self):
        api_path = self.exp_dir / "api_validation_v3.csv"
        if api_path.exists():
            df_api = pd.read_csv(api_path)
            assert (df_api["status"] == "PASS").all()

    def test_scientific_disclaimer_presence(self):
        case_path = self.exp_dir / "v3_case_studies.csv"
        if case_path.exists():
            df_case = pd.read_csv(case_path)
            for disc in df_case["disclaimer"]:
                assert "PREDICTIVE IMPORTANCE" in disc
