import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
from ml.src.modeling.phase6f.decision_support import AtmosIQDecisionSupportService
from ml.src.modeling.phase6f.config import DecisionSupportConfigPhase6F


class TestPhase6F:

    v1_path = ROOT_DIR / "ml" / "data" / "modeling" / "v1" / "feature_dataset_frozen.csv"
    v2_path = ROOT_DIR / "ml" / "data" / "modeling" / "v2" / "feature_dataset_frozen.csv"
    v3_path = ROOT_DIR / "ml" / "data" / "modeling" / "v3" / "feature_dataset_frozen.csv"
    ctrl_model_path = ROOT_DIR / "ml" / "models" / "attribution" / "v1" / "model.joblib"
    v3_prod_model_path = ROOT_DIR / "ml" / "models" / "production" / "v3" / "model.joblib"
    feat_reg_path = ROOT_DIR / "ml" / "models" / "production" / "v3" / "feature_registry.csv"
    prod_unc_path = ROOT_DIR / "ml" / "uncertainty" / "production" / "v1" / "uncertainty_method.json"
    prod_ds_dir = ROOT_DIR / "ml" / "decision_support" / "production" / "v1"
    exp_dir = ROOT_DIR / "ml" / "experiments" / "phase6f"

    v1_expected_hash = "c271bfc6df5dc442737b32e42d2e722c7f08266f77f1820649b7873e0d8b10df"
    v2_expected_hash = "e7645584e48b5fc65d930b9a4fe499560595778759f4c2caf0ad91de256ed301"
    v3_expected_hash = "78b329fbc6f61ccf1a846dfcd140617416c5c9b7e74f1a6bf564d48077d36736"
    ctrl_expected_hash = "55d7f6ab0afc5395dc8647e64302c5fbc7b30b5f9e80d8a7a3ed733c8fc27162"
    v3_model_expected_hash = "9ed048448197dd36574d3b73e8e5c70534e1db7eba3d2f89bb775f4855d88210"

    @classmethod
    def setup_class(cls):
        cls.config = DecisionSupportConfigPhase6F()
        cls.service = AtmosIQDecisionSupportService(cls.config)
        cls.df_v3 = pd.read_csv(cls.v3_path)
        cls.sample_features = {f: float(cls.df_v3.iloc[500][f]) for f in cls.service.features}

    def calculate_sha256(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # 1. Provenance & Immutability
    def test_upstream_lineage_and_immutability(self):
        assert self.calculate_sha256(self.v1_path) == self.v1_expected_hash
        assert self.calculate_sha256(self.v2_path) == self.v2_expected_hash
        assert self.calculate_sha256(self.v3_path) == self.v3_expected_hash
        assert self.calculate_sha256(self.ctrl_model_path) == self.ctrl_expected_hash
        assert self.calculate_sha256(self.v3_prod_model_path) == self.v3_model_expected_hash
        assert self.prod_unc_path.exists()
        assert len(pd.read_csv(self.feat_reg_path)) == 35

    # 2. Prediction & Interval Integration across Levels
    def test_prediction_and_intervals(self):
        res_90 = self.service.predict_with_decision_support(self.sample_features, nominal_coverage=0.90)
        res_80 = self.service.predict_with_decision_support(self.sample_features, nominal_coverage=0.80)
        res_95 = self.service.predict_with_decision_support(self.sample_features, nominal_coverage=0.95)

        assert res_90["prediction"]["value"] >= 0.0
        assert res_90["prediction_interval"]["lower_bound"] >= 0.0
        assert res_90["prediction_interval"]["lower_bound"] <= res_90["prediction_interval"]["upper_bound"]
        assert res_80["prediction_interval"]["interval_width"] < res_90["prediction_interval"]["interval_width"]
        assert res_90["prediction_interval"]["interval_width"] < res_95["prediction_interval"]["interval_width"]

    # 3. Attribution & Group Breakdown
    def test_attribution_and_groups(self):
        res = self.service.predict_with_decision_support(self.sample_features)
        attr = res["attribution"]
        assert len(attr["dominant_features"]) == 3
        assert len(attr["dominant_groups"]) == 2
        groups = [g["group_name"] for g in attr["group_contributions"]]
        assert "pm25_persistence" in groups
        assert "wind_ventilation" in groups
        assert "biomass_burning" in groups

    # 4. Counterfactual Simulation
    def test_counterfactual_simulation(self):
        res = self.service.predict_with_decision_support(self.sample_features, scenario_name="combined_all_favorable")
        cf = res["counterfactual"]
        assert cf["scenario_name"] == "combined_all_favorable"
        assert 0.70 <= cf["directional_stability"] <= 1.0
        assert len(cf["counterfactual_interval_80"]) == 2

    # 5. OOD Assessment
    def test_ood_assessment(self):
        res = self.service.predict_with_decision_support(self.sample_features)
        ood = res["ood_assessment"]
        assert ood["ood_status"] in ["IN_DISTRIBUTION", "NEAR_OOD", "OOD"]
        assert ood["ood_score"] >= 1.0

    # 6. Evidence & Counter-Evidence
    def test_evidence_synthesis(self):
        res = self.service.predict_with_decision_support(self.sample_features)
        assert "supporting_factors" in res["evidence"]
        assert "counter_evidence" in res["evidence"]

    # 7. Decision Support Rules & Reliability Classification
    def test_decision_rules_and_reliability(self):
        res = self.service.predict_with_decision_support(self.sample_features)
        ds = res["decision_support"]
        assert ds["reliability_tier"] in ["HIGH_RELIABILITY", "MODERATE_RELIABILITY", "HIGH_UNCERTAINTY"]
        assert 0.0 <= ds["reliability_index_heuristic"] <= 100.0
        assert len(ds["recommendation_summary"]) > 20

    # 8. Robust Input Validation (Missing, NaN, Inf Handling)
    def test_input_validation_safety(self):
        # Missing feature
        bad_features = self.sample_features.copy()
        del bad_features["pm25_lag_1d"]
        with pytest.raises(ValueError):
            self.service.predict_with_decision_support(bad_features)

        # NaN feature
        nan_features = self.sample_features.copy()
        nan_features["pm25_lag_1d"] = np.nan
        with pytest.raises(ValueError):
            self.service.predict_with_decision_support(nan_features)

        # Inf feature
        inf_features = self.sample_features.copy()
        inf_features["pm25_lag_1d"] = np.inf
        with pytest.raises(ValueError):
            self.service.predict_with_decision_support(inf_features)

    # 9. Production Artifacts & Schema
    def test_production_artifacts_exist(self):
        if self.prod_ds_dir.exists():
            assert (self.prod_ds_dir / "decision_support_schema.json").exists()
            assert (self.prod_ds_dir / "decision_rules.json").exists()
            assert (self.prod_ds_dir / "method_registry.json").exists()
            assert (self.prod_ds_dir / "integration_metadata.json").exists()
            assert (self.prod_ds_dir / "validation_summary.json").exists()
            assert (self.prod_ds_dir / "README.md").exists()
