"""
Unit and Integration Tests for AtmosIQ Phase 9C–9D (Hardening, Calibration & Deployment Interface).
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase9cd import (
    Phase9CDConfig,
    Phase9CDProvenanceManager,
    Phase9CHardener,
    Phase9DInferenceEngine,
    InferenceContractViolation,
    Phase9CDManifestManager,
)
from ml.src.modeling.phase9.models import Phase9TCNModel, Phase9LSTMModel
from ml.src.modeling.phase8g.policy_engine import Phase8GAugmentationPolicyEngine, AugmentationPolicyViolation


class TestPhase9CD:
    @classmethod
    def setup_class(cls):
        cls.config = Phase9CDConfig()
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()
        cls.prov_mgr = Phase9CDProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.hardener = Phase9CHardener(cls.feature_registry, extreme_threshold=250.0)
        cls.policy_engine = Phase8GAugmentationPolicyEngine(0.25, 0.50)

    # 1. Protected Artifacts Freeze
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["drift_count"] == 0

    # 2. Augmentation Policy Engine Enforcement
    def test_augmentation_policy_enforcement(self):
        res_25 = self.policy_engine.validate_augmentation_request(0.25)
        assert res_25["tier"] == "RECOMMENDED_PRODUCTION"

        res_50 = self.policy_engine.validate_augmentation_request(0.50, is_stress_test=True)
        assert res_50["tier"] == "CONTROLLED_UPPER_BOUND"

        with pytest.raises(AugmentationPolicyViolation):
            self.policy_engine.validate_augmentation_request(1.00)

    # 3. Calibration & Conformal Uncertainty Calculations
    def test_calibration_and_uncertainty(self):
        y_val_true = np.array([40.0, 60.0, 100.0, 260.0], dtype=np.float32)
        y_val_pred = np.array([45.0, 65.0, 105.0, 265.0], dtype=np.float32) # Bias = +5.0

        hardener = Phase9CHardener(self.feature_registry, extreme_threshold=250.0)
        hardener.fit_calibration_and_uncertainty(y_val_true, y_val_pred)

        assert np.isclose(hardener.calibration_bias, 5.0)
        assert hardener.conformal_q90 >= 5.0

        y_test_pred = np.array([55.0, 105.0], dtype=np.float32)
        calibrated = hardener.calibrate_predictions(y_test_pred)
        assert np.allclose(calibrated, [50.0, 100.0])

        lower, upper, bound = hardener.compute_prediction_intervals(calibrated, alpha=0.10)
        assert (lower <= calibrated).all()
        assert (upper >= calibrated).all()

        unc_metrics = hardener.evaluate_uncertainty_coverage(
            np.array([50.0, 100.0]), calibrated, lower, upper
        )
        assert unc_metrics["interval_coverage"] == 1.0

    # 4. Residual Diagnostics Engine
    def test_residual_diagnostics(self):
        y_true = np.array([30.0, 50.0, 100.0, 200.0], dtype=np.float32)
        y_pred = np.array([32.0, 48.0, 102.0, 198.0], dtype=np.float32)

        res_diag = self.hardener.compute_residual_diagnostics(y_true, y_pred)
        assert "residual_mean" in res_diag
        assert "residual_std" in res_diag
        assert "lag1_autocorrelation" in res_diag
        assert "heteroscedasticity_corr" in res_diag

    # 5. Permutation Feature Importance
    def test_permutation_explainability(self):
        model = Phase9TCNModel(window_size=14, feature_dim=35, seed=42)
        X_dummy = np.random.randn(8, 14, 35).astype(np.float32)
        y_dummy = np.random.rand(8).astype(np.float32) * 50.0

        df_imp = self.hardener.compute_permutation_explainability(model, X_dummy, y_dummy, n_repeats=1, seed=42)
        assert len(df_imp) == 35
        assert "importance_mae_delta" in df_imp.columns
        assert df_imp["rank"].iloc[0] == 1

    # 6. Inference Engine Contract, Determinism & Robustness
    def test_inference_engine_contract_and_robustness(self):
        model = Phase9TCNModel(window_size=14, feature_dim=35, seed=42)
        engine = Phase9DInferenceEngine(
            model=model,
            feature_registry=self.feature_registry,
            window_size=14,
            feature_dim=35,
            model_version="TEST_MODEL_v1.0.0",
            calibration_bias=2.0,
            interval_bound_90=15.0,
        )

        X_valid = np.random.randn(4, 14, 35).astype(np.float32)
        res = engine.predict(X_valid)
        assert len(res["forecast_pm25"]) == 4
        assert "uncertainty_interval_90" in res

        # Determinism check
        res2 = engine.predict(X_valid)
        p1 = np.array(res["forecast_pm25"])
        p2 = np.array(res2["forecast_pm25"])
        delta = float(np.max(np.abs(p1 - p2)))
        assert delta <= 1e-9

        # Latency profile
        prof = engine.profile_latency(X_valid, n_iterations=5)
        assert prof["single_item_latency_ms"] >= 0.0

        # Adversarial input rejection
        df_rob = engine.run_robustness_test_suite(X_valid)
        assert df_rob["safely_rejected"].all()

        with pytest.raises(InferenceContractViolation):
            engine.validate_input_tensor(np.random.randn(4, 10, 35)) # Wrong W
