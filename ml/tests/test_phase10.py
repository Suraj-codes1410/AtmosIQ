"""
Unit and Integration Tests for AtmosIQ Phase 10 + Phase 10A (Production Validation & Walk-Forward Backtesting).
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase10 import (
    Phase10Config,
    Phase10ProvenanceManager,
    Phase10WalkForwardValidator,
    Phase10RobustnessAuditor,
    Phase10FailureModeAnalyzer,
    Phase10ManifestManager,
)
from ml.src.modeling.phase9.models import Phase9TCNModel
from ml.src.modeling.phase9.trainer import Phase9Trainer
from ml.src.modeling.phase9cd.inference import Phase9DInferenceEngine, InferenceContractViolation
from ml.src.modeling.phase8g.policy_engine import Phase8GAugmentationPolicyEngine, AugmentationPolicyViolation


class TestPhase10:
    @classmethod
    def setup_class(cls):
        cls.config = Phase10Config()
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()
        cls.prov_mgr = Phase10ProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.wf_validator = Phase10WalkForwardValidator(
            feature_registry=cls.feature_registry,
            window_size=cls.config.sequence_window,
            extreme_threshold=250.0
        )
        cls.policy_engine = Phase8GAugmentationPolicyEngine(0.25, 0.50)
        cls.fail_analyzer = Phase10FailureModeAnalyzer(cls.config.benchmarks_dir)

    # 1. Protected Artifacts Freeze
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["drift_count"] == 0

    # 2. Augmentation Policy Engine Enforcement
    def test_production_augmentation_policy(self):
        res_25 = self.policy_engine.validate_augmentation_request(0.25)
        assert res_25["tier"] == "RECOMMENDED_PRODUCTION"

        res_50 = self.policy_engine.validate_augmentation_request(0.50, is_stress_test=True)
        assert res_50["tier"] == "CONTROLLED_UPPER_BOUND"

        with pytest.raises(AugmentationPolicyViolation):
            self.policy_engine.validate_augmentation_request(1.00)

    # 3. Walk-Forward Fold Execution & Leakage Audit
    def test_walkforward_fold_execution(self):
        df_full = pd.read_csv(self.config.dataset_v3_path)
        fold_cfg = self.config.walkforward_folds[0] # Window A
        model = Phase9TCNModel(window_size=14, feature_dim=35, seed=2025)

        metrics, leakage, df_preds = self.wf_validator.execute_walkforward_fold(
            df_full=df_full,
            fold_cfg=fold_cfg,
            model=model,
            aug_ratio=0.25,
            cal_bias=-5.0,
            bound_90=95.0
        )

        assert metrics["mae"] > 0.0
        assert leakage["temporal_firewall_passed"] is True
        assert leakage["status"] == "PASS"
        assert len(df_preds) > 0

    # 4. Temporal & Regime Breakdowns
    def test_temporal_and_regime_breakdowns(self):
        df_dummy = pd.DataFrame({
            "timestamp": ["2021-01-15", "2021-04-15", "2021-07-15", "2021-10-15"],
            "y_true": [25.0, 55.0, 110.0, 260.0],
            "y_pred_cal": [26.0, 52.0, 105.0, 250.0],
            "residual": [1.0, -3.0, -5.0, -10.0],
            "abs_error": [1.0, 3.0, 5.0, 10.0],
        })

        df_sn, df_reg = self.wf_validator.compute_temporal_and_regime_breakdowns(df_dummy)
        assert len(df_sn) == 4
        assert len(df_reg) == 4

    # 5. Operational Input Robustness & Malformed Rejection
    def test_input_robustness_rejection(self):
        model = Phase9TCNModel(window_size=14, feature_dim=35, seed=42)
        engine = Phase9DInferenceEngine(
            model=model,
            feature_registry=self.feature_registry,
            window_size=14,
            feature_dim=35,
            model_version="TEST_v1.0.0",
        )
        auditor = Phase10RobustnessAuditor(engine, self.feature_registry)

        X_valid = np.random.randn(4, 14, 35).astype(np.float32)
        df_rob = auditor.audit_input_robustness(X_valid)
        assert len(df_rob) == 12
        assert (df_rob["pass_fail"] == "PASS").all()

    # 6. Feature Drift Analysis
    def test_feature_drift_analysis(self):
        df_full = pd.read_csv(self.config.dataset_v3_path)
        df_hist = df_full[(df_full["date"] >= "2020-01-01") & (df_full["date"] <= "2021-12-31")]
        df_eval = df_full[(df_full["date"] >= "2022-01-01") & (df_full["date"] <= "2022-12-31")]

        model = Phase9TCNModel(window_size=14, feature_dim=35, seed=42)
        engine = Phase9DInferenceEngine(
            model=model,
            feature_registry=self.feature_registry,
            window_size=14,
            feature_dim=35,
            model_version="TEST_v1.0.0",
        )
        auditor = Phase10RobustnessAuditor(engine, self.feature_registry)

        df_drift = auditor.audit_feature_drift(df_hist, df_eval)
        assert len(df_drift) > 0
        assert "drift_classification" in df_drift.columns

    # 7. Operational Failure Modes Table
    def test_failure_modes_matrix(self):
        df_fm = self.fail_analyzer.generate_failure_matrix()
        assert len(df_fm) >= 6
        assert "severity" in df_fm.columns
        assert "CRITICAL" in df_fm["severity"].values
