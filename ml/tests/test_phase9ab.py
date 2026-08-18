"""
Unit and Integration Tests for AtmosIQ Phase 9A–9B (Certification & Independent Validation).
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase9ab import (
    Phase9ABConfig,
    Phase9ABProvenanceManager,
    Phase9AReconciler,
    Phase9BValidator,
    Phase9ABManifestManager,
)
from ml.src.modeling.phase9.models import Phase9TCNModel
from ml.src.modeling.phase8g.policy_engine import Phase8GAugmentationPolicyEngine, AugmentationPolicyViolation


class TestPhase9AB:
    @classmethod
    def setup_class(cls):
        cls.config = Phase9ABConfig()
        cls.root_dir = cls.config.root_dir
        cls.prov_mgr = Phase9ABProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.reconciler = Phase9AReconciler(cls.config.benchmarks_dir, cls.config.manifests_dir)
        cls.validator = Phase9BValidator(extreme_threshold=250.0)
        cls.manifest_mgr = Phase9ABManifestManager(cls.config.manifests_dir)
        cls.policy_engine = Phase8GAugmentationPolicyEngine(0.25, 0.50)

    # 1. Protected Artifacts Freeze
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["drift_count"] == 0

    # 2. Augmentation Policy Engine Enforcement
    def test_governance_reconciliation_enforcement(self):
        res_25 = self.policy_engine.validate_augmentation_request(0.25)
        assert res_25["tier"] == "RECOMMENDED_PRODUCTION"

        res_50 = self.policy_engine.validate_augmentation_request(0.50, is_stress_test=True)
        assert res_50["tier"] == "CONTROLLED_UPPER_BOUND"

        with pytest.raises(AugmentationPolicyViolation):
            self.policy_engine.validate_augmentation_request(1.00)

    # 3. Model Selection Reconciliation Engine
    def test_reconciliation_logic(self):
        p9_val_csv = self.config.phase9_benchmarks_dir / "phase9_validation_results.csv"
        p9_multi_csv = self.config.phase9_benchmarks_dir / "phase9_multiseed_results.csv"
        df_recon, decision = self.reconciler.reconcile_candidates(p9_val_csv, p9_multi_csv)

        assert len(df_recon) == 12
        assert "TCN_aug50pct" in df_recon["candidate_id"].values
        assert decision["reconciliation_status"] == "GOVERNANCE_RECONCILED"
        assert decision["selected_research_candidate"]["certification_status"] == "CERTIFIED_RESEARCH_CANDIDATE"
        assert decision["selected_production_eligible_candidate"]["certification_status"] == "PRODUCTION_ELIGIBLE"

    # 4. Independent Validation Metrics & Extreme-Event Detection
    def test_independent_validation_metrics(self):
        y_true = np.array([30.0, 50.0, 100.0, 260.0, 310.0], dtype=np.float32)
        y_pred = np.array([32.0, 48.0, 95.0, 250.0, 290.0], dtype=np.float32)

        m = self.validator.evaluate_metrics(y_true, y_pred)
        assert m["mae"] > 0.0
        assert m["extreme_count"] == 2
        assert m["extreme_underpred_rate"] == 1.0
        assert m["physical_validity_rate"] == 1.0
        assert m["negative_predictions_count"] == 0
        assert m["nan_count"] == 0
        assert m["inf_count"] == 0

    # 5. Temporal, Seasonal, and Regime Breakdowns
    def test_breakdowns_and_failure_cases(self):
        y_true = np.array([25.0, 55.0, 110.0, 220.0, 280.0], dtype=np.float32)
        y_pred = np.array([24.0, 52.0, 105.0, 210.0, 260.0], dtype=np.float32)
        dates = ["2022-01-15", "2022-04-15", "2023-07-15", "2023-10-15", "2024-12-15"]

        df_yr = self.validator.evaluate_yearly_breakdowns(y_true, y_pred, dates)
        assert len(df_yr) == 3

        df_sn = self.validator.evaluate_seasonal_breakdowns(y_true, y_pred, dates)
        assert len(df_sn) == 4

        df_reg = self.validator.evaluate_regime_breakdowns(y_true, y_pred)
        assert len(df_reg) == 5

        df_fail = self.validator.extract_failure_cases(y_true, y_pred, dates, top_n=3)
        assert len(df_fail) == 3
        assert df_fail.iloc[0]["absolute_error"] == 20.0

    # 6. Physical Non-Negativity & Checkpoint Inference
    def test_checkpoint_reproducibility(self):
        winner_ckpt_path = self.config.phase9_checkpoints_dir / "checkpoint_TCN_aug50pct_seed2025.json"
        assert winner_ckpt_path.exists()

        model1 = Phase9TCNModel(window_size=14, feature_dim=35, seed=2025)
        model2 = Phase9TCNModel(window_size=14, feature_dim=35, seed=2025)

        with open(winner_ckpt_path) as f:
            state = json.load(f)
        model1.load_state_dict(state["model_state"])
        model2.load_state_dict(state["model_state"])

        X_dummy = np.random.randn(10, 14, 35).astype(np.float32)
        p1 = model1.forward(X_dummy)
        p2 = model2.forward(X_dummy)

        delta = float(np.max(np.abs(p1 - p2)))
        assert delta <= 1e-9
        assert (p1 >= 0.0).all()
