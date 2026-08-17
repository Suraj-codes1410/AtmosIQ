"""
Unit and Integration Tests for AtmosIQ Phase 8B (Controlled Generator Scaling).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase8b import (
    ScalingConfigPhase8B,
    Phase8BProvenanceManager,
    Phase8BPhysicsValidator,
    OODScaleMonitor,
    MemorizationScaleAuditor,
    FidelityScaleMonitor,
    MLUtilityScaleEvaluator,
    ScalingBatchGenerator,
    BatchAcceptanceGate,
    Phase8BReproducibilityAuditor,
)
from ml.src.modeling.phase8a.firewall import EvaluationFirewall, EvaluationFirewallViolation


class TestPhase8B:
    @classmethod
    def setup_class(cls):
        cls.config = ScalingConfigPhase8B()
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()

        df_full = pd.read_csv(cls.config.dataset_v3_path)
        cls.df_real_train = df_full[
            (df_full["date"] >= cls.config.dev_train_start_date) &
            (df_full["date"] <= cls.config.dev_train_end_date)
        ].copy()
        cls.df_real_test = df_full[
            (df_full["date"] >= cls.config.locked_eval_start_date) &
            (df_full["date"] <= cls.config.locked_eval_end_date)
        ].copy()

        def classify_season(m):
            if m in [12, 1, 2]: return "Winter"
            if m in [3, 4, 5]: return "Summer"
            if m in [6, 7, 8, 9]: return "Monsoon"
            return "Post-Monsoon"

        def classify_regime(pm):
            if pm < 60.0: return "Low"
            if pm < 120.0: return "Moderate"
            if pm < 250.0: return "High"
            return "Extreme"

        cls.df_real_train["month"] = pd.to_datetime(cls.df_real_train["date"]).dt.month
        cls.df_real_train["season"] = cls.df_real_train["month"].apply(classify_season)
        cls.df_real_train["pollution_regime"] = cls.df_real_train["pm25"].apply(classify_regime)

        cls.df_real_test["month"] = pd.to_datetime(cls.df_real_test["date"]).dt.month
        cls.df_real_test["season"] = cls.df_real_test["month"].apply(classify_season)
        cls.df_real_test["pollution_regime"] = cls.df_real_test["pm25"].apply(classify_regime)

        cls.batch_gen = ScalingBatchGenerator(cls.config, cls.feature_registry)
        cls.batch_gen.fit(cls.df_real_train)

    # 1. Phase 6F Freeze Gate Verification
    def test_freeze_gate(self):
        prov_mgr = Phase8BProvenanceManager(self.root_dir, self.config.freeze_manifest_path)
        passed, summary = prov_mgr.verify_phase6f_freeze()
        assert passed is True
        assert summary["freeze_status"] == "PASS"
        assert summary["total_protected_artifacts"] == 21

    # 2. Hierarchical Deterministic Seed Derivation
    def test_hierarchical_seeding(self):
        b_seed1 = Phase8BProvenanceManager.derive_batch_seed(42, "batch_0001")
        b_seed2 = Phase8BProvenanceManager.derive_batch_seed(42, "batch_0001")
        b_seed3 = Phase8BProvenanceManager.derive_batch_seed(42, "batch_0002")
        assert b_seed1 == b_seed2
        assert b_seed1 != b_seed3

        t_seed1 = Phase8BProvenanceManager.derive_trajectory_seed(b_seed1, "TRAJ_001")
        t_seed2 = Phase8BProvenanceManager.derive_trajectory_seed(b_seed1, "TRAJ_001")
        t_seed3 = Phase8BProvenanceManager.derive_trajectory_seed(b_seed1, "TRAJ_002")
        assert t_seed1 == t_seed2
        assert t_seed1 != t_seed3

    # 3. Data Isolation Firewall
    def test_data_isolation(self):
        firewall = EvaluationFirewall("2022-01-01")
        assert firewall.verify_training_partition_isolation(self.df_real_train) is True

    # 4. Batch Generation Execution
    def test_batch_generation(self, tmp_path):
        df_batch, b_meta, df_rej = self.batch_gen.generate_batch("test_batch", 4, tmp_path)
        assert len(df_batch) > 0
        assert b_meta["target_trajectories"] == 4
        assert b_meta["accepted_trajectories"] >= 2
        assert (tmp_path / "test_batch_accepted.parquet").exists()

    # 5. Physics Validation
    def test_physics_validation(self, tmp_path):
        val = Phase8BPhysicsValidator()
        df_batch, _, _ = self.batch_gen.generate_batch("test_phys_batch", 2, tmp_path)
        is_ok, report = val.validate_trajectory(df_batch, "test_phys_batch")
        assert is_ok is True
        assert report["violation_count"] == 0

    # 6. OOD Scale Monitoring
    def test_ood_monitoring(self):
        ood_mon = OODScaleMonitor(self.feature_registry)
        ood_mon.fit(self.df_real_train)
        df_dummy = self.df_real_train.head(10).copy()
        sum_dict, df_ann = ood_mon.evaluate_batch_ood(df_dummy, "test_ood")
        assert "ood_distance" in df_ann.columns
        assert "ood_category" in df_ann.columns
        assert sum_dict["outlier_pct"] <= 50.0

    # 7. Memorization Auditing
    def test_memorization_audit(self):
        mem_aud = MemorizationScaleAuditor(self.feature_registry)
        mem_aud.fit(self.df_real_train)
        df_dummy = self.df_real_train.head(10).copy()
        report = mem_aud.audit_batch(df_dummy, "test_mem")
        assert report["exact_duplicate_count"] == 10 # Self-test has exact matches

    # 8. Fidelity Monitoring
    def test_fidelity_monitoring(self, tmp_path):
        fid_mon = FidelityScaleMonitor(self.feature_registry)
        df_batch, _, _ = self.batch_gen.generate_batch("test_fid_batch", 2, tmp_path)
        report = fid_mon.evaluate_batch_fidelity(self.df_real_train, df_batch, "test_fid_batch")
        assert report["mean_normalized_w1"] > 0.0
        assert report["frobenius_correlation_distance"] > 0.0

    # 9. Batch Acceptance Gate
    def test_batch_acceptance_gate(self):
        gate = BatchAcceptanceGate()
        b_meta = {"batch_id": "b1", "acceptance_rate_pct": 80.0}
        fid_rep = {"mean_normalized_w1": 0.48, "frobenius_correlation_distance": 0.21, "mean_acf_error_lags_1_7": 0.18}
        mem_rep = {"exact_duplicate_count": 0}
        ood_rep = {"outlier_pct": 45.0}
        dec, rep = gate.evaluate_batch(b_meta, fid_rep, mem_rep, ood_rep)
        assert dec in ["ACCEPT", "CONDITIONAL_ACCEPT"]

    # 10. ML Utility Scaling Evaluation
    def test_ml_utility_scaling(self, tmp_path):
        df_batch, _, _ = self.batch_gen.generate_batch("test_ml_batch", 2, tmp_path)
        evaluator = MLUtilityScaleEvaluator(self.feature_registry, 42)
        df_res, sum_dict = evaluator.evaluate_scaling_utility(self.df_real_train, df_batch, self.df_real_test)
        assert len(df_res) == 5
        assert sum_dict["real_only_mae"] > 0.0

    # 11. Reproducibility Auditor
    def test_reproducibility(self):
        auditor = Phase8BReproducibilityAuditor()
        df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        passed, max_delta, df_aud = auditor.run_reproducibility_audit(df1, df1)
        assert passed is True
        assert max_delta <= 1e-10
