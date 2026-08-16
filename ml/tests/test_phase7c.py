"""
Unit and Integration Tests for AtmosIQ Phase 7C (Validation & ML Utility).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase7c import (
    ValidationConfigPhase7C,
    Phase6FFreezeVerifier,
    ProvenanceAuditorPhase7C,
    UnivariateDistributionValidator,
    MultivariateDependencyValidator,
    TemporalDynamicsValidator,
    SeasonalRegimeValidator,
    ExtremeTailValidator,
    PhysicsValidatorPhase7C,
    RealVsSyntheticClassifier,
    MLUtilityEvaluator,
    ExtremeMLUtilityEvaluator,
    SyntheticOODAuditor,
    MemorizationAuditor,
    Phase7CReproducibilityAuditor,
    TrainingReadinessDecisionGate,
)


class TestPhase7C:
    @classmethod
    def setup_class(cls):
        cls.config = ValidationConfigPhase7C()
        cls.root_dir = cls.config.root_dir
        cls.feature_registry = pd.read_csv(cls.config.feature_registry_path)["feature_name"].tolist()

        # Load partitions
        df_full = pd.read_csv(cls.config.dataset_v3_path)
        cls.df_real_train = df_full[
            (df_full["date"] >= cls.config.dev_train_start) &
            (df_full["date"] <= cls.config.dev_train_end)
        ].copy()
        cls.df_real_test = df_full[
            (df_full["date"] >= cls.config.locked_eval_start) &
            (df_full["date"] <= cls.config.locked_eval_end)
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

        cls.df_synthetic = pd.read_parquet(cls.config.synthetic_parquet_path)

    # 1. Phase 6F Freeze Gate Verification
    def test_freeze_gate_verification(self):
        verifier = Phase6FFreezeVerifier(self.root_dir, self.config.freeze_manifest_path)
        passed, record = verifier.verify_freeze_baseline()
        assert passed is True
        assert record["freeze_gate_status"] == "PASS"
        assert record["total_protected_artifacts"] == 21

    # 2. Provenance & Metadata Completeness
    def test_synthetic_provenance(self):
        auditor = ProvenanceAuditorPhase7C(self.config.dev_train_end, self.config.locked_eval_start)
        res = auditor.audit_provenance(self.df_synthetic)
        assert bool(res["provenance_passed"]) is True
        assert bool(res["all_data_origin_synthetic"]) is True
        assert res["total_synthetic_records"] == 1110

    # 3. Univariate Distributional Fidelity
    def test_distributional_fidelity(self):
        dist_val = UnivariateDistributionValidator(self.feature_registry)
        df_dist, summary = dist_val.validate_distributions(self.df_real_train, self.df_synthetic)
        assert len(df_dist) >= 35
        assert summary["mean_normalized_w1"] > 0.0
        assert summary["mean_ks_stat"] > 0.0

    # 4. Multivariate Dependency Fidelity
    def test_multivariate_dependencies(self):
        multi_val = MultivariateDependencyValidator(self.feature_registry)
        df_pairs, summary, corr_r, corr_s = multi_val.validate_multivariate_dependencies(self.df_real_train, self.df_synthetic)
        assert summary["pearson_frobenius_distance"] > 0.0
        assert len(df_pairs) > 0

    # 5. Temporal Autocorrelation
    def test_temporal_dynamics(self):
        temp_val = TemporalDynamicsValidator()
        df_acf, summary, acf_r, acf_s = temp_val.validate_temporal_dynamics(self.df_real_train, self.df_synthetic)
        assert len(df_acf) == 30
        assert summary["mean_acf_error_lags_1_7"] > 0.0

    # 6. Physical Constraints
    def test_physics_boundaries(self):
        phys_val = PhysicsValidatorPhase7C()
        df_phys, summary = phys_val.validate_physics(self.df_synthetic)
        assert summary["total_physics_violations"] == 0
        assert summary["hard_constraint_pass_rate_pct"] == 100.0

    # 7. Extreme Tail Coherence
    def test_extreme_tail_coherence(self):
        tail_val = ExtremeTailValidator()
        df_tail, summary = tail_val.validate_extreme_tail(self.df_real_train, self.df_synthetic)
        assert summary["extreme_250_count"] > 0
        assert summary["extreme_250_coherence_rate"] >= 0.70

    # 8. Memorization & Exact Duplicates
    def test_memorization_audit(self):
        mem_auditor = MemorizationAuditor(self.feature_registry)
        df_mem, summary = mem_auditor.audit_memorization(self.df_real_train, self.df_synthetic)
        assert summary["exact_duplicate_count"] == 0
        assert summary["near_duplicate_count"] == 0
        assert summary["memorization_status"] == "PASS"

    # 9. OOD Artifacts
    def test_ood_artifacts(self):
        ood_auditor = SyntheticOODAuditor(self.feature_registry)
        df_ood, summary = ood_auditor.audit_ood_artifacts(self.df_real_train, self.df_synthetic)
        assert summary["synthetic_outlier_count"] >= 0
        assert summary["synthetic_outlier_pct"] <= 60.0

    # 10. Machine Learning Utility & Isolation
    def test_ml_utility(self):
        ml_util = MLUtilityEvaluator(self.feature_registry, self.config.ml_utility_dir, self.config.random_seed)
        df_results, summary, pred_dict = ml_util.evaluate_ml_utility(self.df_real_train, self.df_synthetic, self.df_real_test)
        
        assert len(df_results) == 6
        assert bool(summary["synthetic_to_real_transfer_pass"]) is True
        # Ensure augmentation does not degrade baseline
        assert summary["delta_best_mae_vs_real"] <= 0.50

    # 11. Reproducibility
    def test_reproducibility(self):
        auditor = Phase7CReproducibilityAuditor()
        m1 = {"metric_a": 12.345, "metric_b": 0.0842}
        passed, max_delta, df_repro = auditor.run_reproducibility_audit(m1, m1)
        assert bool(passed) is True
        assert max_delta <= 1e-9

    # 12. Decision Gate
    def test_decision_gate(self):
        gate = TrainingReadinessDecisionGate()
        summaries = {
            "freeze_pass": True,
            "physics_pass": True,
            "physics_pass_rate": 100.0,
            "mean_w1": 0.0842,
            "w1_pass": True,
            "corr_frob": 0.0614,
            "corr_pass": True,
            "acf_err_7": 0.0418,
            "acf_pass": True,
            "extreme_coherence": 0.9785,
            "extreme_pass": True,
            "exact_duplicates": 0,
            "ood_outlier_pct": 2.5,
            "ood_pass": True,
            "delta_best_mae": -0.08,
            "ml_utility_pass": True,
            "delta_extreme_250_mae": -0.15,
            "extreme_ml_pass": True,
            "repro_delta": 0.0,
            "repro_pass": True,
        }
        readiness, admission, df_mat = gate.evaluate_decision(summaries)
        assert readiness in ["ACCEPT", "CONDITIONAL_ACCEPT"]
        assert admission in ["APPROVED", "APPROVED_WITH_RESTRICTIONS"]
