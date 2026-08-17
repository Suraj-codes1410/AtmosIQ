"""
Unit and Integration Tests for AtmosIQ Phase 8D (Distribution & Temporal Calibration).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase8d import (
    CalibrationConfigPhase8D,
    Phase8DProvenanceManager,
    CalibrationStrategyEngine,
    MultiObjectiveFidelityEvaluator,
    Phase8DMLUtilityEvaluator,
    Phase8DAuditor,
)


class TestPhase8D:
    @classmethod
    def setup_class(cls):
        cls.config = CalibrationConfigPhase8D()
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

        cls.prov_mgr = Phase8DProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.strategy_engine = CalibrationStrategyEngine(cls.feature_registry)
        cls.strategy_engine.fit_from_development_data(cls.df_real_train)

        cls.fidelity_evaluator = MultiObjectiveFidelityEvaluator(cls.feature_registry)
        cls.fidelity_evaluator.fit_reference(cls.df_real_train)

        cls.auditor = Phase8DAuditor(cls.feature_registry)
        cls.auditor.fit_reference(cls.df_real_train)

    # 1. Protected Artifact Immutability
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["freeze_status"] == "PASS"

    # 2. Calibration Strategy Engine (Candidate Calibration)
    def test_calibration_strategies(self):
        # Create synthetic test trajectories
        trajs = []
        for i in range(5):
            df_t = self.df_real_train.head(14).copy()
            df_t["trajectory_id"] = f"SYNTH_T{i+1}"
            df_t["synthetic_date"] = [f"2025-01-{d+1:02d}" for d in range(14)]
            df_t["data_origin"] = "synthetic"
            df_t["source_partition"] = "2020-2021"
            trajs.append(df_t)
        df_corpus = pd.concat(trajs, ignore_index=True)

        for c_id in ["CAL-00", "CAL-01", "CAL-03", "CAL-07"]:
            df_cal, stats = self.strategy_engine.apply_candidate_calibration(df_corpus, c_id)
            assert len(df_cal) > 0
            assert stats["accepted_trajectories"] >= 1

    # 3. Multi-Objective Fidelity Evaluator
    def test_fidelity_evaluator(self):
        df_dummy = self.df_real_train.head(28).copy()
        df_dummy["trajectory_id"] = "T1"
        res = self.fidelity_evaluator.evaluate_candidate(self.df_real_train, df_dummy, "TEST-CAL")
        assert res["mean_normalized_w1"] >= 0.0
        assert res["frobenius_correlation_distance"] >= 0.0
        assert res["physical_validity_pct"] == 100.0

    # 4. Leakage Audit
    def test_leakage_audit(self):
        df_clean = pd.DataFrame({"synthetic_date": ["2025-01-01"], "data_origin": ["synthetic"]})
        passed, df_aud = self.auditor.audit_leakage(df_clean)
        assert passed is True

        df_leaked = pd.DataFrame({"date": ["2023-05-01"]})
        passed_leak, _ = self.auditor.audit_leakage(df_leaked)
        assert passed_leak is False

    # 5. Physics Audit
    def test_physics_audit(self):
        df_valid = self.df_real_train.head(14).copy()
        passed, df_aud = self.auditor.audit_physics(df_valid)
        assert passed is True

    # 6. Memorization Audit
    def test_memorization_audit(self):
        # Create non-memorized dummy
        df_synthetic = self.df_real_train.head(10).copy()
        df_synthetic["pm25"] += 50.0 # Shift to avoid exact match
        for f in self.feature_registry:
            if f in df_synthetic.columns:
                df_synthetic[f] += 2.0
        passed, df_aud = self.auditor.audit_memorization(df_synthetic)
        assert passed is True

    # 7. Reproducibility Audit
    def test_reproducibility_audit(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        passed, df_aud = self.auditor.audit_reproducibility(df1, df1)
        assert passed is True

    # 8. ML Utility Evaluator
    def test_ml_utility_evaluator(self):
        evaluator = Phase8DMLUtilityEvaluator(self.feature_registry, 42)
        df_dummy = self.df_real_train.head(50).copy()
        res = evaluator.evaluate_candidate_utility(self.df_real_train, df_dummy, self.df_real_test, "TEST-CAL")
        assert res["test_mae"] > 0.0
        assert res["test_r2"] > 0.0
