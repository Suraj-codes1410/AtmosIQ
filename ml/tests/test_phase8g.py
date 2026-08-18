"""
Unit and Integration Tests for AtmosIQ Phase 8G (Production Integration & Pre-Deep-Learning Gate).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase8g import (
    Phase8GConfig,
    Phase8GProvenanceManager,
    Phase8GAugmentationPolicyEngine,
    AugmentationPolicyViolation,
    Phase8GSequenceBuilder,
    Phase8GInterfaceValidator,
    Phase8GAuditor,
)


class TestPhase8G:
    @classmethod
    def setup_class(cls):
        cls.config = Phase8GConfig()
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

        cls.df_8d_corpus = pd.read_parquet(cls.config.phase8d_corpus_path)

        cls.prov_mgr = Phase8GProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.policy_engine = Phase8GAugmentationPolicyEngine(0.25, 0.50)
        cls.seq_builder = Phase8GSequenceBuilder(cls.feature_registry, "pm25")
        cls.seq_builder.fit_scaler(cls.df_real_train)
        cls.validator = Phase8GInterfaceValidator(cls.feature_registry)
        cls.auditor = Phase8GAuditor(cls.feature_registry)

    # 1. Protected Artifact Freeze
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["drift_count"] == 0

    # 2. Augmentation Policy Engine & 100% Rejection
    def test_augmentation_policy_enforcement(self):
        res_00 = self.policy_engine.validate_augmentation_request(0.0)
        assert res_00["is_valid"] is True

        res_25 = self.policy_engine.validate_augmentation_request(0.25)
        assert res_25["tier"] == "RECOMMENDED_PRODUCTION"

        res_50 = self.policy_engine.validate_augmentation_request(0.50, is_stress_test=True)
        assert res_50["tier"] == "CONTROLLED_UPPER_BOUND"

        # 50% without stress flag raises error
        with pytest.raises(AugmentationPolicyViolation):
            self.policy_engine.validate_augmentation_request(0.50, is_stress_test=False)

        # 100% raises hard rejection
        with pytest.raises(AugmentationPolicyViolation):
            self.policy_engine.validate_augmentation_request(1.00)

    # 3. Temporal Sequence Construction & Trajectory Boundaries
    def test_sequence_construction_boundaries(self):
        # Create 2 disjoint synthetic trajectories of length 14
        df_dummy = pd.concat([
            self.df_real_train.head(14).assign(trajectory_id="T1", data_origin="synthetic"),
            self.df_real_train.head(14).assign(trajectory_id="T2", data_origin="synthetic"),
        ], ignore_index=True)

        X, y, prov = self.seq_builder.create_sequences_from_trajectories(df_dummy, window_size=7, is_synthetic=True)
        # Each 14-day trajectory produces 14 - 7 = 7 sequences. Total = 14.
        assert len(X) == 14
        assert len(y) == 14
        assert len(prov) == 14
        assert X.shape == (14, 7, len(self.feature_registry))

    # 4. Integrated Dataset Assembly (25% Recommended)
    def test_assemble_integrated_dataset_25(self):
        X, y, df_prov, meta = self.seq_builder.assemble_integrated_dataset(
            self.df_real_train, self.df_8d_corpus, augmentation_ratio=0.25, window_size=14, seed=42
        )
        assert meta["real_sequences"] == 717
        assert meta["synthetic_sequences"] == 179
        assert meta["total_sequences"] == 896
        assert X.shape == (896, 14, len(self.feature_registry))
        assert y.shape == (896,)
        assert len(df_prov) == 896

    # 5. Interface Validation & Architecture Smoke Passes
    def test_interface_validation_and_smoke(self):
        X = np.random.randn(64, 14, len(self.feature_registry)).astype(np.float32)
        y = np.random.rand(64).astype(np.float32) * 80.0

        t_pass, t_sum = self.validator.validate_training_tensors(X, y, window_size=14)
        assert t_pass is True

        s_pass, s_sum = self.validator.verify_architecture_smoke_pass(X, y)
        assert s_pass is True
        assert s_sum["LSTM"] == "PASS"
        assert s_sum["TCN"] == "PASS"
        assert s_sum["Transformer"] == "PASS"

    # 6. Data Isolation & Temporal Firewall
    def test_data_isolation_audit(self):
        df_prov_dummy = pd.DataFrame({"source_partition": ["2020-2021"] * 10})
        passed, df_aud = self.auditor.audit_leakage(self.df_real_train, self.df_real_test, df_prov_dummy)
        assert passed is True
        assert (df_aud["violations"] == 0).all()

    # 7. Physical Invariants & Hydrodynamic Identity
    def test_physical_invariants_audit(self):
        passed, df_aud = self.auditor.audit_physical_integrity(self.df_8d_corpus)
        assert passed is True
        assert (df_aud["violations"] == 0).all()

    # 8. Deterministic Rebuild Test
    def test_deterministic_rebuild(self):
        X1 = np.ones((10, 14, 35), dtype=np.float32)
        y1 = np.ones((10,), dtype=np.float32)
        passed, df_aud = self.auditor.audit_deterministic_rebuild(X1, y1, X1, y1)
        assert passed is True
        assert (df_aud["status"] == "PASS").all()
