"""
Unit and Integration Tests for AtmosIQ Phase 8E (Deep-Learning Readiness & Phase 9 Admission Gate).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase8e import (
    Phase8EConfig,
    Phase8EProvenanceManager,
    Phase8DReconciliationManager,
    Phase8ETemporalDataLoader,
    TemporalModelBenchmarkEngine,
    Phase8EAuditor,
    Phase9ContractManager,
)


class TestPhase8E:
    @classmethod
    def setup_class(cls):
        cls.config = Phase8EConfig()
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

        cls.prov_mgr = Phase8EProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.reconciler = Phase8DReconciliationManager(cls.config.phase8d_corpus_path, cls.config.feature_registry_path)
        cls.loader = Phase8ETemporalDataLoader(cls.feature_registry)
        cls.loader.fit_scaler(cls.df_real_train)
        cls.auditor = Phase8EAuditor(cls.feature_registry)

    # 1. Protected Artifact Freeze
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["freeze_status"] == "PASS"

    # 2. Phase 8D Metadata Reconciliation
    def test_phase8d_reconciliation(self):
        passed, recon_dict = self.reconciler.reconcile_candidate_artifact()
        assert passed is True
        assert recon_dict["actual_rows"] == 56088
        assert recon_dict["actual_trajectories"] == 2644
        assert recon_dict["mathematical_row_sum_check"] == "PASS"

    # 3. Preprocessing & Sequence Loader Isolation
    def test_sequence_construction_isolation(self):
        X, y = self.loader.create_sequences(self.df_real_train.head(50), window_size=7, is_synthetic=False)
        assert len(X) == 43
        assert len(y) == 43
        assert X.shape[1] == 7

    # 4. Augmentation Mixing Engine
    def test_augmentation_mixing(self):
        df_dummy_synth = self.df_real_train.head(28).copy()
        df_dummy_synth["trajectory_id"] = "SYNTH_1"
        X_comb, y_comb = self.loader.build_augmented_training_set(
            self.df_real_train.head(50), df_dummy_synth, augmentation_ratio=0.25, window_size=7, seed=42
        )
        assert len(X_comb) >= 43

    # 5. Temporal Models Execution
    def test_temporal_models(self):
        X = np.random.randn(20, 7, len(self.feature_registry)).astype(np.float32)
        y = np.random.rand(20).astype(np.float32) * 100.0

        for arch in ["LSTM", "TCN", "Transformer"]:
            model = TemporalModelBenchmarkEngine.get_model(arch, 7, len(self.feature_registry), random_seed=42)
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == 20
            assert (preds >= 0.0).all()

    # 6. Leakage & Temporal Isolation Audit
    def test_leakage_audit(self):
        df_clean_s = pd.DataFrame({"synthetic_date": ["2025-01-01"], "data_origin": ["synthetic"]})
        passed, df_aud = self.auditor.audit_leakage(self.df_real_train, df_clean_s, df_clean_s)
        assert passed is True

    # 7. Physical Validity & Hydrodynamics
    def test_physical_validity_audit(self):
        df_clean = self.df_real_train.head(14).copy()
        passed, df_aud = self.auditor.audit_physical_validity(df_clean, df_clean)
        assert passed is True

    # 8. Provenance Audit
    def test_provenance_audit(self):
        df_synth = pd.DataFrame({
            "trajectory_id": ["T1", "T2"],
            "data_origin": ["synthetic", "synthetic"],
        })
        passed, df_aud = self.auditor.audit_provenance(df_synth, df_synth)
        assert passed is True

    # 9. Phase 9 Training Contract Generation
    def test_contract_generation(self, tmp_path):
        mgr = Phase9ContractManager(tmp_path)
        contract = mgr.generate_contract(
            preferred_corpus_name="AtmosIQ_Synthetic_Calibrated",
            preferred_corpus_version="v0.1.0",
            preferred_corpus_sha256="264c9c5ec109ad03",
            recommended_augmentation=0.25,
            max_augmentation=0.50,
            admission_status="APPROVED_WITH_RESTRICTIONS"
        )
        assert contract["admission_status"] == "APPROVED_WITH_RESTRICTIONS"
        assert contract["synthetic_training_corpus"]["recommended_augmentation_ratio"] == 0.25
        assert len(contract["scientific_disclaimers"]) >= 5
