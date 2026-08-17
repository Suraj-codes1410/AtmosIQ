"""
Unit and Integration Tests for AtmosIQ Phase 8F (Final Synthetic Governance & Provenance Audit).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.modeling.phase8f import (
    Phase8FConfig,
    Phase8FProvenanceManager,
    Phase8FSchemaAuditor,
    Phase8FIsolationAuditor,
    Phase8FPhysicsAuditor,
    Phase8FProvenanceAuditor,
    Phase8FMemorizationAuditor,
    Phase8FReproducibilityAuditor,
    Phase8FGovernanceEngine,
)


class TestPhase8F:
    @classmethod
    def setup_class(cls):
        cls.config = Phase8FConfig()
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

        cls.df_8c_corpus = pd.read_parquet(cls.config.phase8c_corpus_path)
        cls.df_8d_corpus = pd.read_parquet(cls.config.phase8d_corpus_path)

        cls.prov_mgr = Phase8FProvenanceManager(cls.root_dir, cls.config.freeze_manifest_path)
        cls.schema_auditor = Phase8FSchemaAuditor(str(cls.config.feature_registry_path))
        cls.isolation_auditor = Phase8FIsolationAuditor()
        cls.physics_auditor = Phase8FPhysicsAuditor()
        cls.provenance_auditor = Phase8FProvenanceAuditor()
        cls.memorization_auditor = Phase8FMemorizationAuditor(cls.feature_registry)
        cls.memorization_auditor.fit_reference(cls.df_real_train)
        cls.reproducibility_auditor = Phase8FReproducibilityAuditor()
        cls.governance_engine = Phase8FGovernanceEngine(cls.root_dir, cls.config.manifests_dir, cls.config.governance_dir)

    # 1. Protected Artifact Freeze
    def test_protected_artifacts_freeze(self):
        passed, summary = self.prov_mgr.verify_all_protected_artifacts()
        assert passed is True
        assert summary["drift_count"] == 0

    # 2. CAL-07 Physical Identity & Horizon Distribution
    def test_cal07_physical_identity(self):
        assert len(self.df_8d_corpus) == 56088
        assert self.df_8d_corpus["trajectory_id"].nunique() == 2644
        lens = self.df_8d_corpus.groupby("trajectory_id").size().value_counts().to_dict()
        assert set(lens.keys()) == {14, 30}
        assert lens[14] * 14 + lens[30] * 30 == 56088

    # 3. Schema & Feature Registry Compatibility
    def test_schema_compatibility(self):
        passed, df_schema = self.schema_auditor.audit_corpus_schema(self.df_8d_corpus, "CAL-07")
        assert passed is True
        assert (df_schema["status"] == "PASS").all()

    # 4. Data Isolation & Temporal Firewall
    def test_data_isolation(self):
        passed, df_iso = self.isolation_auditor.audit_isolation(
            self.df_real_train, self.df_real_test, self.df_8c_corpus, self.df_8d_corpus
        )
        assert passed is True
        assert (df_iso["violations"] == 0).all()

    # 5. Physical Invariants & Hydrodynamics
    def test_physics_invariants(self):
        passed, df_phys = self.physics_auditor.audit_physics(self.df_8c_corpus, self.df_8d_corpus)
        assert passed is True
        assert (df_phys["violations"] == 0).all()

    # 6. Provenance Traceability
    def test_provenance_traceability(self):
        passed, df_prov = self.provenance_auditor.audit_provenance(self.df_8c_corpus, self.df_8d_corpus)
        assert passed is True
        assert (df_prov["violations"] == 0).all()

    # 7. Memorization & Duplicate Copying Audit
    def test_memorization_audit(self):
        passed, df_mem = self.memorization_auditor.audit_memorization(self.df_8c_corpus, self.df_8d_corpus)
        assert passed is True
        assert (df_mem["violations"] == 0).all()

    # 8. Reproducibility
    def test_reproducibility(self):
        passed, df_repro = self.reproducibility_auditor.audit_reproducibility(
            self.df_8d_corpus.head(50), self.df_8d_corpus.head(50)
        )
        assert passed is True
        assert (df_repro["status"] == "PASS").all()

    # 9. Augmentation Governance
    def test_augmentation_governance(self):
        policy = self.governance_engine.generate_augmentation_governance()
        assert policy["augmentation_tiers"]["RECOMMENDED_PRODUCTION"]["ratio"] == 0.25
        assert policy["augmentation_tiers"]["CONTROLLED_UPPER_BOUND"]["ratio"] == 0.50
        assert policy["augmentation_tiers"]["PROHIBITED"]["ratio"] == 1.00

    # 10. Release Manifest Generation
    def test_release_manifest_generation(self):
        tracked = [
            {"name": "test_feat", "version": "v1", "path": "ml/models/production/v3/feature_registry.csv", "role": "TEST", "immutable": True, "source_phase": "test"}
        ]
        manifest = self.governance_engine.generate_artifact_manifest(tracked)
        assert manifest["total_artifacts"] == 1
